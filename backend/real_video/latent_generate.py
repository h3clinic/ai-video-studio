"""Noise-only latent generation, decoded into explicit Gaussian fields and splatted."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from PIL import Image,ImageDraw
from .latent import inference_decoder
from .model import GaussianVideoDenoiser,sample
from .representation import render_fields_chunked,decode_state
from .evaluate import save_video,to_frames


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=460001)
    parser.add_argument('--label',default='BabyCrawling')
    parser.add_argument('--all-classes',action='store_true')
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--guidance',type=float,default=1.5)
    parser.add_argument('--frame-chunk',type=int,default=1)
    parser.add_argument('--cuda-graph',action='store_true',help='Cache GPU execution, not generated frames; setup has an up-front cost')
    parser.add_argument('--precast-weights',action='store_true')
    parser.add_argument('--codes',type=Path,help='Rerender previously saved latent codes; no prior sampling')
    args=parser.parse_args()
    if not 2<=args.steps<=1000 or not 0<=args.guidance<=10: parser.error('Invalid sampling settings')
    if not 1<=args.frame_chunk<=8: parser.error('Frame chunk must be in 1..8')
    if args.cuda_graph and args.codes: parser.error('Code replay has no denoiser to graph')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    codec=inference_decoder(ckpt['codec_config'],ckpt['codec'])
    if args.precast_weights:
        if device!='cuda': parser.error('Precast path requires CUDA BF16 autocast')
        from .inference_precision import precast_autocast_weights
        precast_autocast_weights(codec)
    codec=codec.to(device)
    args.out.mkdir(parents=True,exist_ok=False)
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.codes:
        saved=torch.load(args.codes,map_location='cpu',weights_only=True)
        if saved['checkpoint_sha256']!=digest: raise ValueError('Latent codes require their matching decoder checkpoint')
        z=saved['latent'].float().to(device); label_ids=saved['labels'].to(device)
        seed=saved['seed']; prior=None
    else:
        if not args.all_classes and args.label not in ckpt['labels']: parser.error('Unknown label')
        label_ids=torch.arange(len(ckpt['labels']),device=device) if args.all_classes else torch.tensor([ckpt['labels'].index(args.label)],device=device)
        prior=GaussianVideoDenoiser(**ckpt['config']).eval()
        prior.load_state_dict(ckpt['model']); seed=args.seed
        if args.precast_weights: precast_autocast_weights(prior)
        prior=prior.to(device)
    runner=None
    if args.cuda_graph:
        from .graph_sampling import GraphedVelocity
        runner=GraphedVelocity(prior,len(label_ids),tuple(ckpt['latent_shape']),args.guidance,ckpt.get('distilled_guidance'))
    if device=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    if prior is not None:
        normalized=sample(prior,label_ids,seed=seed,steps=args.steps,guidance=args.guidance,shape=tuple(ckpt['latent_shape']),distilled_guidance=ckpt.get('distilled_guidance'),velocity_fn=runner)
        z=normalized*ckpt['latent_std'].to(device)+ckpt['latent_mean'].to(device)
    # The saved fp16 code is the code used for this rendering, so it can be replayed.
    z=z.half().float()
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'): field=codec.decode(z).float()
    fields=field*ckpt['std'].to(device)+ckpt['mean'].to(device)
    videos=render_fields_chunked(fields,frame_chunk=args.frame_chunk)
    if device=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
    r=decode_state(fields.permute(0,2,1,3,4).reshape(-1,9,32,32))['R']
    records=[]
    sheet=Image.new('RGB',(768,len(label_ids)*212),(18,22,29))
    for i,label_id in enumerate(label_ids.tolist()):
        label=ckpt['labels'][label_id]; name=f'generated_{i:02d}_{label}'
        save_video(videos[i],args.out/(name+'.mp4'),'NOISE -> GAUSSIAN LATENT | '+label)
        records.append(dict(label=label,mp4=name+'.mp4',gif=name+'.gif'))
        for j,t in enumerate((0,2,4,7)):
            frame=Image.fromarray(to_frames(videos[i])[t]).resize((192,192),Image.Resampling.NEAREST)
            sheet.paste(frame,(192*j,212*i+20))
        ImageDraw.Draw(sheet).text((5,212*i+3),label+' | Gaussian latent generation',fill=(225,232,241))
    sheet.save(args.out/'generated_contact_sheet.png')
    torch.save(dict(latent=z.half().cpu(),labels=label_ids.cpu(),seed=seed,checkpoint_sha256=digest),args.out/'latent_codes.pt')
    torch.save(dict(fields=fields.cpu(),labels=label_ids.cpu(),seed=seed),args.out/'gaussian_fields.pt')
    result=dict(checkpoint_sha256=digest,seed=seed,samples=records,prior_step=ckpt['step'],
                diffusion_steps=args.steps if prior is not None else 0,guidance=args.guidance if prior is not None else None,
                analysis_encoder_on_gpu=False,
                distilled_guidance=ckpt.get('distilled_guidance'),
                cuda_graph=bool(runner),graph_setup_seconds=runner.setup_seconds if runner else 0.,
                precast_weights=args.precast_weights,
                render_frame_chunk=args.frame_chunk,
                inference_input='Random noise and action labels only' if prior is not None else 'Previously generated latent codes only',
                decoder_output='Nine Gaussian attribute channels; explicit Gaussian splatting produces all pixels',
                latent_values_per_clip=z[0].numel(),fp16_latent_bytes_per_clip=z[0].numel()*2,
                float32_gaussian_field_bytes_per_clip=fields[0].numel()*4,
                raw_uint8_rgb_bytes_per_clip=8*64*64*3,checkpoint_bytes=args.checkpoint.stat().st_size,
                seconds_sampling_and_decode_and_splat=elapsed,peak_cuda_allocated_bytes=peak,
                axes_orthogonality_max=(r.transpose(-1,-2)@r-torch.eye(3,device=device)).abs().max().item(),
                limitations=['Validation development samples, not a new unbiased final test.',
                             'Planar, eight-frame, 64x64, class-conditioned generation only.',
                             'Small latent byte count excludes shared weights, file headers and distortion; not superiority over video codecs.',
                             'No source RGB, analysis encoder, pretrained generator, or synthetic teacher used at inference.',
                             'Latent denoising loss is not numerically comparable to raw-field denoising loss.'])
    (args.out/'generation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
