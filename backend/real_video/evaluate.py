"""Keep actual noise-sampled generation separate from teacher-forced reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .model import GaussianVideoDenoiser,sample,schedule
from .representation import render_fields,decode_state
from .train import validate

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent.parent/'work/real_video'


def to_frames(video):
    return (video.permute(0,2,3,1).clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)


def save_video(video,path,label):
    frames=to_frames(video)
    imageio.mimsave(path,frames,fps=8,codec='libx264',macro_block_size=1)
    # Nearest-neighbor enlargement for legibility, not a learned detail enhancer.
    previews=[]
    for array in frames:
        img=Image.new('RGB',(256,282),(18,22,29))
        img.paste(Image.fromarray(array).resize((256,256),Image.Resampling.NEAREST),(0,26))
        ImageDraw.Draw(img).text((7,6),label,fill=(225,232,241))
        previews.append(img)
    previews[0].save(path.with_suffix('.gif'),save_all=True,append_images=previews[1:],duration=125,loop=0)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--sample-count',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=GaussianVideoDenoiser(**checkpoint['config']).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    mean,std=checkpoint['mean'].cuda(),checkpoint['std'].cuda()
    data=torch.load(WORK/'dataset.pt',weights_only=True,map_location='cpu')
    ids=[i for i,m in enumerate(data['metadata']) if m['split']==args.split]
    fields=data['fields'][ids].float().cuda()
    x=(fields-mean)/std
    labels=torch.tensor([data['labels'].index(data['metadata'][i]['label']) for i in ids],device='cuda')
    loss,baseline=validate(model,x,labels,schedule('cuda'))
    real_rgb=data['videos'][ids].float().permute(0,1,4,2,3).cuda()/255
    recon=[]
    for i in range(0,len(fields),4): recon.append(render_fields(fields[i:i+4]))
    reconstruction=torch.cat(recon)
    reconstruction_mse=(real_rgb-reconstruction).square().mean().item()
    # A representation test only; future target RGB participates in the analysis encoder.
    reconstruction_psnr=-10*np.log10(max(reconstruction_mse,1e-12))
    sample_ids=torch.arange(args.sample_count,device='cuda')%len(data['labels'])
    seed=460009 if args.split=='test' else 460001
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
    normalized=sample(model,sample_ids,seed=seed,steps=50)
    generated_fields=normalized*std+mean
    generated=render_fields(generated_fields)
    torch.cuda.synchronize(); seconds=time.perf_counter()-start
    raw=decode_state(generated_fields.permute(0,2,1,3,4).reshape(-1,9,32,32))
    identity=torch.eye(3,device='cuda')
    frame_error=(raw['R'].transpose(-1,-2)@raw['R']-identity).abs().max().item()
    canonical=render_fields(generated_fields,canonical_axes=True)
    changed=(canonical-generated).abs().mean().item()
    raw_motion=(generated_fields[:,:2,1:]-generated_fields[:,:2,:-1]).abs().mean().item()*16
    records=[]
    for j,mid in enumerate(sample_ids.tolist()):
        name=f'generated_{j:02d}_{data["labels"][mid]}'
        save_video(generated[j],args.out/f'{name}.mp4',f'NOISE GENERATED | {data["labels"][mid]}')
        records.append(dict(label=data['labels'][mid],mp4=name+'.mp4',gif=name+'.gif'))
    # Show first held-out representation roundtrip, explicitly separated from generation.
    save_video(real_rgb[0],args.out/'heldout_reference.mp4','REAL HELD-OUT REFERENCE')
    save_video(reconstruction[0],args.out/'heldout_reconstruction.mp4','RECONSTRUCTION ONLY - NOT GENERATION')
    sheet=Image.new('RGB',(4*192,len(records)*212),(18,22,29))
    for j,record in enumerate(records):
        for k,t in enumerate((0,2,4,7)):
            cell=Image.fromarray(to_frames(generated[j])[t]).resize((192,192),Image.Resampling.NEAREST)
            sheet.paste(cell,(k*192,j*212+20))
        ImageDraw.Draw(sheet).text((5,j*212+3),record['label']+' | generated from noise',fill=(225,232,241))
    sheet.save(args.out/'generated_contact_sheet.png')
    torch.save(dict(fields=generated_fields.cpu(),seed=seed,labels=sample_ids.cpu(),representation='direct Gaussian fields'),args.out/'generated_gaussian_fields.pt')
    # Check closest training example, not a proof of nonmemorization.
    training_ids=[i for i,m in enumerate(data['metadata']) if m['split']=='train']
    training_rgb=data['videos'][training_ids].float().permute(0,1,4,2,3)/255
    small_train=F.interpolate(training_rgb.flatten(0,1),size=16,mode='area').reshape(len(training_ids),8,3,16,16)
    small_generated=F.interpolate(generated.cpu().flatten(0,1),size=16,mode='area').reshape(len(records),8,3,16,16)
    distances=(small_generated[:,None]-small_train[None]).square().mean((2,3,4,5))
    nearest=[dict(sample=i,train_file=data['metadata'][training_ids[distances[i].argmin().item()]]['file'],
                  coarse_mse=distances[i].min().item()) for i in range(len(records))]
    result=dict(split=args.split,heldout_clips=len(ids),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                checkpoint_step=checkpoint['step'],denoising_velocity_mse=loss,zero_velocity_mse=baseline,
                representation_roundtrip_psnr_db=reconstruction_psnr,
                sample_seed=seed,samples=records,sampling_seconds=seconds,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                axes_orthogonality_error=frame_error,canonical_axis_ablation_mean_pixel_change=changed,
                mean_generated_position_step_pixels=raw_motion,nearest_training_clip_coarse_check=nearest,
                normalization_source='training clips only',
                pixel_decoder='normalized anisotropic Gaussian scatter only, no neural image/video decoder',
                limitations=['Denoising loss and reconstruction PSNR are not generation-quality scores.',
                             'Only 64x64, 8-frame, ten-action-class generation; no open-ended language conditioning.',
                             'Planar Gaussian mixture, no recovered 3D depth or alpha-ordered 3DGS.',
                             'Joint clip diffusion; not yet bounded-memory recurrent long-video generation.',
                             'Flow-derived training parameters are approximate and not uniquely determined by RGB.',
                             'No FVD/VBench or modern-model superiority claim; few generated samples.'])
    (args.out/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
