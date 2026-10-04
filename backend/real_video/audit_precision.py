"""Full-pipeline original versus precast+graph equivalence on fresh inputs."""
import argparse
import gc
import json
from pathlib import Path
import torch
from .checkpoint_io import load_verified,digest,keep_windows_awake
from .latent import inference_decoder
from .model import GaussianVideoDenoiser,sample
from .representation import render_fields_chunked
from .inference_precision import precast_autocast_weights
from .graph_sampling import GraphedVelocity
from .evaluate import save_video


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); checkpoint=load_verified(args.checkpoint)
    mean=checkpoint['mean'].cuda(); std=checkpoint['std'].cuda()
    zm=checkpoint['latent_mean'].cuda(); zs=checkpoint['latent_std'].cuda()
    def models(precast):
        prior=GaussianVideoDenoiser(**checkpoint['config']).eval()
        prior.load_state_dict(checkpoint['model'])
        decoder=inference_decoder(checkpoint['codec_config'],checkpoint['codec'])
        if precast: precast_autocast_weights(prior); precast_autocast_weights(decoder)
        return prior.cuda(),decoder.cuda()
    def generate(prior,decoder,seed,label,runner=None):
        z=sample(prior,torch.tensor([label],device='cuda'),seed,steps=50,shape=tuple(checkpoint['latent_shape']),
                 distilled_guidance=checkpoint.get('distilled_guidance'),velocity_fn=runner)
        with torch.autocast('cuda',dtype=torch.bfloat16): f=decoder.decode((z*zs+zm).half().float()).float()
        fields=f*std+mean; video=render_fields_chunked(fields,frame_chunk=1)
        return z.cpu(),fields.cpu(),video.cpu()
    prior,decoder=models(False); reference=[]
    for i in range(20):
        reference.append(generate(prior,decoder,2010000+i,i%10))
    del prior,decoder; gc.collect(); torch.cuda.empty_cache()
    prior,decoder=models(True)
    runner=GraphedVelocity(prior,1,tuple(checkpoint['latent_shape']),distilled_guidance=checkpoint.get('distilled_guidance'))
    records=[]
    for i in range(20):
        outputs=generate(prior,decoder,2010000+i,i%10,runner)
        diffs=[(a-b).abs().max().item() for a,b in zip(outputs,reference[i])]
        finite=all(torch.isfinite(value).all().item() for value in outputs)
        record=dict(seed=2010000+i,label=checkpoint['labels'][i%10],latent_max_error=diffs[0],
                    field_max_error=diffs[1],pixel_max_error=diffs[2],finite=finite,
                    passed=finite and diffs[2]<=2e-5)
        records.append(record); print(json.dumps(record),flush=True)
        if i==0: save_video(outputs[2][0],args.out/'optimized_original.mp4','NOISE -> GAUSSIANS | PRECAST + GRAPH')
    result=dict(checkpoint_sha256=digest(args.checkpoint),cases=records,passed=all(r['passed'] for r in records),
                max_pixel_error=max(r['pixel_max_error'] for r in records),
                comparison='FP32 stored weights + eager autocast versus selective BF16 stored weights + graph autocast',
                normalization_and_embedding_dtype='FP32',checkpoint_modified=False,
                limits=['Equivalence of this configuration and hardware, not a universal precision theorem.',
                        'Not a quality, speed or memory benchmark; no source videos used.'])
    (args.out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['passed']: raise AssertionError('Precision optimization did not preserve pixels within tolerance')


if __name__=='__main__':
    with keep_windows_awake(): main()
