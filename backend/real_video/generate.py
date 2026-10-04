"""Noise -> learned temporal Gaussian parameters -> splatted video. No dataset access."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .model import GaussianVideoDenoiser,sample
from .representation import render_fields,decode_state

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/real_video/gaussian_video_weights.pt')
    parser.add_argument('--label',default='BabyCrawling')
    parser.add_argument('--seed',type=int,default=12345)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not 2<=args.steps<=200: parser.error('Use 2..200 sampling steps')
    torch.set_num_threads(4)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if args.label not in ckpt['labels']: parser.error(f"Choose one of {ckpt['labels']}")
    args.out.mkdir(parents=True,exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=GaussianVideoDenoiser(**ckpt['config']).to(device).eval()
    model.load_state_dict(ckpt['model'])
    condition=torch.tensor([ckpt['labels'].index(args.label)],device=device)
    if device=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start=time.perf_counter()
    normalized=sample(model,condition,seed=args.seed,steps=args.steps)
    field=normalized*ckpt['std'].to(device)+ckpt['mean'].to(device)
    video=render_fields(field)[0]
    if device=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    frames=(video.permute(0,2,3,1).cpu().numpy()*255).round().astype('uint8')
    imageio.mimsave(args.out/'generated.mp4',frames,fps=8,codec='libx264',macro_block_size=1)
    preview=[Image.fromarray(frame).resize((256,256),Image.Resampling.NEAREST) for frame in frames]
    preview[0].save(args.out/'generated.gif',save_all=True,append_images=preview[1:],duration=125,loop=0)
    preview[0].save(args.out/'first_frame.png')
    # Store exactly the generated Gaussian fields, so the video can be rerendered without the model.
    torch.save(dict(field=field.cpu(),seed=args.seed,label=args.label),args.out/'gaussians.pt')
    first=decode_state(field[:,:,0])
    report=dict(model_type='Gaussian-parameter clip diffusion trained from scratch on real UCF101 clips',
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),seed=args.seed,label=args.label,
                inference_input='Random noise and action label only; no reference frame, cached clip, synthetic teacher, or pretrained video generator',
                output='8 frames, 64x64, 8fps; 1024 planar anisotropic Gaussians per frame',sampling_steps=args.steps,
                sampling_seconds=elapsed,gaussian_field_bytes=field.numel()*field.element_size(),
                axes_orthogonality_max=(first['R'].transpose(-1,-2)@first['R']-torch.eye(3,device=device)).abs().max().item(),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                limitation='Not full 3D, long-horizon recurrent, or open-vocabulary text-to-video generation. Low-resolution quality must be inspected.')
    (args.out/'generation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
