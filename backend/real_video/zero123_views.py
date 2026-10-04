"""Pinned Zero123++ research-only view proposal, not Gaussian reconstruction.

The pretrained model is frozen. No generated image is automatically accepted as
ground truth or attached to the cat. v1.2 poses differ from the original paper.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
import torch
from .checkpoint_io import digest,keep_windows_awake

REVISION='2da07e89919e1a130c9b5add1584c70c7aa065fd'
CODE_REVISION='7d0315c31be6eb906b34cf07d91310f8e12e9b95'
REPO=Path('work/zero123plus')
MODEL=Path('work/zero123plus-v1.2')
ROOT=Path('artifacts/real_video/hunyuan_gaussian/v16_zero123_views')

def prepare_input(image,box,size=320):
    x0,y0,x1,y1=box
    if not (0<=x0<x1<=image.width and 0<=y0<y1<=image.height):raise ValueError('Invalid ROI')
    crop=image.convert('RGBA').crop(box)
    side=max(crop.size);margin=round(side*.1)
    canvas=Image.new('RGBA',(side+2*margin,side+2*margin),(127,127,127,0))
    canvas.alpha_composite(crop,((canvas.width-crop.width)//2,(canvas.height-crop.height)//2))
    # This is model conditioning preprocessing, not an output enhancement.
    return canvas.resize((size,size),Image.Resampling.LANCZOS)

def split_views(image):
    if image.size!=(640,960):raise ValueError('Expected v1.2 2-column, 3-row 320px tile output')
    return [image.crop((c*320,r*320,(c+1)*320,(r+1)*320)) for r in range(3) for c in range(2)]

def main(args):
    if subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()!=CODE_REVISION:
        raise RuntimeError('Unreviewed upstream revision')
    from huggingface_hub import snapshot_download
    from diffusers import EulerAncestralDiscreteScheduler
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='preparing',accepted=False,model='sudo-ai/zero123plus-v1.2',revision=REVISION,
                code_revision=CODE_REVISION,license='Model weights CC-BY-NC 4.0; research-only integration.',
                base_weights_modified=False,source=str(args.source),source_sha256=digest(args.source),
                roi=args.box,seed=args.seed,steps=args.steps,
                scope='Six RGB novel-view proposals of an isolated head. NOT Gaussians or video motion.',
                poses=dict(azimuth_relative=[30,90,150,210,270,330],elevation_absolute=[20,-10,20,-10,20,-10],fov_degrees=30),
                gaussian_camera_alignment='Not performed; full-body camera and crop camera are not interchangeable.')
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        conditioning=prepare_input(Image.open(args.source),args.box);conditioning.save(args.output/'conditioning.png')
        print('Downloading pinned safetensors model to workspace cache.',flush=True)
        snapshot_download(report['model'],revision=REVISION,local_dir=MODEL,max_workers=2)
        spec=importlib.util.spec_from_file_location('zero123plus_reviewed',REPO/'diffusers-support/pipeline.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        pipeline=module.Zero123PlusPipeline.from_pretrained(str(MODEL),torch_dtype=torch.float16,
                    use_safetensors=True,low_cpu_mem_usage=True,local_files_only=True)
        pipeline.scheduler=EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config,timestep_spacing='trailing')
        pipeline.to('cuda');pipeline.vae.enable_slicing()
        torch.cuda.synchronize();report['load_seconds']=time.perf_counter()-start
        start=time.perf_counter()
        result=pipeline(conditioning,num_inference_steps=args.steps,generator=torch.Generator('cuda').manual_seed(args.seed)).images[0]
        torch.cuda.synchronize();report['generation_seconds']=time.perf_counter()-start
        report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
        result.save(args.output/'six_views.png')
        for i,view in enumerate(split_views(result)):view.save(args.output/f'view_{i}.png')
        report['status']='generated_pending_visual_review'
        report['weights_sha256']={str(p.relative_to(MODEL)):digest(p) for p in MODEL.rglob('*.safetensors')}
        save();print(json.dumps(report),flush=True)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('artifacts/real_video/hunyuan_gaussian/v4_full/conditioning.png'))
    p.add_argument('--box',type=int,nargs=4,default=[460,75,650,290]);p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--seed',type=int,default=531030);p.add_argument('--steps',type=int,default=75)
    args=p.parse_args();torch.set_num_threads(4)
    with keep_windows_awake():main(args)
