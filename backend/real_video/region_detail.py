"""Category-independent region reconstruction experiment using frozen models.

Explicit image ROI is experiment input, not a learned region detector. Produces
an isolated 3D part for evaluation; never silently grafts it onto a body.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from PIL import Image
from .checkpoint_io import digest,keep_windows_awake,load_verified,save_inference_checkpoint

def crop_region(image,box):
    if len(box)!=4:raise ValueError('ROI needs four coordinates')
    x0,y0,x1,y1=box
    if not (0<=x0<x1<=image.width and 0<=y0<y1<=image.height):raise ValueError('ROI outside image')
    return image.crop(box)

def execute(args):
    root=args.root;shape_root=root/'shape';paint_root=root/'paint'
    if args.stage=='all':
        for stage in ['prepare','shape','paint','inspect']:
            subprocess.run([sys.executable,'-m','real_video.region_detail',stage,'--root',str(root),
                '--source',str(args.source),'--box',*map(str,args.box)],check=True)
        return
    if args.stage=='prepare':
        shape_root.mkdir(parents=True,exist_ok=True)
        if (shape_root/'conditioning.png').exists():raise FileExistsError('Preserve region input')
        image=Image.open(args.source).convert('RGBA');crop=crop_region(image,args.box)
        crop.save(shape_root/'conditioning.png')
        (root/'experiment.json').write_text(json.dumps(dict(source=str(args.source),source_sha256=digest(args.source),
            roi=args.box,region_selection='explicit manual image rectangle; no category-specific architecture',
            scope='Frozen general-purpose model rerun on region; isolated part, not integrated repair or trained refiner',
            accepted=False),indent=2))
    elif args.stage=='shape':
        from . import hunyuan_gaussian as model
        model.ROOT=shape_root;model.SOURCE=shape_root/'conditioning.png'
        model.WEIGHTS=model.WORK/'hunyuan_full/hunyuan3d-dit-v2-0'
        model.MODEL='tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0';model.REVISION='9cd649ba6913f7a852e3286bad86bfa9a2d83dcf'
        model.shape()
    elif args.stage=='paint':
        from . import hunyuan_multiview as model
        model.BASE=shape_root;model.ROOT=paint_root
        model.controls();model.generate()
        # Unload model allocations before rasterization in another process.
    elif args.stage=='inspect':
        from . import hunyuan_multiview as model
        from . import hunyuan_appearance as viewer
        model.BASE=shape_root;model.ROOT=paint_root
        model.bake(diagnostics=False)
        a=load_verified(paint_root/'gaussian_fitted.pt');viewer.ROOT=paint_root
        viewer.views(a,'appearance')
        center=torch.tensor(a['camera_pivot'])
        save_inference_checkpoint(dict(eye=center+torch.tensor([0.,0.,a['camera_distance']]),target=center,fov=42.),paint_root/'camera.pt')
        viewer.inspect()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['all','prepare','shape','paint','inspect'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--box',type=int,nargs=4,required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    with keep_windows_awake():execute(args)
