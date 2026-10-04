"""Render six object sides, six environment directions and camera-only video."""
import json
import math
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import torch
from .checkpoint_io import load_verified,keep_windows_awake,digest
from .gaussian3d import render
from .enclosing_scene import ROOT,VIEWS

@torch.no_grad()
def main():
    out=ROOT/'inspection';out.mkdir(exist_ok=False)
    a=load_verified(ROOT/'scene.pt');n=a['original_count'];data={k:a[k].cuda() for k in ['position','covariance','colour','opacity']}
    cat={k:v[:n] for k,v in data.items()};env={k:v[n:] for k,v in data.items()}
    center=torch.tensor(a['camera_pivot'],device='cuda'); directions=list(VIEWS.items())
    def pic(im):return Image.fromarray((im.cpu().numpy().clip(0,1)*255).astype(np.uint8))
    for kind in ['cat','environment']:
        sheet=Image.new('RGB',(1152,824),'#161d26');draw=ImageDraw.Draw(sheet)
        for i,(name,(direction,up)) in enumerate(directions):
            vec=torch.tensor(direction,device='cuda',dtype=torch.float32)
            if kind=='cat': eye=center-vec*1.85;target=center;fields=cat;fov=42
            else:eye=torch.tensor([0.,.8,0.],device='cuda');target=eye+vec;fields=env;fov=90
            im,alpha=render(**fields,eye=eye,target=target,height=384,width=384,fov=fov,radius=5,ground=False)
            image=pic(im);image.save(out/f'{kind}_{name}.png');x=i%3*384;y=i//3*412
            sheet.paste(image,(x,y+28));draw.text((x+8,y+8),f'{kind}: {name}',fill='white')
        sheet.save(out/f'{kind}_six_views.jpg')
    path=out/'generated_surfaces_camera_only_7s.mp4'
    with imageio.get_writer(path,fps=16,codec='libx264',quality=8,macro_block_size=1) as writer:
        for i in range(112):
            angle=2*math.pi*i/112;eye=center+center.new_tensor([1.85*math.sin(angle),.25,1.85*math.cos(angle)])
            im,_=render(**data,eye=eye,target=center,height=480,width=832,radius=4,ground=False)
            image=pic(im);writer.append_data(np.asarray(image))
            if i in [0,28,56,84]:image.save(out/f'orbit_{i:03d}.png')
            if i%28==0:print(json.dumps(dict(stage='orbit',frame=i)),flush=True)
    (out/'report.json').write_text(json.dumps(dict(video_sha256=digest(path),scope='Static learned image-to-3D completion plus radial environment; camera motion only',accepted=False),indent=2))

if __name__=='__main__':
    torch.set_num_threads(4)
    with keep_windows_awake():main()
