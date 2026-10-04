"""Matched-view appearance audit and orbit; not unseen-view accuracy evidence."""
import json
import math
from pathlib import Path
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .checkpoint_io import load_verified,keep_windows_awake,digest
from .fit_appearance3d import target_data
from .gaussian3d import render,rotation
from .demo_true3d import covariance

ROOT=Path('artifacts/real_video/true3d/v4_appearance')
BASE=Path('artifacts/real_video/true3d/v1')


def masked_edge_error(image,target,mask):
    # Compare edge errors to target, not standalone sharpness that rewards noise.
    kernel=image.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])/8
    kernel=torch.stack((kernel,kernel.T))[:,None]
    def gradient(x): return F.conv2d(x.mean(-1)[None,None],kernel,padding=1)
    return float(((gradient(image)-gradient(target)).square()*mask).sum()/(2*mask.sum()))


@torch.no_grad()
def main():
    if (ROOT/'evaluation.json').exists(): raise FileExistsError('Preserve evaluation')
    old=load_verified(BASE/'cat_asset.pt'); new=load_verified(ROOT/'cat_asset.pt'); cam=load_verified(ROOT/'camera.pt')
    for key in ['position','frame','scale','opacity','ids','face_id','barycentric']:
        assert torch.equal(old[key],new[key]),key
    a={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in new.items()}; cov=covariance(a)
    old_colour=old['colour'].cuda(); eye=cam['eye'].cuda(); target=cam['target'].cuda()
    ref,mask,_=target_data(); ref=ref.cuda(); mask=mask.cuda(); metrics={}
    for label,col in [('before',old_colour),('after',a['colour'])]:
        rgb,alpha=render(a['position'],cov,col,a['opacity'],eye,target,512,512,fov=40,ground=False)
        rgb=rgb+(1-alpha[...,None])*(.5-rgb.new_tensor([.15,.19,.24]))
        metrics[label]=dict(masked_psnr=-10*math.log10(float(((rgb-ref).square()*mask[...,None]).sum()/(3*mask.sum()))),
                            target_relative_edge_mse=masked_edge_error(rgb,ref,mask))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    output=ROOT/'appearance_orbit_comparison_7s.mp4'; writer=imageio.get_writer(output,fps=24,codec='libx264',quality=8,macro_block_size=1)
    elapsed=[]; torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(168):
            angle=2*math.pi*i/168
            camera=target+rotation(eye.new_tensor([0.,1.,0.]),eye.new_tensor(angle))@(eye-target)
            torch.cuda.synchronize(); begin=time.perf_counter()
            images=[]
            for colour in [old_colour,a['colour']]:
                rgb,_=render(a['position'],cov,colour,a['opacity'],camera,target,640,640,fov=40)
                images.append(np.uint8(rgb.cpu().numpy()*255))
            torch.cuda.synchronize(); elapsed.append((time.perf_counter()-begin)*1000)
            canvas=Image.new('RGB',(1280,708),'#151922'); draw=ImageDraw.Draw(canvas)
            for j,(name,pic) in enumerate(zip(['BEFORE: inferred colours','AFTER: source-view colour fitting'],images)):
                canvas.paste(Image.fromarray(pic),(j*640,32)); draw.text((j*640+8,7),name,font=font,fill='white')
            draw.text((8,681),'Same 45,000 XYZ Gaussians | observed appearance reconstruction | orbit only, not generated action',font=font,fill='white')
            writer.append_data(np.asarray(canvas))
            if i in [0,42,84,126]: canvas.save(ROOT/f'comparison_{i:03d}.jpg')
            if i%42==0: print(json.dumps(dict(orbit_frame=i,total=168)),flush=True)
    finally: writer.close()
    reader=imageio.get_reader(output); count=0; seen=set()
    for pic in reader:
        count+=1; seen.add(hashlib.sha256(pic.tobytes()).hexdigest())
    metadata=reader.get_meta_data(); reader.close(); assert count==168 and metadata['fps']==24
    metrics.update(decoded_frames=count,unique_decoded_frames=len(seen),fps=24,seconds=7,median_ms_two_views_including_cpu_copy=float(np.median(elapsed[5:])),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),video_sha256=digest(output),positions_unchanged=True,
        scope='Source-view metrics use the fitted training image; no novel-view ground truth. Orbit exposes errors; not motion generation or matched-quality efficiency evidence.')
    (ROOT/'evaluation.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    with keep_windows_awake(): main()
