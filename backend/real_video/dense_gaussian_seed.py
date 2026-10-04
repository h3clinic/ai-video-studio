"""Fit a sharp observed Gaussian seed, separate from learned future prediction.

Reads exactly THREE original Wan RGB frames. Never fits future frames. Density
ablation uses the same observed frame/mask and optimizes only Gaussian colours.
"""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F

from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake
from .gaussian_motion_memory import select_cat,sample,H,W,ASSETS
from .edit_gaussian_memory import array_image,splat_layer

OUT=Path('artifacts/real_video/learned_motion/v1/dense_seed')
SOURCE=Path('artifacts/real_video/wan_baseline/cat_seed_421001/wan_original.mp4')


class FixedGeometrySplat:
    def __init__(self,points,height=H,width=W,radius=3):
        self.height,self.width=height,width
        centre=points[:,:2]*height-.5; scale=points[:,2:4].exp()*height
        axis=F.normalize(points[:,4:6],dim=-1)
        yy,xx=torch.meshgrid(torch.arange(-radius,radius+1,device=points.device),torch.arange(-radius,radius+1,device=points.device),indexing='ij')
        pixels=centre.floor().long()[:,None]+torch.stack((xx.flatten(),yy.flatten()),-1)[None]
        d=pixels.to(points.dtype)-centre[:,None]
        local=torch.stack(((d*axis[:,None]).sum(-1),-d[...,0]*axis[:,None,1]+d[...,1]*axis[:,None,0]),-1)
        self.weights=(-.5*(local/scale[:,None]).square().sum(-1)).exp()*points[:,None,9]
        self.weights*=((pixels[...,0]>=0)&(pixels[...,0]<width)&(pixels[...,1]>=0)&(pixels[...,1]<height))
        self.index=(pixels[...,1].clamp(0,height-1)*width+pixels[...,0].clamp(0,width-1)).flatten()
        self.denominator=points.new_zeros(height*width).scatter_add(0,self.index,self.weights.flatten())
        self.alpha=self.denominator.clamp(0,1).reshape(1,1,height,width)

    def render(self,colour):
        values=colour[:,None]*self.weights[...,None]
        numerator=colour.new_zeros(self.height*self.width,3).scatter_add(0,self.index[:,None].expand(-1,3),values.reshape(-1,3))
        return (numerator/self.denominator.clamp_min(1e-6)[:,None]).T.reshape(1,3,self.height,self.width),self.alpha


def points_from_image(image,mask,spacing):
    yy,xx=np.meshgrid(np.arange(spacing/2-.5,H,spacing),np.arange(spacing/2-.5,W,spacing),indexing='ij')
    xy=np.stack((xx.ravel(),yy.ravel()),-1).astype(np.float32)
    xy=xy[sample(mask,xy)>.7]
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    grad=np.stack((cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8),-1)
    g=sample(grad,xy); norm=np.linalg.norm(g,axis=-1,keepdims=True)
    axis=np.concatenate((-g[:,1:],g[:,:1]),-1)/np.maximum(norm,1e-6)
    axis[norm[:,0]<.005]=[1,0]
    log=np.tile(np.log(np.array([.65,.55],np.float32)*spacing/H),(len(xy),1))
    colours=sample(image.astype(np.float32)/255,xy)
    return torch.from_numpy(np.concatenate(((xy+.5)/H,log,axis,colours*2-1,np.ones((len(xy),1),np.float32)),1)).float()


def observed_coarse(frames,gh=32,gw=56):
    yy,xx=np.meshgrid((np.arange(gh)+.5)*H/gh-.5,(np.arange(gw)+.5)*W/gw-.5,indexing='ij')
    positions=np.stack((xx.ravel(),yy.ravel()),-1).astype(np.float32)
    states=[]; backward=[]
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    previous=None
    for image in frames:
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        if previous is not None:
            flow=dis.calc(previous,gray,None); reverse=dis.calc(gray,previous,None)
            positions=positions+sample(flow,positions); backward.append(reverse)
        colour=sample(image.astype(np.float32)/255,positions)*2-1
        logs=np.full_like(positions,math.log(.65/gh)); axes=np.zeros_like(positions); axes[:,0]=1
        fields=np.concatenate(((positions+.5)/H,logs,axes,colour),-1)
        states.append(torch.from_numpy(fields.T.reshape(9,gh,gw)).float())
        previous=gray
    return torch.stack(states,1)[None],backward


def fit():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'seed_1px.pt').exists(): raise FileExistsError('Preserve dense seeds')
    cap=cv2.VideoCapture(str(SOURCE)); frames=[]
    try:
        for _ in range(3):
            ok,bgr=cap.read()
            if not ok: raise ValueError('Three seed frames required')
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    finally: cap.release()
    assert len(frames)==3
    coarse,backward=observed_coarse(frames)
    image=frames[-1]; mask=select_cat(image)
    Image.fromarray(image).save(OUT/'observed_frame_2.png')
    Image.fromarray((mask*255).astype(np.uint8)).save(OUT/'mask.png')
    assets=load_verified(ASSETS)
    bg,_=splat_layer(assets['background'].cuda()); bg=bg.detach()
    target=torch.from_numpy(image).permute(2,0,1)[None].cuda().float()/255
    target_mask=torch.from_numpy(mask)[None,None].cuda()
    report=[]; previews=[]
    for spacing in [4,2,1]:
        if (OUT/f'seed_{spacing}px.pt').exists():
            load_verified(OUT/f'seed_{spacing}px.pt')
            report.append(json.loads((OUT/f'fit_{spacing}px.json').read_text()))
            previews.append(np.array(Image.open(OUT/f'fit_{spacing}px.png').convert('RGB')))
            continue
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        points=points_from_image(image,mask,spacing).cuda()
        radius=math.ceil(2*spacing)
        op=FixedGeometrySplat(points,radius=radius)
        colour=((points[:,6:9]+1)/2).clamp(.001,.999)
        logits=torch.nn.Parameter(torch.logit(colour))
        optimizer=torch.optim.Adam([logits],lr=.04)
        initial=op.render(colour)[0]
        initial_mse=(((initial-target)**2)*target_mask).sum().item()/(target_mask.sum().item()*3)
        for step in range(120):
            rgb,_=op.render(logits.sigmoid())
            error=(((rgb-target)**2)*target_mask).sum()/(target_mask.sum()*3)
            loss=error+.00001*(logits.sigmoid()-colour).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            points[:,6:9]=logits.sigmoid()*2-1
            rgb,alpha=op.render(logits.sigmoid()); composite=rgb*alpha+bg*(1-alpha)
            mse=(((rgb-target)**2)*target_mask).sum().item()/(target_mask.sum().item()*3)
            result=array_image(composite)
            # Map current dense points back to the observed first-frame index grid.
            origin=points[:,:2].cpu().numpy()*H-.5
            for reverse in reversed(backward): origin=origin+sample(reverse,origin)
            uv=np.stack((2*(origin[:,0]+.5)/W-1,2*(origin[:,1]+.5)/H-1),-1)
        torch.cuda.synchronize(); seconds=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated()
        sha=save_inference_checkpoint(dict(points=points.detach().cpu(),sample_uv=torch.from_numpy(uv).float(),
                                           observed_coarse=coarse,background=assets['background'],dt=.5,
                                           height=H,width=W,radius=radius,spacing=spacing,decoded_source_frames=3,
                                           source_sha256=digest(SOURCE),scope='FIT TO THREE OBSERVED ORIGINAL WAN FRAMES ONLY; no future motion or keyframes'),OUT/f'seed_{spacing}px.pt')
        record=dict(spacing=spacing,points=len(points),density_vs_4px=(4/spacing)**2,fit_steps=120,
                    initial_masked_psnr=-10*math.log10(initial_mse),fitted_masked_psnr=-10*math.log10(mse),
                    seconds=seconds,peak_allocated_bytes=peak,packet_bytes=(OUT/f'seed_{spacing}px.pt').stat().st_size,sha256=sha,
                    metric='Observed-frame Gaussian colour fitting, approximate foreground mask; NOT future prediction quality')
        report.append(record); previews.append(result); Image.fromarray(result).save(OUT/f'fit_{spacing}px.png')
        (OUT/f'fit_{spacing}px.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)
        del points,op,logits,optimizer,initial,rgb,alpha,composite,colour,loss,error
    sheet=Image.new('RGB',(W*2,(H+32)*2),'#141820'); draw=ImageDraw.Draw(sheet)
    for i,(label,pic) in enumerate([('ORIGINAL OBSERVED FRAME (not predicted)',image)]+[(f'OBSERVED GAUSSIAN FIT: {r["points"]:,} cat splats / {r["spacing"]}px spacing',pic) for r,pic in zip(report,previews)]):
        x,y=(i%2)*W,(i//2)*(H+32); sheet.paste(Image.fromarray(pic),(x,y+32)); draw.text((x+10,y+9),label,fill='white')
    sheet.save(OUT/'density_comparison.jpg')
    (OUT/'results.json').write_text(json.dumps(dict(decoded_frames=3,density=report,source='Original Wan RGB observed frames; not upsampled old blurry Gaussian output',limits='Reconstruction only. Density increases cost. No learned motion measured in this fitting stage.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(4); cv2.setNumThreads(4)
    with keep_windows_awake(): fit()
