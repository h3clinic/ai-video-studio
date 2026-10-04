"""Fixed-camera Gaussian capture with explicit motion tracks and bounded refresh.

This is video-conditioned reconstruction, NOT generated motion or 3D recovery.
Colour is constant within each eight-frame segment. New Gaussian banks at
segment boundaries capture disocclusion/appearance changes instead of hiding
that information in an RGB replay buffer. All saved future tracks count as memory.
"""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from real_video.checkpoint_io import digest, save_inference_checkpoint, keep_windows_awake
from real_video.gaussian_motion_memory import sample, transport_geometry


def fill_enclosed_holes(mask):
    flood=cv2.copyMakeBorder(mask.astype(np.uint8),1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    cv2.floodFill(flood,None,(0,0),1)
    return np.maximum(mask,(flood[1:-1,1:-1]==0).astype(np.float32))


def animal_mask(image):
    small=cv2.resize(image,(416,240),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
    green=(hsv[:,:,0]>32)&(hsv[:,:,0]<100)&(hsv[:,:,1]>45)
    labels=np.full((240,416),cv2.GC_PR_BGD,np.uint8)
    labels[12:235,12:405]=cv2.GC_PR_FGD
    labels[green]=cv2.GC_BGD
    warm=(hsv[:,:,0]<30)&(hsv[:,:,1]>65)&(hsv[:,:,2]>65)
    warm[:12]=False; warm[:, :12]=False; warm[:,405:]=False
    sure=cv2.erode(warm.astype(np.uint8),np.ones((3,3),np.uint8))>0
    labels[sure]=cv2.GC_FGD
    labels[:5]=cv2.GC_BGD; labels[-3:]=cv2.GC_BGD
    cv2.setRNGSeed(71033)
    cv2.grabCut(small,labels,None,np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_MASK)
    mask=((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)).astype(np.uint8)
    count,connected,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if count<2: raise ValueError('Animal segmentation failed')
    mask=(connected==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.float32)
    # GrabCut can classify dark eyes/nose as background. Fill enclosed holes,
    # but retain exterior-connected gaps between legs and under the belly.
    mask=fill_enclosed_holes(mask)
    return cv2.resize(mask,(832,480),interpolation=cv2.INTER_LINEAR)


def capture(source,out,foreground=False,spacing=2,refresh=8):
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    start=time.perf_counter(); process=psutil.Process(); peak=process.memory_info().rss
    cv2.setNumThreads(4); torch.set_num_threads(4)
    cap=cv2.VideoCapture(str(source)); frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames)!=33 or frames[0].shape!=(480,832,3): raise ValueError('Expected exact 33-frame 832x480 clip')
    masks=[animal_mask(f) for f in frames] if foreground else [np.ones((480,832),np.float32)]*len(frames)
    gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in frames]
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    yy,xx=np.meshgrid(np.arange(.5,480,spacing),np.arange(.5,832,spacing),indexing='ij')
    grid=np.stack((xx.ravel(),yy.ravel()),-1).astype(np.float32)
    segments=[]; global_id=0
    for first in range(0,len(frames),refresh):
        xy=grid[sample(masks[first],grid)>.6].copy()
        n=len(xy); colour=sample(frames[first].astype(np.float32)/255,xy)
        axes=np.tile(np.array([1.,0.],np.float32),(n,1))
        logs=np.tile(np.log(np.array([.65,.55],np.float32)*spacing/480),(n,1))
        base=np.concatenate(((xy+.5)/480,logs,axes,colour*2-1,np.ones((n,1),np.float32)),1)
        states=[]
        for t in range(first,min(first+refresh,len(frames))):
            if t>first:
                flow=dis.calc(gray[t-1],gray[t],None)
                delta=sample(flow,xy)
                gradients=np.stack([np.stack([cv2.Sobel(flow[:,:,c],cv2.CV_32F,1,0,ksize=3)/8,
                                             cv2.Sobel(flow[:,:,c],cv2.CV_32F,0,1,ksize=3)/8],-1) for c in range(2)],-2)
                jac=sample(gradients.reshape(480,832,4),xy).reshape(-1,2,2)+np.eye(2,dtype=np.float32)
                u,s,vh=np.linalg.svd(jac); jac=(u*np.clip(s,.85,1.18)[:,None,:])@vh
                logs,axes=transport_geometry(logs,axes,jac)
                xy=xy+delta
            # Absolute from this bank, not unbounded float16 accumulated increments.
            angle=np.arctan2(axes[:,1],axes[:,0])[:,None]
            values=np.concatenate(((xy+.5)/480-base[:,:2],logs-base[:,2:4],angle),1)
            visibility=sample(masks[t],xy).clip(0,1)
            visibility*=((xy[:,0]>=0)&(xy[:,0]<832)&(xy[:,1]>=0)&(xy[:,1]<480))
            states.append(dict(delta=torch.from_numpy(values.astype(np.float16)),visibility=torch.from_numpy((visibility*255).round().astype(np.uint8))))
            peak=max(peak,process.memory_info().rss)
        segments.append(dict(first_frame=first,ids=torch.arange(global_id,global_id+n),base=torch.from_numpy(base),states=states))
        global_id+=n
        print(f'Captured segment {first}: {n} Gaussians, {len(states)} motion states',flush=True)
    packet=dict(segments=segments,height=480,width=832,fps=16,frames=33,source_sha256=digest(source),foreground=foreground,
                role='Recorded planar Gaussian reconstruction; persistent within each segment, new IDs at refresh',
                future_tracks_stored=True,refresh_interval=refresh,no_rgb_frames_in_packet=True)
    path=out/'gaussian_memory.pt'; save_inference_checkpoint(packet,path)
    def tensor_bytes(x):
        if isinstance(x,torch.Tensor): return x.numel()*x.element_size()
        if isinstance(x,dict): return sum(tensor_bytes(v) for v in x.values())
        if isinstance(x,list): return sum(tensor_bytes(v) for v in x)
        return 0
    report=dict(source=str(source),source_sha256=digest(source),capture_seconds=time.perf_counter()-start,
                peak_process_rss_sampled_bytes=peak,file_bytes=path.stat().st_size,tensor_bytes=tensor_bytes(packet),
                gaussian_banks=len(segments),gaussians_per_bank=[len(s['base']) for s in segments],
                raw_rgb_clip_bytes=33*480*832*3,source_mp4_bytes=source.stat().st_size,
                interpretation='All future vectors and appearance refresh banks included. Not constant-size memory or 3D inference.')
    (out/'capture_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for i in [0,16,32]:
        Image.fromarray((masks[i]*255).astype(np.uint8)).save(out/f'mask_{i:03d}.png')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--foreground',action='store_true')
    args=p.parse_args()
    with keep_windows_awake(): print(json.dumps(capture(args.source,args.out,args.foreground),indent=2))
