"""Repair captured-motion drift with Gaussian anchors plus per-dot trajectories.

Explicit reconstruction/codec experiment, NOT a generative dynamics model.
Original motion-memory v1 and its failed fixed-appearance outputs remain intact.
"""
import argparse
import builtins
import json
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import torch

from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake
from .edit_gaussian_memory import splat_layer,array_image
from .gaussian_motion_memory import sample,H,W,SOURCE,ASSETS,OUT as CAPTURE
from .persistent_gaussian import wan_fields_to_state

OUT=Path('artifacts/real_video/gaussian_motion/keyframed_v1')


def tensor_bytes(value):
    if isinstance(value,torch.Tensor): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)): return sum(tensor_bytes(v) for v in value)
    return 0


@torch.no_grad()
def capture():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'motion.pt').exists(): raise FileExistsError('Preserve packet')
    start=time.perf_counter()
    with np.load(CAPTURE/'capture_reference.npz') as archive:
        images=archive['rgb']; masks=archive['masks']
    raw=load_verified(SOURCE)['fields']; background=load_verified(ASSETS)['background']
    anchors=[]; keys=list(range(0,33,4))
    for frame in keys:
        points=wan_fields_to_state(raw[:,:,frame],H,W)[0].flatten(1).T.numpy()
        selected=sample(masks[frame],points[:,:2]*H-.5)>.7
        anchors.append(torch.from_numpy(np.concatenate((points[selected],np.ones((selected.sum(),1),np.float32)),1)))
    gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in images]
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    forward=[]; backward=[]
    for t in range(32):
        forward.append(dis.calc(gray[t],gray[t+1],None)); backward.append(dis.calc(gray[t+1],gray[t],None))
    segments=[]
    for s,(left,right) in enumerate(zip(keys[:-1],keys[1:])):
        p=anchors[s][:,:2].numpy()*H-.5; fd=[]
        for t in range(left,right):
            motion=sample(forward[t],p); p=p+motion; fd.append(motion/H)
        p=anchors[s+1][:,:2].numpy()*H-.5; bd=[]
        for t in range(right-1,left-1,-1):
            motion=sample(backward[t],p); p=p+motion; bd.append(motion/H)
        segments.append(dict(forward=torch.from_numpy(np.stack(fd)).float(),backward=torch.from_numpy(np.stack(bd)).float()))
    packet=dict(anchors=anchors,keyframes=keys,segments=segments,background=background,
                height=H,width=W,fps=16,source_sha256=digest(SOURCE),
                scope='Bidirectional captured Gaussian trajectories between nine appearance keyframes; future anchors stored, not prediction')
    sha=save_inference_checkpoint(packet,OUT/'motion.pt')
    result=dict(seconds=time.perf_counter()-start,keyframes=keys,points_per_anchor=[len(v) for v in anchors],
                anchor_tensor_bytes=tensor_bytes(anchors),displacement_tensor_bytes=tensor_bytes(segments),
                background_tensor_bytes=tensor_bytes(background),total_tensor_bytes=tensor_bytes(packet),
                file_bytes=(OUT/'motion.pt').stat().st_size,sha256=sha,
                prior_failure='Fixed first-frame colours and long tracks tore/degraded; retained in gaussian_motion/v1',
                limitation='Stores future motion plus nine future appearance anchors. Not new action synthesis, no storage superiority established.')
    (OUT/'capture.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def frame_points(packet,t,device='cuda',motion=True):
    if not 0<=t<=32: raise ValueError('Outside captured interval')
    s=min(t//4,7); offset=t-s*4
    left=packet['anchors'][s].to(device).clone(); right=packet['anchors'][s+1].to(device).clone()
    if motion:
        if offset: left[:,:2]+=packet['segments'][s]['forward'][:offset].to(device).sum(0)
        if offset<4: right[:,:2]+=packet['segments'][s]['backward'][:4-offset].to(device).sum(0)
    return left,right,offset/4


@torch.no_grad()
def replay():
    if (OUT/'replay.json').exists(): raise FileExistsError('Preserve replay')
    original=torch.load; original_import=builtins.__import__; loaded=[]
    def guarded(path,*args,**kwargs):
        if Path(path).resolve()!=(OUT/'motion.pt').resolve(): raise AssertionError('Unexpected tensor input')
        loaded.append(str(path)); return original(path,*args,**kwargs)
    def no_frames(*args,**kwargs): raise AssertionError('No source images at replay')
    def guarded_import(name,*args,**kwargs):
        if name=='diffusers' or name.startswith('diffusers.'): raise AssertionError('No diffusion')
        return original_import(name,*args,**kwargs)
    outputs={}; timings={}
    with patch('torch.load',side_effect=guarded),patch('cv2.VideoCapture',side_effect=no_frames),patch('numpy.load',side_effect=no_frames),patch('builtins.__import__',side_effect=guarded_import):
        packet=load_verified(OUT/'motion.pt'); bg,_=splat_layer(packet['background'].cuda())
        for motion in [True,False]:
            name='motion' if motion else 'no_vectors'; images=[]; alphas=[]; times=[]
            torch.cuda.reset_peak_memory_stats()
            for t in range(33):
                torch.cuda.synchronize(); start=time.perf_counter()
                left,right,weight=frame_points(packet,t,motion=motion)
                lr,la=splat_layer(left); rr,ra=splat_layer(right)
                alpha=(1-weight)*la+weight*ra
                rgb=(1-weight)*lr*la+weight*rr*ra+bg*(1-alpha)
                torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
                images.append(array_image(rgb)); alphas.append(alpha.cpu().numpy()[0,0])
            timings[name]=dict(median_update_render_ms_after4=float(np.median(times[4:])),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),all_ms=times)
            outputs[name]=np.stack(images); outputs[name+'_alpha']=np.stack(alphas)
            labeled=[]
            for image in images:
                canvas=Image.new('RGB',(W,H+48),'#141820'); canvas.paste(Image.fromarray(image),(0,48))
                draw=ImageDraw.Draw(canvas)
                draw.text((10,7),f'CAPTURED Gaussian gait | {name} | 9 Gaussian anchors + stored motion; NOT new generation',fill='white')
                draw.text((10,27),'Wheat: Neil Oakes / CC BY-SA 2.0 | no source RGB frames decoded during replay',fill='white')
                labeled.append(np.asarray(canvas))
            imageio.mimwrite(OUT/f'{name}.mp4',labeled,fps=16,codec='libx264',quality=8,macro_block_size=1)
            imageio.mimwrite(OUT/f'{name}_slow.mp4',labeled,fps=8,codec='libx264',quality=8,macro_block_size=1)
        np.savez_compressed(OUT/'replay_outputs.npz',**outputs)
    report=dict(only_tensor_input=loaded,no_source_video_or_npz_read=True,no_diffusion=True,timings=timings,
                fixed_appearance_between_anchors=True,stored_future_appearance_anchors=9,
                total_memory_not_constant_in_duration=True)
    (OUT/'replay.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='timings'},indent=2),flush=True)


def evaluate():
    if (OUT/'evaluation.json').exists(): raise FileExistsError('Preserve evaluation')
    with np.load(CAPTURE/'capture_reference.npz') as f: reference={k:f[k] for k in f.files}
    with np.load(OUT/'replay_outputs.npz') as f: results={k:f[k] for k in f.files}
    packet=load_verified(OUT/'motion.pt')
    with torch.no_grad(): bg=array_image(splat_layer(packet['background'].cuda())[0]).astype(np.float32)/255
    target=reference['rgb'].astype(np.float32)/255*reference['masks'][...,None]+bg[None]*(1-reference['masks'][...,None])
    metrics={}; intermediate=[i for i in range(33) if i%4]
    for mode in ['motion','no_vectors']:
        alpha=results[mode+'_alpha']>.5; mask=reference['masks']>.5
        iou=(alpha&mask).sum((1,2))/np.maximum((alpha|mask).sum((1,2)),1)
        mse=((results[mode].astype(np.float32)/255-target)**2).mean((1,2,3))
        metrics[mode]=dict(mean_silhouette_iou=float(iou.mean()),intermediate_silhouette_iou=float(iou[intermediate].mean()),
                           intermediate_rgb_mse=float(mse[intermediate].mean()),
                           intermediate_rgb_psnr=float(-10*np.log10(mse[intermediate].mean())),
                           note='Capture-source agreement, not held-out generative quality. Approximate masks. Endpoints excluded from intermediate scores.')
    frames=[]
    for t in range(33):
        canvas=Image.new('RGB',(1248,288),'#141820'); d=ImageDraw.Draw(canvas)
        d.text((8,7),'CAPTURED GAUSSIAN GAIT | Motion vectors versus keyframes alone | Reconstruction, not invented action',fill='white')
        panels=[('Source Gaussian clip',reference['rgb'][t]),('Gaussian anchors + local motion',results['motion'][t]),('Same anchors WITHOUT vectors (crossfade)',results['no_vectors'][t])]
        for i,(label,arr) in enumerate(panels):
            d.text((i*416+5,30),label,fill='white'); canvas.paste(Image.fromarray(arr).resize((416,240)),(i*416,48))
        frames.append(np.array(canvas))
        if t in [0,10,21,32]: canvas.save(OUT/f'comparison_{t:02d}.jpg')
    imageio.mimwrite(OUT/'comparison.mp4',frames,fps=8,codec='libx264',quality=8,macro_block_size=1)
    (OUT/'evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['capture','replay','evaluate']); args=p.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(4)
    with keep_windows_awake(): {'capture':capture,'replay':replay,'evaluate':evaluate}[args.mode]()


if __name__=='__main__': main()
