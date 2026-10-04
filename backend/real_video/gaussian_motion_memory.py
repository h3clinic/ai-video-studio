"""Captured non-rigid Gaussian motion memory, not learned novel-action generation.

Capture sees an existing jointly generated Gaussian clip. Isolated playback sees
only initial splats, displacement/deformation/visibility tracks and a background.
Appearance colours are fixed. No RGB frame, image warp or video decoder at replay.
"""
import argparse
import builtins
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from .checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint
from .edit_gaussian_memory import splat_layer, array_image
from .persistent_gaussian import wan_fields_to_state, render_state

OUT=Path('artifacts/real_video/gaussian_motion/v1')
SOURCE=Path('artifacts/real_video/wan_bridge/v2/generated_cat/gaussian_fields.pt')
ASSETS=Path('artifacts/real_video/gaussian_edits/v1/assets.pt')
H,W=480,832


def select_cat(image):
    """Approximate annotated ROI+colour GrabCut; not learned semantic identity."""
    mask=np.full((H,W),cv2.GC_BGD,np.uint8)
    mask[95:476,:650]=cv2.GC_PR_BGD
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    possible=(hsv[:,:,0]<28)&(hsv[:,:,1]>55)&(image[:,:,0]>80)
    possible[:95]=False; possible[476:]=False; possible[:,650:]=False
    mask[possible]=cv2.GC_PR_FGD
    sure=cv2.erode(possible.astype(np.uint8),np.ones((7,7),np.uint8))>0
    mask[sure]=cv2.GC_FGD
    cv2.setRNGSeed(430103)
    cv2.grabCut(image,mask,None,np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_MASK)
    binary=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if n<2: raise ValueError('No cat mask')
    return (labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.float32)


def sample(array,xy):
    # OpenCV remap limits each destination dimension to signed 16-bit range.
    # Dense point sets must be sampled in bounded batches, not an N x 1 map.
    if len(xy)>16384:
        return np.concatenate([sample(array,xy[i:i+16384]) for i in range(0,len(xy),16384)],axis=0)
    return cv2.remap(array,xy[:,0,None].astype(np.float32),xy[:,1,None].astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,0]


def transport_geometry(logscale,axes,jacobian):
    """Covariance pushforward Sigma'=J Sigma J^T; bounded axes for robustness."""
    v=np.stack((-axes[:,1],axes[:,0]),-1)
    frame=np.stack((axes,v),-1)
    sigma=(frame*np.exp(2*logscale)[:,None,:])@frame.transpose(0,2,1)
    transported=jacobian@sigma@jacobian.transpose(0,2,1)
    values,vectors=np.linalg.eigh(transported)
    # Largest eigenvalue first, sign matched to transported previous first axis.
    scales=np.sqrt(np.maximum(values[:,::-1],1e-10)).clip(.8/H,6/H)
    axis=vectors[:,:,1]
    reference=np.einsum('nij,nj->ni',jacobian,axes)
    axis*=np.where((axis*reference).sum(-1)<0,-1,1)[:,None]
    return np.log(scales).astype(np.float32),axis.astype(np.float32)


def packet_bytes(packet):
    return {k:v.numel()*v.element_size() for k,v in packet.items() if isinstance(v,torch.Tensor)}


@torch.no_grad()
def capture():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'motion.pt').exists(): raise FileExistsError('Preserve captured motion')
    start=time.perf_counter(); source=load_verified(SOURCE); backgrounds=load_verified(ASSETS)
    fields=source['fields']; frames=[]
    for t in range(fields.shape[2]):
        explicit=wan_fields_to_state(fields[:,:,t].cuda(),H,W)
        frames.append(array_image(render_state(explicit,H,W,radius=8)))
    initial=wan_fields_to_state(fields[:,:,0],H,W)[0].flatten(1).T.numpy()
    masks=[]
    for t,frame in enumerate(frames):
        masks.append(select_cat(frame))
        if t%8==0: print(json.dumps(dict(stage='cat_masks',frame=t)),flush=True)
    chosen=sample(masks[0],initial[:,:2]*H-.5)>.7
    base=np.concatenate((initial[chosen],np.ones((chosen.sum(),1),np.float32)),1)
    positions=[base[:,:2]*H-.5]; logs=[base[:,2:4]]; axes=[base[:,4:6]]; visibility=[np.ones(len(base),np.float32)]
    steps=[]; scale_steps=[]; angle_steps=[]; diagnostics=[]
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in frames]
    for t in range(1,len(frames)):
        flow=dis.calc(gray[t-1],gray[t],None)
        backward=dis.calc(gray[t],gray[t-1],None)
        xy=positions[-1]; movement=sample(flow,xy)
        # Safety cap records large steps rather than silently accepting explosions.
        lengths=np.linalg.norm(movement,axis=-1)
        movement*=np.minimum(1,40/np.maximum(lengths,1e-6))[:,None]
        nextxy=xy+movement
        fb=np.linalg.norm(movement+sample(backward,nextxy),axis=-1)
        dx=np.stack((cv2.Sobel(flow[:,:,0],cv2.CV_32F,1,0,ksize=3)/8,
                     cv2.Sobel(flow[:,:,1],cv2.CV_32F,1,0,ksize=3)/8),-1)
        dy=np.stack((cv2.Sobel(flow[:,:,0],cv2.CV_32F,0,1,ksize=3)/8,
                     cv2.Sobel(flow[:,:,1],cv2.CV_32F,0,1,ksize=3)/8),-1)
        jac=np.stack((sample(dx,xy),sample(dy,xy)),-1)+np.eye(2,dtype=np.float32)[None]
        u,s,vh=np.linalg.svd(jac); jac=(u*np.clip(s,.85,1.18)[:,None,:])@vh
        nextlog,nextaxis=transport_geometry(logs[-1],axes[-1],jac)
        angle=np.arctan2(axes[-1][:,0]*nextaxis[:,1]-axes[-1][:,1]*nextaxis[:,0],(axes[-1]*nextaxis).sum(-1))
        # Visibility comes from the captured sequence; this is not inferred future memory.
        visible=sample(masks[t],nextxy).clip(0,1)
        visible*=np.exp(-np.maximum(fb-1,0)/4)
        visible*=((nextxy[:,0]>=0)&(nextxy[:,0]<W)&(nextxy[:,1]>=0)&(nextxy[:,1]<H))
        steps.append(movement/H); scale_steps.append(nextlog-logs[-1]); angle_steps.append(angle)
        positions.append(nextxy); logs.append(nextlog); axes.append(nextaxis); visibility.append(visible)
        # Residual after best affine fit distinguishes non-rigid tracks from crop motion;
        # optical-flow error also contributes and is NOT evidence of correct anatomy.
        design=np.concatenate((xy,np.ones((len(xy),1),np.float32)),1)
        fit=np.linalg.lstsq(design,movement,rcond=None)[0]
        residual=movement-design@fit
        diagnostics.append(dict(frame=t,mean_motion_pixels=float(np.linalg.norm(movement,axis=-1).mean()),
                                best_affine_residual_rms_pixels=float(np.sqrt((residual**2).sum(-1).mean())),
                                median_forward_backward_error=float(np.median(fb)),
                                mean_visibility=float(visible.mean()),capped_tracks=int((lengths>40).sum())))
        if t%8==0: print(json.dumps(dict(stage='motion_tracks',**diagnostics[-1])),flush=True)
    packet=dict(base=torch.from_numpy(base),displacements=torch.from_numpy(np.stack(steps)).float(),
                logscale_deltas=torch.from_numpy(np.stack(scale_steps)).float(),
                angle_deltas=torch.from_numpy(np.stack(angle_steps)).float(),
                visibility=torch.from_numpy(np.stack(visibility)).float(),
                background=backgrounds['background'],fps=16,height=H,width=W,
                source_sha256=digest(SOURCE),background_source_sha256=digest(ASSETS),
                note='Captured planar optical-flow trajectories. Fixed first-frame colours; future motion/visibility bank stored. NOT prediction or 3D identity.')
    checksum=save_inference_checkpoint(packet,OUT/'motion.pt')
    np.savez_compressed(OUT/'capture_reference.npz',rgb=np.stack(frames),masks=np.stack(masks))
    sheet=Image.new('RGB',(W*2,H*2))
    for i,t in enumerate([0,10,21,32]):
        pic=frames[t].copy(); pic[masks[t]<.5]=(pic[masks[t]<.5]*.2).astype(np.uint8)
        sheet.paste(Image.fromarray(pic),((i%2)*W,(i//2)*H))
    sheet.save(OUT/'capture_masks.jpg')
    result=dict(capture_seconds=time.perf_counter()-start,points=len(base),frames=len(frames),
                tensor_bytes=packet_bytes(packet),file_bytes=(OUT/'motion.pt').stat().st_size,sha256=checksum,
                diagnostics=diagnostics,constant_colour=True,
                limitation='No new splats for disoccluded surfaces; tracking/segmentation are approximate. Full future vector bank grows with duration.')
    (OUT/'capture.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='diagnostics'},indent=2),flush=True)


class MotionReplay:
    """Current Gaussian state plus CPU-stored future motion bank; not autonomous AI."""
    def __init__(self,packet,device='cuda',mode='full'):
        self.packet=packet; self.mode=mode; self.points=packet['base'].to(device).clone(); self.index=0

    @torch.no_grad()
    def advance(self):
        if self.index>=len(self.packet['displacements']): raise StopIteration('End of captured motion')
        d=self.packet['displacements'][self.index].to(self.points.device)
        if self.mode=='full':
            self.points[:,:2]+=d
            self.points[:,2:4]+=self.packet['logscale_deltas'][self.index].to(self.points.device)
            angle=self.packet['angle_deltas'][self.index].to(self.points.device)
            axis=self.points[:,4:6].clone(); c,s=angle.cos(),angle.sin()
            self.points[:,4]=c*axis[:,0]-s*axis[:,1]; self.points[:,5]=s*axis[:,0]+c*axis[:,1]
            self.points[:,4:6]/=self.points[:,4:6].norm(dim=-1,keepdim=True).clamp_min(1e-8)
            self.points[:,9]=self.packet['visibility'][self.index+1].to(self.points.device)
        elif self.mode=='vectors_only':
            self.points[:,:2]+=d
        elif self.mode=='translation':
            self.points[:,:2]+=d.median(dim=0).values
        elif self.mode!='frozen': raise ValueError(self.mode)
        self.index+=1
        return self.points

    def snapshot(self):
        return dict(points=self.points.detach().cpu().clone(),index=self.index)

    def restore(self,snapshot):
        self.points=snapshot['points'].to(self.points.device).clone(); self.index=snapshot['index']


@torch.no_grad()
def replay():
    if (OUT/'replay.json').exists(): raise FileExistsError('Preserve replay')
    original_load=torch.load; original_import=builtins.__import__; loaded=[]
    def guard_load(path,*args,**kwargs):
        if Path(path).resolve()!=(OUT/'motion.pt').resolve(): raise AssertionError('Unexpected replay tensor input')
        loaded.append(str(path)); return original_load(path,*args,**kwargs)
    def guard_import(name,*args,**kwargs):
        if name=='diffusers' or name.startswith('diffusers.'): raise AssertionError('No diffusion in replay')
        return original_import(name,*args,**kwargs)
    def no_video(*args,**kwargs): raise AssertionError('No video decoding in replay')
    with patch('torch.load',side_effect=guard_load),patch('builtins.__import__',side_effect=guard_import),patch('cv2.VideoCapture',side_effect=no_video),patch('numpy.load',side_effect=no_video):
        packet=load_verified(OUT/'motion.pt'); background=packet['background'].cuda()
        bg,_=splat_layer(background); del background
        timings={}; outputs={}; coverage={}; velocities=[]
        for mode in ['full','vectors_only','translation','frozen']:
            session=MotionReplay(packet,mode=mode); imgs=[]; alphas=[]; times=[]
            torch.cuda.reset_peak_memory_stats()
            for t in range(len(packet['displacements'])+1):
                torch.cuda.synchronize(); start=time.perf_counter()
                if t: session.advance()
                rgb,alpha=splat_layer(session.points)
                composite=rgb*alpha+bg*(1-alpha)
                torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
                arr=array_image(composite); imgs.append(arr); alphas.append(alpha[0,0].cpu().numpy())
                if mode=='full':
                    if t==8: saved=session.snapshot()
                    if t==12: expected=session.snapshot()
                    overlay=Image.fromarray(arr); draw=ImageDraw.Draw(overlay)
                    host_points=session.points.cpu().numpy()
                    pos=host_points[:,:2]*H-.5
                    movement=packet['displacements'][min(t,31)].numpy()*H if t<32 else np.zeros_like(pos)
                    for j in range(0,len(pos),35):
                        if host_points[j,9]<.5: continue
                        a=tuple(pos[j]); b=tuple(pos[j]+movement[j]*3)
                        draw.line((a,b),fill='#00ffff',width=2)
                        draw.ellipse((b[0]-2,b[1]-2,b[0]+2,b[1]+2),fill='#ffff00')
                    velocities.append(np.array(overlay))
                    if t in [0,8,16,24,32]: overlay.save(OUT/f'vectors_{t:02d}.jpg')
            peak=torch.cuda.max_memory_allocated()
            if mode=='full':
                colours_unchanged=torch.equal(session.points[:,6:9].cpu(),packet['base'][:,6:9])
                session.restore(saved)
                for _ in range(4): session.advance()
                restore_error=(session.points.cpu()-expected['points']).abs().max().item()
                assert restore_error==0 and colours_unchanged
            timings[mode]=dict(median_update_render_ms_after4=float(np.median(times[4:])),peak_allocated_bytes=peak,
                               current_points_bytes=session.points.numel()*session.points.element_size(),all_ms=times)
            outputs[mode]=np.stack(imgs); coverage[mode]=np.stack(alphas)
            labeled=[]
            for arr in imgs:
                canvas=Image.new('RGB',(W,H+48),'#141820'); canvas.paste(Image.fromarray(arr),(0,48))
                draw=ImageDraw.Draw(canvas)
                draw.text((10,7),f'CAPTURED Gaussian motion | {mode} | no new action generated',fill='white')
                draw.text((10,27),'First-frame colours retained | Wheat: Neil Oakes, CC BY-SA 2.0',fill='white')
                labeled.append(np.array(canvas))
            imageio.mimwrite(OUT/f'{mode}.mp4',labeled,fps=16,codec='libx264',quality=8,macro_block_size=1)
        vector_frames=[]
        for arr in velocities:
            canvas=Image.new('RGB',(W,H+48),'#141820'); canvas.paste(Image.fromarray(arr),(0,48))
            draw=ImageDraw.Draw(canvas)
            draw.text((10,7),'Captured per-dot displacement to NEXT frame | cyan arrows shown at 3x length',fill='white')
            draw.text((10,27),'No source video at replay | Wheat: Neil Oakes, CC BY-SA 2.0',fill='white')
            vector_frames.append(np.array(canvas))
        imageio.mimwrite(OUT/'vectors_live.mp4',vector_frames,fps=8,codec='libx264',quality=8,macro_block_size=1)
        # Diagnostic RGB/alpha output, never a replay input.
        np.savez_compressed(OUT/'replay_outputs.npz',**outputs,**{k+'_alpha':v for k,v in coverage.items()})
    report=dict(only_tensor_inputs=loaded,no_video_or_reference_read_in_replay=True,no_diffusion=True,
                colours_exactly_fixed=colours_unchanged,snapshot_restore_max_error=restore_error,timings=timings,
                requires_stored_future_motion_bank=True,
                limits='Replay of extracted motion; no invented action, no learned dynamics, no source RGB at playback. Includes future trajectory bank; not bounded total history memory.')
    (OUT/'replay.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='timings'},indent=2),flush=True)


def evaluate():
    if (OUT/'comparison.mp4').exists(): raise FileExistsError('Preserve comparison')
    # Load compressed arrays once, not once per displayed frame.
    with np.load(OUT/'capture_reference.npz') as archive:
        reference={key:archive[key] for key in archive.files}
    with np.load(OUT/'replay_outputs.npz') as archive:
        results={key:archive[key] for key in archive.files}
    metrics={}
    for mode in ['full','vectors_only','translation','frozen']:
        predicted=results[mode+'_alpha']>.5; target=reference['masks']>.5
        intersection=(predicted&target).sum(axis=(1,2)); union=(predicted|target).sum(axis=(1,2))
        metrics[mode]=dict(mean_silhouette_iou=float((intersection/np.maximum(union,1)).mean()),
                           last_silhouette_iou=float(intersection[-1]/max(union[-1],1)),
                           note='Approximate source GrabCut masks, not independent ground-truth annotation')
    frames=[]
    for t in range(33):
        canvas=Image.new('RGB',(1248,560),'#141820'); d=ImageDraw.Draw(canvas)
        d.text((8,7),'CAPTURED GAUSSIAN MOTION: same first-frame colours; stored per-dot trajectories, not new action generation',fill='white')
        d.text((8,24),'Wheat: Neil Oakes / CC BY-SA 2.0 | source clip -> flow extraction -> motion archive -> Gaussian replay',fill='white')
        panels=[('Source Gaussian video (capture input)',reference['rgb'][t]),
                ('Stored motion + deformation + visibility',results['full'][t]),
                ('Displacements ONLY; all other attributes fixed',results['vectors_only'][t]),
                ('Whole-cutout translation control',results['translation'][t]),
                ('Motion disabled: frozen control',results['frozen'][t]),
                ('Source mask (approximate segmentation)',np.repeat((reference['masks'][t]*255).astype(np.uint8)[:,:,None],3,2))]
        for i,(label,arr) in enumerate(panels):
            x,y=(i%3)*416,48+(i//3)*256
            d.text((x+5,y),label,fill='white'); canvas.paste(Image.fromarray(arr).resize((416,240)),(x,y+16))
        frames.append(np.asarray(canvas))
        if t in [0,10,21,32]: canvas.save(OUT/f'comparison_{t:02d}.jpg')
    imageio.mimwrite(OUT/'comparison.mp4',frames,fps=16,codec='libx264',quality=8,macro_block_size=1)
    imageio.mimwrite(OUT/'comparison_slow.mp4',frames,fps=4,codec='libx264',quality=8,macro_block_size=1)
    (OUT/'evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['capture','replay','evaluate']); args=p.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(4)
    with keep_windows_awake(): {'capture':capture,'replay':replay,'evaluate':evaluate}[args.mode]()


if __name__=='__main__': main()
