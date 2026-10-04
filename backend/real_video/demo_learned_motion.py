"""Isolated learned forecast from three observed frames; future RGB only in eval."""
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
import torch.nn.functional as F

from .checkpoint_io import digest,load_verified,keep_windows_awake
from .learned_motion import LearnedGaussianMotion
from .dense_gaussian_seed import OUT as SEEDS,SOURCE,FixedGeometrySplat
from .edit_gaussian_memory import splat_layer,array_image
from .gaussian_motion_memory import select_cat,H,W

ROOT=Path('artifacts/real_video/learned_motion/v1')


def transfer_displacement(coarse_delta,uv):
    return F.grid_sample(coarse_delta,uv[None,None],mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T


def sampled_jacobian(position,uv,aspect=W/H):
    gh,gw=position.shape[-2:]
    dy,dx=torch.gradient(position,spacing=(1/gh,aspect/gw),dim=(-2,-1))
    fields=torch.cat((dx[:,:1],dy[:,:1],dx[:,1:2],dy[:,1:2]),1)
    return transfer_displacement(fields,uv).reshape(-1,2,2)


def deform_covariance(points,jacobian):
    # Standard covariance push-forward, not a learned anatomical deformation.
    u,s,vh=torch.linalg.svd(jacobian)
    jacobian=(u*s.clamp(.5,2)[:,None,:])@vh
    axis=points[:,4:6]; perpendicular=torch.stack((-axis[:,1],axis[:,0]),-1)
    frame=torch.stack((axis,perpendicular),-1)
    basis=jacobian@(frame*points[:,2:4].exp()[:,None,:])
    sigma=basis@basis.transpose(-1,-2)
    # Closed-form symmetric 2x2 eigensystem avoids cuSOLVER's large-batch limit.
    a,b,d=sigma[:,0,0],sigma[:,0,1],sigma[:,1,1]
    gap=((a-d).square()+4*b.square()).clamp_min(0).sqrt()
    values=torch.stack(((a+d+gap)/2,(a+d-gap)/2),-1)
    theta=.5*torch.atan2(2*b,a-d)
    result=points.clone(); result[:,2:4]=.5*values.clamp_min(1e-12).log()
    result[:,4:6]=torch.stack((theta.cos(),theta.sin()),-1)
    return result


@torch.no_grad()
def predict(spacing,transport_shape=False,root=None):
    root=ROOT if root is None else Path(root)
    suffix='_shape' if transport_shape else ''
    out=root/f'forecast_{spacing}px{suffix}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists(): raise FileExistsError('Preserve forecast')
    allowed={(root/'model.pt').resolve(),(SEEDS/f'seed_{spacing}px.pt').resolve()}
    original=torch.load; original_import=builtins.__import__; loaded=[]
    def guard(path,*args,**kwargs):
        if Path(path).resolve() not in allowed: raise AssertionError('Unexpected tensor input')
        loaded.append(str(path)); return original(path,*args,**kwargs)
    def no_frames(*args,**kwargs): raise AssertionError('No video/image/NPZ input during prediction')
    def guard_import(name,*args,**kwargs):
        if name=='diffusers' or name.startswith('diffusers.'): raise AssertionError('No diffusion')
        return original_import(name,*args,**kwargs)
    results={}; timing={}; snapshots={}; counts=[]
    with patch('torch.load',side_effect=guard),patch('cv2.VideoCapture',side_effect=no_frames),patch('numpy.load',side_effect=no_frames),patch('PIL.Image.open',side_effect=no_frames),patch('builtins.__import__',side_effect=guard_import):
        seed=load_verified(SEEDS/f'seed_{spacing}px.pt'); weights=load_verified(root/'model.pt')
        assert seed['decoded_source_frames']==3
        model=LearnedGaussianMotion(weights['config']['hidden']).eval().cuda(); model.load_state_dict(weights['model'])
        observed=seed['observed_coarse'].cuda(); base=seed['points'].cuda(); uv=seed['sample_uv'].cuda()
        bg,_=splat_layer(seed['background'].cuda()); ref=observed[:,:2,2].clone()
        if transport_shape:
            inverse_ref=torch.linalg.pinv(sampled_jacobian(ref,uv))
        for mode in ['learned','velocity','frozen','zero_hidden']:
            state=model.initialize(observed[:,:,0],observed[:,:,1],observed[:,:,2],dt=.5)
            images=[]; alphas=[]; times=[]; movement=[]; torch.cuda.reset_peak_memory_stats()
            for t in range(25):
                torch.cuda.synchronize(); start=time.perf_counter()
                if t:
                    if mode in ['learned','zero_hidden']: state=model.step(state,dt=.5,reset_memory=mode=='zero_hidden')
                    elif mode=='velocity': state=dict(state,fields=torch.cat((state['fields'][:,:2]+.5*state['velocity'],state['fields'][:,2:]),1))
                dense_delta=transfer_displacement(state['fields'][:,:2]-ref,uv)
                points=base.clone(); points[:,:2]+=dense_delta
                if transport_shape and t and mode!='frozen':
                    jacobian=sampled_jacobian(state['fields'][:,:2],uv)@inverse_ref
                    points=deform_covariance(points,jacobian)
                renderer=FixedGeometrySplat(points,radius=seed['radius']*(2 if transport_shape else 1))
                rgb,alpha=renderer.render((points[:,6:9]+1)/2); composite=rgb*alpha+bg*(1-alpha)
                torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
                images.append(array_image(composite)); alphas.append(alpha[0,0].cpu().numpy())
                movement.append(float((dense_delta*H).norm(dim=-1).mean()))
                if mode=='learned':
                    if t in [4,8]: snapshots[t]={k:v.detach().cpu().clone() for k,v in state.items()}
                    counts.append(sum(v.numel()*v.element_size() for v in state.values()))
                del renderer,rgb,alpha,composite,points,dense_delta
            timing[mode]=dict(median_update_render_ms_after4=float(np.median(times[4:])),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              all_ms=times,mean_displacement_pixels=movement)
            results[mode]=np.stack(images); results[mode+'_alpha']=np.stack(alphas)
            writer=imageio.get_writer(out/f'{mode}.mp4',fps=16,codec='libx264',quality=8,macro_block_size=1)
            try:
                for t,arr in enumerate(images):
                    canvas=Image.new('RGB',(W,H+48),'#141820'); canvas.paste(Image.fromarray(arr),(0,48)); draw=ImageDraw.Draw(canvas)
                    draw.text((10,7),f'{mode.upper()} | {len(base):,} Gaussian cat points | '+('observed seed fit' if t==0 else f'forecast +{t}/16 sec; NO future anchors'),fill='white')
                    draw.text((10,27),'3 observed frames; fixed fitted colours | Wheat: Neil Oakes / CC BY-SA 2.0',fill='white')
                    writer.append_data(np.asarray(canvas))
            finally: writer.close()
            if mode=='learned':
                for frame in [0,6,12,24]: Image.fromarray(images[frame]).save(out/f'forecast_{frame:02d}.jpg')
        state={k:v.cuda().clone() for k,v in snapshots[4].items()}
        for _ in range(4): state=model.step(state,dt=.5)
        difference=max((state[k].cpu()-snapshots[8][k]).abs().max().item() for k in state)
        assert difference<1e-6 and len(set(counts))==1
        assert torch.equal(base[:,6:9].cpu(),seed['points'][:,6:9])
        np.savez_compressed(out/'outputs.npz',**results)
        audit=dict(input_paths=loaded,input_seed_sha256=digest(SEEDS/f'seed_{spacing}px.pt'),model_sha256=digest(root/'model.pt'),
                   observed_frames=3,future_anchor_count=0,future_motion_bank_count=0,forecast_frames=24,
                   no_video_image_or_npz_input=True,no_diffusion=True,fitted_colours_unchanged=True,
                   save_restore_max_error=difference,current_coarse_state_bytes=counts[0],
                   dense_gaussian_bytes=base.numel()*base.element_size(),dense_mapping_bytes=uv.numel()*uv.element_size(),
                   coarse_state_size_constant=True,dense_gaussians=len(base),spacing=spacing,timing=timing,
                   covariance_transport=transport_shape,local_stretch_clamp=[.5,2] if transport_shape else None,
                   source_reconstruction_only_for_observed_seed=True,
                   limits='Small real-UCF learned predictor; cat is OOD. Fine splats follow interpolated coarse predicted displacement. No anatomy, new surfaces, text controls, new appearance or depth.')
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='timing'},indent=2),flush=True)


def evaluate(transport_shape=False):
    suffix='_shape' if transport_shape else ''
    if (ROOT/f'cat_evaluation{suffix}.json').exists(): raise FileExistsError('Preserve evaluation')
    # Only AFTER isolated forecasts have been saved, decode reference future frames.
    cap=cv2.VideoCapture(str(SOURCE)); source=[]
    try:
        for _ in range(27):
            ok,bgr=cap.read()
            if not ok: raise ValueError('Reference too short')
            source.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    finally: cap.release()
    masks=np.stack([select_cat(frame) for frame in source[2:]])
    seed=load_verified(SEEDS/'seed_1px.pt')
    with torch.no_grad(): bg=array_image(splat_layer(seed['background'].cuda())[0]).astype(np.float32)/255
    target=np.stack(source[2:]).astype(np.float32)/255*masks[...,None]+bg[None]*(1-masks[...,None])
    all_results={}; metrics={}
    for spacing in [4,2,1]:
        out=ROOT/f'forecast_{spacing}px{suffix if spacing==1 else ""}'
        with np.load(out/'outputs.npz') as f: results={k:f[k] for k in f.files}
        all_results[spacing]=results; modes={}
        for mode in ['learned','velocity','frozen','zero_hidden']:
            error=((results[mode].astype(np.float32)/255-target)**2).mean((1,2,3))
            a=results[mode+'_alpha']>.5; m=masks>.5
            iou=(a&m).sum((1,2))/np.maximum((a|m).sum((1,2)),1)
            modes[mode]=dict(future24_rgb_mse=float(error[1:].mean()),future12_rgb_mse=float(error[1:13].mean()),
                             future24_psnr=float(-10*np.log10(max(error[1:].mean(),1e-12))),future24_silhouette_iou=float(iou[1:].mean()),
                             seed_mse=float(error[0]),note='OOD cat development reference, source-mask composite; not independent final-test or broad generative quality')
        metrics[str(spacing)]=modes
    writer=imageio.get_writer(ROOT/f'density_forecast_comparison{suffix}.mp4',fps=8,codec='libx264',quality=8,macro_block_size=1)
    try:
        for t in range(25):
            canvas=Image.new('RGB',(1248,576),'#141820'); d=ImageDraw.Draw(canvas)
            d.text((8,8),'DENSE GAUSSIAN LEARNED FORECAST | 3 observed frames only | No future anchors; half-speed inspection',fill='white')
            panels=[('Original Wan reference (evaluation only)',source[t+2]),
                    ('~5.9k Gaussian learned forecast',all_results[4]['learned'][t]),
                    ('~23.9k Gaussian learned forecast',all_results[2]['learned'][t]),
                    ('~95.7k learned'+(' + covariance transport' if transport_shape else ''),all_results[1]['learned'][t]),
                    ('~95.7k constant-velocity control',all_results[1]['velocity'][t]),
                    ('~95.7k frozen control',all_results[1]['frozen'][t])]
            for i,(label,arr) in enumerate(panels):
                x,y=(i%3)*416,40+(i//3)*264; d.text((x+5,y),label,fill='white'); canvas.paste(Image.fromarray(arr).resize((416,240)),(x,y+20))
            writer.append_data(np.asarray(canvas))
            if t in [0,6,12,24]: canvas.save(ROOT/f'density_forecast{suffix}_{t:02d}.jpg')
    finally: writer.close()
    (ROOT/f'cat_evaluation{suffix}.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['predict','evaluate']); p.add_argument('--spacing',type=int,choices=[1,2,4],default=1); p.add_argument('--transport-shape',action='store_true'); args=p.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(4)
    with keep_windows_awake(): predict(args.spacing,args.transport_shape) if args.mode=='predict' else evaluate(args.transport_shape)


if __name__=='__main__': main()
