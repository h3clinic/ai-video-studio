"""Isolated graph forecast; reference frames are opened only by separate eval."""
import argparse
import builtins
import json
import time
from pathlib import Path
from unittest.mock import patch
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import torch
from .animal_control_graph import AnimalControlGraph,bind_points,deform_points
from .animal_graph_data import HEIGHT,track_controls
from .prepare_animal_motion import WORK,VAL
from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake
from .dense_gaussian_seed import OUT as DENSE,SOURCE,FixedGeometrySplat
from .gaussian_motion_memory import select_cat
from .edit_gaussian_memory import array_image,splat_layer
from .train_animal_graph import OUT
SEED_ROOT=OUT/'seeds'


def image_points(frame,mask):
    h,w=mask.shape; yy,xx=np.where(mask>.5); xy=np.stack((xx,yy),-1)
    points=np.zeros((len(xy),10),np.float32); points[:,:2]=(xy+.5)/h
    points[:,2:4]=np.log(.65/h); points[:,4]=1; points[:,6:9]=frame[yy,xx]/127.5-1; points[:,9]=1
    return torch.from_numpy(points)


def make_seed(frames,masks,points,background,name):
    controls=track_controls(frames,masks)
    ref=controls['position'][2]; indices,weights=bind_points(points[:,:2],ref,mask=torch.from_numpy(masks[2]).float())
    return dict(controls={k:v[:3] if k in ['position','angle','visibility','confidence'] else v for k,v in controls.items()},
                points=points,index=indices,weight=weights,background=background,name=name,
                observed_frames=3,source='observed recorded frames' if name!='cat' else 'observed original Wan frames',
                time_units='one sampled frame step; DAVIS stride2, cat stride1; physical FPS not verified')


def prepare():
    folder=SEED_ROOT; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists(): raise FileExistsError('Preserve seeds')
    records=[]
    for name in ['cat',*VAL]:
        frames=[]; masks=[]
        if name=='cat':
            cap=cv2.VideoCapture(str(SOURCE))
            try:
                for _ in range(3):
                    ok,bgr=cap.read()
                    if not ok: raise ValueError('Missing observed frame')
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); mask=select_cat(rgb)
                    frames.append(cv2.resize(rgb,(round(rgb.shape[1]*HEIGHT/rgb.shape[0]),HEIGHT),interpolation=cv2.INTER_AREA))
                    masks.append(cv2.resize(mask,(frames[-1].shape[1],HEIGHT),interpolation=cv2.INTER_NEAREST))
            finally: cap.release()
            old=load_verified(DENSE/'seed_1px.pt'); points=old['points']; background=old['background']; metadata=dict(name=name,source_sha256=digest(SOURCE),raw_stride=1,raw_start=0)
        else:
            root=WORK/'extracted/DAVIS'; paths=sorted((root/'JPEGImages/480p'/name).glob('*.jpg'))
            first=np.array(Image.open(root/'Annotations/480p'/name/(paths[0].stem+'.png'))); ids=sorted(set(np.unique(first))-{0,255}); object_id=int(ids[0])
            for i in [0,2,4]:
                rgb=np.array(Image.open(paths[i]).convert('RGB')); width=round(rgb.shape[1]*HEIGHT/rgb.shape[0]); frames.append(cv2.resize(rgb,(width,HEIGHT),interpolation=cv2.INTER_AREA))
                mask=np.array(Image.open(root/'Annotations/480p'/name/(paths[i].stem+'.png')))==object_id
                masks.append(cv2.resize(mask.astype(np.float32),(width,HEIGHT),interpolation=cv2.INTER_NEAREST))
            points=image_points(frames[2],masks[2]); background=torch.full((1,3,HEIGHT,frames[2].shape[1]),.15)
            metadata=dict(name=name,object=object_id,raw_stride=2,raw_start=0,selection='first object ID in first frame, not selected for output quality')
        seed=make_seed(frames,masks,points,background,name); seed.update(metadata)
        sha=save_inference_checkpoint(seed,folder/f'{name}.pt'); records.append(dict(**metadata,gaussians=len(points),sha256=sha))
        print(json.dumps(records[-1]),flush=True)
    (folder/'manifest.json').write_text(json.dumps(records,indent=2))


@torch.no_grad()
def predict():
    folder=OUT/'forecast'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'audit.json').exists(): raise FileExistsError('Preserve forecast')
    names=['cat',*VAL]; allowed={(OUT/'model.pt').resolve(),*((SEED_ROOT/f'{name}.pt').resolve() for name in names)}
    original_load=torch.load; loaded=[]
    def guarded_load(path,*args,**kwargs):
        if Path(path).resolve() not in allowed: raise AssertionError('Unexpected tensor input')
        loaded.append(str(path)); return original_load(path,*args,**kwargs)
    def forbidden(*args,**kwargs): raise AssertionError('No RGB/video/NPZ reads during forecast')
    original_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name=='diffusers' or name.startswith('diffusers.'): raise AssertionError('No diffusion inputs')
        return original_import(name,*args,**kwargs)
    audit={}
    with patch('torch.load',side_effect=guarded_load),patch('cv2.VideoCapture',side_effect=forbidden),patch('PIL.Image.open',side_effect=forbidden),patch('numpy.load',side_effect=forbidden),patch('builtins.__import__',side_effect=guarded_import):
        checkpoint=load_verified(OUT/'model.pt'); model=AnimalControlGraph().cuda().eval(); model.load_state_dict(checkpoint['model'])
        for name in names:
            seed=load_verified(SEED_ROOT/f'{name}.pt'); ctrl={k:v[None].cuda() for k,v in seed['controls'].items()}
            points=seed['points'].cuda(); index=seed['index'].cuda(); weight=seed['weight'].cuda()
            if name=='cat': background=splat_layer(seed['background'].cuda())[0]; h,w=480,832
            else: background=seed['background'].cuda(); h,w=background.shape[-2:]
            modes=['learned','velocity','frozen','zero_hidden']; results={}; statistics={}
            for mode in modes:
                state=model.initialize(ctrl['position'],ctrl['angle'],ctrl['visibility'],ctrl['colour'],ctrl['adjacency'])
                frames=[]; masks=[]; times=[]; bytes_state=[]; changes=[]; torch.cuda.reset_peak_memory_stats()
                snapshot=None
                for t in range(13):
                    torch.cuda.synchronize(); start=time.perf_counter()
                    if t:
                        if mode in ['learned','zero_hidden']: state=model.step(state,reset_memory=mode=='zero_hidden')
                        elif mode=='velocity': state=dict(state,position=state['position']+state['velocity'],angle=state['angle']+state['angular_velocity'])
                    deformed=deform_points(points,state['reference'][0],state['position'][0],state['angle'][0]-state['reference_angle'][0],state['visibility'][0],index,weight)
                    renderer=FixedGeometrySplat(deformed,height=h,width=w,radius=2)
                    rgb,alpha=renderer.render((deformed[:,6:9]+1)/2); composite=rgb*alpha+background*(1-alpha)
                    torch.cuda.synchronize(); times.append(1000*(time.perf_counter()-start))
                    frames.append(array_image(composite)); masks.append(alpha[0,0].cpu().numpy())
                    bytes_state.append(sum(v.numel()*v.element_size() for v in state.values())); changes.append(float((deformed[:,:2]-points[:,:2]).norm(dim=-1).mean()*h))
                    assert torch.equal(deformed[:,6:9],points[:,6:9])
                    if t==4: snapshot={k:v.clone() for k,v in state.items()}
                    if t==8 and mode=='learned':
                        restored=snapshot
                        for _ in range(4): restored=model.step(restored)
                        assert all(torch.equal(restored[k],state[k]) for k in state)
                    del renderer,rgb,alpha,composite,deformed
                statistics[mode]=dict(median_update_render_ms=float(np.median(times[3:])),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      state_bytes=bytes_state[0],state_size_constant=len(set(bytes_state))==1,mean_displacement_pixels=changes)
                results[mode]=np.stack(frames); results[mode+'_alpha']=np.stack(masks)
                if name=='cat' and mode=='learned':
                    for t in [0,6,12]: Image.fromarray(frames[t]).save(folder/f'cat_{t:02d}.jpg')
            np.savez_compressed(folder/f'{name}.npz',**results)
            audit[name]=dict(gaussians=len(points),controls=48,future_rgb_inputs=0,future_tracks_inputs=0,
                             model_sha256=digest(OUT/'model.pt'),seed_sha256=digest(SEED_ROOT/f'{name}.pt'),
                             colours_unchanged=True,restore_exact=True,statistics=statistics)
            print(json.dumps(dict(name=name,learned=statistics['learned'])),flush=True)
        audit['inputs']=loaded; audit['limits']='Conditional planar motion forecast, not text-to-video or novel surfaces. Source frames only in observed seed fitting, reference future only in subsequent evaluation.'
    (folder/'audit.json').write_text(json.dumps(audit,indent=2))


def evaluate():
    folder=OUT/'evaluation'; folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists(): raise FileExistsError('Preserve evaluation')
    report={}
    for name in ['cat',*VAL]:
        seed=load_verified(SEED_ROOT/f'{name}.pt'); frames=[]; masks=[]
        if name=='cat':
            cap=cv2.VideoCapture(str(SOURCE)); all_frames=[]
            try:
                for _ in range(15):
                    ok,bgr=cap.read()
                    if not ok: raise ValueError('Missing reference')
                    all_frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            finally: cap.release()
            frames=all_frames[2:]; masks=[select_cat(f) for f in frames]
            with torch.no_grad(): background=array_image(splat_layer(seed['background'].cuda())[0]).astype(np.float32)/255
        else:
            root=WORK/'extracted/DAVIS'; paths=sorted((root/'JPEGImages/480p'/name).glob('*.jpg'))
            for i in range(4,29,2):
                rgb=np.array(Image.open(paths[i]).convert('RGB')); width=round(rgb.shape[1]*HEIGHT/rgb.shape[0]); frames.append(cv2.resize(rgb,(width,HEIGHT),interpolation=cv2.INTER_AREA))
                mask=np.array(Image.open(root/'Annotations/480p'/name/(paths[i].stem+'.png')))==seed['object']
                masks.append(cv2.resize(mask.astype(np.float32),(width,HEIGHT),interpolation=cv2.INTER_NEAREST))
            background=seed['background'][0].permute(1,2,0).numpy()
        masks=np.stack(masks); reference=np.stack(frames).astype(np.float32)/255*masks[...,None]+background*(1-masks[...,None])
        with np.load(OUT/'forecast'/f'{name}.npz') as f: forecast={k:f[k] for k in f.files}
        scores={}
        for mode in ['learned','velocity','frozen','zero_hidden']:
            output=forecast[mode].astype(np.float32)/255; mse=((output-reference)**2).mean((1,2,3)); predmask=forecast[mode+'_alpha']>.5; gt=masks>.5
            iou=(predmask&gt).sum((1,2))/np.maximum((predmask|gt).sum((1,2)),1)
            union=(predmask|gt).astype(np.float32); region=((output-reference)**2).sum(-1)
            foreground_mse=(region*union).sum((1,2))/(3*np.maximum(union.sum((1,2)),1))
            scores[mode]=dict(future12_mse=float(mse[1:].mean()),future12_psnr=float(-10*np.log10(max(mse[1:].mean(),1e-12))),
                              future12_mask_iou=float(iou[1:].mean()),foreground_union_mse=float(foreground_mse[1:].mean()),seed_mse=float(mse[0]))
        report[name]=scores
        writer=imageio.get_writer(folder/f'{name}_comparison.mp4',fps=8,codec='libx264',quality=8,macro_block_size=1)
        try:
            for t in range(13):
                canvas=Image.new('RGB',(832,600),'#141820'); draw=ImageDraw.Draw(canvas)
                draw.text((8,6),f'{name.upper()} | 48 shared Gaussian controls | 3 observed frames -> autonomous forecast',fill='white')
                draw.text((8,22),'Conditional motion experiment, NOT new text-to-video. Reference is evaluation only.',fill='white')
                panels=[('Recorded/Wan object reference',np.uint8(np.clip(reference[t],0,1)*255)),('Learned graph',forecast['learned'][t]),
                        ('Constant-velocity graph',forecast['velocity'][t]),('Frozen graph',forecast['frozen'][t])]
                for j,(label,img) in enumerate(panels):
                    x,y=j%2*416,42+j//2*266; draw.text((x+5,y),label,fill='white'); canvas.paste(Image.fromarray(img).resize((416,240)),(x,y+20))
                draw.text((8,583),'DAVIS: Pont-Tuset et al. 2017. Cat wheat: Neil Oakes, CC BY-SA 2.0.',fill='white')
                writer.append_data(np.asarray(canvas))
                if t in [0,6,12]: canvas.save(folder/f'{name}_{t:02d}.jpg')
        finally: writer.close()
    (folder/'results.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


def main():
    global OUT,SEED_ROOT
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=['prepare','predict','evaluate'])
    parser.add_argument('--root',type=Path,default=OUT); parser.add_argument('--seed-root',type=Path,default=SEED_ROOT); args=parser.parse_args()
    OUT=args.root; SEED_ROOT=args.seed_root
    torch.set_num_threads(4); cv2.setNumThreads(4)
    with keep_windows_awake(): {'prepare':prepare,'predict':predict,'evaluate':evaluate}[args.mode]()


if __name__=='__main__': main()
