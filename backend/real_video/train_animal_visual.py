"""Render-supervised control refinement. Future RGB is training/eval target only."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .animal_graph_data import DATA,HEIGHT
from .prepare_animal_motion import WORK
from .animal_control_graph import AnimalControlGraph,bind_points,deform_points
from .demo_animal_graph import image_points
from .dense_gaussian_seed import FixedGeometrySplat
from .train_animal_graph import OUT as BASE,batch,initialize,target_loss
from .checkpoint_io import load_verified,save_inference_checkpoint,save_training_checkpoint,digest,keep_windows_awake

CACHE=WORK/'visual_targets_v1.pt'
OUT=Path('artifacts/real_video/animal_graph/visual_v1')


def prepare():
    if CACHE.exists(): raise FileExistsError('Preserve visual target cache')
    data=load_verified(DATA); root=WORK/'extracted/DAVIS'; records=[]
    for i,meta in enumerate(data['metadata']):
        paths=sorted((root/'JPEGImages/480p'/meta['sequence']).glob('*.jpg')); frames=[]; masks=[]
        for j in range(15):
            path=paths[meta['raw_start']+j*meta['raw_stride']]
            image=np.array(Image.open(path).convert('RGB')); width=round(image.shape[1]*HEIGHT/image.shape[0]); image=cv2.resize(image,(width,HEIGHT),interpolation=cv2.INTER_AREA)
            mask=np.array(Image.open(root/'Annotations/480p'/meta['sequence']/(path.stem+'.png')))==meta['object']
            mask=cv2.resize(mask.astype(np.float32),(width,HEIGHT),interpolation=cv2.INTER_NEAREST)
            frames.append(image); masks.append(mask)
        points=image_points(frames[2],masks[2]); pixel=(points[:,:2]*HEIGHT-.5).round().long(); chosen=((pixel%2)==0).all(-1)
        points=points[chosen].clone(); points[:,2:4]=np.log(1.3/HEIGHT)
        index,weight=bind_points(points[:,:2],data['position'][i,2],mask=torch.from_numpy(masks[2]))
        records.append(dict(points=points,index=index,weight=weight,frames=torch.from_numpy(np.stack(frames)),masks=torch.from_numpy(np.stack(masks)).to(torch.uint8)))
        if (i+1)%25==0: print(json.dumps(dict(stage='visual_cache',windows=i+1)),flush=True)
    sha=save_inference_checkpoint(dict(records=records,metadata=data['metadata'],source_tracks_sha256=digest(DATA),scope='Future images/masks are training and evaluation targets only; forbidden at prediction'),CACHE)
    print(json.dumps(dict(sha256=sha,windows=len(records))),flush=True)


def render_loss(state,sample,target,t):
    points=target['points']; position=state['position'][0]; angle=state['angle'][0]-state['reference_angle'][0]
    moved=deform_points(points,state['reference'][0],position,angle,state['visibility'][0],target['index'],target['weight'])
    h,w=target['masks'].shape[-2:]; splat=FixedGeometrySplat(moved,height=h,width=w,radius=3)
    rgb,alpha=splat.render((points[:,6:9]+1)/2); mask=target['masks'][t][None,None].float()
    truth=target['frames'][t].permute(2,0,1)[None].float()/255*mask+.15*(1-mask)
    composite=rgb*alpha+.15*(1-alpha)
    mse=(composite-truth).square().mean()
    dice=1-(2*(alpha*mask).sum()+1)/(alpha.sum()+mask.sum()+1)
    loss=mse+.1*dice+.001*target_loss(state,sample,t)
    return loss,mse,dice,alpha,mask


def to_device(record):
    return {key:value.cuda() for key,value in record.items()}


@torch.no_grad()
def evaluate(model,data,cache,indices,mode='learned'):
    model.eval(); cases=[]
    for i in indices:
        sample=batch(data,[i]); target=to_device(cache['records'][i]); state=initialize(model,sample); rows=[]
        for t in range(3,15):
            if mode in ['learned','zero_hidden']: state=model.step(state,reset_memory=mode=='zero_hidden')
            elif mode=='velocity': state=dict(state,position=state['position']+state['velocity'],angle=state['angle']+state['angular_velocity'])
            elif mode=='damped_velocity':
                v=state['velocity']*.9; omega=state['angular_velocity']*.9
                state=dict(state,position=state['position']+v,angle=state['angle']+omega,velocity=v,angular_velocity=omega)
            elif mode!='frozen': raise ValueError(mode)
            loss,mse,dice,alpha,mask=render_loss(state,sample,target,t)
            pred=alpha>.5; truth=mask>.5; iou=(pred&truth).sum()/(pred|truth).sum().clamp_min(1)
            rows.append(dict(loss=float(loss),rgb_mse=float(mse),silhouette_dice_error=float(dice),mask_iou=float(iou)))
        cases.append(dict(sequence=data['metadata'][i]['sequence'],index=i,scores={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}))
    groups=sorted({c['sequence'] for c in cases}); means={}
    for key in cases[0]['scores']:
        means[key]=float(np.mean([np.mean([c['scores'][key] for c in cases if c['sequence']==g]) for g in groups]))
    return dict(means=means,cases=cases)


def train():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): raise FileExistsError('Preserve visual experiment')
    config=dict(seed=430402,steps=600,lr=.0001,hidden=64,
                objective='RGB MSE + 0.1 silhouette Dice error + 0.001 control target loss',
                selection='minimum sequence-macro validation objective every200 steps; step0 allowed',
                input='3 observations only; future sampled frame is supervised training target',
                training='Source-disjoint train/validation metadata from cache; exact contributing counts stored in results; no diagnostic cat or final test',
                limits='2-pixel observed splat seed, no new appearance. Fine-tunes control model not Wan. Adaptive validation development.')
    (OUT/'protocol.json').write_text(json.dumps(config,indent=2))
    data=load_verified(DATA); cache=load_verified(CACHE); initial=load_verified(BASE/'model.pt')
    assert cache['metadata']==data['metadata']
    if cache['source_tracks_sha256']!=digest(DATA): raise ValueError('Visual cache is bound to different control tracks')
    train_idx=[i for i,m in enumerate(data['metadata']) if m['split']=='train']; val=[i for i,m in enumerate(data['metadata']) if m['split']=='validation']
    assert not ({data['metadata'][i]['sequence'] for i in train_idx}&{data['metadata'][i]['sequence'] for i in val})
    model=AnimalControlGraph().cuda(); model.load_state_dict(initial['model']); optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'])
    torch.manual_seed(config['seed']); baseline=evaluate(model,data,cache,val); best=baseline['means']['loss']; history=[dict(step=0,validation=baseline)]
    save_training_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,step=0,validation=baseline),OUT,0,True)
    (OUT/'baseline.json').write_text(json.dumps(baseline,indent=2)); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for step in range(1,config['steps']+1):
        model.train(); i=train_idx[int(torch.randint(len(train_idx),()).item())]; t=int(torch.randint(3,15,()).item())
        sample=batch(data,[i]); target=to_device(cache['records'][i]); state=initialize(model,sample)
        for _ in range(3,t+1): state=model.step(state)
        loss,_,_,_,_=render_loss(state,sample,target,t)
        if not torch.isfinite(loss): raise ValueError('Nonfinite visual loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
        if step%50==0:
            progress=dict(step=step,total=config['steps'],loss=float(loss.detach()),seconds=time.perf_counter()-start)
            (OUT/'progress.json').write_text(json.dumps(progress,indent=2)); print(json.dumps(progress),flush=True)
        if step%200==0:
            score=evaluate(model,data,cache,val); improved=score['means']['loss']<best; best=min(best,score['means']['loss']); history.append(dict(step=step,validation=score))
            save_training_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,step=step,validation=score,
                                          rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),OUT,step,improved)
            (OUT/'history.json').write_text(json.dumps(history,indent=2)); print(json.dumps(dict(step=step,validation=score['means'])),flush=True)
    selected=load_verified(OUT/'best.pt'); sha=save_inference_checkpoint(dict(model=selected['model'],config=config,selected_step=selected['step'],base_sha256=digest(BASE/'model.pt'),
                                                                          tracks_sha256=digest(DATA),visual_targets_sha256=digest(CACHE)),OUT/'model.pt')
    report=dict(selected_step=selected['step'],baseline=baseline['means'],validation=selected['validation']['means'],sha256=sha,train_windows=len(train_idx),validation_windows=len(val),
                loop_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),wan_weights_changed=False,visually_accepted=False)
    (OUT/'results.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


def main():
    global DATA,CACHE,OUT,BASE
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=['prepare','train'])
    parser.add_argument('--data',type=Path,default=DATA); parser.add_argument('--cache',type=Path,default=CACHE); parser.add_argument('--output',type=Path,default=OUT); parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args(); DATA=args.data; CACHE=args.cache; OUT=args.output; BASE=args.base
    torch.set_num_threads(4); cv2.setNumThreads(4)
    with keep_windows_awake(): prepare() if args.mode=='prepare' else train()


if __name__=='__main__': main()
