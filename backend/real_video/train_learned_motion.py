"""Train causal Gaussian displacement forecasting on real, group-disjoint clips."""
import json
from pathlib import Path
import time
import numpy as np
import torch

from .checkpoint_io import load_verified,save_training_checkpoint,save_inference_checkpoint,keep_windows_awake,digest
from .learned_motion import LearnedGaussianMotion
from .persistent_gaussian import render_state

OUT=Path('artifacts/real_video/learned_motion/v1')
DATA=Path('../../work/real_video/persistent_gaussian_tracks_v1.pt')


def weights(sequence):
    speed=(sequence[:,:2,2]-sequence[:,:2,1]).norm(dim=1,keepdim=True)
    return 1+4*(speed>.005).float()


def loss(predicted,target,previous_target,weight):
    position=((predicted['fields'][:,:2]-target[:,:2])*64).square().sum(1,keepdim=True)
    velocity=((predicted['velocity']-(target[:,:2]-previous_target[:,:2]))*64).square().sum(1,keepdim=True)
    return ((position+.5*velocity)*weight).sum()/weight.sum()


@torch.no_grad()
def evaluate(model,data,indices,baselines=False):
    model.eval(); records=[]
    modes=['learned','frozen','velocity','zero_hidden'] if baselines else ['learned']
    for i in indices:
        sequence=data['fields'][i:i+1].cuda(); weight=weights(sequence); scores={}
        for mode in modes:
            state=model.initialize(sequence[:,:,0],sequence[:,:,1],sequence[:,:,2]); losses=[]; epe=[]; pixels=[]
            for t in range(3,9):
                if mode in ['learned','zero_hidden']: state=model.step(state,reset_memory=mode=='zero_hidden')
                elif mode=='velocity': state=dict(state,fields=torch.cat((state['fields'][:,:2]+state['velocity'],state['fields'][:,2:]),1))
                elif mode=='frozen': state=dict(state,velocity=torch.zeros_like(state['velocity']))
                losses.append(loss(state,sequence[:,:,t],sequence[:,:,t-1],weight).item())
                error=((state['fields'][:,:2]-sequence[:,:2,t])*64).norm(dim=1,keepdim=True)
                epe.append((error*weight).sum().item()/weight.sum().item())
                if baselines:
                    rgb=render_state(state['fields'])
                    truth=data['videos'][i,t].permute(2,0,1).cuda().float()/255
                    pixels.append((rgb[0]-truth).square().mean().item())
            scores[mode]=dict(weighted_motion_loss=float(np.mean(losses)),weighted_position_epe_pixels64=float(np.mean(epe)))
            if baselines: scores[mode]['rgb_mse']=float(np.mean(pixels))
        records.append(dict(file=data['metadata'][i]['file'],scores=scores))
    means={m:{k:float(np.mean([r['scores'][m][k] for r in records])) for k in records[0]['scores'][m]} for m in modes}
    if baselines:
        for m in modes: means[m]['rgb_psnr']=-10*np.log10(max(means[m]['rgb_mse'],1e-12))
    return dict(means=means,cases=records)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): raise FileExistsError('Preserve experiment')
    protocol=dict(seed=430201,steps=3000,hidden=32,batch=4,lr=.0003,seed_frames=3,predict_frames=6,
                  selection='minimum validation motion loss every 500 updates',teacher_forcing=False,
                  curriculum='first 1000 updates random 1-3 future steps, then full 6-step free rollout',
                  input='current Gaussian positions and fixed appearance, recent velocity/acceleration, learned per-slot hidden state',
                  training='279 recorded UCF clips; 57 group-disjoint validation; no cat or test frames',
                  limitation='Grid-index neighbours; image-plane trajectories; not articulated 3D, not trained on cats or text actions')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4); torch.manual_seed(protocol['seed'])
    with keep_windows_awake():
        data=load_verified(DATA)
        train=[i for i,e in enumerate(data['metadata']) if e['split']=='train']
        val=[i for i,e in enumerate(data['metadata']) if e['split']=='validation']
        assert len(train)==279 and len(val)==57
        assert not ({data['metadata'][i]['group'] for i in train}&{data['metadata'][i]['group'] for i in val})
        assert all(e['split'] in ['train','validation'] for e in data['metadata'])
        model=LearnedGaussianMotion().cuda(); optim=torch.optim.AdamW(model.parameters(),lr=protocol['lr'])
        first=evaluate(model,data,val); history=[dict(step=0,validation=first)]
        best=first['means']['learned']['weighted_motion_loss']
        def save(step,improved):
            save_training_checkpoint(dict(model=model.state_dict(),optimizer=optim.state_dict(),config=protocol,step=step,
                                          validation=history[-1]['validation'],rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),OUT,step,improved)
        save(0,True); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for step in range(1,protocol['steps']+1):
            model.train(); idx=[train[i] for i in torch.randint(len(train),(4,)).tolist()]
            seq=data['fields'][idx].cuda(); weight=weights(seq); state=model.initialize(seq[:,:,0],seq[:,:,1],seq[:,:,2])
            horizon=int(torch.randint(1,4,()).item()) if step<=1000 else 6
            value=0
            for t in range(3,3+horizon):
                state=model.step(state); value=value+loss(state,seq[:,:,t],seq[:,:,t-1],weight)/horizon
            if not torch.isfinite(value): raise ValueError('Nonfinite training loss')
            optim.zero_grad(set_to_none=True); value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optim.step()
            if step%100==0:
                progress=dict(step=step,total=protocol['steps'],loss=value.item(),seconds=time.perf_counter()-start)
                (OUT/'progress.json').write_text(json.dumps(progress,indent=2)); print(json.dumps(progress),flush=True)
            if step%500==0:
                score=evaluate(model,data,val); criterion=score['means']['learned']['weighted_motion_loss']; improved=criterion<best; best=min(best,criterion)
                history.append(dict(step=step,validation=score)); save(step,improved)
                (OUT/'history.json').write_text(json.dumps(history,indent=2))
        elapsed=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated()
        selected=load_verified(OUT/'best.pt'); model.load_state_dict(selected['model'])
        final=evaluate(model,data,val,baselines=True)
        sha=save_inference_checkpoint(dict(model=selected['model'],config=protocol,selected_step=selected['step'],data_sha256=digest(DATA)),OUT/'model.pt')
        report=dict(selected_step=selected['step'],initial=first['means'],validation=final,train_clips=len(train),validation_clips=len(val),test_clips=0,
                    seconds=elapsed,peak_training_allocated_bytes=peak,parameters=sum(p.numel() for p in model.parameters()),sha256=sha)
        (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(stage='complete',selected_step=selected['step'],scores=final['means'])),flush=True)


if __name__=='__main__': main()
