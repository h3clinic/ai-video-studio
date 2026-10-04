"""Animal-specific recurrent control training; targets are recorded flow tracks."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from .animal_control_graph import AnimalControlGraph
from .animal_graph_data import DATA
from .checkpoint_io import load_verified,save_training_checkpoint,save_inference_checkpoint,digest,keep_windows_awake

OUT=Path('artifacts/real_video/animal_graph/v1')


def batch(data,indices):
    return {key:data[key][indices].cuda() for key in ['position','angle','visibility','confidence','colour','adjacency']}


def initialize(model,data):
    return model.initialize(data['position'][:,:3],data['angle'][:,:3],data['visibility'][:,:3],data['colour'],data['adjacency'])


def target_loss(state,data,t):
    error=(state['position']-data['position'][:,t])/state['scale']*32
    moving=((data['position'][:,2]-data['position'][:,1])/state['scale']).norm(dim=-1)>.01
    confidence=data['confidence'][:,t].clamp_min(.05)
    weight=(.1+.9*data['visibility'][:,t])*confidence*(1+4*moving)
    position=(error.square().sum(-1)*weight).sum()/weight.sum().clamp_min(1e-8)
    angle=((1-torch.cos(state['angle']-data['angle'][:,t]))*weight).sum()/weight.sum().clamp_min(1e-8)
    visibility=F.binary_cross_entropy(state['visibility'].clamp(1e-5,1-1e-5),data['visibility'][:,t].clamp(0,1))
    return position+.5*angle+visibility


@torch.no_grad()
def evaluate(model,data,indices,controls=False):
    model.eval(); modes=['learned','frozen','velocity','zero_hidden'] if controls else ['learned']; records=[]
    for i in indices:
        sample=batch(data,[i]); scores={}
        for mode in modes:
            state=initialize(model,sample); rows=[]
            for t in range(3,15):
                if mode in ['learned','zero_hidden']: state=model.step(state,reset_memory=mode=='zero_hidden')
                elif mode=='velocity': state=dict(state,position=state['position']+state['velocity'],angle=state['angle']+state['angular_velocity'])
                error=((state['position']-sample['position'][:,t])/state['scale']*32).norm(dim=-1)
                vis=sample['visibility'][:,t]; confidence=sample['confidence'][:,t].clamp_min(.05)
                w=vis*confidence; visible_epe=(error*w).sum()/w.sum().clamp_min(1e-8)
                brier=(state['visibility']-vis).square().mean()
                displacement=((state['position']-sample['position'][:,2])/state['scale']*32).norm(dim=-1).mean()
                rows.append(dict(epe_all=float(error.mean()),visible_confident_epe=float(visible_epe),visibility_brier=float(brier),
                                 displacement=float(displacement),objective=float(target_loss(state,sample,t))))
            scores[mode]={key:float(np.mean([r[key] for r in rows])) for key in rows[0]}
        records.append(dict(**data['metadata'][i],scores=scores))
    groups=sorted({r['sequence'] for r in records}); group_scores={}
    for group in groups:
        members=[r for r in records if r['sequence']==group]
        group_scores[group]={mode:{k:float(np.mean([r['scores'][mode][k] for r in members])) for k in members[0]['scores'][mode]} for mode in modes}
    means={mode:{k:float(np.mean([g[mode][k] for g in group_scores.values()])) for k in records[0]['scores'][mode]} for mode in modes}
    return dict(means=means,sequence_scores=group_scores,cases=records)


def main():
    global DATA,OUT
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--data',type=Path,default=DATA); parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args(); DATA=args.data; OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): raise FileExistsError('Preserve graph experiment')
    protocol=dict(seed=430401,steps=4000,batch=8,lr=.0003,hidden=64,nodes=48,seed_frames=3,predict_frames=12,
                  selection='minimum sequence-macro validation objective every500 updates; no cat-generation diagnostic or test used',
                  curriculum='1000 updates horizon4 then horizon12',
                  representation='fixed appearance + 48 control-point SE2 frames + visibility; controls are not bones',
                  data=f'DAVIS source {DATA}; cat-girl id2 only; see cache metadata for animal-only versus mixed-object selection; optical-flow targets with annotated-mask membership',
                  augmentation='random horizontal reflection applied consistently to positions, angles and adjacency; nodes are shared-weight graph slots',
                  limitations='Tiny dataset; sequence-disjoint not identity-proven; no new appearance, depth, text input or Wan adaptation')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(4); torch.manual_seed(protocol['seed'])
    with keep_windows_awake():
        data=load_verified(DATA); train=[i for i,m in enumerate(data['metadata']) if m['split']=='train']; val=[i for i,m in enumerate(data['metadata']) if m['split']=='validation']
        assert not ({data['metadata'][i]['sequence'] for i in train}&{data['metadata'][i]['sequence'] for i in val})
        assert all(m['split'] in ['train','validation'] for m in data['metadata'])
        model=AnimalControlGraph().cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'])
        initial=evaluate(model,data,val,True); (OUT/'initial.json').write_text(json.dumps(initial,indent=2))
        best=initial['means']['learned']['objective']; history=[]
        def save(step,improved,score):
            save_training_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=protocol,step=step,
                                          score=score,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),OUT,step,improved)
        save(0,True,initial); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for step in range(1,protocol['steps']+1):
            model.train(); indices=[train[i] for i in torch.randint(len(train),(protocol['batch'],)).tolist()]; sample=batch(data,indices)
            mirror=torch.rand(len(indices),device='cuda')<.5
            sample['position'][mirror,:,:,0]*=-1; sample['angle'][mirror]*=-1
            state=initialize(model,sample); horizon=4 if step<=1000 else 12; objective=0
            for t in range(3,3+horizon): state=model.step(state); objective=objective+target_loss(state,sample,t)/horizon
            if not torch.isfinite(objective): raise ValueError('Nonfinite graph training')
            optimizer.zero_grad(set_to_none=True); objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            if step%100==0:
                progress=dict(step=step,total=protocol['steps'],loss=float(objective.detach()),seconds=time.perf_counter()-start)
                (OUT/'progress.json').write_text(json.dumps(progress,indent=2)); print(json.dumps(progress),flush=True)
            if step%500==0:
                score=evaluate(model,data,val); value=score['means']['learned']['objective']; improved=value<best; best=min(best,value)
                history.append(dict(step=step,validation=score)); save(step,improved,score)
                (OUT/'history.json').write_text(json.dumps(history,indent=2)); print(json.dumps(dict(step=step,validation=score['means'])),flush=True)
        seconds=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated(); selected=load_verified(OUT/'best.pt'); model.load_state_dict(selected['model'])
        scores=evaluate(model,data,val,True)
        sha=save_inference_checkpoint(dict(model=selected['model'],config=protocol,selected_step=selected['step'],data_sha256=digest(DATA)),OUT/'model.pt')
        report=dict(selected_step=selected['step'],validation=scores,train_windows=len(train),validation_windows=len(val),
                    train_sequences=sorted({data['metadata'][i]['sequence'] for i in train}),validation_sequences=sorted({data['metadata'][i]['sequence'] for i in val}),
                    parameters=sum(p.numel() for p in model.parameters()),loop_seconds=seconds,peak_allocated_bytes=peak,sha256=sha,
                    wan_weights_changed=False,visually_accepted=False,test_sequences=0)
        (OUT/'results.json').write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!='validation'}),flush=True)
        print(json.dumps(scores['means']),flush=True)


if __name__=='__main__': main()
