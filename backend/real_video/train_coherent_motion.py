"""Bounded self-evaluation: matched fine-tuning control and two coherence losses."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from .checkpoint_io import load_verified,save_training_checkpoint,save_inference_checkpoint,keep_windows_awake,digest
from .learned_motion import LearnedGaussianMotion
from .motion_coherence import make_context,coherence_loss,geometry_metrics,passes_gate
from .persistent_gaussian import render_state
from .train_learned_motion import DATA,weights,loss

OUT=Path('artifacts/real_video/coherent_motion/v1')
BASE=Path('artifacts/real_video/learned_motion/v1/model.pt')


@torch.no_grad()
def evaluate(model,data,indices,mode='learned'):
    model.eval(); records=[]
    for i in indices:
        seq=data['fields'][i:i+1].cuda(); weight=weights(seq)
        state=model.initialize(seq[:,:,0],seq[:,:,1],seq[:,:,2])
        ctx=make_context(seq[:,:,0],seq[:,:,1],seq[:,:,2]); rows=[]
        for t in range(3,9):
            if mode in ('learned','zero_hidden'): state=model.step(state,reset_memory=mode=='zero_hidden')
            elif mode=='velocity': state=dict(state,fields=torch.cat((state['fields'][:,:2]+state['velocity'],state['fields'][:,2:]),1))
            elif mode=='frozen': state=dict(state,velocity=torch.zeros_like(state['velocity']))
            else: raise ValueError(mode)
            epe=((state['fields'][:,:2]-seq[:,:2,t])*64).norm(dim=1,keepdim=True)
            rgb=render_state(state['fields'])
            truth=data['videos'][i,t].permute(2,0,1).cuda().float()/255
            metrics=geometry_metrics(state['fields'],seq[:,:,t],ctx)
            metrics.update(weighted_motion_loss=loss(state,seq[:,:,t],seq[:,:,t-1],weight),
                           weighted_position_epe_pixels64=(epe*weight).sum()/weight.sum(),
                           rgb_mse=(rgb[0]-truth).square().mean(),
                           mean_displacement_pixels64=((state['fields'][:,:2]-seq[:,:2,2])*64).norm(dim=1).mean(),
                           target_displacement_pixels64=((seq[:,:2,t]-seq[:,:2,2])*64).norm(dim=1).mean())
            rows.append({k:v.item() for k,v in metrics.items()})
        records.append(dict(file=data['metadata'][i]['file'],scores={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}))
    means={k:float(np.mean([r['scores'][k] for r in records])) for k in records[0]['scores']}
    means['rgb_psnr']=float(-10*np.log10(max(means['rgb_mse'],1e-12)))
    return dict(means=means,cases=records)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): raise FileExistsError('Preserve experiment')
    protocol=dict(seed=430301,hidden=32,steps_per_candidate=1000,batch=4,lr=.0001,strengths=[0.,.5,2.],
                  validation_every=250,seed_frames=3,forecast_steps=6,
                  training='279 UCF train; 57 source-group-disjoint validation; no test or cat',
                  control='All candidates use identical initial weights, optimizer reset and sampled minibatches',
                  selection='Gate versus original AND equally fine-tuned control at same step; among eligible select lowest edge MSE',
                  gate=dict(edge_mse_ratio_max=.95,epe_ratio_max=1.02,rgb_mse_ratio_max=1.01,
                            fold_mismatch_additive_max=.002,displacement_ratio_min=.8),
                  hypotheses='Target-relative edge supervision may reduce local deformation error without freezing motion',
                  limits='Planar grid edges and colour/velocity gates are not anatomy. Optical-flow targets are imperfect. Validation development only.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(4)
    with keep_windows_awake():
        data=load_verified(DATA); initial=load_verified(BASE)
        train=[i for i,e in enumerate(data['metadata']) if e['split']=='train']
        val=[i for i,e in enumerate(data['metadata']) if e['split']=='validation']
        assert len(train)==279 and len(val)==57
        assert not ({data['metadata'][i]['group'] for i in train}&{data['metadata'][i]['group'] for i in val})
        assert all(e['split'] in ['train','validation'] for e in data['metadata'])
        model=LearnedGaussianMotion().cuda(); model.load_state_dict(initial['model'])
        baseline={m:evaluate(model,data,val,m) for m in ['learned','frozen','velocity','zero_hidden']}
        (OUT/'baseline.json').write_text(json.dumps(baseline,indent=2))
        history=[]; control={}; selected=None; total_start=time.perf_counter()
        for strength in protocol['strengths']:
            tag=f'lambda_{strength:g}'; folder=OUT/tag
            torch.manual_seed(protocol['seed']); model.load_state_dict(initial['model'])
            optim=torch.optim.AdamW(model.parameters(),lr=protocol['lr'])
            torch.cuda.synchronize(); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            for step in range(1,protocol['steps_per_candidate']+1):
                model.train(); idx=[train[j] for j in torch.randint(len(train),(protocol['batch'],)).tolist()]
                seq=data['fields'][idx].cuda(); weight=weights(seq)
                state=model.initialize(seq[:,:,0],seq[:,:,1],seq[:,:,2]); ctx=make_context(seq[:,:,0],seq[:,:,1],seq[:,:,2]); value=0
                for t in range(3,9):
                    state=model.step(state)
                    term=loss(state,seq[:,:,t],seq[:,:,t-1],weight)
                    if strength: term=term+strength*coherence_loss(state['fields'],seq[:,:,t],ctx)
                    value=value+term/6
                if not torch.isfinite(value): raise ValueError('Nonfinite loss')
                optim.zero_grad(set_to_none=True); value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optim.step()
                if step%100==0:
                    progress=dict(candidate=tag,step=step,loss=value.item(),seconds=time.perf_counter()-start)
                    (OUT/'progress.json').write_text(json.dumps(progress,indent=2)); print(json.dumps(progress),flush=True)
                if step%protocol['validation_every']==0:
                    score=evaluate(model,data,val); means=score['means']
                    if strength==0: control[step]=means
                    gate=passes_gate(means,baseline['learned']['means'])
                    control_gate=passes_gate(means,control[step]) if strength else dict(passed=False,checks={})
                    eligible=gate['passed'] and control_gate['passed']
                    row=dict(candidate=tag,strength=strength,step=step,validation=score,gate=gate,matched_control_gate=control_gate,eligible=eligible)
                    history.append(row)
                    improved=eligible and (selected is None or means['edge_mse_pixels64']<selected['validation']['means']['edge_mse_pixels64'])
                    save_training_checkpoint(dict(model=model.state_dict(),optimizer=optim.state_dict(),config=protocol,step=step,
                                                  validation=score,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),folder,step,improved)
                    if improved: selected=row
                    (OUT/'history.json').write_text(json.dumps(history,indent=2))
                    print(json.dumps(dict(candidate=tag,step=step,means=means,eligible=eligible)),flush=True)
            torch.cuda.synchronize()
            (folder/'cost.json').write_text(json.dumps(dict(training_and_validation_seconds=time.perf_counter()-start,
                                                          peak_allocated_bytes=torch.cuda.max_memory_allocated()),indent=2))
        if selected:
            checkpoint=load_verified(OUT/selected['candidate']/'checkpoints'/f"step_{selected['step']:08d}.pt")
            sha=save_inference_checkpoint(dict(model=checkpoint['model'],config=protocol,selected_step=selected['step'],
                                              base_sha256=digest(BASE),data_sha256=digest(DATA)),OUT/'model.pt')
        else: sha=None
        results=dict(selected=selected,exported_sha256=sha,total_candidate_seconds=time.perf_counter()-total_start,
                     candidates=len(protocol['strengths']),evaluated_checkpoints=len(history),test_clips=0,
                     wan_weights_changed=False,automatic_promotion=False,
                     status='validation_gate_passed_not_visual_certified' if selected else 'all_candidates_rejected_keep_original')
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({k:v for k,v in results.items() if k!='selected'}),flush=True)


if __name__=='__main__': main()
