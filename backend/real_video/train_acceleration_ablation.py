"""One fixed external-DAVIS update-rule ablation, never cat/final-test training.

Loss, features, optimizer and sequence-balanced sampling match residual trial.
Acceleration cap .005 versus velocity-offset cap .02 has different units: this
is a declared update-plus-cap ablation, not a perfectly matched-cap experiment.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .checkpoint_io import load_verified, save_training_checkpoint, save_inference_checkpoint, digest, keep_windows_awake
from .train_vector_motion_residual import DATA, batch, targets, loss, evaluate, acceptance
from .vector_motion_acceleration import AccelerationMotionNetwork
from .motion_model_io import load_motion_model

OUT=Path('artifacts/real_video/true3d/learned_motion_loop/integrated_acceleration_v1')
REFERENCE=Path('artifacts/real_video/true3d/learned_motion_loop/residual_velocity_v1/model.pt')


@torch.no_grad()
def acceleration_metrics(model, reference, data, index, device, epsilon=.001):
    values=batch(data,index,device)
    observed,scale,velocity,relative,weight=targets(values)
    initial=values['position'][:,2]-values['position'][:,1]
    predictions={}
    predictions['learned'],_=model.rollout(values['position'][:,:3],values['adjacency'],12,values['confidence'][:,2])
    predictions['unchanged_residual'],_=reference.rollout(values['position'][:,:3],values['adjacency'],12,values['confidence'][:,2])
    horizon=torch.arange(1,13,device=device,dtype=observed.dtype)[None,:,None,None]
    predictions['frozen']=observed[:,None].expand(-1,12,-1,-1)
    predictions['velocity']=observed[:,None]+horizon*initial[:,None]
    predictions['average_velocity']=observed[:,None]+horizon*((values['position'][:,2]-values['position'][:,0])/2)[:,None]
    for damping in [.75,.95]:
        predictions[f'damped{round(damping*100)}']=observed[:,None]+damping*(1-damping**horizon)/(1-damping)*initial[:,None]
    truth_acc=torch.diff(torch.cat((initial[:,None],velocity),1),dim=1)
    quality=values['confidence'].clamp(0,1)*values['visibility'].clamp(0,1)
    acc_weight=torch.minimum(torch.minimum(quality[:,1:-2],quality[:,2:-1]),quality[:,3:])
    # Metric-only, fixed node confidence prevents a changing population centroid
    # from making a constant-velocity baseline appear to reverse direction.
    initial_weight=quality[:,1:].amin(1)
    initial_center=(initial*initial_weight[...,None]).sum(1,keepdim=True)/initial_weight.sum(1,keepdim=True)[...,None].clamp_min(1e-8)
    initial_relative=initial-initial_center
    truth_center=(velocity*initial_weight[:,None,:,None]).sum(2,keepdim=True)/initial_weight.sum(1,keepdim=True)[:,None,:,None].clamp_min(1e-8)
    reversal_truth_relative=velocity-truth_center
    initial_norm=initial_relative.norm(dim=-1)/scale[:,:,0]
    target_norm=reversal_truth_relative.norm(dim=-1)/scale[:,None,:,0]
    true_cos=(reversal_truth_relative*initial_relative[:,None]).sum(-1)/(reversal_truth_relative.norm(dim=-1)*initial_relative.norm(dim=-1)[:,None]).clamp_min(1e-12)
    valid=(initial_norm[:,None]>epsilon)&(target_norm>epsilon)
    reversal_weight=weight*initial_weight[:,None]*valid
    truth_reversed=true_cos<-.25
    groups={name:[j for j,i in enumerate(index) if data['metadata'][i]['sequence']==name] for name in sorted({data['metadata'][i]['sequence'] for i in index})}
    results={}
    for mode,prediction in predictions.items():
        pv=torch.diff(torch.cat((observed[:,None],prediction),1),dim=1)
        pa=torch.diff(torch.cat((initial[:,None],pv),1),dim=1)
        ae=((pa-truth_acc)/scale[:,None]*32).norm(dim=-1)
        case_epe=(ae*acc_weight).sum((1,2))/acc_weight.sum((1,2)).clamp_min(1e-8)
        center=(pv*initial_weight[:,None,:,None]).sum(2,keepdim=True)/initial_weight.sum(1,keepdim=True)[:,None,:,None].clamp_min(1e-8)
        pr=pv-center
        predicted_norm=pr.norm(dim=-1)/scale[:,None,:,0]
        cosine=(pr*initial_relative[:,None]).sum(-1)/(pr.norm(dim=-1)*initial_relative.norm(dim=-1)[:,None]).clamp_min(1e-12)
        predicted_reversed=(cosine<-.25)&(predicted_norm>epsilon)
        masks=dict(tp=predicted_reversed&truth_reversed,fp=predicted_reversed&~truth_reversed,
                   fn=~predicted_reversed&truth_reversed,tn=~predicted_reversed&~truth_reversed)
        counts={key:(value*reversal_weight).sum((1,2)).cpu() for key,value in masks.items()}
        sequence={}
        for name,ids in groups.items():
            masses={key:float(value[ids].sum()) for key,value in counts.items()}
            tp,fp,fn,tn=(masses[key] for key in ['tp','fp','fn','tn'])
            sequence[name]=dict(acceleration_epe=float(case_epe[ids].mean()),weighted_reversal_counts=masses,
                reversal_precision=tp/(tp+fp) if tp+fp else None,
                reversal_recall=tp/(tp+fn) if tp+fn else None,
                reversal_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                reversal_accuracy=(tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else None)
        macro={key:float(np.mean([r[key] for r in sequence.values() if r[key] is not None])) if any(r[key] is not None for r in sequence.values()) else None
               for key in ['acceleration_epe','reversal_precision','reversal_recall','reversal_f1','reversal_accuracy']}
        results[mode]=dict(macro=macro,sequences=sequence)
    return dict(definition='Acceleration EPE = second difference error in objectspan*32/sample^2, weighted by minimum three-frame quality. Reversal compares centroid-relative future velocity with centroid-relative last observed velocity; SAME fixed node weights (minimum quality over observed/evaluation frames1..14) define every centroid, for metrics only. Cosine<-.25 and initial/target/predicted normalized speed>.001 prevent near-zero sign jitter. Frozen prediction counts as no reversal. Sequence macro averages; nullable precision if no predicted events.',
        metric_revision='fixed_centroid_v2',
        speed_floor_objectspan_per_sample=epsilon,results=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--steps',type=int,default=1500)
    args=parser.parse_args(); out=args.output
    if args.steps<1 or args.steps>1500: raise ValueError('Bounded1..1500updates required')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists(): raise FileExistsError('Preserve trial')
    torch.set_num_threads(4); torch.manual_seed(431101)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data=load_verified(DATA)
    train=[i for i,m in enumerate(data['metadata']) if m['split']=='train']
    val=[i for i,m in enumerate(data['metadata']) if m['split']=='validation']
    groups={name:[i for i in train if data['metadata'][i]['sequence']==name] for name in sorted({data['metadata'][i]['sequence'] for i in train})}
    val_names=sorted({data['metadata'][i]['sequence'] for i in val})
    assert not set(groups).intersection(val_names)
    assert all(m['split'] in ['train','validation'] for m in data['metadata'])
    train_values=batch(data,train,'cpu'); _,scale,_,relative,weight=targets(train_values)
    normalized=relative.norm(dim=-1)/scale[:,None,:,0]
    threshold=float(torch.quantile(normalized[weight>=.5],.75))
    reference_packet=load_verified(REFERENCE)
    # Loading a comparator must not consume the candidate's seeded init stream.
    with torch.random.fork_rng(devices=[]):
        reference=load_motion_model(reference_packet).to(device)
    for parameter in reference.parameters(): parameter.requires_grad_(False)
    protocol=dict(architecture=AccelerationMotionNetwork.architecture,model_config=dict(hidden=64,acceleration_scale=.005),seed=431101,
        steps=args.steps,batch_per_sequence=2,batch=2*len(groups),learning_rate=.0003,weight_decay=.01,
        input_frames=3,prediction_steps=12,no_teacher_forcing=True,noise_augmentation=False,
        augmentation='random horizontal reflection only',data_sha256=digest(DATA),train_windows=len(train),validation_windows=len(val),
        train_sequences=sorted(groups),validation_sequences=val_names,validation_reused_for_adaptation=True,final_test_count=0,
        update='a_next=.005*object_span*tanh(head); v_next=v+a_next; p_next=p+v_next; no mandatory damping or initial-velocity anchor',
        cap_caveat='Acceleration cap .005 objectspan/sample^2 differs in units and magnitude from residual velocity-offset cap .02 objectspan/sample. Update-plus-cap ablation, not identical physical cap.',
        objective='Exactly imported residual trainer loss: .5position+velocity+centroid-relative velocity smoothL2; no acceleration/reversal loss added.',
        balancing='Exactly2random windows per training sequence; unchanged per-window topquartile-motion balancing and confidence scheme.',
        selection='Minimum sequence-macro validation objective at0,100,...,1500; initial constantvelocity fallback eligible and explicitly untrained; best trained exported separately.',
        moving_threshold=threshold,threshold_source='Train-only confident relative speed75thpercentile; fixed for validation.',
        unchanged_reference=dict(path=str(REFERENCE),sha256=digest(REFERENCE),selected_step=reference_packet['selected_step']),
        acceleration_reversal_metrics='Final-report diagnostics only, not extra checkpoint selection. Relative motion reversal cosine<-.25 with fixed.001objectspan/sample speed floor.',
        time_units='Sampled frames; physical dt for individual DAVIS sequences unavailable. No m/s or physical acceleration claims.',
        limitations='Deterministic2Dcausal update ablation on noisy optical-flow tracks. No cat inputs, no finaltest, no3Danatomy/Wanweight changes, no natural-video quality claim.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source_names=['vector_motion_acceleration.py','train_acceleration_ablation.py','vector_motion_residual.py','vector_motion_network.py','train_vector_motion_residual.py']
    for name in source_names: shutil.copyfile(Path('real_video')/name,out/name)
    with keep_windows_awake():
        model=AccelerationMotionNetwork(**protocol['model_config']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
        initial=evaluate(model,data,val,device,threshold)
        reference_score=evaluate(reference,data,val,device,threshold)
        (out/'initial.json').write_text(json.dumps(initial,indent=2))
        (out/'unchanged_residual_validation.json').write_text(json.dumps(reference_score,indent=2))
        best=initial['means']['learned']['selection_objective']; trained_best=float('inf'); trained_step=None
        history=[]; start=time.perf_counter()
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        def save(step,score,improved):
            save_training_checkpoint(dict(model={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                model_config=model.config,config=model.config,architecture=model.architecture,step=step,score=score,
                optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None),out,step,improved)
        save(0,initial,True)
        for step in range(1,args.steps+1):
            model.train()
            index=[members[j] for members in groups.values() for j in torch.randint(len(members),(2,)).tolist()]
            values=batch(data,index,device)
            mirror=torch.rand(len(index),device=device)<.5; values['position'][mirror,:,:,0]*=-1
            prediction,_=model.rollout(values['position'][:,:3],values['adjacency'],12,values['confidence'][:,2])
            objective,_=loss(prediction,values)
            if not torch.isfinite(objective): raise ValueError('Nonfinite acceleration objective')
            optimizer.zero_grad(set_to_none=True); objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            if step%100==0 or step==args.steps:
                score=evaluate(model,data,val,device,threshold)
                value=score['means']['learned']['selection_objective']; improved=value<best
                if value<trained_best: trained_best=value; trained_step=step
                best=min(best,value); save(step,score,improved)
                history.append(dict(step=step,train_loss=float(objective.detach()),validation=score,seconds=time.perf_counter()-start))
                (out/'history.json').write_text(json.dumps(history,indent=2))
                progress=dict(step=step,total=args.steps,selected_objective=best,best_trained_objective=trained_best,validation=score['means']['learned'],seconds=time.perf_counter()-start)
                (out/'progress.json').write_text(json.dumps(progress,indent=2))
                print(json.dumps(dict(step=step,objective=value,ade=score['means']['learned']['ade'],seconds=progress['seconds'])),flush=True)
        final={}
        for label,path in [('selected',out/'best.pt'),('best_trained',out/'checkpoints'/f'step_{trained_step:08d}.pt')]:
            checkpoint=load_verified(path); model.load_state_dict(checkpoint['model'],strict=True); model.eval()
            score=evaluate(model,data,val,device,threshold); gates,ratios=acceptance(score)
            extra=acceleration_metrics(model,reference,data,val,device)
            packet=dict(architecture=model.architecture,model_config=model.config,config=model.config,model=checkpoint['model'],
                selected_step=checkpoint['step'],trained=checkpoint['step']>0,data_sha256=digest(DATA),protocol_sha256=digest(out/'protocol.json'),
                inference_contract='Exactly3observedXYframes+adjacency/confidence initialize persistentstate; nofutureinput',learned_dimension=2,time_units=protocol['time_units'])
            destination=out/('model.pt' if label=='selected' else 'best_trained.pt')
            sha=save_inference_checkpoint(packet,destination)
            reference_means=reference_score['means']['learned']; means=score['means']['learned']
            comparison={key:dict(candidate=means[key],unchanged_residual=reference_means[key]) for key in means if means[key] is not None}
            final[label]=dict(step=checkpoint['step'],trained=checkpoint['step']>0,sha256=sha,checkpoint_bytes=destination.stat().st_size,
                validation=score,acceleration_reversal=extra,gates=gates,all_ordinary_gates_pass=all(v for k,v in gates.items() if k!='tenfold_ade'),
                per_sequence_ade_ratio_to_strongest_baseline=ratios,unchanged_residual_comparison=comparison)
        report=dict(architecture=model.architecture,parameters=sum(p.numel() for p in model.parameters()),loop_seconds=time.perf_counter()-start,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,results=final,test_sequences=0,validation_reused=True,
            source_sha256={name:digest(out/name) for name in source_names})
        (out/'selection.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(done=True,selected_step=final['selected']['step'],trained=final['selected']['trained'],all_ordinary_gates_pass=final['selected']['all_ordinary_gates_pass'],seconds=report['loop_seconds'])),flush=True)


if __name__=='__main__': main()
