"""One predeclared motion-preserving recurrent trial, recorded video only.

This adaptively reuses the existing validation split and is NOT a final test.
No Wan/cat diagnostic is read by this trainer or its model-selection routine.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .vector_motion_residual import ResidualVelocityNetwork
from .checkpoint_io import load_verified, save_training_checkpoint, save_inference_checkpoint, digest, keep_windows_awake
from .prepare_animal_motion import WORK

DATA = WORK / 'control_tracks_animal_only_v2.pt'
OUT = Path('artifacts/real_video/true3d/learned_motion_loop/residual_velocity_v1')


def batch(data, index, device):
    return {key:data[key][index].to(device) for key in ['position', 'adjacency', 'confidence', 'visibility']}


def targets(values):
    observed = values['position'][:,2]
    scale = (observed.amax(1,keepdim=True)-observed.amin(1,keepdim=True)).amax(-1,keepdim=True).clamp_min(1e-5)
    velocity = torch.diff(values['position'][:,2:], dim=1)
    weight = values['confidence'][:,3:].clamp(0,1)*values['visibility'][:,3:].clamp(0,1)
    center = (velocity*weight[...,None]).sum(2,keepdim=True)/weight.sum(2,keepdim=True)[...,None].clamp_min(1e-8)
    relative = velocity-center
    return observed, scale, velocity, relative, weight


def smooth_norm(value):
    return torch.sqrt(value.square().sum(-1)+1e-6)-.001


def loss(prediction, values, moving_threshold=None):
    observed, scale, velocity, relative, confidence = targets(values)
    predicted_velocity = torch.diff(torch.cat((observed[:,None],prediction),1),dim=1)
    center = (predicted_velocity*confidence[...,None]).sum(2,keepdim=True)/confidence.sum(2,keepdim=True)[...,None].clamp_min(1e-8)
    predicted_relative = predicted_velocity-center
    activity = (relative.norm(dim=-1)/scale[:,None,:,0]*confidence).sum(1)/confidence.sum(1).clamp_min(1e-8)
    if moving_threshold is None:
        threshold = torch.quantile(activity,.75,dim=-1,keepdim=True)
        moving = activity>threshold
    else:
        moving = activity>moving_threshold
    first = confidence*moving[:,None]
    other = confidence*(~moving[:,None])
    first_mass, other_mass = first.sum((1,2),keepdim=True), other.sum((1,2),keepdim=True)
    has_both = (first_mass>0)&(other_mass>0)
    balanced = .5*first/first_mass.clamp_min(1e-8)+.5*other/other_mass.clamp_min(1e-8)
    normal = confidence/confidence.sum((1,2),keepdim=True).clamp_min(1e-8)
    weight = torch.where(has_both,balanced,normal)
    position_error = smooth_norm((prediction-values['position'][:,3:])/scale[:,None]*32)
    velocity_error = smooth_norm((predicted_velocity-velocity)/scale[:,None]*32)
    relative_error = smooth_norm((predicted_relative-relative)/scale[:,None]*32)
    per_case = ((.5*position_error+velocity_error+relative_error)*weight).sum((1,2))
    return per_case.mean(), per_case


@torch.no_grad()
def evaluate(model,data,index,device,moving_threshold):
    values=batch(data,index,device)
    observed,scale,truth_velocity,truth_relative,weight=targets(values)
    predictions={}
    model.eval()
    predictions['learned'],_=model.rollout(values['position'][:,:3],values['adjacency'],12,values['confidence'][:,2])
    horizon=torch.arange(1,13,device=device,dtype=observed.dtype)[None,:,None,None]
    initial_velocity=values['position'][:,2]-values['position'][:,1]
    predictions['frozen']=observed[:,None].expand(-1,12,-1,-1)
    predictions['velocity']=observed[:,None]+horizon*initial_velocity[:,None]
    predictions['average_velocity']=observed[:,None]+horizon*((values['position'][:,2]-values['position'][:,0])/2)[:,None]
    for damping in [.75,.95]:
        amplitude=damping*(1-damping**horizon)/(1-damping)
        predictions[f'damped{round(damping*100)}']=observed[:,None]+amplitude*initial_velocity[:,None]
    moving=(truth_relative.norm(dim=-1)/scale[:,None,:,0]>moving_threshold)
    motion_weight=weight*moving
    rows={}
    for mode,prediction in predictions.items():
        error=((prediction-values['position'][:,3:])/scale[:,None]*32).norm(dim=-1)
        pv=torch.diff(torch.cat((observed[:,None],prediction),1),dim=1)
        center=(pv*weight[...,None]).sum(2,keepdim=True)/weight.sum(2,keepdim=True)[...,None].clamp_min(1e-8)
        pr=pv-center
        velocity_error=((pv-truth_velocity)/scale[:,None]*32).norm(dim=-1)
        relative_error=((pr-truth_relative)/scale[:,None]*32).norm(dim=-1)
        cosine=(pr*truth_relative).sum(-1)/(pr.norm(dim=-1)*truth_relative.norm(dim=-1)).clamp_min(1e-8)
        _,objective=loss(prediction,values,moving_threshold)
        rows[mode]=dict(ade=((error*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1e-8)).cpu(),
                        fde=((error[:,-1]*weight[:,-1]).sum(1)/weight[:,-1].sum(1).clamp_min(1e-8)).cpu(),
                        velocity_epe=((velocity_error*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1e-8)).cpu(),
                        relative_velocity_epe=((relative_error*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1e-8)).cpu(),
                        selection_objective=objective.cpu(),
                        moving_velocity_numerator=(velocity_error*motion_weight).sum((1,2)).cpu(),
                        moving_relative_velocity_numerator=(relative_error*motion_weight).sum((1,2)).cpu(),
                        direction_numerator=(cosine*motion_weight).sum((1,2)).cpu(),
                        predicted_amplitude=(pr.norm(dim=-1)*motion_weight/scale[:,None,:,0]).sum((1,2)).cpu(),
                        target_amplitude=(truth_relative.norm(dim=-1)*motion_weight/scale[:,None,:,0]).sum((1,2)).cpu(),
                        motion_weight=motion_weight.sum((1,2)).cpu())
    groups={}
    for sequence in sorted({data['metadata'][i]['sequence'] for i in index}):
        ids=[j for j,i in enumerate(index) if data['metadata'][i]['sequence']==sequence]
        groups[sequence]={}
        for mode,row in rows.items():
            mass=float(row['motion_weight'][ids].sum())
            score={key:float(row[key][ids].mean()) for key in ['ade','fde','velocity_epe','relative_velocity_epe','selection_objective']}
            score.update(moving_velocity_epe=float(row['moving_velocity_numerator'][ids].sum())/mass if mass else None,
                         moving_relative_velocity_epe=float(row['moving_relative_velocity_numerator'][ids].sum())/mass if mass else None,
                         moving_signed_direction_cosine=float(row['direction_numerator'][ids].sum())/mass if mass else None,
                         moving_relative_amplitude_ratio=float(row['predicted_amplitude'][ids].sum()/row['target_amplitude'][ids].sum().clamp_min(1e-8)) if mass else None,
                         moving_confident_mass=mass,windows=len(ids))
            groups[sequence][mode]=score
    means={mode:{key:float(np.mean([group[mode][key] for group in groups.values() if group[mode][key] is not None]))
                 if any(group[mode][key] is not None for group in groups.values()) else None
                 for key in next(iter(groups.values()))[mode] if key not in ['windows','moving_confident_mass']} for mode in predictions}
    return dict(means=means,sequence_scores=groups)


def acceptance(score):
    means=score['means']; learned=means['learned']; baselines=[value for key,value in means.items() if key!='learned']
    ade=min(value['ade'] for value in baselines)
    fde=min(value['fde'] for value in baselines)
    motion=min(value['moving_relative_velocity_epe'] for value in baselines)
    direction=max(value['moving_signed_direction_cosine'] for value in baselines)
    ratios={sequence:values['learned']['ade']/min(value['ade'] for key,value in values.items() if key!='learned') for sequence,values in score['sequence_scores'].items()}
    return dict(ade_at_least_10percent_better=learned['ade']<=.9*ade,
                fde_no_regression=learned['fde']<=fde,
                moving_relative_velocity_10percent_better=learned['moving_relative_velocity_epe']<=.9*motion,
                moving_signed_direction_no_regression=learned['moving_signed_direction_cosine']>=direction,
                moving_amplitude_ratio_in_075_to_125=.75<=learned['moving_relative_amplitude_ratio']<=1.25,
                no_sequence_ade_more_than10percent_worse=max(ratios.values())<=1.1,
                tenfold_ade=learned['ade']<=.1*ade),ratios


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--steps',type=int,default=1500)
    args=parser.parse_args(); out=args.output
    if args.steps<1 or args.steps>1500: raise ValueError('Bounded trial requires1..1500updates')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists(): raise FileExistsError('Preserve residual trial')
    torch.set_num_threads(4); torch.manual_seed(431101)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data=load_verified(DATA)
    train=[i for i,m in enumerate(data['metadata']) if m['split']=='train']
    val=[i for i,m in enumerate(data['metadata']) if m['split']=='validation']
    groups={name:[i for i in train if data['metadata'][i]['sequence']==name] for name in sorted({data['metadata'][i]['sequence'] for i in train})}
    val_names=sorted({data['metadata'][i]['sequence'] for i in val})
    assert not set(groups).intersection(val_names)
    assert all(m['split'] in ['train','validation'] for m in data['metadata'])
    train_batch=batch(data,train,'cpu'); _,scale,_,relative,weight=targets(train_batch)
    normalized=relative.norm(dim=-1)/scale[:,None,:,0]
    threshold=float(torch.quantile(normalized[weight>=.5],.75))
    protocol=dict(architecture=ResidualVelocityNetwork.architecture,model_config=dict(hidden=64,residual_scale=.02),seed=431101,
                  steps=args.steps,batch_per_sequence=2,batch=2*len(groups),learning_rate=.0003,weight_decay=.01,
                  input_frames=3,prediction_steps=12,no_teacher_forcing=True,noise_augmentation=False,
                  augmentation='random horizontal reflection only',data_sha256=digest(DATA),train_windows=len(train),validation_windows=len(val),
                  train_sequences=sorted(groups),validation_sequences=val_names,validation_reused_for_adaptation=True,final_test_count=0,
                  update='v_next = v_observed_last + .02*observed_object_span*tanh(learned_residual); p_next=p+v_next; no multiplicative damping',
                  objective='0.5*position smooth L2 norm +1*velocity smooth L2 norm +1*centroid-relative velocity smooth L2 norm; vector epsilon0.001; units objectspan*32',
                  balancing='Every batch takes2random windows fromeachtrainsequence. Perwindowtrainingtopquartile of confidence-weighted future relative node activity and remainder receiveequal0.5 loss mass; neither group synthetic.',
                  confidence='clipped optical-flow consistency times mask membership; no positive floor; zero-mass groups fallback to valid other group',
                  evaluation_moving_threshold=threshold,evaluation_threshold_source='75thpercentile of TRAIN ONLY confident relative per-step speeds/objectspan; heldfixedforvalidation',
                  selection='minimum sequence-macro validation objective atstep0,100,...,1500; step0eligible explicitlyUNTRAINEDconstantvelocityfallback. Exportbesttrainedseparately.',
                  gates=['macroADE<=0.9*strongestbaselineADE','FDE<=strongestbaselineFDE','movingrelativevelocityEPE<=0.9*strongestbaseline',
                         'movingsigneddirectioncosine>=strongestbaseline','movingrelativeamplituderatio between.75and1.25',
                         'everysequenceADE<=1.1*itsstrongestbaseline','separate10xgateADE<=.1*strongestbaseline'],
                  baselines=['frozen','lastvelocity','averageobservedvelocity','0.75dampedvelocity','0.95dampedvelocity'],
                  time_units='sampledframe: DAVISstride2, Wanstride1. Per-sequence physical dt unavailable; no physical-speed transfer claim.',
                  limitations='Still deterministic2Dforecast; tiny noisyflowdataset; no learneddepth/anatomy/action/noise/text/Wanweights. Motion-amplitude alone not natural gait.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for name in ['vector_motion_residual.py','train_vector_motion_residual.py','vector_motion_network.py']:
        shutil.copyfile(Path('real_video')/name,out/name)
    with keep_windows_awake():
        model=ResidualVelocityNetwork(**protocol['model_config']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=.01)
        initial=evaluate(model,data,val,device,threshold)
        (out/'initial.json').write_text(json.dumps(initial,indent=2))
        best=initial['means']['learned']['selection_objective']; trained_best=float('inf'); trained_step=None
        history=[]; start=time.perf_counter()
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        def save(step,score,improved):
            save_training_checkpoint(dict(model={key:value.cpu() for key,value in model.state_dict().items()},
                                          model_config=model.config,config=model.config,architecture=model.architecture,
                                          step=step,score=score,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                                          cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None),out,step,improved)
        save(0,initial,True)
        for step in range(1,args.steps+1):
            model.train()
            index=[members[j] for members in groups.values() for j in torch.randint(len(members),(2,)).tolist()]
            values=batch(data,index,device)
            mirror=torch.rand(len(index),device=device)<.5
            values['position'][mirror,:,:,0]*=-1
            prediction,_=model.rollout(values['position'][:,:3],values['adjacency'],12,values['confidence'][:,2])
            objective,_=loss(prediction,values)
            if not torch.isfinite(objective): raise ValueError('Nonfinite residual-motion objective')
            optimizer.zero_grad(set_to_none=True); objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            if step%100==0 or step==args.steps:
                score=evaluate(model,data,val,device,threshold)
                value=score['means']['learned']['selection_objective']; improved=value<best
                if value<trained_best: trained_best=value; trained_step=step
                best=min(best,value); save(step,score,improved)
                record=dict(step=step,train_loss=float(objective.detach()),validation=score,seconds=time.perf_counter()-start)
                history.append(record); (out/'history.json').write_text(json.dumps(history,indent=2))
                progress=dict(step=step,total=args.steps,selected_objective=best,best_trained_objective=trained_best,
                              validation=score['means']['learned'],seconds=time.perf_counter()-start)
                (out/'progress.json').write_text(json.dumps(progress,indent=2)); print(json.dumps(progress),flush=True)
        final={}
        for label,path in [('selected',out/'best.pt'),('best_trained',out/'checkpoints'/f'step_{trained_step:08d}.pt')]:
            checkpoint=load_verified(path); model.load_state_dict(checkpoint['model'],strict=True)
            score=evaluate(model,data,val,device,threshold); gates,ratios=acceptance(score)
            packet=dict(architecture=model.architecture,model_config=model.config,config=model.config,model=checkpoint['model'],
                        selected_step=checkpoint['step'],trained=checkpoint['step']>0,data_sha256=digest(DATA),protocol_sha256=digest(out/'protocol.json'),
                        inference_contract='initialize exactly3observedXYcontrolframes+observedadjacency/confidence; step consumesonlypersistentstate',
                        learned_dimension=2,time_units=protocol['time_units'])
            destination=out/('model.pt' if label=='selected' else 'best_trained.pt')
            sha=save_inference_checkpoint(packet,destination)
            final[label]=dict(step=checkpoint['step'],trained=checkpoint['step']>0,sha256=sha,checkpoint_bytes=destination.stat().st_size,
                              validation=score,gates=gates,all_ordinary_gates_pass=all(value for key,value in gates.items() if key!='tenfold_ade'),
                              per_sequence_ade_ratio_to_strongest_baseline=ratios)
        report=dict(architecture=model.architecture,parameters=sum(p.numel() for p in model.parameters()),
                    loop_seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                    evaluation_moving_threshold=threshold,results=final,test_sequences=0,validation_reused=True,
                    source_sha256={name:digest(out/name) for name in ['vector_motion_residual.py','train_vector_motion_residual.py','vector_motion_network.py']})
        (out/'selection.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({key:value for key,value in report.items() if key!='results'}),flush=True)
        print(json.dumps({key:{k:v for k,v in value.items() if k!='validation'} for key,value in final.items()}),flush=True)


if __name__=='__main__': main()
