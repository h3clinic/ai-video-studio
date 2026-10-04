"""V4: bounded sequence-balanced, evaluator-aligned projected motion training.

Only DAVIS train sequences train the weights. Validation was reused after v2/v3
failures and is explicitly NOT an unbiased final test. Initial step zero is a
legal fallback, separately identified as untrained, not a learning success.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .vector_motion_network import VectorMotionNetwork
from .train_vector_motion_loop import DATA, sample, evaluate
from .checkpoint_io import load_verified, save_training_checkpoint, save_inference_checkpoint, digest, keep_windows_awake

OUT=Path('artifacts/real_video/true3d/learned_motion_loop/v4')


def sanitize_reports(out):
    """Preserve raw frozen-evaluator NaN diagnostics; JSON null means undefined.

    This is post-run serialization repair only, not a metric/evaluator change.
    The exact source snapshot used to train remains unmodified.
    """
    records=[]
    for path in list(out.rglob('*.json')):
        if path.name.endswith('.raw-nonfinite.json'): continue
        data=json.loads(path.read_text()); changed=[]
        def clean(value,location=''):
            if isinstance(value,float) and not math.isfinite(value): changed.append(location); return None
            if isinstance(value,dict): return {k:clean(v,location+'/'+k) for k,v in value.items()}
            if isinstance(value,list): return [clean(v,location+'/'+str(i)) for i,v in enumerate(value)]
            return value
        safe=clean(data)
        if not changed: continue
        raw=path.with_name(path.stem+'.raw-nonfinite.json')
        if raw.exists(): raise FileExistsError('Preserve previous raw metric report')
        shutil.copy2(path,raw)
        path.write_text(json.dumps(safe,indent=2,allow_nan=False))
        records.append(dict(path=str(path),raw_path=str(raw),raw_sha256=digest(raw),safe_sha256=digest(path),undefined_paths=changed))
    report=dict(reason='Frozen evaluator train edge_error divides by zero for windows with entirely zero adjacency. Training loss, validation ADE/FDE, selection and all weights are unchanged. Undefined train edge diagnostics represented as JSON null, not zero. Raw reports and training source snapshots preserved.',records=records)
    (out/'serialization_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)); print(json.dumps(dict(sanitized_reports=len(records))),flush=True)


def aligned_loss(prediction, values, velocity_weight=.05, epsilon=.01):
    """Robust Euclidean error, equally averaged across windows and horizons.

    Zero-visibility/confidence points contribute exactly zero, never a floor.
    Empty horizons contribute zero exactly as the frozen evaluator does.
    """
    ref=values['position'][:,2]; target=values['position'][:,3:]
    scale=(ref.amax(1,keepdim=True)-ref.amin(1,keepdim=True)).amax(-1,keepdim=True).clamp_min(1e-5)
    weight=values['confidence'][:,3:]*values['visibility'][:,3:]
    def weighted_distance(delta):
        residual=delta/scale[:,None]*32
        norm=(residual.square().sum(-1)+epsilon**2).sqrt()-epsilon
        return ((norm*weight).sum(-1)/weight.sum(-1).clamp_min(1e-8)).mean()
    position=weighted_distance(prediction-target)
    predicted_velocity=torch.diff(torch.cat((ref[:,None],prediction),1),dim=1)
    target_velocity=torch.diff(values['position'][:,2:],dim=1)
    velocity=weighted_distance(predicted_velocity-target_velocity)
    return position+velocity_weight*velocity


def balanced_indices(groups,batch,generator):
    """Choose sequences uniformly, then uniformly choose their train windows."""
    names=sorted(groups); chosen=torch.randint(len(names),(batch,),generator=generator).tolist(); indices=[]
    for index in chosen:
        windows=groups[names[index]]
        indices.append(windows[int(torch.randint(len(windows),(1,),generator=generator))])
    return indices


def gates_for(scores,trained):
    means=scores['means']; modes=['frozen','velocity','average_velocity','damped_velocity','zero_weights']
    best=min(modes,key=lambda x:means[x]['ade']); ratio=means['learned']['ade']/means[best]['ade']
    sequence_ratios={name:item['learned']['ade']/min(item[m]['ade'] for m in modes) for name,item in scores['sequence_scores'].items()}
    gates=dict(macro_ade_10percent=ratio<=.9,no_sequence_10percent_regression=max(sequence_ratios.values())<=1.1,
        fde_no_regression=means['learned']['fde']<=min(means[m]['fde'] for m in modes),
        edge_error_no_10percent_regression=means['learned']['edge_error']<=1.1*min(means[m]['edge_error'] for m in modes),
        learned_beats_disabled_ade=bool(trained and means['learned']['ade']<means['zero_weights']['ade']-1e-9),
        learned_beats_disabled_fde=bool(trained and means['learned']['fde']<means['zero_weights']['fde']-1e-9),tenfold_ade=ratio<=.1)
    return dict(gates=gates,ordinary_gates_pass=all(v for k,v in gates.items() if k!='tenfold_ade'),best_macro_baseline=best,
                ade_ratio_to_best_baseline=ratio,per_sequence_ratio_to_best_baseline=sequence_ratios,
                learned_benefit=bool(gates['learned_beats_disabled_ade'] and gates['learned_beats_disabled_fde']))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=OUT); parser.add_argument('--sanitize-reports',action='store_true'); args=parser.parse_args(); out=args.output
    if args.sanitize_reports: sanitize_reports(out); return
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('Use fresh output; preserve all bounded-loop artifacts')
    torch.set_num_threads(4); device='cuda' if torch.cuda.is_available() else 'cpu'
    data=load_verified(DATA)
    train=[i for i,m in enumerate(data['metadata']) if m['split']=='train']; val=[i for i,m in enumerate(data['metadata']) if m['split']=='validation']
    groups={name:[i for i in train if data['metadata'][i]['sequence']==name] for name in sorted({data['metadata'][i]['sequence'] for i in train})}
    val_sequences=sorted({data['metadata'][i]['sequence'] for i in val})
    if not train or not val or set(groups)&set(val_sequences): raise ValueError('Invalid/disjoint split required')
    if any(m['split'] not in ['train','validation'] for m in data['metadata']): raise ValueError('Final test data prohibited')
    configs=[dict(name='aligned_compact32',hidden=32,residual_scale=.002,damping_init=.75,seed=431006),
             dict(name='aligned_residual64',hidden=64,residual_scale=.005,damping_init=.75,seed=431007)]
    sources=['train_vector_motion_aligned.py','train_vector_motion_loop.py','vector_motion_network.py']
    snapshot=out/'source_snapshot'; snapshot.mkdir()
    for name in sources: shutil.copy2(Path('real_video')/name,snapshot/name)
    protocol=dict(candidates=configs,steps_per_candidate=500,maximum_total_steps=1000,batch=16,learning_rate=.0002,weight_decay=.01,
        evaluate_every=100,gradient_norm_clip=1.,train_windows=len(train),validation_windows=len(val),train_sequence_window_counts={k:len(v) for k,v in groups.items()},
        train_sequences=sorted(groups),validation_sequences=val_sequences,data_sha256=digest(DATA),
        sampling='Uniform train sequence, then uniform window within that sequence, with replacement; no validation sampling for gradients.',
        loss='For each window and each of12futurehorizons, sum(confidence*visibility*smoothEuclidean(error/bboxspan*32))/sum(confidence*visibility); equal window/horizon average. Add0.05sameweightedEuclideanvelocityerror. sqrt(norm^2+0.01^2)-0.01. No confidence floor.',
        augmentation='Train-only x reflection; no added observation noise in this preregistered aligned loop.',
        normalization='Observed frame2 bounding span only, identical frozen evaluator scale. No fitting normalization on validation/test.',
        selected_by='Minimum sequence-macro visible-confidence validation ADE including initial step0; strictly improving tolerance1e-9. Separately retain besttrained step>0.',
        initial_fallback='If step0 wins, export untrained0.75dampingbaseline; learned_benefitfalse. Never force selection of degraded trained model.',
        validation_reused_for_adaptation=True,validation_is_unbiased_final_test=False,adaptation='v3 training/validation mismatch and imbalance analysis; no cat or final-test tuning.',
        evaluator='Imported existing train_vector_motion_loop.evaluate unchanged, source snapshot hashed.',
        gates='10%macroADEgain overstrongestbaseline includingzero; nosequence>10%regression; FDE<=strongestbaseline; edge<=1.1bestbaseline; trainedADEandFDEstrictlybeatzeroweights. 10xADEseparate.',
        seed_frames=3,predicted_frames=12,teacher_forcing=False,test_sequences=0,no_cat_video_selection=True,
        source_files_sha256={name:digest(snapshot/name) for name in sources},
        research_source='https://arxiv.org/html/2608.25956v1 A.4 equations28-29: confidence-normalized robust vector penalties. Our projected48control loss/sequence balancing is an engineering adaptation, not a reproduction.',
        claim='Learned deterministic conditional XY control forecast only. No text/noise-to-video or learned hidden 3D anatomy.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)); start=time.perf_counter(); results=[]
    with keep_windows_awake():
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        for config in configs:
            path=out/config['name']; path.mkdir(); torch.manual_seed(config['seed']); generator=torch.Generator().manual_seed(config['seed']+100)
            model=VectorMotionNetwork(**{k:config[k] for k in ['hidden','residual_scale','damping_init']}).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
            initial=evaluate(model,data,val,device,True); (path/'initial.json').write_text(json.dumps(initial,indent=2))
            best_any=initial['means']['learned']['ade']; best_any_step=0; best_trained=float('inf'); best_trained_step=None; history=[]; sampled_counts=Counter()
            def save_step(step,score,is_best):
                checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),config=model.config,step=step,score=score,
                    torch_rng=torch.get_rng_state(),sampling_rng=generator.get_state(),candidate=config,
                    cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],data_sha256=protocol['data_sha256'],protocol_sha256=digest(out/'protocol.json'))
                save_training_checkpoint(checkpoint,path,step,is_best)
            save_step(0,initial,True)
            for step in range(1,501):
                model.train(); indices=balanced_indices(groups,16,generator); sampled_counts.update(data['metadata'][i]['sequence'] for i in indices)
                values=sample(data,indices,device); mirror=torch.rand(len(indices),device=device)<.5; values['position'][mirror,:,:,0]*=-1
                prediction,_=model.rollout(values['position'][:,:3],values['adjacency'],12,values['confidence'][:,2])
                objective=aligned_loss(prediction,values)
                if not torch.isfinite(objective): raise ValueError('Nonfinite aligned loss')
                optimizer.zero_grad(set_to_none=True); objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                if step%100==0:
                    score=evaluate(model,data,val,device); value=score['means']['learned']['ade']; improved=value<best_any-1e-9
                    if improved: best_any=value; best_any_step=step
                    if value<best_trained: best_trained=value; best_trained_step=step
                    save_step(step,score,improved)
                    progress=dict(candidate=config['name'],step=step,loss=float(objective.detach()),validation_ade=value,best_including_initial=best_any,best_any_step=best_any_step,best_trained=best_trained,best_trained_step=best_trained_step,seconds=time.perf_counter()-start)
                    history.append(dict(**progress,validation=score)); (path/'history.json').write_text(json.dumps(history,indent=2)); (out/'progress.json').write_text(json.dumps(progress,indent=2)); print(json.dumps(progress),flush=True)
            selections={}
            for label,chosen_step in [('including_initial',best_any_step),('trained_only',best_trained_step)]:
                checkpoint=load_verified(path/'checkpoints'/f'step_{chosen_step:08d}.pt'); model.load_state_dict(checkpoint['model']); scores=evaluate(model,data,val,device,True)
                train_scores=evaluate(model,data,train,device,True); trained=chosen_step>0
                packet=dict(model=checkpoint['model'],config=model.config,model_config=model.config,candidate=config,selected_step=chosen_step,
                    trained=trained,selection_kind=label,data_sha256=protocol['data_sha256'],protocol_sha256=digest(out/'protocol.json'),learned_dimension=2,
                    initialization_fallback=not trained,validation_reused_for_adaptation=True,
                    inference_contract='initialize(three observed XY positions, observed adjacency, observedconfidence); step(state) needs no targets or images')
                filename='model.pt' if label=='including_initial' else 'best_trained_model.pt'
                sha=save_inference_checkpoint(packet,path/filename)
                selections[label]=dict(step=chosen_step,trained=trained,weights_sha256=sha,scores=scores,train_scores=train_scores,**gates_for(scores,trained))
            result=dict(name=config['name'],selections=selections,parameters=sum(p.numel() for p in model.parameters()),sampled_sequence_counts=dict(sampled_counts))
            results.append(result); (path/'results.json').write_text(json.dumps(result,indent=2))
        best=min(results,key=lambda r:r['selections']['including_initial']['scores']['means']['learned']['ade']); selected_result=best['selections']['including_initial']
        packet=load_verified(out/best['name']/'model.pt'); sha=save_inference_checkpoint(packet,out/'model.pt')
        best_trained=min(results,key=lambda r:r['selections']['trained_only']['scores']['means']['learned']['ade']); trained_result=best_trained['selections']['trained_only']
        trained_packet=load_verified(out/best_trained['name']/'best_trained_model.pt'); trained_sha=save_inference_checkpoint(trained_packet,out/'best_trained_model.pt')
        report=dict(selected_candidate=best['name'],selected_step=selected_result['step'],selected_weights_sha256=sha,
            selected_is_trained=selected_result['trained'],initialization_fallback=not selected_result['trained'],
            best_trained_candidate=best_trained['name'],best_trained_step=trained_result['step'],best_trained_weights_sha256=trained_sha,
            best_trained_selection=trained_result,**gates_for(selected_result['scores'],selected_result['trained']),
            selected_scores=selected_result['scores'],candidates=results,loop_seconds=time.perf_counter()-start,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,validation_reused_for_adaptation=True,validation_is_unbiased_final_test=False,
            test_sequences=0,no_cat_video_selection=True,learned_depth=False,text_to_video=False,
            quality_claim='Validation2Dcontrolforecast only. No10xRGBquality, true3Dphysicalmotion, or qualitymatchedcompute/storageclaim.')
        (out/'selection.json').write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k not in ['candidates','selected_scores','best_trained_selection']}),flush=True)


if __name__=='__main__': main()
