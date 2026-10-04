"""Causal motion-update comparison on the same unchanged Gaussian surface.

No future source images or fitted future states enter prediction. Rendering and
reference evaluation are separate from producing the immutable motion packet.
"""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from .attach_vector_weights import ASSET, CAMERA, SEED, control_binding, original_intrinsics
from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from .dense_surface_fit import ARAPMesh
from .motion_model_io import load_motion_model

OLD=Path('artifacts/real_video/true3d/learned_motion_loop/v3/model.pt')
ROOT=Path('artifacts/real_video/true3d/motion_update_repair/v1')


@torch.no_grad()
def predict(new_model_path,root=ROOT,steps=12):
    if steps<1: raise ValueError('Positive horizon required')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'forecast.pt').exists(): raise FileExistsError('Preserve motion comparison')
    allowed={p.resolve() for p in [ASSET,CAMERA,SEED,OLD,new_model_path]}; reads=[]; original=torch.load
    def guarded(path,*args,**kwargs):
        if Path(path).resolve() not in allowed: raise AssertionError('Unapproved prediction input')
        reads.append(str(Path(path).resolve())); return original(path,*args,**kwargs)
    def forbidden(*args,**kwargs): raise AssertionError('Image/future inputs forbidden')
    with patch('torch.load',side_effect=guarded),patch('cv2.VideoCapture',side_effect=forbidden),patch('cv2.imread',side_effect=forbidden),patch('PIL.Image.open',side_effect=forbidden),patch('numpy.load',side_effect=forbidden):
        asset=load_verified(ASSET); camera=load_verified(CAMERA); seed=load_verified(SEED)
        old_packet=load_verified(OLD); new_packet=load_verified(new_model_path)
        if new_packet.get('architecture')!='residual_velocity_v1': raise ValueError('Expected explicitly tagged new architecture')
        models={'previous_learned':load_motion_model(old_packet),'residual_learned':load_motion_model(new_packet)}
        observed=seed['controls']['position'][None]; adjacency=seed['controls']['adjacency'][None]; confidence=seed['controls']['confidence'][2][None]
        indices,depth,focal,cam_rotation=control_binding(asset['mesh_vertices'],camera,observed[0,2])
        rest=asset['mesh_vertices'].numpy(); solver=ARAPMesh(rest,asset['mesh_faces'].numpy())
        weight=np.zeros(len(rest),np.float64); node_weight=np.maximum(confidence[0].numpy(),.05)*10
        np.add.at(weight,indices,node_weight); active=weight>0
        vertices={}; controls={}; times={}; final_state={}; measures={}
        for mode in ['previous_learned','residual_learned','constant_velocity']:
            model=models['residual_learned'] if mode!='previous_learned' else models[mode]
            state=model.initialize(observed,adjacency,confidence); reference=state['position'][0].clone()
            geometry=[asset['mesh_vertices'].clone()]; path=[reference.clone()]; previous=rest.copy(); elapsed=[]
            for step in range(steps):
                t0=time.perf_counter()
                state=model.step(state) if mode!='constant_velocity' else dict(state,position=state['position']+state['velocity'],time=state['time']+1)
                displacement=(state['position'][0]-reference).numpy()*480
                world=np.column_stack((displacement*depth[:,None]/focal,np.zeros(len(indices))))@cam_rotation
                target=rest.copy(); accum=np.zeros_like(rest,dtype=np.float64)
                np.add.at(accum,indices,(rest[indices]+world)*node_weight[:,None]); target[active]=accum[active]/weight[active,None]
                previous=solver.solve(target,weight,stiffness=6.,iterations=8,prior=.002,initial=previous)
                geometry.append(torch.from_numpy(previous.copy())); path.append(state['position'][0].clone())
                elapsed.append((time.perf_counter()-t0)*1000)
            vertices[mode]=torch.stack(geometry); controls[mode]=torch.stack(path); times[mode]=float(np.median(elapsed))
            final_state[mode]={k:v.clone() for k,v in state.items()}
            velocity=torch.diff(controls[mode],dim=0)*480
            measures[mode]=dict(initial_speed_px=float(((observed[0,2]-observed[0,1])*480).norm(dim=-1).mean()),
                mean_step_speed_px=float(velocity.norm(dim=-1).mean()),last_step_speed_px=float(velocity[-1].norm(dim=-1).mean()),
                endpoint_control_motion_px=float(((controls[mode][-1]-reference)*480).norm(dim=-1).mean()))
            print(json.dumps(dict(mode=mode,**measures[mode])),flush=True)
        packet=dict(vertices=vertices,controls=controls,final_state=final_state,camera=camera,source_frame=torch.arange(2,steps+3),fps=16.,
            asset_path=str(ASSET),asset_sha256=digest(ASSET),seed_sha256=digest(SEED),old_model_sha256=digest(OLD),new_model_sha256=digest(new_model_path),
            model_paths={'previous_learned':str(OLD),'residual_learned':str(new_model_path)},
            model_architectures={'previous_learned':'damped_velocity_v1','residual_learned':'residual_velocity_v1'},control_vertex_index=torch.from_numpy(indices),
            trained_horizon_steps=12,forecast_beyond_trained_horizon=steps>12,
            classification='Conditional XY motion prediction on inferred XYZ support, unchanged surface solver. No learned hidden depth or text-to-video.',
            new_trained=bool(new_packet.get('trained',new_packet.get('selected_step',0)>0)))
    sha=save_inference_checkpoint(packet,root/'forecast.pt')
    report=dict(forecast_sha256=sha,inputs=reads,future_reads=0,observed_frames=3,predicted_steps=steps,
        new_weights_sha256=digest(new_model_path),new_trained=packet['new_trained'],motion_measures=measures,
        median_network_and_solver_ms=times,surface_solver_unchanged=True,posthoc_motion_gain=1.,
        source_sha256=digest(Path(__file__)),limits='More movement alone is not more accurate articulation. No gradient through this NumPy surface solver during prior control training. Physical time transfer remains uncalibrated.')
    (root/'inference_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)); print(json.dumps(report),flush=True)


@torch.no_grad()
def evaluate(root=ROOT):
    from PIL import Image,ImageDraw
    import cv2
    from .gaussian3d import render,project
    from .vector_recording import GaussianSurfaceMemory
    from .evaluate_attached_vectors import read_evaluation_frames,write_video
    from .gaussian_motion_memory import select_cat
    output=root/'evaluation'; output.mkdir(exist_ok=True)
    if (output/'evaluation.json').exists(): raise FileExistsError('Preserve comparison evaluation')
    packet=load_verified(root/'forecast.pt'); sha=digest(root/'forecast.pt'); indices=packet['source_frame'].tolist()
    if Path(packet['asset_path']).resolve()!=ASSET.resolve() or digest(ASSET)!=packet['asset_sha256']:
        raise ValueError('Forecast and renderer asset provenance differ')
    # This is the FIRST future-source read, after prediction was frozen on disk.
    source,fps=read_evaluation_frames(indices)
    if fps!=16 or any(frame.shape!=(480,832,3) for frame in source):
        raise ValueError('Expected pinned 832x480 16fps diagnostic source')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    asset={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in load_verified(ASSET).items()}
    camera={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in packet['camera'].items()}
    memory=GaussianSurfaceMemory(asset); intrinsics=original_intrinsics(camera); renderer_bg=torch.tensor([.15,.19,.24],device=device)
    masks=[select_cat(frame).astype(np.float32) for frame in source]
    target=[(frame.astype(np.float32)/255)*mask[...,None]+.5*(1-mask[...,None]) for frame,mask in zip(source,masks)]
    rendered={}; metrics={}; activity={}
    for mode,trajectory in packet['vertices'].items():
        frames=[]; rows=[]; projected=[]
        for t,vertices in enumerate(trajectory):
            state=memory.decode(vertices.to(device))
            rgb,alpha=render(state['position'],state['covariance'],asset['colour'],asset['opacity'],camera['eye'],camera['target'],**intrinsics,ground=False)
            rgb=rgb+(1-alpha[...,None])*(.5-renderer_bg)
            array=rgb.clamp(0,1).cpu().numpy(); frames.append((array*255).round().astype(np.uint8))
            uv=project(state['position'],state['covariance'],camera['eye'],camera['target'],**intrinsics)[0].cpu(); projected.append(uv)
            pred_mask=alpha.cpu().numpy()>.5; ref_mask=masks[t]>.5
            rows.append(dict(source_frame=indices[t],roi_mae=float(np.abs(array[80:480,:670]-target[t][80:480,:670]).mean()),
                approximate_iou=float((pred_mask&ref_mask).sum()/max(1,(pred_mask|ref_mask).sum()))))
        rendered[mode]=frames
        metrics[mode]=dict(per_frame=rows,future_mean_mae=float(np.mean([v['roi_mae'] for v in rows[1:]])),future_mean_iou=float(np.mean([v['approximate_iou'] for v in rows[1:]])))
        metrics[mode]['first12_mean_mae']=float(np.mean([v['roi_mae'] for v in rows[1:13]]))
        metrics[mode]['first12_mean_iou']=float(np.mean([v['approximate_iou'] for v in rows[1:13]]))
        if len(rows)>13:
            metrics[mode]['beyond12_mean_mae']=float(np.mean([v['roi_mae'] for v in rows[13:]]))
            metrics[mode]['beyond12_mean_iou']=float(np.mean([v['approximate_iou'] for v in rows[13:]]))
        uv=torch.stack(projected); delta=(uv[-1]-uv[0]).norm(dim=-1)
        activity[mode]=dict(gaussian_endpoint_mean_px=float(delta.mean()),gaussian_endpoint_median_px=float(delta.median()),
            fraction_over_one_px=float((delta>1).float().mean()),fraction_over_ten_px=float((delta>10).float().mean()))
    frames=[]
    titles=['Observed source (evaluation only)','Previous learned update (damped)','New learned velocity-residual update','Constant velocity (NOT learned)']
    for t in range(len(source)):
        row=Image.new('RGB',(1664,1032),'#151b24'); draw=ImageDraw.Draw(row)
        images=[source[t],rendered['previous_learned'][t],rendered['residual_learned'][t],rendered['constant_velocity'][t]]
        for j,(name,array) in enumerate(zip(titles,images)):
            x=(j%2)*832; y=(j//2)*504; row.paste(Image.fromarray(array),(x,y+24)); draw.text((x+8,y+6),name,fill='white')
        horizon_note='BEYOND 12-step training horizon' if t>12 else 'Within 12-step training horizon'
        draw.text((8,1015),f'Native timestep {t}; source frame {indices[t]}; same 45,000 XYZ splats. {horizon_note}. Movement is not proof of correct gait.',fill='white')
        frames.append(np.asarray(row))
    qa=write_video(output/'motion_updates_native.mp4',frames,fps)
    # No repeated/held frames are added to simulate a longer generated duration.
    for t in sorted(set([0,len(frames)//2,len(frames)-1])): Image.fromarray(frames[t]).save(output/f'comparison_{t:02d}.jpg',quality=94)
    if digest(root/'forecast.pt')!=sha: raise AssertionError('Forecast modified during evaluation')
    report=dict(forecast_sha256=sha,metrics=metrics,activity=activity,video=qa,source_reads_only_after_forecast=True,
        source_mask='Heuristic GrabCut, not ground truth',pose_quality_claim=False,posthoc_motion_amplification=False,
        trained_horizon_steps=12,forecast_beyond_trained_horizon=len(indices)>13,
        temporal_scope=f'Native {len(indices)}-state {fps:g}fps forecast, no held/interpolated extension or synthetic gait.',
        limitation='Activity metrics expose freezing but cannot certify correct motion; wrong moving predictions must still fail quality review.')
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['predict','evaluate']); parser.add_argument('--model',type=Path); parser.add_argument('--root',type=Path,default=ROOT); parser.add_argument('--steps',type=int,default=12); args=parser.parse_args()
    torch.set_num_threads(4)
    with keep_windows_awake():
        if args.mode=='predict':
            if args.model is None: parser.error('--model is required for predict')
            predict(args.model,args.root,args.steps)
        else: evaluate(args.root)
