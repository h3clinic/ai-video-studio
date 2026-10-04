"""Attach causal learned projected-motion weights to a persistent XYZ surface.

Forecast reads only three-observation seed, appearance asset, source-view camera
and inference weights. It never reads a fitted full-video motion packet.
Depth comes from the inferred rest asset, NOT learned depth supervision.
"""
import argparse
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial import cKDTree
from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake
from .gaussian3d import project,look_at
from .vector_motion_network import VectorMotionNetwork
from .vector_recording import record_sequence,recording_audit

ROOT=Path('artifacts/real_video/true3d/attached_vector_weights/v1')
ASSET=Path('artifacts/real_video/true3d/v4_appearance/cat_asset.pt')
CAMERA=Path('artifacts/real_video/true3d/v2_appearance/camera.pt')
SEED=Path('artifacts/real_video/animal_graph/v1/seeds/cat.pt')
MODEL=Path('artifacts/real_video/true3d/learned_motion_loop/v3/model.pt')


def original_intrinsics(camera,height=480,width=832):
    principal=[camera['size']/2+camera['x0']-camera['pad_x'],camera['size']/2+camera['y0']-camera['pad_y']]
    fov=math.degrees(2*math.atan(height/camera['size']*math.tan(math.radians(camera['fov'])/2)))
    return dict(height=height,width=width,fov=fov,principal=principal)


def control_binding(vertices,camera,reference):
    intrinsics=original_intrinsics(camera); n=len(vertices)
    uv,_,depth,_,focal=project(vertices,torch.eye(3).expand(n,3,3)*1e-6,camera['eye'],camera['target'],**intrinsics)
    tree=cKDTree(uv.numpy()); distances,indices=tree.query(reference.numpy()*480,k=32)
    selected=[]
    for ds,ids in zip(distances,indices):
        close=ids[ds<=max(float(ds[0])+2,4)]
        selected.append(int(close[depth[close].argmin()]))
    indices=np.array(selected,dtype=np.int64)
    return indices,depth[indices].numpy(),float(focal),look_at(camera['eye'],camera['target']).numpy()


@torch.no_grad()
def predict(model_path=MODEL,steps=12):
    from .dense_surface_fit import ARAPMesh
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'forecast.pt').exists(): raise FileExistsError('Preserve forecast')
    allowed={p.resolve() for p in [ASSET,CAMERA,SEED,model_path]}; reads=[]
    original_load=torch.load
    def guarded_load(path,*args,**kwargs):
        if Path(path).resolve() not in allowed: raise AssertionError('Unexpected forecast input '+str(path))
        reads.append(str(Path(path).resolve())); return original_load(path,*args,**kwargs)
    def forbidden(*args,**kwargs): raise AssertionError('Future/image input forbidden in forecast')
    # All file-loaded inputs are enumerated; future references live in separate eval.
    with patch('torch.load',side_effect=guarded_load),patch('cv2.VideoCapture',side_effect=forbidden),patch('cv2.imread',side_effect=forbidden),patch('PIL.Image.open',side_effect=forbidden),patch('numpy.load',side_effect=forbidden):
        a=load_verified(ASSET); camera=load_verified(CAMERA); seed=load_verified(SEED); checkpoint=load_verified(model_path)
        model=VectorMotionNetwork(**checkpoint.get('model_config',checkpoint['config'])).eval(); model.load_state_dict(checkpoint['model'],strict=True)
        disabled=VectorMotionNetwork(**model.config).eval()
        for parameter in disabled.parameters(): parameter.zero_()
        observed=seed['controls']['position'][None]; adjacency=seed['controls']['adjacency'][None]
        confidence=seed['controls']['confidence'][2][None]
        indices,depth,focal,cam_rotation=control_binding(a['mesh_vertices'],camera,observed[0,2])
        rest=a['mesh_vertices'].numpy(); solver=ARAPMesh(rest,a['mesh_faces'].numpy())
        all_modes={}; histories={}; times={}; state_sizes={}; resume_error=None
        for mode in ['learned','zero_weights','velocity','average_velocity','damped_velocity','frozen']:
            active_model=disabled if mode=='zero_weights' else model
            state=active_model.initialize(observed,adjacency,confidence); reference=state['position'][0].clone()
            if mode=='average_velocity': state=dict(state,velocity=(observed[:,2]-observed[:,0])/2)
            sequence=[torch.from_numpy(rest.copy())]; controls=[reference.clone()]; previous=rest.copy(); elapsed=[]; sizes=[]
            snapshot=None
            for step in range(steps):
                start=time.perf_counter()
                if mode in ['learned','zero_weights']: state=active_model.step(state)
                elif mode in ['velocity','average_velocity','damped_velocity']:
                    velocity=state['velocity']*(.95 if mode=='damped_velocity' else 1.)
                    state=dict(state,position=state['position']+velocity,velocity=velocity,time=state['time']+1)
                displacement=(state['position'][0]-reference).numpy()*480
                world=np.column_stack((displacement*depth[:,None]/focal,np.zeros(len(indices))))@cam_rotation
                target=rest.copy(); weight=np.zeros(len(rest),np.float64); accum=np.zeros_like(rest,dtype=np.float64)
                node_weight=np.maximum(confidence[0].numpy(),.05)*10
                np.add.at(accum,indices,(rest[indices]+world)*node_weight[:,None]); np.add.at(weight,indices,node_weight)
                active=weight>0; target[active]=accum[active]/weight[active,None]
                moved=solver.solve(target,weight,stiffness=6.,iterations=8,prior=.002,initial=previous)
                previous=moved
                elapsed.append((time.perf_counter()-start)*1000); sequence.append(torch.tensor(moved,dtype=torch.float32)); controls.append(state['position'][0].clone())
                sizes.append(sum(v.numel()*v.element_size() for v in state.values() if isinstance(v,torch.Tensor)))
                if mode=='learned' and step==3: snapshot={k:v.clone() for k,v in state.items()}
                if mode=='learned' and step==7:
                    restored=snapshot
                    for _ in range(4): restored=model.step(restored)
                    resume_error=max(float((restored[k]-state[k]).abs().max()) for k in state)
            all_modes[mode]=torch.stack(sequence); histories[mode]=torch.stack(controls); times[mode]=float(np.median(elapsed)); state_sizes[mode]=sizes
            print(json.dumps(dict(mode=mode,states=len(sequence),median_network_and_surface_ms=times[mode])),flush=True)
        packet=dict(vertices=all_modes,controls=histories,camera=camera,source_frame=torch.arange(2,steps+3),fps=16.,asset_path=str(ASSET),asset_sha256=digest(ASSET),
            model_path=str(model_path),model_sha256=digest(model_path),seed_sha256=digest(SEED),control_vertex_index=torch.from_numpy(indices),
            classification='Causal learned projected-motion forecast lifted to persistent inferred XYZ mesh; NOT learned 3D depth or text-to-video.')
        report=dict(inputs=reads,future_image_inputs=0,future_motion_inputs=0,observed_frames=3,predict_steps=steps,
            model_parameters=sum(p.numel() for p in model.parameters()),model_sha256=digest(model_path),model_strictly_loaded=True,
            state_bytes={k:v[0] for k,v in state_sizes.items()},state_size_constant=all(len(set(v))==1 for v in state_sizes.values()),
            recurrent_state_resume_max_error=resume_error,median_network_and_surface_ms=times,renderer_included=False,
            limits='Sparse planar training targets; inferred fixed depth prior. Neither per-dot physical truth nor stochastic text-to-video generation. ARAP factors/workspace and renderer excluded from state_bytes. DAVIS/cat physical time calibration unverified.')
    # Verified atomic export reloads its own temporary file. Keep that validation
    # outside the input-only guard so it cannot be mistaken for a forecast input.
    save_inference_checkpoint(packet,ROOT/'forecast.pt')
    report['forecast_sha256']=digest(ROOT/'forecast.pt')
    report['source_sha256']={str(path):digest(path) for path in [Path(__file__),Path(__file__).with_name('dense_surface_fit.py'),Path(__file__).with_name('vector_motion_network.py')]}
    (ROOT/'inference_audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


def record(audit_only=False):
    if (ROOT/'gaussian_vectors.pt').exists() and not audit_only: raise FileExistsError('Preserve recorded vectors')
    if (ROOT/'recording_audit.json').exists(): raise FileExistsError('Preserve recorded audit')
    p=load_verified(ROOT/'forecast.pt'); a=load_verified(ASSET)
    if audit_only:
        result=load_verified(ROOT/'gaussian_vectors.pt')
        if result['model_sha256']!=p['model_sha256'] or result['asset_sha256']!=p['asset_sha256']: raise ValueError('Record provenance mismatch')
    else:
        result=record_sequence(a,p['vertices']['learned'],p['fps'])
        result.update(classification=p['classification'],model_sha256=p['model_sha256'],asset_sha256=p['asset_sha256'])
        save_inference_checkpoint(result,ROOT/'gaussian_vectors.pt')
    report=recording_audit(result); report.update(packet_bytes=(ROOT/'gaussian_vectors.pt').stat().st_size,model_sha256=p['model_sha256'])
    (ROOT/'recording_audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['predict','record','audit']); parser.add_argument('--model',type=Path,default=MODEL); parser.add_argument('--steps',type=int,default=12); parser.add_argument('--root',type=Path,default=ROOT); args=parser.parse_args()
    ROOT=args.root
    torch.set_num_threads(4)
    with keep_windows_awake():
        if args.mode=='predict': predict(args.model,args.steps)
        else: record(audit_only=args.mode=='audit')
