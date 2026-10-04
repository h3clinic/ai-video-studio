"""Recorded-clip-conditioned 3D acceleration ablation, NOT novel generation.

Uses the already fitted rig and first three fitted poses as an explicit prior.
Those fitted poses saw the clip: this is NOT future-blind seed construction.
The new recurrent weights learn through FK and Gaussian rendering. No scripted
sinusoid or noise is added to make the result look active.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake
from .fit_articulated_weights import ClipJointWeights,to_device,reduced_intrinsics,RIG,OBS,SOURCE
from .fit_motion3d import batched_fk,project_points,PARENTS,read_frames
from .attach_vector_weights import ASSET,CAMERA,original_intrinsics
from .articulated_gaussian import BatchedArticulatedGaussian
from .gaussian3d import render
from .gaussian_motion_memory import select_cat

ROOT=Path('artifacts/real_video/true3d/acceleration_hypothesis/v1/joint_trial')
PRIOR=Path('artifacts/real_video/true3d/articulated_weight_loop/v2_constrained/model.pt')


def seed_model(model,prior,joints,device):
    reference=ClipJointWeights(**prior['model_config']).to(device)
    reference.load_state_dict(prior['candidates']['image_refined']['model'])
    with torch.no_grad():r,x,_=reference(torch.tensor([0.,1/30,2/30],device=device))
    rel=(r[:-1].transpose(-1,-2)@r[1:]).cpu().double().numpy()
    angular=torch.tensor(Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec().reshape(2,17,3),device=device,dtype=torch.float32)*16
    # Both interval angular vectors are expressed in the last body's frame.
    previous=r[1].transpose(-1,-2)@r[2]
    prev_w=(previous.transpose(-1,-2)@angular[0,...,None]).squeeze(-1)
    alpha=(angular[1]-prev_w)*16
    velocity=(x[2]-x[1])*16; acceleration=(x[2]-2*x[1]+x[0])*256
    return model.initialize(rotation=r[2:3],omega=angular[1:2],root_position=x[2:3],
        root_velocity=velocity[None],joints=joints,parents=PARENTS,dt=1/16,
        alpha=alpha[None],root_acceleration=acceleration[None])


def stack_states(states):
    return torch.cat([s['rotation'] for s in states]),torch.cat([s['root_position'] for s in states])


def main(root=ROOT,steps=400,image_steps=60):
    from .joint_acceleration import JointAccelerationNetwork,integrate_joint_state
    from .evaluate_attached_vectors import write_video
    if (root/'protocol.json').exists():raise FileExistsError('Preserve acceleration trial')
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(531001)
    paths=[ASSET,CAMERA,RIG,OBS,SOURCE,PRIOR]
    config=dict(joint_count=17,hidden=48,angular_accel_scale=30.,linear_accel_scale=3.)
    protocol=dict(classification='Full-clip-conditioned 3D dynamics fitting, NOT new-motion generation',
        inputs={str(p):digest(p) for p in paths},model_config=config,steps=steps,image_steps=image_steps,
        seed_source='Three poses from previous full-clip-conditioned fit, NOT future-blind observations',
        image_training_frames=[6,10,18,26,32],withheld_image_frames=[14,22],
        pose_targets='Manually assisted full-video tracks and interpolated intermediate landmarks; not 3D truth',
        selection='Fixed final step, no final-test data; reject on direct visual failures',
        dt_seconds=1/16,all_future_target_frames_used_for_pose_fitting=True,
        appearance_and_binding_frozen=True,geometry_prior_was_already_rejected=True)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    asset=to_device(load_verified(ASSET),device);camera=to_device(load_verified(CAMERA),device)
    prior=load_verified(PRIOR); rig=to_device(load_verified(RIG),device)
    candidate=prior['candidates']['image_refined']
    joints=candidate['joints'].to(device)
    rig.update(joints=joints,vertex_weights=candidate['weights'].to(device))
    decoder=BatchedArticulatedGaussian(asset,rig,train_skinning=False,max_influences=17,min_support=0).to(device)
    observation=to_device(load_verified(OBS),device)
    target=observation['landmarks_condition'][4:33]
    confidence=observation['landmark_confidence'][4:33].clone();confidence[:,[7,10,13,16]]*=2
    model=JointAccelerationNetwork(**config).to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.0007)
    def rollout():
        seed=seed_model(model,prior,joints,device)
        result=model.rollout(seed,28)
        states=[seed]+result['states']; rotations,translations=stack_states(states)
        posed,_=batched_fk(joints,PARENTS,rotations,translations)
        return states,rotations,translations,posed
    def objectives(states,rotations,translations,posed):
        uv=project_points(posed,camera)
        error=(uv-target)/float(camera['width'])
        position=(torch.nn.functional.smooth_l1_loss(error,torch.zeros_like(error),beta=.012,reduction='none').sum(-1)*confidence).sum()/confidence.sum()
        velocity=(torch.diff(uv,dim=0)-torch.diff(target,dim=0))/float(camera['width'])
        vloss=(velocity.square().sum(-1)*torch.minimum(confidence[1:],confidence[:-1])).mean()
        # Dynamics regularization controls runaway; it does not force constant
        # velocity or inject periodic motion. Weak rest-rotation prior is explicit.
        alpha=torch.cat([s['alpha'] for s in states[1:]])
        acceleration=torch.cat([s['root_acceleration'] for s in states[1:]])
        regular=1e-6*alpha.square().mean()+1e-4*acceleration.square().mean()
        root_rotation=(rotations[:,0]-torch.eye(3,device=device)).square().mean()
        return position+2*vloss+regular+.002*root_rotation,position
    history=[];begin=time.perf_counter()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        states,r,x,p=rollout();loss,data=objectives(states,r,x,p)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0 or step+1==steps:
            row=dict(stage='pose',step=step,loss=float(loss.detach()),position=float(data.detach()));history.append(row);print(json.dumps(row),flush=True)
    frames,fps=read_frames();intrinsics=reduced_intrinsics(camera);h,w=intrinsics['height'],intrinsics['width']
    images={}
    for frame in protocol['image_training_frames']:
        mask=cv2.resize(select_cat(frames[frame]),(w,h),interpolation=cv2.INTER_AREA)
        rgb=cv2.resize(frames[frame].astype(np.float32)/255,(w,h),interpolation=cv2.INTER_AREA)
        images[frame]=(torch.tensor(rgb*mask[...,None]+.5*(1-mask[...,None]),device=device),torch.tensor(mask,device=device))
    for group in optimizer.param_groups:group['lr']=.00015
    for step in range(image_steps):
        states,r,x,p=rollout();frame=protocol['image_training_frames'][step%5];index=frame-4
        output=decoder(r[index:index+1],x[index:index+1],return_strain=True)
        rgb,alpha=render(output['position'][0],output['covariance'][0],asset['colour'],asset['opacity'],camera['eye'],camera['target'],**intrinsics,radius=3,ground=False)
        rgb=rgb+(1-alpha[...,None])*(.5-rgb.new_tensor([.15,.19,.24]));target_rgb,mask=images[frame]
        pose,_=objectives(states,r,x,p)
        strain=output['singular_values'].clamp_min(1e-5).log().square().mean()
        loss=(rgb-target_rgb).abs().mean()+4*(alpha-mask).square().mean()+5*pose+.25*strain
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        if step%20==0 or step+1==image_steps:
            row=dict(stage='image',step=step,loss=float(loss.detach()),strain=float(strain.detach()));history.append(row);print(json.dumps(row),flush=True)
    with torch.no_grad():
        states,r,x,p=rollout();learned=[{k:v.detach().cpu().clone() for k,v in s.items()} for s in states]
        seed=seed_model(model,prior,joints,device)
        modes={'learned_acceleration':(r,x)}
        for mode in ['constant_velocity','constant_acceleration']:
            state={k:v.clone() for k,v in seed.items()};baseline=[state]
            for _ in range(28):
                state=integrate_joint_state(state,angular_acceleration=torch.zeros_like(state['alpha']) if mode=='constant_velocity' else seed['alpha'],
                    root_acceleration=torch.zeros_like(state['root_acceleration']) if mode=='constant_velocity' else seed['root_acceleration'])
                baseline.append(state)
            modes[mode]=stack_states(baseline)
        report=dict(protocol=protocol,training_seconds=time.perf_counter()-begin,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
            trained_parameter_tensors_changed=sum(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()),
            accepted=False,reason='Pending explicit visual evaluation; the fixed underlying asset already has anatomy failures.')
        save_inference_checkpoint(dict(model={k:v.cpu() for k,v in model.state_dict().items()},config=config,architecture='joint_acceleration_v1',
            initial_model=initial,initial_state=learned[0],final_state=learned[-1],states=learned,provenance=protocol),root/'model.pt')
        (root/'history.json').write_text(json.dumps(history,indent=2))
        rendered={};vertices={};metrics={};full_intrinsics=original_intrinsics(camera)
        for mode,(rotation,translation) in modes.items():
            views=[];geometry=[]
            for index,frame in enumerate(range(4,33)):
                output=decoder(rotation[index:index+1],translation[index:index+1])
                rgb,alpha=render(output['position'][0],output['covariance'][0],asset['colour'],asset['opacity'],camera['eye'],camera['target'],**full_intrinsics,ground=False)
                rgb=rgb+(1-alpha[...,None])*(.5-rgb.new_tensor([.15,.19,.24]))
                views.append((rgb.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8));geometry.append(output['vertices'][0].cpu())
                if mode=='learned_acceleration' and frame in [14,32]:
                    eye=camera['eye'];center=camera['target'];ang=.45;c,s=np.cos(ang),np.sin(ang)
                    turn=eye.new_tensor([[c,0,s],[0,1,0],[-s,0,c]])
                    oblique,_=render(output['position'][0],output['covariance'][0],asset['colour'],asset['opacity'],center+turn@(eye-center),center,512,512,camera['fov'],ground=False)
                    Image.fromarray((oblique.clamp(0,1).cpu().numpy()*255).astype(np.uint8)).save(root/f'oblique_{frame:02d}.png')
            posed,_=batched_fk(joints,PARENTS,rotation,translation);uv=project_points(posed,camera)
            metrics[mode]=dict(weighted_joint_reprojection_px=float(((uv-target).norm(dim=-1)*confidence).sum()/confidence.sum()*camera['size']/camera['width']))
            rendered[mode]=views;vertices[mode]=torch.stack(geometry)
        comparison=[]
        titles=['SOURCE','LEARNED 3D ACCELERATION | full-clip fitting','CONSTANT VELOCITY | same initial state','CONSTANT ACCELERATION | same initial state']
        for index,frame in enumerate(range(4,33)):
            canvas=Image.new('RGB',(1664,1056),'#151922');draw=ImageDraw.Draw(canvas)
            pictures=[frames[frame]]+[rendered[k][index] for k in modes]
            for slot,(pic,title) in enumerate(zip(pictures,titles)):
                xx=(slot%2)*832;yy=(slot//2)*512;canvas.paste(Image.fromarray(pic),(xx,yy+24));draw.text((xx+8,yy+5),title,fill='white')
            draw.text((8,1034),f'Source frame{frame}/32. Same 45,000 XYZ Gaussians. Acceleration experiment, NOT a new-video generator; quality unapproved.',fill='white')
            comparison.append(np.asarray(canvas))
            if frame in [4,10,14,22,32]:canvas.save(root/f'comparison_{frame:02d}.jpg')
        report['video']=write_video(root/'acceleration_native.mp4',comparison,fps);report['metrics']=metrics
        save_inference_checkpoint(dict(vertices=vertices,camera=to_device(camera,'cpu'),source_frame=torch.arange(4,33),fps=fps,
            asset_path=str(ASSET),asset_sha256=digest(ASSET),model_sha256=digest(root/'model.pt'),classification=protocol['classification']),root/'motion.pt')
        (root/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(saved=str(root),metrics=metrics)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--steps',type=int,default=400);parser.add_argument('--image-steps',type=int,default=60);args=parser.parse_args()
    if args.steps<1 or args.image_steps<0:parser.error('Positive training count required')
    with keep_windows_awake():main(args.root,args.steps,args.image_steps)
