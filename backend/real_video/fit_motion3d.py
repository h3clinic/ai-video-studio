"""Video-conditioned articulated XYZ Gaussian motion fitting, NOT generation.

Manual anatomical image landmarks are propagated with forward/backward optical
flow and fitted using constant-length forward kinematics. Colours/IDs are never
optimized here. Single-view depth and hidden joints remain inferred.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from .gaussian3d import look_at, rotation, forward_kinematics, project, skin, render

SOURCE=Path('artifacts/real_video/wan_baseline/cat_seed_421001/wan_original.mp4')
ASSET=Path('artifacts/real_video/true3d/v1/cat_asset.pt')
OUT=Path('artifacts/real_video/true3d/v2_motion')
NAMES=['root','neck','head','tail_base','tail_tip',
       'rear_near_hip','rear_near_knee','rear_near_foot',
       'rear_far_hip','rear_far_knee','rear_far_foot',
       'front_near_shoulder','front_near_elbow','front_near_foot',
       'front_far_shoulder','front_far_elbow','front_far_foot']
PARENTS=[-1,0,1,0,3,0,5,6,0,8,9,0,11,12,0,14,15]
# Explicit human-assigned image landmarks, original 832x480 source frame 2.
LANDMARKS=np.array([[255,226],[389,229],[465,191],[87,194],[18,262],
                    [113,267],[128,340],[166,403],[90,258],[88,330],[101,381],
                    [361,260],[406,338],[451,417],[317,268],[293,339],[275,391]],np.float32)
# Source-observed lower-limb anchors, not learned predictions. Limb identities
# through projected crossings are manually inferred and remain uncertain.
MANUAL_LIMBS={
    2:LANDMARKS[5:].tolist(),
    10:[[140,256],[164,327],[230,383],[123,256],[118,339],[160,403],
        [377,260],[395,334],[428,415],[330,248],[332,314],[326,370]],
    18:[[166,251],[230,332],[292,403],[138,263],[104,340],[129,400],
        [369,265],[375,346],[385,416],[358,253],[405,329],[462,390]],
    26:[[186,265],[227,348],[277,405],[145,263],[98,340],[110,398],
        [380,260],[362,334],[368,416],[415,262],[447,341],[554,444]],
    32:[[209,265],[226,347],[251,403],[164,263],[113,340],[108,395],
        [365,261],[353,336],[347,416],[439,274],[451,356],[549,449]],
}


def read_frames():
    cap=cv2.VideoCapture(str(SOURCE)); fps=cap.get(cv2.CAP_PROP_FPS); frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames)!=33: raise ValueError('Expected pinned 33-frame Wan source')
    return frames,fps


def contact_sheet():
    OUT.mkdir(parents=True,exist_ok=True)
    frames,_=read_frames(); canvas=Image.new('RGB',(832*2,520*4),'#121820'); d=ImageDraw.Draw(canvas)
    for slot,i in enumerate([2,6,10,14,18,22,26,32]):
        Image.fromarray(frames[i]).save(OUT/f'source_{i:02d}.png')
        x,y=slot%2*832,slot//2*520; pic=Image.fromarray(frames[i]); draw=ImageDraw.Draw(pic)
        if i==2:
            for j,(u,v) in enumerate(LANDMARKS):
                draw.ellipse((u-4,v-4,u+4,v+4),fill='cyan'); draw.text((u+5,v),str(j),fill='yellow')
        canvas.paste(pic,(x,y+30)); d.text((x+8,y+6),f'SOURCE frame {i}',fill='white')
    canvas.save(OUT/'source_contact.jpg')


def track_landmarks(frames, rest=2, initial=LANDMARKS):
    """Sparse LK with forward/backward rejection; no extrapolated confidence."""
    gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in frames]
    track=np.zeros((len(frames),len(initial),2),np.float32)
    confidence=np.zeros((len(frames),len(initial)),np.float32)
    track[rest]=initial; confidence[rest]=1
    args=dict(winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    for direction in [-1,1]:
        for current in range(rest+direction,-1 if direction<0 else len(frames),direction):
            previous=current-direction; p=track[previous].reshape(-1,1,2)
            q,ok,error=cv2.calcOpticalFlowPyrLK(gray[previous],gray[current],p,None,**args)
            back,ok_back,_=cv2.calcOpticalFlowPyrLK(gray[current],gray[previous],q,None,**args)
            fb=np.linalg.norm(back[:,0]-p[:,0],axis=-1)
            valid=ok[:,0].astype(bool)&ok_back[:,0].astype(bool)&(fb<1.5)&(error[:,0]<35)
            valid&=(q[:,0,0]>=0)&(q[:,0,0]<frames[0].shape[1])&(q[:,0,1]>=0)&(q[:,0,1]<frames[0].shape[0])
            track[current]=q[:,0]
            confidence[current]=confidence[previous]*valid.astype(np.float32)*np.exp(-fb*.08)
    return track,confidence


def apply_manual_limb_anchors(track,confidence):
    """Interpolated manual anchors replace failed/swap-prone limb flow.

    Intermediate image-space interpolation is only a fitting observation prior;
    rendered XYZ points still come from rotation-only FK with fixed bones.
    """
    from scipy.interpolate import PchipInterpolator
    times=np.array(sorted(MANUAL_LIMBS)); values=np.array([MANUAL_LIMBS[t] for t in times])
    interpolator=PchipInterpolator(times,values,axis=0)
    result=track.copy(); conf=confidence.copy(); result[2:,5:]=interpolator(np.arange(2,len(track)))
    conf[2:,5:]=.65
    for frame in times:
        result[frame,5:]=MANUAL_LIMBS[frame]; conf[frame,5:]=1.
    return result,conf


def to_condition(points,camera):
    """Original source pixels -> exact TripoSR conditioning crop pixels."""
    shift=points.new_tensor([camera['pad_x']-camera['x0'],camera['pad_y']-camera['y0']])
    return (points+shift)*(camera['width']/camera['size'])


def project_points(points,camera):
    cam=look_at(camera['eye'],camera['target']); p=(points-camera['eye'])@cam.T
    focal=.5*camera['height']/np.tan(np.deg2rad(camera['fov'])*.5)
    return p[...,:2]/p[...,2:3].clamp_min(.01)*focal+points.new_tensor([camera['width']/2,camera['height']/2])


def lift_landmarks(position,camera,pixels):
    """Lift observed joints with explicit geometry-depth prior, not 3D truth."""
    pixels=to_condition(pixels,camera); projected=project_points(position,camera)
    cam=look_at(camera['eye'],camera['target']); depths=((position-camera['eye'])@cam.T)[:,2]
    _,nearest=torch.cdist(pixels,projected).topk(160,largest=False)
    zs=[]
    for j,ids in enumerate(nearest):
        # Image-facing limbs use nearer volume; far-side joints use farther one.
        q=.25 if j in [5,6,7,11,12,13] else .75 if j in [8,9,10,14,15,16] else .5
        zs.append(torch.quantile(depths[ids],q))
    z=torch.stack(zs)
    focal=.5*camera['height']/np.tan(np.deg2rad(camera['fov'])*.5)
    xy=(pixels-pixels.new_tensor([camera['width']/2,camera['height']/2]))/focal*z[:,None]
    joints=torch.cat((xy,z[:,None]),-1)@cam+camera['eye']
    # Nearest surface rays can put both rear legs on the same side. A shared
    # bilateral depth prior is explicit and keeps source projection unchanged.
    midpoint=(position[:,2].quantile(.1)+position[:,2].quantile(.9))*.5
    halfwidth=(position[:,2].quantile(.9)-position[:,2].quantile(.1))*.30
    rays=torch.cat(((pixels-pixels.new_tensor([camera['width']/2,camera['height']/2]))/focal,torch.ones(len(pixels),1)),1)@cam
    for ids,sign in [([5,6,7,11,12,13],1.),([8,9,10,14,15,16],-1.)]:
        desired_z=midpoint+sign*halfwidth
        depth=(desired_z-camera['eye'][2])/rays[ids,2]
        joints[ids]=camera['eye']+rays[ids]*depth[:,None]
    return joints


def bind_asset(position,joints,parents=PARENTS,all_weights=False):
    """Associate a bone parent->child with its PARENT rotation, not child.

    Spatial anatomical gating stops near/far limb cross-binding. These are still
    inferred soft skin weights, not a trained or anatomically certified rig.
    """
    children=torch.arange(1,len(joints),device=position.device)
    parent=torch.tensor(parents[1:],device=position.device)
    starts=joints[parent]; ends=joints[children]; vec=ends-starts
    t=((position[:,None]-starts)*vec).sum(-1)/vec.square().sum(-1).clamp_min(1e-8)
    closest=starts+t.clamp(0,1)[...,None]*vec
    distance=(position[:,None]-closest).norm(dim=-1)
    span=(position.amax(0)-position.amin(0)).max()
    # A central torso support avoids assigning belly to the closest dangling paw.
    for child in range(1,len(joints)):
        slot=child-1
        if child in [5,8,11,14]:
            # Root->hip/shoulder translates with root; never rotates whole torso by a limb.
            continue
        if child in [6,7,9,10,12,13,15,16]:
            hip=5 if child<8 else 8 if child<11 else 11 if child<14 else 14
            # Near/far depth partition smoothly penalizes wrong-side attachments.
            midpoint=(joints[5,2]+joints[8,2]+joints[11,2]+joints[14,2])/4
            near=hip in [5,11]
            near_greater=(joints[5,2]+joints[11,2])>(joints[8,2]+joints[14,2])
            side_sign=1. if (near==bool(near_greater)) else -1.
            distance[:,slot]+=torch.relu(position[:,1]-joints[hip,1]-span*.04)*4
            distance[:,slot]+=torch.relu(-(position[:,2]-midpoint)*side_sign)*2
        if child in [3,4]:
            distance[:,slot]+=torch.relu(position[:,0]-joints[3,0]-span*.07)*4
        if child==2:
            distance[:,slot]+=torch.relu(joints[1,0]-span*.06-position[:,0])*4
    if all_weights:
        bone_weights=torch.softmax(-distance/(span*.04),dim=-1)
        weight=position.new_zeros(len(position),len(joints))
        weight.scatter_add_(1,parent[None].expand(len(position),-1),bone_weights)
        return weight
    values,slots=distance.topk(3,largest=False)
    index=parent[slots]; weight=torch.softmax(-values/(span*.055),dim=-1)
    return index.to(torch.int16),weight


def mesh_weights(asset,joints,smoothing=15.):
    """Solve continuous mesh weights: (I + lambda L) W = W_distance.

    L is the symmetric uniform mesh graph Laplacian. Its M-matrix inverse
    preserves nonnegativity and constants. No hard per-point part boundary.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu
    vertices=asset['mesh_vertices']; faces=asset['mesh_faces'].long()
    prior=bind_asset(vertices,joints,all_weights=True).numpy()
    f=faces.numpy(); edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]),0)
    row=np.concatenate((edges[:,0],edges[:,1])); col=np.concatenate((edges[:,1],edges[:,0]))
    adjacency=sparse.coo_matrix((np.ones(len(row)),(row,col)),shape=(len(vertices),len(vertices))).tocsr()
    adjacency.data[:]=1.; degree=np.asarray(adjacency.sum(1)).ravel()
    laplacian=sparse.diags(degree)-adjacency
    matrix=sparse.eye(len(vertices),format='csc')+smoothing*laplacian.tocsc()
    weights=splu(matrix).solve(prior.astype(np.float64)).clip(0)
    weights/=weights.sum(1,keepdims=True)
    vw=torch.tensor(weights,dtype=torch.float32)
    attached=(vw[faces[asset['face_id'].long()]]*asset['barycentric'][:,:,None]).sum(1)
    index=torch.arange(len(joints),dtype=torch.int16)[None].expand(len(attached),-1).clone()
    return vw,index,attached


def anchored_mesh_weights(asset,rig,camera):
    """Projected distal-bone support seeds with a connected harmonic extension.

    These image-assisted anatomy seeds correct the inferred mesh's swapped rear
    depth order. They do NOT alter fitted joint vectors or extend short bones.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu
    vertices=asset['mesh_vertices']; faces=asset['mesh_faces'].long(); n=len(vertices)
    projected=project_points(vertices,camera); observed=to_condition(torch.tensor(LANDMARKS),camera)
    scores=torch.full((n,),float('inf')); label=torch.full((n,),-1,dtype=torch.long)
    for hip,knee,foot in [(5,6,7),(8,9,10),(11,12,13),(14,15,16)]:
        for start,end,owner in [(hip,knee,hip),(knee,foot,knee)]:
            a,b=observed[start],observed[end]; vector=b-a
            t=((projected-a)*vector).sum(1)/vector.square().sum()
            delta=(projected-(a+t.clamp(0,1)[:,None]*vector)).norm(dim=1)
            eligible=(t>.25)&(t<1.05)&(delta<7.)
            # Select a coherent volumetric limb rather than imposing the rig's
            # bilateral prior on every vertex: all connected distal surface seeds.
            candidate=delta/7.
            choose=eligible&(candidate<scores); scores[choose]=candidate[choose]; label[choose]=owner
        distance=(projected-observed[foot]).norm(dim=1)
        choose=(distance<11.)&((distance/11.)<scores)
        scores[choose]=distance[choose]/11.; label[choose]=knee
    # Stable torso/head anchors keep leg seeds from diffusing through the body.
    body=(projected[:,1]<observed[5,1]-15)&(projected[:,0]>observed[3,0]+10)&(projected[:,0]<observed[1,0]-10)
    label[body]=0
    head=(projected[:,0]>observed[1,0]+15)&(projected[:,1]<observed[1,1]+15)
    label[head]=1
    seeded=label>=0; seed_strength=seeded.numpy().astype(np.float64)*200.
    target=np.zeros((n,len(rig['joints'])),np.float64)
    target[np.flatnonzero(seeded.numpy()),label[seeded].numpy()]=1.
    f=faces.numpy(); edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]),0)
    row=np.concatenate((edges[:,0],edges[:,1])); col=np.concatenate((edges[:,1],edges[:,0]))
    adjacency=sparse.coo_matrix((np.ones(len(row)),(row,col)),shape=(n,n)).tocsr(); adjacency.data[:]=1.
    degree=np.asarray(adjacency.sum(1)).ravel(); laplacian=sparse.diags(degree)-adjacency
    # Weak original prior gives unseeded tail/hidden supports a unique solution.
    matrix=(laplacian+sparse.diags(seed_strength+.03)).tocsc()
    rhs=target*seed_strength[:,None]+.03*rig['vertex_weights'].numpy()
    result=splu(matrix).solve(rhs).clip(0); result/=result.sum(1,keepdims=True)
    weights=torch.tensor(result,dtype=torch.float32)
    attachment=(weights[faces[asset['face_id'].long()]]*asset['barycentric'][:,:,None]).sum(1)
    return weights,attachment,label


@torch.no_grad()
def rebind(source_root,asset_path=ASSET):
    from .surface_skin3d import SurfaceSkinner
    torch.set_num_threads(4); OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'rig.pt').exists(): raise FileExistsError('Preserve rebound experiment')
    asset=load_verified(asset_path); original=load_verified(source_root/'rig.pt'); motion=load_verified(source_root/'motion.pt')
    rig=dict(original); vw,weight,labels=anchored_mesh_weights(asset,rig,motion['camera'])
    rig.update(vertex_weights=vw,weight=weight,source='Projected anatomical support seeds + anchored harmonic mesh weights; identical surface_refined trajectory.')
    save_inference_checkpoint(rig,OUT/'rig.pt'); save_inference_checkpoint(motion,OUT/'motion.pt')
    np.save(OUT/'mesh_seed_labels.npy',labels.numpy())
    old=SurfaceSkinner(asset,original); new=SurfaceSkinner(asset,rig)
    rest=project_points(asset['mesh_vertices'],motion['camera']); targets=to_condition(torch.tensor(LANDMARKS),motion['camera'])
    feet=[7,10,13,16]; picked=[int((rest-targets[j]).norm(dim=1).argmin()) for j in feet]
    report=dict(source_root=str(source_root),trajectory_same=bool(torch.equal(load_verified(OUT/'motion.pt')['local_rotation'],motion['local_rotation'])),
                seed_counts={NAMES[k]:int((labels==k).sum()) for k in range(len(NAMES))},endpoint_diagnostics=[])
    canvas=Image.new('RGB',(1536,1080),'#151922'); draw=ImageDraw.Draw(canvas)
    for frame in [2,10,18,26,32]:
        _,_,_,ov,_=old(motion['local_rotation'][frame],motion['root_translation'][frame])
        _,_,_,nv,_=new(motion['local_rotation'][frame],motion['root_translation'][frame])
        old_p=project_points(ov[picked],motion['camera']); new_p=project_points(nv[picked],motion['camera'])
        joint_p=project_points(motion['joint_positions'][frame,feet],motion['camera']); actual=motion['landmarks_condition'][frame,feet]
        factor=motion['camera']['size']/motion['camera']['width']
        report['endpoint_diagnostics'].append(dict(frame=frame,joints=[NAMES[j] for j in feet],
              joint_target_error_source_px=((joint_p-actual).norm(dim=1)*factor).tolist(),
              old_vertex_joint_error_source_px=((old_p-joint_p).norm(dim=1)*factor).tolist(),
              new_vertex_joint_error_source_px=((new_p-joint_p).norm(dim=1)*factor).tolist(),
              old_vertex_target_error_source_px=((old_p-actual).norm(dim=1)*factor).tolist(),
              new_vertex_target_error_source_px=((new_p-actual).norm(dim=1)*factor).tolist()))
    report['limits']='Same fitted vectors. This tests binding only; per-joint endpoint errors expose failures hidden by mean reprojection. Single-view support seeds are manual assistance, not anatomical ground truth.'
    (OUT/'binding_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


@torch.no_grad()
def binding_audit(source_root,asset_path=ASSET):
    from .surface_skin3d import SurfaceSkinner
    torch.set_num_threads(4)
    asset=load_verified(asset_path); old_rig=load_verified(source_root/'rig.pt'); new_rig=load_verified(OUT/'rig.pt'); motion=load_verified(OUT/'motion.pt')
    old=SurfaceSkinner(asset,old_rig); new=SurfaceSkinner(asset,new_rig)
    feet=[7,10,13,16]; rest=project_points(asset['mesh_vertices'],motion['camera']); target=to_condition(torch.tensor(LANDMARKS),motion['camera'])
    picked=[int((rest-target[j]).norm(dim=1).argmin()) for j in feet]
    factor=motion['camera']['size']/motion['camera']['width']; measures={k:[] for k in ['joint_target','old_vertex_target','new_vertex_target','old_vertex_joint','new_vertex_joint']}
    for frame in range(len(motion['local_rotation'])):
        _,_,_,ov,_=old(motion['local_rotation'][frame],motion['root_translation'][frame]); _,_,_,nv,_=new(motion['local_rotation'][frame],motion['root_translation'][frame])
        po=project_points(ov[picked],motion['camera']); pn=project_points(nv[picked],motion['camera']); pj=project_points(motion['joint_positions'][frame,feet],motion['camera']); observed=motion['landmarks_condition'][frame,feet]
        for name,delta in [('joint_target',pj-observed),('old_vertex_target',po-observed),('new_vertex_target',pn-observed),('old_vertex_joint',po-pj),('new_vertex_joint',pn-pj)]:
            measures[name].append(delta.norm(dim=1)*factor)
    report=dict(units='Original832x480sourcepixels',frames=33,feet=[NAMES[j] for j in feet],metrics={},
                candidate_accepted=False,reason='Targeted rear support follows the fitted endpoint better but creates visibly stretched rear-leg sheets. Keep surface_refined as the less-distorted diagnostic. Natural gait still rejected.')
    for name,values in measures.items():
        t=torch.stack(values); report['metrics'][name]=dict(median=t.quantile(.5,dim=0).tolist(),maximum=t.amax(0).tolist())
    (OUT/'binding_all_frames.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


def batched_fk(joints,parents,local_rotation,translation):
    """Batched differentiable fixed-bone FK with the public module's convention."""
    rotations=[]; points=[]
    for j,parent in enumerate(parents):
        if parent<0:
            rotations.append(local_rotation[:,j]); points.append(joints[j]+translation)
        else:
            rotations.append(rotations[parent]@local_rotation[:,j])
            points.append(points[parent]+(rotations[parent]@(joints[j]-joints[parent])[:,None]).squeeze(-1))
    return torch.stack(points,1),torch.stack(rotations,1)


def rotation_vectors(v):
    zero=torch.zeros_like(v[...,0]); x,y,z=v.unbind(-1)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*v.shape[:-1],3,3)
    return torch.matrix_exp(skew)


def fit_trajectory(joints,target,confidence,camera,steps=650,rest=2):
    angles=torch.zeros(len(target),len(joints),3,requires_grad=True)
    shift=torch.zeros(len(target),3,requires_grad=True)
    optimizer=torch.optim.Adam([{'params':[angles],'lr':.025},{'params':[shift],'lr':.002}])
    mask=torch.ones(len(target),1,1); mask[rest]=0
    # Monocular reprojection alone lets a free 3-axis model invent severe twist.
    # Use bounded camera-facing hinges for this constrained reconstruction.
    limits=torch.full((len(joints),1),1.6); limits[0]=.12; limits[1]=.18; limits[3]=.5
    hinge_axis=look_at(camera['eye'],camera['target'])[2]
    def parameters():
        rv=torch.tanh(angles[...,2:3])*limits*hinge_axis*mask
        translation=shift*mask[:,0]*shift.new_tensor([1.,1.,0.])
        return rv,translation
    hist=[]; scale=float(camera['width'])
    for step in range(steps):
        rv,translation=parameters(); local=rotation_vectors(rv)
        xyz,_=batched_fk(joints,PARENTS,local,translation)
        prediction=project_points(xyz,camera)
        residual=(prediction-target)/scale
        data=(torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.012,reduction='none').sum(-1)*confidence).sum()/confidence.sum().clamp_min(1)
        smooth=(angles[2:]-2*angles[1:-1]+angles[:-2]).square().mean()
        root_smooth=(shift[2:]-2*shift[1:-1]+shift[:-2]).square().mean()
        # Strong out-of-plane prior because one view cannot identify arbitrary 3D twist.
        regular=.00025*angles.square().mean()+.001*angles[...,:2].square().mean()
        ground=torch.relu(-xyz[:,[7,10,13,16],1]-.015).square().mean()
        loss=data+.04*smooth+.2*root_smooth+regular+.1*ground
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step%100==0 or step==steps-1:
            hist.append(dict(step=step,loss=float(loss.detach()),data=float(data.detach())))
    with torch.no_grad():
        rv,translation=parameters(); local=rotation_vectors(rv)
        xyz,_=batched_fk(joints,PARENTS,local,translation)
    return dict(local_rotation=local,root_translation=translation,rotation_vectors=rv,joint_positions=xyz,history=hist)


def fit(camera_path,steps=650,manual=False):
    if (OUT/'motion.pt').exists(): raise FileExistsError('Preserve existing motion fit')
    OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); start=time.perf_counter()
    asset=load_verified(ASSET); camera=load_verified(camera_path)
    frames,fps=read_frames(); track,confidence=track_landmarks(frames)
    if manual: track,confidence=apply_manual_limb_anchors(track,confidence)
    target=to_condition(torch.from_numpy(track),camera); conf=torch.from_numpy(confidence)
    joints=lift_landmarks(asset['position'],camera,torch.from_numpy(LANDMARKS))
    vertex_weights,index,weight=mesh_weights(asset,joints)
    result=fit_trajectory(joints,target,conf,camera,steps)
    rig=dict(joints=joints,parents=PARENTS,names=NAMES,index=index,weight=weight,ids=asset['ids'],vertex_weights=vertex_weights,
             source='Manual frame-2 anatomical landmark lift with inferred mesh depth; incoming segments bound to parent transforms.')
    save_inference_checkpoint(rig,OUT/'rig.pt')
    result.update(camera=camera,source_frame=torch.arange(len(frames)),fps=fps,rest_frame=2,source_video_sha256=digest(SOURCE),
                  source='Video-conditioned motion reconstruction from original Wan clip. Not new learned motion generation.',
                  landmarks_source=torch.from_numpy(track),landmarks_condition=target,landmark_confidence=conf,
                  manual_rest_landmarks=torch.from_numpy(LANDMARKS),asset_sha256=digest(ASSET))
    result['manual_limb_anchors']=MANUAL_LIMBS if manual else {}
    save_inference_checkpoint(result,OUT/'motion.pt')
    prediction=project_points(result['joint_positions'],camera)
    baseline=project_points(joints,camera)[None].expand_as(prediction)
    metric=lambda x:float(((x-target).norm(dim=-1)*conf).sum()/conf.sum())
    lengths=torch.tensor([(joints[j]-joints[p]).norm() for j,p in enumerate(PARENTS) if p>=0])
    moved_lengths=torch.stack([(result['joint_positions'][:,j]-result['joint_positions'][:,p]).norm(dim=-1) for j,p in enumerate(PARENTS) if p>=0],-1)
    report=dict(frames=len(frames),fps=fps,duration_seconds=len(frames)/fps,fit_seconds=time.perf_counter()-start,
                static_weighted_reprojection_px=metric(baseline),fitted_weighted_reprojection_px=metric(prediction),
                accepted_observation_fraction=float((conf>.1).float().mean()),max_bone_length_error=float((moved_lengths-lengths).abs().max()),
                skinning='Continuous shared-mesh weights: (I+15L)W=distance-prior; barycentric material attachment',
                motion_scalar_count=result['rotation_vectors'].numel()+result['root_translation'].numel(),
                manually_assisted=manual,manual_limb_keyframes=sorted(MANUAL_LIMBS) if manual else [],
                source_video_sha256=digest(SOURCE),asset_sha256=digest(ASSET),
                limits='Reconstruction, no unseen-motion generation or efficiency proof. Optical flow is not anatomical ground truth. Hidden joints and depth inferred. Fixed rest geometry may have wrong proportions.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    canvas=Image.new('RGB',(832*2,520*4),'#151922'); draw=ImageDraw.Draw(canvas)
    source_prediction=prediction.numpy()*camera['size']/camera['width']-np.array([camera['pad_x']-camera['x0'],camera['pad_y']-camera['y0']])
    for slot,i in enumerate([2,6,10,14,18,22,26,32]):
        pic=Image.fromarray(frames[i]); d=ImageDraw.Draw(pic)
        for j,parent in enumerate(PARENTS):
            if parent>=0: d.line([tuple(source_prediction[i,j]),tuple(source_prediction[i,parent])],fill='cyan',width=2)
            u,v=track[i,j]; d.ellipse((u-3,v-3,u+3,v+3),fill='yellow' if confidence[i,j]>.1 else 'red')
        x,y=slot%2*832,slot//2*520; canvas.paste(pic,(x,y+30)); draw.text((x+8,y+6),f'frame {i}: fitted 3D skeleton cyan | tracked target yellow | rejected red',fill='white')
    canvas.save(OUT/'fitted_skeleton.jpg'); print(json.dumps(report),flush=True)


@torch.no_grad()
def preview(asset_path=ASSET):
    torch.set_num_threads(4)
    asset=load_verified(asset_path); rig=load_verified(OUT/'rig.pt'); motion=load_verified(OUT/'motion.pt'); camera=motion['camera']
    if not torch.equal(asset['ids'],rig['ids']): raise ValueError('Asset IDs differ from skinning packet')
    cov=(asset['frame']*asset['scale'][:,None].square())@asset['frame'].transpose(-1,-2)
    from .surface_skin3d import SurfaceSkinner
    support=SurfaceSkinner(asset,rig)
    canvas=Image.new('RGB',(1536,1080),'#151922'); d=ImageDraw.Draw(canvas)
    for slot,i in enumerate([2,6,10,18,26,32]):
        xyz,co,_,_,_=support(motion['local_rotation'][i],motion['root_translation'][i])
        rgb,_=render(xyz,co,asset['colour'],asset['opacity'],camera['eye'],camera['target'],512,512,camera['fov'],ground=False)
        pic=Image.fromarray(np.uint8(rgb.numpy()*255)); pic.save(OUT/f'render_{i:02d}.png')
        x,y=slot%3*512,slot//3*540; canvas.paste(pic,(x,y+28)); d.text((x+8,y+8),f'Fitted XYZ Gaussian motion | source frame {i}',fill='white')
    canvas.save(OUT/'motion_preview.jpg')


@torch.no_grad()
def audit(asset_path=ASSET):
    from scipy.spatial import cKDTree
    from .surface_skin3d import SurfaceSkinner
    torch.set_num_threads(4)
    asset=load_verified(asset_path); rig=load_verified(OUT/'rig.pt'); motion=load_verified(OUT/'motion.pt')
    support=SurfaceSkinner(asset,rig); p=asset['position']
    distance,indices=cKDTree(p.numpy()).query(p.numpy(),k=2)
    near=distance[:,1]<.005; ids=torch.from_numpy(indices[:,1]); worst=0.; separations=0; min_eigen=1.; min_area=100.; max_area=0.
    for frame in range(len(motion['local_rotation'])):
        xyz,cov,_,_,area=support(motion['local_rotation'][frame],motion['root_translation'][frame])
        gap=(xyz-xyz[ids]).norm(dim=-1)[near]
        worst=max(worst,float(gap.max())); separations=max(separations,int((gap>.05).sum()))
        min_eigen=min(min_eigen,float(torch.linalg.eigvalsh(cov).amin()))
        min_area=min(min_area,float(area.min())); max_area=max(max_area,float(area.max()))
    rest,_,_,_,_=support(motion['local_rotation'][2],motion['root_translation'][2])
    report=dict(frames=len(motion['local_rotation']),same_ids=bool(torch.equal(asset['ids'],rig['ids'])),
                rest_position_max_error=float((rest-p).abs().max()),rest_rotation_max_error=float((motion['local_rotation'][2]-torch.eye(3)).abs().max()),
                vertex_weight_row_error=float((rig['vertex_weights'].sum(1)-1).abs().max()),vertex_weight_min=float(rig['vertex_weights'].min()),
                minimum_covariance_eigenvalue=min_eigen,minimum_face_area_ratio=min_area,maximum_face_area_ratio=max_area,
                max_nearest_pair_gap=worst,max_near_pairs_separating_over_005=separations,
                near_pair_threshold=.005,gap_threshold=.05,unused_degenerate_faces=support.unused_degenerate_faces,
                motion_packet_sha256=digest(OUT/'motion.pt'),rig_packet_sha256=digest(OUT/'rig.pt'),asset_sha256=digest(asset_path),
                math_sources=['https://arxiv.org/html/2504.01204v1','https://arxiv.org/html/2602.04271v1'],
                interpretation='Numerical geometry checks are NOT motion accuracy or learned generation. Shared mesh prevents independent attachment seams but does not prove collision-free or correct anatomy.')
    (OUT/'connected_audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['inspect','fit','preview','audit','rebind','binding_audit']); p.add_argument('--camera',type=Path,default=Path('artifacts/real_video/true3d/v2_appearance/camera.pt')); p.add_argument('--steps',type=int,default=650)
    p.add_argument('--manual-anchors',action='store_true'); p.add_argument('--output',type=Path)
    p.add_argument('--asset',type=Path,default=ASSET)
    p.add_argument('--source-root',type=Path,default=Path('artifacts/real_video/true3d/v2_motion/surface_refined'))
    args=p.parse_args()
    if args.output is not None: OUT=args.output
    with keep_windows_awake():
        if args.mode=='inspect': contact_sheet()
        elif args.mode=='fit': fit(args.camera,args.steps,args.manual_anchors)
        elif args.mode=='preview': preview(args.asset)
        elif args.mode=='rebind': rebind(args.source_root,args.asset)
        elif args.mode=='binding_audit': binding_audit(args.source_root,args.asset)
        else: audit(args.asset)
