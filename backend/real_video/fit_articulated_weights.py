"""Source-conditioned neural joint/skin calibration, NOT a video generator.

The network maps clip time to 3D local joint rotations and root translation.
It is supervised by this observed clip, including future poses/images, so its
weights are an asset-specific motion representation, NOT causal prediction.
Appearance and Gaussian IDs remain immutable. Every selected/rejected run stays.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw
from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from .fit_motion3d import read_frames, project_points, rotation_vectors, PARENTS, NAMES
from .attach_vector_weights import ASSET, CAMERA, original_intrinsics
from .gaussian3d import render, look_at
from .gaussian_motion_memory import select_cat

ROOT = Path('artifacts/real_video/true3d/articulated_weight_loop/v1')
RIG = Path('artifacts/real_video/true3d/v2_motion/surface_refined/rig.pt')
OBS = Path('artifacts/real_video/true3d/v2_motion/surface_refined/motion.pt')
SOURCE = Path('artifacts/real_video/wan_baseline/cat_seed_421001/wan_original.mp4')
TRAIN_IMAGES = [2, 6, 10, 18, 26, 32]
HELD_IMAGES = [14, 22]


class ClipJointWeights(nn.Module):
    """Time-conditioned pose fitting weights, no claim of novel generation."""
    def __init__(self, joint_count=17, hidden=96, constrained=False):
        super().__init__()
        self.joint_count = joint_count
        self.net = nn.Sequential(nn.Linear(10, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, joint_count*3+3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        limits = torch.full((joint_count, 3), 2.2)
        limits[0] = .25
        limits[1:3] = .35
        limits[3:5] = .8
        if constrained:
            limits[0] = .10
            limits[1:3] = .18
            limits[3:5] = .5
            # World-axis bounds are conservative priors, not recovered anatomy.
            limits[5:] = torch.tensor([.35, .35, 1.8])
        self.register_buffer('limits', limits)

    @staticmethod
    def encoding(t):
        f = t[:, None] * t.new_tensor([1., 2., 3., 4.])[None] * torch.pi
        return torch.cat((t[:, None], t[:, None].square(), f.sin(), f.cos()), -1)

    def forward(self, t):
        raw = self.net(self.encoding(t)) - self.net(self.encoding(t.new_zeros(1)))
        rv = raw[:, :self.joint_count*3].reshape(-1, self.joint_count, 3).tanh()*self.limits
        shift = raw[:, -3:].tanh()*raw.new_tensor([.4, .2, .2])
        return rotation_vectors(rv), shift, rv


def to_device(packet, device):
    return {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in packet.items()}


def bone_lengths(joints):
    return torch.stack([(joints[j]-joints[p]).norm(dim=-1) for j,p in enumerate(PARENTS) if p >= 0], -1)


def symmetry_loss(joints):
    lengths = bone_lengths(joints)
    near = torch.tensor([5, 6, 11, 12], device=joints.device)
    far = torch.tensor([8, 9, 14, 15], device=joints.device)
    return ((lengths[near]-lengths[far])/((lengths[near]+lengths[far])*.5).detach().clamp_min(.01)).square().mean()


def reduced_intrinsics(camera, factor=.25):
    intrinsics = original_intrinsics(camera)
    return dict(height=round(intrinsics['height']*factor), width=round(intrinsics['width']*factor),
                fov=intrinsics['fov'], principal=[x*factor for x in intrinsics['principal']])


def fit(root=ROOT, pose_steps=600, image_steps=180, train_skin=True, seed=530101, constrained=False):
    from .articulated_gaussian import BatchedArticulatedGaussian
    if (root/'model.pt').exists() or (root/'protocol.json').exists():
        raise FileExistsError('Preserve previous trial; choose a fresh output directory')
    root.mkdir(parents=True, exist_ok=True)
    input_hashes={str(path):digest(path) for path in [ASSET,CAMERA,RIG,OBS,SOURCE]}
    protocol = dict(classification='Source-conditioned 3D joint and skinning fitting, NOT generation',
        training_images=TRAIN_IMAGES, held_out_images=HELD_IMAGES,
        manual_pose_observation_frames=[2,10,18,26,32], pose_steps=pose_steps, image_steps=image_steps,
        train_skinning=train_skin, seed=seed, constrained=constrained, input_hashes=input_hashes,
        asset_sha256=digest(ASSET), source_sha256=digest(SOURCE),
        intermediate_pose_targets='PCHIP interpolation of manual source anchors; not additional ground truth' if constrained else 'Manual keyframes only',
        intended_acceptance='Numerical metrics plus direct source/oblique visual review; any articulation fail rejects.',
        heldout_limit='Same-clip interpolation between observed keyframes, not independent video or future-motion generalization.')
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    torch.manual_seed(seed); torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    asset = to_device(load_verified(ASSET),device); camera = to_device(load_verified(CAMERA),device)
    rig = to_device(load_verified(RIG),device); observed = to_device(load_verified(OBS),device)
    decoder = BatchedArticulatedGaussian(asset,rig,train_skinning=train_skin,max_influences=6).to(device)
    model = ClipJointWeights(constrained=constrained).to(device)
    initial_model = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_weights = decoder.weights().detach().clone()
    initial_logits = decoder.skinning_logits.detach().clone()
    # Depth calibration stays on original camera rays: rest-image joints do not
    # slide in XY. Depth remains inferred, with explicit bilateral length prior.
    joints0 = rig['joints'].clone()
    ray = (joints0-camera['eye'])
    ray = ray/ray.norm(dim=-1,keepdim=True)
    depth_raw = nn.Parameter(torch.zeros(len(joints0),device=device))
    with torch.no_grad():
        depth_raw[[9,10,15,16]] = torch.tensor([.10,.24,.10,.24],device=device)
    depth_mask = torch.ones(len(joints0),1,device=device); depth_mask[:5] = 0
    def rest_joints():
        return joints0 + .25*depth_raw.tanh()[:,None]*ray*depth_mask
    pose_ids = torch.arange(2,33,device=device) if constrained else torch.tensor([2,10,18,26,32],device=device)
    times = (pose_ids.float()-2)/30
    target = observed['landmarks_condition'][pose_ids]
    confidence = observed['landmark_confidence'][pose_ids].clone()
    confidence[:, [7,10,13,16]] *= 2
    optim = torch.optim.Adam([{'params':model.parameters(),'lr':.002}, {'params':[depth_raw],'lr':.004}])
    start = time.perf_counter(); history=[]
    initial_lengths = bone_lengths(joints0).detach()
    bilateral_target = initial_lengths.clone()
    for a,b in [(5,8),(6,9),(11,14),(12,15)]:
        bilateral_target[a] = bilateral_target[b] = torch.maximum(initial_lengths[a],initial_lengths[b])
    def pose_loss():
        local, tr, rv = model(times)
        out = decoder.vertices(local,tr,joints=rest_joints())
        uv = project_points(out['joints'],camera)
        delta = (uv-target)/float(camera['width'])
        data = (torch.nn.functional.smooth_l1_loss(delta,torch.zeros_like(delta),beta=.02,reduction='none').sum(-1)*confidence).sum()/confidence.sum()
        _, all_shift, all_rv = model(torch.linspace(0,1,31,device=device))
        smooth = (all_rv[2:]-2*all_rv[1:-1]+all_rv[:-2]).square().mean()
        regular = .0003*rv.square().mean()+.004*symmetry_loss(rest_joints())+.0003*depth_raw.tanh().square().mean()
        if constrained:
            regular = regular + .004*((bone_lengths(rest_joints())-bilateral_target)/bilateral_target.clamp_min(.01)).square().mean()
        return data + .02*smooth + regular, data
    for step in range(pose_steps):
        loss,data=pose_loss(); optim.zero_grad(); loss.backward(); optim.step()
        if step % 100 == 0 or step+1 == pose_steps:
            row=dict(stage='pose',step=step,loss=float(loss.detach()),pose=float(data.detach()),symmetry=float(symmetry_loss(rest_joints()).detach()))
            history.append(row); print(json.dumps(row),flush=True)
    # Store a pose-only candidate before image fitting; visual review can reject
    # either/both without erasing the failed outcome.
    def snapshot():
        return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    weights=decoder.weights().detach().cpu().clone(),joints=rest_joints().detach().cpu().clone())
    candidates = {'pose_only':snapshot()}
    frames,fps = read_frames()
    intrinsics = reduced_intrinsics(camera)
    h,w=intrinsics['height'],intrinsics['width']
    targets={}
    for frame in TRAIN_IMAGES:
        mask=select_cat(frames[frame]); small=cv2.resize(mask,(w,h),interpolation=cv2.INTER_AREA)
        image=cv2.resize(frames[frame].astype(np.float32)/255,(w,h),interpolation=cv2.INTER_AREA)
        image=image*small[...,None]+.5*(1-small[...,None])
        targets[frame]=(torch.tensor(image,device=device),torch.tensor(small,device=device))
    edges=torch.cat([asset['mesh_faces'][:,[0,1]],asset['mesh_faces'][:,[1,2]],asset['mesh_faces'][:,[2,0]]]).long()
    edges=torch.unique(edges.sort(dim=-1).values,dim=0)
    rest_edges=(asset['mesh_vertices'][edges[:,0]]-asset['mesh_vertices'][edges[:,1]]).norm(dim=-1)
    valid_edge=rest_edges>1e-5; edges=edges[valid_edge]; rest_edges=rest_edges[valid_edge]
    parameters=[{'params':model.parameters(),'lr':.0006},{'params':[depth_raw],'lr':.001}]
    if train_skin: parameters.append({'params':decoder.parameters(),'lr':.008})
    optim=torch.optim.Adam(parameters)
    gradient_evidence={}
    for step in range(image_steps):
        frame=TRAIN_IMAGES[step % len(TRAIN_IMAGES)]
        local,tr,rv=model(torch.tensor([(frame-2)/30],device=device))
        out=decoder(local,tr,joints=rest_joints(),return_strain=constrained)
        rgb,alpha=render(out['position'][0],out['covariance'][0],asset['colour'],asset['opacity'],
            camera['eye'],camera['target'],**intrinsics,radius=3,ground=False)
        rgb=rgb+(1-alpha[...,None])*(.5-rgb.new_tensor([.15,.19,.24]))
        target_rgb,target_mask=targets[frame]
        silhouette=(alpha-target_mask).square().mean()
        photometric=(rgb-target_rgb).abs().mean()
        length=(out['vertices'][0,edges[:,0]]-out['vertices'][0,edges[:,1]]).norm(dim=-1)/rest_edges
        strain=(torch.relu(length-1.4).square()+torch.relu(.65-length).square()).mean()
        weights=decoder.weights()
        weight_prior=(weights-initial_weights).square().mean()
        weight_smooth=(weights[edges[:,0]]-weights[edges[:,1]]).square().mean()
        positional,_=pose_loss()
        loss=4*silhouette+photometric+3*positional+.1*strain+.02*weight_prior+.002*weight_smooth
        if constrained:
            # Edge lengths alone miss shear in thin triangles. Penalize full
            # material deformation, without silently clipping rendered states.
            singular=out['singular_values'].clamp_min(1e-5)
            material=torch.log(singular).square().mean()
            loss=4*silhouette+photometric+5*positional+.2*strain+.5*material+.2*weight_prior+.08*weight_smooth
        optim.zero_grad(); loss.backward()
        if step==1:
            gradient_evidence=dict(network_head_grad_norm=float(model.net[-1].weight.grad.norm()),
                skin_grad_norm=float(sum(p.grad.square().sum() for p in decoder.parameters() if p.grad is not None).sqrt()) if train_skin else 0.,
                rest_depth_grad_norm=float(depth_raw.grad.norm()),
                note='Combined rendered, pose and regularization gradient; separate rendered-only regression test required for complete path proof.')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if constrained and train_skin:
            with torch.no_grad():
                decoder.skinning_logits.copy_(initial_logits+(decoder.skinning_logits-initial_logits).clamp(-.5,.5))
        if step%20==0 or step+1==image_steps:
            row=dict(stage='rendered',step=step,source_frame=frame,loss=float(loss.detach()),silhouette=float(silhouette.detach()),rgb=float(photometric.detach()),strain=float(strain.detach()))
            history.append(row); print(json.dumps(row),flush=True)
    candidates['image_refined']=snapshot()
    packet=dict(architecture='clip_joint_weights_v1',model_config={'constrained':constrained},classification=protocol['classification'],trained=True,
        initial_model=initial_model,candidates=candidates,source_frame=torch.arange(2,33),fps=fps,
        provenance=protocol,gradient_evidence=gradient_evidence,training_seconds=time.perf_counter()-start,
        model_parameter_count=sum(p.numel() for p in model.parameters()),
        learned_skinning_parameter_count=sum(p.numel() for p in decoder.parameters()),
        appearance_unchanged=True,gaussian_ids_unchanged=True,wan_weights_changed=False,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    save_inference_checkpoint(packet,root/'model.pt')
    (root/'history.json').write_text(json.dumps(history,indent=2))
    print(json.dumps(dict(saved=str(root/'model.pt'),seconds=packet['training_seconds'])),flush=True)


@torch.no_grad()
def evaluate(root=ROOT):
    from .articulated_gaussian import BatchedArticulatedGaussian
    from .evaluate_attached_vectors import write_video
    if any((root/name).exists() for name in ['evaluation.json','motion.pt','joint_skinning_native.mp4']):
        raise FileExistsError('Preserve completed evaluation artifacts')
    device='cuda' if torch.cuda.is_available() else 'cpu'; torch.set_num_threads(4)
    packet=load_verified(root/'model.pt')
    input_hashes=packet['provenance'].get('input_hashes')
    if input_hashes is None:
        # Early trial recovery requires an explicitly labeled external manifest;
        # it must not retroactively pretend hashes were recorded at train start.
        manifest_path=root/'input_manifest.json'
        if not manifest_path.is_file(): raise ValueError('Input provenance missing; no unverified evaluation')
        input_hashes=json.loads(manifest_path.read_text())['input_hashes']
    expected={str(path) for path in [ASSET,CAMERA,RIG,OBS,SOURCE]}
    if set(input_hashes)!=expected or any(digest(Path(path))!=sha for path,sha in input_hashes.items()):
        raise ValueError('Training/evaluation input hash mismatch')
    asset=to_device(load_verified(ASSET),device)
    camera=to_device(load_verified(CAMERA),device); rig=to_device(load_verified(RIG),device)
    frames,fps=read_frames(); reference=load_verified(OBS)
    masks={f:select_cat(frames[f]) for f in range(2,33)}
    intrinsics=original_intrinsics(camera); bg=torch.tensor([.15,.19,.24],device=device)
    rendered={}; metrics={}; trajectories={}
    for name,candidate in packet['candidates'].items():
        model=ClipJointWeights(**packet.get('model_config',{})).to(device); model.load_state_dict(candidate['model'],strict=True)
        local_rig=dict(rig,joints=candidate['joints'].to(device),vertex_weights=candidate['weights'].to(device))
        decoder=BatchedArticulatedGaussian(asset,local_rig,train_skinning=False,max_influences=17,min_support=0).to(device)
        trajectory=[]; views=[]; sideviews=[]; rows=[]
        for frame in range(2,33):
            local,tr,_=model(torch.tensor([(frame-2)/30],device=device))
            out=decoder(local,tr)
            rgb,alpha=render(out['position'][0],out['covariance'][0],asset['colour'],asset['opacity'],camera['eye'],camera['target'],**intrinsics,ground=False)
            rgb=rgb+(1-alpha[...,None])*(.5-bg)
            array=rgb.clamp(0,1).cpu().numpy(); views.append((array*255).round().astype(np.uint8))
            mask=masks[frame]; predicted=alpha.cpu().numpy()>.5
            target=frames[frame]/255*mask[...,None]+.5*(1-mask[...,None])
            joint_pixels=project_points(out['joints'][0],camera).cpu()
            joint_target=reference['landmarks_condition'][frame]
            feet=(joint_pixels[[7,10,13,16]]-joint_target[[7,10,13,16]]).norm(dim=-1)*camera['size']/camera['width']
            rows.append(dict(frame=frame,roi_mae=float(np.abs(array[80:480,:670]-target[80:480,:670]).mean()),
                iou=float((predicted&(mask>.5)).sum()/max(1,(predicted|(mask>.5)).sum())),
                foot_reprojection_px=feet.tolist()))
            trajectory.append(out['vertices'][0].cpu())
            if frame in [2,10,14,22,32]:
                eye=camera['eye']; center=camera['target']; offset=eye-center
                angle=torch.tensor(.45,device=device); c,s=angle.cos(),angle.sin()
                rotate=eye.new_tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
                rotate[0,0]=c;rotate[0,2]=s;rotate[2,0]=-s;rotate[2,2]=c
                oblique,_=render(out['position'][0],out['covariance'][0],asset['colour'],asset['opacity'],center+rotate@offset,center,512,512,camera['fov'],ground=False)
                pic=Image.fromarray((oblique.clamp(0,1).cpu().numpy()*255).astype(np.uint8))
                pic.save(root/f'{name}_oblique_{frame:02d}.png')
            if frame%10==0:print(json.dumps(dict(stage='render',candidate=name,frame=frame)),flush=True)
        rendered[name]=views;trajectories[name]=torch.stack(trajectory)
        allframes=[r for r in rows if r['frame']>2]
        held=[r for r in rows if r['frame'] in HELD_IMAGES]
        metrics[name]=dict(per_frame=rows,mean_roi_mae=float(np.mean([r['roi_mae'] for r in allframes])),
            heldout_roi_mae=float(np.mean([r['roi_mae'] for r in held])),heldout_iou=float(np.mean([r['iou'] for r in held])),
            bone_lengths=bone_lengths(candidate['joints']).tolist(),symmetry_error=float(symmetry_loss(candidate['joints'])))
    output=[]
    for index,frame in enumerate(range(2,33)):
        canvas=Image.new('RGB',(2496,544),'#151922'); draw=ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(frames[frame]),(0,28))
        for column,name in enumerate(['pose_only','image_refined'],1):
            canvas.paste(Image.fromarray(rendered[name][index]),(column*832,28))
        for x,title in [(8,'Observed source'),(840,'Learned 3D joint pose | source-conditioned fit'),(1672,'3D pose + learned skinning | rendered-loss fit')]: draw.text((x,7),title,fill='white')
        draw.text((8,516),f'Source frame {frame}/32 | persistent 45,000 XYZ Gaussians | NOT new-motion generation | visual acceptance pending',fill='white')
        output.append(np.asarray(canvas))
        if frame in [2,10,14,22,32]:canvas.save(root/f'comparison_{frame:02d}.jpg')
    write_video(root/'joint_skinning_native.mp4',output,fps)
    save_inference_checkpoint(dict(vertices=trajectories,camera=to_device(camera,'cpu'),source_frame=torch.arange(2,33),fps=fps,
        asset_path=str(ASSET),asset_sha256=digest(ASSET),model_sha256=digest(root/'model.pt'),classification=packet['classification']),root/'motion.pt')
    report=dict(metrics=metrics,video_sha256=digest(root/'joint_skinning_native.mp4'),frames=31,fps=fps,
        classification=packet['classification'],accepted=False,reason='Numerics cannot accept this run; requires hashed visual review.',
        pose_metric_limit='Comparison to manually inferred anatomical anchors and interpolated intermediate observations, not measured 3D pose truth.',
        training_seconds=packet['training_seconds'],peak_cuda_allocated_bytes=packet['peak_cuda_allocated_bytes'])
    (root/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='metrics'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['fit','evaluate'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--pose-steps',type=int,default=600)
    p.add_argument('--image-steps',type=int,default=180);p.add_argument('--freeze-skin',action='store_true')
    p.add_argument('--constrained',action='store_true');args=p.parse_args()
    if args.pose_steps<1 or args.image_steps<0: p.error('Positive pose steps and nonnegative image steps required')
    with keep_windows_awake():
        if args.command=='fit':fit(args.root,args.pose_steps,args.image_steps,not args.freeze_skin,constrained=args.constrained)
        else:evaluate(args.root)
