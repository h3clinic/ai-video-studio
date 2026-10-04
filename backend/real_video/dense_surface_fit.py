"""Source-conditioned connected XYZ surface fitting, NOT motion generation.

Dense forward/backward image flow supplies uncertain 2D constraints. Their lift
uses the canonical inferred depth: hidden 3D trajectories are not measured.
Appearance/IDs/barycentric Gaussian ownership are immutable. A fixed three-run
loop trades correspondence fit against local shape preservation; all candidates
and rejected outcomes are retained. The validation split is vertex-wise within
this one observed video, not novel-video or novel-motion generalization.
"""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.sparse.linalg import splu
import torch
from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from .fit_motion3d import read_frames, project_points
from .gaussian3d import look_at, render

ROOT=Path('artifacts/real_video/true3d/dense_fit_loop/v1')
ASSET=Path('artifacts/real_video/true3d/v4_appearance/cat_asset.pt')
CAMERA=Path('artifacts/real_video/true3d/v4_appearance/camera.pt')
BASELINE=Path('artifacts/real_video/true3d/v2_motion/surface_refined')
SOURCE=Path('artifacts/real_video/wan_baseline/cat_seed_421001/wan_original.mp4')


class ARAPMesh:
    """Local/global shared-vertex ARAP with soft position observations.

    Inputs/outputs are CPU NumPy arrays. This is a soft regularizer, NOT a
    nonfolding or collision-free guarantee. A weak rest prior fixes null spaces.
    """
    def __init__(self, rest_vertices, faces, edge_weighting='uniform'):
        self.rest=np.asarray(rest_vertices,dtype=np.float64)
        self.faces=np.asarray(faces,dtype=np.int64)
        if self.rest.ndim!=2 or self.rest.shape[1]!=3 or not np.isfinite(self.rest).all():
            raise ValueError('Expected finite Vx3 vertices')
        if self.faces.ndim!=2 or self.faces.shape[1]!=3 or self.faces.min()<0 or self.faces.max()>=len(self.rest):
            raise ValueError('Invalid mesh face indices')
        edges=np.concatenate([self.faces[:,[0,1]],self.faces[:,[1,2]],self.faces[:,[2,0]]])
        self.edges=np.unique(np.sort(edges,axis=1),axis=0)
        self.i,self.j=self.edges.T
        self.edge=self.rest[self.i]-self.rest[self.j]
        self.length=np.linalg.norm(self.edge,axis=1)
        if edge_weighting not in ['uniform','relative']: raise ValueError('Unknown edge weighting')
        self.edge_weighting=edge_weighting
        self.weights=np.ones(len(self.edges)) if edge_weighting=='uniform' else (np.median(self.length)/np.maximum(self.length,1e-5)).clip(.2,10.)**2
        row=np.r_[self.i,self.j]; col=np.r_[self.j,self.i]
        adjacency=sparse.coo_matrix((np.r_[self.weights,self.weights],(row,col)),shape=(len(self.rest),len(self.rest))).tocsr()
        self.laplacian=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency

    def solve(self, targets, confidence, stiffness=6., iterations=8, prior=.002, initial=None, prior_target=None):
        targets=np.asarray(targets,dtype=np.float64); confidence=np.asarray(confidence,dtype=np.float64)
        if targets.shape!=self.rest.shape or confidence.shape!=(len(self.rest),):
            raise ValueError('Target/confidence shape mismatch')
        if not np.isfinite(targets).all() or not np.isfinite(confidence).all() or np.any(confidence<0):
            raise ValueError('Nonfinite/negative observations')
        if stiffness<=0 or prior<=0 or iterations<1: raise ValueError('Positive solver controls required')
        matrix=(stiffness*self.laplacian+sparse.diags(confidence+prior)).tocsc()
        factor=splu(matrix)
        current=self.rest.copy() if initial is None else np.asarray(initial,dtype=np.float64).copy()
        if current.shape!=self.rest.shape or not np.isfinite(current).all(): raise ValueError('Invalid initial vertices')
        prior_target=self.rest if prior_target is None else np.asarray(prior_target,dtype=np.float64)
        if prior_target.shape!=self.rest.shape or not np.isfinite(prior_target).all(): raise ValueError('Invalid prior target')
        fixed=confidence[:,None]*targets+prior*prior_target
        for _ in range(iterations):
            delta=current[self.i]-current[self.j]
            covariance=np.zeros((len(self.rest),3,3),np.float64)
            products=delta[:,:,None]*self.edge[:,None,:]*self.weights[:,None,None]
            np.add.at(covariance,self.i,products); np.add.at(covariance,self.j,products)
            u,_,vt=np.linalg.svd(covariance)
            correction=np.ones((len(self.rest),3)); correction[:,-1]=np.linalg.det(u@vt)
            rotation=(u*correction[:,None,:])@vt
            rhs_edges=((rotation[self.i]+rotation[self.j])@self.edge[:,:,None])[:,:,0]*(.5*self.weights[:,None])
            rhs=np.zeros_like(self.rest)
            np.add.at(rhs,self.i,rhs_edges); np.add.at(rhs,self.j,-rhs_edges)
            current=factor.solve(stiffness*rhs+fixed)
        return current.astype(np.float32)


def source_pixels(vertices,camera):
    pixels=project_points(torch.as_tensor(vertices,dtype=torch.float32),camera).numpy()
    return pixels*camera['size']/camera['width']-np.array([camera['pad_x']-camera['x0'],camera['pad_y']-camera['y0']])


def lift_pixels(pixels,rest,camera):
    """Camera-ray lift preserving each vertex's canonical inferred depth."""
    rotation=look_at(camera['eye'],camera['target']).numpy(); eye=camera['eye'].numpy()
    depth=((rest-eye)@rotation.T)[:,2]
    condition=(pixels+np.array([camera['pad_x']-camera['x0'],camera['pad_y']-camera['y0']]))*camera['width']/camera['size']
    focal=.5*camera['height']/np.tan(np.deg2rad(camera['fov'])*.5)
    xy=(condition-np.array([camera['width']/2,camera['height']/2]))/focal*depth[:,None]
    return (np.concatenate([xy,depth[:,None]],axis=1)@rotation+eye).astype(np.float32)


def sample(image,points):
    p=np.asarray(points,np.float32)
    return cv2.remap(image,p[:,0:1],p[:,1:2],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(len(p),*image.shape[2:])


def track_dense(frames,initial,rest=2):
    """Cumulative dense DIS flow with FB, bounds and photo confidence gates."""
    gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in frames]
    n=len(initial); tracks=np.zeros((len(frames),n,2),np.float32); conf=np.zeros((len(frames),n),np.float32)
    tracks[rest]=initial; conf[rest]=1.; dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    records=[]
    for direction in [-1,1]:
        for t in range(rest+direction,-1 if direction<0 else len(frames),direction):
            previous=t-direction; forward=dis.calc(gray[previous],gray[t],None); backward=dis.calc(gray[t],gray[previous],None)
            p=tracks[previous]; q=p+sample(forward,p); fb=np.linalg.norm(sample(backward,q)+q-p,axis=1)
            photo=np.abs(sample(gray[previous].astype(np.float32),p)-sample(gray[t].astype(np.float32),q))
            inside=(q[:,0]>2)&(q[:,0]<frames[0].shape[1]-3)&(q[:,1]>2)&(q[:,1]<frames[0].shape[0]-3)
            valid=inside&(fb<2.)&(photo<45.)
            local=np.exp(-.5*(fb/1.5)**2)*np.exp(-.5*(photo/25.)**2)*valid
            # A cumulative minimum, not a confidence product, avoids exponential
            # decay while still permanently rejecting a failed correspondence.
            conf[t]=np.minimum(conf[previous],local); tracks[t]=q
            records.append(dict(frame=t,valid_fraction=float((conf[t]>.15).mean()),median_fb_px=float(np.median(fb))))
    return tracks,conf,records


def track_lk(frames,initial,rest=2):
    """Independent sparse tracker used only for fixed validation vertices."""
    gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in frames]; tracks=np.zeros((len(frames),len(initial),2),np.float32)
    confidence=np.zeros(tracks.shape[:2],np.float32); tracks[rest]=initial; confidence[rest]=1.
    settings=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
    for direction in [-1,1]:
        for t in range(rest+direction,-1 if direction<0 else len(frames),direction):
            p=tracks[t-direction].reshape(-1,1,2)
            q,ok,error=cv2.calcOpticalFlowPyrLK(gray[t-direction],gray[t],p,None,**settings)
            back,okback,_=cv2.calcOpticalFlowPyrLK(gray[t],gray[t-direction],q,None,**settings)
            fb=np.linalg.norm(back[:,0]-p[:,0],axis=1)
            valid=ok[:,0].astype(bool)&okback[:,0].astype(bool)&(fb<1.)&(error[:,0]<25.)
            valid&=(q[:,0,0]>2)&(q[:,0,0]<frames[0].shape[1]-3)&(q[:,0,1]>2)&(q[:,0,1]<frames[0].shape[0]-3)
            tracks[t]=q[:,0]; confidence[t]=confidence[t-direction]*valid
    return tracks,confidence


def visible_vertices(asset,camera):
    """Material-contribution visibility, transferred through immutable faces."""
    a={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in asset.items()}
    cov=(a['frame']*a['scale'][:,None].square())@a['frame'].transpose(-1,-2)
    with torch.no_grad():
        _,_,cache=render(a['position'],cov,a['colour'],a['opacity'],camera['eye'].cuda(),camera['target'].cuda(),512,512,camera['fov'],ground=False,return_cache=True)
        mass=torch.zeros(len(a['position']),device='cuda').index_add_(0,cache['ids'],cache['weight'])
        vertex_mass=torch.zeros(len(a['mesh_vertices']),device='cuda')
        vertices=a['mesh_faces'].long()[a['face_id'].long()]
        vertex_mass.index_add_(0,vertices.flatten(),(a['barycentric']*mass[:,None]).flatten())
    return vertex_mass.cpu().numpy()>.3


def geometry_metrics(rest,moved,faces,edges):
    valid=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)>1e-5
    ratio=np.linalg.norm(moved[edges[:,0]]-moved[edges[:,1]],axis=1)[valid]/np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)[valid]
    rt=rest[faces]; mt=moved[faces]
    rnormal=np.cross(rt[:,1]-rt[:,0],rt[:,2]-rt[:,0]); normal=np.cross(mt[:,1]-mt[:,0],mt[:,2]-mt[:,0])
    area=np.linalg.norm(rnormal,axis=1); area_new=np.linalg.norm(normal,axis=1); valid_face=area>1e-7
    ar=area_new[valid_face]/area[valid_face]
    # A negative rest-normal dot flags a large orientation change; articulation
    # can rotate valid faces, so this is not by itself a 3D inversion proof.
    opposed=(normal[valid_face]*rnormal[valid_face]).sum(1)<0
    return dict(edge_ratio_p01=float(np.quantile(ratio,.01)),edge_ratio_p99=float(np.quantile(ratio,.99)),
                edge_ratio_min=float(ratio.min()),edge_ratio_max=float(ratio.max()),
                area_ratio_p01=float(np.quantile(ar,.01)),area_ratio_p99=float(np.quantile(ar,.99)),
                tiny_area_fraction=float((ar<.1).mean()),opposed_rest_normal_fraction=float(opposed.mean()))


def epe(vertices,indices,tracks,confidence,camera):
    prediction=np.stack([source_pixels(v[indices],camera) for v in vertices])
    error=np.linalg.norm(prediction-tracks,axis=-1); weight=confidence.copy(); weight[2]=0.
    return float((error*weight).sum()/max(weight.sum(),1)),error


def gaussian_state(asset,vertices):
    """Preview only. Root's memory exporter owns durable per-dot trajectories."""
    from .surface_skin3d import triangle_basis
    faces=asset['mesh_faces'].long(); ids=asset['face_id'].long()
    rest,area=triangle_basis(asset['mesh_vertices'][faces]); rest=rest.clone(); rest[area<1e-12]=torch.eye(3,device=rest.device)
    basis,newarea=triangle_basis(vertices[faces]); gradient=basis@torch.linalg.inv(rest)
    position=(vertices[faces[ids]]*asset['barycentric'][:,:,None]).sum(1)
    covariance=(asset['frame']*asset['scale'][:,None].square())@asset['frame'].transpose(-1,-2)
    g=gradient[ids]; covariance=g@covariance@g.transpose(-1,-2)
    return position,covariance


def source_crop(frame,camera,resolution=512):
    yy,xx=np.mgrid[:resolution,:resolution].astype(np.float32)
    source_x=xx*camera['size']/resolution-camera['pad_x']+camera['x0']
    source_y=yy*camera['size']/resolution-camera['pad_y']+camera['y0']
    return cv2.remap(frame,source_x,source_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(128,128,128))


def evaluate_render(asset,camera,vertices,frames,directory):
    """Fixed full crop and edge evaluation, not a changing candidate mask."""
    a={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in asset.items()}; c={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in camera.items()}
    scores=[]; pics=[]
    for t in [2,6,14,22,30]:
        xyz,cov=gaussian_state(a,torch.from_numpy(vertices[t]).cuda())
        with torch.no_grad(): rgb,alpha=render(xyz,cov,a['colour'],a['opacity'],c['eye'],c['target'],512,512,c['fov'],ground=False)
        # Evaluate foreground-union ROI using fixed source-frame-2 mask advected
        # nowhere: broad central ROI, same pixels for baseline and candidates.
        rendered=rgb.cpu().numpy(); observed=source_crop(frames[t],camera).astype(np.float32)/255
        roi=np.zeros((512,512),bool); roi[62:454,38:496]=True
        mae=np.abs(rendered-observed)[roi].mean(); mse=((rendered-observed)**2)[roi].mean()
        er=cv2.Sobel(cv2.cvtColor(rendered,cv2.COLOR_RGB2GRAY),cv2.CV_32F,1,0)
        eo=cv2.Sobel(cv2.cvtColor(observed,cv2.COLOR_RGB2GRAY),cv2.CV_32F,1,0)
        scores.append(dict(frame=t,roi_mae=float(mae),roi_psnr=float(-10*np.log10(mse)),roi_edge_mae=float(np.abs(er-eo)[roi].mean())))
        pic=np.uint8(rendered.clip(0,1)*255); Image.fromarray(pic).save(directory/f'render_{t:02d}.png'); pics.append((t,pic,observed))
    canvas=Image.new('RGB',(1024,5*544),'#151922'); d=ImageDraw.Draw(canvas)
    for row,(t,pic,observed) in enumerate(pics):
        canvas.paste(Image.fromarray(np.uint8(observed*255)),(0,row*544+32)); canvas.paste(Image.fromarray(pic),(512,row*544+32))
        d.text((8,row*544+8),f'Frame {t}: SOURCE left | fixed Gaussian appearance + fitted shared XYZ mesh right',fill='white')
    canvas.save(directory/'comparison.jpg')
    return scores


def run(weighted_prior=False):
    torch.set_num_threads(4); cv2.setNumThreads(4); ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'selected_motion.pt').exists(): raise FileExistsError('Preserve previous loop results')
    start=time.perf_counter(); asset=load_verified(ASSET); camera=load_verified(CAMERA); frames,fps=read_frames()
    rest=asset['mesh_vertices'].numpy(); faces=asset['mesh_faces'].long().numpy(); solver=ARAPMesh(rest,faces,'relative' if weighted_prior else 'uniform')
    visible=visible_vertices(asset,camera); pixels=source_pixels(rest,camera)
    visible&=(pixels[:,0]>3)&(pixels[:,0]<829)&(pixels[:,1]>3)&(pixels[:,1]<477)
    indices=np.flatnonzero(visible); validation=indices[(indices*2654435761%97)<19]; training=np.setdiff1d(indices,validation)
    protocol=dict(configurations=[dict(name='flexible',stiffness=1.),dict(name='balanced',stiffness=6.),dict(name='stiff',stiffness=24.)],
        iterations=10,prior=.002,rest_frame=2,training_vertices=training.tolist(),validation_vertices=validation.tolist(),
        selection='minimum independent LK validation EPE among candidates passing edge p01>=0.35,p99<=2.5,tiny-area fraction<=0.002 and EPE<baseline; else no accepted candidate',
        source_scope='All 33 source frames are observed for reconstruction. Deterministic vertex-wise validation; NOT held-out-frame/video or generation generalization.',
        photo_scope='Fixed broad 512px crop ROI [62:454,38:496], all methods identical; includes background mismatch and is secondary diagnostic.',
        sources=['https://arxiv.org/html/2308.09713v1','https://arxiv.org/html/2312.14937v1'],
        mathematics='min_X sum_v c_v||X_v-Y_v||^2 + lambda sum_edges ||(X_i-X_j)-R_i(V_i-V_j)||^2 + prior||X-V||^2; local SVD rotations + sparse global shared-vertex solves. Lift Y uses inferred rest camera depth; not 3D evidence.')
    if weighted_prior:
        protocol.update(configurations=[dict(name='relative_4',stiffness=4.),dict(name='relative_12',stiffness=12.),dict(name='relative_36',stiffness=36.)],iterations=16,prior=.03,
            edge_weighting='relative: clipped(median_rest_edge_length/rest_edge_length,0.2,10)^2',
            hidden_prior='Weak source-fitted skeleton mesh prior on every vertex (.03); visibility-filtered DIS data dominates observed surface. Prior uses existing source/manual observations, not learned generation.',
            previous_loop='v1 all three uniform-edge candidates failed the fixed strain gate. This new protocol was fixed before running v2; v1 is retained.')
    protocol_path=ROOT/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:
            raise ValueError('Existing immutable loop protocol differs; use a new output directory')
    else: protocol_path.write_text(json.dumps(protocol,indent=2))
    print(json.dumps(dict(stage='protocol',training=len(training),validation=len(validation))),flush=True)
    if (ROOT/'tracks.pt').exists():
        packet=load_verified(ROOT/'tracks.pt')
        if packet['source_video_sha256']!=digest(SOURCE) or not np.array_equal(packet['training_vertices'].numpy(),training) or not np.array_equal(packet['validation_vertices'].numpy(),validation):
            raise ValueError('Cached tracking provenance/split mismatch')
        targets=packet['tracks'].numpy(); confidence=packet['confidence'].numpy(); valtrack=packet['validation_tracks'].numpy(); valconf=packet['validation_confidence'].numpy()
    else:
        targets,confidence,records=track_dense(frames,pixels[training]); valtrack,valconf=track_lk(frames,pixels[validation])
        save_inference_checkpoint(dict(tracks=torch.from_numpy(targets),confidence=torch.from_numpy(confidence),training_vertices=torch.from_numpy(training),validation_tracks=torch.from_numpy(valtrack),validation_confidence=torch.from_numpy(valconf),validation_vertices=torch.from_numpy(validation),source_video_sha256=digest(SOURCE)),ROOT/'tracks.pt')
        (ROOT/'tracking_report.json').write_text(json.dumps(records,indent=2))
    from .surface_skin3d import SurfaceSkinner
    original=load_verified(BASELINE/'motion.pt'); rig=load_verified(BASELINE/'rig.pt'); support=SurfaceSkinner(asset,rig)
    baseline=np.stack([support(original['local_rotation'][t],original['root_translation'][t])[3].numpy() for t in range(len(frames))])
    baseline_epe,_=epe(baseline,validation,valtrack,valconf,camera)
    basedir=ROOT/'baseline'; basedir.mkdir(exist_ok=True); baseline_photo=evaluate_render(asset,camera,baseline,frames,basedir)
    reports=[]; all_vertices={}
    for config in protocol['configurations']:
        name=config['name']; directory=ROOT/name; directory.mkdir(exist_ok=True); path=directory/'motion.pt'
        if path.exists():
            packet=load_verified(path)
            if packet['asset_sha256']!=digest(ASSET) or packet['source_video_sha256']!=digest(SOURCE) or packet['configuration']!=config:
                raise ValueError('Cached candidate provenance/configuration mismatch')
            for key in ['eye','target']:
                if not torch.equal(packet['camera'][key],camera[key]): raise ValueError('Cached candidate camera mismatch')
            vertices=packet['vertices'].numpy()
        else:
            vertices=np.zeros((len(frames),len(rest),3),np.float32); vertices[2]=rest
            for direction in [-1,1]:
                for t in range(2+direction,-1 if direction<0 else len(frames),direction):
                    full_pixels=pixels.copy(); full_pixels[training]=targets[t]
                    lifted=lift_pixels(full_pixels,rest,camera); weights=np.zeros(len(rest),np.float32)
                    weights[training]=confidence[t]*(confidence[t]>.15)
                    vertices[t]=solver.solve(lifted,weights,stiffness=config['stiffness'],iterations=protocol['iterations'],prior=protocol['prior'],initial=vertices[t-direction],prior_target=baseline[t] if weighted_prior else None)
                    if t%8==0: print(json.dumps(dict(stage=name,frame=t)),flush=True)
            packet=dict(vertices=torch.from_numpy(vertices),source_frame=torch.arange(len(frames)),fps=fps,camera=camera,rest_frame=2,
                asset_path=str(ASSET),asset_sha256=digest(ASSET),source_video_sha256=digest(SOURCE),baseline_path=str(BASELINE),
                ids=asset['ids'],configuration=config,source='Offline 33-frame source-conditioned dense-flow reconstruction, no learned future generation. Hidden depth prior inferred from single image.',
                weighted_skeleton_prior=weighted_prior,
                depth_prior='Canonical camera depth used only for observation lift; ARAP can adjust unobserved depth. No metric 3D tracking truth.',
                gaussian_memory='Fixed canonical IDs, face_id, barycentric, colour, opacity; Gaussians derive from shared vertices. Per-dot vectors exported separately.')
            save_inference_checkpoint(packet,path)
        score,_=epe(vertices,validation,valtrack,valconf,camera); train_score,_=epe(vertices,training,targets,confidence,camera)
        geometry=[geometry_metrics(rest,v,faces,solver.edges) for v in vertices]
        accepted=all(g['edge_ratio_p01']>=.35 and g['edge_ratio_p99']<=2.5 and g['tiny_area_fraction']<=.002 for g in geometry) and score<baseline_epe
        photo=evaluate_render(asset,camera,vertices,frames,directory)
        report=dict(**config,validation_lk_epe_source_px=score,fit_dis_epe_source_px=train_score,baseline_lk_epe_source_px=baseline_epe,
            epe_improvement_ratio=baseline_epe/max(score,1e-9),accepted=accepted,geometry=geometry,photometry=photo,
            gates='Numerical strain and correspondence gates only; not a natural gait or hidden-geometry certification.')
        (directory/'report.json').write_text(json.dumps(report,indent=2)); reports.append(report); all_vertices[name]=vertices
        print(json.dumps({k:v for k,v in report.items() if k not in ['geometry','photometry']}),flush=True)
    passing=[r for r in reports if r['accepted']]; best=min(passing or reports,key=lambda r:r['validation_lk_epe_source_px'])
    selected=load_verified(ROOT/best['name']/'motion.pt'); selected['passed_selection_gates']=bool(passing); selected['selection_report']=str(ROOT/'loop_report.json')
    save_inference_checkpoint(selected,ROOT/'selected_motion.pt')
    report=dict(selected=best['name'],accepted=bool(passing),baseline_lk_epe_source_px=baseline_epe,selected_lk_epe_source_px=best['validation_lk_epe_source_px'],
        improvement_ratio=baseline_epe/best['validation_lk_epe_source_px'],configs=reports,baseline_photometry=baseline_photo,seconds=time.perf_counter()-start,
        claims='Reconstruction from one observed Wan clip, not generated motion, not true hidden 3D motion. 10x quality not established by these metrics.',
        protocol_sha256=digest(ROOT/'protocol.json'),source_video_sha256=digest(SOURCE),asset_sha256=digest(ASSET))
    (ROOT/'loop_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k not in ['configs','baseline_photometry']}),flush=True)


def audit_regions():
    """Expose survivor bias explicitly; tracking loss is not total limb accuracy."""
    p=load_verified(ROOT/'tracks.pt'); ids=p['validation_vertices'].numpy(); tracks=p['validation_tracks'].numpy(); confidence=p['validation_confidence'].numpy()
    regions={'body':tracks[2,:,1]<300,'limbs':tracks[2,:,1]>=300,'feet':tracks[2,:,1]>=370}
    reports={}
    for path in sorted(ROOT.glob('*/motion.pt')):
        m=load_verified(path); _,error=epe(m['vertices'].numpy(),ids,tracks,confidence,m['camera']); result={}
        for name,mask in regions.items():
            weight=confidence*mask[None]; weight[2]=0
            result[name]=dict(initial_points=int(mask.sum()),last_frame_valid=int((confidence[-1]*mask).sum()),
                endpoint_survival_fraction=float((confidence[-1]*mask).sum()/max(mask.sum(),1)),
                surviving_track_epe_source_px=float((error*weight).sum()/max(weight.sum(),1)),
                per_frame_valid=(confidence*mask[None]).sum(1).astype(int).tolist())
        reports[path.parent.name]=result
    report=dict(region_rule='Canonical source image y<300 body, y>=300 limbs, y>=370 feet. Heuristic spatial groups, not anatomical truth.',
        warning='Severe survivor bias: occluded or failed tracks have zero confidence. Low EPE does NOT imply accurate motion of all paws or all 45000 Gaussians. Independent LK is an estimated 2D correspondence, not 3D motion ground truth.',
        candidates=reports,tracking_sha256=digest(ROOT/'tracks.pt'))
    (ROOT/'coverage_audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


@torch.no_grad()
def export_video():
    """Native source replay + explicitly slowed inspection; NOT generation."""
    from .surface_skin3d import SurfaceSkinner
    torch.set_num_threads(4); cv2.setNumThreads(4)
    packet=load_verified(ROOT/'selected_motion.pt'); camera=packet['camera']; vertices=packet['vertices']
    asset=load_verified(packet['asset_path']); source,fps=read_frames()
    original=load_verified(BASELINE/'motion.pt'); support=SurfaceSkinner(asset,load_verified(BASELINE/'rig.pt'))
    a={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in asset.items()}; eye=camera['eye'].cuda(); target=camera['target'].cuda()
    native=ROOT/'source_fit_comparison_native.mp4'; slowed=ROOT/'source_fit_comparison_7s.mp4'
    if native.exists() or slowed.exists(): raise FileExistsError('Preserve existing source-fit videos')
    writer=cv2.VideoWriter(str(native),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1536,570)); images=[]
    if not writer.isOpened(): raise RuntimeError('Video encoder unavailable')
    started=time.perf_counter()
    for t in range(len(vertices)):
        baseline_vertices=support(original['local_rotation'][t],original['root_translation'][t])[3].cuda()
        rendered=[]
        for v in [baseline_vertices,vertices[t].cuda()]:
            xyz,cov=gaussian_state(a,v)
            rgb,_=render(xyz,cov,a['colour'],a['opacity'],eye,target,512,512,camera['fov'],ground=False)
            rendered.append(np.uint8(rgb.cpu().numpy().clip(0,1)*255))
        canvas=Image.new('RGB',(1536,570),'#131c27'); d=ImageDraw.Draw(canvas)
        for x,title,pic in [(0,'Observed Wan source (reconstruction target)',source_crop(source[t],camera)),(512,'Previous: manually assisted skeleton fit',rendered[0]),(1024,'Now: dense-flow + connected XYZ mesh fit',rendered[1])]:
            canvas.paste(Image.fromarray(pic),(x,32)); d.text((x+8,10),title,fill='white')
        d.text((8,551),f'SOURCE-CONDITIONED RECONSTRUCTION | frame {t}/32 | appearance/45000 IDs retained | hidden anatomy and gait still unverified',fill='white')
        image=np.array(canvas); writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR)); images.append(image)
        if t in [0,8,16,24,32]: canvas.save(ROOT/f'video_comparison_{t:02d}.jpg')
    writer.release(); slowwriter=cv2.VideoWriter(str(slowed),cv2.VideoWriter_fourcc(*'mp4v'),24.,(1536,570))
    mapping=np.rint(np.linspace(0,len(images)-1,168)).astype(int)
    for index in mapping:
        canvas=Image.fromarray(images[index].copy()); d=ImageDraw.Draw(canvas)
        d.rectangle((0,545,1536,570),fill='#131c27'); d.text((8,551),f'SLOWED INSPECTION: original 33 source-conditioned states stretched to 7 seconds | source frame {index}/32 | NOT 7 seconds of generated motion',fill='white')
        slowwriter.write(cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR))
    slowwriter.release(); decoded={}
    for path in [native,slowed]:
        cap=cv2.VideoCapture(str(path)); rate=cap.get(cv2.CAP_PROP_FPS); count=0
        while True:
            ok,_=cap.read()
            if not ok: break
            count+=1
        cap.release(); decoded[path.name]=dict(decoded_frames=count,fps=rate,seconds=count/rate,sha256=digest(path))
    if decoded[native.name]['decoded_frames']!=33 or decoded[slowed.name]['decoded_frames']!=168: raise RuntimeError('Video decode count failed')
    report=dict(videos=decoded,render_and_encode_seconds=time.perf_counter()-started,selected_motion_sha256=digest(ROOT/'selected_motion.pt'),
        scope='Reconstruction comparison. The 7-second version repeats display frames from 33 original states; no motion interpolation or novel generation.')
    (ROOT/'video_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=ROOT); parser.add_argument('--audit-only',action='store_true'); parser.add_argument('--weighted-prior',action='store_true'); parser.add_argument('--video-only',action='store_true'); args=parser.parse_args(); ROOT=args.output
    with keep_windows_awake():
        if args.video_only: export_video()
        elif args.audit_only: audit_regions()
        else: run(args.weighted_prior); audit_regions()
