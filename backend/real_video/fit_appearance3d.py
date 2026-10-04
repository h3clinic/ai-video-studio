"""Source-view 3D Gaussian appearance fitting. NOT new-view ground truth.

Only camera and persistent colours change; the full XYZ body stays fixed.
The target is observed Wan frame 2. This is reconstruction, not generation.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from .checkpoint_io import load_verified,save_inference_checkpoint,keep_windows_awake,digest
from .gaussian3d import render
from .demo_true3d import covariance

BASE=Path('artifacts/real_video/true3d/v1')
ROOT=Path('artifacts/real_video/true3d/v2_appearance')
OBS=Path('artifacts/real_video/learned_motion/v1/dense_seed')


def target_data(resolution=512):
    image=np.array(Image.open(OBS/'observed_frame_2.png').convert('RGB'))
    mask=np.array(Image.open(OBS/'mask.png').convert('L'),dtype=np.float32)/255
    yy,xx=np.where(mask>.5); y0,y1,x0,x1=yy.min(),yy.max()+1,xx.min(),xx.max()+1
    rgb=image[y0:y1,x0:x1].astype(np.float32)/255; alpha=mask[y0:y1,x0:x1]
    size=int(max(rgb.shape[:2])/.85); pad_y=(size-rgb.shape[0])//2; pad_x=(size-rgb.shape[1])//2
    canvas=np.full((size,size,3),.5,np.float32); m=np.zeros((size,size),np.float32)
    canvas[pad_y:pad_y+len(rgb),pad_x:pad_x+rgb.shape[1]]=rgb*alpha[...,None]+.5*(1-alpha[...,None])
    m[pad_y:pad_y+len(rgb),pad_x:pad_x+rgb.shape[1]]=alpha
    target=F.interpolate(torch.from_numpy(canvas).permute(2,0,1)[None],size=(resolution,resolution),mode='bilinear',align_corners=False)[0].permute(1,2,0)
    target_mask=F.interpolate(torch.from_numpy(m)[None,None],size=(resolution,resolution),mode='bilinear',align_corners=False)[0,0]
    crop=dict(x0=int(x0),y0=int(y0),size=size,pad_x=pad_x,pad_y=pad_y)
    return target,target_mask,crop


def asset_cuda():
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in load_verified(BASE/'cat_asset.pt').items()}


def outward_sign(vertices,faces):
    """Closed isosurface winding can be inward; frames alone don't specify it."""
    faces=faces.long()
    edges=torch.cat((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),0)
    canonical=edges.sort(dim=1).values
    _,inverse,count=torch.unique(canonical,dim=0,return_inverse=True,return_counts=True)
    balance=torch.zeros_like(count).index_add_(0,inverse,torch.where(edges[:,0]<edges[:,1],1,-1))
    if bool((count!=2).any()) or bool((balance!=0).any()):
        raise ValueError('Visibility normal orientation requires closed, consistently wound mesh')
    triangles=vertices[faces.long()]
    volume=(triangles[:,0]*torch.linalg.cross(triangles[:,1],triangles[:,2])).sum()/6
    if abs(float(volume))<1e-8: raise ValueError('Cannot orient zero-volume surface')
    return volume.sign()


def camera_from(parameters,z):
    azimuth,elevation,log_distance,tx,ty=parameters
    target=torch.stack((tx,ty,z)); distance=log_distance.exp()
    direction=torch.stack((azimuth.sin()*elevation.cos(),elevation.sin(),azimuth.cos()*elevation.cos()))
    return target+distance*direction,target


def camera_fit():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'camera.pt').exists(): raise FileExistsError('Preserve camera')
    a=asset_cuda(); p=a['position']; cov=covariance(a); target,mask,crop=target_data(192); mask=mask.cuda()
    center=(p.amin(0)+p.amax(0))*.5
    parameters=torch.nn.Parameter(p.new_tensor([0.,0.,math.log(1.9),0.,float(center[1])]))
    optimizer=torch.optim.Adam([parameters],lr=.008); best=None; best_loss=float('inf'); initial=None
    start=time.perf_counter()
    for step in range(220):
        eye,look=camera_from(parameters,center[2])
        _,alpha=render(p,cov,a['colour'],a['opacity'],eye,look,192,192,fov=40,ground=False,radius=3)
        loss=(alpha-mask).square().mean()
        if initial is None: initial=float(loss.detach())
        if float(loss.detach())<best_loss: best_loss=float(loss.detach()); best=parameters.detach().clone()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            parameters[:2].clamp_(-.5,.5); parameters[2].clamp_(math.log(1.1),math.log(2.8))
        if step%55==0: print(json.dumps(dict(stage='camera',step=step,loss=float(loss.detach()))),flush=True)
    eye,look=camera_from(best,center[2]); packet=dict(eye=eye.detach().cpu(),target=look.detach().cpu(),fov=40.,height=512,width=512,**crop,
        source='Silhouette camera fit to observed Wan frame2; inferred3D remains unchanged')
    save_inference_checkpoint(packet,ROOT/'camera.pt')
    record=dict(initial_mse=initial,best_mse=best_loss,parameters=best.tolist(),seconds=time.perf_counter()-start)
    (ROOT/'camera_report.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record),flush=True)


class FixedColourOperator:
    def __init__(self,cache,alpha):
        self.cache=cache; self.alpha=alpha; self.height,self.width=alpha.shape
    def __call__(self,colour):
        c=self.cache
        rgb=colour.new_zeros(self.height*self.width,3).index_add(0,c['pixel'],c['weight'][:,None]*colour[c['ids']])
        return rgb.reshape(self.height,self.width,3)+(1-self.alpha[...,None])*.5


def colour_fit():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'cat_asset.pt').exists(): raise FileExistsError('Preserve fitted appearance')
    a=asset_cuda(); c=load_verified(ROOT/'camera.pt'); target,mask,_=target_data(); target=target.cuda(); mask=mask.cuda()
    with torch.no_grad():
        _,alpha,cache=render(a['position'],covariance(a),a['colour'],a['opacity'],c['eye'].cuda(),c['target'].cuda(),512,512,fov=c['fov'],ground=False,return_cache=True)
        colour_mass=torch.zeros(len(a['position']),device='cuda').index_add_(0,cache['ids'],cache['weight'])
        sign=outward_sign(a['mesh_vertices'],a['mesh_faces'])
        facing=(sign*a['frame'][:,:,2]*(c['eye'].cuda()-a['position'])).sum(1)>0
        # Enforce exactly unchanged unobserved/back-facing appearance, rather
        # than relying on a small soft prior during one-view fitting.
        trainable=((colour_mass>.1)&facing)[:,None]
    op=FixedColourOperator(cache,alpha); original=a['colour'].clone()
    logits=torch.nn.Parameter(torch.logit(original.clamp(.001,.999))); optim=torch.optim.Adam([logits],lr=.08)
    before=op(original).detach(); start=time.perf_counter(); losses=[]
    for step in range(400):
        colour=torch.where(trainable,logits.sigmoid(),original); rgb=op(colour)
        mse=((rgb-target).square()*mask[...,None]).sum()/(3*mask.sum())
        # Tiny prior leaves unobserved colours unchanged; no invented detail claim.
        loss=mse+1e-5*(colour-original).square().mean()
        optim.zero_grad(set_to_none=True); loss.backward(); optim.step(); losses.append(float(mse.detach()))
        if step%100==0: print(json.dumps(dict(stage='appearance',step=step,mse=losses[-1])),flush=True)
    with torch.no_grad():
        a['colour']=torch.where(trainable,logits.sigmoid(),original).detach(); after=op(a['colour'])
        before_mse=float(((before-target).square()*mask[...,None]).sum()/(3*mask.sum()))
        after_mse=float(((after-target).square()*mask[...,None]).sum()/(3*mask.sum()))
        iou=float(((alpha>.5)&(mask>.5)).sum()/((alpha>.5)|(mask>.5)).sum())
    a['appearance_source']='Persistent colour fitting to observed frame2. Excluded/back-facing or low-contribution splats keep prior. Geometry fixed; visibility is heuristic.'
    a['source_asset_sha256']=digest(BASE/'cat_asset.pt')
    save_inference_checkpoint({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in a.items()},ROOT/'cat_asset.pt')
    canvas=Image.new('RGB',(1536,548),'#151922'); draw=ImageDraw.Draw(canvas)
    for j,(title,pic) in enumerate([('Observed source (reconstruction target)',target),('Before: TripoSR colours',before),('After: fitted persistent Gaussian colours',after)]):
        canvas.paste(Image.fromarray(np.uint8(pic.clamp(0,1).cpu().numpy()*255)),(j*512,36)); draw.text((j*512+8,10),title,fill='white')
    canvas.save(ROOT/'appearance_comparison.jpg')
    for name,pic in [('before',before),('after',after)]: Image.fromarray(np.uint8(pic.clamp(0,1).cpu().numpy()*255)).save(ROOT/f'{name}.png')
    report=dict(gaussians=len(original),initial_masked_psnr=-10*math.log10(before_mse),final_masked_psnr=-10*math.log10(after_mse),
        silhouette_iou=iou,seconds=time.perf_counter()-start,steps=400,visible_colour_count=int(trainable.sum()),
        hidden_colours_exactly_unchanged=bool(torch.equal(a['colour'][~trainable[:,0]],original[~trainable[:,0]])),
        mesh_winding_sign=float(sign),
        metrics_scope='Training-view reconstruction only; not held-out/generalization or motion accuracy.',
        positions_unchanged=True,ids_unchanged=True,unseen_surface_validation=False)
    (ROOT/'appearance_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['camera','colour']); parser.add_argument('--root',type=Path,default=ROOT); args=parser.parse_args()
    ROOT=args.root
    torch.set_num_threads(4)
    with keep_windows_awake(): {'camera':camera_fit,'colour':colour_fit}[args.mode]()
