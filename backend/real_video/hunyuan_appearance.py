"""Asset-specific Gaussian adaptation of Hunyuan geometry, not base-model training.

Unobserved appearance stays neutral gray, deliberately not invented by projection.
"""
import argparse
import json
import math
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import trimesh
from .hunyuan_gaussian import ROOT, SOURCE
from .checkpoint_io import load_verified, save_inference_checkpoint, keep_windows_awake
from .gaussian3d import render
from .fit_appearance3d import camera_from, FixedColourOperator


def target_data(size):
    rgb=np.asarray(Image.open(SOURCE).convert('RGB')).astype(np.float32)/255
    mask=np.asarray(Image.open(ROOT/'mask.png'))>0
    y,x=np.where(mask); rgb=rgb[y.min():y.max()+1,x.min():x.max()+1]; mask=mask[y.min():y.max()+1,x.min():x.max()+1]
    side=math.ceil(max(mask.shape)/.85); top=(side-mask.shape[0])//2; left=(side-mask.shape[1])//2
    canvas=np.full((side,side,3),.5,np.float32); alpha=np.zeros((side,side),np.float32)
    canvas[top:top+len(mask),left:left+mask.shape[1]]=np.where(mask[...,None],rgb,.5)
    alpha[top:top+len(mask),left:left+mask.shape[1]]=mask
    image=F.interpolate(torch.from_numpy(canvas).permute(2,0,1)[None],(size,size),mode='bilinear',align_corners=False)[0].permute(1,2,0)
    alpha=F.interpolate(torch.from_numpy(alpha)[None,None],(size,size),mode='bilinear',align_corners=False)[0,0]
    return image.cuda(),alpha.cuda()


def sample_surface(vertices,faces,count=180000):
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    rng=np.random.default_rng(531020)
    face=rng.choice(len(faces),count,p=mesh.area_faces/mesh.area)
    uv=rng.random((count,2)); uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    bary=np.column_stack((1-uv.sum(1),uv)); xyz=(vertices[faces[face]]*bary[:,:,None]).sum(1)
    normal=torch.tensor(np.asarray(mesh.face_normals)[face],dtype=torch.float32)
    helper=torch.tensor([0.,1.,0.]).expand_as(normal).clone(); helper[normal[:,1].abs()>.9]=torch.tensor([1.,0.,0.])
    tangent=F.normalize(torch.linalg.cross(helper,normal),dim=-1); frame=torch.stack((tangent,torch.linalg.cross(normal,tangent),normal),-1)
    sigma=math.sqrt(mesh.area/count)*.62
    scale=torch.tensor([sigma,sigma,sigma*.15]).expand(count,-1)
    cov=(frame*scale[:,None,:].square())@frame.transpose(-1,-2)
    return dict(position=torch.tensor(xyz,dtype=torch.float32),covariance=cov,normal=normal,
        colour=torch.full((count,3),.62),opacity=torch.full((count,),.98),ids=torch.arange(count),
        face_id=torch.tensor(face),barycentric=torch.tensor(bary,dtype=torch.float32))


def cuda(a):
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in a.items()}


@torch.no_grad()
def views(a,name):
    a=cuda(a); p=a['position']; center=(p.amin(0)+p.amax(0))/2
    distance=float((p.amax(0)-p.amin(0)).max())*1.9
    sheet=Image.new('RGB',(1536,1064),'#222222'); draw=ImageDraw.Draw(sheet)
    for i,(label,direction) in enumerate([('front',(0,0,1)),('right',(1,0,0)),('back',(0,0,-1)),('left',(-1,0,0)),('above',(0,1,0)),('oblique',(.7,.25,1))]):
        v=p.new_tensor(direction); eye=center+distance*v/v.norm()
        # Lighting is diagnostic only, never written into persistent color.
        shade=.35+.65*(a['normal']*F.normalize(eye-p,dim=-1)).sum(-1).abs()
        colour=a['colour']*shade[:,None] if name=='geometry' else a['colour']
        pic,_=render(p,a['covariance'],colour,a['opacity'],eye,center,512,512,ground=False)
        image=Image.fromarray(np.uint8(pic.cpu().numpy()*255)); image.save(ROOT/f'{name}_{label}.png')
        x=(i%3)*512; y=(i//3)*532; sheet.paste(image,(x,y+20)); draw.text((x+8,y+3),label,fill='white')
    sheet.save(ROOT/f'{name}_six_views.jpg')


def sample():
    if (ROOT/'gaussian_geometry.pt').exists(): raise FileExistsError('Preserve asset')
    mesh=load_verified(ROOT/'shape.pt')
    a=sample_surface(mesh['vertices'].numpy(),mesh['faces'].numpy())
    a['scope']='Hunyuan3D geometry sampled into persistent surface-bound Gaussians; no learned texture or motion.'
    save_inference_checkpoint(a,ROOT/'gaussian_geometry.pt'); views(a,'geometry')


def fit():
    if (ROOT/'gaussian_fitted.pt').exists(): raise FileExistsError('Preserve fit')
    start=time.perf_counter(); a=cuda(load_verified(ROOT/'gaussian_geometry.pt')); p=a['position']; center=(p.amin(0)+p.amax(0))/2
    _,mask=target_data(160); parameters=torch.nn.Parameter(p.new_tensor([0.,0.,math.log(2.),float(center[0]),float(center[1])]))
    optimizer=torch.optim.Adam([parameters],lr=.012); best=None; best_loss=float('inf')
    # Coarser Gaussian subset for camera fitting only; full asset remains fixed.
    for step in range(160):
        eye,look=camera_from(parameters,center[2])
        _,alpha=render(p[::6],a['covariance'][::6]*6,a['colour'][::6],a['opacity'][::6],eye,look,160,160,ground=False,radius=3)
        loss=(alpha-mask).square().mean()
        if float(loss.detach())<best_loss: best_loss=float(loss.detach()); best=parameters.detach().clone()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad(): parameters[:2].clamp_(-.8,.8); parameters[2].clamp_(math.log(1.),math.log(4.))
        if step%40==0: print(json.dumps(dict(stage='camera',step=step,loss=float(loss.detach()))),flush=True)
    eye,look=camera_from(best,center[2]); target,mask=target_data(768)
    with torch.no_grad():
        _,alpha,cache=render(p,a['covariance'],a['colour'],a['opacity'],eye,look,768,768,ground=False,radius=4,return_cache=True)
        mass=torch.zeros(len(p),device='cuda').index_add_(0,cache['ids'],cache['weight'])
        visible=(mass>.1)&((a['normal']*(eye-p)).sum(-1)>0)
    op=FixedColourOperator(cache,alpha); original=a['colour'].clone()
    logits=torch.nn.Parameter(torch.logit(original)); optimizer=torch.optim.Adam([logits],lr=.1)
    for step in range(300):
        colour=torch.where(visible[:,None],logits.sigmoid(),original); rgb=op(colour)
        loss=((rgb-target).square()*mask[...,None]).sum()/(3*mask.sum())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step%75==0: print(json.dumps(dict(stage='colour',step=step,mse=float(loss.detach()))),flush=True)
    with torch.no_grad():
        a['colour']=torch.where(visible[:,None],logits.sigmoid(),original).detach(); pic=op(a['colour'])
        mse=float(((pic-target).square()*mask[...,None]).sum()/(3*mask.sum()))
        iou=float(((alpha>.5)&(mask>.5)).sum()/((alpha>.5)|(mask>.5)).sum())
    a['appearance_observed']=visible; a['scope']='Asset-specific 768px Gaussian RGB fit to sharp Wan source. Unseen color neutral gray; base weights unchanged; no motion.'
    a['original_count']=len(p); a['camera_pivot']=look.tolist(); a['camera_distance']=float((eye-look).norm())
    save_inference_checkpoint({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in a.items()},ROOT/'gaussian_fitted.pt')
    save_inference_checkpoint(dict(eye=eye.cpu(),target=look.cpu(),fov=42.),ROOT/'camera.pt')
    for name,img in [('fit_source',target),('fit_result',pic)]: Image.fromarray(np.uint8(img.detach().clamp(0,1).cpu().numpy()*255)).save(ROOT/f'{name}.png')
    report=dict(seconds=time.perf_counter()-start,gaussians=len(p),visible_colours=int(visible.sum()),masked_training_psnr=-10*math.log10(mse),silhouette_iou=iou,
        positions_unchanged=True,unseen_colour_unchanged=bool(torch.equal(a['colour'][~visible],original[~visible])),base_weights_modified=False,
        scope='Source-view asset reconstruction; not held-out quality, texture generation, motion generation or speed comparison.',accepted=False)
    (ROOT/'appearance_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)
    views(a,'appearance')


@torch.no_grad()
def inspect():
    import imageio.v2 as imageio
    a=cuda(load_verified(ROOT/'gaussian_fitted.pt')); c=load_verified(ROOT/'camera.pt')
    target=c['target'].cuda(); distance=float((c['eye']-c['target']).norm())
    path=ROOT/'camera_only_7s.mp4'
    if path.exists(): raise FileExistsError('Preserve diagnostic')
    with imageio.get_writer(path,fps=16,codec='libx264',quality=8) as writer:
        for i in range(112):
            angle=2*math.pi*i/112; eye=target+target.new_tensor([math.sin(angle)*distance,.08,math.cos(angle)*distance])
            pic,_=render(a['position'],a['covariance'],a['colour'],a['opacity'],eye,target,512,768,ground=False)
            frame=np.uint8(pic.clamp(0,1).cpu().numpy()*255)
            writer.append_data(frame)
            if i in [0,28,56,84]: Image.fromarray(frame).save(ROOT/f'orbit_{i:03d}.png')
    print(json.dumps(dict(video=str(path),scope='Static inferred asset, camera movement only; NOT generated cat motion.')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['sample','fit','inspect']); args=parser.parse_args()
    torch.set_num_threads(4)
    with keep_windows_awake(): globals()[args.stage]()
