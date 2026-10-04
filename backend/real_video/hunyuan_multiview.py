"""Hunyuan Paint adapter with portable triangle controls and Gaussian baking.

Shape and texture are distinct pretrained models. No base weights are trained.
Control-map conventions reproduce the pinned upstream renderer; rasterization
uses PyTorch rather than requiring a locally compiled CUDA extension.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
import trimesh
from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake
from .hunyuan_gaussian import ROOT as BASE,WORK

ROOT=BASE.parent/'v2_paint'
WEIGHTS=WORK/'hunyuan_paint/hunyuan3d-paint-v2-0'
REVISION='9cd649ba6913f7a852e3286bad86bfa9a2d83dcf'
VIEWS=[(0,0),(0,90),(0,180),(0,270),(90,0),(-90,180)]


def canonical(vertices):
    v=vertices.clone(); v[:,[0,1]]=-v[:,[0,1]]; v=v[:,[0,2,1]]
    center=(v.amax(0)+v.amin(0))/2; diameter=(v-center).norm(dim=1).max()*2
    return (v-center)*(1.15/diameter),center,diameter


def camera(elev,azim,device):
    e=math.radians(-elev); a=math.radians(azim+90)
    eye=torch.tensor([math.cos(e)*math.cos(a),math.cos(e)*math.sin(a),math.sin(e)],device=device)*1.45
    forward=F.normalize(-eye,dim=0); up=eye.new_tensor([0.,0.,1.])
    # Analytic limit remains unit length at the poles; normalize(cross(...))
    # collapses under torch's epsilon when cos(elevation) is nearly zero.
    right=eye.new_tensor([-math.sin(a),math.cos(a),0.]); up=F.normalize(torch.linalg.cross(right,forward),dim=0)
    return eye,torch.stack((right,up,-forward))


def screen_position(v,elev,azim,size):
    eye,rot=camera(elev,azim,v.device); p=(v-eye)@rot.T
    # Upstream raster uses +NDC-y as increasing image row (no image flip).
    xy=(p[:,:2]/1.2+.5)*(size-1)+.5
    return torch.cat((xy,-p[:,2:3]),1)


@torch.no_grad()
def rasterize(screen,faces,size):
    """Orthographic z-buffer, nearest triangle, exact barycentric attributes."""
    tri=screen[faces]; lo=tri[:,:,:2].amin(1).floor().long().clamp(0,size-1)
    hi=tri[:,:,:2].amax(1).ceil().long().clamp(0,size-1); width=hi[:,0]-lo[:,0]+1
    count=width*(hi[:,1]-lo[:,1]+1)
    if int(count.sum())>40000000: raise ValueError('Raster candidate budget exceeded')
    ids=torch.repeat_interleave(torch.arange(len(faces),device=screen.device),count)
    offset=torch.arange(len(ids),device=screen.device)-torch.repeat_interleave(count.cumsum(0)-count,count)
    x=lo[ids,0]+offset%width[ids]; y=lo[ids,1]+offset//width[ids]
    t=tri[ids]; dx=x+.5-t[:,0,0]; dy=y+.5-t[:,0,1]
    ab=t[:,1,:2]-t[:,0,:2]; ac=t[:,2,:2]-t[:,0,:2]
    det=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]; safe=torch.where(det.abs()>1e-12,det,torch.ones_like(det))
    b=(dx*ac[:,1]-dy*ac[:,0])/safe; c=(ab[:,0]*dy-ab[:,1]*dx)/safe
    bary=torch.stack((1-b-c,b,c),1); z=(bary*t[:,:,2]).sum(1)
    good=(bary>=-1e-6).all(1)&(det.abs()>1e-12)&(z>0)
    ids,bary,z,pix=ids[good],bary[good],z[good],(y*size+x)[good]
    depth=screen.new_full((size*size,),float('inf')); depth.scatter_reduce_(0,pix,z,reduce='amin')
    near=z==depth[pix]; rank=torch.arange(len(ids),device=screen.device)
    winner=torch.full((size*size,),len(ids),device=screen.device,dtype=torch.long)
    winner.scatter_reduce_(0,pix[near],rank[near],reduce='amin')
    valid=winner<len(ids); face=torch.full_like(winner,-1); weights=screen.new_zeros(size*size,3)
    face[valid]=ids[winner[valid]]; weights[valid]=bary[winner[valid]]
    return face.reshape(size,size),weights.reshape(size,size,3),depth.reshape(size,size)


@torch.no_grad()
def controls():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'controls.pt').exists():raise FileExistsError('Preserve control checkpoint')
    shape=load_verified(BASE/'shape.pt'); v,center,diameter=canonical(shape['vertices'].cuda()); faces=shape['faces'].cuda().long()
    mesh=trimesh.Trimesh(vertices=v.cpu().numpy(),faces=faces.cpu().numpy(),process=False)
    normals=torch.tensor(np.asarray(mesh.vertex_normals),dtype=torch.float32,device='cuda')
    records=[]; sheet=Image.new('RGB',(1536,1024),'white')
    for i,(e,a) in enumerate(VIEWS):
        face,bary,depth=rasterize(screen_position(v,e,a,512),faces,512); mask=face>=0
        vertex=faces[face.clamp_min(0)]; pos=(v[vertex]*bary[...,None]).sum(2); normal=(normals[vertex]*bary[...,None]).sum(2)
        maps=[torch.where(mask[...,None],(normal+1)/2,1),torch.where(mask[...,None],.5-pos/1.15,1)]
        for name,img in zip(['normal','position'],maps):
            pil=Image.fromarray(np.uint8(img.clamp(0,1).cpu().numpy()*255));pil.save(ROOT/f'{name}_{i}.png')
            if name=='normal':sheet.paste(pil,((i%3)*512,(i//3)*512))
        records.append(dict(face=face.cpu(),barycentric=bary.cpu(),depth=torch.where(mask,depth,0).cpu(),elevation=e,azimuth=a))
    save_inference_checkpoint(dict(views=records,center=center.cpu(),diameter=diameter.cpu()),ROOT/'controls.pt')
    sheet.save(ROOT/'control_sheet.jpg')
    image=Image.open(BASE/'conditioning.png').convert('RGBA'); box=image.getchannel('A').getbbox(); image=image.crop(box)
    side=math.ceil(max(image.size)*1.4); canvas=Image.new('RGBA',(side,side),(255,255,255,0)); canvas.paste(image,((side-image.width)//2,(side-image.height)//2))
    canvas.resize((512,512),Image.Resampling.LANCZOS).save(ROOT/'reference.png')


@torch.inference_mode()
def generate():
    if (ROOT/'generated_0.png').exists():raise FileExistsError('Preserve generated views')
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from diffusers import UNet2DConditionModel,AutoencoderKL,EulerAncestralDiscreteScheduler
    sys.path.insert(0,str(WORK/'Hunyuan3D-2/hy3dgen/texgen'))
    from hunyuanpaint.pipeline import HunyuanPaintPipeline
    from hunyuanpaint.unet.modules import UNet2p5DConditionModel
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    config=json.loads((WEIGHTS/'unet/config.json').read_text())
    print('Loading multiview UNet with strict safetensors assignment',flush=True)
    with init_empty_weights(): unet=UNet2p5DConditionModel(UNet2DConditionModel.from_config(config))
    state=load_file(str(WEIGHTS/'unet/diffusion_pytorch_model.safetensors'))
    result=unet.load_state_dict(state,strict=True,assign=True);del state
    unet=unet.half().cuda().eval();gc.collect()
    vae=AutoencoderKL.from_pretrained(WEIGHTS/'vae',torch_dtype=torch.float16,local_files_only=True).cuda().eval();vae.enable_slicing()
    scheduler=EulerAncestralDiscreteScheduler.from_pretrained(WEIGHTS/'scheduler',timestep_spacing='trailing',local_files_only=True)
    # Pipeline uses learned unconditional/conditional embeddings from UNet, not a text encoder.
    pipe=HunyuanPaintPipeline(vae,None,None,unet,scheduler,None)
    seed=531022;torch.manual_seed(seed)
    camera_info=[(((a//30)+9)%12)//({0:1,90:3,-90:3}[e])+{0:12,90:40,-90:36}[e] for e,a in VIEWS]
    print('Generating all six texture views jointly',flush=True)
    images=pipe(Image.open(ROOT/'reference.png'),num_inference_steps=30,width=512,height=512,num_in_batch=6,
        camera_info_gen=[camera_info],camera_info_ref=[[0]],normal_imgs=[[Image.open(ROOT/f'normal_{i}.png') for i in range(6)]],
        position_imgs=[[Image.open(ROOT/f'position_{i}.png') for i in range(6)]],generator=torch.Generator('cuda').manual_seed(seed)).images
    sheet=Image.new('RGB',(1536,1024))
    for i,img in enumerate(images): img.save(ROOT/f'generated_{i}.png');sheet.paste(img,((i%3)*512,(i//3)*512))
    sheet.save(ROOT/'generated_sheet.jpg')
    report=dict(model='tencent/Hunyuan3D-2/hunyuan3d-paint-v2-0',revision=REVISION,seed=seed,steps=30,views=6,
        seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),strict_missing=result.missing_keys,strict_unexpected=result.unexpected_keys,
        weights_sha256=digest(WEIGHTS/'unet/diffusion_pytorch_model.safetensors'),base_weights_modified=False,
        geometry_source_sha256=digest(BASE/'shape.pt'),
        limitations=['Shape fixed to supplied geometry; paint cannot repair geometry.','Delighting omitted; reference illumination may be baked in.','Portable rasterizer replaces CUDA extension.','Generated texture, not verified hidden appearance or motion.'])
    (ROOT/'generation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


@torch.no_grad()
def bake(sigma=.8, fusion='mean', texture_root=None, diagnostics=True):
    from scipy.spatial import cKDTree
    from .hunyuan_appearance import sample_surface
    if (ROOT/'gaussian_fitted.pt').exists():raise FileExistsError('Preserve baked asset')
    if not 0 < sigma <= 2: raise ValueError('Invalid sigma')
    if fusion not in ('mean','winner'): raise ValueError('Invalid fusion')
    texture_root=Path(texture_root) if texture_root else ROOT
    ROOT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    mesh=load_verified(BASE/'shape.pt'); a=sample_surface(mesh['vertices'].numpy(),mesh['faces'].numpy(),400000)
    # Denser surface with overlap reduces pinholes; no arbitrary sharpening.
    a['covariance']*= (sigma/.62)**2
    control=load_verified(texture_root/'controls.pt'); p=a['position'].cuda(); v=p.clone();v[:,[0,1]]=-v[:,[0,1]];v=v[:,[0,2,1]]
    v=(v-control['center'].cuda())*(1.15/control['diameter'].cuda())
    n=a['normal'].cuda().clone();n[:,[0,1]]=-n[:,[0,1]];n=n[:,[0,2,1]]
    total=torch.zeros(len(p),device='cuda'); colour=torch.zeros_like(p); seen=torch.zeros(len(p),6,dtype=torch.bool,device='cuda')
    best=torch.zeros_like(total);winner=torch.zeros_like(colour)
    for i,record in enumerate(control['views']):
        e,az=record['elevation'],record['azimuth']; screen=screen_position(v,e,az,512)
        pixel=screen[:,:2].floor().long().clamp(0,511); depth=record['depth'].cuda()[pixel[:,1],pixel[:,0]]
        visible=(depth>0)&((depth-screen[:,2]).abs()<.008)&(screen[:,:2]>=0).all(1)&(screen[:,:2]<512).all(1)
        eye,_=camera(e,az,p.device); direction=F.normalize(eye,dim=0)
        weight=(n@direction).abs().pow(8)*visible;seen[:,i]=visible
        rgb=torch.tensor(np.asarray(Image.open(texture_root/f'generated_{i}.png').convert('RGB')).copy(),device='cuda',dtype=torch.float32)/255
        grid=(screen[:,:2]-.5)/511*2-1
        sampled=F.grid_sample(rgb.permute(2,0,1)[None],grid[None,None],mode='bilinear',padding_mode='border',align_corners=True)[0,:,0].T
        better=weight>best;winner[better]=sampled[better];best=torch.maximum(best,weight)
        total+=weight;colour+=weight[:,None]*sampled
    covered=total>1e-6; colour/=total.clamp_min(1e-6)[:,None]
    if fusion=='winner':colour=winner
    if not covered.any():raise ValueError('No visibility-supported colors')
    fallback=(~covered).cpu().numpy(); values=colour.cpu().numpy()
    if fallback.any():
        nearest=cKDTree(a['position'][covered.cpu()].numpy()).query(a['position'][~covered.cpu()].numpy())[1]
        values[fallback]=values[~fallback][nearest]
    a['colour']=torch.tensor(values);a['texture_view_visibility']=seen.cpu();a['texture_interpolated']=torch.tensor(fallback)
    a['original_count']=len(p);a['camera_pivot']=((p.amin(0)+p.amax(0))/2).tolist();a['camera_distance']=float((p.amax(0)-p.amin(0)).max())*1.65
    a['scope']='Hunyuan joint six-view generated texture baked into persistent surface Gaussians; geometry inferred; static, no base weight modification.'
    save_inference_checkpoint(a,ROOT/'gaussian_fitted.pt')
    report=dict(gaussians=len(p),direct_multiview_coverage=float(covered.float().mean()),interpolated_count=int(fallback.sum()),
        geometry_source_sha256=digest(BASE/'shape.pt'),texture_generation_source=str(texture_root/'generation_report.json'),accepted=False,
        sigma=sigma,fusion=fusion,seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Generated multiview appearance, not ground truth or Gaussian-native motion. Remaining occluded points nearest-color interpolation.')
    (ROOT/'bake_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    if diagnostics:inspect()


@torch.no_grad()
def inspect():
    a=load_verified(ROOT/'gaussian_fitted.pt');p=a['position'].cuda()
    from . import hunyuan_appearance as inspect_module
    inspect_module.ROOT=ROOT
    inspect_module.views(a,'appearance')
    from .gaussian3d import render,project
    # Explicit diagnostic crop for this +X-facing cat, not general head detection.
    lo,hi=p.amin(0),p.amax(0);head=p[(p[:,0]>lo[0]+.76*(hi[0]-lo[0]))&(p[:,1]>lo[1]+.55*(hi[1]-lo[1]))]
    focus=(head.amin(0)+head.amax(0))/2
    face_distance=float((head.amax(0)-head.amin(0)).max())*1.85
    for name,direction in [('face_front',[1.,.05,0.]),('face_oblique',[1.,.1,.55])]:
        eye=focus+F.normalize(p.new_tensor(direction),dim=0)*face_distance
        cov=a['covariance'].cuda();mean,screen_cov,depth,_,_=project(p,cov,eye,focus,768,768)
        mid=(screen_cov[:,0,0]+screen_cov[:,1,1])/2
        delta=(((screen_cov[:,0,0]-screen_cov[:,1,1])/2).square()+screen_cov[:,0,1].square()).sqrt()
        margin=3*(mid+delta).clamp_min(0).sqrt()
        keep=(depth>.02)&(mean[:,0]+margin>=0)&(mean[:,0]-margin<768)&(mean[:,1]+margin>=0)&(mean[:,1]-margin<768)
        picture,_=render(p[keep],cov[keep],a['colour'].cuda()[keep],a['opacity'].cuda()[keep],eye,focus,768,768,ground=False,radius=7)
        Image.fromarray(np.uint8(picture.clamp(0,1).cpu().numpy()*255)).save(ROOT/f'{name}.png')
    center=torch.tensor(a['camera_pivot']);save_inference_checkpoint(dict(eye=center+torch.tensor([0.,0.,a['camera_distance']]),target=center,fov=42.),ROOT/'camera.pt')
    inspect_module.inspect()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['controls','generate','bake','inspect']);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--sigma',type=float,default=.8);parser.add_argument('--fusion',choices=['mean','winner'],default='mean');parser.add_argument('--texture-root',type=Path)
    args=parser.parse_args();ROOT=args.root;BASE=args.base
    torch.set_num_threads(4)
    with keep_windows_awake():
        if args.stage=='bake':bake(args.sigma,args.fusion,args.texture_root)
        else:globals()[args.stage]()
