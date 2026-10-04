"""Experimental image-conditioned full object + six-direction environment.

Existing pretrained weights, not a trained Gaussian-native generalizer.
Environment is an inferred radial depth surface with an assumed ground plane.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
from .checkpoint_io import digest, load_verified, save_inference_checkpoint, keep_windows_awake

ROOT=Path('artifacts/real_video/enclosing_scene/v1')
SOURCE=Path('artifacts/real_video/prompt_generation/orange_cat_seed_531002/frame_000.png')
VIEWS={'front':([0,0,-1],[0,1,0]),'right':([1,0,0],[0,1,0]),
       'back':([0,0,1],[0,1,0]),'left':([-1,0,0],[0,1,0]),
       'up':([0,1,0],[0,0,1]),'down':([0,-1,0],[0,0,-1])}

def rays(size, fov, direction, up):
    forward=np.asarray(direction,dtype=float); forward/=np.linalg.norm(forward)
    right=np.cross(forward,up); right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    v=(np.arange(size)+.5)/size*2-1
    x,y=np.meshgrid(v,v); s=math.tan(math.radians(fov)/2)
    result=forward+x[...,None]*s*right-y[...,None]*s*up
    return result/np.linalg.norm(result,axis=-1,keepdims=True)

def pano_directions(height=768):
    lon=(np.arange(height*2)+.5)/(height*2)*2*np.pi-np.pi
    lat=np.pi/2-(np.arange(height)+.5)/height*np.pi
    x,y=np.meshgrid(lon,lat)
    return np.stack([np.cos(y)*np.sin(x),np.sin(y),-np.cos(y)*np.cos(x)],-1)

def sample_pano(pano, directions):
    h,w=pano.shape[:2]
    lon=np.arctan2(directions[...,0],-directions[...,2]); lat=np.arcsin(np.clip(directions[...,1],-1,1))
    x=((lon+np.pi)/(2*np.pi)*w-.5).astype(np.float32)
    y=((np.pi/2-lat)/np.pi*h-.5).astype(np.float32)
    return cv2.remap(pano,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)

def integrate(pano, known, image, directions, fov, forward, up, validity=None):
    forward=np.asarray(forward,float); right=np.cross(forward,up); right/=np.linalg.norm(right); up=np.cross(right,forward)
    z=directions@forward; s=math.tan(math.radians(fov)/2)
    u=(directions@right)/np.maximum(z,1e-8)/s
    v=-(directions@up)/np.maximum(z,1e-8)/s
    n=image.shape[0]; x=((u+1)*n/2-.5).astype(np.float32); y=((v+1)*n/2-.5).astype(np.float32)
    valid=(z>0)&(abs(u)<.98)&(abs(v)<.98)&(~known)
    if validity is not None:
        valid &= cv2.remap(validity.astype(np.uint8),x,y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
    values=cv2.remap(image,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    pano[valid]=values[valid]; known[valid]=True

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'mask.png').exists(): raise FileExistsError('Preserve input mask')
    rgb=np.asarray(Image.open(SOURCE).convert('RGB')); hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    warm=(hsv[...,0]<30)&(hsv[...,1]>50)&(rgb[...,0]>85)
    warm[:30]=False; warm[-12:]=False
    labels,n=ndimage.label(warm); counts=np.bincount(labels.ravel()); counts[0]=0
    keep=np.where(counts>70)[0]; seed=np.isin(labels,keep)
    possible=ndimage.binary_dilation(seed,iterations=14)
    marks=np.zeros(rgb.shape[:2],np.uint8);marks[possible]=cv2.GC_PR_BGD;marks[seed]=cv2.GC_PR_FGD
    marks[ndimage.binary_erosion(seed,iterations=3)]=cv2.GC_FGD
    cv2.setRNGSeed(531010);cv2.grabCut(rgb,marks,None,np.zeros((1,65)),np.zeros((1,65)),6,cv2.GC_INIT_WITH_MASK)
    mask=ndimage.binary_fill_holes((marks==cv2.GC_FGD)|(marks==cv2.GC_PR_FGD))
    Image.fromarray(np.uint8(mask)*255).save(ROOT/'mask.png')
    rgba=np.dstack([rgb,np.uint8(mask)*255]);Image.fromarray(rgba).save(ROOT/'segmented.png')
    print(json.dumps(dict(stage='prepared',source=str(SOURCE),mask_pixels=int(mask.sum()))),flush=True)

def cat():
    from .infer_3d_cat import main
    main(SOURCE,ROOT/'mask.png',ROOT/'cat',count=90000,resolution=192)

@torch.inference_mode()
def environment():
    from diffusers import StableDiffusionInpaintPipeline
    out=ROOT/'environment';out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();torch.set_num_threads(4)
    local=Path('../../work/background_inpainting/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb')
    pipe=StableDiffusionInpaintPipeline.from_pretrained(local,torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True,
        safety_checker=None,feature_extractor=None,requires_safety_checker=False).to('cuda')
    pano=np.zeros((768,1536,3),np.uint8);known=np.zeros((768,1536),bool);directions=pano_directions()
    # Square perspective crop of known background anchors the panorama; cat pixels are unknown.
    rgb=np.asarray(Image.open(SOURCE).convert('RGB'));mask=np.asarray(Image.open(ROOT/'mask.png'))>0
    mask=ndimage.binary_dilation(mask,iterations=18)
    h,w=rgb.shape[:2];x0=(w-h)//2
    integrate(pano,known,rgb[:,x0:x0+h],directions,42,[0,0,-1],[0,1,0],~mask[:,x0:x0+h])
    records=[]
    # Progressive overlapping perspective generation: shared angular memory, not independent six pictures.
    for index,(name,(forward,up)) in enumerate(VIEWS.items()):
        rr=rays(512,110,forward,up);initial=sample_pano(pano,rr)
        valid=sample_pano(known.astype(np.float32),rr)>.999
        initial[~valid]=127
        prompt={'up':'Photograph looking straight up into a blue sky with soft white clouds, above a quiet green garden, daylight, no objects.',
                'down':'Photograph looking straight down at continuous dense green lawn grass underfoot, natural small grass blades, daylight, no objects.'}.get(name,
                'Photograph of an empty quiet garden, continuous green lawn across lower half, distant leafy shrubs and trees, blue sky above, natural daylight, eye level landscape, no animals.')
        print(json.dumps(dict(stage='environment_diffusion',face=name,known_fraction=float(valid.mean()))),flush=True)
        generated=np.asarray(pipe(prompt=prompt,negative_prompt='animal, cat, dog, person, legs, body, orange fur, text, watermark, frame, collage',
            image=Image.fromarray(initial),mask_image=Image.fromarray(np.uint8(~valid)*255),height=512,width=512,num_inference_steps=30,
            guidance_scale=6.5,generator=torch.Generator('cuda').manual_seed(531010+index)).images[0])
        completed=np.where(valid[...,None],initial,generated)
        Image.fromarray(completed).save(out/f'{name}_generated.png')
        integrate(pano,known,completed,directions,110,forward,up)
        records.append(dict(face=name,seed=531010+index,prompt=prompt,known_fraction=float(valid.mean())))
        Image.fromarray(pano).save(out/'panorama.png')
    del pipe;gc.collect();torch.cuda.empty_cache()
    coverage=float(known.mean())
    if coverage<.999: raise ValueError(f'Incomplete panorama coverage: {coverage}')
    from transformers import AutoModelForDepthEstimation
    from transformers.models.dpt.image_processing_pil_dpt import DPTImageProcessorPil
    depth_dir=Path('../../work/detail_lift_depth/5426e4f0f36572d16453bbda7a8389317b1bef99')
    processor=DPTImageProcessorPil.from_pretrained(depth_dir,local_files_only=True)
    model=AutoModelForDepthEstimation.from_pretrained(depth_dir,local_files_only=True).cuda().eval()
    distances=np.zeros(known.shape,np.float32); weights=np.zeros(known.shape,np.float32)
    # Blend relative inverse-depth evidence from overlapping views into one spherical field.
    for name,(forward,up) in VIEWS.items():
        rgb=sample_pano(pano,rays(512,110,forward,up));Image.fromarray(rgb).save(out/f'{name}_consistent.png')
        pred=model(**processor(images=Image.fromarray(rgb),return_tensors='pt').to('cuda')).predicted_depth
        pred=torch.nn.functional.interpolate(pred[:,None],size=(512,512),mode='bicubic',align_corners=False)[0,0].cpu().numpy()
        lo,hi=np.quantile(pred,[.02,.98]);depth=5+7*(1-np.clip((pred-lo)/max(hi-lo,1e-6),0,1))
        f=np.array(forward,float);r=np.cross(f,up);u=np.cross(r,f);z=directions@f
        xx=(directions@r)/np.maximum(z,1e-6)/math.tan(math.radians(55)); yy=-(directions@u)/np.maximum(z,1e-6)/math.tan(math.radians(55))
        good=(z>0)&(abs(xx)<1)&(abs(yy)<1); wt=np.where(good,np.maximum(0,1-np.maximum(abs(xx),abs(yy)))**2,0)
        dd=cv2.remap(depth.astype(np.float32),((xx+1)*256-.5).astype(np.float32),((yy+1)*256-.5).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        distances+=dd*wt;weights+=wt
    depth=distances/np.maximum(weights,1e-8)
    np.save(out/'radial_depth.npy',depth)
    report=dict(model='stable-diffusion-v1-5/stable-diffusion-inpainting',revision=local.name,views=records,panorama_resolution=[1536,768],coverage=coverage,
        seconds=time.perf_counter()-start,depth_model='DepthAnythingV2Small',depth_units='Assumed 5..12 scene units, per-view relative depth blended angularly; not metric.',
        limitations=['Sequential diffusion is not jointly trained multiview generation; seams/semantic inconsistency possible.', 'Shared spherical depth is single-layer; not complete free-roaming geometry.', 'Ground plane imposed later; no ground-contact inference.'])
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

def assemble():
    asset=load_verified(ROOT/'cat/cat_asset.pt');p=asset['position'];center=(p.amin(0)+p.amax(0))*.5
    p=p-center; p[:,1]-=p[:,1].min()
    cov=(asset['frame']*asset['scale'][:,None].square())@asset['frame'].transpose(-1,-2)
    pano=np.asarray(Image.open(ROOT/'environment/panorama.png'));dep=np.load(ROOT/'environment/radial_depth.npy')
    envp=[];envc=[];envcov=[];faceid=[]
    origin=np.array([0.,.8,0.]);n=128
    for index,(name,(forward,up)) in enumerate(VIEWS.items()):
        rr=rays(n,90,forward,up);distance=sample_pano(dep,rr)
        # Explicit ground-plane prior under the object, not learned ground geometry.
        floor=rr[...,1]<-.08;distance=np.where(floor,np.minimum(distance,.8/np.maximum(-rr[...,1],1e-6)),distance)
        xyz=origin+rr*distance[...,None]; sigma=distance*2/n*.8
        outer=rr[..., :,None]*rr[...,None,:]
        cc=sigma[...,None,None]**2*(np.eye(3)-.96*outer)
        envp.append(xyz.reshape(-1,3));envcov.append(cc.reshape(-1,3,3));envc.append(sample_pano(pano,rr).reshape(-1,3)/255);faceid.append(np.full(n*n,index))
    envp=torch.tensor(np.concatenate(envp),dtype=torch.float32)
    result=dict(position=torch.cat([p,envp]),covariance=torch.cat([cov,torch.tensor(np.concatenate(envcov),dtype=torch.float32)]),
        colour=torch.cat([asset['colour'],torch.tensor(np.concatenate(envc),dtype=torch.float32)]),
        opacity=torch.cat([asset['opacity'],torch.full((len(envp),),.995)]),original_count=len(p),background_count=len(envp),
        group=torch.cat([torch.zeros(len(p),dtype=torch.int16),torch.ones(len(envp),dtype=torch.int16)]),
        environment_face=torch.tensor(np.concatenate(faceid),dtype=torch.int16),
        scope='TripoSR inferred full object plus generated depth-shell environment; static, unverified, not trained Gaussian-native model',
        camera_pivot=[0,float(p[:,1].max())*.5,0],camera_distance=3.5)
    save_inference_checkpoint(result,ROOT/'scene.pt')
    report=dict(cat_gaussians=len(p),environment_gaussians=len(envp),cat_extents=(p.amax(0)-p.amin(0)).tolist(),
        finite=all(torch.isfinite(result[k]).all().item() for k in ['position','covariance','colour','opacity']),
        minimum_covariance_eigenvalue=float(torch.linalg.eigvalsh(result['covariance']).min()),accepted=False,
        cat_source_sha256=digest(ROOT/'cat/cat_asset.pt'),environment_sha256=digest(ROOT/'environment/panorama.png'))
    (ROOT/'assembly_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','cat','environment','assemble']);args=parser.parse_args()
    with keep_windows_awake(): globals()[args.stage]()
