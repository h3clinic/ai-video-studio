"""Remote-only development trial: learned 3D apple, depth-lifted observed video.

Not new motion generation, a complete reconstructed scene, or a trained agent.
All final pixels are Gaussian-rendered; no RGB image is composited over video.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from real_video.gaussian3d import render, project

ROOT = Path('/workspace/object_edit')
OUT = ROOT/'output'
W,H,T = 512,288,40
FOV=42.
FIELDS=('position','covariance','colour','opacity')
PROMPT='a single ripe red apple with a short brown stem, realistic fruit'


def save_report(report):
    (OUT/'report.json').write_text(json.dumps(report,indent=2))


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def raster(a, h=H,w=W,fov=FOV,principal=None):
    return render(*(a[k].cuda() for k in FIELDS),torch.zeros(3,device='cuda'),
        torch.tensor([0.,0.,1.],device='cuda'),height=h,width=w,fov=fov,
        radius=2,ground=False,principal=principal)[0]


def main():
    if not torch.cuda.is_available(): raise RuntimeError('Remote CUDA required')
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(6)
    cv2.setNumThreads(2)
    torch.manual_seed(104003)
    np.random.seed(104003)
    report=dict(status='model_loading',prompt=PROMPT,seed=104003,accepted=False,
        classification='Local 3D object insertion into depth-lifted observed video; NOT new motion',
        source_sha256=hashlib.sha256((ROOT/'source.mp4').read_bytes()).hexdigest(),
        stages={},frames=T,resolution=[W,H],gpu=torch.cuda.get_device_name(),
        limitations=['Manual fruit ROI; orange peel outside it remains.',
        'Scene is single-view relative-depth lifting, not recovered hidden geometry.',
        'Per-frame scene samples are observed state, not verified persistent material tracks.',
        'Pretrained Shap-E generates one apple; no model weights trained in this trial.',
        'No newly generated donkey action, contact, shadows or disocclusion repair.',
        'Hessian-guided fitting is not implemented in this trial.'])
    save_report(report)
    pos,rgb=generate_or_reuse_apple(report)
    report['status']='writing_observed_gaussian_scene';save_report(report)
    run_scene(report,pos,rgb)


def generate_or_reuse_apple(report):
    cached=ROOT/'input_apple.npz'
    if cached.exists():
        import shutil
        data=np.load(cached,allow_pickle=False)
        pos,rgb=data['position'],data['colour']
        if pos.shape!=(20000,3) or rgb.shape!=pos.shape or not np.isfinite(pos).all():
            raise ValueError('Invalid reused apple asset')
        shutil.copyfile(cached,OUT/'apple_canonical.npz')
        report['reused_apple_sha256']=hashlib.sha256(cached.read_bytes()).hexdigest()
        report['stages']['object_generation_seconds']=0
        report['object_generation_note']='Reused exact learned mesh samples from preserved v1 trial; prior generation cost excluded here, not free.'
        return pos,rgb
    from shap_e.models.download import load_model,load_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.util.notebooks import decode_latent_mesh
    start=sync_time()
    xm=load_model('transmitter',device=torch.device('cuda'))
    model=load_model('text300M',device=torch.device('cuda'))
    diffusion=diffusion_from_config(load_config('diffusion'))
    report['stages']['object_model_load_download_seconds']=sync_time()-start
    report['status']='generating_3d_apple';save_report(report)
    start=sync_time()
    with torch.no_grad():
        latent=sample_latents(batch_size=1,model=model,diffusion=diffusion,
            guidance_scale=15.,model_kwargs=dict(texts=[PROMPT]),progress=True,
            clip_denoised=True,use_fp16=True,use_karras=True,karras_steps=64,
            sigma_min=1e-3,sigma_max=160,s_churn=0)[0]
    report['stages']['object_generation_seconds']=sync_time()-start
    start=sync_time()
    with torch.no_grad(): mesh=decode_latent_mesh(xm,latent).tri_mesh()
    report['stages']['mesh_decode_seconds']=sync_time()-start
    verts=np.asarray(mesh.verts,dtype=np.float32)
    faces=np.asarray(mesh.faces)
    colors=np.stack([mesh.vertex_channels[c] for c in ('R','G','B')],axis=-1).astype(np.float32)
    if colors.max()>1: colors/=255.
    # Area-weighted surface samples, not a procedural fruit primitive.
    tri=verts[faces];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    chosen=np.random.choice(len(faces),20000,p=areas/areas.sum())
    uv=np.random.random((20000,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    bary=np.column_stack((1-uv.sum(1),uv))
    pos=(tri[chosen]*bary[:,:,None]).sum(1)
    rgb=(colors[faces[chosen]]*bary[:,:,None]).sum(1)
    # Shap-E is Z-up; put Z along our Y and retain a right-handed transform.
    pos=pos[:,[0,2,1]];pos[:,2]*=-1
    pos-=.5*(pos.min(0)+pos.max(0));pos/=np.ptp(pos,axis=0).max()
    np.savez_compressed(OUT/'apple_canonical.npz',position=pos.astype('float32'),colour=rgb.astype('float32'),
        faces=faces,vertices=verts,vertex_colour=colors,latent=latent.float().cpu().numpy())
    del xm,model,diffusion,latent
    torch.cuda.empty_cache()
    return pos,rgb


def run_scene(report,pos,rgb):
    from transformers import DepthAnythingForDepthEstimation,DPTImageProcessor,DepthAnythingConfig
    from huggingface_hub import hf_hub_download
    start=sync_time()
    # Explicit official files + config avoid the failed AutoConfig resolution.
    config=json.loads((ROOT/'depth_config.json').read_text())
    processor=DPTImageProcessor.from_dict(json.loads((ROOT/'depth_preprocessor.json').read_text()))
    weights=hf_hub_download('depth-anything/Depth-Anything-V2-Small-hf','model.safetensors',revision=json.loads((ROOT/'depth_revision.json').read_text())['sha'])
    from safetensors.torch import load_file
    depth_model=DepthAnythingForDepthEstimation(DepthAnythingConfig.from_dict(config)).cuda().eval()
    depth_model.load_state_dict(load_file(weights),strict=True)
    report['stages']['depth_model_load_download_seconds']=sync_time()-start
    cap=cv2.VideoCapture(str(ROOT/'source.mp4'))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    duration=count/fps
    focal=.5*H/math.tan(math.radians(FOV)*.5)
    yy,xx=np.mgrid[:H,:W].astype('float32')
    indices=np.floor(np.arange(T)*count/T).astype(int)
    stats=[];before_frames=[];after_frames=[]
    (OUT/'state').mkdir()
    # ROI covers main peeled fruit only, excludes loose pieces and donkey muzzle.
    roi=(xx-350.)**2/33.**2+(yy-214.)**2/30.**2<1
    torch.cuda.reset_peak_memory_stats()
    for fi,index in enumerate(indices):
        start=sync_time();cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,bgr=cap.read()
        if not ok: raise RuntimeError('Source decode failure')
        im=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(W,H),interpolation=cv2.INTER_AREA)
        prediction=depth_model(**{k:v.cuda() for k,v in processor(images=Image.fromarray(im),return_tensors='pt').items()}).predicted_depth
        d=torch.nn.functional.interpolate(prediction.reshape(1,1,*prediction.shape[-2:]),size=(H,W),mode='bicubic',align_corners=False)[0,0].cpu().numpy()
        lo,hi=np.percentile(d,[2,98]);z=2.+2.*(1-np.clip((d-lo)/max(hi-lo,1e-6),0,1))
        xyz=np.stack((-(xx+.5-W/2)*z/focal,-(yy+.5-H/2)*z/focal,z),axis=-1).reshape(-1,3)
        sigma=(.38*z/focal).reshape(-1)
        cov=np.zeros((H*W,3,3),dtype='float32');cov[:,0,0]=sigma**2;cov[:,1,1]=sigma**2;cov[:,2,2]=sigma**2*.1
        a=dict(position=torch.from_numpy(xyz.astype('float32')),covariance=torch.from_numpy(cov),
            colour=torch.from_numpy(im.reshape(-1,3).astype('float32')/255),opacity=torch.ones(H*W)*.995)
        # Save actual per-frame Gaussian fields, including immutable non-fruit state.
        np.savez_compressed(OUT/'state'/f'{fi:03d}.npz',**{k:v.numpy() for k,v in a.items()},object_mask=roi,source_index=index)
        write_seconds=sync_time()-start
        start=sync_time();base=raster(a);baseline_seconds=sync_time()-start
        # Apple occupies the same approximate image-space object extent, in shared XYZ.
        center_z=float(np.median(z[roi]))
        center=np.array([-(350.5-W/2)*center_z/focal,-(211.5-H/2)*center_z/focal,center_z],dtype='float32')
        scale=59*center_z/focal
        apple_p=pos.astype('float32')*scale+center
        apple_cov=np.tile(np.eye(3,dtype='float32')[None]*(scale*.010)**2,(len(pos),1,1))
        keep=~torch.from_numpy(roi.reshape(-1))
        edited={k:a[k][keep] for k in FIELDS}
        apple=dict(position=torch.from_numpy(apple_p),covariance=torch.from_numpy(apple_cov),
            colour=torch.from_numpy(rgb.astype('float32')),opacity=torch.ones(len(pos))*.9)
        edited={k:torch.cat((edited[k],apple[k])) for k in FIELDS}
        # One object-local generation is reused. Matching source scene state is never edited.
        start=sync_time();full=raster(edited);full_seconds=sync_time()-start
        start=sync_time()
        eye=torch.zeros(3);target=torch.tensor([0.,0.,1.])
        means,_,depth,_,_=project(edited['position'],edited['covariance'],eye,target,H,W,FOV)
        ap=means[-len(pos):]
        x0=max(0,min(315,int(ap[:,0].min())-3));x1=min(W,max(386,int(ap[:,0].max())+4))
        y0=max(0,min(181,int(ap[:,1].min())-3));y1=min(H,max(248,int(ap[:,1].max())+4))
        selected=(depth>.02)&(means[:,0].floor()+2>=x0)&(means[:,0].floor()-2<x1)&(means[:,1].floor()+2>=y0)&(means[:,1].floor()-2<y1)
        patch_fov=math.degrees(2*math.atan((y1-y0)/(2*focal)))
        patch=raster({k:v[selected] for k,v in edited.items()},y1-y0,x1-x0,patch_fov,(W/2-x0,H/2-y0))
        incremental=base.clone();incremental[y0:y1,x0:x1]=patch
        incremental_seconds=sync_time()-start
        err=float((incremental-full).abs().max())
        outside=torch.ones((H,W),dtype=torch.bool,device='cuda');outside[y0:y1,x0:x1]=False
        unchanged=bool(torch.equal(base[outside],incremental[outside]))
        before_frames.append((base.cpu().numpy().clip(0,1)*255).round().astype('uint8'))
        after_frames.append((incremental.cpu().numpy().clip(0,1)*255).round().astype('uint8'))
        stats.append(dict(frame=fi,scene_write_seconds=write_seconds,baseline_raster_seconds=baseline_seconds,
            edited_full_raster_seconds=full_seconds,incremental_seconds_including_scan_copy=incremental_seconds,
            full_vs_incremental_max_error=err,outside_exact=unchanged,dirty_pixels=(x1-x0)*(y1-y0),
            selected_gaussians=int(selected.sum()),total_gaussians=len(edited['position']),
            apple_center=center.tolist(),apple_scale=scale,apple_covariance_scale=.010,
            dirty_box=[x0,y0,x1,y1],
            nonfruit_state_exact=all(torch.equal(a[k][keep],edited[k][:-len(pos)]) for k in FIELDS)))
        report.update(completed_frames=fi+1,frame_metrics=stats);save_report(report)
    cap.release()
    start=sync_time()
    for name,frames in [('gaussian_before.mp4',before_frames),('gaussian_apple.mp4',after_frames)]:
        imageio.mimsave(OUT/name,frames,fps=T/duration,codec='libx264',macro_block_size=1)
    report['stages']['encoding_seconds']=sync_time()-start
    for i in (0,10,20,30,39):
        Image.fromarray(np.concatenate((before_frames[i],after_frames[i]),axis=1)).save(OUT/f'comparison_{i:03d}.png')
    # Three true perspective views of generated object, not used as action video.
    obj={k:v.cuda() for k,v in apple.items()};c=torch.from_numpy(center).cuda()
    views=[]
    for angle in (-.5,0,.5):
        eye=c+torch.tensor([math.sin(angle)*scale*2,0.,-math.cos(angle)*scale*2],device='cuda')
        pic=render(*(obj[k] for k in FIELDS),eye,c,height=256,width=256,radius=2,ground=False)[0]
        views.append((pic.cpu().numpy().clip(0,1)*255).round().astype('uint8'))
    Image.fromarray(np.concatenate(views,axis=1)).save(OUT/'apple_views.png')
    report.update(status='completed_visual_review_required',duration_seconds=duration,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),training_seconds=0,
        gaussian_native_new_motion=False,quality_matched_savings_proven=False,
        state_bytes=sum(p.stat().st_size for p in (OUT/'state').glob('*')),
        apple_bytes=(OUT/'apple_canonical.npz').stat().st_size)
    report['output_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='report.json'}
    save_report(report)


if __name__=='__main__':
    with torch.inference_mode(): main()
