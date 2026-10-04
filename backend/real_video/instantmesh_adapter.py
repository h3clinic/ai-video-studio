"""Local InstantMesh inference: staged models, chunked decoding, Gaussian export.

No base weights trained. Vertex-colored extraction avoids optional CUDA rasterizer.
Upstream source is reviewed and locally patched; patch hashes are recorded.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import numpy as np
from PIL import Image
import torch
from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake

REPO=Path('work/InstantMesh').resolve()
REVISION='b785b4ecfb6636ef34a08c748f96f6a5686244d0'
CODE='08822c52fdc399b93ea00e4fa9e596344ed52ccc'
ROOT=Path('artifacts/real_video/instantmesh/v1')

def regular_grid(self,res):
    """Same lexicographic grid and corner order without duplicate-vertex unique."""
    if not isinstance(res,int) or res<1:raise ValueError('Positive cubic resolution required')
    v=torch.stack(torch.meshgrid(*[torch.arange(res+1,device=self.device)]*3,indexing='ij'),-1).reshape(-1,3)
    p=torch.round(v.float()/res*1e5)/1e5-.5
    c=torch.stack(torch.meshgrid(*[torch.arange(res,device=self.device)]*3,indexing='ij'),-1).reshape(-1,3)
    corners=c[:,None]+self.cube_corners.long()[None]
    indices=(corners[...,0]*(res+1)+corners[...,1])*(res+1)+corners[...,2]
    return p,indices

def chunk_geometry(self,planes,coords,indices,chunk=8192):
    from src.models.renderer.utils.renderer import sample_from_planes
    b,n,_=coords.shape;c=planes.shape[2]*3;axes=self.plane_axes.to(planes.device)
    features=torch.empty((b,n,c),device=planes.device,dtype=planes.dtype)
    for start in range(0,n,chunk):
        f=sample_from_planes(axes,planes,coords[:,start:start+chunk],padding_mode='zeros',box_warp=self.rendering_kwargs['box_warp'])
        features[:,start:start+chunk]=f.permute(0,2,1,3).reshape(b,-1,c)
    sdf=[];deformation=[];weight=[]
    for start in range(0,n,chunk):
        f=features[:,start:start+chunk];sdf.append(self.decoder.net_sdf(f));deformation.append(self.decoder.net_deformation(f))
    for start in range(0,len(indices),chunk):
        ix=indices[start:start+chunk];f=features[:,ix.reshape(-1)].reshape(b,len(ix),8*c)
        weight.append(self.decoder.net_weight(f)*.1)
    return torch.cat(sdf,1).float(),torch.cat(deformation,1).float(),torch.cat(weight,1).float()

def export_gaussians(vertices,faces,colours,count):
    from .hunyuan_appearance import sample_surface
    # Canonical InstantMesh Z-up -> our Gaussian Y-up; proper rotation, not mirror.
    vertices=vertices[:,[0,2,1]].copy();vertices[:,2]*=-1
    a=sample_surface(vertices,faces,count)
    a['colour']=(torch.tensor(colours,dtype=torch.float32)[torch.tensor(faces)[a['face_id']]]*a['barycentric'][...,None]).sum(1)
    center=(a['position'].amin(0)+a['position'].amax(0))/2
    a.update(original_count=count,camera_pivot=center.tolist(),camera_distance=float((a['position'].amax(0)-a['position'].amin(0)).max())*1.9,
        scope='Frozen InstantMesh geometry and vertex appearance, sampled into surface-bound Gaussians. Static inferred object, no motion or base weight training.')
    return a,vertices

def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    report_path=args.output/f'{args.stage}_report.json'
    if report_path.exists():raise FileExistsError('Preserve stage report; use a new output or stage')
    if subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()!=CODE:raise RuntimeError('Unexpected upstream revision')
    sys.path.insert(0,str(REPO));torch.set_num_threads(4);torch.manual_seed(531031);np.random.seed(531031)
    report=dict(stage=args.stage,status='running',accepted=False,code_revision=CODE,weights_revision=REVISION,
        weights_trained=False,scope='Static image-conditioned mesh reconstruction then Gaussian sampling; not motion generation.',
        source_sha256=digest(args.source),source=str(args.source))
    report['command_arguments']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    def save():report_path.write_text(json.dumps(report,indent=2))
    save();start=time.perf_counter()
    try:
        if args.stage=='views':
            from .zero123_views import MODEL,split_views,prepare_input
            from diffusers import EulerAncestralDiscreteScheduler
            spec=importlib.util.spec_from_file_location('instantmesh_zp',REPO/'zero123plus/pipeline.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            image=Image.open(args.source).convert('RGBA')
            if args.head:image=prepare_input(image,[460,75,650,290])
            else:
                alpha=np.asarray(image.getchannel('A'));y,x=np.where(alpha>0)
                image=prepare_input(image,[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)])
            image.save(args.output/'conditioning.png')
            report['conditioning_sha256']=digest(args.output/'conditioning.png')
            report['manual_head_roi']=[460,75,650,290] if args.head else None
            pipeline=module.Zero123PlusPipeline.from_pretrained(str(MODEL),torch_dtype=torch.float16,use_safetensors=True,local_files_only=True)
            weights=REPO/'ckpts/diffusion_pytorch_model.bin'
            state=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
            pipeline.unet.load_state_dict(state,strict=True);del state
            pipeline.scheduler=EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config,timestep_spacing='trailing')
            pipeline.to('cuda');torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            result=pipeline(image,num_inference_steps=75,generator=torch.Generator('cuda').manual_seed(531031)).images[0]
            torch.cuda.synchronize();report['inference_seconds']=time.perf_counter()-t
            report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();report['weights_sha256']=digest(weights)
            result.save(args.output/'six_views.png')
            for i,v in enumerate(split_views(result)):v.save(args.output/f'view_{i}.png')
        elif args.stage=='mesh':
            from .zero123_views import split_views
            from src.models.lrm_mesh import InstantMesh
            from src.models.encoder.dino_wrapper import DinoWrapper
            from src.models.encoder.dino import ViTModel
            from src.models.geometry.rep_3d.flexicubes import FlexiCubes
            from src.utils.camera_util import get_zero123plus_input_cameras
            from transformers import ViTConfig,ViTImageProcessor
            # Full InstantMesh checkpoint includes DINO. Initialize architecture only;
            # strict loading below must cover every parameter (no random fallback).
            def build_dino(name):
                return ViTModel(ViTConfig(hidden_size=768,num_hidden_layers=12,num_attention_heads=12,intermediate_size=3072,patch_size=16),add_pooling_layer=False),ViTImageProcessor(image_mean=[.485,.456,.406],image_std=[.229,.224,.225])
            DinoWrapper._build_dino=staticmethod(build_dino)
            FlexiCubes.construct_voxel_grid=regular_grid
            model=InstantMesh(grid_res=128,grid_scale=2.1)
            weights=REPO/'ckpts/instant_mesh_large.ckpt'
            state=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)['state_dict']
            state={k[14:]:v for k,v in state.items() if k.startswith('lrm_generator.')}
            model.load_state_dict(state,strict=True);del state;gc.collect()
            model=model.eval().cuda();model.init_flexicubes_geometry('cuda',fovy=30,extraction_only=True)
            model.synthesizer.get_geometry_prediction=types.MethodType(chunk_geometry,model.synthesizer)
            images=torch.stack([torch.tensor(np.asarray(v).copy()).permute(2,0,1).float()/255 for v in split_views(Image.open(args.output/'six_views.png').convert('RGB'))])[None].cuda()
            cameras=get_zero123plus_input_cameras(1,radius=4).cuda()
            torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                planes=model.forward_planes(images,cameras)
                vertices,faces,colours=model.extract_mesh(planes,use_texture_map=False)
            torch.cuda.synchronize();report['inference_seconds']=time.perf_counter()-t
            report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated();report['weights_sha256']=digest(weights)
            report.update(vertices=len(vertices),faces=len(faces),mixed_precision=True,grid_resolution=128)
            a,rotated=export_gaussians(vertices,faces,colours.astype(np.float32)/255,args.count)
            # Query the learned color field at each splat center rather than
            # interpolating quantized vertex colors (which can add avoidable blur).
            original_xyz=a['position'][:,[0,2,1]].clone();original_xyz[:,1]*=-1
            splat_colours=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                for offset in range(0,args.count,8192):
                    rgb=model.synthesizer.get_texture_prediction(planes,original_xyz[offset:offset+8192][None].cuda())
                    splat_colours.append(rgb[0].clamp(0,1).float().cpu())
            a['colour']=torch.cat(splat_colours)
            a['scope']+=' Colors queried directly from learned InstantMesh texture field, not vertex RGB interpolation.'
            report['gaussian_appearance']='Direct learned texture-field queries at Gaussian centers'
            report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
            save_inference_checkpoint(dict(vertices=torch.tensor(rotated),faces=torch.tensor(faces),vertex_colours=torch.tensor(colours)),args.output/'shape.pt')
            save_inference_checkpoint(a,args.output/'gaussian_fitted.pt')
            report['gaussians']=args.count
        elif args.stage=='inspect':
            from . import hunyuan_appearance as viewer
            a=load_verified(args.output/'gaussian_fitted.pt');viewer.ROOT=args.output
            viewer.views(a,'appearance');center=torch.tensor(a['camera_pivot'])
            save_inference_checkpoint(dict(eye=center+torch.tensor([0.,0.,a['camera_distance']]),target=center,fov=42.),args.output/'camera.pt')
            viewer.inspect()
        report.update(status='completed_pending_visual_review',elapsed_seconds=time.perf_counter()-start)
        report['source_patch']=subprocess.check_output(['git','-C',str(REPO),'diff'],text=True)
        save();print(json.dumps({k:v for k,v in report.items() if k!='source_patch'}),flush=True)
    except Exception as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}');save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['views','mesh','inspect']);p.add_argument('--head',action='store_true')
    p.add_argument('--source',type=Path,default=Path('artifacts/real_video/hunyuan_gaussian/v4_full/conditioning.png'))
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--count',type=int,default=400000)
    with keep_windows_awake():main(p.parse_args())
