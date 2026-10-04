"""Pinned Hunyuan3D geometry experiment with a Gaussian-output adaptation.

Hunyuan base weights are unchanged. Any later Gaussian appearance fitting is
asset-specific, not a trained generalizable replacement for its ShapeVAE.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from .checkpoint_io import digest, save_inference_checkpoint, keep_windows_awake

ROOT=Path('artifacts/real_video/hunyuan_gaussian/v1')
WORK=Path('../../work/real_video').resolve()
SOURCE=Path('artifacts/real_video/prompt_generation/orange_cat_seed_531002/paired_decoder_benchmark_v1/0_wan/frame_000.png')
WEIGHTS=WORK/'hunyuan_mini/hunyuan3d-dit-v2-mini'
REVISION='f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6'
MODEL='tencent/Hunyuan3D-2mini'

def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    rgb=np.asarray(Image.open(SOURCE).convert('RGB'))
    prior=np.asarray(Image.open('artifacts/real_video/enclosing_scene/v1/mask.png'))>0
    possible=ndimage.binary_dilation(prior,iterations=14)
    mark=np.zeros(prior.shape,np.uint8);mark[possible]=cv2.GC_PR_BGD;mark[prior]=cv2.GC_PR_FGD
    mark[ndimage.binary_erosion(prior,iterations=8)]=cv2.GC_FGD
    cv2.setRNGSeed(531020);cv2.grabCut(rgb,mark,None,np.zeros((1,65)),np.zeros((1,65)),7,cv2.GC_INIT_WITH_MASK)
    mask=(mark==cv2.GC_FGD)|(mark==cv2.GC_PR_FGD)
    Image.fromarray(np.uint8(mask)*255).save(ROOT/'mask.png')
    Image.fromarray(np.dstack([rgb,np.uint8(mask)*255])).save(ROOT/'conditioning.png')
    Image.fromarray(np.where(mask[...,None],rgb,255)).save(ROOT/'reference.png')
    (ROOT/'input_provenance.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=digest(SOURCE),
        scope='Same Wan latent, original VAE decoder instead of the blurry learned Gaussian decoder; reconstruction reference, not new generation'),indent=2))

@torch.inference_mode()
def shape():
    if (ROOT/'shape.pt').exists():raise FileExistsError('Preserve generated geometry')
    import yaml
    from safetensors import safe_open
    from accelerate import init_empty_weights
    sys.path.insert(0,str(WORK/'Hunyuan3D-2'))
    from hy3dgen.shapegen.pipelines import instantiate_from_config,Hunyuan3DDiTFlowMatchingPipeline
    cfg=yaml.safe_load((WEIGHTS/'config.yaml').read_text())
    allowed={'hy3dgen.shapegen.models.Hunyuan3DDiT','hy3dgen.shapegen.models.ShapeVAE',
        'hy3dgen.shapegen.models.SingleImageEncoder','hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler',
        'hy3dgen.shapegen.preprocessors.ImageProcessorV2','hy3dgen.shapegen.pipelines.Hunyuan3DDiTFlowMatchingPipeline'}
    if {v['target'] for v in cfg.values()}!=allowed:raise ValueError('Unreviewed config target')
    start=time.perf_counter();parts={};loads={}
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    with safe_open(WEIGHTS/'model.fp16.safetensors',framework='pt',device='cpu') as sf:
        for name in ['model','vae','conditioner']:
            print(json.dumps(dict(stage='load',component=name)),flush=True)
            with init_empty_weights():module=instantiate_from_config(cfg[name])
            state={k[len(name)+1:]:sf.get_tensor(k) for k in sf.keys() if k.startswith(name+'.')}
            if name=='vae' and not any(k.startswith('encoder.') for k in state):
                # Official inference weights omit the training-only encoder.
                # Remove unused meta modules; require every decoder weight.
                del module.encoder
                del module.pre_kl
            result=module.load_state_dict(state,strict=True,assign=True)
            loads[name]=dict(missing=result.missing_keys,unexpected=result.unexpected_keys,parameters=sum(p.numel() for p in module.parameters()))
            parts[name]=module.to(device='cuda',dtype=torch.float16).eval();del state;gc.collect()
    pipe=Hunyuan3DDiTFlowMatchingPipeline(**parts,scheduler=instantiate_from_config(cfg['scheduler']),
        image_processor=instantiate_from_config(cfg['image_processor']),device='cuda',dtype=torch.float16)
    image=Image.open(ROOT/'conditioning.png').convert('RGBA')
    print(json.dumps(dict(stage='shape_diffusion',steps=50,seed=531020)),flush=True)
    latent=pipe(image=image,num_inference_steps=50,guidance_scale=5.,generator=torch.Generator('cuda').manual_seed(531020),output_type='latent')
    save_inference_checkpoint(dict(latent=latent.cpu(),model=MODEL,revision=REVISION),ROOT/'shape_latent.pt')
    # Unload the denoiser and image encoder before high-resolution SDF extraction.
    pipe.model.cpu();pipe.conditioner.cpu();gc.collect();torch.cuda.empty_cache()
    print(json.dumps(dict(stage='extract_mesh',resolution=380)),flush=True)
    mesh=pipe._export(latent,output_type='trimesh',octree_resolution=380,num_chunks=2000,mc_algo='mc')[0]
    if mesh is None:raise ValueError('No surface extracted')
    mesh.export(ROOT/'shape.ply')
    save_inference_checkpoint(dict(vertices=torch.tensor(mesh.vertices,dtype=torch.float32),faces=torch.tensor(mesh.faces,dtype=torch.int64),
        scope=f'Unchanged {MODEL} inferred mesh; hidden geometry not verified'),ROOT/'shape.pt')
    report=dict(model=MODEL,revision=REVISION,code_commit='f8db63096c8282cb27354314d896feba5ba6ff8a',
        weights_sha256=digest(WEIGHTS/'model.fp16.safetensors'),strict_loads=loads,seconds=time.perf_counter()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),vertices=len(mesh.vertices),faces=len(mesh.faces),
        seed=531020,steps=50,octree_resolution=380,watertight=bool(mesh.is_watertight),accepted=False,
        license='Tencent Hunyuan3D 2.0 Community License; not MIT; inspect upstream license before distribution',
        source_sha256=digest(SOURCE),base_weights_modified=False)
    (ROOT/'shape_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','shape']);parser.add_argument('--full',action='store_true');args=parser.parse_args()
    if args.full:
        ROOT=ROOT.parent/'v4_full';WEIGHTS=WORK/'hunyuan_full/hunyuan3d-dit-v2-0';MODEL='tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0';REVISION='9cd649ba6913f7a852e3286bad86bfa9a2d83dcf'
    with keep_windows_awake():globals()[args.stage]()
