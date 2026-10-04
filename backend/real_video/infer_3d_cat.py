"""Infer actual XYZ cat geometry with pinned TripoSR, then store 3D Gaussians.

Single-image inferred geometry, NOT measured ground truth or a learned motion model.
No source video after the observed frame; no procedural body extrusion.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from .checkpoint_io import digest,save_inference_checkpoint,keep_windows_awake

WORK=Path('../../work/real_video').resolve()
OUT=Path('artifacts/real_video/true3d/v1')
OBS=Path('artifacts/real_video/learned_motion/v1/dense_seed')


@torch.no_grad()
def main(source_path=None, mask_path=None, output=None, count=45000, resolution=160):
    OUT = Path(output) if output is not None else globals()['OUT']
    source_path = Path(source_path) if source_path is not None else OBS/'observed_frame_2.png'
    mask_path = Path(mask_path) if mask_path is not None else OBS/'mask.png'
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'cat_asset.pt').exists(): raise FileExistsError('Preserve inferred asset')
    sys.path.insert(0,str(WORK/'TripoSR'))
    from omegaconf import OmegaConf
    from tsr.system import TSR
    from skimage.measure import marching_cubes
    import trimesh
    torch.set_num_threads(4); torch.manual_seed(430601)
    image=np.array(Image.open(source_path).convert('RGB'))
    mask=np.array(Image.open(mask_path).convert('L'))/255
    if mask.shape != image.shape[:2] or not (mask>.5).any(): raise ValueError('Invalid foreground mask')
    yy,xx=np.where(mask>.5); y0,y1,x0,x1=yy.min(),yy.max()+1,xx.min(),xx.max()+1
    rgb=image[y0:y1,x0:x1].astype(np.float32)/255; alpha=mask[y0:y1,x0:x1,None]
    rgb=rgb*alpha+.5*(1-alpha)
    size=int(max(rgb.shape[:2])/.85); canvas=np.full((size,size,3),.5,np.float32)
    y=(size-rgb.shape[0])//2; x=(size-rgb.shape[1])//2; canvas[y:y+rgb.shape[0],x:x+rgb.shape[1]]=rgb
    condition=Image.fromarray(np.uint8(np.clip(canvas,0,1)*255)); condition.save(OUT/'conditioning.png')
    config=OmegaConf.load(WORK/'triposr_weights/config.yaml'); OmegaConf.resolve(config)
    allowed={'tsr.models.tokenizers.image.DINOSingleImageTokenizer','tsr.models.tokenizers.triplane.Triplane1DTokenizer',
             'tsr.models.transformer.transformer_1d.Transformer1D','tsr.models.network_utils.TriplaneUpsampleNetwork',
             'tsr.models.network_utils.NeRFMLP','tsr.models.nerf_renderer.TriplaneNeRFRenderer'}
    assert {config[k] for k in config if k.endswith('_cls')}==allowed
    config.image_tokenizer.pretrained_model_name_or_path=str(WORK/'dino_config')
    start=time.perf_counter(); model=TSR(config)
    weights=torch.load(WORK/'triposr_weights/model.ckpt',map_location='cpu',weights_only=True)
    # Transformers 5 renamed ViT modules. Values stay unchanged; strict loading
    # below rejects any missing, extra or mismatched parameter after translation.
    if 'image_tokenizer.model.layers.0.attention.q_proj.weight' in model.state_dict():
        replacements=[('.encoder.layer.','.layers.'),('.attention.attention.query.','.attention.q_proj.'),
                      ('.attention.attention.key.','.attention.k_proj.'),('.attention.attention.value.','.attention.v_proj.'),
                      ('.attention.output.dense.','.attention.o_proj.'),('.intermediate.dense.','.mlp.fc1.'),('.output.dense.','.mlp.fc2.')]
        translated={}
        for key,value in weights.items():
            if key.startswith('image_tokenizer.model.'):
                for old,new in replacements: key=key.replace(old,new)
            if key in translated: raise ValueError('Checkpoint mapping collision')
            translated[key]=value
        weights=translated
    model.load_state_dict(weights,strict=True); del weights
    model=model.cuda().eval(); model.renderer.set_chunk_size(8192)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); inference_start=time.perf_counter()
    codes=model(condition,device='cuda'); torch.cuda.synchronize(); inference_seconds=time.perf_counter()-inference_start
    print(json.dumps(dict(stage='inferred_3d_latent',seconds=inference_seconds,shape=list(codes.shape))),flush=True)
    radius=float(model.renderer.cfg.radius)
    axis=torch.linspace(-radius,radius,resolution,device='cuda')
    grid=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    density=model.renderer.query_triplane(model.decoder,grid,codes[0])['density_act'].reshape(resolution,resolution,resolution)
    vertices,faces,_,_=marching_cubes(density.cpu().numpy(),level=25.)
    vertices=vertices/(resolution-1)*2*radius-radius
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    # Preserve complete inferred mesh, including any model-generated defects.
    mesh.export(OUT/'inferred_cat.ply')
    rng=np.random.default_rng(430601)
    face_id=rng.choice(len(faces),count,p=mesh.area_faces/mesh.area_faces.sum())
    uv=rng.random((count,2)); uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    bary=np.column_stack((1-uv.sum(1),uv)); xyz=(vertices[faces[face_id]]*bary[:,:,None]).sum(1)
    colours=model.renderer.query_triplane(model.decoder,torch.tensor(xyz,dtype=torch.float32,device='cuda'),codes[0])['color'].cpu().numpy()
    # TripoSR world x-back/y-right/z-up -> our x-right/y-up/z-back.
    xyz=xyz[:,[1,2,0]]; normals=np.asarray(mesh.face_normals)[face_id][:,[1,2,0]]
    floor=float(xyz[:,1].min()); xyz[:,1]-=floor
    normal=torch.tensor(normals,dtype=torch.float32); normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    helper=torch.tensor([0.,1.,0.]).expand_as(normal).clone(); helper[normal[:,1].abs()>.9]=torch.tensor([1.,0.,0.])
    tangent=torch.nn.functional.normalize(torch.linalg.cross(helper,normal),dim=-1); bitangent=torch.linalg.cross(normal,tangent)
    frames=torch.stack((tangent,bitangent,normal),-1)
    sigma=float(np.sqrt(mesh.area/count)*.72)
    scales=torch.tensor([sigma,sigma,sigma*.22]).expand(count,-1).clone()
    packet=dict(position=torch.tensor(xyz,dtype=torch.float32),frame=frames,scale=scales,colour=torch.tensor(colours,dtype=torch.float32),
                opacity=torch.full((count,),.9),ids=torch.arange(count),face_id=torch.tensor(face_id,dtype=torch.int32),barycentric=torch.tensor(bary,dtype=torch.float32),
                mesh_vertices=torch.tensor(vertices[:,[1,2,0]]-np.array([0.,floor,0.]),dtype=torch.float32),mesh_faces=torch.tensor(faces.astype(np.int32)),
                source='TripoSR learned single-image XYZ geometry and colour; fixed surface sampling, not fitted 3DGS',
                geometry_truth=False,source_image_sha256=digest(source_path))
    sha=save_inference_checkpoint(packet,OUT/'cat_asset.pt')
    torch.cuda.synchronize()
    report=dict(model='stabilityai/TripoSR',revision='5b521936b01fbe1890f6f9baed0254ab6351c04a',
                code_commit='107cefdc244c39106fa830359024f6a2f1c78871',dino_config_revision='f205d5d8e640a89a2b8ef0369670dfc37cc07fc2',
                weights_sha256=digest(WORK/'triposr_weights/model.ckpt'),asset_sha256=sha,
                source_paper='https://arxiv.org/html/2403.02151v1',licence='MIT per official repository',
                inference_seconds=inference_seconds,load_infer_extract_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                gaussians=count,mesh_vertices=len(vertices),mesh_faces=len(faces),xyz_extent=np.ptp(xyz,axis=0).tolist(),
                position_covariance_eigenvalues=np.linalg.eigvalsh(np.cov(xyz.T)).tolist(),
                limitations='Unseen geometry inferred, not verified. No learned motion or new video generator. Surface-sampled Gaussians are not optimized for multiview photometric reconstruction.')
    (OUT/'asset_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    with keep_windows_awake(): main()
