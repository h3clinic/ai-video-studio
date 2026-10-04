"""Retarget an existing clip-conditioned joint network, NOT new motion generation.

The original Hunyuan appearance and Gaussian attachments are immutable. An
explicit approximate rig moves shared mesh vertices before covariance transport.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from .checkpoint_io import load_verified, save_inference_checkpoint, digest
from .fit_articulated_weights import ClipJointWeights
from .fit_motion3d import PARENTS, rotation_vectors
from .surface_skin3d import SurfaceSkinner
from .gaussian3d import render

BASE = Path('artifacts/real_video')
OUT = BASE/'hunyuan_gaussian/motion_transfer_v1'


def approximate_rig(vertices):
    # Inferred joints for THIS asset: +X head, +Y up, +/-Z bilateral.
    joints = vertices.new_tensor([
        [0,0,0],[.48,.08,0],[.72,.16,0],[-.66,.14,0],[-.86,.48,0],
        [-.55,-.12,.16],[-.65,-.37,.17],[-.66,-.64,.17],
        [-.45,-.12,-.16],[-.40,-.39,-.16],[-.26,-.61,-.16],
        [.39,-.06,.16],[.38,-.37,.16],[.41,-.66,.16],
        [.40,-.06,-.16],[.52,-.32,-.16],[.55,-.52,-.16]])
    ends = joints.clone()
    for start,end in [(1,2),(3,4),(5,6),(6,7),(8,9),(9,10),(11,12),(12,13),(14,15),(15,16)]:
        ends[start] = joints[end]
    starts = joints.clone(); starts[0] = vertices.new_tensor([-.55,.03,0]); ends[0]=vertices.new_tensor([.4,.03,0])
    ends[2] = vertices.new_tensor([.88,.15,0])
    segment = ends-starts
    t = ((vertices[:,None]-starts)*segment).sum(-1)/segment.square().sum(-1).clamp_min(1e-8)
    closest = starts+t.clamp(0,1)[...,None]*segment
    distance = (vertices[:,None]-closest).square().sum(-1)
    weights = torch.softmax(-distance/.008,dim=-1)
    return dict(joints=joints, parents=PARENTS, vertex_weights=weights)


@torch.no_grad()
def run():
    OUT.mkdir(parents=True,exist_ok=True)
    video=OUT/'motion_transfer_7s.mp4'
    if video.exists(): raise FileExistsError('Preserve prior experiment')
    source=BASE/'hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt'
    meshpath=BASE/'hunyuan_gaussian/v4_full/shape.pt'
    modelpath=BASE/'true3d/articulated_weight_loop/v2_constrained/model.pt'
    a=load_verified(source); mesh=load_verified(meshpath)
    a={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in a.items()}
    a['mesh_vertices']=mesh['vertices'].cuda(); a['mesh_faces']=mesh['faces'].cuda()
    decomposed=[torch.linalg.eigh(chunk) for chunk in a['covariance'].split(16000)]
    eigenvalues=torch.cat([item[0] for item in decomposed]); frame=torch.cat([item[1] for item in decomposed])
    frame[:,:,0]*=torch.linalg.det(frame).sign()[:,None]
    a['frame']=frame; a['scale']=eigenvalues.clamp_min(1e-12).sqrt()
    rig=approximate_rig(a['mesh_vertices']); skinner=SurfaceSkinner(a,rig)
    packet=load_verified(modelpath); model=ClipJointWeights(**packet['model_config']).cuda()
    model.load_state_dict(packet['candidates']['pose_only']['model']); model.eval()
    identity=torch.eye(3,device='cuda').repeat(17,1,1)
    p,c,*_=skinner(identity)
    rest_error=float((p-a['position']).abs().max())
    covariance_error=float((c-a['covariance']).abs().max())
    assert rest_error<1e-5 and covariance_error<1e-5
    save_inference_checkpoint({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in rig.items()},OUT/'rig.pt')
    centre=(a['position'].amin(0)+a['position'].amax(0))/2
    times=torch.linspace(0,1,112,device='cuda')
    _,_,rv=model(times); rv=rv*.45
    # Relative learned rotations only. No extrapolation, procedural gait or
    # translation: 1.875 seconds of fitted motion slowed to a 7-second inspection.
    poses=rotation_vectors(rv)
    save_inference_checkpoint(dict(rotations=poses.cpu(),source_time=times.cpu(),fps=16,
        classification='retargeted video-conditioned motion fit; not novel generation'),OUT/'motion.pt')
    elapsed=[]; metrics=[]; screenshots=[]
    torch.cuda.reset_peak_memory_stats()
    with imageio.get_writer(video,fps=16,codec='libx264',quality=8) as writer:
        for i,pose in enumerate(poses):
            torch.cuda.synchronize(); start=time.perf_counter()
            p,c,_,v,areas=skinner(pose)
            panels=[]
            for direction in ([0,.08,1],[.7,.22,1]):
                direction=p.new_tensor(direction); eye=centre+direction/direction.norm()*3.5
                rgb,_=render(p,c,a['colour'],a['opacity'],eye,centre,384,512,ground=False)
                panels.append(np.uint8(rgb.clamp(0,1).cpu().numpy()*255))
            frame=Image.fromarray(np.concatenate(panels,axis=1)); draw=ImageDraw.Draw(frame)
            draw.text((12,10),'Original cat | learned clip-motion transfer (slowed; NOT new generation)',fill='white')
            draw.text((524,32),'Fixed oblique camera',fill='white')
            writer.append_data(np.asarray(frame)); torch.cuda.synchronize()
            elapsed.append(time.perf_counter()-start)
            metrics.append(dict(frame=i,area_min=float(areas.min()),area_max=float(areas.max()),
                mean_displacement=float((p-a['position']).norm(dim=-1).mean())))
            if i in [0,28,56,84,111]:
                frame.save(OUT/f'frame_{i:03d}.png'); screenshots.append(frame.copy())
            if i%28==0: print(json.dumps(metrics[-1]),flush=True)
    sheet=Image.new('RGB',(1024,384*len(screenshots)))
    for j,im in enumerate(screenshots): sheet.paste(im,(0,384*j))
    sheet.save(OUT/'motion_sheet.jpg')
    report=dict(classification='motion transfer of an observed clip, not new generative training',
        source_asset=str(source),asset_sha256=digest(source),model_sha256=digest(modelpath),
        gaussians=len(p),frames=112,fps=16,duration_seconds=7,source_motion_seconds=30/16,
        amplitude_multiplier=.45,rest_position_max_error=rest_error,rest_covariance_max_error=covariance_error,
        gaussian_ids_and_colours_unchanged=True,rig='manual approximate bone-distance skinning',
        model_weights_changed=False,wan_weights_changed=False,
        median_deform_and_two_view_render_seconds=float(np.median(elapsed)),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),frames_diagnostics=metrics,
        efficiency_claim='No quality-matched Wan comparison; these are replay/render costs only',
        research_source='https://arxiv.org/html/2402.04796v1',
        mathematics='mu=sum barycentric*deformed_vertices; covariance=F Sigma F^T',
        limitations=['Inferred rig; no contact solver','Earlier motion checkpoint failed anatomical quality',
                     'Surface attachment does not guarantee correct anatomy or non-folding'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='frames_diagnostics'}),flush=True)


if __name__=='__main__': run()
