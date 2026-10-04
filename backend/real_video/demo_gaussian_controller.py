"""Run released neural motion weights against the original persistent cat.

No Wan, input video, or diffusion model is loaded. Retargeting remains an
approximate, separately evaluated adaptation from dog locomotion to a cat.
"""
import argparse
import json
import time
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio
from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from .quadruped_controller import QuadrupedController, DEMO, VENDOR
from .controller_rig import guarded_rig, CatRetargeter, DualSurfaceSkinner
from .gaussian3d import render

BASE=Path('artifacts/real_video')


@torch.no_grad()
def run(gait='Walk',version='v1',mode='ik',bundle=None):
    out=BASE/'hunyuan_gaussian'/f'neural_controller_{version}'/gait.lower()
    out.mkdir(parents=True,exist_ok=True)
    video=out/'gaussian_motion_7s.mp4'
    if video.exists(): raise FileExistsError('Preserve completed experiment')
    torch.set_num_threads(2)
    controller=QuadrupedController(bundle=bundle)
    exported=json.loads((Path(bundle)/'export_report.json').read_text()) if bundle else None
    source_model_hash=exported['source_weights_sha256'] if exported else digest(DEMO/'Network.pt')
    state0=controller.snapshot()
    speed={'Walk':.7,'Trot':2.,'Idle':0.}[gait]
    samples=[]; worlds=[]
    for _ in range(70):
        world,relative,_=controller.step(gait,speed)
        samples.append(relative); worlds.append(world)
    generated=torch.cat(samples); world=torch.cat(worlds)
    initial=torch.cat([v.flatten() for v in state0.values()])
    resumed=controller.snapshot()
    first=controller.step(gait,speed)[1]
    controller.restore(resumed); second=controller.step(gait,speed)[1]
    assert torch.equal(first,second), 'Snapshot must resume exactly'
    source=BASE/'hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt'
    shape=BASE/'hunyuan_gaussian/v4_full/shape.pt'
    a=load_verified(source); mesh=load_verified(shape)
    a={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in a.items()}
    a['mesh_vertices']=mesh['vertices'].cuda(); a['mesh_faces']=mesh['faces'].cuda()
    parts=[torch.linalg.eigh(x) for x in a['covariance'].split(16000)]
    values=torch.cat([p[0] for p in parts]); frames=torch.cat([p[1] for p in parts])
    frames[:,:,0]*=torch.linalg.det(frames).sign()[:,None]
    a['frame']=frames; a['scale']=values.clamp_min(1e-12).sqrt()
    rig=guarded_rig(a['mesh_vertices']); skin=DualSurfaceSkinner(a,rig)
    identity=torch.eye(3,device='cuda').repeat(17,1,1)
    p,c,*_=skin(identity)
    rest_error=float((p-a['position']).abs().max()); assert rest_error<1e-5
    targeter=CatRetargeter(rig,controller.guidances['Walk'].cuda(),mode=mode)
    save_inference_checkpoint(dict(generated_positions=generated,generated_world=world,
        initial_state=state0,resume_state=resumed,fps=30,gait=gait,speed=speed,
        model_sha256=source_model_hash,classification='pretrained neural skeletal generation; approximate cat retarget'),out/'motion.pt')
    save_inference_checkpoint({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in rig.items()},out/'rig.pt')
    poses=[]; roots=[]; rows=[]; latency=[]; evidence=[]
    centre=(a['position'].amin(0)+a['position'].amax(0))/2
    torch.cuda.reset_peak_memory_stats()
    with imageio.get_writer(video,fps=15,codec='libx264',quality=8) as writer:
        for i,(s,w) in enumerate(zip(generated,world)):
            local,shift,feet,error,contacts=targeter.step(s.cuda(),(w[0]-s[0]).cuda())
            poses.append(local.cpu()); roots.append(shift.cpu())
            if i%2: continue
            torch.cuda.synchronize(); start=time.perf_counter()
            shift=shift+shift.new_tensor([0.,.72,0.])
            p,c,_,v,area=skin(local,shift)
            panels=[]; target=centre+shift
            for direction in ([0,.13,1],[.7,.23,1]):
                direction=p.new_tensor(direction); eye=target+direction/direction.norm()*3.5
                rgb,_=render(p,c,a['colour'],a['opacity'],eye,target,384,512,ground=True)
                panels.append(np.uint8(rgb.clamp(0,1).cpu().numpy()*255))
            frame=Image.fromarray(np.concatenate(panels,1)); draw=ImageDraw.Draw(frame)
            draw.text((10,8),f'Pretrained neural {gait.lower()} | persistent 400,000 Gaussians | no Wan',fill='white')
            draw.text((522,28),'Oblique view | cameras follow root | approximate cat rig',fill='white')
            writer.append_data(np.asarray(frame)); torch.cuda.synchronize(); latency.append(time.perf_counter()-start)
            frame_index=i//2
            row=dict(frame=frame_index,ik_max_error=float(error.max()),contacts=contacts,
                area_min=float(area.min()),area_max=float(area.max()),
                area_bad_fraction=float(((area<.2)|(area>5)).float().mean()))
            rows.append(row)
            if frame_index in [0,20,40,60,80,104]:
                frame.save(out/f'frame_{frame_index:03d}.png'); evidence.append(frame)
                print(json.dumps(row),flush=True)
    sheet=Image.new('RGB',(1024,384*len(evidence)))
    for j,frame in enumerate(evidence): sheet.paste(frame,(0,j*384))
    sheet.save(out/'sheet.jpg')
    save_inference_checkpoint(dict(rotations=torch.stack(poses),root_translation=torch.stack(roots),fps=30),out/'retargeted.pt')
    parameter_bytes=sum(x.numel()*x.element_size() for x in controller.model.parameters())
    asset_bytes=sum(x.numel()*x.element_size() for x in a.values() if isinstance(x,torch.Tensor))
    report=dict(gait=gait,retarget_mode=mode,contact_constraints_applied=mode=='ik',classification='pretrained conditional skeletal generation, not clip replay',
        cat_scope='Dog motion prior with manually inferred cat rig; not validated cat anatomy',
        upstream='https://github.com/facebookresearch/ai4animationpy',
        upstream_commit='bfb5866681f7ea6dac9984be05181de5955eb48b',
        license='CC-BY-NC-4.0',weights_sha256=source_model_hash,
        gaussian_asset_sha256=digest(source),gaussians=len(p),frames=105,fps=15,duration_seconds=7,
        pose_output_fps=30,controller_call_hz=10,lookahead_frames=16,
        controller_median_ms=float(np.median(controller.latencies))*1000,
        deform_and_two_view_render_median_ms=float(np.median(latency))*1000,
        weights_parameter_bytes=None if exported else parameter_bytes,
        frozen_inference_file_bytes=exported['frozen_file_bytes'] if exported else None,
        controller_recurrent_state_tensor_bytes=initial.numel()*initial.element_size(),
        asset_tensor_bytes_including_derived_fields=asset_bytes,skin_weight_bytes=rig['vertex_weights'].numel()*4,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),resume_exact=True,
        rest_position_max_error=rest_error,appearance_ids_fixed=True,wan_calls=0,new_training=False,
        quality_matched_wan_speedup=None,
        limitations=['Headless adapter omits upstream PID and learned contact postprocessor',
            'IK mode uses heuristic contacts; relative mode has no contact correction. Neither is dynamics simulation',
            'Straight-line root control; generated root yaw ignored; dog-to-cat retarget unvalidated',
            'No held-out benchmark accuracy claim; guidance poses are released static controls, not video input'],
        diagnostics=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='diagnostics'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--gait',choices=['Walk','Trot','Idle'],default='Walk'); parser.add_argument('--version',default='v1'); parser.add_argument('--mode',choices=['ik','relative'],default='ik'); parser.add_argument('--bundle')
    args=parser.parse_args()
    with keep_windows_awake(): run(args.gait,args.version,args.mode,args.bundle)
