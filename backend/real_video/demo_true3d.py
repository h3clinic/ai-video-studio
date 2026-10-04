"""Seven-second actual 3D orbit and explicitly scripted rig inspection.

The asset is learned image-to-3D inference. Camera/joint trajectories are
diagnostic controls, not a learned action generator and not a measured cat gait.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake
from .gaussian3d import render,rotation,skin,project

ROOT=Path('artifacts/real_video/true3d/v1')


def covariance(asset):
    return (asset['frame']*asset['scale'][:,None].square())@asset['frame'].transpose(-1,-2)


@torch.no_grad()
def previews():
    cpu=load_verified(ROOT/'cat_asset.pt'); a={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
    p=a['position']; center=(p.amin(0)+p.amax(0))*.5; extent=float((p.amax(0)-p.amin(0)).max()); distance=extent*1.9
    canvas=Image.new('RGB',(1280,1024),'#151922'); draw=ImageDraw.Draw(canvas)
    for j,degrees in enumerate([0,90,180,270]):
        angle=math.radians(degrees); eye=center+p.new_tensor([math.sin(angle)*distance,distance*.23,math.cos(angle)*distance])
        image,_=render(p,covariance(a),a['colour'],a['opacity'],eye,center)
        pic=np.uint8(image.cpu().numpy()*255); Image.fromarray(pic).save(ROOT/f'view_{degrees}.png')
        x,y=j%2*640,j//2*512; canvas.paste(Image.fromarray(pic),(x,y+32)); draw.text((x+8,y+8),f'True XYZ Gaussian asset | azimuth {degrees} degrees | inferred geometry',fill='white')
    canvas.save(ROOT/'four_views.jpg')
    print(json.dumps(dict(extent=extent,center=center.tolist(),bounds=[p.amin(0).tolist(),p.amax(0).tolist()])))


def make_rig(cpu):
    """Initial hand-specified normalized rig, NOT extracted animal anatomy.

    Body long axis is assumed image-horizontal x. Four limbs branch from torso;
    soft bindings use bone distance with an explicit torso/limb partition.
    """
    p=cpu['position']; lo=p.amin(0); hi=p.amax(0); span=hi-lo
    # Relative landmarks are a declared rig assumption, not fitted ground truth.
    coords=[[.5,.58,.5],[.69,.66,.5],[.83,.77,.5],[.28,.57,.5],[.1,.55,.5]]
    names=['root','neck','head','tail_base','tail']; parents=[-1,0,1,0,3]
    for label,x in [('rear',.32),('front',.69)]:
        for side,z in [('near',.72),('far',.28)]:
            parent=len(coords); coords.extend([[x,.53,z],[x,.28,z],[x,.06,z]])
            names.extend([label+'_'+side+'_hip',label+'_'+side+'_knee',label+'_'+side+'_foot']); parents.extend([0,parent,parent+1])
    joints=lo+torch.tensor(coords)*span
    # Segment distance, with a nonzero torso support so points do not freely drift.
    starts=torch.stack([joints[max(parent,0)] for parent in parents]); ends=joints
    vector=ends-starts; t=((p[:,None]-starts)*vector).sum(-1)/vector.square().sum(-1).clamp_min(1e-8)
    near=starts+t.clamp(0,1)[...,None]*vector; dist=(p[:,None]-near).norm(dim=-1)
    rel=(p-lo)/span
    allowed=torch.ones_like(dist,dtype=torch.bool)
    allowed[:,5:]=rel[:,1:2]<.6
    allowed[:,:5]=rel[:,1:2]>.24
    # Avoid binding across near/far limbs solely due to projected overlap.
    for k in range(5,len(joints)):
        z=coords[k][2]; allowed[:,k]&=(rel[:,2]>.5) if z>.5 else (rel[:,2]<=.5)
    dist=dist.masked_fill(~allowed,100)
    values,indices=dist.topk(3,largest=False)
    weights=torch.softmax(-values/max(float(span.max())*.055,1e-4),1)
    return dict(joints=joints,parents=parents,names=names,index=indices.to(torch.int16),weight=weights,
                source='Manually specified normalized prototype rig. Not predicted anatomical landmarks or learned gait.')


@torch.no_grad()
def videos():
    if (ROOT/'video_report.json').exists(): raise FileExistsError('Preserve videos')
    cpu=load_verified(ROOT/'cat_asset.pt'); rig=make_rig(cpu)
    save_inference_checkpoint(rig,ROOT/'rig.pt')
    a={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
    r={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in rig.items()}
    p=a['position']; cov=covariance(a); center=(p.amin(0)+p.amax(0))*.5
    extent=float((p.amax(0)-p.amin(0)).max()); distance=extent*1.9
    frames=168; fps=24; font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17); report={}
    for mode in ['orbit','rig_test']:
        path=ROOT/f'cat_3d_{mode}_7s.mp4'; writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=1)
        times=[]; max_bone_error=0.; min_height=100.; pose_records=[]; eye_records=[]
        torch.cuda.reset_peak_memory_stats()
        try:
            for frame in range(frames):
                seconds=frame/fps; azimuth=2*math.pi*frame/frames if mode=='orbit' else .35
                eye=center+p.new_tensor([math.sin(azimuth)*distance,distance*.23,math.cos(azimuth)*distance])
                eye_records.append(eye.cpu())
                torch.cuda.synchronize(); start=time.perf_counter(); pos=p; co=cov; transforms=None
                angles=p.new_zeros(len(r['joints']))
                if mode=='rig_test':
                    # Explicit small-amplitude control test: do NOT label this learned motion.
                    phase=2*math.pi*seconds*.5
                    angles[1]=.12*math.sin(phase); angles[3]=.18*math.sin(phase); angles[4]=.2*math.sin(phase+.3)
                    for base,offset in [(5,0),(8,math.pi),(11,math.pi),(14,0)]:
                        angles[base]=.17*math.sin(phase+offset); angles[base+1]=.12*math.sin(phase+offset+.8)
                    axes=p.new_tensor([0.,0.,1.]).expand(len(angles),3)
                    local=rotation(axes,angles)
                    pos,co,transforms=skin(p,cov,r['joints'],r['parents'],local,r['index'].long(),r['weight'])
                    # Vertical support correction only: no physics or no-slip contact claim.
                    shift=pos[:,1].min(); pos[:,1]-=shift; transforms[:,1,3]-=shift
                    for child,parent in enumerate(r['parents']):
                        if parent>=0:
                            error=abs(float((transforms[child,:3,3]-transforms[parent,:3,3]).norm()-(r['joints'][child]-r['joints'][parent]).norm()))
                            max_bone_error=max(max_bone_error,error)
                pose_records.append(angles.cpu()); min_height=min(min_height,float(pos[:,1].min()))
                rgb,_=render(pos,co,a['colour'],a['opacity'],eye,center)
                torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
                canvas=Image.new('RGB',(640,552),'#141820'); draw=ImageDraw.Draw(canvas)
                draw.text((8,5),'ACTUAL 3D GAUSSIANS | '+('camera orbit' if mode=='orbit' else 'scripted joint diagnostic'),font=font,fill='white')
                canvas.paste(Image.fromarray(np.uint8(rgb.cpu().numpy()*255)),(0,32))
                draw.text((8,516),'TripoSR-inferred geometry | 45,000 persistent XYZ splats',font=font,fill='white')
                draw.text((8,535),'Not learned action generation. Unseen geometry may be wrong.',font=font,fill='#ccd1d7')
                if transforms is not None:
                    projected,*_=project(transforms[:,:3,3],torch.eye(3,device=p.device).expand(len(angles),3,3)*.0001,eye,center,480,640)
                    # Small joint markers provide inspection without replacing actual Gaussian pixels.
                    for child,parent in enumerate(r['parents']):
                        if parent>=0:
                            v=projected[[child,parent]].cpu().numpy(); draw.line([(float(x),float(y)+32) for x,y in v],fill='#55eeaa',width=1)
                writer.append_data(np.asarray(canvas))
                if frame in [0,42,84,126]: canvas.save(ROOT/f'{mode}_{frame:03d}.jpg')
                if frame%42==0: print(json.dumps(dict(mode=mode,frame=frame,total=frames)),flush=True)
        finally: writer.close()
        save_inference_checkpoint(dict(angles=torch.stack(pose_records),camera=torch.stack(eye_records),fps=fps,
                                       source='Explicit camera or scripted-joint controls, not learned movement'),ROOT/f'{mode}_controls.pt')
        report[mode]=dict(frames=frames,fps=fps,seconds=frames/fps,median_ms=float(np.median(times[5:])),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                          max_bone_length_error=max_bone_error,min_gaussian_center_height=min_height,video_sha256=digest(path),
                          motion_source='camera trajectory only' if mode=='orbit' else 'scripted joint angles, diagnostic only')
    report['asset_sha256']=digest(ROOT/'cat_asset.pt'); report['persistent_ids']=len(p)
    report['limits']='Not trained 3D action generation, no full physical contact/collision model. Center-height correction is not foot locking. Shape/colour inferred from one image. Neither speed nor memory compared at matched quality.'
    (ROOT/'video_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['previews','videos']); args=parser.parse_args()
    torch.set_num_threads(4)
    with keep_windows_awake(): {'previews':previews,'videos':videos}[args.mode]()
