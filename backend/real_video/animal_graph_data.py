"""Mask-bound control tracks from recorded animal clips; future tracks are targets."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from .prepare_animal_motion import WORK,TRAIN,VAL
from .gaussian_motion_memory import sample
from .checkpoint_io import keep_windows_awake,save_inference_checkpoint,digest

DATA=WORK/'control_tracks_v1.pt'
HEIGHT=128
NODES=48


def farthest_points(mask,count=NODES):
    yy,xx=np.where(mask>.5); candidates=np.stack((xx,yy),-1).astype(np.float32)
    if len(candidates)<count: raise ValueError('Object too small for controls')
    mean=candidates.mean(0); index=int(((candidates-mean)**2).sum(1).argmin()); result=[]
    distance=np.full(len(candidates),np.inf)
    for _ in range(count):
        result.append(candidates[index]); distance=np.minimum(distance,((candidates-candidates[index])**2).sum(1)); index=int(distance.argmax())
    return np.stack(result)


def adjacency(position,mask,height):
    pixels=position*height-.5; distance=np.linalg.norm(position[:,None]-position[None],axis=-1)
    inside=[]
    for t in np.linspace(0,1,9):
        points=pixels[:,None]*(1-t)+pixels[None]*t
        inside.append(sample(mask,points.reshape(-1,2)).reshape(len(position),len(position)))
    allowed=np.min(inside,axis=0)>.5
    rank=np.argsort(np.argsort(distance,axis=1),axis=1)<9
    weight=np.exp(-distance**2/.04)*allowed*rank
    weight/=np.maximum(weight.sum(1,keepdims=True),1e-8)
    return weight.astype(np.float32)


def track_controls(frames,masks,flows=None):
    height=frames[0].shape[0]; positions=farthest_points(masks[0]); states=[positions.copy()]
    angle=np.zeros(NODES,np.float32); angles=[angle.copy()]; visibility=[np.ones(NODES,np.float32)]; confidences=[np.ones(NODES,np.float32)]
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    grays=[cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) for frame in frames]
    for t in range(1,len(frames)):
        forward,backward=flows[t-1] if flows is not None else (dis.calc(grays[t-1],grays[t],None),dis.calc(grays[t],grays[t-1],None))
        dy,dx=np.gradient(forward,axis=(0,1)); jx=sample(dx,positions); jy=sample(dy,positions)
        angular=np.arctan2(jx[:,1]-jy[:,0],2+jx[:,0]+jy[:,1])
        step=sample(forward,positions); positions=positions+step
        fb=np.linalg.norm(step+sample(backward,positions),axis=-1)
        valid=(positions[:,0]>=0)&(positions[:,0]<frames[0].shape[1])&(positions[:,1]>=0)&(positions[:,1]<height)
        vis=sample(masks[t],positions)*valid
        angle=angle+np.clip(angular,-.5,.5); states.append(positions.copy()); angles.append(angle.copy()); visibility.append(vis.astype(np.float32)); confidences.append(np.exp(-fb**2/4).astype(np.float32))
    positions=(np.stack(states)+.5)/height; angles=np.stack(angles); visible=np.stack(visibility); confidence=np.stack(confidences)
    colour=sample(frames[2].astype(np.float32)/255,positions[2]*height-.5)*2-1
    return dict(position=torch.from_numpy(positions).float(),angle=torch.from_numpy(angles).float(),visibility=torch.from_numpy(visible).float(),
                confidence=torch.from_numpy(confidence).float(),colour=torch.from_numpy(colour).float(),
                adjacency=torch.from_numpy(adjacency(positions[2],masks[2],height)))


def main():
    if DATA.exists(): raise FileExistsError('Preserve animal data cache')
    cv2.setNumThreads(4); torch.set_num_threads(4); records=[]; metadata=[]; rejected=[]
    root=WORK/'extracted/DAVIS'
    with keep_windows_awake():
        for name in TRAIN+VAL:
            paths=sorted((root/'JPEGImages/480p'/name).glob('*.jpg'))[::2]; frames=[]; labels=[]
            for path in paths:
                rgb=np.array(Image.open(path).convert('RGB')); width=round(rgb.shape[1]*HEIGHT/rgb.shape[0])
                frames.append(cv2.resize(rgb,(width,HEIGHT),interpolation=cv2.INTER_AREA))
                mask=np.array(Image.open(root/'Annotations/480p'/name/(path.stem+'.png')))
                labels.append(cv2.resize(mask,(width,HEIGHT),interpolation=cv2.INTER_NEAREST))
            gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in frames]; dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flows=[(dis.calc(a,b,None),dis.calc(b,a,None)) for a,b in zip(gray[:-1],gray[1:])]
            object_ids=[2] if name=='cat-girl' else sorted(set(np.unique(labels[0]))-{0})
            for start in range(0,len(frames)-14,4):
                for object_id in object_ids:
                    masks=[(m==object_id).astype(np.float32) for m in labels[start:start+15]]
                    if masks[0].sum()<150 or masks[2].sum()<150:
                        rejected.append(dict(sequence=name,start=start,object=int(object_id),reason='small observed object')); continue
                    record=track_controls(frames[start:start+15],masks,flows[start:start+14]); records.append(record)
                    metadata.append(dict(sequence=name,start_sampled=start,raw_start=start*2,raw_stride=2,object=int(object_id),
                                         split='train' if name in TRAIN else 'validation',seed_frames=3,target_frames=12))
            print(json.dumps(dict(sequence=name,windows=len(records))),flush=True)
        packet={key:torch.stack([r[key] for r in records]) for key in records[0]}
        packet.update(metadata=metadata,config=dict(nodes=NODES,height=HEIGHT,window_frames=15,seed_frames=3,stride=2,
                      split='official subset, sequence-disjoint; not verified physical-identity-disjoint',
                      visibility='annotated object-mask membership of flow-advected control, NOT true surface visibility',
                      cat_annotation='cat-girl object 2 verified in first image and annotation; person object 1 excluded'),
                      source_sha256=json.loads((WORK/'provenance.json').read_text())['archive_sha256'])
        sha=save_inference_checkpoint(packet,DATA)
        report=dict(windows=len(records),train=sum(m['split']=='train' for m in metadata),validation=sum(m['split']=='validation' for m in metadata),
                    train_sequences=TRAIN,validation_sequences=VAL,rejected=rejected,sha256=sha)
        (WORK/'control_tracks_v1.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__': main()
