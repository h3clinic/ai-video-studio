"""Experimental quantized state-delta playback. Not a competitive video codec.

Stores an initial scene plus translation and rotation increments. Appearance and
scale are static in this prototype. Recurrent h is not part of visual playback.
"""
from pathlib import Path
import numpy as np
import torch
from .geometry import exp_so3


def log_so3_small(R):
    v=torch.stack((R[...,2,1]-R[...,1,2],R[...,0,2]-R[...,2,0],R[...,1,0]-R[...,0,1]),-1)*0.5
    sine=v.norm(dim=-1,keepdim=True)
    cosine=((R.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*0.5).clamp(-1,1)[...,None]
    theta=torch.atan2(sine,cosine)
    if (theta>3.0).any():
        raise ValueError('Rotation step near pi is outside this incremental codec scope')
    return v*(theta/sine.clamp_min(1e-8)).where(sine>1e-7,torch.ones_like(sine))


def encode(states,path,quantum=1e-5,dt=0.08):
    path=Path(path)
    initial={k:v.detach().cpu().numpy().astype(np.float32) for k,v in states[0].items() if k!='velocity'}
    p=torch.stack([s['p'].detach().cpu() for s in states])
    R=torch.stack([s['R'].detach().cpu() for s in states])
    dp=p[1:]-p[:-1]
    dw=log_so3_small(R[:-1].transpose(-1,-2)@R[1:])
    delta=torch.cat((dp,dw),-1)/quantum
    if delta.abs().max()>32767:
        raise ValueError('Delta exceeds int16 range; increase quantum rather than silently clip')
    np.savez_compressed(path,**initial,updates=delta.round().numpy().astype(np.int16),
                        quantum=np.array(quantum),dt=np.array(dt),format_version=np.array(1))
    return path.stat().st_size


def decode(path):
    with np.load(path,allow_pickle=False) as f:
        if int(f['format_version'])!=1:
            raise ValueError('Unsupported codec version')
        state={k:torch.from_numpy(f[k].copy()) for k in ('p','R','scale','color','alpha')}
        updates=torch.from_numpy(f['updates'].copy()).float()*float(f['quantum'])
    yield state
    for delta in updates:
        state=dict(state,p=state['p']+delta[...,:3],R=state['R']@exp_so3(delta[...,3:]))
        yield state
