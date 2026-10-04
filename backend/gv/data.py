"""On-the-fly synthetic TRAINING distribution; never used by learned inference.

Three controlled motion families: orbit, spring, coupled rotation + translation.
Labels are finite controls, not open-ended natural-language understanding.
"""
import torch
import torch.nn.functional as F
from .geometry import exp_so3, world

MODES = ('orbit','spring','drift')
NOISE_DIM = 12
DT = 0.08


def controls(batch, device, gen, mode=None):
    ids = torch.randint(3,(batch,),device=device,generator=gen) if mode is None else torch.full((batch,),mode,device=device)
    kind = F.one_hot(ids,3).float()
    rate = 0.55 + 0.75*torch.rand(batch,1,device=device,generator=gen)
    axis = torch.tensor([0.,1.,0.],device=device).expand(batch,3)
    center = torch.zeros(batch,3,device=device)
    return {'kind':kind,'rate':rate,'axis':axis,'center':center}


def sample_noise(batch,n,device,gen):
    return torch.randn(batch,n,NOISE_DIM,device=device,generator=gen)


def synthetic_sequence(z, control, steps, dt=DT):
    b,n,_=z.shape
    kind = control['kind']
    rate = control['rate'][:,None,:]
    base = torch.tanh(z[...,:3])*torch.tensor([0.75,0.4,0.6],device=z.device)
    R0 = exp_so3(0.5*torch.tanh(z[...,3:6]))
    scales = torch.tensor([0.075,0.025,0.035],device=z.device)*(0.8+0.4*torch.sigmoid(z[...,6:9]))
    palette = torch.tensor([[0.12,0.72,0.95],[0.95,0.40,0.18],[0.58,0.35,0.95]],device=z.device)
    colors = (kind@palette)[:,None,:]*0.8 + 0.2*torch.sigmoid(z[...,9:12])
    alpha = torch.full((b,n,1),0.82,device=z.device)
    axis = control['axis'][:,None,:].expand(b,n,3)
    omega = torch.stack((0.25*torch.tanh(z[...,3]),rate.squeeze(-1).expand(b,n),0.2*torch.tanh(z[...,5])),-1)
    seq=[]
    for t in range(steps+1):
        time = t*dt
        W = exp_so3(axis*rate*time)
        orbit = world(W,base)
        spring = base*torch.cos(rate*time)
        drift = world(exp_so3(axis*rate*time*0.45),base) + axis*(0.18*time)
        pos = (orbit*kind[:,None,0:1] + spring*kind[:,None,1:2] + drift*kind[:,None,2:3]) + control['center'][:,None,:]
        orbit_v = torch.linalg.cross(axis*rate,orbit,dim=-1)
        spring_v = -base*rate*torch.sin(rate*time)
        drift_v = torch.linalg.cross(axis*rate*0.45,drift-axis*(0.18*time),dim=-1)+axis*0.18
        vel = orbit_v*kind[:,None,0:1]+spring_v*kind[:,None,1:2]+drift_v*kind[:,None,2:3]
        # Physical time, not a diffusion-denoising timestep.
        R = R0 @ exp_so3(omega*time)
        seq.append(dict(p=pos,R=R,velocity=vel,scale=scales,color=colors,alpha=alpha))
    return seq


def make_batch(seed,batch=8,n=48,steps=24,device='cpu',mode=None):
    gen=torch.Generator(device=device).manual_seed(seed)
    control=controls(batch,device,gen,mode)
    z=sample_noise(batch,n,device,gen)
    return z,control,synthetic_sequence(z,control,steps)
