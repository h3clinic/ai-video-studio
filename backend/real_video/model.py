"""Small from-scratch diffusion network whose outputs are temporal Gaussian fields."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t,width=128):
    frequencies=torch.exp(-math.log(10000)*torch.arange(width//2,device=t.device)/(width//2-1))
    phase=t[:,None].float()*frequencies[None]
    return torch.cat((phase.sin(),phase.cos()),-1)


class Residual(nn.Module):
    def __init__(self,inc,outc,dropout=0.):
        super().__init__()
        self.norm1=nn.GroupNorm(8,inc)
        self.conv1=nn.Conv3d(inc,outc,3,padding=1)
        self.affine=nn.Linear(128,2*outc)
        self.norm2=nn.GroupNorm(8,outc)
        self.conv2=nn.Conv3d(outc,outc,3,padding=1)
        self.skip=nn.Conv3d(inc,outc,1) if inc!=outc else nn.Identity()
        self.dropout=dropout

    def forward(self,x,condition):
        h=self.conv1(F.silu(self.norm1(x)))
        scale,bias=self.affine(F.silu(condition)).chunk(2,-1)
        h=self.norm2(h)*(1+scale[:,:,None,None,None])+bias[:,:,None,None,None]
        return self.skip(x)+self.conv2(F.dropout(F.silu(h),p=self.dropout,training=self.training))


class ContextRefinement(nn.Module):
    """Identity-initialized extra global processing; no change to Gaussian outputs."""
    def __init__(self,channels):
        super().__init__()
        self.residual=Residual(channels,channels)
        self.norm=nn.GroupNorm(8,channels)
        self.qkv=nn.Conv1d(channels,channels*3,1)
        self.output=nn.Conv1d(channels,channels,1)
        nn.init.zeros_(self.residual.conv2.weight); nn.init.zeros_(self.residual.conv2.bias)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self,x,condition):
        x=self.residual(x,condition)
        b,c,t,h,w=x.shape
        q,k,v=self.qkv(self.norm(x).flatten(2)).chunk(3,1)
        def heads(value): return value.reshape(b,4,c//4,-1).transpose(-1,-2)
        update=F.scaled_dot_product_attention(heads(q),heads(k),heads(v))
        update=update.transpose(-1,-2).reshape(b,c,-1)
        return x+self.output(update).reshape_as(x)


class GaussianVideoDenoiser(nn.Module):
    def __init__(self,classes,width=32,channels=9,refinement_blocks=0,dropout=0.):
        super().__init__()
        self.config=dict(classes=classes,width=width,channels=channels,refinement_blocks=refinement_blocks,dropout=dropout)
        self.classes=classes
        self.time=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        self.label=nn.Embedding(classes+1,128)
        self.input=nn.Conv3d(channels,width,3,padding=1)
        self.r1=Residual(width,width,dropout)
        self.down1=nn.Conv3d(width,width*2,3,stride=(1,2,2),padding=1)
        self.r2=Residual(width*2,width*2,dropout)
        self.down2=nn.Conv3d(width*2,width*3,3,stride=(1,2,2),padding=1)
        self.mid1=Residual(width*3,width*3,dropout)
        self.attn_norm=nn.GroupNorm(8,width*3)
        self.qkv=nn.Conv1d(width*3,width*9,1)
        self.attn_out=nn.Conv1d(width*3,width*3,1)
        self.mid2=Residual(width*3,width*3,dropout)
        self.refinements=nn.ModuleList([ContextRefinement(width*3) for _ in range(refinement_blocks)])
        self.coordinates=nn.Conv3d(3,width*3,1) if refinement_blocks else None
        if self.coordinates is not None:
            nn.init.zeros_(self.coordinates.weight); nn.init.zeros_(self.coordinates.bias)
        self.u2=Residual(width*5,width*2,dropout)
        self.u1=Residual(width*3,width,dropout)
        self.output=nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv3d(width,channels,3,padding=1))

    def forward(self,x,t,label):
        condition=self.time(timestep_embedding(t))+self.label(label)
        a=self.r1(self.input(x),condition)
        b=self.r2(self.down1(a),condition)
        h=self.mid1(self.down2(b),condition)
        batch,channels,time,height,width=h.shape
        q,k,v=self.qkv(self.attn_norm(h).flatten(2)).chunk(3,1)
        def heads(value): return value.reshape(batch,4,channels//4,-1).transpose(-1,-2)
        attended=F.scaled_dot_product_attention(heads(q),heads(k),heads(v))
        attended=attended.transpose(-1,-2).reshape(batch,channels,-1)
        h=h+self.attn_out(attended).reshape_as(h)
        h=self.mid2(h,condition)
        if self.coordinates is not None:
            axes=[torch.linspace(-1,1,n,device=h.device,dtype=h.dtype) for n in h.shape[2:]]
            coordinates=torch.stack(torch.meshgrid(*axes,indexing='ij'),0)[None]
            h=h+self.coordinates(coordinates)
            for block in self.refinements: h=block(h,condition)
        h=self.u2(torch.cat((F.interpolate(h,size=b.shape[2:],mode='nearest'),b),1),condition)
        h=self.u1(torch.cat((F.interpolate(h,size=a.shape[2:],mode='nearest'),a),1),condition)
        return self.output(h)


def schedule(device,steps=1000):
    t=torch.arange(steps+1,device=device,dtype=torch.float64)/steps
    alphas=torch.cos((t+0.008)/1.008*math.pi/2).square()
    betas=(1-alphas[1:]/alphas[:-1]).clamp(0,0.999)
    return torch.cumprod(1-betas,0).float()


def guided_velocity(model,x,t,labels,guidance,distilled_guidance=None):
    """A distilled checkpoint folds fixed CFG into one conditional prediction."""
    if distilled_guidance is not None and abs(guidance-distilled_guidance)>1e-8:
        raise ValueError('Guidance differs from the fixed distilled guidance')
    with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=x.device.type=='cuda'):
        if distilled_guidance is not None:
            return model(x,t,labels).float()
        null=torch.full_like(labels,model.classes)
        both=model(torch.cat((x,x)),t.repeat(2),torch.cat((null,labels))).float()
    unconditional,conditional=both.chunk(2)
    return unconditional+guidance*(conditional-unconditional)


def ddim_step(x,velocity,alpha,next_alpha=None):
    a=alpha.sqrt(); s=(1-alpha).sqrt()
    predicted=(a*x-s*velocity).clamp(-4,4)
    if next_alpha is None: return predicted
    noise=s*x+a*velocity
    return next_alpha.sqrt()*predicted+(1-next_alpha).sqrt()*noise


@torch.no_grad()
def sample(model,labels,seed,steps=50,guidance=1.5,shape=(8,32,32),distilled_guidance=None,observer=None,velocity_fn=None):
    if distilled_guidance is not None and abs(guidance-distilled_guidance)>1e-8:
        raise ValueError('Guidance differs from the fixed distilled guidance')
    if velocity_fn is not None:
        if getattr(velocity_fn,'model',model) is not model or getattr(velocity_fn,'guidance',guidance)!=guidance or getattr(velocity_fn,'distilled_guidance',distilled_guidance)!=distilled_guidance:
            raise ValueError('Velocity runner does not match model/guidance settings')
    device=next(model.parameters()).device
    gen=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn(len(labels),model.config['channels'],*shape,device=device,generator=gen)
    alpha=schedule(device)
    times=torch.linspace(999,0,steps,device=device).long()
    for i,t in enumerate(times):
        tt=t.expand(len(labels))
        # Velocity-prediction diffusion. All outputs are normalized Gaussian parameters.
        velocity=guided_velocity(model,x,tt,labels,guidance,distilled_guidance) if velocity_fn is None else velocity_fn(x,tt,labels)
        if observer is not None: observer(x,tt,velocity)
        x=ddim_step(x,velocity,alpha[t],alpha[times[i+1]] if i+1<len(times) else None)
    return x
