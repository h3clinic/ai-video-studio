"""Small differentiable orthographic renderer; not an optimized 3DGS engine."""
import numpy as np
import torch


def render(state,size=96,extent=1.8):
    # Input is one scene; returns H,W,3 float RGB. View along +z; smaller z is near.
    p,R,s,color,alpha=(state[k] for k in ('p','R','scale','color','alpha'))
    device=p.device
    xy=torch.linspace(-extent,extent,size,device=device)
    yy,xx=torch.meshgrid(xy.flip(0),xy,indexing='ij')
    grid=torch.stack((xx,yy),-1)
    cov=(R*s[:,None,:].square())@R.transpose(-1,-2)
    cov2=cov[:,:2,:2]+torch.eye(2,device=device)[None]*(extent/size*0.45)**2
    inv=torch.linalg.inv(cov2)
    delta=grid[None]-p[:,None,None,:2]
    power=torch.einsum('nhwi,nij,nhwj->nhw',delta,inv,delta)
    a=(torch.exp(-0.5*power)*alpha[:,0,None,None]).clamp(max=0.98)
    order=p[:,2].argsort()
    a=a[order]
    color=color[order]
    transmission=torch.cumprod(torch.cat((torch.ones_like(a[:1]),1-a+1e-8),0),0)
    weights=a*transmission[:-1]
    bg=p.new_tensor([0.018,0.025,0.045])
    return (torch.einsum('nhw,nc->hwc',weights,color)+transmission[-1,:,:,None]*bg).clamp(0,1)


def as_image(tensor):
    from PIL import Image
    return Image.fromarray((tensor.detach().cpu().numpy()*255).round().astype(np.uint8))
