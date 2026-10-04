import torch
from torch import Tensor
import torch.nn.functional as F


def skew(v: Tensor) -> Tensor:
    x,y,z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*v.shape[:-1],3,3)


def exp_so3(v: Tensor) -> Tensor:
    """Stable Rodrigues map, including differentiable zero rotation."""
    angle = v.norm(dim=-1,keepdim=True)
    K = skew(v)
    # sinc(x) = sin(pi*x)/(pi*x); avoids 0/0 and hard Taylor-branch gradients.
    a = torch.sinc(angle / torch.pi)[...,None]
    b = (0.5 * torch.sinc(angle / (2*torch.pi)).square())[...,None]
    eye = torch.eye(3,device=v.device,dtype=v.dtype)
    return eye + a*K + b*(K@K)


def rotation_6d(x: Tensor) -> Tensor:
    """Continuous network output projected to right-handed frames (columns)."""
    a,b = x[...,:3],x[...,3:]
    e1 = F.normalize(a,dim=-1,eps=1e-7)
    e2 = F.normalize(b-(e1*b).sum(-1,keepdim=True)*e1,dim=-1,eps=1e-7)
    return torch.stack((e1,e2,torch.linalg.cross(e1,e2,dim=-1)),dim=-1)


def local(R: Tensor, v: Tensor) -> Tensor:
    return (R.transpose(-1,-2)@v.unsqueeze(-1)).squeeze(-1)


def world(R: Tensor, v: Tensor) -> Tensor:
    return (R@v.unsqueeze(-1)).squeeze(-1)


def frame_error(R: Tensor) -> dict:
    eye = torch.eye(3,device=R.device,dtype=R.dtype)
    return {'orthogonality_max':(R.transpose(-1,-2)@R-eye).abs().max().item(),
            'determinant_error_max':(torch.linalg.det(R)-1).abs().max().item()}
