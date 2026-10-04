"""Explicit 3D Gaussians, perspective projection, depth alpha blending and FK.

Reference PyTorch implementation, not an optimized production rasterizer.
Surface IDs persist when occluded. This module does not learn/generate motion.
"""
import math
import torch
import torch.nn.functional as F


def rotation(axis,angle):
    axis=F.normalize(axis,dim=-1); x,y,z=axis.unbind(-1); zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*axis.shape[:-1],3,3)
    eye=torch.eye(3,device=axis.device,dtype=axis.dtype).expand_as(skew)
    return eye+angle.sin()[...,None,None]*skew+(1-angle.cos())[...,None,None]*(skew@skew)


def forward_kinematics(rest,parents,local_rotation):
    """Global joint frames; rest translations ensure exact joint connectivity."""
    if rest.shape!=(len(parents),3) or local_rotation.shape!=(len(parents),3,3): raise ValueError('Joint shape mismatch')
    transforms=[]
    for i,parent in enumerate(parents):
        if parent>=i or parent< -1: raise ValueError('Parents must precede children')
        transform=torch.eye(4,device=rest.device,dtype=rest.dtype)
        transform[:3,:3]=local_rotation[i]
        transform[:3,3]=rest[i] if parent<0 else rest[i]-rest[parent]
        transforms.append(transform if parent<0 else transforms[parent]@transform)
    return torch.stack(transforms)


def skin(position,covariance,joints,parents,local_rotation,indices,weights):
    transforms=forward_kinematics(joints,parents,local_rotation)
    matrices=transforms[:,:3,:3]
    offsets=transforms[:,:3,3]-(matrices@joints[...,None]).squeeze(-1)
    r=matrices[indices]; t=offsets[indices]
    moved=((r@position[:,None,:,None]).squeeze(-1)+t)*weights[...,None]
    jac=(r*weights[...,None,None]).sum(1)
    return moved.sum(1),jac@covariance@jac.transpose(-1,-2),transforms


def look_at(eye,target):
    forward=F.normalize(target-eye,dim=0)
    if float((target-eye).detach().norm()) < 1e-8: raise ValueError('Camera eye and target coincide')
    reference=eye.new_tensor([0.,1.,0.])
    if abs(float(forward[1].detach())) > .999: reference=eye.new_tensor([0.,0.,1.])
    right=F.normalize(torch.linalg.cross(forward,reference),dim=0)
    up=torch.linalg.cross(right,forward)
    return torch.stack((right,-up,forward))


def project(position,covariance,eye,target,height,width,fov=42.,principal=None):
    camera=look_at(eye,target); p=(position-eye)@camera.T; z=p[:,2]
    focal=.5*height/math.tan(math.radians(fov)*.5); safe=z.clamp_min(.01)
    center=p.new_tensor([width/2,height/2]) if principal is None else torch.as_tensor(principal,device=p.device,dtype=p.dtype)
    mean=p[:,:2]/safe[:,None]*focal+center
    jac=p.new_zeros(len(p),2,3); jac[:,0,0]=focal/safe; jac[:,1,1]=focal/safe
    jac[:,0,2]=-focal*p[:,0]/safe.square(); jac[:,1,2]=-focal*p[:,1]/safe.square()
    cov=camera@covariance@camera.T
    screen=jac@cov@jac.transpose(-1,-2)+torch.eye(2,device=p.device)[None]*.16
    return mean,screen,z,camera,focal


def render(position,covariance,colour,opacity,eye,target,height=480,width=640,fov=42.,radius=4,ground=True,return_cache=False,principal=None):
    """Finite-stencil ellipses, sorted by pixel then Gaussian center depth.

    Includes full alpha transmittance, unlike prior normalized 2D blending.
    Truncated footprints and center-depth ordering are explicit approximations.
    """
    mean,cov,depth,camera,focal=project(position,covariance,eye,target,height,width,fov,principal)
    order=depth.argsort(stable=True); mean,cov,depth,colour,opacity=mean[order],cov[order],depth[order],colour[order],opacity[order]
    yy,xx=torch.meshgrid(torch.arange(height,device=eye.device)+.5,torch.arange(width,device=eye.device)+.5,indexing='ij')
    cx,cy=(width/2,height/2) if principal is None else principal
    rays=torch.stack(((xx-cx)/focal,(yy-cy)/focal,torch.ones_like(xx)),-1)@camera
    floor_t=-eye[1]/torch.where(rays[...,1].abs()>1e-7,rays[...,1],torch.ones_like(rays[...,1])*1e-7)
    floor_valid=(floor_t>0)&(rays[...,1]<-1e-7)&ground
    hit=eye+floor_t[...,None]*rays
    checker=((torch.floor(hit[...,0]*4)+torch.floor(hit[...,2]*4)).long()%2).float()
    bg=eye.new_tensor([.15,.19,.24]).expand(height,width,3).clone()
    floor_colour=(.28+.12*checker)[...,None]*eye.new_tensor([.9,1.,.85])
    bg=torch.where(floor_valid[...,None],floor_colour,bg)
    offsets=torch.arange(-radius,radius+1,device=eye.device)
    oy,ox=torch.meshgrid(offsets,offsets,indexing='ij')
    pixel=mean.floor().long()[:,None]+torch.stack((ox.flatten(),oy.flatten()),-1)[None]
    delta=pixel.float()+.5-mean[:,None]
    a,b,d=cov[:,0,0],cov[:,0,1],cov[:,1,1]; det=(a*d-b*b).clamp_min(1e-12)
    power=(d[:,None]*delta[...,0].square()-2*b[:,None]*delta[...,0]*delta[...,1]+a[:,None]*delta[...,1].square())/det[:,None]
    alpha=(opacity[:,None]*torch.exp(-.5*power)).clamp(0,.995)
    valid=(depth[:,None]>.02)&(pixel[...,0]>=0)&(pixel[...,0]<width)&(pixel[...,1]>=0)&(pixel[...,1]<height)&(alpha>1e-4)
    flat=pixel[...,1].clamp(0,height-1)*width+pixel[...,0].clamp(0,width-1)
    if ground: valid&=(~floor_valid.flatten()[flat])|(depth[:,None]<=floor_t.flatten()[flat])
    ids=torch.arange(len(position),device=eye.device)[:,None].expand_as(flat)[valid]
    pix=flat[valid]; al=alpha[valid]
    sort=(pix*(len(position)+1)+ids).argsort(); pix,ids,al=pix[sort],ids[sort],al[sort]
    # Float64 segmented prefix avoids cancellation between different pixels.
    log=torch.log1p(-al.double()); prefix=torch.cat((log.new_zeros(1),log.cumsum(0)))
    start=torch.ones_like(pix,dtype=torch.bool); start[1:]=pix[1:]!=pix[:-1]
    bases=log.new_zeros(height*width); bases[pix[start]]=prefix[:-1][start]
    trans=torch.exp(prefix[:-1]-bases[pix]); weight=(al*trans).float()
    rgb=position.new_zeros(height*width,3).index_add_(0,pix,weight[:,None]*colour[ids])
    alpha_sum=position.new_zeros(height*width).index_add_(0,pix,weight)
    image=rgb.reshape(height,width,3)+(1-alpha_sum.reshape(height,width,1))*bg
    if return_cache:
        # Fixed geometry gives a linear colour operator. Original IDs, not
        # depth-sort ranks, are returned so appearance fitting preserves memory.
        return image.clamp(0,1),alpha_sum.reshape(height,width),dict(pixel=pix,ids=order[ids],weight=weight)
    return image.clamp(0,1),alpha_sum.reshape(height,width)
