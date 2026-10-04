"""Persistent flow-tracked planar Gaussian fields. Not recovered 3D scene geometry."""
import numpy as np
import torch
import torch.nn.functional as F

GRID=32
SIZE=64
FRAMES=8
CHANNELS=9


def mirror_fields(fields):
    """Horizontal reflection of physical Gaussian fields, including local axes."""
    result=fields.flip(-1).clone()
    result[:,0]=-result[:,0]
    result[:,4]=-result[:,4]
    return result


def refresh_coverage_positions(px, py, image_size, grid):
    """Observed-frame reseeding, NOT material tracking or unseen-content generation.

    Retain one existing track per occupied sampling cell. Reuse duplicate slots
    at missing cell centers; callers must increment those slots' track versions.
    This is a development reconstruction writer for disocclusion, not a prior.
    """
    px, py = np.asarray(px), np.asarray(py)
    if (px.shape != (grid, grid) or py.shape != px.shape
            or not np.isfinite(px).all() or not np.isfinite(py).all()):
        raise ValueError('Finite grid positions required')
    if type(image_size) is not int or type(grid) is not int or image_size < 1 or grid < 1:
        raise ValueError('Positive integer image/grid sizes required')
    cx = np.floor((px.flatten()+.5)*grid/image_size).astype(np.int64).clip(0,grid-1)
    cy = np.floor((py.flatten()+.5)*grid/image_size).astype(np.int64).clip(0,grid-1)
    cells = cy*grid+cx
    order = np.argsort(cells, kind='stable')
    first = np.r_[True, np.diff(cells[order]) != 0]
    redundant = order[~first]
    missing = np.setdiff1d(np.arange(grid*grid), cells[order[first]], assume_unique=True)
    if len(redundant) != len(missing):
        raise AssertionError('Coverage reassignment lost slots')
    new_x, new_y = px.copy().reshape(-1), py.copy().reshape(-1)
    new_x[redundant] = (missing%grid+.5)*image_size/grid-.5
    new_y[redundant] = (missing//grid+.5)*image_size/grid-.5
    reset = np.zeros(grid*grid, dtype=bool)
    reset[redundant] = True
    return new_x.reshape(px.shape), new_y.reshape(py.shape), reset.reshape(px.shape)


def encode_clip(frames,grid=GRID,*,refresh_coverage=False,return_generations=False):
    """Fixed optical-flow analysis encoder used only to create training targets.

    Channels: displacement x/y in units of 16 pixels, log scales x/y,
    direction cos/sin, RGB mapped to [-1,1]. All dots persist over the clip.
    """
    import cv2
    if frames.ndim!=4 or frames.shape[1]!=frames.shape[2] or frames.shape[-1]!=3:
        raise ValueError('Expected square RGB video')
    image_size=frames.shape[1]
    # Preserve the established canonical 64-pixel coordinate system at all sizes.
    base=(np.arange(grid,dtype=np.float32)+0.5)*image_size/grid-0.5
    yy,xx=np.meshgrid(base,base,indexing='ij')
    px,py=xx.copy(),yy.copy()
    fields=[]
    generations=np.zeros((grid,grid),dtype=np.int32)
    generation_history=[]
    previous=None
    for frame in frames:
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        if previous is not None:
            flow=cv2.calcOpticalFlowFarneback(previous,gray,None,0.5,3,15,3,5,1.2,0)
            vx=cv2.remap(flow[:,:,0],px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            vy=cv2.remap(flow[:,:,1],px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            px=np.clip(px+vx,0,image_size-1); py=np.clip(py+vy,0,image_size-1)
        if refresh_coverage:
            px,py,reset=refresh_coverage_positions(px,py,image_size,grid)
            generations+=reset.astype(np.int32)
        if return_generations:
            generation_history.append(generations.copy())
        rgb=cv2.remap(frame.astype(np.float32)/255,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        gx=cv2.Sobel(gray.astype(np.float32)/255,cv2.CV_32F,1,0,ksize=3)/8
        gy=cv2.Sobel(gray.astype(np.float32)/255,cv2.CV_32F,0,1,ksize=3)/8
        gx=cv2.remap(gx,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        gy=cv2.remap(gy,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        strength=np.sqrt(gx*gx+gy*gy)
        # Tangent to the local image edge; background gets the canonical unit axis.
        cosine=np.where(strength>0.01,-gy/np.maximum(strength,1e-6),1.)
        sine=np.where(strength>0.01,gx/np.maximum(strength,1e-6),0.)
        contrast=np.clip(strength*5,0,1)
        scale_x=(1.15+0.4*contrast)*GRID/grid; scale_y=(1.05-0.35*contrast)*GRID/grid
        fields.append(np.stack(((px-xx)/(image_size/4),(py-yy)/(image_size/4),np.log(scale_x),np.log(scale_y),
                                cosine,sine,*(rgb*2-1).transpose(2,0,1)),0))
        previous=gray
    result=np.stack(fields,axis=1).astype(np.float32)
    if return_generations:
        return result,np.stack(generation_history)
    return result


def decode_state(field,min_log_scale=-0.8):
    """B,C,H,W raw field -> B,N explicit positions, unit frames, scale and RGB."""
    b,channels,g,_=field.shape
    if channels!=CHANNELS: raise ValueError('Expected nine Gaussian channels')
    x=field.flatten(2).transpose(1,2)
    grid=(torch.arange(g,device=x.device,dtype=x.dtype)+0.5)*SIZE/g-0.5
    yy,xx=torch.meshgrid(grid,grid,indexing='ij')
    base=torch.stack((xx,yy),-1).reshape(1,-1,2)
    p=base+16*x[:,:,:2].clamp(-4,4)
    direction=x[:,:,4:6]
    norm=direction.norm(dim=-1,keepdim=True)
    canonical=torch.zeros_like(direction); canonical[:,:,0]=1
    direction=torch.where(norm>1e-6,direction/norm.clamp_min(1e-6),canonical)
    c,s=direction.unbind(-1)
    zero=torch.zeros_like(c); one=torch.ones_like(c)
    R=torch.stack((c,-s,zero,s,c,zero,zero,zero,one),-1).reshape(b,-1,3,3)
    return dict(p=torch.cat((p,zero[:,:,None]),-1),R=R,
                scale=torch.cat((x[:,:,2:4].clamp(min_log_scale,0.7).exp(),one[:,:,None]),-1),
                color=((x[:,:,6:9]+1)/2).clamp(0,1))


def render_fields(fields,size=SIZE,radius=5,canonical_axes=False,min_log_scale=-0.8):
    """Normalized planar Gaussian splatting, local support; no pixel/video bypass.

    fields: B,C,T,G,G -> B,T,3,S,S. No depth, opacity ordering, neural pixel decoder.
    Mean locations have differentiable subpixel coordinates; support indices are discrete.
    """
    b,c,t,g,_=fields.shape
    state=decode_state(fields.permute(0,2,1,3,4).reshape(b*t,c,g,g),min_log_scale=min_log_scale)
    # Scale pixel centres, not corner coordinates, when changing resolution.
    p=state['p'][...,:2] if size==SIZE else (state['p'][...,:2]+0.5)*(size/SIZE)-0.5
    scale=state['scale'][...,:2]*(size/SIZE)
    R=state['R']
    if canonical_axes:
        R=torch.eye(3,device=fields.device,dtype=fields.dtype).expand_as(R)
    yy,xx=torch.meshgrid(torch.arange(-radius,radius+1,device=p.device),torch.arange(-radius,radius+1,device=p.device),indexing='ij')
    offsets=torch.stack((xx.flatten(),yy.flatten()),-1)
    pixels=p.floor().long()[:,:,None,:]+offsets[None,None]
    delta=pixels.to(p.dtype)-p[:,:,None,:]
    local=torch.einsum('bnki,bnij->bnkj',delta,R[...,:2,:2])
    weights=torch.exp(-0.5*(local/scale[:,:,None,:]).square().sum(-1))
    valid=(pixels[...,0]>=0)&(pixels[...,0]<size)&(pixels[...,1]>=0)&(pixels[...,1]<size)
    weights=weights*valid
    indices=(pixels[...,1].clamp(0,size-1)*size+pixels[...,0].clamp(0,size-1)).reshape(b*t,-1)
    weights=weights.reshape(b*t,-1)
    denominator=weights.new_zeros(b*t,size*size).scatter_add(1,indices,weights)
    colors=(state['color'][:,:,None,:]*weights.reshape(b*t,g*g,-1,1)).reshape(b*t,-1,3)
    numerator=fields.new_zeros(b*t,size*size,3).scatter_add(1,indices[:,:,None].expand(-1,-1,3),colors)
    image=(numerator/denominator.clamp_min(1e-6)[:,:,None]).reshape(b,t,size,size,3).permute(0,1,4,2,3)
    return image.clamp(0,1)


def render_fields_chunked(fields,frame_chunk=1,**kwargs):
    """Same splats, bounded frame-wise intermediates; still stores the input clip.

    Does not make the joint diffusion prior recurrent or its memory horizon-free.
    """
    if frame_chunk<1: raise ValueError('frame_chunk must be positive')
    return torch.cat([render_fields(fields[:,:,i:i+frame_chunk],**kwargs)
                      for i in range(0,fields.shape[2],frame_chunk)],dim=1)
