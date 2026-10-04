"""New middle-surface layer from two fixed side cross-sections.

Controlled reconstruction diagnostic, not learned anatomical completion. The
two-slice interpolator excludes middle samples. The optional support guard DOES
consult the original middle surface and colors, so it is not unseen completion.
Original splats persist, with opacity faded in the supported replacement region.
"""
import json
import argparse
import time
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import torch
import trimesh
from .checkpoint_io import load_verified,save_inference_checkpoint,keep_windows_awake,digest
from .hunyuan_appearance import sample_surface

ROOT=Path('artifacts/real_video/hunyuan_gaussian/v12_middle_surface')

def middle_field(left,right,steps,pitch):
    if left.shape!=right.shape or steps<3 or pitch<=0:raise ValueError('Invalid boundary fields')
    def sdf(mask):
        mask=ndimage.binary_fill_holes(mask)
        if not mask.any() or mask.all():raise ValueError('Boundary needs inside and outside')
        return (ndimage.distance_transform_edt(~mask)-ndimage.distance_transform_edt(mask))*pitch
    a,b=sdf(left),sdf(right);t=np.linspace(0,1,steps)
    return ((1-t)*a[...,None]+t*b[...,None]).astype(np.float32)

def replacement_weight(p,cutx,cuty,center,halfwidth,collar):
    w=torch.minimum((p[:,0]-cutx)/collar,(p[:,1]-cuty)/collar)
    w=torch.minimum(w,(halfwidth-(p[:,2]-center).abs())/collar).clamp(0,1)
    return w*w*(3-2*w)

def distance_confidence(distance,inner=.008,outer=.025):
    """Conservative source-prior support, NOT a learned anatomical confidence."""
    if not 0<=inner<outer:raise ValueError('Invalid confidence radii')
    t=np.clip((np.asarray(distance)-inner)/(outer-inner),0,1)
    return 1-t*t*(3-2*t)

def support_weights(old_position,new_position,old_weight,new_weight):
    """Never erase an old splat when no supported new splat covers it."""
    if len(new_position)==0:return torch.zeros_like(old_weight),torch.zeros_like(new_weight)
    distance,_=cKDTree(old_position.numpy()).query(new_position.numpy())
    new_weight=new_weight*torch.tensor(distance_confidence(distance),dtype=new_weight.dtype)
    valid=new_weight>1e-5
    if not valid.any():return torch.zeros_like(old_weight),new_weight
    distance,index=cKDTree(new_position[valid].numpy()).query(old_position.numpy())
    coverage=new_weight[valid][torch.tensor(index)]*torch.tensor(distance_confidence(distance),dtype=old_weight.dtype)
    return torch.minimum(old_weight,coverage),new_weight

def main(solid=False,half_fraction=.23,guarded=False):
    if not 0<half_fraction<.5:raise ValueError('Invalid band width')
    ROOT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    base=ROOT.parent;shape=load_verified(base/'v4_full/shape.pt');old=load_verified(base/'v5_full_paint/gaussian_fitted.pt')
    v=shape['vertices'];lo,hi=v.amin(0),v.amax(0);span=hi-lo
    cutx=float(lo[0]+.74*span[0]);cuty=float(lo[1]+.51*span[1]);center=float((lo[2]+hi[2])/2)
    halfwidth=float(span[2])*half_fraction;pitch=.005;collar=min(.025,halfwidth*.6)
    mesh=trimesh.Trimesh(v.numpy(),shape['faces'].numpy(),process=False)
    vox=mesh.voxelized(pitch)
    if solid:vox=vox.fill()
    origin=vox.transform[:3,3]
    left=int(round((center-halfwidth-origin[2])/pitch));right=int(round((center+halfwidth-origin[2])/pitch))
    field=middle_field(vox.matrix[:,:,left],vox.matrix[:,:,right],right-left+1,pitch)
    vertices,faces,_,_=marching_cubes(field,0,spacing=(pitch,pitch,pitch))
    vertices+=origin+np.array([0,0,left*pitch])
    tri=vertices[faces];select=(tri[:,:,0].max(1)>cutx)&(tri[:,:,1].max(1)>cuty)
    faces=faces[select]
    patch=sample_surface(vertices,faces,80000);patch['covariance']*=(.8/.62)**2
    op=old['position'];p=patch['position'];oldw=replacement_weight(op,cutx,cuty,center,halfwidth,collar)
    neww=replacement_weight(p,cutx,cuty,center,halfwidth,collar)
    raw_oldw=oldw.clone()
    if guarded:oldw,neww=support_weights(op,p,oldw,neww)
    usable=neww>1e-5
    for key in patch:patch[key]=patch[key][usable]
    p=patch['position'];neww=neww[usable]
    # Legacy ablation uses side-only appearance. Guarded repair reuses observed
    # center colors too: it is not a held-out missing-region completion claim.
    side=(op[:,0]>cutx-collar)&(op[:,1]>cuty-collar)&((op[:,2]-center).abs()>=halfwidth)
    if int(side.sum())<4:raise ValueError('Insufficient retained side context')
    distances,nearest=cKDTree(op[side].numpy()).query(p.numpy(),k=4)
    weights=1/np.maximum(distances,1e-5)**2;weights/=weights.sum(1,keepdims=True)
    patch['colour']=(old['colour'][side][torch.tensor(nearest)]*torch.tensor(weights,dtype=torch.float32)[...,None]).sum(1)
    if guarded:
        distance,index=cKDTree(op.numpy()).query(p.numpy(),k=16)
        index=torch.tensor(index)
        agreement=(old['normal'][index]*patch['normal'][:,None]).sum(-1).abs()
        score=torch.tensor(distance,dtype=torch.float32)+.02*(1-agreement)
        chosen=index[torch.arange(len(p)),score.argmin(1)]
        patch['colour']=old['colour'][chosen].clone()
    patch['opacity']*=neww
    out={k:old[k].clone() for k in ['position','covariance','colour','opacity','normal','ids']}
    out['opacity']*=1-oldw
    for key in ['position','covariance','colour','opacity','normal']:out[key]=torch.cat([out[key],patch[key]])
    out['ids']=torch.cat([old['ids'],torch.arange(len(p))+int(old['ids'].max())+1])
    out['group']=torch.cat([torch.zeros(len(op),dtype=torch.int16),torch.ones(len(p),dtype=torch.int16)])
    out['original_count']=len(out['position']);out['camera_pivot']=old['camera_pivot'];out['camera_distance']=old['camera_distance']
    out['scope']='Experimental middle surface; static, no learned fill. '+('Source-supported replacement and source appearance reuse.' if guarded else 'Side-only color interpolation.')
    outside=oldw==0
    for key in ['position','covariance','colour','opacity','normal','ids']:
        assert torch.equal(out[key][:len(op)][outside],old[key][outside]),key
    save_inference_checkpoint(out,ROOT/'gaussian_fitted.pt')
    save_inference_checkpoint(dict(vertices=torch.tensor(vertices),faces=torch.tensor(faces),scope='Middle patch only, open along join boundaries'),ROOT/'patch.pt')
    report=dict(accepted=False,source_sha256=digest(base/'v5_full_paint/gaussian_fitted.pt'),
        retained_original_gaussians=len(op),new_middle_gaussians=len(p),opacity_modified_originals=int((oldw>0).sum()),
        outside_render_attributes_exact=True,original_positions_colors_covariances_exact=True,
        solid_voxel_fill=solid,
        half_fraction=half_fraction,guarded=guarded,
        originals_preserved_by_support=int(((raw_oldw>0)&(oldw==0)).sum()),
        seconds=time.perf_counter()-start,pitch=pitch,side_slice_indices=[left,right],
        method='Linear interpolation of signed-distance cross-sections; zero-level new surface. Not trained model.',
        limitations=['Controlled middle replacement on Hunyuan asset, not located historic mirrored asset.',
                    'No original center samples passed to middle_field. Boundary masks inferred from voxelized sides.',
                    'Solid fill, when enabled, uses source mesh closure; this is not held-out middle reconstruction.',
                    'Opacity crossfade is not a watertight topological weld.',
                    'Guarded variant reuses original middle geometry as a support prior and original colors; not unseen-region generation.',
                    'Anatomy unverified. Opacity confidence is heuristic, not calibrated probability.'])
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    from . import hunyuan_multiview as viewer
    viewer.ROOT=ROOT;viewer.inspect()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--solid',action='store_true')
    parser.add_argument('--half-fraction',type=float,default=.23)
    parser.add_argument('--guarded',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();ROOT=args.output
    torch.set_num_threads(4)
    with keep_windows_awake():main(args.solid,args.half_fraction,args.guarded)
