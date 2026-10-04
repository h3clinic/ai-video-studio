"""Anchored surface transition-band experiment, not learned hole filling.

Existing connectivity is retained. No artificial hole, mirroring, or symmetry
constraint. Only explicitly selected vertices can move. Gaussian IDs persist.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch
from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake

def fair_band(vertices,faces,editable,screen=.03,max_displacement=.02):
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces,dtype=np.int64)
    mask=np.asarray(editable,dtype=bool)
    if mask.shape!=(len(v),) or not mask.any() or mask.all():raise ValueError('A nonempty band and fixed context are required')
    if screen<=0 or max_displacement<=0:raise ValueError('Positive screening and displacement cap required')
    if not np.isfinite(v).all():raise ValueError('Nonfinite geometry')
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);edges=np.concatenate([edges,edges[:,::-1]])
    adj=sparse.coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(v),len(v))).tocsr()
    adj.data[:]=1;degree=np.asarray(adj.sum(1)).ravel()
    lap=sparse.eye(len(v),format='csr')-sparse.diags(1/np.maximum(degree,1))@adj
    free=np.flatnonzero(mask);A=lap[:,free];rows=np.unique(A.nonzero()[0]);A=A[rows]
    # min ||L V_new||^2 + screen ||V_free_new - V_free_old||^2
    rhs=A@v[free]-(lap@v)[rows]
    normal=A.T@A+screen*sparse.eye(len(free),format='csr')
    proposed=spsolve(normal.tocsc(),A.T@rhs+screen*v[free])
    delta=proposed-v[free];norm=np.linalg.norm(delta,axis=1)
    delta*=np.minimum(1,max_displacement/np.maximum(norm,1e-12))[:,None]
    affected=mask[f].any(1);old=v[f[affected]]
    old_cross=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);old_area=np.linalg.norm(old_cross,axis=1)
    valid=old_area>1e-12;factor=1.
    for _ in range(18):
        result=v.copy();result[free]+=factor*delta
        tri=result[f[affected]];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        area=np.linalg.norm(cross,axis=1)
        if np.all((cross[valid]*old_cross[valid]).sum(1)>0) and np.all(area[valid]>.25*old_area[valid]):break
        factor*=.5
    else:raise ValueError('No non-flipping bounded band update')
    result=result.astype(np.asarray(vertices).dtype)
    assert np.array_equal(result[~mask],np.asarray(vertices)[~mask])
    return result,dict(editable_vertices=int(mask.sum()),affected_faces=int(affected.sum()),step_factor=factor,
        max_displacement=float(np.linalg.norm(result-v,axis=1).max()),outside_vertices_exact=True,
        flipped_faces=0,minimum_area_ratio=float((area[valid]/old_area[valid]).min()),
        laplacian_energy_before=float(np.square((lap@v)[rows]).sum()),
        laplacian_energy_after=float(np.square((lap@result)[rows]).sum()),
        screen=screen,self_intersections_checked=False,topology_unchanged=True)

def transport(asset,old,new,faces):
    from .surface_skin3d import triangle_basis
    out={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in asset.items()}
    owned=asset['face_id'].long();changed=(old!=new).any(1)[faces].any(1);selected=changed[owned]
    face=faces[owned[selected]];b0,area=triangle_basis(old[face]);b1,_=triangle_basis(new[face])
    if bool((area<1e-12).any()):raise ValueError('Degenerate occupied triangle')
    gradient=b1@torch.linalg.inv(b0)
    out['position'][selected]=(new[face]*asset['barycentric'][selected,:,None]).sum(1)
    out['covariance'][selected]=gradient@asset['covariance'][selected]@gradient.transpose(-1,-2)
    normal=b1[:,:,2];out['normal'][selected]=normal
    assert torch.equal(out['position'][~selected],asset['position'][~selected])
    out['scope']='Anchored deterministic facial-band fairing; persistent IDs and colors; no learned fill or motion.'
    return out,int(selected.sum())

def main(args):
    base=Path('artifacts/real_video/hunyuan_gaussian');args.root.mkdir(parents=True,exist_ok=False)
    shape=load_verified(base/'v4_full/shape.pt');asset=load_verified(base/'v5_full_paint/gaussian_fitted.pt')
    v=shape['vertices'];f=shape['faces'];lo,hi=v.amin(0),v.amax(0);span=hi-lo
    # Experiment selection only: this asset faces +X and has Y-up.
    center=(lo[2]+hi[2])/2
    mask=(v[:,0]>lo[0]+.76*span[0])&(v[:,1]>lo[1]+.55*span[1])&((v[:,2]-center).abs()<.12*span[2])
    start=time.perf_counter();new,report=fair_band(v.numpy(),f.numpy(),mask.numpy(),args.screen,.025)
    new=torch.from_numpy(new);out,count=transport(asset,v,new,f)
    save_inference_checkpoint(out,args.root/'gaussian_fitted.pt')
    save_inference_checkpoint(dict(vertices=new,faces=f,scope=out['scope']),args.root/'shape.pt')
    report.update(seconds=time.perf_counter()-start,affected_gaussians=count,gaussians=len(out['position']),
        original_shape_sha256=digest(base/'v4_full/shape.pt'),accepted=False,
        region='Manual central facial band; no automatic seam detection',
        limitations=['Existing topology retained, not missing-face generation.','No learned anatomy, no texture regeneration.',
                    'No actual half seam in source. No global self-intersection guarantee.'])
    (args.root/'weld_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    from . import hunyuan_multiview as viewer
    viewer.ROOT=args.root;viewer.inspect()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--screen',type=float,default=.03)
    args=parser.parse_args();torch.set_num_threads(4)
    with keep_windows_awake():main(args)
