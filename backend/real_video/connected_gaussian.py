"""Persistent planar triangle attachments with ARAP and a no-flip line search.

Engineering adaptation of arXiv:2402.04796 and 2312.14937, not their reproduction.
Connectivity is image-space, NOT an anatomical or 3D reconstruction guarantee.
"""
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import factorized


def build_surface(points, spacing):
    """Build only occupied grid triangles; each Gaussian has a fixed material ID.

    Grid vertices can join nearby unrelated visible parts. No unseen surface is
    inferred. Construction uses the observed Gaussian packet only.
    """
    if spacing <= 0 or points.ndim != 2 or points.shape[1] != 10 or not torch.isfinite(points).all():
        raise ValueError('Finite N,10 points and positive spacing required')
    xy=points[:,:2].detach().cpu().double().numpy()
    cell=np.floor(xy/spacing).astype(np.int64); uv=xy/spacing-cell
    lower=uv.sum(1)<=1
    offsets=np.where(lower[:,None,None],np.array([[0,0],[1,0],[0,1]]),np.array([[1,1],[0,1],[1,0]]))
    triples=cell[:,None]+offsets
    vertices,inverse=np.unique(triples.reshape(-1,2),axis=0,return_inverse=True)
    faces,face_id=np.unique(inverse.reshape(-1,3),axis=0,return_inverse=True)
    weights=np.where(lower[:,None],np.stack((1-uv.sum(1),uv[:,0],uv[:,1]),1),np.stack((uv.sum(1)-1,1-uv[:,0],1-uv[:,1]),1))
    rest=vertices*spacing
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    basis=np.stack((rest[faces[:,1]]-rest[faces[:,0]],rest[faces[:,2]]-rest[faces[:,0]]),-1)
    return dict(points=points.detach().cpu().clone(),rest=torch.tensor(rest,dtype=torch.float32),
                faces=torch.tensor(faces,dtype=torch.int32),face_id=torch.tensor(face_id,dtype=torch.int32),
                bary=torch.tensor(weights[:,:2],dtype=torch.float32),edges=torch.tensor(edges,dtype=torch.int32),
                inverse_basis=torch.tensor(np.linalg.inv(basis),dtype=torch.float32),spacing=float(spacing),
                identity='Gaussian ID is immutable row index within this asset; no sorting or deletion')


class ConnectedSurface:
    """Local/global 2D ARAP with fixed rest edges and a positive-Jacobian guard."""
    def __init__(self,asset,stiffness=4.,iterations=8,guard='local'):
        self.rest=asset['rest'].double().numpy().copy()
        self.faces=asset['faces'].long().numpy()
        self.edges=asset['edges'].long().numpy()
        self.inv=asset['inverse_basis'].double().numpy()
        self.stiffness=stiffness; self.iterations=iterations
        if guard not in ['local','global']: raise ValueError('Unknown guard')
        self.guard=guard
        self.previous=self.rest.copy()
        a,b=self.edges.T; n=len(self.rest)
        self.a,self.b=a,b; self.edge=self.rest[a]-self.rest[b]
        lap=sparse.coo_matrix((np.concatenate((np.ones(2*len(a)),-np.ones(2*len(a)))),
                             (np.concatenate((a,b,a,b)),np.concatenate((a,b,b,a)))),shape=(n,n)).tocsc()
        self.solve=factorized(sparse.eye(n,format='csc')+stiffness*lap)

    def jacobians(self,vertices):
        f=self.faces
        basis=np.stack((vertices[f[:,1]]-vertices[f[:,0]],vertices[f[:,2]]-vertices[f[:,0]]),-1)
        return basis@self.inv

    def quality(self,vertices):
        jac=self.jacobians(vertices); sv=np.linalg.svd(jac,compute_uv=False)
        return dict(min_determinant=float(np.linalg.det(jac).min()),min_stretch=float(sv.min()),max_stretch=float(sv.max()))

    def step(self,target):
        target=np.asarray(target,dtype=np.float64)
        if target.shape!=self.rest.shape or not np.isfinite(target).all(): raise ValueError('Invalid vertex proposal')
        x=target.copy(); a,b=self.a,self.b
        for _ in range(self.iterations):
            current=x[a]-x[b]
            dot=(current*self.edge).sum(1); cross=self.edge[:,0]*current[:,1]-self.edge[:,1]*current[:,0]
            c=np.zeros(len(x)); s=np.zeros(len(x))
            np.add.at(c,a,dot); np.add.at(c,b,dot); np.add.at(s,a,cross); np.add.at(s,b,cross)
            norm=np.hypot(c,s); valid=norm>1e-15
            c=np.where(valid,c/np.maximum(norm,1e-15),1); s=np.where(valid,s/np.maximum(norm,1e-15),0)
            ce=(c[a]+c[b])*.5; se=(s[a]+s[b])*.5
            rotated=np.stack((ce*self.edge[:,0]-se*self.edge[:,1],se*self.edge[:,0]+ce*self.edge[:,1]),1)
            rhs=target.copy(); np.add.at(rhs,a,self.stiffness*rotated); np.add.at(rhs,b,-self.stiffness*rotated)
            x=self.solve(rhs)
        if self.guard=='local':
            # A rigidly transported previous surface is feasible and preserves
            # root motion even when the proposed non-rigid deformation fails.
            old=self.previous-self.previous.mean(0); new=x-x.mean(0)
            theta=np.arctan2((old[:,0]*new[:,1]-old[:,1]*new[:,0]).sum(),(old*new).sum())
            c,s=np.cos(theta),np.sin(theta); rigid=old@np.array([[c,s],[-s,c]])+x.mean(0)
            fraction=np.ones(len(x)); accepted=False
            for _ in range(24):
                candidate=rigid+fraction[:,None]*(x-rigid)
                jac=self.jacobians(candidate); sv=np.linalg.svd(jac,compute_uv=False)
                bad=(np.linalg.det(jac)<.2)|(sv.min(1)<.45)|(sv.max(1)>2.2)
                if not bad.any(): accepted=True; break
                fraction[np.unique(self.faces[bad])]*=.5
            if not accepted: candidate=rigid; fraction[:]=0
            q=self.quality(candidate); self.previous=candidate.copy()
            return candidate,dict(q,accepted_fraction=float(fraction.mean()),min_local_fraction=float(fraction.min()),
                                  rigid_fallback=not accepted,proposal_correction=float(np.linalg.norm(candidate-target,axis=1).mean()))
        accepted=False
        for attempt in range(13):
            fraction=2.**(-attempt); candidate=self.previous+fraction*(x-self.previous); q=self.quality(candidate)
            if q['min_determinant']>=.2 and q['min_stretch']>=.45 and q['max_stretch']<=2.2:
                accepted=True; break
        if not accepted: candidate=self.previous.copy(); fraction=0.; q=self.quality(candidate)
        self.previous=candidate.copy()
        return candidate,dict(q,accepted_fraction=fraction,proposal_correction=float(np.linalg.norm(candidate-target,axis=1).mean()))


def deform_surface(asset,vertices):
    """Gaussian means and covariances are transported, not an RGB image warp."""
    points=asset['points']; faces=asset['faces'].long(); ids=asset['face_id'].long()
    if vertices.shape!=asset['rest'].shape or not torch.isfinite(vertices).all(): raise ValueError('Invalid vertices')
    tri=vertices[faces]; basis=torch.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),-1)
    jac=basis@asset['inverse_basis']
    if (torch.linalg.det(jac)<=0).any(): raise ValueError('Collapsed or reversed triangle')
    bary=torch.cat((asset['bary'],1-asset['bary'].sum(1,keepdim=True)),1)
    output=points.clone(); output[:,:2]=(tri[ids]*bary[:,:,None]).sum(1)
    u=points[:,4:6]; u=u/u.norm(dim=1,keepdim=True).clamp_min(1e-12)
    v=torch.stack((-u[:,1],u[:,0]),1); axes=torch.stack((u,v),-1)
    covariance=(axes*points[:,2:4].exp().square()[:,None])@axes.transpose(-1,-2)
    transform=jac[ids]; covariance=transform@covariance@transform.transpose(-1,-2)
    # Closed-form symmetric 2x2 eigensystem avoids vendor limits on large
    # batches of tiny matrices (the cat has 95,704 splats).
    a,b,d=covariance[:,0,0],covariance[:,0,1],covariance[:,1,1]
    delta=torch.sqrt((a-d).square()+4*b.square())
    values=torch.stack(((a+d+delta)*.5,(a+d-delta)*.5),1)
    anisotropic=delta>1e-16
    theta=.5*torch.atan2(torch.where(anisotropic,2*b,torch.zeros_like(b)),torch.where(anisotropic,a-d,torch.ones_like(a)))
    output[:,2:4]=.5*values.clamp_min(1e-16).log()
    output[:,4:6]=torch.stack((theta.cos(),theta.sin()),1)
    return output
