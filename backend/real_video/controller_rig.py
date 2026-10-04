"""Explicit retargeting/contact constraints and shared-surface Gaussian skinning.

Manual rig and kinematic constraints, not learned anatomy or physics simulation.
Dual-quaternion blending follows the standard rigid-motion construction.
"""
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from .gaussian3d import forward_kinematics
from .surface_skin3d import SurfaceSkinner, triangle_basis
from .original_cat_motion import approximate_rig


def qmul(a,b):
    av,aw=a[...,:3],a[...,3:]; bv,bw=b[...,:3],b[...,3:]
    return torch.cat((aw*bv+bw*av+torch.linalg.cross(av,bv),aw*bw-(av*bv).sum(-1,keepdim=True)),-1)


def dual_quaternion_skin(vertices,weights,rotation,offset):
    # Only J rotations cross the CPU boundary, never the dense mesh.
    qr=torch.as_tensor(Rotation.from_matrix(rotation.detach().cpu().numpy()).as_quat(),device=vertices.device,dtype=vertices.dtype)
    qd=.5*qmul(torch.cat((offset,torch.zeros_like(offset[:,:1])),-1),qr)
    reference=qr[weights.argmax(-1)]
    signs=torch.where(reference@qr.T<0,-1.,1.)
    signed=weights*signs
    real=signed@qr; dual=signed@qd; length=real.norm(dim=-1,keepdim=True)
    if bool((length<1e-7).any()): raise ValueError('Degenerate dual-quaternion blend')
    real=real/length; dual=dual/length
    dual=dual-real*(real*dual).sum(-1,keepdim=True)
    conjugate=torch.cat((-real[:,:3],real[:,3:]),-1)
    displacement=2*qmul(dual,conjugate)[:,:3]
    vq=torch.cat((vertices,torch.zeros_like(vertices[:,:1])),-1)
    return qmul(qmul(real,vq),conjugate)[:,:3]+displacement


class DualSurfaceSkinner(SurfaceSkinner):
    def __call__(self,local_rotation,root_translation=None):
        a=self.asset; r=self.rig
        transforms=forward_kinematics(r['joints'],r['parents'],local_rotation)
        rotation=transforms[:,:3,:3]
        offset=transforms[:,:3,3]-(rotation@r['joints'][...,None]).squeeze(-1)
        moved=dual_quaternion_skin(a['mesh_vertices'],r['vertex_weights'],rotation,offset)
        if root_translation is not None: moved=moved+root_translation
        triangles=moved[self.faces]; basis,area=triangle_basis(triangles)
        if bool(((area<1e-12)&self.active_faces).any()): raise ValueError('Collapsed active face')
        gradient=basis@self.inverse_basis
        position=(triangles[self.face_id]*a['barycentric'][...,None]).sum(1)
        f=gradient[self.face_id]
        covariance=f@self.covariance@f.transpose(-1,-2)
        return position,covariance,transforms,moved,area[self.active_faces]/self.rest_area[self.active_faces]


def guarded_rig(vertices):
    rig=approximate_rig(vertices)
    # Keep broad torso out of limb-control influence. Smooth transition avoids
    # independent part translations; this is a heuristic, not automatic rigging.
    w=rig['vertex_weights']; leg=w[:,5:].sum(-1)
    gate=torch.sigmoid((-vertices[:,1]-.18)/.04)*torch.sigmoid(((vertices[:,0]+.06).abs()-.36)/.035)
    removed=leg*(1-gate); w[:,5:]*=gate[:,None]; w[:,0]+=removed
    return rig


def from_to(a,b):
    a=F.normalize(a,dim=-1); b=F.normalize(b,dim=-1)
    dot=(a*b).sum().clamp(-1,1)
    if float(dot)<-.9999:
        axis=torch.zeros_like(a); axis[int(a.abs().argmin())]=1
        axis=F.normalize(torch.linalg.cross(a,axis),dim=0)
        return 2*axis[:,None]*axis[None]-torch.eye(3,device=a.device)
    cross=torch.linalg.cross(a,b); x,y,z=cross
    skew=torch.stack((x*0,-z,y,z,x*0,-x,-y,x,x*0)).reshape(3,3)
    return torch.eye(3,device=a.device)+skew+skew@skew/(1+dot).clamp_min(1e-8)


def two_bone(hip,target,pole,length1,length2):
    delta=target-hip; requested=delta.norm()
    direction=F.normalize(delta,dim=0)
    if float(requested)<1e-7: direction=hip.new_tensor([0.,-1.,0.])
    reach=requested.clamp(abs(length1-length2)+1e-5,length1+length2-1e-5)
    along=(length1**2-length2**2+reach**2)/(2*reach)
    height=(length1**2-along**2).clamp_min(0).sqrt()
    bend=pole-hip-direction*((pole-hip)*direction).sum()
    if float(bend.norm())<1e-6:
        auxiliary=torch.zeros_like(direction); auxiliary[int(direction.abs().argmin())]=1
        bend=auxiliary-direction*(auxiliary*direction).sum()
    knee=hip+along*direction+height*F.normalize(bend,dim=0)
    foot=hip+reach*direction
    return knee,foot,(foot-target).norm()


class CatRetargeter:
    """Fixed-length IK with approximate contact latching; straight-line only."""
    chains=[(5,6,7,16,17,19),(8,9,10,20,21,23),(11,12,13,7,8,10),(14,15,16,12,13,15)]
    def __init__(self,rig,guidance,scale=1.7,mode='ik'):
        if mode not in ('ik', 'relative', 'rest_ik'):
            raise ValueError('Unknown cat retarget mode')
        self.rig=rig; self.scale=scale
        self.mode=mode
        self.reference=self.convert(guidance)
        self.anchors=[None]*4
        self.floor=-.67

    @staticmethod
    def convert(points):
        # Proper rotation: source +Z forward becomes cat +X forward.
        return torch.stack((points[...,2],points[...,1],-points[...,0]),-1)

    def step(self,source,root):
        source=self.convert(source); shift=self.convert(root)*self.scale
        j=self.rig['joints']; local=torch.eye(3,device=j.device).repeat(17,1,1)
        errors=[]; feet=[]; contacts=[]
        for index,(hip,knee,foot,sh,sk,sf) in enumerate(self.chains):
            if self.mode=='relative':
                r1=from_to(self.reference[sk]-self.reference[sh],source[sk]-source[sh])
                r2=from_to(self.reference[sf]-self.reference[sk],source[sf]-source[sk])
                local[hip]=r1; local[knee]=r1.T@r2
                k=j[hip]+r1@(j[knee]-j[hip]); f=k+r2@(j[foot]-j[knee])
                feet.append(f+shift); errors.append(j.new_tensor(0.)); contacts.append(False)
                continue
            l1=(j[knee]-j[hip]).norm(); l2=(j[foot]-j[knee]).norm()
            source_length=(self.reference[sk]-self.reference[sh]).norm()+(self.reference[sf]-self.reference[sk]).norm()
            if self.mode == 'rest_ik':
                # Retain the asset's actual bind pose. The old absolute target
                # forced already-posed cat legs into the dog guidance stance.
                # This branch has NO contact correction: reference identity
                # must not be silently broken by a guessed ground height.
                delta=(source[sf]-source[sh])-(self.reference[sf]-self.reference[sh])
                target=j[foot]+delta*(l1+l2)/source_length+shift
                pole=j[knee]+shift+(source[sk]-source[sh]-self.reference[sk]+self.reference[sh])*(l1+l2)/source_length
                k,f,error=two_bone(j[hip]+shift,target,pole,l1,l2)
                r1=from_to(j[knee]-j[hip],k-(j[hip]+shift))
                r2=from_to(j[foot]-j[knee],f-k)
                local[hip]=r1; local[knee]=r1.T@r2
                errors.append(error); feet.append(f); contacts.append(False)
                continue
            target=j[hip]+(source[sf]-source[sh])*(l1+l2)/source_length+shift
            target[1]=target[1].clamp_min(self.floor+.025)
            contact=bool(source[sf,1]<.045)
            if contact:
                if self.anchors[index] is None: self.anchors[index]=target.clone()
                # A height heuristic is uncertain: soft anchoring, not an
                # indefinite hard latch that pulls a planted foot out of reach.
                target=torch.lerp(target,self.anchors[index],.8)
                self.anchors[index]=target.clone()
            else: self.anchors[index]=None
            pole=j[knee]+shift+(source[sk]-self.reference[sk])*self.scale*.25
            k,f,error=two_bone(j[hip]+shift,target,pole,l1,l2)
            r1=from_to(j[knee]-j[hip],k-(j[hip]+shift))
            r2=from_to(j[foot]-j[knee],f-k)
            local[hip]=r1; local[knee]=r1.T@r2
            errors.append(error); feet.append(f); contacts.append(contact)
        return local,shift,torch.stack(feet),torch.stack(errors),contacts
