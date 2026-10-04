"""Shared XYZ mesh deformation with persistent Gaussian face attachments.

Connected support prevents independent-splat drift. This does not ensure
correct anatomy, nonintersection or a noncollapsed/fold-free articulated mesh.
"""
import torch
from .gaussian3d import forward_kinematics


def triangle_basis(triangles):
    first=triangles[:,1]-triangles[:,0]; second=triangles[:,2]-triangles[:,0]
    cross=torch.linalg.cross(first,second); area2=cross.norm(dim=-1)
    normal=cross/area2.clamp_min(1e-12)[:,None]
    return torch.stack((first,second,normal),-1),area2


class SurfaceSkinner:
    def __init__(self,asset,rig):
        self.asset=asset; self.rig=rig
        self.faces=asset['mesh_faces'].long(); self.face_id=asset['face_id'].long()
        basis,area=triangle_basis(asset['mesh_vertices'][self.faces])
        self.active_faces=torch.zeros(len(self.faces),device=self.faces.device,dtype=torch.bool)
        self.active_faces[self.face_id]=True
        invalid=area<1e-12
        if bool((invalid&self.active_faces).any()): raise ValueError('Degenerate Gaussian-owned rest face')
        # TripoSR marching cubes can retain zero-area triangles with no sampled
        # Gaussians. Preserve the mesh; only their unused basis is replaced.
        self.unused_degenerate_faces=int(invalid.sum())
        basis=basis.clone(); basis[invalid]=torch.eye(3,device=basis.device,dtype=basis.dtype)
        self.inverse_basis=torch.linalg.inv(basis)
        self.rest_area=area
        self.covariance=(asset['frame']*asset['scale'][:,None].square())@asset['frame'].transpose(-1,-2)
        w=rig['vertex_weights']
        if w.shape!=(len(asset['mesh_vertices']),len(rig['joints'])): raise ValueError('Vertex weight shape mismatch')
        if bool((w<0).any()) or not torch.allclose(w.sum(-1),torch.ones_like(w[:,0]),atol=1e-5): raise ValueError('Skin weights must form a partition of unity')

    def __call__(self,local_rotation,root_translation=None):
        a=self.asset; r=self.rig; vertices=a['mesh_vertices']
        transforms=forward_kinematics(r['joints'],r['parents'],local_rotation)
        rotation=transforms[:,:3,:3]; offset=transforms[:,:3,3]-(rotation@r['joints'][...,None]).squeeze(-1)
        # Each mesh vertex is evaluated ONCE, shared by every adjacent face.
        matrices=torch.einsum('vj,jab->vab',r['vertex_weights'],rotation)
        shifts=r['vertex_weights']@offset
        moved=(matrices@vertices[...,None]).squeeze(-1)+shifts
        if root_translation is not None:
            moved=moved+root_translation
            transforms=transforms.clone(); transforms[:,:3,3]+=root_translation
        triangles=moved[self.faces]; basis,area=triangle_basis(triangles)
        if bool(((area<1e-12)&self.active_faces).any()): raise ValueError('Gaussian-owned deformed face collapsed')
        gradient=basis@self.inverse_basis
        position=(triangles[self.face_id]*a['barycentric'][...,None]).sum(1)
        f=gradient[self.face_id]
        covariance=f@self.covariance@f.transpose(-1,-2)
        # Area diagnostics concern the faces actually carrying Gaussians.
        return position,covariance,transforms,moved,area[self.active_faces]/self.rest_area[self.active_faces]
