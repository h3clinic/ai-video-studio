"""Explicit persistent per-Gaussian motion records from a connected XYZ surface.

Records exactly describe evaluated model state, not unobserved physical truth.
The orthonormal frame is polar-transported material orientation, not generally
the covariance eigenframe: covariance is authoritative under shear.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .surface_skin3d import triangle_basis


class GaussianSurfaceMemory:
    def __init__(self,asset):
        self.asset=asset
        required=['ids','mesh_vertices','mesh_faces','face_id','barycentric','frame','scale']
        if any(k not in asset or not isinstance(asset[k],torch.Tensor) for k in required): raise ValueError('Missing tensor material fields')
        n=len(asset['ids']); vertices=asset['mesh_vertices']; faces=asset['mesh_faces']; face_id=asset['face_id']
        if asset['ids'].shape!=(n,) or vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or len(faces)==0:
            raise ValueError('Invalid material/mesh dimensions')
        expected={'face_id':(n,),'barycentric':(n,3),'frame':(n,3,3),'scale':(n,3)}
        if any(asset[k].shape!=shape for k,shape in expected.items()): raise ValueError('Material field count mismatch')
        if not all(bool(torch.isfinite(asset[k]).all()) for k in required): raise ValueError('Nonfinite material fields')
        if bool((faces<0).any()) or bool((faces>=len(vertices)).any()) or bool((face_id<0).any()) or bool((face_id>=len(faces)).any()): raise ValueError('Material indices out of bounds')
        if faces.dtype not in [torch.int32,torch.int64] or face_id.dtype not in [torch.int32,torch.int64] or asset['ids'].dtype not in [torch.int32,torch.int64]: raise ValueError('Integral material indices required')
        bary=asset['barycentric']; frame=asset['frame']
        if bool((bary<0).any()) or not torch.allclose(bary.sum(-1),torch.ones_like(bary[:,0]),atol=1e-5): raise ValueError('Invalid barycentric attachment')
        if bool((asset['scale']<=0).any()): raise ValueError('Positive Gaussian scales required')
        if not torch.allclose(frame.transpose(-1,-2)@frame,torch.eye(3,device=frame.device,dtype=frame.dtype).expand_as(frame),atol=1e-5) or bool((torch.det(frame)<.9999).any()): raise ValueError('Proper orthonormal material frame required')
        self.faces=asset['mesh_faces'].long(); self.face_id=asset['face_id'].long()
        basis,area=triangle_basis(asset['mesh_vertices'][self.faces])
        self.active_faces=torch.zeros(len(self.faces),device=self.faces.device,dtype=torch.bool)
        self.active_faces[self.face_id]=True
        invalid=area<1e-12
        if bool((invalid&self.active_faces).any()): raise ValueError('Degenerate owned rest face')
        basis=basis.clone(); basis[invalid]=torch.eye(3,device=basis.device,dtype=basis.dtype)
        self.inverse_basis=torch.linalg.inv(basis); self.rest_area=area
        self.rest_covariance=(asset['frame']*asset['scale'][:,None].square())@asset['frame'].transpose(-1,-2)
        if len(asset['ids'].unique())!=len(asset['ids']): raise ValueError('Persistent Gaussian IDs must be unique')

    def decode(self,vertices):
        if vertices.shape!=self.asset['mesh_vertices'].shape or not bool(torch.isfinite(vertices).all()): raise ValueError('Invalid shared vertices')
        triangles=vertices[self.faces]; basis,area=triangle_basis(triangles)
        if bool(((area<1e-12)&self.active_faces).any()): raise ValueError('Owned deformed face collapsed')
        gradient=basis@self.inverse_basis
        # Polar decomposition on owned faces only avoids unused degeneracies.
        u,_,vh=torch.linalg.svd(gradient[self.active_faces])
        proper=torch.linalg.det(u@vh)
        correction=torch.ones_like(proper)[:,None].expand(-1,3).clone(); correction[:,-1]=proper
        q_active=(u*correction[:,None,:])@vh
        q=torch.eye(3,device=vertices.device,dtype=vertices.dtype).expand(len(self.faces),3,3).clone()
        q[self.active_faces]=q_active
        f=gradient[self.face_id]
        position=(triangles[self.face_id]*self.asset['barycentric'][...,None]).sum(1)
        covariance=f@self.rest_covariance@f.transpose(-1,-2)
        frame=q[self.face_id]@self.asset['frame']
        return dict(position=position,covariance=covariance,frame=frame,area_ratio=area[self.active_faces]/self.rest_area[self.active_faces])


def _validated_fps(fps):
    try:
        if isinstance(fps,(bool,np.bool_)): raise ValueError
        value=float(fps)
    except (TypeError,ValueError,OverflowError):
        raise ValueError('Positive finite scalar fps required') from None
    if not np.isfinite(value) or value<=0: raise ValueError('Positive finite fps required')
    return value


@torch.no_grad()
def record_sequence(asset,vertices,fps):
    """Decode every dot at every state, plus world-space finite differences.

Angular velocity is the principal finite-interval SO(3) logarithm in radians/s.
It cannot identify rotations greater than pi between samples. Linear and angular
velocities belong to interval midpoints (k+1/2)/fps. Accelerations are differences
of adjacent world-frame interval velocities times fps, at internal state times
k/fps, k=1..T-2. They are sampled model-state derivatives, not physical forces or
instantaneous ground-truth derivatives. Angular acceleration uses a fixed world
basis, not differences of incompatible body-frame vectors. Principal-log branch
changes near pi can still cause aliased derivative spikes. Exporting all states
is intentionally redundant diagnostic storage, not a compression claim.
"""
    fps=_validated_fps(fps)
    if vertices.ndim!=3 or len(vertices)<2: raise ValueError('At least two mesh states required')
    if not bool(torch.isfinite(vertices).all()): raise ValueError('Nonfinite trajectory')
    memory=GaussianSurfaceMemory(asset); rows=[memory.decode(v) for v in vertices]
    position=torch.stack([r['position'].detach().cpu() for r in rows])
    frame=torch.stack([r['frame'].detach().cpu() for r in rows])
    covariance=torch.stack([r['covariance'].detach().cpu() for r in rows])
    delta=position[1:]-position[:-1]
    relative=frame[1:]@frame[:-1].transpose(-1,-2)
    angular=Rotation.from_matrix(relative.double().numpy().reshape(-1,3,3)).as_rotvec().reshape(len(vertices)-1,len(asset['ids']),3)*fps
    velocity=delta*fps; angular_velocity=torch.from_numpy(angular).float()
    acceleration=(velocity[1:]-velocity[:-1])*fps
    angular_acceleration=(angular_velocity[1:]-angular_velocity[:-1])*fps
    return dict(ids=asset['ids'].detach().cpu().clone(),position=position,frame=frame,covariance=covariance,
        delta=delta,velocity=velocity,angular_velocity=angular_velocity,fps=fps,
        acceleration=acceleration,angular_acceleration=angular_acceleration,record_version=2,
        state_time_seconds=torch.arange(len(vertices),dtype=torch.float64)/fps,
        velocity_time_seconds=(torch.arange(len(vertices)-1,dtype=torch.float64)+.5)/fps,
        acceleration_time_seconds=torch.arange(1,len(vertices)-1,dtype=torch.float64)/fps,
        acceleration_sample_index=torch.arange(1,len(vertices)-1),
        interval_start=torch.arange(len(vertices)-1),frame_semantics='Polar-transported orthonormal material frame, NOT covariance eigenvectors',
        motion_semantics='Exact evaluated model-state displacement; finite-interval velocity; not measured physical 3D truth',
        acceleration_semantics='World-frame finite differences at internal state times; linear in model-coordinate units/s^2 and angular in rad/s^2; principal-log aliasing remains possible',
        time_origin='Seconds relative to the first recorded state; not absolute source-video timestamps',
        storage_semantics='Expanded diagnostic records, not compact inference state or compression evidence')


def recording_audit(record):
    """Arithmetic audit, including optional v2 acceleration fields in old records."""
    fps=_validated_fps(record['fps'])
    p=record['position']; frame=record['frame']; n=len(p)-1
    if p.ndim!=3 or len(p)<2 or p.shape[1]<1 or p.shape[2]!=3: raise ValueError('Invalid recorded positions')
    count=p.shape[1]
    if record['ids'].shape!=(count,) or len(record['ids'].unique())!=count: raise ValueError('Invalid or duplicate recorded IDs')
    expected_shapes={'frame':(n+1,count,3,3),'delta':(n,count,3),'velocity':(n,count,3),'angular_velocity':(n,count,3)}
    for key,shape in expected_shapes.items():
        value=record[key]
        if value.shape!=shape or not bool(torch.isfinite(value).all()): raise ValueError(f'Invalid recorded {key}')
    if not bool(torch.isfinite(p).all()): raise ValueError('Invalid recorded positions')
    reconstructed=torch.cat((p[:1],p[:1]+record['delta'].cumsum(0)),0)
    step=Rotation.from_rotvec(record['angular_velocity'].double().numpy().reshape(-1,3)/fps).as_matrix()
    step=torch.from_numpy(step).float().reshape(n,len(record['ids']),3,3)
    expected=step@frame[:-1]
    result=dict(gaussians=len(record['ids']),states=len(p),intervals=n,fps=fps,
        position_replay_max_error=float((reconstructed-p).abs().max()),
        velocity_from_delta_max_error=float((record['velocity']-record['delta']*fps).abs().max()),
        rotation_step_max_error=float((expected-frame[1:]).abs().max()),
        orthogonality_max_error=float((frame.transpose(-1,-2)@frame-torch.eye(3)).abs().max()),
        minimum_frame_determinant=float(torch.det(frame).min()),
        expanded_tensor_bytes=sum(v.numel()*v.element_size() for v in record.values() if isinstance(v,torch.Tensor)),
        scope='Arithmetic replay/orthogonality audit only, not physical accuracy or model generalization')
    for acceleration_key,velocity_key,prefix in (
        ('acceleration','velocity','linear'),
        ('angular_acceleration','angular_velocity','angular'),
    ):
        value=record.get(acceleration_key)
        result[f'{prefix}_acceleration_available']=value is not None
        result[f'{prefix}_acceleration_samples']=0 if value is None else len(value)
        result[f'{prefix}_velocity_replay_max_error']=None
        result[f'{prefix}_acceleration_definition_max_error']=None
        if value is None: continue
        if value.shape!=(n-1,count,3) or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Invalid recorded {acceleration_key}')
        velocity=record[velocity_key]
        replay=torch.cat((velocity[:1],velocity[:1]+value.cumsum(0)/fps),0)
        difference=value-(velocity[1:]-velocity[:-1])*fps
        result[f'{prefix}_velocity_replay_max_error']=float((replay-velocity).abs().max())
        result[f'{prefix}_acceleration_definition_max_error']=float(difference.abs().max()) if difference.numel() else 0.
    expected_times={
        'state_time_seconds':torch.arange(n+1,dtype=torch.float64)/fps,
        'velocity_time_seconds':(torch.arange(n,dtype=torch.float64)+.5)/fps,
        'acceleration_time_seconds':torch.arange(1,n,dtype=torch.float64)/fps,
        'acceleration_sample_index':torch.arange(1,n),
    }
    for key,expected in expected_times.items():
        if key not in record: continue  # Old v1 records have no explicit clocks.
        actual=record[key]
        if actual.shape!=expected.shape or not bool(torch.isfinite(actual).all()) or not torch.allclose(actual.double(),expected.double(),rtol=0,atol=1e-12):
            raise ValueError(f'Invalid recorded {key}')
    return result


def main():
    """Expand a saved fitted/forecast mesh packet without reading source images."""
    import argparse
    import json
    from pathlib import Path
    from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion',type=Path)
    parser.add_argument('--asset',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',default=None,help='Key when the vertices field contains several forecast modes')
    parser.add_argument('--audit-only',action='store_true',help='Verify a completed immutable record without regenerating or overwriting it')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.out.exists() and not args.audit_only: raise FileExistsError('Preserve prior vector export')
    if args.out.with_suffix('.audit.json').exists(): raise FileExistsError('Preserve prior vector audit')
    with keep_windows_awake():
        packet=load_verified(args.motion); asset=load_verified(args.asset)
        if packet.get('asset_sha256') and packet['asset_sha256']!=digest(args.asset):
            raise ValueError('Motion and appearance asset hashes differ')
        vertices=packet['vertices']
        if isinstance(vertices,dict):
            if args.mode not in vertices: raise ValueError('Select an available forecast mode')
            vertices=vertices[args.mode]
        if args.audit_only:
            record=load_verified(args.out)
            if record['motion_sha256']!=digest(args.motion) or record['asset_sha256']!=digest(args.asset):
                raise ValueError('Record provenance mismatch')
            checksum=digest(args.out)
        else:
            record=record_sequence(asset,vertices,packet['fps'])
            record.update(motion_sha256=digest(args.motion),asset_sha256=digest(args.asset),
                          classification=packet.get('classification','Source-conditioned fitted mesh trajectory; not new motion generation'))
            if 'source_frame' in packet: record['source_frame']=packet['source_frame'].clone()
            if 'model_sha256' in packet: record['model_sha256']=packet['model_sha256']
            checksum=save_inference_checkpoint(record,args.out)
        audit=recording_audit(record)
        audit.update(sha256=checksum,packet_bytes=args.out.stat().st_size,motion_path=str(args.motion),
                     motion_sha256=record['motion_sha256'],asset_sha256=record['asset_sha256'],
                     compact_mesh_tensor_bytes=vertices.numel()*vertices.element_size(),
                     compact_mesh_note='Trajectory tensor only; excludes asset, metadata, solver and model. Expanded record deliberately redundant.')
        args.out.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        print(json.dumps(audit),flush=True)


if __name__=='__main__': main()
