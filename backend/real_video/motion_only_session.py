"""Reusable, CPU motion-only session for a separately stored Gaussian character.

No RGB history, diffusion model, API, or per-frame text model. Released dog
locomotion weights generate poses; manual cat retargeting is NOT validated
anatomy. This does not retrain the controller or implement learned fur residuals.
Output joint blocks feed the existing shared-surface Gaussian renderer.
"""
import copy
import math
from pathlib import Path
import torch

from .checkpoint_io import digest, load_verified
from .quadruped_controller import QuadrupedController
from .controller_rig import CatRetargeter


ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'artifacts/real_video/neural_motion_controller/v1'
ASSET=ROOT/'artifacts/real_video/hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt'
SHAPE=ROOT/'artifacts/real_video/hunyuan_gaussian/v4_full/shape.pt'
RIG=ROOT/'artifacts/real_video/hunyuan_gaussian/neural_controller_v1/walk/rig.pt'
DEFAULT_SPEED={'idle':0.,'walk':.7,'trot':2.}


def command(action, speed=None):
    if not isinstance(action,str) or action not in DEFAULT_SPEED:
        raise ValueError('Supported actions: idle, walk, trot; unsupported text must not silently map to walking')
    speed=DEFAULT_SPEED[action] if speed is None else speed
    if isinstance(speed,bool) or not isinstance(speed,(int,float)) or not math.isfinite(speed) or not 0<=speed<=4:
        raise ValueError('Finite speed in [0,4] required')
    if action=='idle' and speed!=0:
        raise ValueError('Idle requires zero speed')
    return dict(action=action,speed=float(speed))


def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(tensor_bytes(v) for v in value)
    return 0


class MotionOnlySession:
    """Predict 3 frames per call (30fps); action changes at 10Hz block boundaries.

    Static Gaussian/mesh files are content-bound but not loaded into the motion
    predictor. Renderer loads them once separately. Appearance is never emitted
    or rewritten by this predictor. Snapshot excludes static assets and weights.
    """
    def __init__(self,bundle=BUNDLE,asset=ASSET,shape=SHAPE,rig=RIG):
        paths=dict(controller=Path(bundle)/'controller.ts',asset=Path(asset),shape=Path(shape),rig=Path(rig))
        self.bindings={k:dict(path=str(v.resolve()),sha256=digest(v)) for k,v in paths.items()}
        self.controller=QuadrupedController(bundle=bundle)
        loaded=load_verified(rig)
        # The dense skinning weights belong to the renderer, not recurrent motion.
        self.rig=dict(joints=loaded['joints'].cpu(),parents=loaded['parents'])
        self.targeter=CatRetargeter(self.rig,self.controller.guidances['Walk'],mode='rest_ik')
        self.current=command('idle')
        self.frame_index=0

    def set_action(self,action,speed=None):
        self.current=command(action,speed)

    @torch.no_grad()
    def advance(self):
        world,relative,_=self.controller.step(self.current['action'].title(),self.current['speed'])
        rotations=[];roots=[];errors=[]
        for w,s in zip(world,relative):
            local,shift,_,error,_=self.targeter.step(s,w[0]-s[0])
            rotations.append(local);roots.append(shift);errors.append(error)
        block=dict(local_rotation=torch.stack(rotations),root_translation=torch.stack(roots),
            ik_reach_error=torch.stack(errors),first_frame=self.frame_index,fps=30,
            action=dict(self.current),classification='pretrained motion generation / unvalidated manual cat retarget')
        if not all(torch.isfinite(block[k]).all() for k in ('local_rotation','root_translation','ik_reach_error')):
            raise ValueError('Nonfinite generated controls')
        self.frame_index+=3
        return block

    def snapshot(self):
        return dict(schema='motion_only_session_v1',bindings=copy.deepcopy(self.bindings),
            controller=self.controller.snapshot(),command=dict(self.current),frame_index=self.frame_index,
            retarget_mode='rest_ik',fps=30,appearance_updated=False,
            geometry_quality_accepted=False)

    def restore(self,snapshot):
        if (snapshot.get('schema')!='motion_only_session_v1' or snapshot.get('bindings')!=self.bindings
                or snapshot.get('retarget_mode')!='rest_ik' or snapshot.get('fps')!=30):
            raise ValueError('Checkpoint model/asset/rig/schema mismatch')
        frame=snapshot.get('frame_index')
        if type(frame) is not int or frame<0 or frame%3:
            raise ValueError('Invalid frame boundary')
        selected=command(**snapshot['command'])
        state=snapshot['controller'];expected=self.controller.snapshot()
        if state.keys()!=expected.keys():raise ValueError('Controller state keys differ')
        for key,ref in expected.items():
            value=state[key]
            if (not isinstance(value,torch.Tensor) or value.shape!=ref.shape or value.dtype!=ref.dtype
                    or value.device.type!='cpu' or not torch.isfinite(value).all()):
                raise ValueError('Invalid controller state')
        self.controller.restore(state)
        self.current=selected;self.frame_index=frame

    def render_permission(self,*,diagnostic=False):
        # Existing source is explicitly rejected. Never promote it because the
        # neural controller runs quickly or changes its limb positions.
        if not diagnostic:
            raise ValueError('Canonical cat anatomy/rig remains rejected; only explicitly labelled diagnostic rendering permitted')
        return dict(quality_accepted=False,bindings=copy.deepcopy(self.bindings),
            decoder='real_video.controller_rig.DualSurfaceSkinner',
            covariance_transport='shared surface F Sigma F.T',
            appearance='immutable canonical colour/opacity; fixed barycentric IDs')
