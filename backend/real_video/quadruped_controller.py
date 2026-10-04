"""Headless adapter for the released AI4AnimationPy quadruped weights.

Upstream: facebookresearch/ai4animationpy, CC-BY-NC-4.0. This adapter does
not retrain or claim ownership of the pretrained network. No image generator.
"""
import importlib
import sys
import types
import time
import json
from collections import deque
from pathlib import Path
import numpy as np
import torch

VENDOR=Path(__file__).resolve().parents[1]/'work/ai4animationpy'
DEMO=VENDOR/'Demos/Locomotion/Quadruped'


def load_model():
    # Load only neural modules; the upstream package initializer also imports
    # desktop GUI/audio dependencies unnecessary for headless inference.
    for name in ['ai4animation','ai4animation.AI','ai4animation.AI.Library','ai4animation.AI.Models']:
        if name not in sys.modules:
            package=types.ModuleType(name)
            package.__path__=[str(VENDOR/Path(*name.split('.')))]
            sys.modules[name]=package
    path=DEMO/'Network.pt'
    explicit={
        'ai4animation.AI.Models.CxM.MotionPrior','ai4animation.AI.Models.CxM.MotionSampler',
        'ai4animation.AI.Models.CxM.MotionModel','ai4animation.AI.Models.CategoricalEncoderDecoder.Model',
        'ai4animation.AI.Library.Blocks.SpaceTimeBlock','ai4animation.AI.Library.Blocks.LinearBlock',
        'ai4animation.AI.Library.Blocks.FiLMLinearBlock','ai4animation.AI.Library.Layers.CodebookLayer',
        'ai4animation.AI.Library.Layers.LinearLayer','ai4animation.AI.Library.Layers.FiLMLinearLayer',
        'ai4animation.AI.Library.Layers.FiLMLayer','ai4animation.AI.Library.Statistics.RunningStatistics',
        'numpy.ndarray','numpy.dtype','numpy.core.multiarray._reconstruct',
        'torch.nn.functional.elu','torch.nn.modules.linear.Linear'}
    found=set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if not found<=explicit: raise ValueError(f'Unreviewed checkpoint globals: {found-explicit}')
    allowed=[(getattr(importlib.import_module(name.rsplit('.',1)[0]),name.rsplit('.',1)[1]),name) for name in found]
    allowed.extend([type(np.dtype('float32')),type(np.dtype('float64'))])
    with torch.serialization.safe_globals(allowed):
        model=torch.load(path,map_location='cpu',weights_only=True).eval()
    return model


class QuadrupedController:
    """Bounded autoregressive state; 16-pose lookahead, no history of RGB frames.

This first adapter uses one network call per three output frames, straight-line
root control, and the released static gait guidance. It is NOT a full port of
the upstream PID/contact-IK runtime. Generated poses are recentered each call.
"""
    def __init__(self, device='cpu', bundle=None):
        self.device=device
        self.frozen=bundle is not None
        if self.frozen:
            from .checkpoint_io import load_verified, digest
            bundle=Path(bundle)
            metadata=json.loads((bundle/'export_report.json').read_text())
            if digest(bundle/'controller.ts')!=metadata['export_sha256']: raise ValueError('Controller digest mismatch')
            self.model=torch.jit.load(str(bundle/'controller.ts'),map_location=device).eval()
            data=load_verified(bundle/'initial_state.pt')
            self.guidances={k:v.to(device) for k,v in data['guidances'].items()}
            self.names=data['joint_names']; self.restore(data['state'])
            self.calls=0; self.latencies=deque(maxlen=256)
            return
        self.model=load_model().to(device)
        self.guidances={}
        for path in (DEMO/'Guidances').glob('*.npz'):
            with np.load(path,allow_pickle=False) as data:
                self.names=data['Names'].tolist()
                self.guidances[path.stem]=torch.tensor(data['Positions'],dtype=torch.float32,device=device)
        self.position=self.guidances['Walk'].clone()
        self.velocity=torch.zeros_like(self.position)
        self.root=torch.zeros(3,device=device)
        self.calls=0
        self.latencies=deque(maxlen=256)

    def snapshot(self):
        return {k:getattr(self,k).clone() for k in ('position','velocity','root')}

    def restore(self,state):
        for key in ('position','velocity','root'): setattr(self,key,state[key].to(self.device).clone())

    @torch.no_grad()
    def step(self,gait='Walk',speed=.7):
        if gait not in self.guidances: raise ValueError('Unknown guidance')
        if not 0<=speed<=4: raise ValueError('Speed outside released locomotion range')
        t=torch.linspace(0,.5,16,device=self.device)
        trajectory=torch.stack((torch.zeros_like(t),t*speed),-1)
        facing=torch.tensor([0.,1.],device=self.device).expand(16,2)
        velocity=torch.tensor([0.,speed],device=self.device).expand(16,2)
        inputs=torch.cat([self.position.flatten(),self.velocity.flatten(),trajectory.flatten(),
            facing.flatten(),velocity.flatten(),self.guidances[gait].flatten()])[None]
        assert inputs.shape==(1,339)
        if self.device=='cuda': torch.cuda.synchronize()
        start=time.perf_counter()
        output=(self.model(inputs) if self.frozen else self.model(inputs,iterations=1,sample=False))[0]
        if self.device=='cuda': torch.cuda.synchronize()
        self.latencies.append(time.perf_counter()-start); self.calls+=1
        if not torch.isfinite(output).all(): raise ValueError('Nonfinite motion prediction')
        # Decoder feature order audited against upstream Program.Predict.
        position=output[:,3:84].reshape(16,27,3)
        velocity=output[:,246:327].reshape(16,27,3)
        root_delta=output[:,:3].clone(); root_delta[:,1]=0
        root_delta[0]=0; root_delta=root_delta.cumsum(0)
        world=position[1:4]+self.root+root_delta[1:4,None]
        self.position=position[3].clone(); self.velocity=velocity[3].clone()
        self.root=self.root+root_delta[3]
        return world,position[1:4],output[1:4,84:246].reshape(3,2,27,3)


if __name__=='__main__':
    model=load_model()
    print(model)
    print('input',model.input_dim(),'parameters',sum(p.numel() for p in model.parameters()))
