"""Reusable CUDA graph for one velocity evaluation; inputs change every step.

No sampled video or noise trajectory is cached. Only the GPU operation graph
and its static memory buffers persist. A runner belongs to one frozen model,
shape and guidance configuration; do not update weights after constructing it.
The static buffers are not thread-safe: serialize calls on a runner.
"""
import time
import torch
from .model import guided_velocity


class GraphedVelocity:
    @torch.no_grad()
    def __init__(self,model,batch,shape,guidance=1.5,distilled_guidance=None):
        if next(model.parameters()).device.type!='cuda' or model.training:
            raise ValueError('CUDA eval model required')
        if distilled_guidance is not None and abs(guidance-distilled_guidance)>1e-8:
            raise ValueError('Guidance differs from the fixed distilled guidance')
        self.model=model; self.guidance=guidance; self.distilled_guidance=distilled_guidance
        self.x=torch.zeros(batch,model.config['channels'],*shape,device='cuda')
        self.t=torch.zeros(batch,dtype=torch.long,device='cuda')
        self.labels=torch.zeros(batch,dtype=torch.long,device='cuda')
        torch.cuda.synchronize(); started=time.perf_counter()
        side=torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.output=guided_velocity(model,self.x,self.t,self.labels,guidance,distilled_guidance)
        torch.cuda.current_stream().wait_stream(side)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output=guided_velocity(model,self.x,self.t,self.labels,guidance,distilled_guidance)
        torch.cuda.synchronize(); self.setup_seconds=time.perf_counter()-started

    @torch.no_grad()
    def __call__(self,x,t,labels):
        if x.shape!=self.x.shape or t.shape!=self.t.shape or labels.shape!=self.labels.shape:
            raise ValueError('CUDA graph shape mismatch; create a new runner')
        if x.device!=self.x.device or x.dtype!=self.x.dtype or t.dtype!=torch.long or labels.dtype!=torch.long:
            raise ValueError('CUDA graph device/dtype mismatch')
        self.x.copy_(x); self.t.copy_(t); self.labels.copy_(labels)
        self.graph.replay()
        # Caller may retain this prediction across another evaluation.
        return self.output.clone()
