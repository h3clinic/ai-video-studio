"""Freeze released pretrained inference weights and verify a standalone runtime."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .quadruped_controller import QuadrupedController, DEMO
from .checkpoint_io import digest,save_inference_checkpoint

OUT=Path('artifacts/real_video/neural_motion_controller/v1')


class InferenceOnly(torch.nn.Module):
    def __init__(self,model):
        super().__init__(); self.model=model
    def forward(self,inputs): return self.model(inputs,iterations=1,sample=False)


@torch.no_grad()
def run():
    torch.set_num_threads(2); OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'controller.ts'
    if path.exists(): raise FileExistsError('Preserve exported model')
    controller=QuadrupedController()
    model=InferenceOnly(controller.model).eval()
    # JIT inspects every property, including unused training-only accumulators.
    # Reset those accumulators in this in-memory export copy; trained Norm
    # tensors and Active flags, which control inference, remain untouched.
    for module in model.modules():
        if hasattr(module,'Norm') and hasattr(module,'Dim'):
            module.n=2.; module.m=np.zeros(module.Dim); module.s=np.ones(module.Dim)
    inputs=controller.model.Sampler.FeatureStatistics.GetMean()[None].clone()
    frozen=torch.jit.freeze(torch.jit.trace(model,inputs))
    torch.jit.save(frozen,str(path)); loaded=torch.jit.load(str(path)).eval()
    errors=[]
    for scale in [0.,.01,.1]:
        torch.manual_seed(531030)
        x=inputs+torch.randn_like(inputs)*scale
        errors.append(float((loaded(x)-model(x)).abs().max()))
    assert max(errors)<1e-5
    state=controller.snapshot(); first=controller.step('Walk',.7)[1]
    controller.restore(state); second=controller.step('Walk',.7)[1]
    assert torch.equal(first,second)
    controller.restore(state); idle=controller.step('Idle',0.)[1]
    response=float((first-idle).abs().mean()); assert response>1e-4
    before=sum(x.numel()*x.element_size() for x in state.values())
    for _ in range(300): controller.step('Walk',.7)
    after=sum(x.numel()*x.element_size() for x in controller.snapshot().values())
    assert before==after
    save_inference_checkpoint(dict(state=state,guidances=controller.guidances,joint_names=controller.names),OUT/'initial_state.pt')
    report=dict(source='https://github.com/facebookresearch/ai4animationpy',license='CC-BY-NC-4.0',
        source_weights_sha256=digest(DEMO/'Network.pt'),export_sha256=digest(path),
        frozen_file_bytes=path.stat().st_size,export_parity_max_error=max(errors),
        recurrent_tensor_bytes=before,recurrent_tensor_bytes_after_30_seconds=after,
        resume_exact=True,command_response_mean_abs_difference=response,
        cpu_median_call_ms=float(np.median(controller.latencies))*1000,
        new_training=False,wan_calls=0,frame_history_input=False,
        limitations='State bytes exclude weights, static guidance, Python objects, temporary lookahead, Gaussian asset and renderer. Control response is not a quality metric.')
    (OUT/'export_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__': run()
