"""Warmed single-clip cost measurements. NOT quality-matched model superiority."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from torch import nn
from .checkpoint_io import load_verified,digest,keep_windows_awake
from .latent import inference_decoder
from .model import GaussianVideoDenoiser,sample
from .representation import render_fields_chunked


class ModuleMACs:
    """Conv/linear and QK/AV MACs only; excludes norms, activation and splatting."""
    def __init__(self,modules):
        self.value=0; self.hooks=[]
        def count(module,inputs,output):
            if isinstance(module,(nn.Conv1d,nn.Conv3d)):
                self.value+=output.numel()*(module.in_channels//module.groups)*math_product(module.kernel_size)
            elif isinstance(module,nn.Linear): self.value+=output.numel()*module.in_features
        def attention(module,inputs,output):
            b,three_c,length=output.shape
            self.value+=2*b*(three_c//3)*length*length
        for root in modules:
            for name,module in root.named_modules():
                if isinstance(module,(nn.Conv1d,nn.Conv3d,nn.Linear)): self.hooks.append(module.register_forward_hook(count))
                if name.endswith('qkv'): self.hooks.append(module.register_forward_hook(attention))

    def close(self):
        for hook in self.hooks: hook.remove()


def math_product(values):
    result=1
    for value in values: result*=value
    return result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['direct_gaussian','latent_gaussian','latent_replay','pixel_cost_probe'])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--codes',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--frame-chunk',type=int,default=8)
    parser.add_argument('--cuda-graph',action='store_true')
    parser.add_argument('--precast-weights',action='store_true',help='Pre-store autocast Conv/Linear weights as BF16; leave embedding/norm FP32')
    args=parser.parse_args()
    if args.mode!='pixel_cost_probe' and not args.checkpoint: parser.error('Checkpoint required')
    if args.mode=='latent_replay' and not args.codes: parser.error('Generated codes required')
    if args.cuda_graph and args.mode=='latent_replay': parser.error('Replay has no denoiser to graph')
    if args.repeats<3: parser.error('Use at least three measurements')
    if not 1<=args.batch<=16: parser.error('Batch must be in 1..16')
    if not 1<=args.frame_chunk<=8: parser.error('Frame chunk must be in 1..8')
    torch.set_num_threads(2); torch.manual_seed(991)
    device='cuda'; modules=[]; shape=(8,32,32); decoder=None; code=None
    checkpoint=load_verified(args.checkpoint) if args.checkpoint else None
    if args.mode in ('latent_gaussian','latent_replay'):
        decoder=inference_decoder(checkpoint['codec_config'],checkpoint['codec'])
        if args.precast_weights:
            from .inference_precision import precast_autocast_weights
            precast_autocast_weights(decoder)
        decoder=decoder.cuda(); modules.append(decoder)
        mean=checkpoint['mean'].cuda(); std=checkpoint['std'].cuda()
        latent_mean=checkpoint['latent_mean'].cuda(); latent_std=checkpoint['latent_std'].cuda()
        shape=tuple(checkpoint['latent_shape'])
    if args.mode=='latent_replay':
        saved=torch.load(args.codes,map_location='cpu',weights_only=True)
        if saved['checkpoint_sha256']!=digest(args.checkpoint): raise ValueError('Code/checkpoint mismatch')
        code=saved['latent'][:args.batch].float().cuda(); prior=None; model_config=None
        if len(code)!=args.batch: parser.error('Insufficient saved codes for requested batch')
    else:
        model_config=checkpoint['config'] if checkpoint else dict(classes=10,width=32,channels=3)
        prior=GaussianVideoDenoiser(**model_config).eval()
        if checkpoint: prior.load_state_dict(checkpoint['model'])
        if args.precast_weights:
            from .inference_precision import precast_autocast_weights
            precast_autocast_weights(prior)
        prior=prior.cuda(); modules.append(prior)
        if args.mode=='pixel_cost_probe': shape=(8,64,64)
        if args.mode=='direct_gaussian': mean=checkpoint['mean'].cuda(); std=checkpoint['std'].cuda()
    labels=torch.arange(args.batch,device=device)%(prior.classes if prior else len(checkpoint['labels']))
    runner=None
    if args.cuda_graph:
        from .graph_sampling import GraphedVelocity
        runner=GraphedVelocity(prior,args.batch,shape,distilled_guidance=checkpoint.get('distilled_guidance') if checkpoint else None)
    def generate(frame_chunk=args.frame_chunk,use_graph=True,seed=460001):
        values=code if prior is None else sample(prior,labels,seed=seed,steps=args.steps,shape=shape,distilled_guidance=checkpoint.get('distilled_guidance') if checkpoint else None,velocity_fn=runner if use_graph else None)
        if args.mode=='pixel_cost_probe': pixels=((values+1)/2).clamp(0,1).permute(0,2,1,3,4)
        else:
            if decoder is not None:
                if prior is not None: values=values*latent_std+latent_mean
                values=values.half().float()
                with torch.autocast('cuda',dtype=torch.bfloat16): values=decoder.decode(values).float()
            pixels=render_fields_chunked(values*std+mean,frame_chunk=frame_chunk)
        return pixels.cpu()
    reference=generate(frame_chunk=8,use_graph=False)
    candidate=generate()
    max_difference=(reference-candidate).abs().max().item()
    if max_difference>2e-5: raise AssertionError('Chunking changed pixels beyond numerical tolerance')
    graph_differences=[]
    if runner:
        for check_seed,label_id in ((460002,1),(460003,9)):
            labels.fill_(label_id)
            delta=(generate(seed=check_seed)-generate(seed=check_seed,use_graph=False)).abs().max().item()
            graph_differences.append(dict(seed=check_seed,label=label_id,max_pixel_difference=delta))
            if delta>2e-5: raise AssertionError('Graph changed fresh-input pixels beyond tolerance')
        labels.copy_(torch.arange(args.batch,device=device)%prior.classes)
    del reference,candidate
    torch.cuda.empty_cache()
    for _ in range(2): generate()
    torch.cuda.synchronize()
    resident=torch.cuda.memory_allocated(); times=[]; peaks=[]; reserved=[]
    for _ in range(args.repeats):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
        output=generate(); torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        peaks.append(torch.cuda.max_memory_allocated()); reserved.append(torch.cuda.max_memory_reserved())
    counter=ModuleMACs(modules)
    # Replaying a CUDA graph skips Python hooks, so count the identical eager operations.
    try: generate(use_graph=False)
    finally: counter.close()
    report=dict(mode=args.mode,checkpoint_sha256=digest(args.checkpoint) if checkpoint else None,
                batch=args.batch,output_shape=list(output.shape),steps=args.steps if prior else 0,guidance=1.5,
                render_frame_chunk=args.frame_chunk,
                distilled_guidance=checkpoint.get('distilled_guidance') if checkpoint else None,
                cuda_graph=bool(runner),graph_setup_seconds=runner.setup_seconds if runner else 0.,
                graph_fresh_input_checks=graph_differences,
                same_seed_whole_clip_max_abs_difference=max_difference,
                precision=('BF16 Conv/Linear weights, FP32 embedding/norm weights' if args.precast_weights else 'FP32 weights')+', BF16 network autocast; FP32 splatting',
                precast_weights=args.precast_weights,
                model_config=model_config,decoder_parameters=sum(p.numel() for p in decoder.parameters()) if decoder else 0,
                live_model_parameters=sum(p.numel() for m in modules for p in m.parameters()),
                analysis_encoder_on_gpu=False,device=torch.cuda.get_device_name(0),torch_version=torch.__version__,
                resident_cuda_allocated_bytes=resident,peak_cuda_allocated_bytes=max(peaks),
                incremental_peak_cuda_allocated_bytes=max(peaks)-resident,peak_cuda_reserved_bytes=max(reserved),
                warmed_seconds=times,median_seconds=statistics.median(times),
                median_seconds_per_clip=statistics.median(times)/args.batch,clips_per_second=args.batch/statistics.median(times),
                module_macs=counter.value,module_flops_2_per_mac=2*counter.value,
                limits=['Compute includes conv/linear and attention QK/AV only, not all FLOPs.',
                        'Timing includes sampling, Gaussian decoding/splatting and CPU output transfer; excludes imports, checkpoint load and video encoding.',
                        'Memory includes live model tensors and intermediate/output GPU tensors; not CUDA context, CPU RAM, other applications or full process VRAM.',
                        'Pixel cost probe has random weights and is NOT a trained or quality-matched baseline.',
                        'Direct and compact Gaussian models differ in architecture, training and quality. No quality-matched saving claim.',
                        'Replay starts from an already generated code; generation and replay are different tasks.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    with keep_windows_awake(): main()
