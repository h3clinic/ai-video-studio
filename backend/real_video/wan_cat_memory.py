"""Experimental Gaussian-memory conditioning and real Wan attention LoRA.

Single-subject appearance adaptation, not direct Gaussian motion generation.
The immutable full Gaussian asset remains authoritative; voxel tokens are lossy
conditioning, not a replacement for its fine geometry. No external test data.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from PIL import Image

from real_video.checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.enabled = True

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        return original + ((x.float() @ self.down.T) @ self.up.T).to(original.dtype)


def install_lora(model, rank):
    names = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and ('.attn1.' in name or '.attn2.' in name):
            if name.endswith(('to_q', 'to_k', 'to_v', 'to_out.0')):
                parent, key = name.rsplit('.', 1)
                setattr(model.get_submodule(parent), key, LowRankLinear(module, rank))
                names.append(name)
    if not names:
        raise ValueError('No Wan attention projections found')
    return names


@torch.no_grad()
def merge_lora_for_inference(model):
    """Merge trained updates into in-memory base only; never write upstream files.

    One-way on this model instance. Reload original weights for a true baseline.
    BF16 rounding can change outputs slightly; benchmark before claiming parity.
    """
    modules = [m for m in model.modules() if isinstance(m, LowRankLinear)]
    for module in modules:
        if getattr(module, 'merged', False):
            raise ValueError('Adapter already merged')
        if not module.enabled:
            raise ValueError('Enable adapter before merging')
    for module in modules:
        module.base.weight.copy_((module.base.weight.float() + module.up @ module.down).to(module.base.weight.dtype))
        module.enabled = False
        module.merged = True
    return len(modules)


def gaussian_tokens(asset):
    """Fixed 4^3 spatial cells; mean geometry/appearance and occupancy.

    No camera frames, animation, or fabricated backside supervision is used.
    Canonical positions normalized using this immutable asset's own bounds.
    """
    p = asset['position'].float()
    lo, hi = p.amin(0), p.amax(0)
    q = ((p-lo)/(hi-lo).clamp_min(1e-6)).clamp(0, 1)
    cell = (q * 4).long().clamp_max(3)
    idx = cell[:, 0]*16 + cell[:, 1]*4 + cell[:, 2]
    values = torch.cat([q*2-1, asset['covariance'].float().flatten(1),
                        asset['normal'].float(), asset['colour'].float(),
                        asset['opacity'].float().reshape(-1, 1)], 1)
    result = torch.zeros(64, values.shape[1])
    count = torch.bincount(idx, minlength=64).float()
    result.index_add_(0, idx, values)
    result /= count.clamp_min(1)[:, None]
    return torch.cat([result, (count / len(p))[:, None]], 1)


class GaussianMemory(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.register_buffer('features', features)
        self.net = nn.Sequential(nn.Linear(features.shape[-1], 128), nn.SiLU(), nn.Linear(128, 4096))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, text):
        # Residual into the fixed padded text context: exact base parity at init.
        residual = self.net(self.features).to(text.dtype)[None]
        return torch.cat([text[:, :-64], text[:, -64:] + residual], 1)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=48)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--sample-steps', type=int, default=24)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=448)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new experiment directory')
    args.out.mkdir(parents=True)
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline, UniPCMultistepScheduler
    root = Path('artifacts/real_video')
    model_dir = Path('../../work/wan21_13b')
    source = root/'prompt_generation/orange_cat_seed_531002/paired_decoder_benchmark_v1/0_wan'
    asset_path = root/'hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt'
    embeddings_path = root/'wan_baseline/cat_seed_421001/prompt_embeddings.pt'
    torch.set_num_threads(6)
    torch.manual_seed(71031)
    features = gaussian_tokens(load_verified(asset_path))
    embeddings = load_verified(embeddings_path)
    positive = embeddings['positive'].to('cuda', dtype=torch.bfloat16)
    negative = embeddings['negative'].to('cuda', dtype=torch.bfloat16)
    report = dict(scope='Single-cat Wan LoRA + Gaussian voxel-memory residual conditioning; RGB output, not direct Gaussian dynamics',
                  quality_accepted=False, training_frames=[0,8,16,24], validation_frames=[32],
                  validation_scope='Same-source held-out frame, not independent identity/generalization validation',
                  source=str(source), gaussian_asset_sha256=digest(asset_path),
                  prompt_embeddings_sha256=digest(embeddings_path), rank=args.rank,
                  source_hashes={str(i):digest(source/f'frame_{i:03d}.png') for i in [0,8,16,24,32]},
                  height=args.height, width=args.width, training_steps=args.steps,
                  research={'paper':'2502.14844v1','cached_sha256':'d7d791f171e496638fb7b6353528709045bc252c2a59ee2463081381e02c8fe4',
                            'math':'W_eff=W+BA; x_s=(1-s)x+s*noise; target=noise-x; squared flow error',
                            'departure':'Custom Gaussian-memory residual. Appearance-only pilot, not replication of two-stage method.'},
                  training=[], samples={}, limitations=['No demonstrated generalization to other cats','No full-resolution fur guarantee','No compute savings established'])
    def status(message):
        print(message, flush=True)
        (args.out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with keep_windows_awake():
        status('Encoding sharp source frames, not blurred Gaussian renders')
        vae = AutoencoderKLWan.from_pretrained(model_dir/'vae', torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).cuda()
        mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1,-1,1,1,1)
        std = torch.tensor(vae.config.latents_std, device='cuda').view(1,-1,1,1,1)
        latents = []
        with torch.no_grad():
            for index in [0,8,16,24,32]:
                image = Image.open(source/f'frame_{index:03d}.png').convert('RGB').resize((args.width,args.height),Image.Resampling.LANCZOS)
                x = torch.from_numpy(np.array(image)).permute(2,0,1)[None,:,None].float().cuda()/127.5-1
                latents.append(((vae.encode(x).latent_dist.mode()-mean)/std).detach().to(torch.bfloat16))
        del vae, mean, std, x
        free()
        status('Loading frozen Wan base and attaching trainable attention updates')
        model = WanTransformer3DModel.from_pretrained(model_dir/'transformer', torch_dtype=torch.bfloat16, local_files_only=True).eval().requires_grad_(False).cuda()
        report['lora_modules'] = install_lora(model, args.rank)
        memory = GaussianMemory(features).cuda()
        parameters = [p for p in model.parameters() if p.requires_grad] + list(memory.parameters())
        report['trainable_parameters'] = sum(p.numel() for p in parameters)
        base = [(name, p) for name,p in model.named_parameters() if not p.requires_grad]
        # Version-counter guard on every frozen parameter; no base tensors saved/overwritten.
        base_versions = [(name,p._version) for name,p in base]
        model.enable_gradient_checkpointing()
        optimizer = torch.optim.AdamW(parameters, lr=8e-5, weight_decay=.01)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        def prediction(x, sigma, noise, condition):
            noisy = ((1-sigma)*x + sigma*noise).requires_grad_(True)
            return model(hidden_states=noisy,timestep=sigma.reshape(1)*1000,
                         encoder_hidden_states=condition,return_dict=False)[0]
        validation_noise = torch.randn(latents[4].shape, generator=torch.Generator(device='cuda').manual_seed(991), device='cuda',dtype=torch.bfloat16)
        sigma_v = torch.tensor(.5,device='cuda')
        with torch.no_grad():
            report['validation_base_mse'] = (prediction(latents[4],sigma_v,validation_noise,positive).float()-(validation_noise-latents[4]).float()).square().mean().item()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            x = latents[step % 4]
            sigma = torch.rand((),device='cuda')*.90+.05
            noise = torch.randn_like(x)
            pred = prediction(x,sigma,noise,memory(positive))
            loss = (pred.float()-(noise-x).float()).square().mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            optimizer.step()
            report['training'].append(dict(step=step+1,loss=loss.item(),gradient_norm=grad.item()))
            if step % 4 == 0 or step == args.steps-1: status(f'Training {step+1}/{args.steps}: flow loss {loss.item():.5f}')
        report['training_seconds'] = time.perf_counter()-start
        report['training_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['frozen_parameter_versions_unchanged'] = all(p._version==version for (name,p),(other,version) in zip(base,base_versions))
        assert report['frozen_parameter_versions_unchanged']
        with torch.no_grad():
            report['validation_adapted_mse'] = (prediction(latents[4],sigma_v,validation_noise,memory(positive)).float()-(validation_noise-latents[4]).float()).square().mean().item()
        adapters = {name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}
        report['nonzero_lora_up_tensors'] = sum(bool(value.count_nonzero()) for name,value in adapters.items() if name.endswith('.up'))
        save_inference_checkpoint(dict(lora=adapters,memory={k:v.detach().cpu() for k,v in memory.state_dict().items()},report=report),args.out/'cat_memory_adapter.pt')
        del optimizer, parameters, latents, pred, loss, noise, x
        free()
        model.disable_gradient_checkpointing()
        samples = {}
        with torch.inference_mode():
            for mode in ['base','adapted']:
                for module in model.modules():
                    if isinstance(module,LowRankLinear): module.enabled=mode=='adapted'
                scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,num_train_timesteps=1000,flow_shift=8.)
                pipe=WanPipeline(tokenizer=None,text_encoder=None,transformer=model,vae=None,scheduler=scheduler)
                condition=positive if mode=='base' else memory(positive)
                status(f'Generating paired {mode} video from fresh noise (same seed 71032)')
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                samples[mode]=pipe(prompt_embeds=condition,negative_prompt_embeds=negative,height=args.height,width=args.width,
                                   num_frames=9,num_inference_steps=args.sample_steps,guidance_scale=6.,
                                   generator=torch.Generator(device='cuda').manual_seed(71032),output_type='latent').frames.cpu()
                torch.cuda.synchronize()
                report['samples'][mode]=dict(denoise_seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),seed=71032,frames=9,steps=args.sample_steps)
                save_inference_checkpoint(dict(latent=samples[mode]),args.out/f'{mode}_latent.pt')
                del pipe
        del model, memory, base, positive, negative, condition
        free()
        vae=AutoencoderKLWan.from_pretrained(model_dir/'vae',torch_dtype=torch.float32,local_files_only=True).eval().cuda()
        mean=torch.tensor(vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
        std=torch.tensor(vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
        import imageio.v2 as imageio
        sheet=Image.new('RGB',(args.width*3,args.height*2))
        with torch.inference_mode():
            for row,(mode,latent) in enumerate(samples.items()):
                video=vae.decode(latent.cuda().float()*std+mean,return_dict=False)[0]
                frames=((video[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
                imageio.mimwrite(args.out/f'{mode}.mp4',frames,fps=8,codec='libx264',quality=8,macro_block_size=1)
                for col,index in enumerate([0,4,8]):
                    image=Image.fromarray(frames[index]); image.save(args.out/f'{mode}_{index:03d}.png')
                    sheet.paste(image,(col*args.width,row*args.height))
        sheet.save(args.out/'comparison.jpg')
        status('Completed pilot; visual acceptance pending, no efficiency or Gaussian-motion claim')


if __name__=='__main__':
    main()
