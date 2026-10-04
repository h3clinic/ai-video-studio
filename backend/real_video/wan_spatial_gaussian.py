"""Experimental spatial Gaussian conditioning INSIDE Wan; RGB video output.

Paired renderer-derived supervision, not natural-video generalization. Gaussian
geometry is projected before denoising; no object is pasted into decoded frames.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from real_video.checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from real_video.gaussian3d import render, project
from real_video.wan_cat_memory import install_lora


class SpatialControl(nn.Module):
    """Zero-initialized residuals at spatially corresponding DiT tokens.

    Condition remains fixed through forward/backward checkpoint recomputation.
    First-time-only conditioning leaves future time slots for Wan's motion prior.
    This does not predict Gaussian trajectories or replace Wan's RGB VAE.
    """
    def __init__(self, dim, blocks=(0, 5, 10, 15, 20, 25), channels=14):
        super().__init__()
        self.blocks = tuple(blocks)
        self.encoder = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(64, dim) for _ in blocks])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.condition = None
        self.grid = None
        self.enabled = True

    def bind(self, condition, grid):
        if len(grid) != 3 or any(int(n) < 1 for n in grid):
            raise ValueError('Expected positive T,H,W token grid')
        self.condition, self.grid = condition, tuple(grid)

    def forward(self, hidden, head_index):
        if not self.enabled or self.condition is None:
            return hidden
        t, h, w = self.grid
        if hidden.shape[1] != t*h*w:
            raise ValueError('Spatial condition/token ordering mismatch')
        features = F.adaptive_avg_pool2d(self.condition.float(), (h, w))
        tokens = self.encoder(features).flatten(2).transpose(1, 2)
        delta = self.heads[head_index](tokens)
        if delta.shape[0] == 1:
            delta = delta.expand(hidden.shape[0], -1, -1)
        if delta.shape[0] != hidden.shape[0]:
            raise ValueError('Condition batch mismatch')
        # Only first latent frame, without in-place mutation of hidden states.
        if t > 1:
            delta = torch.cat([delta, delta.new_zeros(hidden.shape[0], (t-1)*h*w, delta.shape[-1])], 1)
        return hidden + delta.to(hidden.dtype)


def attach_control(model, control):
    handles = []
    for j, index in enumerate(control.blocks):
        def hook(module, inputs, output, j=j):
            return control(output, j)
        handles.append(model.blocks[index].register_forward_hook(hook))
    model.add_module('gaussian_spatial_control', control)
    return handles


@torch.no_grad()
def projected_condition(asset, camera, height, width, dx=0., scale=1.):
    """World-space transformation; never refit camera/bounds per object.

    Visibility-weighted RGB, alpha, depth, normal, XYZ, projected covariance.
    Fixed units preserve relative size/location. Full 3D covariance scales s^2.
    """
    p = asset['position'].cuda().float()*scale
    p = p + p.new_tensor([dx, 0., 0.])
    covariance = asset['covariance'].cuda().float()*scale**2
    eye, target = camera['eye'].cuda().float(), camera['target'].cuda().float()
    rgb, alpha, cache = render(p, covariance, asset['colour'].cuda().float(),
                              asset['opacity'].cuda().float().reshape(-1), eye, target,
                              height, width, float(camera['fov']), radius=2,
                              ground=False, return_cache=True)
    _, cov2, depth, _, _ = project(p, covariance, eye, target, height, width, float(camera['fov']))
    cov_features = torch.stack([cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]], -1)/height**2
    values = torch.cat([asset['colour'].cuda().float(), depth[:, None]/4.,
                        asset['normal'].cuda().float(), p, cov_features], -1)
    summed = p.new_zeros(height*width, 13)
    # Limit temporary feature gather memory, not the number of Gaussians.
    for start in range(0, len(cache['ids']), 100000):
        end = start+100000
        summed.index_add_(0, cache['pixel'][start:end],
                          cache['weight'][start:end, None]*values[cache['ids'][start:end]])
    features = torch.cat([summed.reshape(height, width, 13), alpha[..., None]], -1)
    features = F.avg_pool2d(features.permute(2, 0, 1)[None], 8).cpu()
    return rgb.cpu(), features, alpha.cpu()


def free():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=448)
    parser.add_argument('--sample-steps', type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh run directory')
    if args.height % 16 or args.width % 16:
        raise ValueError('Resolution must be divisible by 16')
    args.out.mkdir(parents=True)
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline, UniPCMultistepScheduler
    root = Path('artifacts/real_video')
    asset_path = root/'hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt'
    model_dir = Path('../../work/wan21_13b')
    lora_path = root/'wan_cat_memory/v1/cat_memory_adapter.pt'
    report = dict(scope='Spatial Gaussian control inside Wan, RGB video output', quality_accepted=False,
                  supervision='Rendered views of existing inferred cat asset; NOT external real-video validation',
                  gaussian_sha256=digest(asset_path), starting_lora_sha256=digest(lora_path),
                  condition='First latent frame only; same condition on both CFG branches',
                  training=[], samples={}, limitations=['No learned Gaussian dynamics',
                  'No demonstrated multi-object composition', 'No compute savings established',
                  'Small single-asset renderer-supervised pilot, no generalization claim'])
    def status(message):
        print(message, flush=True)
        (args.out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    torch.set_num_threads(6)
    torch.manual_seed(72101)
    with keep_windows_awake():
        asset = load_verified(asset_path)
        camera = load_verified(root/'hunyuan_gaussian/v5_full_paint/camera.pt')
        configs = [(x, s) for x in [-.45, 0., .45] for s in [.8, 1.2]] + [(0., 1.), (.25, 1.1), (.45, 1.)]
        data = []
        for i, (dx, scale) in enumerate(configs):
            status(f'Projecting all {len(asset["position"])} Gaussians: paired view {i+1}/{len(configs)}')
            rgb, features, alpha = projected_condition(asset, camera, args.height, args.width, dx, scale)
            Image.fromarray((rgb.numpy()*255).round().astype(np.uint8)).save(args.out/f'condition_{i}.png')
            data.append(dict(rgb=rgb, features=features, alpha=alpha, dx=dx, scale=scale))
        report['training_views'] = list(range(6))
        report['heldout_views'] = [6, 7]
        report['validation_scope'] = 'Same asset, unseen transforms; not independent identity validation'
        save_inference_checkpoint(dict(data=data), args.out/'projected_data.pt')
        del asset
        free()
        status('Encoding paired render targets')
        vae = AutoencoderKLWan.from_pretrained(model_dir/'vae', torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).cuda()
        mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)
        with torch.no_grad():
            latents = [((vae.encode(d['rgb'].permute(2, 0, 1)[None, :, None].cuda()*2-1).latent_dist.mode()-mean)/std).to(torch.bfloat16) for d in data]
        del vae, mean, std
        free()
        embeddings = load_verified(root/'wan_baseline/cat_seed_421001/prompt_embeddings.pt')
        positive, negative = [embeddings[key].cuda().to(torch.bfloat16) for key in ['positive', 'negative']]
        model = WanTransformer3DModel.from_pretrained(model_dir/'transformer', torch_dtype=torch.bfloat16, local_files_only=True).eval().requires_grad_(False).cuda()
        old = load_verified(lora_path)
        install_lora(model, old['report']['rank'])
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if set(trainable) != set(old['lora']):
            raise ValueError('LoRA schema mismatch')
        with torch.no_grad():
            for name, p in trainable.items(): p.copy_(old['lora'][name])
        model.requires_grad_(False)
        # Freeze previous appearance LoRA; isolate the new spatial connection.
        control = SpatialControl(model.config.num_attention_heads*model.config.attention_head_dim).cuda()
        attach_control(model, control)
        base = [(p, p._version) for n, p in model.named_parameters() if not p.requires_grad]
        params = list(control.parameters())
        report['trainable_parameters'] = sum(p.numel() for p in params)
        optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=.01)
        model.enable_gradient_checkpointing()
        grid = (1, args.height//16, args.width//16)
        def predict(x, noise, sigma, view, enabled=True):
            control.enabled = enabled
            control.bind(data[view]['features'].cuda(), grid)
            return model(hidden_states=((1-sigma)*x+sigma*noise).requires_grad_(True),
                         timestep=sigma.reshape(1)*1000, encoder_hidden_states=positive, return_dict=False)[0]
        def validation():
            result = []
            with torch.no_grad():
                for view in [6, 7]:
                    x = latents[view]
                    noise = torch.randn(x.shape, device='cuda', dtype=x.dtype, generator=torch.Generator(device='cuda').manual_seed(72200+view))
                    sigma = torch.tensor(.7, device='cuda')
                    result.append({mode: (predict(x, noise, sigma, view, enabled).float()-(noise-x).float()).square().mean().item()
                                   for mode, enabled in [('off', False), ('on', True)]})
            return result
        report['before_validation'] = validation()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(args.steps):
            view = step % 6
            x = latents[view]
            sigma = torch.rand((), device='cuda')*.9+.05
            noise = torch.randn_like(x)
            optimizer.zero_grad(set_to_none=True)
            pred = predict(x, noise, sigma, view)
            loss = (pred.float()-(noise-x).float()).square().mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            report['training'].append(dict(step=step+1, loss=loss.item(), gradient_norm=grad.item()))
            if step % 16 == 0 or step == args.steps-1:
                status(f'Spatial control training {step+1}/{args.steps}: {loss.item():.5f}')
        torch.cuda.synchronize()
        report['training_seconds'] = time.perf_counter()-start
        report['training_peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        report['frozen_weights_unchanged'] = all(p._version == version for p, version in base)
        assert report['frozen_weights_unchanged']
        report['after_validation'] = validation()
        report['nonzero_control_heads'] = sum(bool(h.weight.count_nonzero()) for h in control.heads)
        save_inference_checkpoint(dict(control={k: v.cpu() for k, v in control.state_dict().items()},
                                       lora=old['lora'], report=report), args.out/'spatial_adapter.pt')
        del optimizer, pred, loss, noise, x, latents, old, trainable, params, base
        free()
        model.disable_gradient_checkpointing()
        samples = {}
        for mode, view, enabled in [('off', 6, False), ('center', 6, True), ('translated', 8, True)]:
            control.enabled = enabled
            control.bind(data[view]['features'].cuda(), (3, args.height//16, args.width//16))
            scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=8.)
            pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=model, vae=None, scheduler=scheduler)
            status(f'Generating {mode} from identical noise; no output compositing')
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                samples[mode] = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
                                     height=args.height, width=args.width, num_frames=9,
                                     num_inference_steps=args.sample_steps, guidance_scale=6.,
                                     generator=torch.Generator(device='cuda').manual_seed(72102), output_type='latent').frames.cpu()
            torch.cuda.synchronize()
            report['samples'][mode] = dict(denoise_seconds=time.perf_counter()-start,
                                          peak_cuda_bytes=torch.cuda.max_memory_allocated(), view=view,
                                          steps=args.sample_steps, seed=72102, frames=9)
            save_inference_checkpoint(dict(latent=samples[mode]), args.out/f'{mode}_latent.pt')
            del pipe
        control.condition = None
        del model, control, positive, negative
        free()
        vae = AutoencoderKLWan.from_pretrained(model_dir/'vae', torch_dtype=torch.float32, local_files_only=True).eval().cuda()
        mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)
        import imageio.v2 as imageio
        sheet = Image.new('RGB', (args.width*3, args.height*3))
        with torch.inference_mode():
            for row, (mode, latent) in enumerate(samples.items()):
                decoded = vae.decode(latent.cuda().float()*std+mean, return_dict=False)[0]
                frames = ((decoded[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
                imageio.mimwrite(args.out/f'{mode}.mp4', frames, fps=8, codec='libx264', quality=8, macro_block_size=1)
                for col, i in enumerate([0, 4, 8]):
                    im = Image.fromarray(frames[i]); im.save(args.out/f'{mode}_{i:03d}.png')
                    sheet.paste(im, (col*args.width, row*args.height))
        sheet.save(args.out/'comparison.jpg')
        status('Pilot finished; visual/spatial-control acceptance still pending')


if __name__ == '__main__':
    main()
