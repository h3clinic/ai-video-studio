"""Isolated inference ablations: no retraining or selection on these outputs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from real_video.wan_spatial_gaussian import SpatialControl, attach_control, projected_condition, free
from real_video.wan_cat_memory import install_lora
from real_video.checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake


def colour_extent(rgb):
    """Orange-only heuristic, NOT semantic segmentation or anatomy accuracy."""
    r, g, b = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)/255.
    yy, xx = np.nonzero((r > .3) & (r > g*1.15) & (r > b*1.4))
    if len(xx) < 10: return None
    return dict(center_x=float(xx.mean()), center_y=float(yy.mean()), pixels=len(xx),
                x_span_90=float(np.percentile(xx, 95)-np.percentile(xx, 5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline, UniPCMultistepScheduler
    root = Path('artifacts/real_video')
    model_dir = Path('../../work/wan21_13b')
    torch.set_num_threads(6)
    report = dict(checkpoint_sha256=digest(args.checkpoint), quality_accepted=False,
                  metric_limit='Orange pixel extent is a rough diagnostic, not semantic segmentation or proof of identity.',
                  protocol='Fixed checkpoint, seed, text, camera and generation settings. Change only Gaussian dx OR scale.',
                  outputs={}, no_post_generation_compositing=True)
    with keep_windows_awake():
        asset = load_verified(root/'hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt')
        camera = load_verified(root/'hunyuan_gaussian/v5_full_paint/camera.pt')
        conditions = {}
        for name, dx, scale in [('center', 0., 1.), ('shift_only', .45, 1.), ('scale_only', 0., .65)]:
            rgb, features, _ = projected_condition(asset, camera, 256, 448, dx, scale)
            pixels = (rgb.numpy()*255).round().astype(np.uint8)
            Image.fromarray(pixels).save(args.out/f'{name}_condition.png')
            conditions[name] = features
            report['outputs'][name] = dict(dx=dx, scale=scale, condition_extent=colour_extent(pixels))
        del asset
        free()
        state = load_verified(args.checkpoint)
        model = WanTransformer3DModel.from_pretrained(model_dir/'transformer', torch_dtype=torch.bfloat16, local_files_only=True).eval().requires_grad_(False).cuda()
        rank = next(v.shape[0] for k, v in state['lora'].items() if k.endswith('.down'))
        install_lora(model, rank)
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in state['lora'].items(): params[name].copy_(value)
        model.requires_grad_(False)
        control = SpatialControl(1536).cuda()
        control.load_state_dict(state['control'], strict=True)
        control.requires_grad_(False)
        attach_control(model, control)
        embeddings = load_verified(root/'wan_baseline/cat_seed_421001/prompt_embeddings.pt')
        positive, negative = [embeddings[key].cuda().to(torch.bfloat16) for key in ['positive', 'negative']]
        samples = {}
        with torch.inference_mode():
            for name, features in conditions.items():
                print('Generating isolated ablation:', name, flush=True)
                control.bind(features.cuda(), (3, 16, 28))
                pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=model, vae=None,
                                   scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=8.))
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                samples[name] = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
                                     height=256, width=448, num_frames=9, num_inference_steps=24, guidance_scale=6.,
                                     generator=torch.Generator(device='cuda').manual_seed(72102), output_type='latent').frames.cpu()
                torch.cuda.synchronize()
                report['outputs'][name].update(denoise_seconds=time.perf_counter()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated())
                save_inference_checkpoint(dict(latent=samples[name]), args.out/f'{name}_latent.pt')
                del pipe
        del model, control, params, state, positive, negative
        free()
        vae = AutoencoderKLWan.from_pretrained(model_dir/'vae', torch_dtype=torch.float32, local_files_only=True).eval().cuda()
        mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)
        import imageio.v2 as imageio
        sheet = Image.new('RGB', (448*4, 256*3))
        with torch.inference_mode():
            for row, (name, latent) in enumerate(samples.items()):
                decoded = vae.decode(latent.cuda().float()*std+mean, return_dict=False)[0]
                frames = ((decoded[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
                imageio.mimwrite(args.out/f'{name}.mp4', frames, fps=8, codec='libx264', quality=8, macro_block_size=1)
                sheet.paste(Image.open(args.out/f'{name}_condition.png'), (0, row*256))
                for col, index in enumerate([0, 4, 8]):
                    im = Image.fromarray(frames[index]); im.save(args.out/f'{name}_{index:03d}.png')
                    sheet.paste(im, ((col+1)*448, row*256))
                report['outputs'][name]['first_frame_extent'] = colour_extent(frames[0])
        sheet.save(args.out/'comparison.jpg')
        (args.out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
