"""Pinned, staged Wan baseline. This is NOT our Gaussian generator.

Keeps generated latents for a future latent-to-Gaussian decoder experiment.
No remote Python code, global installs, or source video inputs are used.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_XET_NUM_CONCURRENT_RANGE_GETS', '4')

import torch
from .checkpoint_io import keep_windows_awake, save_inference_checkpoint, load_verified, digest

MODEL = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
REVISION = '0fad780a534b6463e45facd96134c9f345acfa5b'
PROMPT = ('A realistic orange tabby cat walks slowly across a green garden lawn in daylight. '
          'Full body, natural walking motion, detailed fur, steady camera, continuous shot.')
NEGATIVE = ('blurry, low quality, distorted anatomy, extra legs, subtitles, watermark, '
            'static image, illustration, cartoon')


def validate_args(args):
    if args.height % 16 or args.width % 16 or min(args.height, args.width) < 64:
        raise ValueError('Spatial dimensions must be positive multiples of 16, at least 64')
    if args.frames < 5 or (args.frames - 1) % 4:
        raise ValueError('Frames must be 4k+1, at least 5')
    if args.steps < 1:
        raise ValueError('Steps must be positive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/real_video/wan_baseline/cat_seed_421001'))
    parser.add_argument('--model-dir', type=Path, default=Path('../../work/wan21_13b'))
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--frames', type=int, default=33)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=421001)
    parser.add_argument('--prompt', default=PROMPT)
    args = parser.parse_args()
    validate_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(model=MODEL, revision=REVISION, prompt=args.prompt, negative_prompt=NEGATIVE,
                  height=args.height, width=args.width, frames=args.frames, steps=args.steps,
                  seed=args.seed, guidance_scale=6.0, scheduler='UniPCMultistepScheduler',
                  flow_shift=8.0, fps=16, role='unchanged_pretrained_Wan_RGB_baseline_NOT_Gaussian',
                  text_encoder_device='cpu', transformer_dtype='bfloat16', vae_dtype='float32')
    config_path = args.out / 'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing output belongs to a different protocol; choose a new directory')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    if (args.out / 'metrics.json').exists():
        raise FileExistsError('Completed run already exists; choose a new output directory')
    metrics = dict(config=config, timings={}, torch=torch.__version__)

    def status(stage, **extra):
        record = dict(stage=stage, time=time.time(), **extra)
        (args.out / 'progress.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        with (args.out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def free():
        gc.collect()
        torch.cuda.empty_cache()

    try:
        with keep_windows_awake(), torch.inference_mode():
            torch.set_num_threads(8)
            from huggingface_hub import snapshot_download
            status('download', model=MODEL, revision=REVISION)
            start = time.perf_counter()
            snapshot_download(MODEL, revision=REVISION, local_dir=args.model_dir,
                              allow_patterns=['model_index.json', 'scheduler/*', 'tokenizer/*',
                                              'text_encoder/*', 'transformer/*', 'vae/*'], max_workers=2)
            metrics['timings']['download_seconds'] = time.perf_counter() - start
            from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
            from transformers import AutoTokenizer, UMT5EncoderModel
            import diffusers
            import transformers
            metrics.update(diffusers=diffusers.__version__, transformers=transformers.__version__,
                           gpu=torch.cuda.get_device_name())
            scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True,
                                               num_train_timesteps=1000, flow_shift=8.0)
            embeddings_path = args.out / 'prompt_embeddings.pt'
            if not embeddings_path.exists():
                status('load_text_encoder_cpu')
                free()
                import psutil
                available = psutil.virtual_memory().available
                metrics['cpu_available_before_text_load_bytes'] = available
                if available < 13.5 * 1024**3:
                    raise MemoryError('Less than 13.5 GiB CPU RAM available for staged BF16 text encoding. '
                                      'Downloads are retained; restart with more free RAM.')
                start = time.perf_counter()
                encoder = UMT5EncoderModel.from_pretrained(args.model_dir / 'text_encoder',
                                                          dtype=torch.bfloat16, local_files_only=True).eval()
                tokenizer = AutoTokenizer.from_pretrained(args.model_dir / 'tokenizer', local_files_only=True)
                pipe = WanPipeline(tokenizer=tokenizer, text_encoder=encoder, vae=None,
                                   scheduler=scheduler, transformer=None)
                metrics['timings']['text_load_seconds'] = time.perf_counter() - start
                status('encode_prompt_cpu')
                start = time.perf_counter()
                positive, negative = pipe.encode_prompt(args.prompt, NEGATIVE, device=torch.device('cpu'),
                                                        dtype=torch.bfloat16, max_sequence_length=512)
                metrics['timings']['text_encode_seconds'] = time.perf_counter() - start
                save_inference_checkpoint(dict(positive=positive.cpu(), negative=negative.cpu(), config=config),
                                          embeddings_path)
                del pipe, encoder, tokenizer, positive, negative
                free()
            else:
                metrics['prompt_embeddings_reused'] = True
            embeddings = load_verified(embeddings_path)
            latent_path = args.out / 'generated_latent.pt'
            if not latent_path.exists():
                status('load_transformer')
                start = time.perf_counter()
                transformer = WanTransformer3DModel.from_pretrained(args.model_dir / 'transformer',
                                    torch_dtype=torch.bfloat16, local_files_only=True).eval().to('cuda')
                pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=None,
                                   scheduler=scheduler, transformer=transformer)
                metrics['timings']['transformer_load_seconds'] = time.perf_counter() - start
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()

                def progress(pipeline, step, timestep, kwargs):
                    torch.cuda.synchronize()
                    status('denoising', step=step + 1, total=args.steps,
                           elapsed_seconds=time.perf_counter() - start,
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                    return kwargs

                latent = pipe(prompt_embeds=embeddings['positive'].to('cuda'),
                              negative_prompt_embeds=embeddings['negative'].to('cuda'),
                              height=args.height, width=args.width, num_frames=args.frames,
                              num_inference_steps=args.steps, guidance_scale=6.0,
                              generator=torch.Generator(device='cuda').manual_seed(args.seed),
                              output_type='latent', callback_on_step_end=progress).frames
                torch.cuda.synchronize()
                metrics['timings']['denoise_seconds'] = time.perf_counter() - start
                metrics['denoise_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                save_inference_checkpoint(dict(latent=latent.cpu(), config=config), latent_path)
                del pipe, transformer, latent
                free()
            else:
                metrics['latent_reused'] = True
            del embeddings
            status('decode_original_vae')
            start = time.perf_counter()
            vae = AutoencoderKLWan.from_pretrained(args.model_dir / 'vae', torch_dtype=torch.float32,
                                                  local_files_only=True).eval().to('cuda')
            vae.enable_tiling()
            metrics['timings']['vae_load_seconds'] = time.perf_counter() - start
            latent = load_verified(latent_path)['latent'].to('cuda', dtype=torch.float32)
            mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            video = vae.decode(latent * std + mean, return_dict=False)[0]
            torch.cuda.synchronize()
            metrics['timings']['decode_seconds'] = time.perf_counter() - start
            metrics['decode_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            if not torch.isfinite(video).all():
                raise ValueError('Nonfinite decoded output')
            frames = ((video[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
            del vae, latent, video, mean, std
            free()
            import imageio.v2 as imageio
            from PIL import Image
            output = args.out / 'wan_original.mp4'
            imageio.mimwrite(output, frames, fps=16, codec='libx264', quality=8, macro_block_size=1)
            indices = [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]
            sheet = Image.new('RGB', (args.width*2, args.height*2))
            for i, index in enumerate(indices):
                sheet.paste(Image.fromarray(frames[index]), ((i%2)*args.width, (i//2)*args.height))
            sheet.save(args.out / 'contact_sheet.jpg')
            metrics.update(video_sha256=digest(output), latent_sha256=digest(latent_path),
                           decoded_shape=list(frames.shape), video_bytes=output.stat().st_size,
                           memory_note='PyTorch allocated tensors, not whole-process VRAM or CPU RAM; cold single run',
                           no_gaussian_claim=True)
            (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            status('complete', video=str(output.resolve()))
    except Exception:
        status('failed', traceback=traceback.format_exc())
        (args.out / 'partial_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
