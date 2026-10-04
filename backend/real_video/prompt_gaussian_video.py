"""Fresh text/noise -> Wan latent -> trained planar Gaussians -> video.

No input video, image, prior scene latent, RGB VAE, mesh or cat asset. Cached
text embeddings are allowed only for the EXACT requested text/model revision.
This is the existing hybrid generator, not the experimental 3D dynamics model.
"""
import argparse
import builtins
from contextlib import contextmanager
import gc
import io
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch
from .checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint
from .wan_baseline import MODEL, REVISION, PROMPT, NEGATIVE, validate_args


def validate_text_packet(packet, prompt, negative):
    config = packet['config']
    if any(config.get(k) != v for k, v in dict(prompt=prompt, negative_prompt=negative,
                                               model=MODEL, revision=REVISION).items()):
        raise ValueError('Text embedding provenance does not match requested prompt/model')
    for name in ['positive', 'negative']:
        value = packet[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 512, 4096):
            raise ValueError('Expected [1,512,4096] text embeddings')
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('Text embeddings must be finite floating tensors')


@contextmanager
def inference_input_guard(model_dir, embeddings_path, decoder_path):
    """Python-level read guards, not an OS sandbox or a total I/O proof.

    Model libraries can use native I/O. The explicit pipeline below only loads
    local pretrained transformer weights and two named tensor packets.
    """
    directory = Path(model_dir).resolve()
    allowed = {Path(embeddings_path).resolve(), Path(decoder_path).resolve()}
    original_open, original_io, original_load = builtins.open, io.open, torch.load
    observed = []
    media = {'.mp4', '.avi', '.mov', '.mkv', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.npy', '.npz'}

    def check(path, mode='r'):
        if isinstance(path, int) or not isinstance(path, (str, bytes, os.PathLike)):
            return
        resolved = Path(os.fsdecode(path)).resolve()
        if '+' not in str(mode) and any(flag in str(mode) for flag in ['w', 'a', 'x']):
            return
        ext = resolved.suffix.lower()
        if ext in media or (ext in {'.pt', '.pth'} and resolved not in allowed):
            raise AssertionError(f'Source media/prior-scene tensor input forbidden: {resolved}')
        if ext in {'.safetensors', '.bin'} and not resolved.is_relative_to(directory):
            raise AssertionError(f'Unapproved pretrained weights: {resolved}')
        if ext in {'.pt', '.pth', '.safetensors', '.bin'}:
            observed.append(str(resolved))

    def guarded_open(path, mode='r', *args, **kwargs):
        check(path, mode)
        return original_open(path, mode, *args, **kwargs)

    def guarded_io(path, mode='r', *args, **kwargs):
        check(path, mode)
        return original_io(path, mode, *args, **kwargs)

    def guarded_load(path, *args, **kwargs):
        if not isinstance(path, (str, bytes, os.PathLike)):
            raise AssertionError('Anonymous tensor inputs forbidden')
        check(path)
        return original_load(path, *args, **kwargs)

    with patch('builtins.open', side_effect=guarded_open), patch('io.open', side_effect=guarded_io), \
         patch('torch.load', side_effect=guarded_load):
        yield observed


def main(args):
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError('This local inference runner requires CUDA')
    if args.out.exists():
        raise FileExistsError('Choose a new run directory; preserve old experiments')
    if not args.embeddings.with_suffix('.pt.sha256.json').exists() or not args.checkpoint.with_suffix('.pt.sha256.json').exists():
        raise ValueError('Verified embedding/decoder sidecars are required')
    embeddings = load_verified(args.embeddings)
    validate_text_packet(embeddings, args.prompt, NEGATIVE)
    del embeddings
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    config = dict(model=MODEL, revision=REVISION, prompt=args.prompt, negative_prompt=NEGATIVE,
        height=args.height, width=args.width, frames=args.frames, steps=args.steps,
        seed=args.seed, guidance_scale=6., flow_shift=8., fps=16,
        role='fresh_text_noise_Wan_latent_to_trained_planar_Gaussians',
        geometry='2D anisotropic Gaussian fields; not persistent 3D scene dynamics',
        input_image=False, input_video=False, old_scene_latent_reused=False,
        pretrained_Wan_denoiser_unchanged=True, original_RGB_VAE_used=False,
        reused_only_exact_prompt_text_embeddings=True)
    inputs = dict(text_embeddings=dict(path=str(args.embeddings.resolve()), sha256=digest(args.embeddings)),
                  gaussian_decoder=dict(path=str(args.checkpoint.resolve()), sha256=digest(args.checkpoint)),
                  transformer_files={str(p.resolve()):digest(p) for p in sorted((args.model_dir/'transformer').glob('*'))
                                     if p.is_file() and p.suffix in {'.safetensors', '.json'}})
    if not any(p.endswith('.safetensors') for p in inputs['transformer_files']):
        raise FileNotFoundError('Local pretrained transformer weights missing')
    report = dict(config=config, inputs=inputs, timings={}, quality_accepted=False,
                  quality_reason='Awaiting direct visual inspection; no automatic acceptance',
                  memory_scope='PyTorch allocated GPU tensors, not total system memory; no savings claim')
    (args.out/'protocol.json').write_text(json.dumps(report, indent=2))
    (args.out/'runner.py').write_bytes(Path(__file__).read_bytes())

    def status(stage, **extra):
        record = dict(stage=stage, unix_time=time.time(), **extra)
        (args.out/'progress.json').write_text(json.dumps(record, indent=2))
        with (args.out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)

    begin = time.perf_counter()
    try:
        with keep_windows_awake(), torch.inference_mode():
            from diffusers import WanPipeline, WanTransformer3DModel, UniPCMultistepScheduler
            from .generate_wan_gaussian import generate
            status('load_local_transformer')
            with inference_input_guard(args.model_dir, args.embeddings, args.checkpoint) as reads:
                embeddings = load_verified(args.embeddings)
                validate_text_packet(embeddings, args.prompt, NEGATIVE)
                started = time.perf_counter()
                transformer = WanTransformer3DModel.from_pretrained(args.model_dir/'transformer',
                    torch_dtype=torch.bfloat16, local_files_only=True).eval().to('cuda')
                scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction',
                    use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=8.)
                pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=None,
                    scheduler=scheduler, transformer=transformer)
                assert pipe.vae is None
                report['timings']['transformer_load_seconds'] = time.perf_counter()-started
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                started = time.perf_counter()

                def progress(pipeline, step, timestep, kwargs):
                    torch.cuda.synchronize()
                    status('fresh_noise_denoising', step=step+1, total=args.steps,
                           elapsed_seconds=time.perf_counter()-started,
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                    return kwargs

                latent = pipe(prompt_embeds=embeddings['positive'].to('cuda'),
                    negative_prompt_embeds=embeddings['negative'].to('cuda'),
                    height=args.height, width=args.width, num_frames=args.frames,
                    num_inference_steps=args.steps, guidance_scale=6., output_type='latent',
                    generator=torch.Generator(device='cuda').manual_seed(args.seed),
                    callback_on_step_end=progress).frames
                torch.cuda.synchronize()
                report['timings']['fresh_denoise_seconds'] = time.perf_counter()-started
                report['denoise_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                latent_packet = dict(latent=latent.cpu(), config=config)
                del pipe, transformer, latent, embeddings
                gc.collect(); torch.cuda.empty_cache()
                denoise_reads = list(reads)
            latent_path = args.out/'generated_latent.pt'
            report['latent_sha256'] = save_inference_checkpoint(latent_packet, latent_path)
            status('learned_Gaussian_decode_and_splat')
            with inference_input_guard(args.model_dir, args.embeddings, args.checkpoint) as reads:
                checkpoint = load_verified(args.checkpoint)
                if checkpoint.get('model_revision') != REVISION:
                    raise ValueError('Gaussian decoder was trained for a different Wan revision')
                frames, fields, metrics = generate(checkpoint, latent_packet)
                decoder_reads = list(reads)
            report['decoder'] = metrics
            report['gaussian_fields_sha256'] = save_inference_checkpoint(dict(fields=fields,
                height=args.height, width=args.width, checkpoint_sha256=inputs['gaussian_decoder']['sha256'],
                latent_sha256=report['latent_sha256'], config=config), args.out/'gaussian_fields.pt')
            import imageio.v2 as imageio
            from PIL import Image
            video = args.out/'orange_cat_gaussian.mp4'
            imageio.mimwrite(video, frames, fps=16, codec='libx264', quality=8, macro_block_size=1)
            indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
            for index in indices:
                Image.fromarray(frames[index]).save(args.out/f'frame_{index:03d}.png')
            sheet = Image.new('RGB', (args.width*3, args.height*2), '#171b22')
            for index, frame_index in enumerate(indices):
                sheet.paste(Image.fromarray(frames[frame_index]),
                            ((index%3)*args.width, (index//3)*args.height))
            sheet.save(args.out/'contact_sheet.jpg')
            report.update(video_sha256=digest(video), frame_shape=list(frames.shape),
                gaussian_field_shape=list(fields.shape), duration_seconds=len(frames)/16,
                python_guarded_weight_reads=sorted(set(denoise_reads+decoder_reads)),
                input_guard_limit='Python read guard and inspected path, not OS-level sandbox; native model weight loading may bypass Python open.',
                generation_source_rgb_reads=0, old_cat_asset_loaded=False,
                total_seconds=time.perf_counter()-begin)
            (args.out/'metrics.json').write_text(json.dumps(report, indent=2))
            status('complete', video=str(video.resolve()))
    except Exception:
        status('failed', traceback=traceback.format_exc())
        (args.out/'partial_metrics.json').write_text(json.dumps(report, indent=2))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/real_video/prompt_generation/orange_cat_seed_531002'))
    parser.add_argument('--model-dir', type=Path, default=Path('../../work/wan21_13b'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/real_video/wan_bridge/v2/gaussian_decoder.pt'))
    parser.add_argument('--embeddings', type=Path, default=Path('artifacts/real_video/wan_baseline/cat_seed_421001/prompt_embeddings.pt'))
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--seed', type=int, default=531002)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--frames', type=int, default=33)
    parser.add_argument('--steps', type=int, default=50)
    main(parser.parse_args())
