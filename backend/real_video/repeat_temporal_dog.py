"""Matched dog insertion experiment, not post-generation video composition.

A: moving Gaussian cat condition + trained temporal adapter + joint prompt.
B: original Wan + same joint prompt. C: same trained weights as A, control off.
All outputs are RGB video. No generated dog geometry is written to Gaussians.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

import imageio.v2 as imageio
import torch
from PIL import Image, ImageDraw

from real_video.checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint
from real_video.dog_scene_experiment import Measure, PROMPTS
from real_video.sample_gaussian_temporal_memory import (
    MODEL, ASSET, BUNDLE, HEIGHT, WIDTH, FRAMES, FPS, EVIDENCE, prepare_condition,
)
from real_video.wan_cat_memory import LowRankLinear, install_lora
from real_video.wan_temporal_control import TemporalSpatialControl, attach_temporal_control

CASES = ('A_gaussian_memory', 'B_original_wan', 'C_adapted_without_memory')
EMBEDDINGS = Path('artifacts/real_video/dog_scene_comparison/v1/joint/prompt_embeddings.pt')


def configure_case(model, control, case):
    if case not in CASES:
        raise ValueError('Unknown experiment case')
    adapted = case != 'B_original_wan'
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            module.enabled = adapted
    if control is not None:
        control.enabled = case == 'A_gaussian_memory'
    return dict(lora_enabled=adapted, gaussian_control_enabled=case == 'A_gaussian_memory')


def free():
    gc.collect()
    torch.cuda.empty_cache()


def temperature():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=5)
        return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def record(out, report):
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


@torch.inference_mode()
def sample_case(out, case, checkpoint, embeddings, steps, condition_path):
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline, UniPCMultistepScheduler
    target = out/case
    target.mkdir()
    measured = dict(case=case, quality_accepted=False, start_gpu_temperature_c=temperature())
    with Measure() as loading:
        model = WanTransformer3DModel.from_pretrained(MODEL/'transformer', torch_dtype=torch.bfloat16,
            local_files_only=True).eval().requires_grad_(False).cuda()
        control = None
        if case != 'B_original_wan':
            packet = load_verified(checkpoint)
            if packet.get('rank') != 4 or packet.get('control_channels') != 5:
                raise ValueError('Expected rank4/5-channel temporal checkpoint')
            install_lora(model, 4)
            control = TemporalSpatialControl(1536, channels=5).cuda()
            attach_temporal_control(model, control)
            parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
            if parameters.keys() != packet['weights'].keys():
                raise ValueError('Strict checkpoint parameter mismatch')
            for name, parameter in parameters.items():
                value = packet['weights'][name]
                if parameter.shape != value.shape:
                    raise ValueError(f'Checkpoint shape mismatch: {name}')
                parameter.copy_(value)
            measured['trained_parameters'] = sum(p.numel() for p in parameters.values())
            del parameters, packet, value, parameter
        if case == 'A_gaussian_memory':
            saved = load_verified(condition_path)
            if saved['gaussian_sha256'] != digest(ASSET):
                raise ValueError('Gaussian provenance mismatch')
            condition = saved['condition'].cuda()
            if tuple(condition.shape) != (1, 5, 15, 32, 56):
                raise ValueError('Unexpected condition shape')
            control.bind(condition, (15, HEIGHT//16, WIDTH//16))
            measured['condition_bytes'] = condition.numel()*condition.element_size()
            del saved, condition
        measured.update(configure_case(model, control, case))
        model.requires_grad_(False)
        positive, negative = (embeddings[k].cuda().to(torch.bfloat16) for k in ('positive', 'negative'))
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True,
            num_train_timesteps=1000, flow_shift=8.)
        pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=model, vae=None, scheduler=scheduler)
    measured['transformer_and_adapter_load'] = loading.result
    free()
    print(f'Generating {case}: {FRAMES} frames, {steps} steps', flush=True)
    with Measure() as denoising:
        latent = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
            height=HEIGHT, width=WIDTH, num_frames=FRAMES, num_inference_steps=steps,
            guidance_scale=5., generator=torch.Generator(device='cuda').manual_seed(73007),
            output_type='latent').frames.cpu()
    measured['denoise'] = denoising.result
    measured['end_denoise_gpu_temperature_c'] = temperature()
    save_inference_checkpoint(dict(latent=latent, case=case, prompt=embeddings['prompt']), target/'latent.pt')
    if control is not None:
        control.condition = None
    del pipe, model, control, positive, negative
    free()
    with Measure() as vae_loading:
        vae = AutoencoderKLWan.from_pretrained(MODEL/'vae', torch_dtype=torch.float32,
            local_files_only=True).eval().requires_grad_(False).cuda()
        vae.enable_tiling()
        mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)
    measured['vae_load'] = vae_loading.result
    with Measure() as decoding:
        video = vae.decode(latent.cuda().float()*std+mean, return_dict=False)[0]
        frames = ((video[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
    measured['decode'] = decoding.result
    if frames.shape != (FRAMES, HEIGHT, WIDTH, 3):
        raise ValueError('Unexpected decoded shape')
    imageio.mimwrite(target/'video.mp4', frames, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    sheet = Image.new('RGB', (WIDTH*len(EVIDENCE), HEIGHT+24))
    ImageDraw.Draw(sheet).text((6, 5), case, fill='white')
    for col, index in enumerate(EVIDENCE):
        im = Image.fromarray(frames[index])
        im.save(target/f'frame_{index:03d}.png')
        sheet.paste(im, (col*WIDTH, 24))
    sheet.save(target/'samples.jpg')
    measured['video_sha256'] = digest(target/'video.mp4')
    measured['output_representation'] = 'Wan RGB video; no emitted or updated Gaussian dog geometry'
    measured['video_file_bytes'] = (target/'video.mp4').stat().st_size
    (target/'metrics.json').write_text(json.dumps(measured, indent=2), encoding='utf-8')
    del vae, video, latent, frames, mean, std
    free()
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/real_video/wan_temporal_memory/v3/temporal_adapter.pt'))
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--case', choices=('prepare',)+CASES+('all',), default='all')
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError('Positive steps required')
    if args.case in ('prepare', 'all'):
        if args.out.exists():
            raise FileExistsError('Preserve existing experiment')
        args.out.mkdir(parents=True)
        report = dict(scope='Matched model-level cat-memory plus requested dog; not legacy planar dog composition',
            prompt=PROMPTS['joint'], checkpoint_sha256=digest(args.checkpoint),
            embeddings_sha256=digest(EMBEDDINGS), frames=FRAMES, fps=FPS, duration_seconds=FRAMES/FPS,
            width=WIDTH, height=HEIGHT, steps=args.steps, seed=73007, guidance_scale=5.,
            cases={}, quality_accepted=False, generated_gaussian_writeback=False,
            size_review_gate='Both animals fully visible, same ground/depth; adult dog not miniaturized. If unoccluded comparable shoulder height is measurable, target dog/cat ratio 1.4 to 2.5. This is a review criterion, NOT model-enforced.',
            exclusions='Text embeddings cached from exact old joint prompt; no text-encoding cost included. No new training cost. Loading and Gaussian preparation reported separately.',
            limitations=['Single seed, no reranking or statistical efficiency claim',
                'A vs C isolates conditioning with fixed adapted weights; A vs B also changes weights',
                'No hard dog bounding-box control or dog-specific Gaussian memory yet',
                'Full precomputed Gaussian condition, not an online causal Wan rollout',
                'GPU temperature recorded; serial timings are descriptive, not stable speedup estimates'])
        record(args.out, report)
    else:
        report = json.loads((args.out/'report.json').read_text())
        if report['checkpoint_sha256'] != digest(args.checkpoint) or report['steps'] != args.steps:
            raise ValueError('Immutable protocol mismatch')
    torch.set_num_threads(4)
    with keep_windows_awake():
        if args.case in ('prepare', 'all'):
            gaussian = args.out/'gaussian_condition'
            gaussian.mkdir()
            preparation = dict(quality_accepted=False, scope='Fresh controller-generated motion on stored Gaussian cat')
            prepare_condition(gaussian, preparation, BUNDLE, 'ik')
            report['gaussian_preparation'] = preparation['gaussian_preparation']
            record(args.out, report)
            free()
        embeddings = load_verified(EMBEDDINGS)
        if embeddings.get('prompt') != report['prompt']:
            raise ValueError('Cached embeddings do not match exact joint prompt')
        selected = CASES if args.case == 'all' else (() if args.case == 'prepare' else (args.case,))
        for case in selected:
            report['cases'][case] = sample_case(args.out, case, args.checkpoint, embeddings, args.steps,
                args.out/'gaussian_condition/temporal_condition.pt')
            record(args.out, report)
        if set(report['cases']) == set(CASES):
            sheet = Image.new('RGB', (WIDTH*len(EVIDENCE), (HEIGHT+24)*3))
            for row, case in enumerate(CASES):
                sheet.paste(Image.open(args.out/case/'samples.jpg'), (0, row*(HEIGHT+24)))
            sheet.save(args.out/'comparison.jpg')
            report['generation_finished'] = True
            record(args.out, report)


if __name__ == '__main__':
    main()
