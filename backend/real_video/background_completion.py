"""One-time local diffusion completion -> stored hidden-background Gaussians.

Static scene completion, NOT Wan training or generated animal motion.
Mask is an approximate colour/GrabCut heuristic for this orange-cat diagnostic.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
import torch.nn.functional as F
from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from .detail_lift import lift
from .gaussian3d import render
from .experiment_detail_lift import picture, score

MODEL = 'stable-diffusion-v1-5/stable-diffusion-inpainting'


def cat_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    warm = (hsv[..., 0] < 28) & (hsv[..., 1] > 65) & (rgb[..., 0] > 95)
    labels, count = ndimage.label(warm)
    if count == 0: raise ValueError('No orange foreground found')
    counts = np.bincount(labels.ravel()); counts[0] = 0
    seed = labels == counts.argmax()
    possible = ndimage.binary_dilation(seed, iterations=9)
    marks = np.full(rgb.shape[:2], cv2.GC_BGD, np.uint8)
    marks[possible] = cv2.GC_PR_BGD
    marks[seed] = cv2.GC_PR_FGD
    marks[ndimage.binary_erosion(seed, iterations=2)] = cv2.GC_FGD
    cv2.setRNGSeed(531004)
    cv2.grabCut(rgb, marks, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    mask = (marks == cv2.GC_FGD) | (marks == cv2.GC_PR_FGD)
    return ndimage.binary_fill_holes(mask)


def preserve_known(original, generated, mask):
    if original.shape != generated.shape or mask.shape != original.shape[:2]:
        raise ValueError('Shape mismatch')
    return np.where(mask[..., None], generated, original)


def behind_depth(completed, foreground, mask, margin=.15):
    if completed.shape != foreground.shape or mask.shape != completed.shape or margin <= 0:
        raise ValueError('Invalid depth constraint')
    if not np.isfinite(completed).all() or not np.isfinite(foreground).all():
        raise ValueError('Nonfinite depth')
    return np.where(mask, np.maximum(completed, foreground+margin), completed)


def main(args):
    if args.out.exists(): raise FileExistsError('Preserve previous completion run')
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    start = time.perf_counter()
    report = dict(scope='Single static view, inferred hidden background stored as 3D Gaussians',
        accepted=False, model=MODEL, seed=args.seed, no_new_animal_motion=True,
        mask_method='Orange-component seeded GrabCut, not general learned segmentation',
        source_sha256=digest(args.source), asset_sha256=digest(args.asset))
    def log(stage, **data):
        print(json.dumps(dict(stage=stage, **data)), flush=True)
        (args.out/'progress.json').write_text(json.dumps(dict(stage=stage, **data)))
        (args.out/'metrics.json').write_text(json.dumps(report, indent=2))
    rgb = np.asarray(Image.open(args.source).convert('RGB'))
    h, w = rgb.shape[:2]
    foreground = cat_mask(rgb)
    mask = ndimage.binary_dilation(foreground, iterations=20 if args.texture_init else 12)
    generation_mask = mask.copy()
    if args.rectangular_context:
        my, mx = np.where(mask)
        generation_mask[:] = False
        generation_mask[my.min():my.max()+1, mx.min():mx.max()+1] = True
    Image.fromarray(np.uint8(foreground)*255).save(args.out/'foreground_mask.png')
    Image.fromarray(np.uint8(mask)*255).save(args.out/'inpaint_mask.png')
    Image.fromarray(np.uint8(generation_mask)*255).save(args.out/'generation_mask.png')
    report['rectangular_generation_context'] = args.rectangular_context
    report['texture_initialized_diffusion'] = args.texture_init
    with keep_windows_awake():
        from huggingface_hub import HfApi, snapshot_download
        from diffusers import StableDiffusionInpaintPipeline
        revision = HfApi().model_info(MODEL).sha
        report['revision'] = revision
        local = Path('../../work/background_inpainting')/revision
        log('download_inpainting_model', revision=revision)
        snapshot_download(MODEL, revision=revision, local_dir=local,
            allow_patterns=['model_index.json', 'scheduler/*.json', 'tokenizer/*', 'unet/config.json',
                'unet/diffusion_pytorch_model.fp16.safetensors', 'vae/config.json',
                'vae/diffusion_pytorch_model.fp16.safetensors', 'text_encoder/config.json',
                'text_encoder/model.fp16.safetensors'], max_workers=2)
        log('load_inpainting_model')
        pipe = StableDiffusionInpaintPipeline.from_pretrained(local, torch_dtype=torch.float16,
            variant='fp16', use_safetensors=True, local_files_only=True,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False).to('cuda')
        prompt = 'Photograph of uninterrupted green grass lawn, close-up grass texture across the lower area, horizontal dark leafy green hedge across the very top, flat grass, daylight, empty landscape, matching the surrounding photograph.'
        log('generate_hidden_background')
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            scale = 2 if (args.rectangular_context or args.texture_init) else 1
            initialization = cv2.inpaint(rgb, np.uint8(mask)*255, 5, cv2.INPAINT_TELEA) if args.texture_init else rgb
            Image.fromarray(initialization).save(args.out/'diffusion_initialization.png')
            generated = pipe(prompt=prompt, negative_prompt='path, pavement, walkway, curved border, cat, dog, cow, deer, horse, animal, creature, cartoon, illustration, statue, orange fur, legs, tail, person, text, watermark',
                image=Image.fromarray(initialization).resize((w*scale, h*scale)),
                mask_image=Image.fromarray(np.uint8(generation_mask)*255).resize((w*scale, h*scale), Image.Resampling.NEAREST),
                height=h*scale, width=w*scale, num_inference_steps=35, guidance_scale=5.0 if args.texture_init else 7.5,
                strength=.65 if args.texture_init else 1.,
                generator=torch.Generator(device='cuda').manual_seed(args.seed)).images[0]
            generated = generated.resize((w, h), Image.Resampling.LANCZOS)
        generated.save(args.out/'raw_completion.png')
        complete = preserve_known(rgb, np.asarray(generated), mask)
        Image.fromarray(complete).save(args.out/'completed_background.png')
        report['outside_mask_max_pixel_change'] = int(np.abs(complete.astype(int)-rgb.astype(int))[~mask].max())
        report['inpainting_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['model_files'] = {str(p.relative_to(local)):digest(p) for p in local.rglob('*.safetensors')}
        del pipe; gc.collect(); torch.cuda.empty_cache()
        log('estimate_background_depth')
        from transformers import AutoModelForDepthEstimation
        from transformers.models.dpt.image_processing_pil_dpt import DPTImageProcessorPil
        depth_dir = Path('../../work/detail_lift_depth/5426e4f0f36572d16453bbda7a8389317b1bef99')
        processor = DPTImageProcessorPil.from_pretrained(depth_dir, local_files_only=True)
        depth_model = AutoModelForDepthEstimation.from_pretrained(depth_dir, local_files_only=True).cuda().eval()
        with torch.no_grad():
            predictions = []
            for image in [rgb, complete]:
                pred = depth_model(**processor(images=Image.fromarray(image), return_tensors='pt').to('cuda')).predicted_depth
                predictions.append(F.interpolate(pred[:, None], size=(h, w), mode='bicubic', align_corners=False)[0, 0].cpu().numpy())
        del depth_model; gc.collect(); torch.cuda.empty_cache()
        old_inverse, new_inverse = predictions
        lo, hi = np.quantile(old_inverse, [.02, .98])
        original_inverse = .25+.75*np.clip((old_inverse-lo)/(hi-lo), 0, 1)
        # Align new relative inverse depth to the old scale on known background.
        known = ~mask
        design = np.stack([new_inverse[known], np.ones(known.sum())], -1)
        coefficients = np.linalg.lstsq(design, original_inverse[known], rcond=None)[0]
        aligned = np.clip(new_inverse*coefficients[0]+coefficients[1], .15, 1.2)
        bg_depth = behind_depth(1/aligned, 1/original_inverse, mask)
        report['inverse_depth_alignment'] = coefficients.tolist()
        report['minimum_background_depth_margin'] = float((bg_depth-1/original_inverse)[mask].min())
        log('lift_hidden_background')
        asset = load_verified(args.asset)
        position, covariance, colour = [asset[k].cuda() for k in ['position', 'covariance', 'colour']]
        opacity = torch.full((len(position),), asset['opacity'], device='cuda')
        # Extra Gaussians only in the completed region; never duplicate cat colour.
        yy, xx = np.mgrid[0:h:2, 0:w:2]
        chosen = mask[yy, xx]; y, x = yy[chosen], xx[chosen]
        uv = torch.tensor(np.stack([x+.5, y+.5], -1), dtype=torch.float32, device='cuda')
        depths = torch.tensor(bg_depth[y, x], dtype=torch.float32, device='cuda')
        cov2 = torch.eye(2, device='cuda')[None].repeat(len(uv), 1, 1)*1.6
        bg_xyz, bg_cov = lift(uv, cov2, depths, h/2/math.tan(math.radians(42)/2), (w/2, h/2))
        bg_colour = torch.tensor(complete[y, x]/255., dtype=torch.float32, device='cuda')
        bg_opacity = torch.full((len(uv),), .995, device='cuda')
        combined = dict(position=torch.cat([position, bg_xyz]), covariance=torch.cat([covariance, bg_cov]),
            colour=torch.cat([colour, bg_colour]), opacity=torch.cat([opacity, bg_opacity]))
        save_inference_checkpoint(dict(**{k:v.cpu() for k,v in combined.items()},
            original_count=len(position), background_count=len(bg_xyz),
            scope='Static scene completion; original Gaussians unchanged; new hidden background inferred'), args.out/'completed_scene.pt')
        report.update(original_gaussians=len(position), added_background_gaussians=len(bg_xyz),
            original_geometry_and_colour_unchanged=True)
        log('render_frozen_scene_inspection')
        pivot = position.new_tensor([0., 0., -float((-position[:, 2]).median())])
        radius = float(-pivot[2]); video = []; coverage = []
        with torch.no_grad():
            for index in range(112):
                angle = math.radians(18)*math.sin(2*math.pi*index/112)
                eye = pivot+position.new_tensor([radius*math.sin(angle), 0., radius*math.cos(angle)])
                before, ab = render(position, covariance, colour, opacity, eye, pivot, h, w, radius=4, ground=False)
                after, aa = render(**combined, eye=eye, target=pivot, height=h, width=w, radius=4, ground=False)
                canvas = Image.new('RGB', (w*2, h+32), '#171b22')
                canvas.paste(picture(before), (0, 32)); canvas.paste(picture(after), (w, 32))
                ImageDraw.Draw(canvas).text((8, 8), 'Before | Stored AI-completed background; camera motion only, no per-frame inpainting', fill='white')
                video.append(np.asarray(canvas))
                if index in [0, 28, 56, 84, 111]:
                    canvas.save(args.out/f'inspection_{index:03d}.png')
                    hole = ab < .1
                    coverage.append(dict(frame=index, before_hole_pixels=int(hole.sum()),
                        remaining_hole_pixels=int((aa < .1).sum()),
                        formerly_empty_pixels_now_covered=int((hole & (aa > .5)).sum())))
                if index == 0: report['front_view_drift_from_original_asset'] = score(after, before)
        imageio.mimwrite(args.out/'background_completion_7seconds.mp4', video, fps=16, codec='libx264', quality=8, macro_block_size=1)
        report.update(coverage=coverage, total_seconds=time.perf_counter()-start,
            video_sha256=digest(args.out/'background_completion_7seconds.mp4'),
            limitations=['Unobserved background is invented, not recovered ground truth.',
                'No scene-boundary outpainting or hidden cat anatomy completion.',
                'Single reference-view ordering does not guarantee correct ordering from all viewpoints.',
                'No speedup claim; completion cost is additional and one-time.'])
        log('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/real_video/detail_lift/v4_tangent/source_000.png'))
    parser.add_argument('--asset', type=Path, default=Path('artifacts/real_video/detail_lift/v4_tangent/frame_000_opacity_0.95.pt'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/real_video/background_completion/v1'))
    parser.add_argument('--seed', type=int, default=531004)
    parser.add_argument('--rectangular-context', action='store_true')
    parser.add_argument('--texture-init', action='store_true')
    main(parser.parse_args())
