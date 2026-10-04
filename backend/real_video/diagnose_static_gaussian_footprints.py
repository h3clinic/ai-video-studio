"""Frozen-state raster diagnostic, not video or model-generation evaluation."""
import argparse
import gc
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw
import torch

from .adaptive_gaussian_renderer import render as adaptive_render
from .checkpoint_io import digest, keep_windows_awake, load_verified
from .gaussian3d import project, render as legacy_render


@torch.inference_mode()
def run(scene_path, output, wall_seconds=280, max_candidate_pixels=300_000_000, fragment_budget=500_000,
        slope_guard=1.3):
    if output.exists():
        raise FileExistsError('Use a fresh output directory; diagnostic evidence is immutable')
    output.mkdir(parents=True)
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.47)
    scene_hash = digest(scene_path)
    scene = load_verified(scene_path)
    keys = ('position', 'covariance', 'colour', 'opacity')
    tensors = {k: scene[k].cuda() for k in keys}
    target = tensors['position'].new_tensor(scene['camera_pivot'])
    direction = target.new_tensor([0., .12, 1.]); direction /= direction.norm()
    eye = target+scene['camera_distance']*direction
    common = dict(eye=eye, target=target, height=540, width=960, ground=False)
    mean, screen, depth, *_ = project(tensors['position'], tensors['covariance'], eye, target, 540, 960)
    radius = torch.sqrt(2*torch.log(tensors['opacity']/1e-4)[:, None]*screen.diagonal(dim1=-2, dim2=-1)).amax(1)
    visible = (depth > .02)&(mean[:, 0]>=0)&(mean[:, 0]<960)&(mean[:, 1]>=0)&(mean[:, 1]<540)
    groups = {}
    for oid, name in ((0, 'cat'), (1, 'dog'), (2, 'environment')):
        selected = visible & (scene['object_ids'].cuda() == oid)
        values = radius[selected]
        groups[name] = dict(visible_centers=int(selected.sum()),
            threshold_radius_quantiles_pixels=torch.quantile(values, values.new_tensor([.1, .5, .9, .99])).tolist(),
            fraction_support_exceeds_radius3=float((values > 3).float().mean()),
            fraction_support_exceeds_radius5=float((values > 5).float().mean()))
    del mean, screen, depth, radius, visible
    records, images = {}, []
    for name, radius in (('legacy_radius3', 3), ('legacy_radius5', 5), ('adaptive_threshold', None)):
        if time.perf_counter()-start > wall_seconds:
            records[name] = dict(status='not_run_wall_budget'); break
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        try:
            if radius is not None:
                rgb, alpha = legacy_render(**tensors, **common, radius=radius)
                stats = dict(fixed_radius=radius, implementation='unmodified gaussian3d.render')
            else:
                rgb, alpha, stats = adaptive_render(**tensors, **common, fragment_budget=fragment_budget,
                    max_candidate_pixels=max_candidate_pixels, return_stats=True,
                    slope_guard=slope_guard,
                    wall_seconds=max(.001, wall_seconds-(time.perf_counter()-start)))
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-before
            image = Image.fromarray((rgb.clamp(0, 1)*255).round().byte().cpu().numpy())
            path = output/f'{name}.png'; image.save(path)
            record = dict(status='rendered', seconds=elapsed, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                          png_sha256=digest(path), opaque_pixel_fraction=float((alpha>.95).float().mean()),
                          alpha_mean=float(alpha.mean()), renderer=stats)
            images.append((name, image))
            del rgb, alpha
        except (torch.cuda.OutOfMemoryError, ValueError, TimeoutError) as exc:
            record = dict(status='resource_or_work_limit', error=str(exc),
                          seconds=time.perf_counter()-before, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            gc.collect(); torch.cuda.empty_cache()
        records[name] = record
        print(json.dumps(dict(case=name, **record)), flush=True)
        (output/'partial_metrics.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    sheet = Image.new('RGB', (960, len(images)*568))
    for i, (name, image) in enumerate(images):
        sheet.paste(image, (0, i*568+28))
        ImageDraw.Draw(sheet).text((10, i*568+8), f'{name}: frozen geometry/texture, 960 x 540', fill='white')
    sheet.save(output/'comparison.png')
    if digest(scene_path) != scene_hash:
        raise AssertionError('Source checkpoint changed during diagnostic')
    report = dict(status='completed_diagnostic', scope='Static frozen scene rendering only; no motion or generation test',
        scene_path=str(scene_path), scene_sha256=scene_hash, source_checkpoint_unchanged=True,
        camera=dict(eye=eye.tolist(), target=target.tolist(), height=540, width=960, fov=42.),
        source_counts={name: int((scene['object_ids']==oid).sum()) for oid, name in ((0,'cat'),(1,'dog'),(2,'environment'))},
        footprint_statistics=groups, comparisons=records, total_seconds=time.perf_counter()-start,
        comparison_sha256=digest(output/'comparison.png'), quality_accepted=False, visual_review='pending',
        interpretation=['A larger complete projected footprint is correct coverage, not new fine detail.',
            'Environment consists of six 128x128 Gaussian grids sampled from a generated panorama and inferred single-layer radial depth.',
            'Changing raster support cannot repair the dog mesh, infer missing fur, solve articulation or create an autonomous video generator.',
            'Opaque-pixel fraction is a coverage diagnostic, not perceptual quality or reference accuracy.',
            'Peak allocation includes canonical CUDA scene and temporaries but excludes CPU RAM, browser, historical model inference and training.'],
        mathematics=dict(primary_source='https://arxiv.org/html/2311.16493v1',
            source_sha256='7c4b5c70f3f2c21416e7f1f449d6f3fef1d2d335fa2e4f968cc81acaf0a061c2',
            inspected='Sections 3.2, 5.1, 5.2, equations 3-5, 9-10',
            implemented='Threshold q=2 log(opacity/epsilon); axis support sqrt(q*Sigma_aa); exact ordered alpha chunk composition.',
            covariance_slope_guard=slope_guard,
            reference_implementation='https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L65',
            not_implemented='No 3D frequency regularization or determinant-normalized 2D Mip filter; unchanged +0.16 screen dilation for matched diagnostic.'))
    (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), total_seconds=report['total_seconds'], quality_accepted=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=Path('artifacts/real_video/gaussian_dog_insertion/v1/scene.pt'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/real_video/static_footprint_diagnostic/v1'))
    parser.add_argument('--wall-seconds', type=float, default=280)
    parser.add_argument('--max-candidate-pixels', type=int, default=300_000_000)
    parser.add_argument('--fragment-budget', type=int, default=500_000)
    parser.add_argument('--slope-guard', type=float, default=1.3)
    args = parser.parse_args()
    with keep_windows_awake():
        run(args.scene, args.out, args.wall_seconds, args.max_candidate_pixels, args.fragment_budget, args.slope_guard)


if __name__ == '__main__':
    main()
