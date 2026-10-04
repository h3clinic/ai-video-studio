"""Persist an inferred 3D dog inside the same Gaussian cat/garden environment.

Not a cropped video overlay. Not Wan emitting Gaussians. Object creation,
canonical validation, state insertion, scene assembly and rendering have
separate costs. The diagnostic orbit moves ONLY the camera, not the animals.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from real_video.checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint
from real_video.gaussian_scene_objects import (
    GaussianSceneObject, merge_scene_objects, object_bounds, overlap_report, placement_report,
)
from real_video.gaussian3d import render, project

BASE = Path('artifacts/real_video')
CAT = BASE/'hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt'
ENV = BASE/'enclosing_scene/v1/scene.pt'


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    return 0


def serial_transform(obj):
    return {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in obj['transform'].items()}


def compose(out):
    if (out/'scene.pt').exists():
        raise FileExistsError('Preserve previous scene')
    begin = time.perf_counter()
    dog_path = out/'dog/cat_asset.pt'  # Legacy generic TripoSR export filename.
    cat, dog, environment = [load_verified(p) for p in (CAT, dog_path, ENV)]
    hashes = {name: digest(p) for name, p in [('cat', CAT), ('dog', dog_path), ('environment', ENV)]}
    original = environment['original_count']
    environment = {k: environment[k][original:].clone() for k in ('position', 'covariance', 'colour', 'opacity')}
    environment['ids'] = torch.arange(len(environment['position']), dtype=torch.int64)
    load_seconds = time.perf_counter()-begin
    start = time.perf_counter()
    cat_object = GaussianSceneObject(0, cat, hashes['cat'])
    dog_object = GaussianSceneObject(1, dog, hashes['dog'])
    environment_object = GaussianSceneObject(2, environment, hashes['environment'])
    validation_seconds = time.perf_counter()-start
    # Chosen scene-unit priors, explicitly not recovered physical dimensions.
    cat_height, dog_height, gap = .75, 1.1, .25
    cat0, dog0 = cat_object.grounded(cat_height), dog_object.grounded(dog_height)
    clo, chi = object_bounds(cat0)
    dlo, dhi = object_bounds(dog0)
    cat_x, dog_x = -.5*(float(dhi[0]-dlo[0])+gap), .5*(float(chi[0]-clo[0])+gap)
    start = time.perf_counter()
    cat_world = cat_object.grounded(cat_height, center_xz=(cat_x, 0.))
    dog_world = dog_object.grounded(dog_height, center_xz=(dog_x, 0.))
    placement_seconds = time.perf_counter()-start
    checks = {name: placement_report(world, (-4., 0., -4.), (4., 2., 4.), height_range=(h-.001, h+.001))
        for name, world, h in [('cat', cat_world, cat_height), ('dog', dog_world, dog_height)]}
    overlap = overlap_report(cat_world, dog_world, sigma=3.)
    if not all(x['accepted'] for x in checks.values()) or overlap['interior_aabb_overlap']:
        raise ValueError('Object bounds/ground/size constraint failed')
    start = time.perf_counter()
    scene = merge_scene_objects([cat_world, dog_world, environment_object.transform()])
    assembly_seconds = time.perf_counter()-start
    for oid, world in [(0, cat_world), (1, dog_world)]:
        rows = scene['object_ids'] == oid
        for key in ('position', 'covariance', 'colour', 'opacity'):
            if not torch.equal(scene[key][rows], world[key]):
                raise AssertionError(f'Scene merge altered object {oid} {key}')
        if not torch.equal(scene['local_ids'][rows], world['ids']):
            raise AssertionError('Scene merge altered persistent IDs')
    if not torch.equal(cat_world['colour'], cat['colour']) or not torch.equal(dog_world['colour'], dog['colour']):
        raise AssertionError('Insertion must not repaint canonical assets')
    object_count = len(cat['position'])+len(dog['position'])
    scene.update(original_count=object_count, camera_pivot=[0., .55, 0.], camera_distance=4.5,
        scope='Persistent inferred 3D cat and dog plus existing generated depth-shell Gaussian garden. STATIC scene; no image plane or RGB/video overlay; hidden anatomy/texture unverified.')
    start = time.perf_counter()
    checksum = save_inference_checkpoint(scene, out/'scene.pt')
    save_seconds = time.perf_counter()-start
    timings = []
    for _ in range(7):
        start = time.perf_counter()
        moved = dog_object.grounded(dog_height, center_xz=(dog_x, .05))
        timings.append(time.perf_counter()-start)
        del moved
    from real_video.export_orbit_viewer import main as export
    export_start = time.perf_counter()
    export(out/'scene.pt', 'dog_scene')
    export_seconds = time.perf_counter()-export_start
    inference = json.loads((out/'dog/asset_report.json').read_text())
    report = dict(scope=scene['scope'], scene_sha256=checksum, source_hashes=hashes,
        cat_gaussians=len(cat['position']), dog_gaussians=len(dog['position']),
        environment_gaussians=len(environment['position']), total_gaussians=len(scene['position']),
        geometry_and_colour_preserved_exactly_on_merge=True, canonical_colours_unchanged=True,
        identity='Stable pair (object_id, local_gaussian_id), not a newly assigned raster row',
        cat_transform=serial_transform(cat_world), dog_transform=serial_transform(dog_world),
        placement_checks=checks, three_sigma_overlap=overlap,
        size_convention='Chosen full center-height .75 cat / 1.10 dog in common arbitrary units; NOT shoulder heights or recovered metres',
        gaussian_scene_tensor_bytes=tensor_bytes(scene), scene_checkpoint_bytes=(out/'scene.pt').stat().st_size,
        timing=dict(existing_asset_load_verify_seconds=load_seconds, canonical_validation_seconds=validation_seconds,
            two_object_placement_seconds=placement_seconds, full_scene_merge_validate_seconds=assembly_seconds,
            scene_save_seconds=save_seconds, viewer_export_seconds=export_seconds,
            cached_dog_reposition_median_seconds=float(np.median(timings)),
            dog_creation_load_infer_extract_seconds=inference['load_infer_extract_seconds'],
            dog_neural_inference_seconds=inference['inference_seconds']),
        dog_creation_peak_cuda_bytes=inference['peak_cuda_bytes'],
        neural_calls_during_insertion_or_reposition=0, image_planes_or_output_compositing=False,
        generated_motion=False, wan_weights_modified_by_this_insertion=False,
        quality_accepted=False,
        limitations=['Single-image inferred dog; visible backside texture and paw defects, not ground truth',
            'Grounding lowest Gaussian centers is not a solved anatomical stance/contact model',
            'Garden is an earlier generated radial depth shell with seams, not complete free-roaming scene geometry',
            'Cheap asset placement/re-rendering is not a quality-matched replacement for full generative video inference',
            'Tensor count excludes temporary copies, renderer, model weights, source assets and browser allocation',
            'Camera-only inspection must not be described as generated animal motion'])
    (out/'insertion_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['timing'], indent=2), flush=True)


@torch.inference_mode()
def inspect(out):
    if (out/'camera_only_360.mp4').exists():
        raise FileExistsError('Preserve prior inspection')
    scene = load_verified(out/'scene.pt')
    a = {k: scene[k].cuda() for k in ('position', 'covariance', 'colour', 'opacity')}
    target = a['position'].new_tensor(scene['camera_pivot'])
    distance = scene['camera_distance']
    height, width = 384, 640
    sheet = Image.new('RGB', (width*3, (height+22)*2))
    diagnostics, elapsed = [], []
    torch.cuda.reset_peak_memory_stats()
    for index, (name, direction) in enumerate([('front', (0., .12, 1.)), ('oblique', (.7, .25, 1.)),
        ('back', (0., .12, -1.)), ('right', (1., .12, 0.)), ('left', (-1., .12, 0.)), ('above', (0., 1., .1))]):
        vec = target.new_tensor(direction); vec /= vec.norm()
        eye = target+distance*vec
        start = time.perf_counter()
        rgb, alpha = render(**a, eye=eye, target=target, height=height, width=width, ground=False, radius=3)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
        im = Image.fromarray((rgb.clamp(0, 1)*255).round().byte().cpu().numpy())
        im.save(out/f'scene_{name}.png')
        x, y = (index%3)*width, (index//3)*(height+22)
        sheet.paste(im, (x, y+22)); ImageDraw.Draw(sheet).text((x+6, y+4), name, fill='white')
        mean, _, depth, _, _ = project(a['position'][:scene['original_count']], a['covariance'][:scene['original_count']], eye, target, height, width)
        framed = (depth>.02)&(mean[:, 0]>=0)&(mean[:, 0]<width)&(mean[:, 1]>=0)&(mean[:, 1]<height)
        diagnostics.append(dict(view=name, object_center_frustum_fraction=float(framed.float().mean()),
            warning='Frustum inclusion does not imply visibility: objects can occlude each other'))
    sheet.save(out/'scene_six_views.jpg')
    start = time.perf_counter()
    with imageio.get_writer(out/'camera_only_360.mp4', fps=8, codec='libx264', quality=8, macro_block_size=1) as writer:
        for frame in range(57):
            angle = 2*math.pi*frame/56
            eye = target+target.new_tensor([math.sin(angle)*distance, .35, math.cos(angle)*distance])
            rgb, _ = render(**a, eye=eye, target=target, height=height, width=width, ground=False, radius=3)
            pixels = (rgb.clamp(0, 1)*255).round().byte().cpu().numpy()
            writer.append_data(pixels)
            if frame%14 == 0:
                print(f'Camera-only scene orbit {frame+1}/57', flush=True)
    torch.cuda.synchronize()
    result = dict(scope='Static assets, camera orbit only; NOT generated animal movement',
        view_diagnostics=diagnostics, static_view_render_median_seconds=float(np.median(elapsed)),
        orbit_render_encode_seconds=time.perf_counter()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        frames=57, fps=8, duration_seconds=57/8, width=width, height=height)
    (out/'inspection_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE/'gaussian_dog_insertion/v1')
    parser.add_argument('--phase', choices=('compose', 'inspect', 'all'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with keep_windows_awake():
        if args.phase in ('compose', 'all'):
            compose(args.out)
            gc.collect()
        if args.phase in ('inspect', 'all'):
            inspect(args.out)


if __name__ == '__main__':
    main()
