"""Bounded source-view fitting experiment; NOT a 3D video generator.

No Wan weights change. Per-frame estimated depth is not material tracking.
The orbit is explicitly a camera-only inspection of frame zero.
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
import torch.nn.functional as F

from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from .detail_lift import lift
from .gaussian3d import render, project
from .wan_gaussian import gaussian_state, splat_frame

MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'


def picture(tensor):
    return Image.fromarray((tensor.detach().clamp(0, 1).cpu().numpy()*255).round().astype('uint8'))


def score(image, target):
    mse = float((image-target).square().mean())
    edge = float(((image[:, 1:]-image[:, :-1])-(target[:, 1:]-target[:, :-1])).abs().mean())
    return dict(psnr_db=-10*math.log10(max(mse, 1e-12)), mse=mse, horizontal_edge_mae=edge)


def main(args):
    if args.out.exists(): raise FileExistsError('Preserve previous experiment')
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    report = dict(scope='2D Gaussian to visible-surface 3D lift and per-image appearance fitting',
        accepted=False, source_sha256=digest(args.source), depth_model=MODEL,
        resolution=[240, 416], frame_indices=[0, 16, 32],
        no_new_video_generation=True, no_persistent_motion_identity=True,
        depth_units='Arbitrary relative inverse-depth remapping, not calibrated metric geometry',
        fitting_split='Checkerboard image pixels: fit on 80%, select on 20%. Same-view diagnostic, NOT independent scene generalization.',
        limitations=['Unseen surfaces absent', 'Source decoder blur remains', 'No matched-quality efficiency claim'],
        candidates=[], surface_tangent=args.surface_tangent)

    def log(stage, **kwargs):
        print(json.dumps(dict(stage=stage, **kwargs)), flush=True)
        (args.out/'progress.json').write_text(json.dumps(dict(stage=stage, **kwargs)))

    with keep_windows_awake():
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoModelForDepthEstimation
        from transformers.models.dpt.image_processing_pil_dpt import DPTImageProcessorPil
        revision = HfApi().model_info(MODEL).sha
        report['depth_revision'] = revision
        log('download_depth_prior', revision=revision)
        model_dir = Path('../../work/detail_lift_depth')/revision
        snapshot_download(MODEL, revision=revision, local_dir=model_dir,
                          allow_patterns=['config.json', 'preprocessor_config.json', '*.safetensors'])
        processor = DPTImageProcessorPil.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True,
                                                            trust_remote_code=False).cuda().eval()
        report['depth_files'] = {p.name:digest(p) for p in model_dir.glob('*') if p.is_file()}
        fields = load_verified(args.source)['fields']
        h, w = 240, 416
        eye = torch.zeros(3, device='cuda'); target_camera = eye.new_tensor([0., 0., -1.])
        focal = h/2/math.tan(math.radians(42)/2)
        assets = []
        torch.cuda.reset_peak_memory_stats()
        for frame_index in report['frame_indices']:
            log('estimate_depth', frame=frame_index)
            raw = fields[:, :, frame_index].cuda()
            with torch.no_grad():
                ref = splat_frame(raw, h, w)[0][0].permute(1, 2, 0)
                state = gaussian_state(raw, h, w)
                inputs = processor(images=picture(ref), return_tensors='pt').to('cuda')
                prediction = model(**inputs).predicted_depth
                inv = F.interpolate(prediction[:, None], size=(h, w), mode='bicubic', align_corners=False)[0, 0]
                lo, hi = torch.quantile(inv, inv.new_tensor([.02, .98]))
                if float(hi-lo) < 1e-6: raise ValueError('Depth prior collapsed')
                relative = ((inv-lo)/(hi-lo)).clamp(0, 1)
                depth_map = 1/(.25+.75*relative)
                grid = state['centre']/state['centre'].new_tensor([w-1, h-1])*2-1
                depth = F.grid_sample(depth_map[None, None], grid[:, None], align_corners=True, padding_mode='border')[0, 0, 0]
                axes, scales = state['axes'][0], state['scale'][0]
                screen_cov = axes @ torch.diag_embed(scales.square()) @ axes.transpose(-1, -2)
                # Renderer uses pixel+.5; 2D splatter uses integer pixels.
                normals = None
                grazing_fraction = 0.
                if args.surface_tangent:
                    py, px = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
                    rays = torch.stack(((px+.5-w/2)/focal, (py+.5-h/2)/focal, torch.ones_like(px)), -1)
                    cloud = rays*depth_map[..., None]
                    dy, dx = torch.gradient(cloud, dim=(0, 1))
                    normal_map = F.normalize(torch.linalg.cross(dx, dy), dim=-1)
                    normals = F.grid_sample(normal_map.permute(2, 0, 1)[None], grid[:, None], align_corners=True,
                                            padding_mode='border')[0, :, 0].T
                    normals = F.normalize(normals, dim=-1)
                    sampled_rays = torch.cat(((state['centre'][0]+.5-state['centre'].new_tensor([w/2, h/2]))/focal,
                                              torch.ones_like(depth[:, None])), -1)
                    sampled_rays = F.normalize(sampled_rays, dim=-1)
                    grazing = ((normals*sampled_rays).sum(-1).abs() < .15)
                    grazing_fraction = float(grazing.float().mean())
                    normals = torch.where(grazing[:, None], sampled_rays, normals)
                xyz, covariance = lift(state['centre'][0]+.5, screen_cov, depth, focal, (w/2, h/2), normals=normals)
                means, projected, _, _, _ = project(xyz, covariance, eye, target_camera, h, w)
                asset = dict(position=xyz, covariance=covariance, colour=state['colour'][0], reference=ref,
                             depth=depth_map, frame=frame_index, grazing_fallback_fraction=grazing_fraction,
                             projection_centre_max_error=float((means-state['centre'][0]-.5).abs().max()),
                             projection_covariance_max_error=float((projected-screen_cov).abs().max()))
                assets.append(asset)
                picture(ref).save(args.out/f'source_{frame_index:03d}.png')
                picture(relative[..., None].expand(-1, -1, 3)).save(args.out/f'depth_{frame_index:03d}.png')
        del model, inputs, prediction
        gc.collect(); torch.cuda.empty_cache()
        yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
        validation = ((xx+3*yy)%5 == 0).flatten()
        winners = []
        for asset in assets:
            frame_index = asset['frame']; ref = asset['reference']
            candidates = []
            # Select opacity only on first diagnostic frame, freeze for later poses.
            opacities = [.35, .65, .95] if frame_index == 0 else [winners[0]['opacity']]
            for opacity in opacities:
                log('fit_appearance', frame=frame_index, opacity=opacity, iterations=args.steps)
                with torch.no_grad():
                    initial, coverage, cache = render(asset['position'], asset['covariance'], asset['colour'],
                        torch.full_like(asset['depth'].flatten()[:len(asset['position'])], opacity),
                        eye, target_camera, h, w, radius=4, ground=False, return_cache=True)
                    bg = (1-coverage.flatten()[:, None])*eye.new_tensor([.15, .19, .24])
                colours = asset['colour'].detach().clone().requires_grad_()
                optimizer = torch.optim.Adam([colours], lr=.025)
                best_error = float('inf'); best = colours.detach().clone(); trace = []
                for step in range(args.steps+1):
                    image = torch.zeros(h*w, 3, device='cuda').index_add(0, cache['pixel'],
                        cache['weight'][:, None]*colours[cache['ids']])+bg
                    residual = (image-ref.reshape(-1, 3)).square().mean(-1)
                    ve = float(residual[validation].mean().detach())
                    if ve < best_error: best_error = ve; best = colours.detach().clone()
                    if step % 20 == 0: trace.append(dict(step=step, validation_mse=ve))
                    if step == args.steps: break
                    loss = residual[~validation].mean()+1e-5*(colours-asset['colour']).square().mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    with torch.no_grad(): colours.clamp_(0, 1)
                with torch.no_grad():
                    fitted, _ = render(asset['position'], asset['covariance'], best,
                        torch.full((len(best),), opacity, device='cuda'), eye, target_camera, h, w, radius=4, ground=False)
                name = f'frame_{frame_index:03d}_opacity_{opacity:.2f}'
                picture(initial).save(args.out/f'{name}_initial.png')
                picture(fitted).save(args.out/f'{name}_fitted.png')
                record = dict(name=name, frame=frame_index, opacity=opacity,
                    before=score(initial, ref), after=score(fitted, ref), validation_mse=best_error,
                    colour_mean_absolute_change=float((best-asset['colour']).abs().mean()), trace=trace,
                    coverage_mean=float(coverage.mean()))
                report['candidates'].append(record)
                save_inference_checkpoint(dict(position=asset['position'].cpu(), covariance=asset['covariance'].cpu(),
                    colour=best.cpu(), opacity=opacity, source_frame=frame_index,
                    identity_scope='Within this single static inferred view only'), args.out/f'{name}.pt')
                candidates.append(dict(opacity=opacity, colours=best, image=fitted, record=record))
                (args.out/'metrics.json').write_text(json.dumps(report, indent=2))
                del cache, optimizer, image, residual, loss, colours
            winner = min(candidates, key=lambda c:c['record']['validation_mse'])
            winners.append(winner)
        log('render_camera_only_inspection')
        # Static object, moving camera: deliberately NOT called generated motion.
        asset, winner = assets[0], winners[0]
        pivot = eye.new_tensor([0., 0., -float(asset['depth'].median())])
        frames = []
        for index in range(112):
            angle = math.radians(18)*math.sin(2*math.pi*index/112)
            radius = float(-pivot[2])
            camera_eye = pivot+eye.new_tensor([radius*math.sin(angle), 0., radius*math.cos(angle)])
            with torch.no_grad():
                oblique, _ = render(asset['position'], asset['covariance'], winner['colours'],
                    torch.full((len(winner['colours']),), winner['opacity'], device='cuda'),
                    camera_eye, pivot, h, w, radius=4, ground=False)
            canvas = Image.new('RGB', (w*2, h+32), '#171b22')
            canvas.paste(picture(asset['reference']), (0, 32)); canvas.paste(picture(oblique), (w, 32))
            ImageDraw.Draw(canvas).text((8, 8), f'2D reference | 3D camera-only inspection ({math.degrees(angle):+.1f} deg); NOT animal motion', fill='white')
            frames.append(np.asarray(canvas))
            if index in [0, 28, 56, 84, 111]: canvas.save(args.out/f'inspection_{index:03d}.png')
        imageio.mimwrite(args.out/'camera_only_7seconds.mp4', frames, fps=16, codec='libx264', quality=8, macro_block_size=1)
        sheet = Image.new('RGB', (w*3, (h+24)*3), '#171b22')
        for row, (asset, winner) in enumerate(zip(assets, winners)):
            with torch.no_grad():
                oblique, _ = render(asset['position'], asset['covariance'], winner['colours'],
                    torch.full((len(winner['colours']),), winner['opacity'], device='cuda'),
                    eye.new_tensor([.4, 0, 0]), eye.new_tensor([.4, 0, -1]), h, w, radius=4, ground=False)
            for col, value in enumerate([asset['reference'], winner['image'], oblique]):
                sheet.paste(picture(value), (col*w, row*(h+24)+24))
            ImageDraw.Draw(sheet).text((8, row*(h+24)+5), f'Source frame {asset["frame"]}: 2D | fitted 3D reference view | translated camera', fill='white')
        sheet.save(args.out/'three_pose_comparison.jpg')
        report.update(total_seconds=time.perf_counter()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            projection_checks=[{k:a[k] for k in ['frame', 'grazing_fallback_fraction', 'projection_centre_max_error', 'projection_covariance_max_error']} for a in assets],
            selected=[w['record']['name'] for w in winners], visual_review='pending',
            video_sha256=digest(args.out/'camera_only_7seconds.mp4'))
        (args.out/'metrics.json').write_text(json.dumps(report, indent=2))
        log('complete', seconds=report['total_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/real_video/prompt_generation/orange_cat_seed_531002/gaussian_fields.pt'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/real_video/detail_lift/v1'))
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--surface-tangent', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.steps <= 200: parser.error('Bounded fitting requires 1-200 steps')
    main(args)
