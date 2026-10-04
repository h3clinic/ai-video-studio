"""Independent future-image evaluation of an ALREADY SAVED causal forecast.

This module never optimizes weights, camera, geometry, colours or forecast state.
Future source frames/masks are evaluation-only and cannot feed the predictor.
"""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .attach_vector_weights import original_intrinsics
from .checkpoint_io import load_verified, digest, keep_windows_awake
from .gaussian3d import render
from .gaussian_motion_memory import select_cat
from .vector_recording import GaussianSurfaceMemory

ROOT = Path('artifacts/real_video/true3d/attached_vector_weights/v2')
SOURCE = Path('artifacts/real_video/wan_baseline/cat_seed_421001/wan_original.mp4')


def verify_video(path, expected_frames, expected_fps):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    while True:
        valid, frame = capture.read()
        if not valid:
            break
        if frame.shape[:2] != (height, width):
            raise ValueError('Inconsistent decoded dimensions')
        decoded += 1
    capture.release()
    if decoded != expected_frames or abs(fps - expected_fps) > 1e-4:
        raise ValueError(f'Export verification failed: {path}, {decoded} frames, {fps} fps')
    return dict(decoded_frames=decoded, fps=fps, duration_seconds=decoded / fps, width=width, height=height, sha256=digest(path))


def write_video(path, frames, fps):
    writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return verify_video(path, len(frames), fps)


def read_evaluation_frames(indices):
    """First future read occurs here, after caller loaded immutable forecast."""
    capture = cv2.VideoCapture(str(SOURCE))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    wanted = set(indices)
    frames = {}
    frame_id = 0
    while frame_id <= max(indices):
        valid, bgr = capture.read()
        if not valid:
            break
        if frame_id in wanted:
            frames[frame_id] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame_id += 1
    capture.release()
    if set(frames) != wanted:
        raise ValueError('Source lacks requested diagnostic frames')
    return [frames[i] for i in indices], fps


def overlay_row(source, learned, disabled, source_frame, slow=False):
    row = Image.new('RGB', (2496, 572), '#151b24')
    draw = ImageDraw.Draw(row)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    labels = ['Observed source (evaluation only)', 'Learned conditional motion forecast', 'All-zero weights: 0.75-damped velocity']
    for column, (picture, label) in enumerate(zip([source, learned, disabled], labels)):
        row.paste(Image.fromarray(picture), (column * 832, 40))
        draw.text((column * 832 + 12, 10), label, fill='white', font=font)
    draw.text((12, 527), f'Source frame {source_frame:02d} | 45,000 persistent XYZ Gaussians | Learned 3D depth NOT trained | Quality gate FAILED', fill='#ffda8a', font=font)
    draw.text((12, 550), '7-second HELD inspection of 13 states, NOT 7 seconds of newly generated motion' if slow else '13 states at nominal 16 fps; source frame2 is the shared observed rest state; frames3-14 are forecasts', fill='white', font=font)
    return np.asarray(row)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    out = root / 'evaluation'
    if out.exists():
        raise FileExistsError('Preserve independent evaluation; use a new forecast root')
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    # Load/save identity before any source image reads; no training import/call.
    packet = load_verified(root / 'forecast.pt')
    forecast_sha = digest(root / 'forecast.pt')
    asset_path = Path(packet['asset_path'])
    if digest(asset_path) != packet['asset_sha256']:
        raise ValueError('Forecast appearance asset digest changed')
    cpu_asset = load_verified(asset_path)
    indices = packet['source_frame'].tolist()
    if indices != list(range(2, 15)):
        raise ValueError('This fixed diagnostic expects source frames2 through14')
    source, source_fps = read_evaluation_frames(indices)
    if source_fps != 16 or any(frame.shape != (480, 832, 3) for frame in source):
        raise ValueError('Expected pinned 832x480,16fps diagnostic')
    out.mkdir(parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    asset = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in cpu_asset.items()}
    camera = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in packet['camera'].items()}
    intrinsics = original_intrinsics(packet['camera'])
    memory = GaussianSurfaceMemory(asset)
    masks = [select_cat(frame) for frame in source]
    neutral_targets = [frame.astype(np.float32) / 255 * mask[..., None] + .5 * (1 - mask[..., None]) for frame, mask in zip(source, masks)]
    rendered = {}
    scores = {}
    physical = {}
    start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for mode in ['learned', 'zero_weights']:
        images = []
        rows = []
        geometry = []
        for state_index, vertices in enumerate(packet['vertices'][mode]):
            decoded = memory.decode(vertices.to(device))
            rgb, alpha = render(decoded['position'], decoded['covariance'], asset['colour'], asset['opacity'], camera['eye'], camera['target'],
                                **intrinsics, ground=False, radius=4)
            # Replace only renderer's constant background with neutral grey.
            # No observed/future pixels enter a generated panel.
            background = rgb.new_tensor([.15, .19, .24])
            neutral = (rgb - (1 - alpha[..., None]) * background + (1 - alpha[..., None]) * .5).clamp(0, 1)
            prediction = neutral.cpu().numpy()
            predicted_mask = alpha.cpu().numpy() > .5
            target_mask = masks[state_index] > .5
            # Same predeclared rectangle for all modes/states. Mask heuristic is
            # evaluation only; no per-model intersection masking can hide errors.
            error = prediction[80:480, :670] - neutral_targets[state_index][80:480, :670]
            mse = float(np.mean(error ** 2))
            union = np.logical_or(predicted_mask, target_mask).sum()
            iou = float(np.logical_and(predicted_mask, target_mask).sum() / max(union, 1))
            rows.append(dict(source_frame=indices[state_index], roi_mse=mse, roi_psnr=-10 * math.log10(max(mse, 1e-12)),
                             roi_mae=float(np.abs(error).mean()), approximate_mask_iou=iou))
            area = decoded['area_ratio']
            geometry.append(dict(source_frame=indices[state_index], minimum_gaussian_center_y=float(decoded['position'][:, 1].min()),
                                 below_rest_ground_fraction=float((decoded['position'][:, 1] < -1e-4).float().mean()),
                                 owned_face_area_min=float(area.min()), owned_face_area_p01=float(torch.quantile(area, .01)),
                                 owned_face_area_p99=float(torch.quantile(area, .99)), owned_face_area_max=float(area.max())))
            image = np.uint8(prediction * 255 + .5)
            images.append(image)
            if state_index in [0, 4, 8, 12]:
                Image.fromarray(image).save(out / f'{mode}_source{indices[state_index]:02d}.png')
            print(json.dumps(dict(mode=mode, state=state_index, source_frame=indices[state_index], roi_psnr=rows[-1]['roi_psnr'], iou=iou)), flush=True)
        rendered[mode] = images
        scores[mode] = dict(per_frame=rows, future_mean_roi_psnr=float(np.mean([row['roi_psnr'] for row in rows[1:]])),
                            future_mean_roi_mae=float(np.mean([row['roi_mae'] for row in rows[1:]])),
                            future_mean_mask_iou=float(np.mean([row['approximate_mask_iou'] for row in rows[1:]])))
        physical[mode] = geometry
    native = [overlay_row(src, rendered['learned'][i], rendered['zero_weights'][i], indices[i]) for i, src in enumerate(source)]
    native_path = out / 'conditional_comparison_native.mp4'
    native_qa = write_video(native_path, native, 16.)
    holding = np.floor(np.arange(168) * 13 / 168).astype(int)
    slow_states = [overlay_row(source[i], rendered['learned'][i], rendered['zero_weights'][i], indices[i], slow=True) for i in range(13)]
    slowed = [slow_states[i] for i in holding]
    slow_path = out / 'conditional_comparison_7s_held.mp4'
    slow_qa = write_video(slow_path, slowed, 24.)
    learned_path = out / 'learned_forecast_native.mp4'
    learned_qa = write_video(learned_path, rendered['learned'], 16.)
    # Reduced contact sheet retains all three full-camera views at four times.
    sheet = Image.new('RGB', (1872, 1716), '#151b24')
    for row, state_index in enumerate([0, 4, 8, 12]):
        panel = Image.fromarray(native[state_index]).resize((1872, 429), Image.Resampling.LANCZOS)
        sheet.paste(panel, (0, row * 429))
    sheet.save(out / 'contact_sheet.jpg', quality=94)
    if digest(root / 'forecast.pt') != forecast_sha:
        raise AssertionError('Evaluation changed forecast')
    report = dict(forecast_sha256=forecast_sha, model_sha256=packet['model_sha256'], asset_sha256=packet['asset_sha256'],
                  source_video_sha256=digest(SOURCE), evaluator_source_sha256=digest(Path(__file__)),
                  observation_count=3, forecast_states=12, source_frames=indices, source_fps=source_fps,
                  inference_future_reads=0, evaluation_reads_future_frames=True, inference_output_unchanged=True,
                  RGB_diagnostic='Same fixed ROI x[0,670),y[80,480) for every method/frame. Source cat cutouts on neutral grey use approximate GrabCut, NOT ground-truth segmentation. Renderer has its own neutral background; no source pixels used in predictions.',
                  geometric_scope='World units arbitrary; centre-Y/face-area diagnostics cannot verify real contact, depth or anatomy. No learned depth.',
                  temporal_scope='Native13states16fps =0.8125sec file,0.75sec sample span. Seven-second file holds these states without interpolation or extra model steps.',
                  failed_model_gate=True, model_accepted=False, learning_or_fitting_performed=False, metrics=scores, geometry=physical,
                  video_qa={str(native_path):native_qa, str(slow_path):slow_qa, str(learned_path):learned_qa},
                  seconds=time.perf_counter()-start, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0)
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key:value for key,value in report.items() if key not in ['metrics','geometry']}), flush=True)


if __name__ == '__main__':
    with keep_windows_awake():
        main()
