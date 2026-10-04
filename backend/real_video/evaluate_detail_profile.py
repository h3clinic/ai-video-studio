"""Bounded source-conditioned detail ablation, never autonomous generation.

The saved Gaussian coefficients alone produce the two candidate videos. All
source/old-output accesses for fidelity scoring happen after rendering finishes.
"""
import argparse
import json
import math
from pathlib import Path
import threading
import time
from unittest.mock import patch

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import psutil
import torch

from .checkpoint_io import digest, load_verified, save_inference_checkpoint, keep_windows_awake
from .gaussian_edit_workers import (source_worker, appearance_worker,
                                    iter_rendered_frames, resource_gate, StageJournal, resolve_render_device)
from .inspect_runway_pilot import extract


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def check_deadline(deadline):
    if time.perf_counter() > deadline:
        raise TimeoutError('Detail experiment exceeded its wall budget')


def stream_video(packet, path, deadline, device='cpu'):
    """Crop only stored letterbox padding; all scene pixels come from splats."""
    x, y, w, h = packet['content_box']
    raster, packing, encoding = 0., 0., 0.
    start = time.perf_counter()
    writer = imageio.get_writer(path, fps=packet['fps'], codec='libx264',
                               macro_block_size=1, ffmpeg_params=['-crf', '18'])
    encoding += time.perf_counter()-start
    try:
        iterator = iter(iter_rendered_frames(packet, device=device))
        # A regression guard: replay may not open the source or another video.
        with patch('cv2.VideoCapture', side_effect=AssertionError('Source I/O during replay')):
            for expected in range(packet['frames']):
                check_deadline(deadline)
                start = time.perf_counter()
                index, frame = next(iterator)
                raster += time.perf_counter()-start
                if index != expected:
                    raise ValueError('Frame order changed')
                start = time.perf_counter()
                pixels = (frame[:, y:y+h, x:x+w].clamp(0, 1)*255).round().byte().permute(1,2,0).numpy()
                packing += time.perf_counter()-start
                start = time.perf_counter()
                writer.append_data(pixels)
                encoding += time.perf_counter()-start
    finally:
        start = time.perf_counter()
        writer.close()
        encoding += time.perf_counter()-start
    return dict(rasterization_seconds=raster, packing_seconds=packing,
                encoding_call_seconds=encoding, source_io_blocked=True,
                encoding_note='Encoder runs asynchronously; append/close wall time is not isolated encoder CPU time.',
                width=w, height=h, frames=packet['frames'], fps=packet['fps'])


def decoded(cap, index=None):
    if index is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, bgr = cap.read()
    if not ok:
        raise ValueError('Video decode failed')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def gradient_energy(rgb):
    gray = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)/8
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)/8
    return float(np.mean(dx*dx+dy*dy))


def matched_metrics(source, baseline_video, baseline_packet, candidate_video, packet, out, deadline):
    if (packet['source_sha256'] != baseline_packet['source_sha256']
            or packet['source_indices'] != baseline_packet['source_indices']):
        raise ValueError('Matched comparison requires identical source and frame indices')
    caps = [cv2.VideoCapture(str(p)) for p in (source, baseline_video, candidate_video)]
    totals = {'baseline':dict(mse=0., gradient=0.), 'candidate':dict(mse=0., gradient=0.)}
    reference_gradient = 0.
    indices = sorted(set(round(i*(packet['frames']-1)/3) for i in range(4)))
    montage = Image.new('RGB', (1440, 4*294), (20,28,36))
    draw = ImageDraw.Draw(montage)
    x,y,w,h = baseline_packet['content_box']
    source_size = None
    try:
        for frame_index, source_index in enumerate(packet['source_indices']):
            check_deadline(deadline)
            reference = decoded(caps[0], source_index)
            baseline = decoded(caps[1])[y:y+h, x:x+w]
            candidate = decoded(caps[2])
            sh, sw = reference.shape[:2]
            source_size = [sw, sh]
            reference_gradient += gradient_energy(reference)
            aligned = []
            for key, rgb in (('baseline',baseline), ('candidate',candidate)):
                rgb = cv2.resize(rgb, (sw,sh), interpolation=cv2.INTER_CUBIC)
                delta = (rgb.astype(np.float32)-reference.astype(np.float32))/255
                totals[key]['mse'] += float(np.mean(delta*delta, dtype=np.float64))
                totals[key]['gradient'] += gradient_energy(rgb)
                aligned.append(rgb)
            if frame_index in indices:
                row = indices.index(frame_index)
                for col,(label,rgb) in enumerate(zip(('SOURCE','OLD GAUSSIANS','DETAIL GAUSSIANS'),(reference,*aligned))):
                    preview = Image.fromarray(rgb)
                    preview.thumbnail((480,270))
                    montage.paste(preview, (col*480,row*294))
                    draw.text((col*480+8,row*294+273),f'{label} | source frame {source_index}',fill='white')
    finally:
        for cap in caps:
            cap.release()
    montage.save(out/'comparison.png')
    count = packet['frames']
    for metrics in totals.values():
        metrics['psnr_db'] = -10*math.log10(max(metrics.pop('mse')/count,1e-12))
        metrics['gradient_energy_ratio'] = metrics.pop('gradient')/max(reference_gradient,1e-12)
    return dict(comparison=totals, reference_resolution=source_size, evaluated_frames=count,
                source_indices=packet['source_indices'], output_resize='OpenCV cubic to original source dimensions',
                split='Iterated development reconstruction; no held-out generative claim',
                pixel_metric_scope='Decoded MP4 vs exact sampled original RGB frame; NOT low-resolution target PSNR',
                psnr_gain_db=totals['candidate']['psnr_db']-totals['baseline']['psnr_db'])


def run(source, baseline_dir, out, budget_seconds=540, refresh_coverage=False, device='cpu'):
    if not 10 <= budget_seconds <= 600:
        raise ValueError('Bounded experiment required')
    entry_ram = resource_gate()
    render_device = resolve_render_device(device)
    if render_device.type == 'cuda':
        torch.cuda.synchronize(render_device)
        torch.cuda.reset_peak_memory_stats(render_device)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    source, baseline_dir, out = map(Path, (source, baseline_dir, out))
    out.mkdir(parents=True, exist_ok=False)
    journal = StageJournal(out/'stages.jsonl')
    save_json(out/'input_manifest.json', dict(
        source=str(source.resolve()), source_sha256=digest(source),
        code={name:digest(Path(__file__).parent/name) for name in
              ('evaluate_detail_profile.py','gaussian_edit_workers.py','representation.py')},
        baseline_sha256=digest(baseline_dir/'gaussians.pt'),
        parameters=dict(size=512,grid=256,refresh_coverage=refresh_coverage,budget_seconds=budget_seconds,device=str(render_device))))
    begun = time.perf_counter()
    deadline = begun+budget_seconds
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        baseline = load_verified(baseline_dir/'gaussians.pt')
        if digest(source) != baseline['source_sha256']:
            raise ValueError('Source hash mismatch')
        with journal.stage('higher_detail_analysis') as artifacts:
            start = time.perf_counter()
            packet, reference = source_worker(source, size=512, grid=256, frames=baseline['frames'],
                                              terms=baseline['terms'], profile='detail', refresh_coverage=refresh_coverage)
            fitting = time.perf_counter()-start
            del reference
            if packet['source_indices'] != baseline['source_indices']:
                raise ValueError('Source sampling changed')
            start = time.perf_counter()
            save_inference_checkpoint(packet, out/'gaussians.pt')
            packet = load_verified(out/'gaussians.pt')
            checkpoint_seconds = time.perf_counter()-start
            artifacts.append(out/'gaussians.pt')
        with journal.stage('detail_gaussian_render') as artifacts:
            original = stream_video(packet, out/'gaussian_replay.mp4', deadline, render_device)
            artifacts.append(out/'gaussian_replay.mp4')
        with journal.stage('coefficient_edit') as artifacts:
            start = time.perf_counter()
            edited = appearance_worker(packet, rgb_gain=(1.05,1.,.95))
            edit_seconds = time.perf_counter()-start
            save_inference_checkpoint(edited, out/'edited_gaussians.pt')
            edited = load_verified(out/'edited_gaussians.pt')
            artifacts.append(out/'edited_gaussians.pt')
        with journal.stage('edited_gaussian_render') as artifacts:
            edited_timings = stream_video(edited, out/'gaussian_edited.mp4', deadline, render_device)
            artifacts.append(out/'gaussian_edited.mp4')
        with journal.stage('matched_source_resolution_evaluation') as artifacts:
            metrics = matched_metrics(source, baseline_dir/'gaussian_replay.mp4', baseline,
                                      out/'gaussian_replay.mp4', packet, out, deadline)
            artifacts.append(out/'comparison.png')
        with journal.stage('actual_decoded_media') as artifacts:
            extract(out, 'gaussian_replay.mp4', 'review_original')
            extract(out, 'gaussian_edited.mp4', 'review_edited')
            artifacts.extend([out/'review_original'/'contact.png',out/'review_edited'/'contact.png'])
        report = dict(kind='development_source_conditioned_planar_detail_ablation',
                      source_sha256=digest(source), baseline_sha256=digest(baseline_dir/'gaussians.pt'),
                      profile=dict(size=512,grid=256,gaussians=256**2,frames=packet['frames'],terms=packet['terms']),
                      coverage_refresh=refresh_coverage, track_refresh_events=packet.get('coverage_refresh_events',0),
                      codec=dict(candidate='libx264 CRF18',baseline='Historical imageio default quality; not re-encoded to match CRF18'),
                      analysis_seconds=fitting, checkpoint_save_reload_seconds=checkpoint_seconds,
                      coefficient_edit_seconds=edit_seconds, original=original, edited=edited_timings,
                      gaussian_checkpoint_bytes=(out/'gaussians.pt').stat().st_size,
                      baseline_checkpoint_bytes=(baseline_dir/'gaussians.pt').stat().st_size,
                      source_mp4_bytes=source.stat().st_size, entry_available_ram_bytes=entry_ram,
                      peak_process_rss_bytes=max(peak[0], process.memory_info().rss),
                      wall_seconds=time.perf_counter()-begun, local_api_calls=0, local_cuda_used=render_device.type=='cuda',
                      render_device=str(render_device),
                      gpu_name=torch.cuda.get_device_name(render_device) if render_device.type=='cuda' else None,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(render_device) if render_device.type=='cuda' else 0,
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(render_device) if render_device.type=='cuda' else 0,
                      training_seconds=0, neural_generation_seconds=0,
                      accepted_autonomous_video=False, accepted_3d=False, **metrics,
                      limitations=['Same observed action; all source frames used during fitting.',
                                   'Grid IDs use unverified optical-flow tracks and time-varying color.',
                                   'With coverage_refresh, reused slots get a new track generation; newly exposed color is observed, not imagined.',
                                   'Resolution, splat density, scale floor and encoding differ together: not a single-factor causal ablation.',
                                   'Not modified neural generator weights or newly generated motion.',
                                   'More detail uses more state and rasterization; not demonstrated savings.',
                                   'Only black letterbox padding removed; no subject crop/overlay animation.',
                                   'Visual review remains required; quantitative score cannot accept anatomy.'])
        save_json(out/'report.json', report)
        return report
    finally:
        stop.set()
        monitor_thread.join(timeout=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=float, default=540)
    parser.add_argument('--refresh-coverage', action='store_true')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    with keep_windows_awake():
        print(json.dumps(run(args.source, args.baseline_dir, args.out, args.budget_seconds, args.refresh_coverage, args.device), indent=2))
