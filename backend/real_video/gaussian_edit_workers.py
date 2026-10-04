"""Bounded specialist workers for source-conditioned PLANAR Gaussian edits.

These are deterministic program agents, not LLMs or a trained motion generator.
The input action comes from the source video. No recovered depth is claimed.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time
from contextlib import contextmanager

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import psutil

from .checkpoint_io import digest, save_inference_checkpoint, load_verified, keep_windows_awake
from .representation import encode_clip, render_fields
from .replay_fit import temporal_basis, physical_fields


KIND = 'source_conditioned_planar_gaussian_reconstruction'
PROFILES = {'pilot': (192, 96), 'detail': (512, 256)}
MAX_ESTIMATED_WORKING_BYTES = 2 * 1024**3
MAX_COEFFICIENT_BYTES = 256 * 1024**2


def profile_bounds(size, grid, frames, terms, profile='pilot', *, element_size=4):
    """Validate before allocating. Estimate is a conservative guard, not measured RAM.

    Accounts for analysis/coefficient copies and one frame's radius-five stencil.
    The detail profile is opt-in; old packets without the key remain pilot packets.
    """
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError('Unknown CPU analysis profile')
    if not all(type(v) is int for v in (size, grid, frames, terms, element_size)):
        raise ValueError('Dimensions and element size must be integers')
    max_size, max_grid = PROFILES[profile]
    if not (32 <= size <= max_size and 8 <= grid <= max_grid
            and 2 <= frames <= 64 and 1 <= terms <= frames and element_size in (2, 4, 8)):
        raise ValueError('Packet exceeds explicit CPU profile bounds')
    coefficients = 9 * terms * grid * grid * element_size
    # Rendering promotes coefficients to float32, hence max(..., 4).
    estimate = (4 * 9 * terms * grid * grid * max(element_size, 4)
                + 2 * 9 * frames * grid * grid * 4
                + frames * size * size * 3
                + grid * grid * 121 * 96)
    if coefficients > MAX_COEFFICIENT_BYTES or estimate > MAX_ESTIMATED_WORKING_BYTES:
        raise ValueError('Profile exceeds bounded CPU memory estimate')
    return estimate


def resource_gate():
    available = psutil.virtual_memory().available
    if available < 8 * 1024**3:
        raise RuntimeError('CPU conversion requires at least 8 GiB available RAM')
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged and battery.percent <= 20:
        raise RuntimeError('Connect AC power before CPU conversion')
    return available


class StageJournal:
    """Append-only evidence; failures are retained, but automatic resume is absent."""
    def __init__(self, path):
        self.path = Path(path)

    def emit(self, stage, status, **values):
        record = dict(schema_version=1, stage=stage, status=status, unix_seconds=time.time(), **values)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')

    @contextmanager
    def stage(self, name):
        self.emit(name, 'begin')
        started = time.perf_counter()
        artifacts = []
        try:
            yield artifacts
            evidence = [dict(path=str(Path(p).resolve()), sha256=digest(p)) for p in artifacts]
            self.emit(name, 'complete', seconds=time.perf_counter()-started, artifacts=evidence)
        except Exception as error:
            self.emit(name, 'failed', seconds=time.perf_counter()-started, error_type=type(error).__name__)
            raise


def validate_packet(packet):
    if packet.get('kind') != KIND or packet.get('geometry') != '2D planar; no inferred depth':
        raise ValueError('Only explicit planar reconstruction packets are supported')
    if packet.get('generated_motion') is not False:
        raise ValueError('This worker cannot claim generated motion')
    frames, grid, terms, size = (packet[k] for k in ('frames', 'grid', 'terms', 'size'))
    profile = packet.get('profile', 'pilot')
    profile_bounds(size, grid, frames, terms, profile)
    coef = packet['coefficients']
    if not isinstance(coef, torch.Tensor) or coef.device.type != 'cpu' or not coef.is_floating_point():
        raise ValueError('Coefficients must be floating CPU tensors')
    if coef.shape != (9, terms, grid, grid) or not torch.isfinite(coef).all():
        raise ValueError('Invalid Gaussian coefficients')
    profile_bounds(size, grid, frames, terms, profile, element_size=coef.element_size())
    if not torch.equal(packet['ids'], torch.arange(grid * grid, dtype=torch.int64)):
        raise ValueError('Persistent grid IDs changed')
    refresh = packet.get('coverage_refresh', False)
    if type(refresh) is not bool:
        raise ValueError('Coverage refresh flag must be explicit boolean')
    if refresh:
        generations = packet.get('track_generations')
        if (not isinstance(generations, torch.Tensor) or generations.device.type != 'cpu'
                or generations.dtype != torch.int32 or generations.shape != (frames,grid,grid)
                or (generations < 0).any() or (generations[0] != 0).any()):
            raise ValueError('Reseeded slots need valid track generations')
        changes = generations[1:] - generations[:-1]
        if (changes < 0).any() or (changes > 1).any():
            raise ValueError('Invalid track lifetime transition')
    if not math.isfinite(packet['fps']) or not 0 < packet['fps'] <= 60:
        raise ValueError('Invalid fps')
    if packet.get('radius') != 5 or packet.get('position_limit') is not None:
        raise ValueError('Unsupported renderer settings')
    floor = packet.get('min_log_scale', -2.)
    if (type(floor) not in (int, float) or not math.isfinite(floor)
            or floor != (-3. if profile == 'detail' else -2.)):
        raise ValueError('Unsupported scale floor for this profile')
    source_hash = packet.get('source_sha256')
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(c not in '0123456789abcdef' for c in source_hash)):
        raise ValueError('Source identity must be a lowercase SHA-256 hex digest')
    # A malformed ROI silently clips in Python slicing (or yields an empty
    # tensor and NaN PSNR), invalidating the recorded fidelity measurement.
    box = packet.get('content_box')
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(type(v) is int for v in box)):
        raise ValueError('Content box must contain four integer coordinates')
    x, y, width, height = box
    if not (0 <= x < size and 0 <= y < size and width > 0 and height > 0
            and x + width <= size and y + height <= size):
        raise ValueError('Content box must be nonempty and inside the image')
    return packet


def source_worker(source, *, size=128, grid=64, frames=40, terms=24, profile='pilot', refresh_coverage=False):
    """Analyze sampled source, retaining its duration and complete aspect ratio."""
    estimated_bytes = profile_bounds(size, grid, frames, terms, profile)
    source = Path(source)
    cap = cv2.VideoCapture(str(source))
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 2 or not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError('Unreadable source timing')
        duration = count / source_fps
        if not 0 < duration <= 10:
            raise ValueError('Source must be at most ten seconds')
        frames = min(frames, count)
        terms = min(terms, frames)
        indices = np.floor(np.arange(frames) * count / frames).astype(int)
        images = []
        content_box = None
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = cap.read()
            if not ok:
                raise ValueError('Source frame decode failed')
            h, w = bgr.shape[:2]
            ratio = size / max(h, w)
            nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
            left, top = (size - nw) // 2, (size - nh) // 2
            box = [left, top, nw, nh]
            if content_box is not None and box != content_box:
                raise ValueError('Changing source dimensions')
            content_box = box
            image = np.zeros((size, size, 3), dtype=np.uint8)
            image[top:top + nh, left:left + nw] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (nw, nh), interpolation=cv2.INTER_AREA)
            images.append(image)
    finally:
        cap.release()
    images = np.stack(images)
    start = time.perf_counter()
    if type(refresh_coverage) is not bool:
        raise ValueError('Coverage refresh must be boolean')
    encoded, generations = encode_clip(images, grid=grid, refresh_coverage=refresh_coverage, return_generations=True)
    fields = torch.from_numpy(encoded)
    basis = temporal_basis(frames, terms)
    coefficients = torch.einsum('tk,cthw->ckhw', basis, fields)
    packet = dict(kind=KIND, geometry='2D planar; no inferred depth', generated_motion=False,
                  coefficients=coefficients, ids=torch.arange(grid * grid), frames=frames,
                  grid=grid, terms=terms, size=size, radius=5, position_limit=None,
                  profile=profile, min_log_scale=-3. if profile == 'detail' else -2.,
                  estimated_working_bytes=estimated_bytes,
                  fps=frames / duration, source_sha256=digest(source), content_box=content_box,
                  source_indices=indices.tolist(), source_fps=source_fps,
                  dot_identity='Persistent grid IDs with unverified optical-flow correspondences',
                  appearance='Source-derived time-varying colors; not immutable material appearance',
                  analysis_seconds=time.perf_counter() - start, edits=[])
    if refresh_coverage:
        packet.update(coverage_refresh=True, track_generations=torch.from_numpy(generations),
                      dot_identity='Storage slots persist; material track identity is (slot_id, track_generation). Observed-frame reseeding invalidates old material memory.',
                      coverage_refresh_events=int(generations[-1].sum()))
    return validate_packet(packet), images


def appearance_worker(packet, *, rgb_gain=(1., 1., 1.)):
    """Source-free global color edit; no semantic region or geometry invention."""
    validate_packet(packet)
    gain = torch.as_tensor(rgb_gain, dtype=torch.float32)
    if gain.shape != (3,) or not torch.isfinite(gain).all() or (gain < .5).any() or (gain > 1.5).any():
        raise ValueError('RGB gains must be three finite values in [0.5, 1.5]')
    result = copy.deepcopy(packet)
    # Field colors x=2*c-1: x_new=g*x+(g-1). Constant DCT coefficient is sqrt(T).
    result['coefficients'][6:9] *= gain[:, None, None, None]
    result['coefficients'][6:9, 0] += (gain - 1)[:, None, None] * math.sqrt(packet['frames'])
    result['edits'].append(dict(worker='appearance', rgb_gain=gain.tolist(), scope='whole image'))
    return validate_packet(result)


def resolve_render_device(device='cpu'):
    """Require an explicit CPU/CUDA backend; never silently downgrade CUDA."""
    if not isinstance(device, (str, torch.device)):
        raise ValueError('Render device must be cpu, cuda or cuda:<index>')
    try:
        selected = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError('Render device must be cpu, cuda or cuda:<index>') from error
    if selected.type not in ('cpu', 'cuda') or (selected.type == 'cpu' and selected.index is not None):
        raise ValueError('Only explicit CPU or CUDA rendering is supported')
    if selected.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA rendering requested but CUDA is unavailable; CPU fallback is disabled')
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise RuntimeError('Requested CUDA device does not exist; CPU fallback is disabled')
    return selected


def iter_rendered_frames(packet, frame_indices=None, *, device='cpu'):
    """Yield (frame_index, CPU CHW) from coefficients only, retaining no output history.

    Detail uses a recorded -3 scale floor: exp(-2) would wrongly inflate the
    encoder's minimum grid256 sigma (0.7*32/256=0.0875 canonical pixels).
    This is not a Mip-Splatting implementation or a guarantee of alias-free output.
    Coefficients and the CPU-computed basis are transferred once per iterator.
    CUDA-to-CPU frame copies are blocking so consumed-frame timings include GPU
    completion and transfer. The serialized packet remains CPU-only and unchanged.
    """
    validate_packet(packet)
    indices = list(range(packet['frames'])) if frame_indices is None else list(frame_indices)
    if (not 1 <= len(indices) <= packet['frames']
            or any(type(i) is not int or not 0 <= i < packet['frames'] for i in indices)):
        raise ValueError('Frame indices must be a bounded nonempty sequence of in-range integers')
    selected = resolve_render_device(device)
    with torch.inference_mode():
        coefficients = packet['coefficients'].to(device=selected, dtype=torch.float32)
        basis = temporal_basis(packet['frames'], packet['terms']).to(selected)
    for index in indices:
        with torch.inference_mode():
            fields = physical_fields(coefficients, basis[index:index+1], packet.get('position_limit'))[None]
            frame = render_fields(fields, size=packet['size'], radius=packet['radius'],
                                  min_log_scale=packet.get('min_log_scale', -2.))[0, 0].cpu()
        yield index, frame


def rendering_worker(packet, *, device='cpu'):
    """Compatibility wrapper retaining the full output; use iterator for long runs."""
    return torch.stack([frame for _, frame in iter_rendered_frames(packet, device=device)])


def evaluation_worker(packet, output, reference):
    validate_packet(packet)
    if (not isinstance(output, torch.Tensor) or output.device.type != 'cpu'
            or not output.is_floating_point()
            or tuple(output.shape) != (packet['frames'], 3, packet['size'], packet['size'])
            or not torch.isfinite(output).all()):
        raise ValueError('Invalid rendered output')
    if (not isinstance(reference, np.ndarray) or reference.dtype != np.uint8
            or reference.shape != (packet['frames'], packet['size'], packet['size'], 3)):
        raise ValueError('Reference must be finite uint8 RGB frames of the matching shape')
    target = torch.from_numpy(reference).permute(0, 3, 1, 2).float() / 255
    if target.shape != output.shape:
        raise ValueError('Reference mismatch')
    x, y, w, h = packet['content_box']
    mse = float((output[:, :, y:y+h, x:x+w] - target[:, :, y:y+h, x:x+w]).square().mean())
    return dict(content_psnr_db=-10 * math.log10(max(mse, 1e-12)),
                visual_acceptance='pending independent source/output inspection',
                autonomous_generation_accepted=False, geometry_3d_accepted=False,
                compression_or_compute_savings_demonstrated=False)


def run(source, out, *, size=128, grid=64, frames=40, terms=24, rgb_gain=(1., 1., 1.), profile='pilot'):
    profile_bounds(size, grid, frames, terms, profile)
    available_ram = resource_gate()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    journal = StageJournal(out / 'stages.jsonl')
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    started = time.perf_counter()
    with journal.stage('source_analysis') as artifacts:
        packet, reference = source_worker(source, size=size, grid=grid, frames=frames, terms=terms, profile=profile)
        analysis = time.perf_counter() - started
        save_inference_checkpoint(packet, out / 'gaussians.pt')
        packet = load_verified(out / 'gaussians.pt')
        artifacts.append(out / 'gaussians.pt')
    started = time.perf_counter()
    with journal.stage('rendering'):
        output = rendering_worker(packet)
    rasterization = time.perf_counter() - started
    with journal.stage('evaluation'):
        quality = evaluation_worker(packet, output, reference)
    started = time.perf_counter()
    with journal.stage('appearance') as artifacts:
        edited = appearance_worker(packet, rgb_gain=rgb_gain)
        editing = time.perf_counter() - started
        save_inference_checkpoint(edited, out / 'edited_gaussians.pt')
        edited = load_verified(out / 'edited_gaussians.pt')
        artifacts.append(out / 'edited_gaussians.pt')
    started = time.perf_counter()
    with journal.stage('encoding') as artifacts:
        frames_u8 = (output.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
        imageio.mimsave(out / 'gaussian_replay.mp4', frames_u8, fps=packet['fps'], codec='libx264', macro_block_size=1)
        artifacts.append(out / 'gaussian_replay.mp4')
    encoding = time.perf_counter() - started
    started = time.perf_counter()
    with journal.stage('edited_rendering'):
        edited_output = rendering_worker(edited)
    edit_rasterization = time.perf_counter() - started
    started = time.perf_counter()
    with journal.stage('edited_encoding') as artifacts:
        edited_u8 = (edited_output.clamp(0, 1)*255).round().byte().permute(0, 2, 3, 1).numpy()
        imageio.mimsave(out/'gaussian_edited.mp4', edited_u8, fps=packet['fps'], codec='libx264', macro_block_size=1)
        artifacts.append(out/'gaussian_edited.mp4')
    edit_encoding = time.perf_counter() - started
    report = dict(kind=KIND, workers=['source_analysis', 'appearance', 'rendering', 'evaluation'],
                  agent_type='Deterministic scoped workers; no LLM inference',
                  source_sha256=packet['source_sha256'], source=str(Path(source).resolve()),
                  duration_seconds=packet['frames']/packet['fps'], grid=grid, size=size, frames=packet['frames'],
                  profile=profile, min_log_scale=packet['min_log_scale'],
                  estimated_working_bytes=packet['estimated_working_bytes'],
                  source_analysis_seconds=analysis, coefficient_edit_seconds=editing,
                  gaussian_rasterization_seconds=rasterization, encoding_seconds=encoding,
                  edited_rasterization_seconds=edit_rasterization, edited_encoding_seconds=edit_encoding,
                  entry_available_ram_bytes=available_ram, local_cuda_used=False,
                  training_seconds=0, local_neural_generation_seconds=0, local_api_calls=0,
                  coefficient_bytes=packet['coefficients'].numel()*packet['coefficients'].element_size(),
                  source_mp4_bytes=Path(source).stat().st_size, gaussian_checkpoint_bytes=(out/'gaussians.pt').stat().st_size,
                  edit_rendered=True, rgb_gain=list(rgb_gain), **quality,
                  limits=['Source-conditioned 2D fitting, not 3D reconstruction or new actions.',
                          'Temporal DCT truncation and low-resolution sampling lose detail.',
                          'Flow IDs are not verified physical material correspondences.',
                          'Gaussian conversion is extra work after paid source generation.',
                          'Replay video shows original coefficients; edited video shows only global color adjustment.',
                          'Stage journal records failures but does not implement automatic resume.'])
    with journal.stage('report') as artifacts:
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        artifacts.append(out / 'report.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--grid', type=int, default=64)
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--terms', type=int, default=24)
    parser.add_argument('--profile', choices=tuple(PROFILES), default='pilot')
    parser.add_argument('--rgb-gain', type=float, nargs=3, default=(1., 1., 1.))
    args = parser.parse_args()
    with keep_windows_awake():
        print(json.dumps(run(args.source, args.out, size=args.size, grid=args.grid, frames=args.frames, terms=args.terms, rgb_gain=args.rgb_gain, profile=args.profile), indent=2))
