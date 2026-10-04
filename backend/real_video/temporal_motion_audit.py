"""CPU-only motion diagnostics, not an automatic video-quality assessment.

The dense-flow/affine residuals distinguish some frozen or rigidly translated
fixtures from locally changing motion. They do NOT establish articulation,
3D motion, identity retention, generation, or physically plausible movement.
Use an explicit evidence-linked visual review before accepting a result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


LIMITATIONS = [
    'Optical flow estimates image motion, not Gaussian identities or 3D trajectories.',
    'Flow can be wrong on fur, occlusion, blur, low texture, lighting changes, and compression.',
    'Affine residual can indicate articulation, but also parallax, tearing, hallucination, or flow error.',
    'A rigid slide can coexist with residual motion; these thresholds are uncalibrated diagnostic heuristics.',
    'Moving support is selected by flow magnitude, not an animal segmentation mask.',
    'Sampled frames and numeric analysis do not attest to continuous playback being viewed.',
    'Source/output comparisons are aggregate diagnostics, not aligned reconstruction error or external validation.',
]


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {'count': 0, 'mean': None, 'median': None, 'p90': None, 'p95': None}
    return dict(count=int(values.size), mean=float(values.mean()),
                median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                p95=float(np.percentile(values, 95)))


def affine_residual(flow, valid=None, sample_step=4, ransac_threshold=0.75):
    """Robust full-affine fit to x -> x + flow, returning fit and residual.

    Flow and residual units are analysis-image pixels per frame transition.
    The default grid bounds fitting cost; residuals are evaluated on ALL valid
    pixels, including RANSAC outliers. Excluding outliers there would hide the
    very non-affine motion this diagnostic is intended to measure.
    """
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError('Expected finite H x W x 2 flow.')
    if sample_step < 1 or ransac_threshold <= 0:
        raise ValueError('sample_step and ransac_threshold must be positive.')
    h, w = flow.shape[:2]
    mask = np.isfinite(flow).all(axis=2)
    if valid is not None:
        if np.asarray(valid).shape != (h, w):
            raise ValueError('valid mask shape differs from flow.')
        mask &= np.asarray(valid, dtype=bool)
    y, x = np.mgrid[:h, :w]
    xy = np.stack((x, y), axis=-1).astype(np.float32)
    sampled = mask & (y % sample_step == 0) & (x % sample_step == 0)
    source = xy[sampled]
    target = source + flow[sampled]
    empty = dict(available=False, matrix=None, fit_sample_count=int(len(source)),
                 inlier_fraction=None, residual_px=_stats([]),
                 affine_explained_fraction=None)
    if len(source) < 6:
        return empty, np.full((h, w), np.nan, dtype=np.float32)
    matrix, inliers = cv2.estimateAffine2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold,
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    if matrix is None or not np.isfinite(matrix).all():
        return empty, np.full((h, w), np.nan, dtype=np.float32)
    prediction = xy @ matrix[:, :2].T + matrix[:, 2] - xy
    residual = np.linalg.norm(flow - prediction, axis=-1)
    residual[~mask] = np.nan
    energy = float(np.square(flow[mask]).sum())
    residual_energy = float(np.square(residual[mask]).sum())
    # Can be negative for a bad fit; never clip it into an apparent success.
    explained = 1.0 - residual_energy / energy if energy > 1e-8 else None
    report = dict(available=True, matrix=matrix.tolist(), fit_sample_count=int(len(source)),
                  inlier_fraction=float(inliers.mean()), residual_px=_stats(residual[mask]),
                  affine_explained_fraction=explained)
    return report, residual


def _gray_rgb(image, max_side):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected RGB uint8 H x W x 3 frames.')
    h, w = image.shape[:2]
    if h < 16 or w < 16:
        raise ValueError('Frames must be at least 16 x 16.')
    if max_side < 16:
        raise ValueError('max_side must be at least 16.')
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (max(16, round(w * scale)), max(16, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def pair_motion(previous, current, max_side=384):
    """Analyze one transition of equal-size RGB uint8 frames, on CPU only."""
    if np.asarray(previous).shape != np.asarray(current).shape:
        raise ValueError('A frame pair must have equal shape.')
    before, after = _gray_rgb(previous, max_side), _gray_rgb(current, max_side)
    h, w = before.shape
    options = dict(pyr_scale=0.5, levels=4, winsize=21, iterations=4,
                   poly_n=7, poly_sigma=1.5, flags=0)
    forward = cv2.calcOpticalFlowFarneback(before, after, None, **options)
    backward = cv2.calcOpticalFlowFarneback(after, before, None, **options)
    y, x = np.mgrid[:h, :w].astype(np.float32)
    destination = np.stack((x, y), axis=-1) + forward
    reverse_at_destination = cv2.remap(
        backward, destination[..., 0], destination[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    magnitude = np.linalg.norm(forward, axis=-1)
    fb_error = np.linalg.norm(forward + reverse_at_destination, axis=-1)
    inside = ((destination[..., 0] >= 1) & (destination[..., 0] < w - 2)
              & (destination[..., 1] >= 1) & (destination[..., 1] < h - 2))
    valid = inside & (fb_error <= 0.5 + 0.1 * magnitude) & np.isfinite(magnitude)
    # Texture-free pixels are not reliable affine fitting observations.
    texture = cv2.GaussianBlur(cv2.Sobel(before, cv2.CV_32F, 1, 0) ** 2
                               + cv2.Sobel(before, cv2.CV_32F, 0, 1) ** 2,
                               (5, 5), 0) > 16
    reliable = valid & texture
    scene_fit, _ = affine_residual(forward, reliable)
    moving = reliable & (magnitude >= 0.5)
    moving_fit, _ = affine_residual(forward, moving)
    delta = np.abs(after.astype(np.float32) - before.astype(np.float32))
    flow_stats = _stats(magnitude[reliable])
    pixel_mad = float(delta.mean() / 255.0)
    frozen = bool(pixel_mad < 0.003 and (flow_stats['p95'] is None or flow_stats['p95'] < 0.15))
    rigid = bool(not frozen and moving_fit['available']
                 and moving_fit['affine_explained_fraction'] is not None
                 and moving_fit['affine_explained_fraction'] > 0.9
                 and moving_fit['residual_px']['p90'] < 0.5)
    residual = moving_fit['residual_px']['p90']
    return dict(
        analysis_width=w, analysis_height=h,
        gray_pixel_mean_absolute_change_0_to_1=pixel_mad,
        gray_changed_fraction_above_8_levels=float((delta > 8).mean()),
        flow_reliable_fraction=float(reliable.mean()),
        moving_reliable_fraction=float(moving.mean()),
        flow_magnitude_px=flow_stats,
        scene_affine=scene_fit, moving_support_affine=moving_fit,
        heuristics=dict(frozen_candidate=frozen, rigid_motion_candidate=rigid,
                        residual_motion_detected=bool(residual is not None and residual >= 0.5)),
    )


def analyze_video(path, max_side=384):
    """Stream every adjacent decoded frame pair without loading the video in RAM."""
    path = Path(path).resolve(strict=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = None
    container_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    container_count = int(container_count) if np.isfinite(container_count) and container_count > 0 else None
    previous, pairs, frame_count, original_shape = None, [], 0, None
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if original_shape is None:
                original_shape = rgb.shape
            if rgb.shape != original_shape:
                raise ValueError('Variable-resolution video is unsupported.')
            if previous is not None:
                pair = pair_motion(previous, rgb, max_side=max_side)
                pair.update(from_frame=frame_count - 1, to_frame=frame_count)
                pairs.append(pair)
            previous = rgb
            frame_count += 1
    finally:
        capture.release()
    if not frame_count:
        raise ValueError(f'No decodable frames: {path}')
    def aggregate(key):
        return _stats([pair[key] for pair in pairs])
    def fraction(flag):
        return float(np.mean([pair['heuristics'][flag] for pair in pairs])) if pairs else None
    magnitudes = [p['flow_magnitude_px']['p95'] for p in pairs
                  if p['flow_magnitude_px']['p95'] is not None]
    scene_residual = [p['scene_affine']['residual_px']['p90'] for p in pairs
                      if p['scene_affine']['available']]
    moving_residual = [p['moving_support_affine']['residual_px']['p90'] for p in pairs
                       if p['moving_support_affine']['available']]
    diagonal = np.hypot(pairs[0]['analysis_width'], pairs[0]['analysis_height']) if pairs else 1
    return dict(
        path=str(path), sha256=_hash(path), fps_metadata=fps, decoded_frame_count=frame_count,
        container_frame_count_metadata=container_count,
        decoded_count_matches_metadata=(frame_count == container_count) if container_count else None,
        duration_seconds=frame_count / fps if fps else None,
        first_to_last_frame_seconds=(frame_count - 1) / fps if fps else None,
        width=original_shape[1], height=original_shape[0], analyzed_transition_count=len(pairs),
        summary=dict(
            pixel_mean_absolute_change=aggregate('gray_pixel_mean_absolute_change_0_to_1'),
            changed_pixel_fraction=aggregate('gray_changed_fraction_above_8_levels'),
            reliable_flow_fraction=aggregate('flow_reliable_fraction'),
            flow_p95_px=_stats(magnitudes),
            flow_p95_diagonals_per_second=_stats(np.asarray(magnitudes) * fps / diagonal) if fps else None,
            scene_affine_residual_p90_px=_stats(scene_residual),
            moving_support_residual_p90_px=_stats(moving_residual),
            frozen_candidate_transition_fraction=fraction('frozen_candidate'),
            rigid_motion_candidate_transition_fraction=fraction('rigid_motion_candidate'),
            residual_motion_transition_fraction=fraction('residual_motion_detected'),
        ), pairs=pairs,
    )


def _sample_video(report, directory, prefix, count=5):
    indices = np.unique(np.linspace(0, report['decoded_frame_count'] - 1, count).round().astype(int))
    wanted = set(indices.tolist())
    capture = cv2.VideoCapture(report['path'])
    evidence, images = [], []
    index = 0
    try:
        while wanted:
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('Sample extraction ended before the recorded frame count.')
            if index in wanted:
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                path = directory / f'{prefix}_{index:06d}.png'
                image.save(path)
                evidence.append(dict(path=str(path.resolve()), frame_index=index, sha256=_hash(path)))
                images.append(image)
                wanted.remove(index)
            index += 1
    finally:
        capture.release()
    return evidence, images


def write_audit(video, out, source=None, max_side=384):
    """Create a NEW audit directory. Never overwrite input or prior evaluations."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite audit directory: {out}')
    video = Path(video).resolve(strict=True)
    source = Path(source).resolve(strict=True) if source else None
    if not video.is_file() or (source is not None and not source.is_file()):
        raise ValueError('Inputs must be video files.')
    out.mkdir(parents=True, exist_ok=False)
    reports, rows = {}, []
    for name, path in [('output', video)] + ([('source', source)] if source else []):
        report = analyze_video(path, max_side=max_side)
        evidence, images = _sample_video(report, out, name)
        report['sampled_frames'] = evidence
        reports[name] = report
        rows.append((name, images, evidence))
    cell_w, cell_h, label_h = 320, 200, 30
    sheet = Image.new('RGB', (cell_w * max(len(row[1]) for row in rows),
                              (cell_h + label_h) * len(rows)), '#202020')
    draw = ImageDraw.Draw(sheet)
    for row_index, (name, images, evidence) in enumerate(rows):
        for column, (frame, item) in enumerate(zip(images, evidence)):
            frame = frame.copy()
            frame.thumbnail((cell_w, cell_h))
            x, y = column * cell_w, row_index * (cell_h + label_h)
            sheet.paste(frame, (x + (cell_w - frame.width) // 2, y + label_h))
            draw.text((x + 6, y + 6), f'{name} frame {item["frame_index"]}', fill='white')
    sheet.save(out / 'sampled_frames.jpg', quality=95)
    # Numeric analysis cannot provide observations for these fields.
    if __package__:
        from real_video.articulation_quality import create_review
    else:  # Also support `python real_video/temporal_motion_audit.py ...`.
        from articulation_quality import create_review
    pending_review = create_review(video, reports['output']['sampled_frames'])
    pending_review['limitations'] = list(LIMITATIONS)
    (out / 'visual_review_pending.json').write_text(json.dumps(pending_review, indent=2), encoding='utf-8')
    report = dict(schema_version=1, method='Farneback + forward/backward consistency + RANSAC full affine',
                  execution_device='cpu', max_analysis_side=max_side,
                  quality_accepted=False, automatic_quality_acceptance=False,
                  continuous_playback_observed=False,
                  threshold_units='analysis-image pixels per adjacent frame unless specified',
                  limitations=LIMITATIONS, videos=reports,
                  sampled_sheet=str((out / 'sampled_frames.jpg').resolve()))
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--source')
    parser.add_argument('--out', required=True, help='New output directory; existing paths are refused.')
    parser.add_argument('--max-side', type=int, default=384)
    args = parser.parse_args()
    report = write_audit(args.video, args.out, source=args.source, max_side=args.max_side)
    print(json.dumps({name: value['summary'] for name, value in report['videos'].items()}, indent=2))
    print(f'Audit written to {Path(args.out).resolve()}; no automatic quality acceptance.')


if __name__ == '__main__':
    main()
