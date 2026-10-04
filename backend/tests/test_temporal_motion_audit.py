"""Local analytic/image fixtures test diagnostics, NOT video generation."""
import cv2
import numpy as np

from real_video.temporal_motion_audit import affine_residual, analyze_video, pair_motion, write_audit


def test_affine_removal_translation_and_local_nonrigid_residual():
    flow = np.zeros((96, 128, 2), dtype=np.float32)
    flow[..., 0], flow[..., 1] = 3, -2
    fit, residual = affine_residual(flow)
    assert fit['available'] and np.nanmax(residual) < 1e-5
    assert fit['affine_explained_fraction'] > 0.9999
    # A minority independently moving region must not disappear as RANSAC outliers.
    flow[32:72, 40:88, 1] += 5
    nonrigid, residual = affine_residual(flow)
    assert nonrigid['available']
    assert nonrigid['residual_px']['p90'] > 4.5
    assert np.median(residual[32:72, 40:88]) > 4.5


def test_affine_removal_rotation_scale_and_shear():
    y, x = np.mgrid[:96, :128]
    points = np.stack((x, y), axis=-1).astype(np.float32)
    linear = np.array([[1.05, -0.08], [0.12, 0.96]], dtype=np.float32)
    flow = points @ linear.T + [2, -1] - points
    fit, residual = affine_residual(flow)
    assert fit['available'] and np.nanmax(residual) < 1e-4


def test_insufficient_and_nonfinite_flow_are_unavailable_not_success():
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    valid = np.zeros((32, 32), dtype=bool)
    valid[0, 0] = True
    fit, residual = affine_residual(flow, valid)
    assert not fit['available'] and fit['affine_explained_fraction'] is None
    flow[:] = np.nan
    fit, residual = affine_residual(flow)
    assert not fit['available'] and np.isnan(residual).all()


def _texture():
    rng = np.random.default_rng(734)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (128, 160), dtype=np.uint8), (3, 3), 0)
    return np.repeat(gray[..., None], 3, axis=2)


def test_actual_frames_freeze_and_translation():
    frame = _texture()
    freeze = pair_motion(frame, frame)
    assert freeze['heuristics']['frozen_candidate']
    transform = np.array([[1, 0, 3], [0, 1, 1]], dtype=np.float32)
    moved = cv2.warpAffine(frame, transform, (160, 128), borderMode=cv2.BORDER_REFLECT)
    translation = pair_motion(frame, moved)
    assert not translation['heuristics']['frozen_candidate']
    assert translation['flow_magnitude_px']['median'] > 2.5
    assert translation['moving_support_affine']['affine_explained_fraction'] > 0.95
    assert translation['heuristics']['rigid_motion_candidate']


def test_actual_frames_opposed_local_motion_has_residual():
    frame = _texture()
    y, x = np.mgrid[:128, :160].astype(np.float32)
    # Smoothly opposite displacement on top/bottom regions: not one affine field.
    displacement = 4 * np.sin(y / 128 * 2 * np.pi)
    moved = cv2.remap(frame, x - displacement, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    result = pair_motion(frame, moved)
    assert not result['heuristics']['frozen_candidate']
    assert result['moving_support_affine']['residual_px']['p90'] > 0.5
    assert result['heuristics']['residual_motion_detected']


def test_array_contract_rejects_mismatched_or_float_frames():
    frame = _texture()
    for other in (frame[:100], frame.astype(np.float32)):
        try:
            pair_motion(frame, other)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')


def test_existing_output_is_never_overwritten():
    from pathlib import Path
    try:
        write_audit(__file__, Path(__file__).parent)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing output path accepted')


def test_single_frame_and_missing_fps_remain_undefined_not_motion_pass():
    from unittest.mock import patch
    class FakeCapture:
        def __init__(self):
            self.read_count, self.released = 0, False
        def isOpened(self):
            return True
        def get(self, key):
            return 0.0
        def read(self):
            self.read_count += 1
            return (True, _texture()) if self.read_count == 1 else (False, None)
        def release(self):
            self.released = True
    capture = FakeCapture()
    with patch('real_video.temporal_motion_audit.cv2.VideoCapture', return_value=capture):
        report = analyze_video(__file__)
    assert capture.released
    assert report['decoded_frame_count'] == 1
    assert report['duration_seconds'] is None
    assert report['summary']['frozen_candidate_transition_fraction'] is None
    assert report['summary']['flow_p95_px']['count'] == 0
