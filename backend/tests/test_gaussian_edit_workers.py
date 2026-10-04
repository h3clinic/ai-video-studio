import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from real_video.gaussian_edit_workers import (KIND, validate_packet, appearance_worker,
                                             source_worker, rendering_worker, evaluation_worker,
                                             StageJournal, resource_gate, run, profile_bounds,
                                             iter_rendered_frames, resolve_render_device,
                                             MAX_ESTIMATED_WORKING_BYTES)
from real_video.replay_fit import replay


def fixture():
    coeff = torch.zeros(9, 2, 8, 8)
    coeff[4, 0] = 2
    return dict(kind=KIND, geometry='2D planar; no inferred depth', generated_motion=False,
                coefficients=coeff, ids=torch.arange(64), frames=4, grid=8, terms=2,
                size=32, radius=5, position_limit=None, fps=2., source_sha256='a'*64,
                edits=[], content_box=[0, 0, 32, 32])


class GaussianEditWorkerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_unsupported_claims_fail_closed(self):
        for key, value in [('kind', '3D'), ('geometry', '3D'), ('generated_motion', True)]:
            packet = fixture()
            packet[key] = value
            with self.assertRaises(ValueError):
                validate_packet(packet)

    def test_bad_coefficients_and_ids_rejected(self):
        for corruption in ('nan', 'shape', 'ids'):
            packet = fixture()
            if corruption == 'nan': packet['coefficients'][0, 0, 0, 0] = float('nan')
            if corruption == 'shape': packet['coefficients'] = packet['coefficients'][:8]
            if corruption == 'ids': packet['ids'][1] = 0
            with self.assertRaises(ValueError): validate_packet(packet)

    def test_detail_profile_explicit_and_bounded(self):
        estimate = profile_bounds(512, 256, 64, 64, 'detail')
        self.assertLessEqual(estimate, MAX_ESTIMATED_WORKING_BYTES)
        for values in ((512, 256, 64, 64, 'pilot'), (513, 256, 64, 64, 'detail'),
                       (512, 257, 64, 64, 'detail'), (512, 256, 65, 64, 'detail'),
                       (512, 256, 64, 65, 'detail'), (512, 256, 64, 0, 'detail'),
                       (512, 256, 64, 64, 'unbounded'), (True, 8, 4, 2, 'pilot')):
            with self.subTest(values=values), self.assertRaises(ValueError):
                profile_bounds(*values)
        with self.assertRaises(ValueError):
            profile_bounds(512, 256, 64, 64, 'detail', element_size=8)

    def test_packet_profile_scale_floor_and_legacy(self):
        packet = fixture()
        validate_packet(packet)  # Existing packet has neither new key.
        packet['size'] = 256
        packet['content_box'] = [0, 0, 256, 256]
        with self.assertRaises(ValueError): validate_packet(packet)
        packet.update(profile='detail', min_log_scale=-3.)
        validate_packet(packet)
        for value in (-2., float('nan'), -20., True):
            packet['min_log_scale'] = value
            with self.assertRaises(ValueError): validate_packet(packet)

    def test_identity_edit_does_not_mutate(self):
        packet = fixture()
        before = copy.deepcopy(packet)
        result = appearance_worker(packet)
        self.assertTrue(torch.equal(packet['coefficients'], before['coefficients']))
        self.assertTrue(torch.equal(result['coefficients'], packet['coefficients']))
        self.assertTrue(torch.equal(result['ids'], packet['ids']))
        self.assertEqual(packet['edits'], [])

    def test_invalid_content_boxes_fail_before_evaluation(self):
        output = torch.zeros(4, 3, 32, 32)
        reference = np.zeros((4, 32, 32, 3), np.uint8)
        invalid = (None, [], [0, 0, 32], [0, 0, 0, 32],
                   [0, 0, 32, -1], [-1, 0, 32, 32], [0, 0, 33, 32],
                   [1, 1, 32, 32], [32, 0, 1, 32], [0, 32, 32, 1],
                   [0., 0, 32, 32], [False, 0, 32, 32])
        for box in invalid:
            with self.subTest(box=box):
                packet = fixture()
                packet['content_box'] = box
                with self.assertRaises(ValueError):
                    evaluation_worker(packet, output, reference)

    def test_valid_letterbox_evaluation_ignores_padding(self):
        packet = fixture()
        packet['content_box'] = [0, 8, 32, 16]
        output = torch.ones(4, 3, 32, 32)
        output[:, :, 8:24] = 0
        report = evaluation_worker(packet, output, np.zeros((4, 32, 32, 3), np.uint8))
        self.assertEqual(report['content_psnr_db'], 120.)

    def test_invalid_reference_rejected_without_nan_metrics(self):
        packet = fixture()
        output = torch.zeros(4, 3, 32, 32)
        for reference in (np.zeros((4, 32, 32, 3), np.float32),
                          np.full((4, 32, 32, 3), np.nan, np.float32),
                          np.zeros((4, 32, 32, 3), np.int16),
                          np.zeros((4, 32, 31, 3), np.uint8), None):
            with self.assertRaises(ValueError):
                evaluation_worker(packet, output, reference)

    def test_malformed_source_identity_rejected(self):
        for source_hash in (None, 123, ['a'] * 64, 'g' * 64, 'a' * 63, 'A' * 64):
            with self.subTest(source_hash=source_hash):
                packet = fixture()
                packet['source_sha256'] = source_hash
                with self.assertRaises(ValueError):
                    validate_packet(packet)

    def test_color_transform_exact_and_geometry_unchanged(self):
        packet = fixture()
        result = appearance_worker(packet, rgb_gain=(1.2, 1., .8))
        self.assertTrue(torch.equal(result['coefficients'][:6], packet['coefficients'][:6]))
        torch.testing.assert_close(result['coefficients'][6:9, 0, 0, 0], torch.tensor([.4, 0., -.4]))
        self.assertEqual(result['source_sha256'], packet['source_sha256'])

    def test_invalid_edit_rejected(self):
        for gains in ((float('nan'), 1, 1), (2, 1, 1), (1, 1)):
            with self.assertRaises(ValueError): appearance_worker(fixture(), rgb_gain=gains)

    def test_replay_has_no_source_io(self):
        with patch('cv2.VideoCapture', side_effect=AssertionError('source accessed')):
            result = rendering_worker(fixture())
        self.assertEqual(tuple(result.shape), (4, 3, 32, 32))
        self.assertTrue(torch.isfinite(result).all())

    def test_streamed_frames_match_legacy_exactly_without_source(self):
        packet = fixture()
        generator = torch.Generator().manual_seed(6731)
        packet['coefficients'] = torch.randn((9, 2, 8, 8), generator=generator) * .1
        expected = replay(packet, device='cpu', chunk=1)
        with patch('cv2.VideoCapture', side_effect=AssertionError('source accessed')):
            actual = list(iter_rendered_frames(packet, [3, 0, 2]))
        self.assertEqual([i for i, _ in actual], [3, 0, 2])
        for index, frame in actual:
            torch.testing.assert_close(frame, expected[index], rtol=0, atol=0)
        torch.testing.assert_close(rendering_worker(packet), expected, rtol=0, atol=0)

    def test_explicit_cpu_render_keeps_legacy_packet_and_avoids_cuda(self):
        packet = fixture()
        before = copy.deepcopy(packet)
        with patch('torch.cuda.is_available', side_effect=AssertionError('CPU path checked CUDA')):
            explicit = rendering_worker(packet, device=torch.device('cpu'))
            default = rendering_worker(packet)
        torch.testing.assert_close(explicit, default, rtol=0, atol=0)
        self.assertEqual(explicit.device.type, 'cpu')
        self.assertEqual(packet.keys(), before.keys())
        torch.testing.assert_close(packet['coefficients'], before['coefficients'], rtol=0, atol=0)

    def test_render_device_validation(self):
        for device in (None, 0, True, [], '', 'automatic', 'mps', 'meta', 'cpu:0', 'cuda:-1'):
            with self.subTest(device=device), self.assertRaises(ValueError):
                list(iter_rendered_frames(fixture(), [0], device=device))

    def test_unavailable_cuda_fails_closed_before_render(self):
        with patch('torch.cuda.is_available', return_value=False), \
                patch('real_video.gaussian_edit_workers.render_fields',
                      side_effect=AssertionError('render must not start')):
            with self.assertRaisesRegex(RuntimeError, 'CPU fallback is disabled'):
                rendering_worker(fixture(), device='cuda')

    def test_cuda_index_validation_without_allocating(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=1):
            self.assertEqual(resolve_render_device('cuda:0'), torch.device('cuda:0'))
            with self.assertRaisesRegex(RuntimeError, 'does not exist'):
                resolve_render_device('cuda:1')

    @unittest.skipUnless(torch.cuda.is_available(), 'Optional CUDA parity requires a CUDA device')
    def test_tiny_cuda_parity_streams_cpu_frames_without_mutating_packet(self):
        packet = fixture()
        generator = torch.Generator().manual_seed(6731)
        packet['coefficients'] = torch.randn((9, 2, 8, 8), generator=generator) * .1
        before = packet['coefficients'].clone()
        expected = list(iter_rendered_frames(packet, [3, 0], device='cpu'))
        with patch('cv2.VideoCapture', side_effect=AssertionError('source accessed')):
            actual = list(iter_rendered_frames(packet, [3, 0], device='cuda'))
        self.assertEqual([index for index, _ in actual], [3, 0])
        for (_, frame), (_, reference) in zip(actual, expected):
            self.assertEqual(frame.device.type, 'cpu')
            self.assertTrue(torch.isfinite(frame).all())
            torch.testing.assert_close(frame, reference, rtol=1e-5, atol=1e-5)
        self.assertEqual(packet['coefficients'].device.type, 'cpu')
        torch.testing.assert_close(packet['coefficients'], before, rtol=0, atol=0)

    def test_streamed_frame_selection_rejects_invalid_requests(self):
        for indices in ([], [4], [-1], [0.0], [False], [0]*5):
            with self.assertRaises(ValueError):
                list(iter_rendered_frames(fixture(), indices))

    def test_detail_iterator_honors_recorded_scale_floor(self):
        packet = fixture()
        packet.update(profile='detail', min_log_scale=-3.)
        with patch('real_video.gaussian_edit_workers.render_fields',
                   return_value=torch.zeros(1, 1, 3, 32, 32)) as render:
            result = list(iter_rendered_frames(packet, [1]))
        self.assertEqual(len(result), 1)
        self.assertEqual(render.call_args.kwargs['min_log_scale'], -3.)

    def test_quality_never_auto_accepts(self):
        packet = fixture()
        output = rendering_worker(packet)
        report = evaluation_worker(packet, output, np.zeros((4, 32, 32, 3), np.uint8))
        self.assertFalse(report['autonomous_generation_accepted'])
        self.assertFalse(report['geometry_3d_accepted'])
        self.assertIn('pending', report['visual_acceptance'])

    def test_source_letterbox_duration_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 4., (64, 32))
            for _ in range(8): writer.write(np.full((32, 64, 3), 128, np.uint8))
            writer.release()
            packet, images = source_worker(path, size=32, grid=8, frames=4, terms=2)
            self.assertEqual(packet['content_box'], [0, 8, 32, 16])
            self.assertEqual(packet['frames']/packet['fps'], 2.)
            self.assertTrue((images[:, :8] == 0).all())
            self.assertEqual(len(packet['source_sha256']), 64)
            detail, detail_images = source_worker(path, size=224, grid=8, frames=4, terms=2, profile='detail')
            self.assertEqual(detail['source_indices'], packet['source_indices'])
            self.assertEqual(detail['min_log_scale'], -3.)
            self.assertEqual(detail_images.shape, (4, 224, 224, 3))

    def test_resource_gate_rejects_low_ram(self):
        from types import SimpleNamespace
        with patch('psutil.virtual_memory', return_value=SimpleNamespace(available=1024)):
            with self.assertRaises(RuntimeError): resource_gate()

    def test_stage_failure_preserves_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'stages.jsonl'
            journal = StageJournal(path)
            with self.assertRaises(ValueError):
                with journal.stage('appearance'):
                    raise ValueError('sensitive detail is not logged')
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['status'] for r in records], ['begin', 'failed'])
            self.assertEqual(records[-1]['error_type'], 'ValueError')
            self.assertNotIn('sensitive', path.read_text())

    def test_end_to_end_both_videos_and_hash_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 4., (64,32))
            for _ in range(4): writer.write(np.full((32,64,3), 128, np.uint8))
            writer.release()
            with patch('real_video.gaussian_edit_workers.resource_gate', return_value=9*1024**3):
                result = run(source, root/'out', size=32, grid=8, frames=4, terms=4, rgb_gain=(1.1,1.,.9))
            self.assertTrue(result['edit_rendered'])
            self.assertGreater((root/'out'/'gaussian_edited.mp4').stat().st_size, 0)
            records = [json.loads(line) for line in (root/'out'/'stages.jsonl').read_text().splitlines()]
            completed = [r for r in records if r['status']=='complete']
            self.assertEqual(len(completed), 8)
            self.assertEqual(len(completed[-1]['artifacts'][0]['sha256']), 64)


if __name__ == '__main__': unittest.main()
