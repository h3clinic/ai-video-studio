"""CPU first-frame memory and native TI2V protocol algebra, not video quality."""

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from real_video.checkpoint_io import load_verified, save_inference_checkpoint
from real_video.gaussian_latent_memory import GaussianLatentMemory
from real_video.wan22_gaussian_anchor import (
    apply_ti2v_anchor, build_gaussian_anchor, recall_gaussian_anchor,
    ti2v_denoising_inputs, ti2v_first_frame_mask,
)


DIGEST = 'c'*64


class Wan22GaussianAnchorTests(unittest.TestCase):
    def test_native_48_channels_default_dense_coverage_and_error_reporting(self):
        source = torch.randn(48, 4, 6, generator=torch.Generator().manual_seed(7))
        recalled, metrics, bank = build_gaussian_anchor(source, DIGEST)
        self.assertEqual(recalled.shape, source.shape)
        self.assertEqual(bank['features'].shape, (24, 48))
        self.assertEqual(metrics['gaussian_count'], 24)
        self.assertGreaterEqual(metrics['coverage_min'], .99-1e-6)
        self.assertLessEqual(metrics['coverage_max'], 1.000001)
        self.assertLess(metrics['relative_latent_rmse'], .15)
        self.assertGreater(metrics['relative_latent_rmse'], 0.)
        self.assertGreater(metrics['recalled_to_source_rms_ratio'], .85)
        self.assertFalse(metrics['coverage_normalized'])
        json.dumps(metrics, allow_nan=False)

    def test_real_memory_writes_once_and_reader_restores_independently(self):
        source = torch.arange(72).float().reshape(3, 4, 6)
        seen = []
        original = GaussianLatentMemory.write
        def write(memory, latent, cache, **kwargs):
            seen.append(latent.clone())
            return original(memory, latent, cache, **kwargs)
        with patch.object(GaussianLatentMemory, 'write', write):
            recalled, _, bank = build_gaussian_anchor(source, DIGEST)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'bank.pt'
                save_inference_checkpoint(bank, path)
                restored, coverage = recall_gaussian_anchor(load_verified(path))
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], source))
        self.assertTrue(torch.equal(recalled, restored))
        self.assertEqual(coverage.shape, (4, 6))

    def test_constant_feature_read_is_coverage_weighted_not_normalized(self):
        recalled, _, bank = build_gaussian_anchor(torch.full((48, 3, 4), 2.), DIGEST)
        _, coverage = recall_gaussian_anchor(bank)
        torch.testing.assert_close(recalled, 2*coverage[None].expand_as(recalled))
        self.assertFalse(torch.equal(recalled, torch.full_like(recalled, 2.)))

    def test_only_first_frame_api_and_snapshot_without_source_history(self):
        self.assertEqual(set(inspect.signature(build_gaussian_anchor).parameters), {'anchor', 'source_image_digest', 'sigma', 'ids'})
        self.assertEqual(set(inspect.signature(recall_gaussian_anchor).parameters), {'snapshot'})
        source = torch.ones(2, 2, 2)
        _, _, bank = build_gaussian_anchor(source, DIGEST)
        self.assertFalse(any(key in bank for key in ('anchor', 'rgb', 'tracks', 'future', 'targets', 'frame_history')))
        self.assertEqual(bank['writes'], 1)
        self.assertEqual(bank['canonical_first_centers'].shape, (4, 2))
        source.zero_()
        self.assertGreater(float(recall_gaussian_anchor(bank)[0].sum()), 0.)

    def test_arbitrary_channels_and_one_pixel_zero_anchor(self):
        for channels in (1, 16, 48):
            recalled, metrics, _ = build_gaussian_anchor(torch.zeros(channels, 1, 1), DIGEST)
            self.assertEqual(recalled.count_nonzero(), 0)
            self.assertEqual(metrics['relative_latent_rmse'], 0.)
            json.dumps(metrics, allow_nan=False)

    def test_fixed_custom_ids_and_detached_inputs(self):
        source = torch.ones(2, 2, 2, requires_grad=True)
        ids = torch.tensor([50, 9, -4, 22])
        recalled, _, bank = build_gaussian_anchor(source, DIGEST, ids=ids)
        self.assertTrue(torch.equal(ids, bank['ids']))
        self.assertFalse(recalled.requires_grad)
        bank['ids'][0] = 0
        self.assertEqual(ids[0], 50)
        with self.assertRaises(ValueError):
            recall_gaussian_anchor(bank)

    def test_fp32_rounding_and_bfloat16_errors_measured(self):
        source = torch.tensor([[[1.0000000001, 1.123456789123]]], dtype=torch.float64)
        _, metrics, _ = build_gaussian_anchor(source, DIGEST)
        self.assertGreater(metrics['fp32_cast_rmse'], 0.)
        self.assertGreater(metrics['direct_bf16_cast_rmse'], 0.)
        self.assertGreater(metrics['paired_bf16_latent_rmse'], 0.)

    def test_bad_anchor_identity_and_sigma_rejected(self):
        cases = [torch.ones(1, 1, 2, 2), torch.ones(2, 2, 2).long(),
                 torch.zeros(0, 2, 2), torch.full((1, 2, 2), float('nan'))]
        for source in cases:
            with self.assertRaises(ValueError):
                build_gaussian_anchor(source, DIGEST)
        for kwargs in ({'ids':torch.arange(3)}, {'ids':torch.zeros(4, dtype=torch.long)}, {'sigma':0}):
            with self.assertRaises(ValueError):
                build_gaussian_anchor(torch.ones(1, 2, 2), DIGEST, **kwargs)

    def test_modified_canonical_geometry_or_protocol_rejected(self):
        for key, value in [('opacity', .5), ('source_image_digest', 'a'*64), ('writes', 2),
                           ('coordinate_convention', '3D'), ('sigma', float('nan'))]:
            _, _, bank = build_gaussian_anchor(torch.ones(1, 2, 2), DIGEST)
            bank[key] = value
            with self.assertRaises(ValueError):
                recall_gaussian_anchor(bank)
        _, _, bank = build_gaussian_anchor(torch.ones(1, 2, 2), DIGEST)
        bank['canonical_first_centers'][0, 0] += 1
        with self.assertRaises(ValueError):
            recall_gaussian_anchor(bank)

    def test_native_mask_and_model_input_do_not_copy_future_anchor(self):
        latents = torch.arange(2*48*3*4*6).float().reshape(2, 48, 3, 4, 6)
        anchor = torch.full((48, 4, 6), -9.)
        original = latents.clone()
        conditioned, tokens, mask = ti2v_denoising_inputs(latents, anchor, torch.tensor(700))
        self.assertEqual(mask.shape, (1, 1, 3, 4, 6))
        self.assertEqual(tokens.shape, (2, 18))
        self.assertTrue(torch.equal(tokens[:, :6], torch.zeros(2, 6)))
        self.assertTrue(torch.equal(tokens[:, 6:], torch.full((2, 12), 700.)))
        self.assertTrue(torch.equal(conditioned[:, :, 0], anchor[None].expand(2, -1, -1, -1)))
        self.assertTrue(torch.equal(conditioned[:, :, 1:], latents[:, :, 1:]))
        self.assertTrue(torch.equal(latents, original))

    def test_ti2v_protocol_matches_inspected_pipeline_expression(self):
        latents = torch.randn(1, 48, 5, 4, 6)
        anchor = torch.randn(1, 48, 1, 4, 6)
        result, actual_time, actual_mask = ti2v_denoising_inputs(latents, anchor, 321.)
        mask = torch.ones(1, 1, 5, 4, 6)
        mask[:, :, 0] = 0
        self.assertTrue(torch.equal(actual_mask, mask))
        self.assertTrue(torch.equal(result, (1-mask)*anchor + mask*latents))
        self.assertTrue(torch.equal(actual_time, (mask[0, 0, :, ::2, ::2]*321.).flatten()[None]))

    def test_future_frames_and_incompatible_protocol_fail_closed(self):
        latent = torch.zeros(1, 48, 5, 4, 6)
        with self.assertRaises(ValueError):
            apply_ti2v_anchor(latent, torch.ones_like(latent))
        for time in (-1, float('nan'), torch.tensor([1, 2])):
            with self.assertRaises(ValueError):
                ti2v_denoising_inputs(latent, torch.ones(48, 4, 6), time)
        with self.assertRaises(ValueError):
            ti2v_denoising_inputs(latent, torch.ones(48, 4, 6), 1, patch_size=(1, 4, 4))
        with self.assertRaises(ValueError):
            ti2v_denoising_inputs(torch.ones(1, 48, 5, 3, 6), torch.ones(48, 3, 6), 1)


if __name__ == '__main__':
    unittest.main()
