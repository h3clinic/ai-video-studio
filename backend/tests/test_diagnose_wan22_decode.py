"""CPU protocol tests with a fake VAE; not evidence of video quality."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from real_video.diagnose_wan22_decode import (
    comparison_metrics, decode_normalized, encode_observed, latent_stats, main, run,
)


class FakeVAE:
    def __init__(self):
        self.config = SimpleNamespace(latents_mean=[.25]*48,latents_std=[2.]*48)
        self.calls = []
        self.fail = False

    def clear_cache(self): self.calls.append('clear')
    def enable_tiling(self): self.calls.append('tiled')
    def disable_tiling(self): self.calls.append('untiled')

    def decode(self,value,return_dict):
        self.calls.append('decode')
        self.decode_input = value.clone()
        if self.fail: raise RuntimeError('test failure')
        return (torch.full((1,3,9,256,448),.1),)

    def encode(self,value):
        self.calls.append('encode')
        self.encode_input = value.clone()
        if self.fail: raise RuntimeError('test failure')
        return SimpleNamespace(latent_dist=SimpleNamespace(
            mode=lambda:torch.full((1,48,3,16,28),1.25)))


class DecodeDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.old_threads)

    def test_pair_uses_exact_same_latents_inverse_normalization_and_clean_cache(self):
        vae = FakeVAE()
        latent = torch.arange(48*3*16*28).reshape(1,48,3,16,28).float()/10000
        original = latent.clone()
        decode_normalized(vae,latent,tiled=True)
        first = vae.decode_input.clone()
        decode_normalized(vae,latent,tiled=False)
        self.assertTrue(torch.equal(first,latent*2+.25))
        self.assertTrue(torch.equal(first,vae.decode_input))
        self.assertTrue(torch.equal(original,latent))
        self.assertEqual(vae.calls,['tiled','clear','decode','clear','untiled','clear','decode','clear'])

    def test_decode_failure_also_clears_cache(self):
        vae = FakeVAE(); vae.fail = True
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            decode_normalized(vae,torch.zeros(1,48,3,16,28),tiled=False)
        self.assertEqual(vae.calls[-1],'clear')

    def test_wrong_latent_clock_shape_and_nonfinite_fail_before_vae(self):
        for shape in ((1,48,1,16,28),(1,16,3,16,28),(1,48,3,28,16)):
            with self.assertRaises(ValueError): decode_normalized(FakeVAE(),torch.zeros(shape),tiled=True)
        bad = torch.zeros(1,48,3,16,28); bad[0,0,0,0,0] = float('nan')
        with self.assertRaises(ValueError): decode_normalized(FakeVAE(),bad,tiled=True)

    def test_observed_control_passes_all_real_frames_once_not_repeated_anchor(self):
        vae = FakeVAE()
        observed = torch.linspace(-1,1,9)[None,None,:,None,None].expand(1,3,9,256,448)
        normalized = encode_observed(vae,observed)
        self.assertTrue(torch.equal(vae.encode_input,observed))
        self.assertTrue(torch.equal(normalized,torch.full((1,48,3,16,28),.5)))
        self.assertFalse(torch.equal(vae.encode_input[:,:,0],vae.encode_input[:,:,8]))
        self.assertEqual(vae.calls,['untiled','clear','encode','clear'])

    def test_encoder_failure_clears_cache(self):
        vae = FakeVAE(); vae.fail = True
        with self.assertRaises(RuntimeError): encode_observed(vae,torch.zeros(1,3,9,256,448))
        self.assertEqual(vae.calls[-1],'clear')

    def test_invalid_rgb_and_normalization_rejected(self):
        with self.assertRaises(ValueError): encode_observed(FakeVAE(),torch.full((1,3,9,256,448),2.))
        vae = FakeVAE(); vae.config.latents_std[0] = 0
        with self.assertRaises(ValueError): decode_normalized(vae,torch.zeros(1,48,3,16,28),tiled=True)

    def test_metrics_keep_temporal_order_and_zero_difference_json_safe(self):
        x = torch.arange(3).float().reshape(1,1,3,1,1)
        self.assertEqual([v['rms'] for v in latent_stats(x)['per_frame']],[0.,1.,2.])
        rgb = torch.zeros(1,3,9,256,448)
        metrics = comparison_metrics(rgb,rgb)
        self.assertEqual(metrics['rmse'],0.)
        self.assertTrue(all(frame['exact'] and frame['psnr_db'] is None for frame in metrics['per_frame']))
        other = rgb.clone(); other[:,:,5] = 1
        self.assertEqual(comparison_metrics(rgb,other)['per_frame'][5]['mse'],1.)

    def test_existing_output_rejected_before_loading_or_cuda(self):
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.mem_get_info') as gpu:
            with self.assertRaises(FileExistsError): run(Path(directory))
            gpu.assert_not_called()

    def test_cli_only_launches_guarded_explicit_worker(self):
        with patch('real_video.guarded_worker.run_guarded_worker',return_value='guarded') as guard:
            result = main(['--out','artifacts/diagnostic_test','--wall-seconds','120'])
        self.assertEqual(result,'guarded')
        args,kwargs = guard.call_args
        self.assertEqual(args[0],'real_video.diagnose_wan22_decode')
        self.assertIn('--worker',args[1])
        self.assertEqual(kwargs,dict(wall_seconds=120,resource='gpu'))

    def test_internal_worker_cannot_skip_live_parent_check(self):
        with patch('real_video.guarded_worker.require_supervised_child',side_effect=RuntimeError('not supervised')):
            with patch('real_video.diagnose_wan22_decode.run') as worker:
                with self.assertRaisesRegex(RuntimeError,'not supervised'):
                    main(['--out','artifacts/diagnostic_test','--worker'])
                worker.assert_not_called()


if __name__ == '__main__': unittest.main()
