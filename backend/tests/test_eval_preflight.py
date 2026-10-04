"""Offline CPU admission/schema tests; no pretrained loading or CUDA work."""
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from cloud import eval_preflight as preflight


class EvalPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cg = self.root/'cgroup'
        self.cg.mkdir()
        self.membership = self.root/'membership'
        self.membership.write_text('0::/worker\n', encoding='utf-8')

    def pair(self, path, limit, current, v1=False):
        path.mkdir(parents=True, exist_ok=True)
        names = ('memory.limit_in_bytes', 'memory.usage_in_bytes') if v1 else ('memory.max', 'memory.current')
        for name, value in zip(names, (limit, current)):
            (path/name).write_text(str(value), encoding='utf-8')

    def budget(self, host=500*2**30):
        return preflight.effective_memory(host, cgroup_root=self.cg, membership_path=self.membership,
                                         mountinfo_path=self.root/'mountinfo')

    def test_custom_mount_root_resolves_descendant(self):
        custom = self.root/'custom-memory'
        self.membership.write_text('0::/host/worker\n', encoding='utf-8')
        self.pair(custom, 'max', 0)
        self.pair(custom/'worker', 220*2**30, 40*2**30)
        (self.root/'mountinfo').write_text(
            f'1 2 0:1 /host {custom} rw - cgroup2 cgroup rw\n', encoding='utf-8')
        self.assertEqual(self.budget()['effective_available_bytes'], 180*2**30)

    def test_v2_child_and_parent_limits_both_apply(self):
        self.pair(self.cg, 250*2**30, 100*2**30)
        self.pair(self.cg/'worker', 300*2**30, 10*2**30)
        self.assertEqual(self.budget()['effective_available_bytes'], 150*2**30)
        with self.assertRaisesRegex(RuntimeError, '192 GiB'):
            preflight.require_host_memory(500*2**30, cgroup_root=self.cg,
                membership_path=self.membership, mountinfo_path=self.root/'mountinfo')

    def test_host_available_is_upper_bound(self):
        self.pair(self.cg, 300*2**30, 10*2**30)
        self.assertEqual(self.budget(100*2**30)['effective_available_bytes'], 100*2**30)

    def test_unlimited_v2_and_v1(self):
        self.pair(self.cg, 'max', 123)
        self.assertEqual(self.budget()['effective_available_bytes'], 500*2**30)
        self.pair(self.cg/'memory', 9223372036854771712, 123, v1=True)
        self.assertEqual(self.budget()['effective_available_bytes'], 500*2**30)

    def test_v1_membership(self):
        self.membership.write_text('5:memory:/worker\n', encoding='utf-8')
        self.pair(self.cg/'memory', 400*2**30, 10*2**30, v1=True)
        self.pair(self.cg/'memory'/'worker', 220*2**30, 40*2**30, v1=True)
        self.assertEqual(self.budget()['effective_available_bytes'], 180*2**30)

    def test_missing_accounting_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'accounting unavailable'):
            self.budget()

    def test_missing_current_and_malformed_limit_fail_closed(self):
        (self.cg/'memory.max').write_text('1000', encoding='utf-8')
        with self.assertRaises(FileNotFoundError): self.budget()
        self.pair(self.cg, 'invalid', 2)
        with self.assertRaises(ValueError): self.budget()

    def test_negative_or_exhausted_limits(self):
        self.pair(self.cg, 5, 10)
        self.assertEqual(self.budget()['effective_available_bytes'], 0)
        self.pair(self.cg, -1, 10)
        with self.assertRaises(ValueError): self.budget()

    def test_nonlinux_no_cgroup_uses_host(self):
        report = preflight.effective_memory(100, cgroup_root=self.root/'absent',
            membership_path=self.root/'absent2', mountinfo_path=self.root/'mountinfo')
        self.assertEqual(report['effective_available_bytes'], 100)

    def adapter(self):
        # Shared scalar storage keeps fixtures tiny; every expected shape is real.
        zero = torch.zeros(1)
        return dict(repo=preflight.REPO, revision=preflight.REVISION,
                    expert='low_noise_transformer_2',
                    weights={n:zero.expand(s) for n,s in preflight.expected_adapter_shapes().items()})

    def test_adapter_expected_parameter_count(self):
        report = preflight.validate_adapter(self.adapter())
        self.assertEqual(report['tensors'], 646)
        self.assertEqual(report['parameters'], 14419072)

    def test_broadcastable_adapter_shape_rejected(self):
        saved = self.adapter()
        saved['weights']['blocks.0.attn1.to_q.down'] = torch.zeros(1, 5120)
        with self.assertRaisesRegex(ValueError, 'shape/dtype/value'):
            preflight.validate_adapter(saved)

    def test_wrong_backbone_and_missing_key_rejected(self):
        saved = self.adapter(); saved['revision'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'backbone'): preflight.validate_adapter(saved)
        saved = self.adapter(); saved['weights'].pop('blocks.0.attn1.to_q.down')
        with self.assertRaisesRegex(ValueError, 'layout'): preflight.validate_adapter(saved)

    def test_nonfinite_or_wrong_dtype_rejected(self):
        for value in (torch.full((4,5120), float('nan')), torch.zeros(4,5120,dtype=torch.float16)):
            saved = self.adapter(); saved['weights']['blocks.0.attn1.to_q.down'] = value
            with self.assertRaisesRegex(ValueError, 'shape/dtype/value'): preflight.validate_adapter(saved)

    def test_memory_rejects_wrong_anchor_and_grid(self):
        from real_video.gaussian_latent_memory import GaussianLatentMemory
        memory = GaussianLatentMemory(torch.arange(6240), 16, 'a'*64, generations=torch.zeros(6240,dtype=torch.long))
        snapshot = memory.snapshot(); snapshot['writes'] = 1
        self.assertEqual(preflight.validate_memory(snapshot, 'a'*64)['gaussians'], 6240)
        with self.assertRaisesRegex(ValueError, 'schema/anchor'): preflight.validate_memory(snapshot, 'b'*64)
        snapshot['features'] = torch.zeros(1,16)
        with self.assertRaisesRegex(ValueError, 'shape/dtype'): preflight.validate_memory(snapshot, 'a'*64)

    def test_asset_hash_checked_before_deserialization(self):
        (self.root/'anchor.png').write_bytes(b'not the saved anchor')
        with patch('real_video.checkpoint_io.load_verified') as load:
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): preflight.validate_assets(self.root)
            load.assert_not_called()

    def test_missing_sidecar_rejected_before_deserialization(self):
        hashes = {}
        for name in preflight.PILOT_HASHES:
            (self.root/name).write_bytes(b'fixture')
            hashes[name] = hashlib.sha256(b'fixture').hexdigest()
        with patch.object(preflight, 'PILOT_HASHES', hashes), patch('real_video.checkpoint_io.load_verified') as load:
            with self.assertRaises(FileNotFoundError): preflight.validate_assets(self.root)
            load.assert_not_called()

    def test_runner_preflight_before_model_loading(self):
        from cloud.evaluate_a14b import run
        code = inspect.getsource(run)
        self.assertLess(code.index('validate_assets(args.asset)'), code.index('validate_model(args.model)'))
        self.assertLess(code.index('require_host_memory('), code.index('from_pretrained('))
        self.assertIn('stage_end_process_rss_bytes', code)
        self.assertIn('retained_evaluation_rgb_bytes', code)
        self.assertLess(code.index("if branch=='gaussian_memory':"), code.index('memory=GaussianLatentMemory('))


if __name__ == '__main__':
    unittest.main()
