"""CPU-only manifest/profile regressions; never load real model weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from real_video.prepare_wan22 import REPO, REVISION
from real_video.sample_wan22_memory import check_manifest, validate_size, gpu_memory_profile, sampling_schedule


class Wan22RunnerTests(unittest.TestCase):
    def test_sampling_schedule_preserves_native_default_and_labels_overrides(self):
        native = sampling_schedule({'flow_shift':5.})
        self.assertEqual(native['flow_shift'],5.)
        self.assertEqual(native['source'],'downloaded_scheduler')
        self.assertFalse(native['quality_validated'])
        explicit = sampling_schedule({'flow_shift':5.},3.)
        self.assertEqual(explicit['flow_shift'],3.)
        self.assertEqual(explicit['downloaded_flow_shift'],5.)
        self.assertEqual(explicit['source'],'explicit_experimental_override')
        for value in (None, True, 0., 11., float('nan'), float('inf'), '5'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): sampling_schedule({'flow_shift':value})
        for value in (False, float('nan'), 0.):
            with self.assertRaises(ValueError): sampling_schedule({'flow_shift':5.},value)

    def test_explicit_small_profile_is_not_a_general_gpu_guard_reduction(self):
        small = gpu_memory_profile(256, 448, 33)
        self.assertTrue(small['lower_resolution_feasibility'])
        self.assertEqual(small['patch_tokens'], 1008)
        self.assertEqual(small['required_free_cuda_bytes'], 10021521792+2**30)
        for shape in ((480,832,49), (256,448,49), (288,512,33), (256,448,17)):
            profile = gpu_memory_profile(*shape)
            self.assertFalse(profile['lower_resolution_feasibility'])
            self.assertEqual(profile['required_free_cuda_bytes'], 11*2**30)
        self.assertEqual(gpu_memory_profile(480,832,49)['patch_tokens'], 5070)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'model'
        self.root.mkdir()
        self.files = {
            'model_index.json': {'_class_name': 'WanPipeline'},
            'transformer/config.json': {'_class_name': 'WanTransformer3DModel', 'in_channels': 48,
                'out_channels': 48, 'patch_size': [1, 2, 2], 'text_dim': 4096, 'image_dim': None},
            'transformer/diffusion_pytorch_model.safetensors.index.json': {
                'weight_map': {'weight': 'diffusion_pytorch_model-00001-of-00001.safetensors'}},
            'transformer/diffusion_pytorch_model-00001-of-00001.safetensors': b'fake transformer weights',
            'vae/config.json': {'_class_name': 'AutoencoderKLWan', 'z_dim': 48,
                'scale_factor_spatial': 16, 'scale_factor_temporal': 4,
                'latents_mean': [0.]*48, 'latents_std': [1.]*48},
            'vae/diffusion_pytorch_model.safetensors': b'fake vae weights',
            'scheduler/scheduler_config.json': {'_class_name': 'UniPCMultistepScheduler',
                'prediction_type': 'flow_prediction', 'use_flow_sigmas': True, 'use_dynamic_shifting': False},
        }
        self.manifest = dict(repo=REPO, revision=REVISION, license='apache-2.0', status='verified')
        self.rewrite()

    def rewrite(self):
        rows = []
        for name, value in self.files.items():
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            path.write_bytes(data)
            checksum = hashlib.sha256(data).hexdigest()
            rows.append(dict(path=name, size=len(data), sha256=checksum,
                remote_lfs_sha256=checksum if name.endswith('.safetensors') else None))
        self.manifest.update(files=rows, download_bytes=sum(row['size'] for row in rows))
        self.write_manifest()

    def write_manifest(self):
        (self.root/'download_manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_complete_native_manifest_and_default_t2v_index_are_allowed(self):
        self.assertEqual(check_manifest(self.root)['revision'], REVISION)

    def test_unfinished_download_is_not_a_runtime_failure(self):
        self.manifest['status'] = 'downloading'; self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Complete pinned'):
            check_manifest(self.root)

    def test_wrong_repo_revision_and_license_fail(self):
        for key, value in (('repo', 'someone/other'), ('revision', '0'*40), ('license', 'unknown')):
            with self.subTest(key=key):
                original = self.manifest[key]
                self.manifest[key] = value; self.write_manifest()
                with self.assertRaises(ValueError): check_manifest(self.root)
                self.manifest[key] = original

    def test_same_size_weight_tampering_is_detected(self):
        target = self.root/'vae/diffusion_pytorch_model.safetensors'
        target.write_bytes(b'x'*target.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            check_manifest(self.root)

    def test_official_lfs_hash_is_checked_independently(self):
        row = next(row for row in self.manifest['files'] if row['path'].endswith('.safetensors'))
        row['remote_lfs_sha256'] = '0'*64; self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'LFS'): check_manifest(self.root)

    def test_declared_download_cannot_omit_completed_files(self):
        self.manifest['files'].pop(); self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'complete declared download'):
            check_manifest(self.root)

    def test_missing_weight_index_cannot_be_forged_as_complete(self):
        del self.files['transformer/diffusion_pytorch_model.safetensors.index.json']; self.rewrite()
        with self.assertRaisesRegex(ValueError, 'weights/index missing'):
            check_manifest(self.root)

    def test_missing_shard_and_unsafe_shard_are_rejected(self):
        index = self.files['transformer/diffusion_pytorch_model.safetensors.index.json']
        for name in ('missing.safetensors', '../outside.safetensors'):
            with self.subTest(name=name):
                index['weight_map']['weight'] = name; self.rewrite()
                with self.assertRaisesRegex(ValueError, 'shard'): check_manifest(self.root)

    def test_manifest_paths_cannot_escape_or_be_duplicated(self):
        original = self.manifest['files'][0]['path']
        for name in ('../outside.json', '/absolute.json', 'C:/outside.json', 'a\\b.json', './model_index.json'):
            with self.subTest(name=name):
                self.manifest['files'][0]['path'] = name; self.write_manifest()
                with self.assertRaisesRegex(ValueError, 'path'): check_manifest(self.root)
        self.manifest['files'][0]['path'] = original
        self.manifest['files'].append(self.manifest['files'][0].copy()); self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Duplicate'): check_manifest(self.root)

    def test_vae_native_shape_and_normalization_are_validated(self):
        vae = self.files['vae/config.json']
        for key, bad in (('z_dim', 16), ('scale_factor_spatial', 8), ('scale_factor_temporal', 8),
                         ('latents_mean', [0.]*16), ('latents_std', [0.]*48)):
            with self.subTest(key=key):
                original = vae[key]; vae[key] = bad; self.rewrite()
                with self.assertRaises(ValueError): check_manifest(self.root)
                vae[key] = original

    def test_scheduler_and_transformer_mismatch_fail(self):
        self.files['scheduler/scheduler_config.json']['use_flow_sigmas'] = False; self.rewrite()
        with self.assertRaisesRegex(ValueError, 'scheduler'): check_manifest(self.root)
        self.files['scheduler/scheduler_config.json']['use_flow_sigmas'] = True
        self.files['transformer/config.json']['in_channels'] = 16; self.rewrite()
        with self.assertRaisesRegex(ValueError, 'architecture'): check_manifest(self.root)

    def test_bounded_resolution_and_clock_profile(self):
        validate_size(256, 448, 33, 20)
        validate_size(480, 832, 49, 30)
        validate_size(256, 448, 33, 50)
        for values in ((256, 448, 32, 20), (480, 832, 81, 30), (720, 1280, 49, 30),
                       (256, 448, 33, 19), (256, 448, 33, 51), (256., 448, 33, 30), (True, 448, 33, 30)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError): validate_size(*values)


if __name__ == '__main__':
    unittest.main()
