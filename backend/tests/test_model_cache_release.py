import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cloud import model_cache_release as release

PRODUCTION_IDENTITY = release._identity


class ModelCacheReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root/release.REVISION
        self.snapshot.mkdir(); (self.snapshot/'transformer').mkdir()
        self.contents = {'model_index.json':b'{}', 'transformer/model.safetensors':b'model bytes'}
        files = []
        for name, data in self.contents.items():
            (self.snapshot/name).write_bytes(data)
            files.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), lfs_verified=False))
        self.record = dict(repo_id=release.REPO, revision=release.REVISION,
            status='verified', local_path=str(self.snapshot), expected_bytes=sum(map(len,self.contents.values())), files=files)
        self.manifest = self.root/'download_manifest.json'
        self.pin = self.write_manifest()
        if os.name == 'nt':
            # This Linux-only implementation is exercised through mocked POSIX
            # syscalls on Windows. Python 3.12 Windows stat(path).st_ctime_ns
            # reports creation time, while fstat(fd).st_ctime_ns can report the
            # later write time (reproduced on 183/500 tiny fixtures). Provide a
            # deterministic synthetic Linux change-time for this emulation,
            # retaining device/inode/size/mtime. Production code is unchanged;
            # native Linux tests continue using actual stat/fstat timestamps.
            identity = patch.object(release, '_identity', side_effect=lambda info:
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mtime_ns))
            identity.start(); self.addCleanup(identity.stop)
        self.linux = patch.object(release.sys, 'platform', 'linux'); self.linux.start(); self.addCleanup(self.linux.stop)
        self.advice_patch = patch.object(release.os, 'posix_fadvise', create=True)
        self.advice = self.advice_patch.start(); self.addCleanup(self.advice_patch.stop)
        self.constant = patch.object(release.os, 'POSIX_FADV_DONTNEED', 4, create=True)
        self.constant.start(); self.addCleanup(self.constant.stop)
        self.sync_patch = patch.object(release.os, 'fsync'); self.sync = self.sync_patch.start(); self.addCleanup(self.sync_patch.stop)

    def write_manifest(self):
        data = json.dumps(self.record).encode()
        self.manifest.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def run_release(self):
        return release.release_model_cache(self.snapshot, self.manifest, self.pin)

    def test_valid_exact_files_advised_without_claiming_reclaimed_ram(self):
        report = self.run_release()
        self.assertEqual(report['verified_bytes'], 13)
        self.assertEqual(report['advised_bytes'], 13)
        self.assertIsNone(report['cache_reclaimed_bytes'])
        self.assertFalse(report['memory_admission_passed'])
        self.assertTrue(report['fresh_memory_measurement_required'])
        self.assertEqual(self.advice.call_count, 2); self.assertEqual(self.sync.call_count, 2)
        for call in self.advice.call_args_list:
            self.assertEqual(call.args[1:], (0, 0, 4))
        for name, data in self.contents.items(): self.assertEqual((self.snapshot/name).read_bytes(), data)

    def test_production_identity_includes_distinct_nanosecond_change_time(self):
        info = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4001, st_ctime_ns=5002)
        self.assertEqual(PRODUCTION_IDENTITY(info), (1,2,3,4001,5002))
        info.st_ctime_ns += 1
        self.assertEqual(PRODUCTION_IDENTITY(info), (1,2,3,4001,5003))

    def test_unverified_or_wrong_repo_and_revision_rejected(self):
        for key, bad in (('status','verifying'), ('repo_id','other'), ('revision','other')):
            old = self.record[key]; self.record[key] = bad; self.pin = self.write_manifest()
            with self.assertRaises(ValueError): self.run_release()
            self.record[key] = old
        self.advice.assert_not_called()

    def test_wrong_manifest_pin_rejected(self):
        self.pin = '0'*64
        with self.assertRaisesRegex(ValueError, 'pin mismatch'): self.run_release()
        self.advice.assert_not_called()

    def test_unsafe_paths_and_duplicates_rejected_before_advice(self):
        for name in ('../outside', '/outside', 'transformer/../outside', 'unrelated/a', 'model_index.json'):
            self.record['files'][1]['path'] = name; self.pin = self.write_manifest()
            with self.assertRaises(ValueError): self.run_release()
        self.advice.assert_not_called()

    def test_size_mismatch_prevents_all_advice(self):
        (self.snapshot/'transformer/model.safetensors').write_bytes(b'wrong size')
        with self.assertRaisesRegex(ValueError, 'type/size'): self.run_release()
        self.advice.assert_not_called()

    def test_same_size_bad_hash_prevents_advising_bad_file(self):
        self.record['files'][0]['sha256'] = '0'*64
        self.pin = self.write_manifest()
        # Isolate the digest guard from Windows' differing path/FD timestamps;
        # production is Linux-only and its identity guard has a separate test.
        with patch.object(release, '_identity', return_value=(1,2,3,4,5)):
            with self.assertRaisesRegex(ValueError, 'SHA-256/identity'): self.run_release()
        self.advice.assert_not_called(); self.sync.assert_not_called()

    def test_fd_identity_changed_before_read_rejected(self):
        with patch.object(release, '_identity', side_effect=[(1,), (2,), (3,)]):
            with self.assertRaisesRegex(ValueError, 'changed before'): self.run_release()
        self.advice.assert_not_called()

    def test_sum_and_local_snapshot_mismatch(self):
        self.record['expected_bytes'] += 1; self.pin = self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'total'): self.run_release()
        self.record['expected_bytes'] -= 1; self.record['local_path'] = str(self.root)
        self.pin = self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'exact snapshot'): self.run_release()
        self.advice.assert_not_called()

    def test_fsync_or_advice_errors_fail_closed(self):
        self.sync.side_effect = OSError('file sync failed')
        with self.assertRaises(OSError): self.run_release()
        self.advice.assert_not_called()
        self.sync.side_effect = None; self.advice.side_effect = OSError('advice failed')
        with self.assertRaises(OSError): self.run_release()

    def test_deadline_stops_before_advice(self):
        with patch.object(release.time, 'monotonic', side_effect=[0,301]):
            with self.assertRaises(TimeoutError): self.run_release()
        self.advice.assert_not_called()

    def test_nonlinux_fails_without_touching_model(self):
        with patch.object(release.sys, 'platform', 'win32'):
            with self.assertRaises(RuntimeError): self.run_release()
        self.advice.assert_not_called()

    def test_symlink_outside_snapshot_rejected(self):
        outside = self.root/'outside'; outside.write_bytes(b'{}')
        path = self.snapshot/'model_index.json'; path.unlink()
        try: path.symlink_to(outside)
        except OSError: self.skipTest('OS forbids unprivileged symlinks')
        with self.assertRaisesRegex(ValueError, 'outside pinned'): self.run_release()
        self.advice.assert_not_called()

    def test_evaluator_rechecks_unchanged_memory_gate(self):
        from cloud.evaluate_a14b import run
        source = inspect.getsource(run)
        self.assertLess(source.index('release_model_cache(args.model'), source.index('resources=require_host_memory('))
        self.assertLess(source.index('resources=require_host_memory('), source.index('from_pretrained('))
        from cloud.eval_preflight import MIN_HOST_AVAILABLE_BYTES
        self.assertEqual(MIN_HOST_AVAILABLE_BYTES, 192*2**30)


if __name__ == '__main__': unittest.main()
