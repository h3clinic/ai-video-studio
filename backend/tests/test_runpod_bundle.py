import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch
import zipfile

from real_video.build_runpod_bundle import ALLOWLIST, CASE, build, selected_files


class BundleTests(unittest.TestCase):
    def fixture(self, root):
        for relative in ALLOWLIST:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('fixture:' + relative).encode())
        checkpoint = root / CASE / 'gaussian_cpu/gaussians.pt'
        checkpoint.with_suffix('.pt.sha256.json').write_text(json.dumps(
            {'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))

    def test_explicit_allowlist_excludes_credentials_and_models(self):
        self.assertEqual(len(ALLOWLIST), len(set(ALLOWLIST)))
        for path in ALLOWLIST:
            self.assertNotIn('..', Path(path).parts)
            self.assertFalse(Path(path).is_absolute())
            for forbidden in ('.env', '.git', 'settings', 'dpapi', 'runway_bootstrap',
                              'real_video/runway_pilot.py', 'models/', 'secrets', 'credentials'):
                self.assertNotIn(forbidden, path)

    def test_only_allowed_content_and_manifest_matches_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / '.env').write_text('must never be packaged')
            (root / 'private.txt').write_text('not relevant')
            out = root / 'bundle.zip'
            report = build(root, out)
            with zipfile.ZipFile(out) as archive:
                self.assertEqual(set(archive.namelist()), set(ALLOWLIST) | {'bundle_manifest.json'})
                manifest = json.loads(archive.read('bundle_manifest.json'))
                self.assertEqual(manifest, report['manifest'])
                self.assertNotIn(str(root), json.dumps(manifest))
                for entry in manifest['files']:
                    data = archive.read(entry['path'])
                    self.assertEqual(entry['bytes'], len(data))
                    self.assertEqual(entry['sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(report['sha256'], hashlib.sha256(out.read_bytes()).hexdigest())
            original = out.read_bytes()
            with self.assertRaises(FileExistsError):
                build(root, out)
            self.assertEqual(out.read_bytes(), original)

    def test_missing_input_fails_before_archive_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                build(root, root / 'bundle.zip')
            self.assertFalse((root / 'bundle.zip').exists())

    def test_bad_checkpoint_hash_fails_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / CASE / 'gaussian_cpu/gaussians.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'sidecar mismatch'):
                build(root, root / 'bundle.zip')
            self.assertFalse((root / 'bundle.zip').exists())


class JobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'cloud/runpod_job.py'
        spec = importlib.util.spec_from_file_location('runpod_job_test', path)
        cls.job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.job)

    def test_job_rejects_unbounded_runtime_before_launch(self):
        for budget in (0, 601):
            with self.assertRaises(ValueError):
                self.job.run(Path('.'), 'cpu', budget)

    def test_failed_child_persists_exact_exit_code_and_no_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(self.job, 'verify_workspace', return_value=Path(tmp)), \
                    patch.object(self.job.subprocess, 'Popen') as popen:
                popen.return_value.wait.return_value = 7
                result = self.job.run(tmp, 'cpu', 10)
            self.assertEqual(result['exit_code'], 7)
            self.assertEqual(result['result'], 'failed')
            self.assertEqual(result['local_api_calls'], 0)
            self.assertIn('--device', result['command'])
            self.assertEqual(result['command'][-1], 'cpu')
            saved = list(Path(tmp).glob('artifacts/cloud/jobs/*/job_status.json'))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text()), result)

    def test_posix_descendants_are_killed_even_when_parent_already_exited(self):
        process = MagicMock(pid=12345)
        process.poll.return_value = 0
        process.wait.return_value = 0
        with patch.object(self.job.os, 'name', 'posix'), \
                patch.object(self.job.os, 'killpg', create=True) as killpg, \
                patch.object(self.job.signal, 'SIGKILL', 9, create=True), \
                patch.object(self.job.time, 'monotonic', side_effect=[0, 4]), \
                patch.object(self.job.time, 'sleep') as sleep:
            self.job.stop_child(process)
        self.assertEqual(killpg.call_args_list, [
            call(12345, self.job.signal.SIGTERM), call(12345, 0),
            call(12345, 9)])
        process.wait.assert_called_once_with(timeout=3)
        sleep.assert_not_called()

    def test_posix_disappeared_group_is_not_signalled_again(self):
        process = MagicMock(pid=12345)
        with patch.object(self.job.os, 'name', 'posix'), \
                patch.object(self.job.os, 'killpg', create=True,
                             side_effect=[None, ProcessLookupError]) as killpg, \
                patch.object(self.job.time, 'sleep') as sleep:
            self.job.stop_child(process)
        self.assertEqual(killpg.call_args_list, [
            call(12345, self.job.signal.SIGTERM), call(12345, 0)])
        process.wait.assert_called_once_with(timeout=3)
        sleep.assert_not_called()

    def test_windows_cleanup_escalates_only_its_own_process(self):
        process = MagicMock(pid=12345)
        process.wait.side_effect = [self.job.subprocess.TimeoutExpired('child', 3), -9]
        with patch.object(self.job.os, 'name', 'nt'), \
                patch.object(self.job.os, 'killpg', create=True) as killpg:
            self.job.stop_child(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list, [call(timeout=3), call(timeout=3)])
        killpg.assert_not_called()

    def test_timeout_path_records_cleanup_and_persists_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = MagicMock(pid=12345, returncode=-9)
            process.wait.side_effect = self.job.subprocess.TimeoutExpired('child', 10)
            with patch.object(self.job, 'verify_workspace', return_value=Path(tmp)), \
                    patch.object(self.job.subprocess, 'Popen', return_value=process) as popen, \
                    patch.object(self.job, 'stop_child') as stop:
                result = self.job.run(tmp, 'cpu', 10)
            stop.assert_called_once_with(process)
            self.assertEqual(result['result'], 'timed_out')
            self.assertEqual(result['exit_code'], -9)
            self.assertEqual(popen.call_args.kwargs['start_new_session'], self.job.os.name == 'posix')
            self.assertIn('does NOT stop', result['billing_warning'])
            saved = next(Path(tmp).glob('artifacts/cloud/jobs/*/job_status.json'))
            self.assertEqual(json.loads(saved.read_text()), result)


if __name__ == '__main__':
    unittest.main()
