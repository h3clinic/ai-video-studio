"""Independent downloader supervision: tiny local children, no network/GPU."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil

from real_video import prepare_wan22


REAL_POPEN = subprocess.Popen


class Clock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def time_ns(self):
        return time.time_ns()


class Wan22DownloadSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name)/'model'
        self.children = []
        self.addCleanup(self.cleanup_children)
        changes = [patch.object(prepare_wan22, 'MODEL', self.model),
            patch.object(prepare_wan22.psutil, 'virtual_memory', return_value=SimpleNamespace(available=10*2**30)),
            patch.object(prepare_wan22.psutil, 'sensors_battery', return_value=SimpleNamespace(percent=90., power_plugged=True))]
        for change in changes:
            change.start()
            self.addCleanup(change.stop)

    def cleanup_children(self):
        for child in self.children:
            if child.poll() is None:
                try:
                    for descendant in psutil.Process(child.pid).children(recursive=True):
                        descendant.kill()
                except psutil.NoSuchProcess:
                    pass
                child.kill()
            child.wait(timeout=3)

    def worker(self, source="print('local supervisor test only', flush=True)", clock=None):
        def launch(command, **kwargs):
            self.assertEqual(command[:3], [sys.executable, '-m', 'real_video.prepare_wan22'])
            self.assertIn('--worker', command)
            self.assertNotIn('shell', kwargs)
            child = REAL_POPEN([sys.executable, '-c', source], **kwargs)
            self.children.append(child)
            if clock is not None:
                clock.now = 601.
            return child
        return patch('subprocess.Popen', side_effect=launch)

    def report(self):
        paths = list((self.model/'attempts').glob('*/result.json'))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text()), paths[0].parent

    def test_child_success_logs_identity_and_returns_verified(self):
        with self.worker():
            report = prepare_wan22.bounded_run(600, 'xet')
        saved, attempt = self.report()
        self.assertEqual(saved['status'], 'verified')
        self.assertTrue(saved['model_ready'])
        self.assertEqual(saved['worker_returncode'], 0)
        self.assertEqual(saved['worker_pid'], self.children[0].pid)
        self.assertIsInstance(saved['worker_created'], float)
        self.assertIn('local supervisor test only', (attempt/'worker.log').read_text())
        self.assertEqual(report['transport'], 'xet')

    def test_deadline_kills_independent_process_before_new_sensor_call(self):
        clock = Clock()
        battery = prepare_wan22.psutil.sensors_battery
        with patch.object(prepare_wan22, 'time', clock), self.worker('import time;time.sleep(60)', clock):
            with self.assertRaisesRegex(RuntimeError, 'wall limit'):
                prepare_wan22.bounded_run(600)
        saved, _ = self.report()
        self.assertEqual(saved['status'], 'stopped')
        self.assertEqual(saved['reason'], 'download wall limit')
        self.assertFalse(saved['model_ready'])
        self.assertEqual(battery.call_count, 1)  # only preflight, never after expiry
        self.assertIsNotNone(self.children[0].poll())
        self.assertTrue(any(item.get('action', '').startswith('killed_owned') for item in saved['termination']))

    def test_runtime_ram_floor_stops_only_launched_worker(self):
        prepare_wan22.psutil.virtual_memory.side_effect = [SimpleNamespace(available=9*2**30), SimpleNamespace(available=2*2**30)]
        with self.worker('import time;time.sleep(60)'):
            with self.assertRaisesRegex(RuntimeError, 'RAM floor'):
                prepare_wan22.bounded_run()
        saved, _ = self.report()
        self.assertEqual(saved['reason'], 'available RAM floor')
        self.assertIsNotNone(self.children[0].poll())

    def test_successful_poll_crossing_deadline_is_not_verified(self):
        clock = Clock()
        process = SimpleNamespace(pid=987654321, create_time=lambda: 123.,
                                  children=lambda recursive=True: [], kill=lambda: None)
        def late_poll():
            clock.now = 601.
            return 0
        child = SimpleNamespace(pid=process.pid, poll=late_poll, wait=lambda timeout: 0,
                                kill=lambda: None)
        with patch.object(prepare_wan22, 'time', clock), \
                patch('subprocess.Popen', return_value=child), \
                patch.object(prepare_wan22.psutil, 'Process', return_value=process), \
                patch.object(prepare_wan22.psutil, 'wait_procs', return_value=([], [])):
            with self.assertRaisesRegex(RuntimeError, 'wall limit'):
                prepare_wan22.bounded_run(600)
        saved, _ = self.report()
        self.assertEqual(saved['status'], 'stopped')
        self.assertFalse(saved['model_ready'])

    def test_preflight_ram_battery_and_sensor_errors_never_launch(self):
        prepare_wan22.psutil.virtual_memory.return_value = SimpleNamespace(available=7*2**30)
        with patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(RuntimeError, 'RAM preflight'):
                prepare_wan22.bounded_run()
        saved, _ = self.report()
        self.assertEqual(saved['worker_pid'], None)
        self.assertFalse(saved['model_ready'])

    def test_low_battery_preflight_is_saved_and_not_launched(self):
        prepare_wan22.psutil.sensors_battery.return_value = SimpleNamespace(percent=20., power_plugged=False)
        with patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(RuntimeError, 'battery floor'):
                prepare_wan22.bounded_run()
        self.assertEqual(self.report()[0]['reason'], 'battery floor')

    def test_sensor_exception_fails_closed_and_terminates_worker(self):
        prepare_wan22.psutil.sensors_battery.side_effect = [SimpleNamespace(percent=80., power_plugged=True), OSError('sensor failed')]
        with self.worker('import time;time.sleep(60)'):
            with self.assertRaisesRegex(RuntimeError, 'sensor failure'):
                prepare_wan22.bounded_run()
        saved, _ = self.report()
        self.assertIn('sensor failed', saved['reason'])
        self.assertIsNotNone(self.children[0].poll())

    def test_nonzero_child_exit_is_not_model_success(self):
        with self.worker('raise SystemExit(7)'):
            with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                prepare_wan22.bounded_run()
        saved, _ = self.report()
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(saved['worker_returncode'], 7)
        self.assertFalse(saved['model_ready'])

    def test_exact_spawned_descendant_tree_is_terminated(self):
        clock = Clock()
        source = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                  "print(p.pid,flush=True);time.sleep(60)")
        descendant = []
        def launch(command, **kwargs):
            child = REAL_POPEN([sys.executable, '-c', source], **kwargs)
            self.children.append(child)
            log = Path(kwargs['stdout'].name)
            deadline = time.monotonic()+3
            while time.monotonic() < deadline and not log.read_text().strip():
                time.sleep(.02)
            descendant.append(int(log.read_text().strip()))
            clock.now = 601
            return child
        with patch.object(prepare_wan22, 'time', clock), patch('subprocess.Popen', side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, 'wall limit'):
                prepare_wan22.bounded_run()
        saved, _ = self.report()
        self.assertIn(descendant[0], [item['pid'] for item in saved['process_tree']])
        self.assertFalse(psutil.pid_exists(descendant[0]))
        self.assertIsNotNone(self.children[0].poll())

    def test_invalid_profile_rejected_without_launch(self):
        for wall, transport in [(0, 'ranges'), (601, 'ranges'), (True, 'ranges'), (30., 'xet'), (600, 'unknown')]:
            with self.assertRaises(ValueError):
                prepare_wan22.bounded_run(wall, transport)
        self.assertFalse(self.model.exists())

    def test_sensor_none_is_allowed_for_desktop_without_battery(self):
        prepare_wan22.psutil.sensors_battery.return_value = None
        with self.worker():
            report = prepare_wan22.bounded_run()
        self.assertTrue(report['model_ready'])


if __name__ == '__main__':
    unittest.main()
