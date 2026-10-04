"""CPU child-process supervision tests. No model, network or CUDA operations."""

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from filelock import FileLock, Timeout
import psutil

from real_video import guarded_worker as guard


REAL_POPEN = subprocess.Popen


class Clock:
    def __init__(self): self.now = 0.
    def monotonic(self): return self.now
    def time_ns(self): return time.time_ns()


class GuardedWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root/'real_video').mkdir()
        (self.root/'real_video/local_test_worker.py').write_text('# not executed; test uses a tiny local child\n')
        self.output = self.root/'artifacts/result'
        self.children = []
        self.addCleanup(self.cleanup)
        for item in [patch.object(guard.psutil, 'virtual_memory', return_value=SimpleNamespace(available=10*2**30)),
                     patch.object(guard.psutil, 'sensors_battery', return_value=SimpleNamespace(percent=80., power_plugged=True))]:
            item.start()
            self.addCleanup(item.stop)

    def cleanup(self):
        for child in self.children:
            if child.poll() is None:
                try:
                    for process in psutil.Process(child.pid).children(recursive=True): process.kill()
                except psutil.NoSuchProcess:
                    pass
                child.kill()
            child.wait(timeout=3)

    def worker(self, source="print('local CPU test', flush=True)", clock=None, wait=False):
        def launch(command, **kwargs):
            self.assertEqual(command[:3], [sys.executable, '-m', 'real_video.local_test_worker'])
            self.assertIn('--worker', command)
            self.assertEqual(kwargs['env']['GV_GUARDED_PARENT_PID'], str(os.getpid()))
            self.assertNotIn('shell', kwargs)
            child = REAL_POPEN([sys.executable, '-c', source], **kwargs)
            self.children.append(child)
            if wait: child.wait(timeout=3)
            if clock is not None: clock.now = 601.
            return child
        return patch('subprocess.Popen', side_effect=launch)

    def run_worker(self, **kwargs):
        return guard.run_guarded_worker('real_video.local_test_worker', ['--worker'], self.output,
                                        root=self.root, resource=kwargs.pop('resource', 'cpu'), **kwargs)

    def report(self):
        records = list((self.root/guard.STATE/'guarded_workers').glob('*/result.json'))
        self.assertEqual(len(records), 1)
        return json.loads(records[0].read_text()), records[0].parent

    def test_completed_execution_never_claims_quality_and_logs_identity(self):
        with self.worker(): report = self.run_worker()
        saved, attempt = self.report()
        self.assertEqual(saved['status'], 'completed')
        self.assertFalse(saved['quality_accepted'])
        self.assertTrue(saved['execution_completed'])
        self.assertEqual(saved['worker_pid'], self.children[0].pid)
        self.assertEqual(saved['worker_returncode'], 0)
        self.assertIn('local CPU test', (attempt/'worker.log').read_text())
        self.assertFalse(self.output.exists())  # supervisor did not reserve/mutate worker output
        self.assertEqual(report['resource_policy']['entry_free_ram_bytes'], 8*2**30)

    def test_wall_deadline_kills_worker_before_any_new_sensor_query(self):
        clock = Clock()
        battery = guard.psutil.sensors_battery
        with patch.object(guard, 'time', clock), self.worker('import time;time.sleep(60)', clock):
            with self.assertRaisesRegex(RuntimeError, 'wall limit'): self.run_worker()
        self.assertEqual(battery.call_count, 1)
        self.assertEqual(self.report()[0]['reason'], 'wall limit')
        self.assertIsNotNone(self.children[0].poll())

    def test_late_success_after_tree_inspection_cannot_pass_deadline(self):
        clock = Clock()
        original = guard.OwnedProcessTree.remember
        def delayed(tree):
            original(tree)
            clock.now = 601.
        with patch.object(guard, 'time', clock), self.worker('pass', wait=True), patch.object(guard.OwnedProcessTree, 'remember', delayed):
            with self.assertRaisesRegex(RuntimeError, 'wall limit'): self.run_worker()
        report, _ = self.report()
        self.assertEqual(report['worker_returncode'], 0)
        self.assertEqual(report['status'], 'stopped')
        self.assertFalse(report['execution_completed'])

    def test_sensor_failure_is_saved_and_kills_child(self):
        guard.psutil.sensors_battery.side_effect = [SimpleNamespace(percent=90., power_plugged=True), OSError('sensor failure')]
        with self.worker('import time;time.sleep(60)'):
            with self.assertRaisesRegex(RuntimeError, 'sensor failure'): self.run_worker()
        self.assertIsNotNone(self.children[0].poll())
        self.assertIn('sensor failure', self.report()[0]['reason'])

    def test_runtime_low_ram_is_fail_closed(self):
        guard.psutil.virtual_memory.side_effect = [SimpleNamespace(available=9*2**30), SimpleNamespace(available=2*2**30)]
        with self.worker('import time;time.sleep(60)'):
            with self.assertRaisesRegex(RuntimeError, 'RAM floor'): self.run_worker()
        self.assertIsNotNone(self.children[0].poll())

    def test_preflight_low_battery_does_not_launch(self):
        guard.psutil.sensors_battery.return_value = SimpleNamespace(percent=20., power_plugged=False)
        with patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(RuntimeError, 'battery floor'): self.run_worker()
        self.assertIsNone(self.report()[0]['worker_pid'])

    def test_gpu_lock_blocks_second_worker_without_launch(self):
        state = self.root/guard.STATE
        state.mkdir(parents=True)
        with FileLock(str(state/'gpu.lock'), timeout=0), patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaises(Timeout): self.run_worker(resource='gpu')
        self.assertFalse(self.report()[0]['execution_completed'])

    def test_gpu_lock_remains_held_until_exception_cleanup_has_killed_worker(self):
        original = guard.OwnedProcessTree.stop
        checks = []
        def stop(tree):
            if not tree.stopped:
                with self.assertRaises(Timeout):
                    with FileLock(str(self.root/guard.STATE/'gpu.lock'), timeout=0): pass
                checks.append(True)
            return original(tree)
        with patch.object(guard, '_gpu_preflight'), self.worker('import time;time.sleep(60)'), \
                patch.object(guard.OwnedProcessTree, 'remember', side_effect=OSError('tree fault')), \
                patch.object(guard.OwnedProcessTree, 'stop', stop):
            with self.assertRaisesRegex(OSError, 'tree fault'): self.run_worker(resource='gpu')
        self.assertEqual(checks, [True])
        self.assertIsNotNone(self.children[0].poll())
        with FileLock(str(self.root/guard.STATE/'gpu.lock'), timeout=0): pass

    def test_exact_descendant_is_killed_and_created_artifact_preserved(self):
        clock = Clock()
        source = ("import subprocess,sys,time;from pathlib import Path;"
                  f"out=Path({str(self.output)!r});out.mkdir(parents=True);(out/'latent.pt').write_bytes(b'preserve');"
                  "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                  "print(p.pid,flush=True);time.sleep(60)")
        descendants = []
        def launch(command, **kwargs):
            child = REAL_POPEN([sys.executable, '-c', source], **kwargs)
            self.children.append(child)
            log = Path(kwargs['stdout'].name)
            deadline = time.monotonic()+3
            while time.monotonic() < deadline and not log.read_text().strip(): time.sleep(.02)
            descendants.append(int(log.read_text().strip()))
            clock.now = 601.
            return child
        with patch.object(guard, 'time', clock), patch('subprocess.Popen', side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, 'wall limit'): self.run_worker()
        self.assertFalse(psutil.pid_exists(descendants[0]))
        self.assertEqual((self.output/'latent.pt').read_bytes(), b'preserve')

    def test_nonzero_child_exit_fails_execution(self):
        with self.worker('raise SystemExit(4)'):
            with self.assertRaisesRegex(RuntimeError, 'exited 4'): self.run_worker()
        report, _ = self.report()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['worker_returncode'], 4)

    def test_gpu_preflight_rejects_compute_pids_without_terminating_them(self):
        with patch.object(guard.subprocess, 'run', return_value=SimpleNamespace(stdout='12345\n')) as query:
            with self.assertRaisesRegex(RuntimeError, 'compute processes'): guard._gpu_preflight(self.root)
        self.assertEqual(query.call_args.args[0][0], 'nvidia-smi')
        self.assertEqual(query.call_args.kwargs['timeout'], 5)

    def test_coordinator_running_gpu_row_blocks_even_before_nvidia_query(self):
        state = self.root/guard.STATE
        state.mkdir(parents=True)
        with sqlite3.connect(state/'program.sqlite3') as connection:
            connection.execute('CREATE TABLE jobs(id TEXT, resource TEXT, status TEXT)')
            connection.execute("INSERT INTO jobs VALUES('known-job','gpu','running')")
        connection.close()
        with patch.object(guard.subprocess, 'run', side_effect=AssertionError('do not query')):
            with self.assertRaisesRegex(RuntimeError, 'Coordinator'): guard._gpu_preflight(self.root)

    def test_existing_output_and_invalid_bounds_fail_without_worker(self):
        for kwargs in ({'wall_seconds':601}, {'wall_seconds':True}, {'resource':'anything'}):
            with self.assertRaises(ValueError): self.run_worker(**kwargs)
        self.output.mkdir(parents=True)
        (self.output/'report.json').write_text('preserve')
        with self.assertRaises(FileExistsError): self.run_worker()
        self.assertEqual((self.output/'report.json').read_text(), 'preserve')

    def test_worker_mode_requires_matching_live_parent_identity(self):
        with patch.dict(os.environ, {'GV_GUARDED_PARENT_PID':'-1', 'GV_GUARDED_PARENT_CREATED':'0'}):
            with self.assertRaises(RuntimeError): guard.require_supervised_child()
        parent = psutil.Process().ppid()
        with patch.dict(os.environ, {'GV_GUARDED_PARENT_PID':str(parent),
                                   'GV_GUARDED_PARENT_CREATED':repr(psutil.Process(parent).create_time())}):
            guard.require_supervised_child()
        with patch.dict(os.environ, {'GV_GUARDED_PARENT_PID':str(parent),
                                   'GV_GUARDED_PARENT_CREATED':repr(psutil.Process(parent).create_time()+1)}):
            with self.assertRaises(RuntimeError): guard.require_supervised_child()

    def test_actual_venv_python_launcher_chain_passes_ancestor_validation(self):
        environment = os.environ.copy()
        environment['GV_GUARDED_PARENT_PID'] = str(os.getpid())
        environment['GV_GUARDED_PARENT_CREATED'] = repr(psutil.Process().create_time())
        source = ("import os,psutil;from real_video.guarded_worker import require_supervised_child;"
                  "require_supervised_child();print('validated',os.getpid(),psutil.Process().ppid(),flush=True)")
        result = subprocess.run([sys.executable, '-c', source], cwd=guard.ROOT, env=environment,
            capture_output=True, text=True, timeout=20, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.assertIn('validated', result.stdout)


if __name__ == '__main__':
    unittest.main()
