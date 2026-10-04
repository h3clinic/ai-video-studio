"""Independent parent supervision for a single project inference worker.

No model or CUDA context is loaded here. Execution completion is not a video
quality verdict. A shared GPU file lock serializes this helper's invocations;
the existing coordinator's running GPU rows and active NVIDIA compute PIDs
also block launch. Legacy code that bypasses this helper cannot be protected
against a simultaneous start solely by a file lock it does not acquire.

The parent's wall deadline includes worker startup/imports and stops active
work at at most 600 seconds. Terminated process reaping can take another five
seconds. Child output directories and partial artifacts are never deleted or
rewritten by this supervisor; its evidence lives in a separate attempt folder.
"""

from contextlib import closing, contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import uuid

from filelock import FileLock
import psutil

from .checkpoint_io import keep_windows_awake
from .gaussian_program import write_json


ROOT = Path(__file__).resolve().parents[1]
STATE = Path('artifacts/real_video/program')


def require_supervised_child():
    """Require the exact live supervisor somewhere in the ancestor chain.

    A Windows venv python.exe can launch a separate runtime interpreter, so
    direct PPID equality is not sufficient. ``Process.parent()`` checks the
    current process identity while traversing; the expected supervisor's
    creation time is checked explicitly instead of trusting its PID alone.
    """
    try:
        parent = int(os.environ['GV_GUARDED_PARENT_PID'])
        created = float(os.environ['GV_GUARDED_PARENT_CREATED'])
        if parent < 1 or not math.isfinite(created) or created <= 0:
            raise ValueError('invalid supervisor identity')
        process = psutil.Process()
        seen = {process.pid}
        for _ in range(64):
            process = process.parent()
            if process is None or process.pid in seen:
                break
            seen.add(process.pid)
            if process.pid == parent:
                if process.create_time() != created:
                    raise ValueError('supervisor creation time mismatch')
                return
        raise ValueError('supervisor is not a live ancestor')
    except (KeyError, ValueError, psutil.Error) as error:
        raise RuntimeError('Internal worker mode requires its live guarded parent') from error


def _resource_reason(required_free, expired):
    try:
        free = psutil.virtual_memory().available
        if not isinstance(free, (int, float)) or not math.isfinite(free) or free < 0:
            return 'invalid available-RAM sensor value'
        if free < required_free:
            return 'available RAM preflight' if required_free > 3*2**30 else 'available RAM floor'
        if expired():
            return 'wall limit'
        battery = psutil.sensors_battery()
        if battery is not None:
            if (not isinstance(battery.percent, (int, float)) or not math.isfinite(battery.percent)
                    or not 0 <= battery.percent <= 100 or type(battery.power_plugged) is not bool):
                return 'invalid battery sensor value'
            if not battery.power_plugged and battery.percent <= 20:
                return 'battery floor'
        return 'wall limit' if expired() else None
    except Exception as error:
        return 'resource sensor failure: '+repr(error)


def _gpu_preflight(root):
    """Read-only checks; no CUDA initialization and no unrelated termination."""
    db = root/STATE/'program.sqlite3'
    if db.exists():
        with closing(sqlite3.connect(db.as_uri()+'?mode=ro', uri=True, timeout=1)) as connection:
            active = connection.execute("SELECT id FROM jobs WHERE resource='gpu' AND status='running'").fetchall()
        if active:
            raise RuntimeError('Coordinator already owns a running GPU job; no new worker launched')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line == 'No running processes found':
            continue
        if not line.isdecimal() or int(line) < 1:
            raise RuntimeError('Unrecognized NVIDIA compute-process status; fail closed')
        pids.append(int(line))
    if pids:
        raise RuntimeError('Other NVIDIA compute processes are active; no process was terminated: '+str(pids))


class OwnedProcessTree:
    """Only the Popen handle and observed PID/creation-time descendants."""
    def __init__(self, child):
        self.child = child
        self.created = None
        self.identities = {}
        self.termination = []
        self.stopped = False
        try:
            self.created = psutil.Process(child.pid).create_time()
            self.identities[child.pid] = self.created
        except psutil.NoSuchProcess:
            if child.poll() is None:
                raise RuntimeError('Unable to establish live worker identity')

    def remember(self):
        if self.created is None:
            return
        try:
            root = psutil.Process(self.child.pid)
            if root.create_time() != self.created:
                raise RuntimeError('Worker PID identity changed')
            for process in [root]+root.children(recursive=True):
                try:
                    self.identities[process.pid] = process.create_time()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        try:
            self.remember()
        except Exception as error:
            self.termination.append(dict(error='tree inspection: '+repr(error)))
        killed = []
        for pid in sorted(self.identities, key=lambda value: value != self.child.pid):
            try:
                process = psutil.Process(pid)
                if process.create_time() != self.identities[pid]:
                    self.termination.append(dict(pid=pid, action='identity_changed_not_touched'))
                    continue
                process.kill()
                killed.append(process)
                self.termination.append(dict(pid=pid, action='killed_owned_process'))
            except psutil.NoSuchProcess:
                self.termination.append(dict(pid=pid, action='already_exited'))
            except Exception as error:
                self.termination.append(dict(pid=pid, error=repr(error)))
        if self.child.poll() is None:
            try:
                self.child.kill()
                self.termination.append(dict(pid=self.child.pid, action='killed_owned_popen_handle'))
            except ProcessLookupError:
                pass
            except Exception as error:
                self.termination.append(dict(pid=self.child.pid, error=repr(error)))
        if killed:
            _, alive = psutil.wait_procs(killed, timeout=5)
            if alive:
                self.termination.append(dict(error='owned descendants not reaped', pids=[p.pid for p in alive]))
        try:
            self.child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.termination.append(dict(error='owned worker not reaped after kill'))


@contextmanager
def _termination_scope(get_owned, get_child):
    """Reap owned work BEFORE the outer GPU lock/awake contexts can exit."""
    try:
        yield
    finally:
        owned, child = get_owned(), get_child()
        if owned is not None:
            owned.stop()
        elif child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def run_guarded_worker(module, arguments, output, *, wall_seconds=600, resource='gpu', root=ROOT):
    """Launch ``python -m module <arguments>`` under an independent parent.

    ``arguments`` must include the module's internal ``--worker`` switch. The
    caller's unchanged worker remains responsible for artifact/model validity;
    this supervisor records execution only and never marks quality accepted.
    """
    root = Path(root).resolve()
    if not isinstance(module, str) or re.fullmatch(r'real_video\.[a-zA-Z0-9_]+', module) is None:
        raise ValueError('One explicit local real_video module required')
    if not (root/Path(*module.split('.'))).with_suffix('.py').is_file():
        raise ValueError('Worker module is not present under project root')
    if (not isinstance(arguments, (list, tuple)) or any(not isinstance(a, str) for a in arguments)
            or '--worker' not in arguments):
        raise ValueError('Explicit string arguments with --worker required; no shell commands')
    if type(wall_seconds) is not int or not 30 <= wall_seconds <= 600:
        raise ValueError('Total wall budget must be an integer 30..600 seconds')
    if resource not in ('cpu', 'gpu'):
        raise ValueError('Resource must be cpu or gpu')
    output = (root/Path(output)).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError('Output must be a named child of this project')
    if output.exists():
        raise FileExistsError('Immutable worker output already exists')
    state = root/STATE
    state.mkdir(parents=True, exist_ok=True)
    attempt = state/'guarded_workers'/(str(time.time_ns())+'-'+uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True)
    start = time.monotonic()
    child = owned = None
    report = dict(schema='guarded_local_worker_v1', status='starting', module=module,
        output=str(output), resource=resource, wall_seconds=wall_seconds,
        parent_pid=os.getpid(), worker_pid=None, worker_created=None,
        worker_log=str(attempt/'worker.log'), quality_accepted=False,
        automatic_retry=False, algorithm_modified=False,
        resource_policy=dict(entry_free_ram_bytes=8*2**30, runtime_free_ram_bytes=3*2**30,
                             battery_floor_without_ac=20, total_wall_seconds=wall_seconds))
    expired = lambda: time.monotonic()-start >= wall_seconds
    def stop(reason):
        report.update(status='stopped', reason=reason)
        if owned is not None:
            owned.stop()
        raise RuntimeError(reason)
    lock = FileLock(str(state/'gpu.lock'), timeout=0) if resource == 'gpu' else nullcontext()
    try:
        with lock, keep_windows_awake(), _termination_scope(lambda:owned, lambda:child), (attempt/'worker.log').open('wb') as log:
            if expired():
                stop('wall limit')
            reason = _resource_reason(8*2**30, expired)
            if reason:
                stop(reason)
            if resource == 'gpu':
                _gpu_preflight(root)
            if expired():
                stop('wall limit')
            if output.exists():
                raise FileExistsError('Output appeared before launch; not overwritten')
            command = [sys.executable, '-m', module, *arguments]
            environment = os.environ.copy()
            environment['GV_GUARDED_PARENT_PID'] = str(os.getpid())
            environment['GV_GUARDED_PARENT_CREATED'] = repr(psutil.Process().create_time())
            child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            owned = OwnedProcessTree(child)
            report.update(status='running', command=command, worker_pid=child.pid, worker_created=owned.created)
            write_json(attempt/'result.json', report)
            while True:
                if expired():
                    stop('wall limit')
                reason = _resource_reason(3*2**30, expired)
                if reason:
                    stop(reason)
                owned.remember()
                code = child.poll()
                # Poll/tree inspection may cross the deadline. Never accept a
                # late code0 simply because the earlier resource check passed.
                if expired():
                    stop('wall limit')
                if code is not None:
                    report['worker_returncode'] = code
                    if code != 0:
                        report.update(status='failed', reason='worker failed')
                        owned.stop()
                        raise RuntimeError(f'Worker exited {code}; inspect {attempt / "worker.log"}')
                    report.update(status='completed', execution_completed=True)
                    break
                try:
                    child.wait(timeout=max(.001, min(.5, wall_seconds-(time.monotonic()-start))))
                except subprocess.TimeoutExpired:
                    pass
        return report
    except BaseException as error:
        if report['status'] not in ('stopped', 'failed'):
            report.update(status='failed', reason='supervisor or preflight failure')
        report.update(error=repr(error), execution_completed=False)
        if owned is not None:
            owned.stop()
        elif child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        if child is not None:
            report['worker_returncode'] = child.poll()
        if owned is not None:
            report['process_tree'] = [dict(pid=pid, created=created) for pid,created in owned.identities.items()]
            report['termination'] = owned.termination
        write_json(attempt/'result.json', report)
