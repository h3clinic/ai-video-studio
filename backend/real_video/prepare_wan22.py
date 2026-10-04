"""Fetch pinned official, non-executable Wan2.2 weights; never remote code.

Only transformer/VAE/config/license assets are fetched. Reuse the existing
UMT5 encoder separately. Stable partial byte ranges are resumable, not model success.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import psutil
from huggingface_hub import HfApi, hf_hub_download

from .checkpoint_io import digest, keep_windows_awake
from .gaussian_program import write_json
from .resumable_model_download import download_verified

REPO = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
REVISION = 'b8fff7315c768468a5333511427288870b2e9635'
MODEL = Path('../../work/wan22_5b')


def selected(name):
    return (name.startswith(('transformer/', 'vae/', 'scheduler/')) or
            name in ('README.md', 'LICENSE', 'LICENSE.txt', 'model_index.json'))


def run(transport='ranges'):
    if transport not in ('ranges', 'xet'):
        raise ValueError('Explicit ranges or official Xet transport required')
    if transport == 'xet':
        # Hub 1.33 keeps this helper in _runtime (not utils.__init__).
        from huggingface_hub.utils._runtime import is_xet_available
        if not is_xet_available():
            raise RuntimeError('Official hf_xet must be installed and enabled')
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.gated or info.private:
        raise ValueError('Only the pinned ungated official model is allowed')
    if info.card_data.get('license') != 'apache-2.0':
        raise ValueError('Model license changed; stop')
    files = [f for f in info.siblings if selected(f.rfilename)]
    total = sum(f.size for f in files)
    if shutil.disk_usage('.').free < total * 2 + 20 * 2**30:
        raise RuntimeError('Insufficient conservative disk headroom')
    if psutil.virtual_memory().available < 8 * 2**30:
        raise RuntimeError('Need 8 GiB available RAM before fetching model')
    battery = psutil.sensors_battery()
    if battery and not battery.power_plugged and battery.percent <= 20:
        raise RuntimeError('Low battery')
    MODEL.mkdir(parents=True, exist_ok=True)
    report = dict(repo=REPO, revision=REVISION, license='apache-2.0', transport=transport,
                  download_bytes=total, status='downloading', files=[],
                  inference_tested=False, remote_code_executed=False,
                  runtime_precision='planned bfloat16; official files are float32',
                  standard_720p_documented_vram_gb=24,
                  planned_low_resolution_test='Up to 832x480; separate model stages, no 720p claim')
    start = time.perf_counter()
    for f in files:
        print('Fetching ' + f.rfilename, flush=True)
        expected = getattr(f.lfs, 'sha256', None) if f.lfs else None
        if f.rfilename.endswith('.safetensors'):
            if not expected:
                raise ValueError('Official LFS SHA256 required before fetching weights')
            destination = MODEL/f.rfilename
            if destination.exists():
                if destination.stat().st_size != f.size or digest(destination) != expected:
                    raise ValueError('Existing model file differs; it will not be overwritten')
                path = destination
            elif transport == 'xet':
                # The Hub's failed transfer may delete its own temporary file.
                # Keep this staging separate from our durable range partial.
                path = Path(hf_hub_download(REPO, f.rfilename, revision=REVISION,
                                           local_dir=MODEL/'xet_staging'))
                if path.stat().st_size != f.size or digest(path) != expected:
                    raise ValueError('Official Xet staging failed publisher hash verification')
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, destination)  # atomic no-overwrite, same-volume publish
                path = destination
            else:
                path = download_verified(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{f.rfilename}',
                                         destination, f.size, expected)
        else:
            path = hf_hub_download(REPO, f.rfilename, revision=REVISION, local_dir=MODEL)
        actual = digest(path)
        if Path(path).stat().st_size != f.size or (expected and actual != expected):
            raise ValueError('Downloaded file failed verification: ' + f.rfilename)
        report['files'].append(dict(path=f.rfilename, size=f.size, sha256=actual,
                                    remote_lfs_sha256=expected))
        report['elapsed_seconds'] = time.perf_counter() - start
        write_json(MODEL/'download_manifest.json', report)
    report['status'] = 'verified'
    write_json(MODEL/'download_manifest.json', report)
    print(json.dumps(dict(status='verified', total_bytes=total,
                          elapsed_seconds=report['elapsed_seconds'])), flush=True)


def bounded_run(wall_seconds=600, transport='ranges'):
    """Supervise a separate downloader process; never share its Python GIL.

    The parent owns the lock, awake request, attempt record and worker log.
    Wall expiry is checked before sensors and before accepting child success.
    Sensor errors fail closed. Only this Popen process and descendants whose
    PID/creation times were observed through it may be terminated. A stop
    requests immediate termination; up to five additional seconds may be
    needed to reap already-killed processes, not to continue downloading.
    """
    import math
    import subprocess
    import sys
    from filelock import FileLock

    if type(wall_seconds) is not int or not 30 <= wall_seconds <= 600:
        raise ValueError('Download wall budget must be an integer 30..600 seconds')
    if transport not in ('ranges', 'xet'):
        raise ValueError('Download transport must be ranges or xet')
    MODEL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(MODEL/'prepare.lock'), timeout=0):
        attempt = MODEL/'attempts'/str(time.time_ns())
        attempt.mkdir(parents=True)
        start = time.monotonic()
        child = None
        identities = {}
        stopped = False
        report = dict(status='starting', transport=transport, wall_seconds=wall_seconds,
                      model_ready=False, supervisor_pid=os.getpid(), worker_pid=None,
                      worker_created=None, worker_log=str(attempt/'worker.log'),
                      process_tree=[], termination=[], deadline_checked_before_sensors=True)

        def elapsed():
            return time.monotonic()-start

        def expired():
            return elapsed() >= wall_seconds

        def resources(required_free):
            """Read-only sensor failures are blockers, not permission to run."""
            try:
                available = psutil.virtual_memory().available
                if not isinstance(available, (int, float)) or not math.isfinite(available):
                    return 'invalid available-RAM sensor value'
                if available < required_free:
                    return 'available RAM preflight' if required_free > 3*2**30 else 'available RAM floor'
                # Never enter the battery query if the wall deadline already
                # passed during the RAM query. Check again after it returns.
                if expired():
                    return 'download wall limit'
                battery = psutil.sensors_battery()
                if battery is not None:
                    if (not isinstance(battery.percent, (int, float))
                            or not math.isfinite(battery.percent)
                            or not 0 <= battery.percent <= 100
                            or type(battery.power_plugged) is not bool):
                        return 'invalid battery sensor value'
                    if not battery.power_plugged and battery.percent <= 20:
                        return 'battery floor'
                return 'download wall limit' if expired() else None
            except Exception as error:
                return 'resource sensor failure: '+repr(error)

        def remember_tree():
            if child is None or report['worker_created'] is None:
                return
            try:
                root = psutil.Process(child.pid)
                if root.create_time() != report['worker_created']:
                    raise RuntimeError('Worker PID identity changed; refusing unrelated process')
                for process in [root]+root.children(recursive=True):
                    try:
                        identities[process.pid] = process.create_time()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass
            report['process_tree'] = [dict(pid=pid, created=created) for pid,created in identities.items()]

        def stop_tree():
            nonlocal stopped
            if child is None or stopped:
                return
            stopped = True
            try:
                remember_tree()
            except Exception as error:
                report['termination'].append(dict(error='tree inspection: '+repr(error)))
            # Stop the root first so it cannot create another downloader.
            ordered = sorted(identities, key=lambda pid: pid != child.pid)
            killed = []
            for pid in ordered:
                try:
                    process = psutil.Process(pid)
                    if process.create_time() != identities[pid]:
                        report['termination'].append(dict(pid=pid, action='identity_changed_not_touched'))
                        continue
                    process.kill()
                    killed.append(process)
                    report['termination'].append(dict(pid=pid, action='killed_owned_process'))
                except psutil.NoSuchProcess:
                    report['termination'].append(dict(pid=pid, action='already_exited'))
                except Exception as error:
                    report['termination'].append(dict(pid=pid, error=repr(error)))
            # The Popen handle refers to exactly the process we launched even
            # if creation-time lookup failed during its very short lifetime.
            if child.poll() is None:
                try:
                    child.kill()
                    report['termination'].append(dict(pid=child.pid, action='killed_owned_popen_handle'))
                except ProcessLookupError:
                    pass
                except Exception as error:
                    report['termination'].append(dict(pid=child.pid, error=repr(error)))
            if killed:
                _, alive = psutil.wait_procs(killed, timeout=5)
                report['remaining_owned_processes'] = [p.pid for p in alive]
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                report['termination'].append(dict(error='owned worker not reaped after kill'))

        try:
            with keep_windows_awake(), (attempt/'worker.log').open('wb') as worker_log:
                reason = 'download wall limit' if expired() else resources(8*2**30)
                if reason:
                    report.update(status='stopped', reason=reason)
                    raise RuntimeError(reason)
                command = [sys.executable, '-m', 'real_video.prepare_wan22',
                           '--worker', '--transport', transport]
                report['command'] = command
                child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                    stdin=subprocess.DEVNULL, stdout=worker_log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                report['worker_pid'] = child.pid
                try:
                    report['worker_created'] = psutil.Process(child.pid).create_time()
                    identities[child.pid] = report['worker_created']
                except psutil.NoSuchProcess:
                    # A child that completed before it could be inspected has
                    # no live tree to kill; its return code still gates success.
                    if child.poll() is None:
                        raise RuntimeError('Unable to establish live worker identity')
                report['status'] = 'running'
                write_json(attempt/'result.json', dict(report, elapsed_seconds=elapsed()))
                while True:
                    # This precedes sensor calls and even a nominal exit code0.
                    if expired():
                        report.update(status='stopped', reason='download wall limit')
                        stop_tree()
                        raise RuntimeError(report['reason'])
                    reason = resources(3*2**30)
                    if reason:
                        report.update(status='stopped', reason=reason)
                        stop_tree()
                        raise RuntimeError(reason)
                    remember_tree()
                    result = child.poll()
                    # Tree inspection or polling can cross the deadline.
                    if expired():
                        report.update(status='stopped', reason='download wall limit')
                        stop_tree()
                        raise RuntimeError(report['reason'])
                    if result is not None:
                        report['worker_returncode'] = result
                        if result != 0:
                            report.update(status='failed', reason='download worker failed')
                            stop_tree()
                            raise RuntimeError(f'Download worker exited {result}; see {attempt / "worker.log"}')
                        report.update(status='verified', model_ready=True)
                        break
                    try:
                        child.wait(timeout=max(.001, min(.5, wall_seconds-elapsed())))
                    except subprocess.TimeoutExpired:
                        pass
            return report
        except BaseException as error:
            if report['status'] not in ('stopped', 'failed'):
                report.update(status='failed', reason='supervisor failure')
            report.update(error=repr(error), model_ready=False)
            stop_tree()
            raise
        finally:
            report['elapsed_seconds'] = elapsed()
            if child is not None:
                report['worker_returncode'] = child.poll()
            write_json(attempt/'result.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-seconds',type=int,default=600)
    parser.add_argument('--transport',choices=('ranges','xet'),default='ranges')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run(args.transport)
    else:
        bounded_run(args.wall_seconds, args.transport)
