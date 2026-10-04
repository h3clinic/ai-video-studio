"""Run one <=600s local reconstruction child; timeout DOES NOT stop Pod billing."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import uuid


CASE = 'artifacts/real_video/runway_agents/donkey_orange_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_workspace(workspace):
    workspace = Path(workspace).resolve(strict=True)
    if Path(__file__).resolve() != workspace / 'cloud/runpod_job.py':
        raise ValueError('Run the job from its own explicitly specified extracted workspace')
    manifest = json.loads((workspace / 'bundle_manifest.json').read_text(encoding='utf-8'))
    entries = manifest.get('files', [])
    if manifest.get('schema_version') != 1 or not entries:
        raise ValueError('Missing valid bundle manifest')
    seen = set()
    for entry in entries:
        relative = entry['path']
        parts = PurePosixPath(relative)
        if (parts.is_absolute() or '..' in parts.parts or '\\' in relative
                or ':' in relative or relative in seen):
            raise ValueError('Unsafe or duplicate manifest path')
        seen.add(relative)
        path = (workspace / relative).resolve(strict=True)
        if not path.is_relative_to(workspace) or not path.is_file():
            raise ValueError('Manifest escaped the workspace')
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Bundle file hash mismatch')
    required = {f'{CASE}/source/video.mp4', f'{CASE}/gaussian_cpu/gaussians.pt',
                f'{CASE}/gaussian_cpu/gaussians.pt.sha256.json',
                f'{CASE}/gaussian_cpu/gaussian_replay.mp4',
                'real_video/evaluate_detail_profile.py', 'cloud/runpod_job.py'}
    if not required.issubset(seen):
        raise ValueError('Incomplete experiment bundle')
    return workspace


def stop_child(process):
    """Stop only the child session created by this launcher's Popen call.

    The child PID is its process-group ID because start_new_session=True.
    A parent exiting does not prove its encoder descendants have exited.
    """
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=3)
            return
        deadline = time.monotonic() + 3
        while True:
            # Reap the direct child if it exited, but independently inspect its
            # original group. Never derive a different group from another PID.
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The group exited between the probe and escalation.
                break
            time.sleep(.05)
        process.wait(timeout=3)
    else:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def run(workspace, device='cuda', timeout_seconds=600):
    if device not in ('cpu', 'cuda') or not 10 <= timeout_seconds <= 600:
        raise ValueError('Use cpu/cuda and a 10..600 second process budget')
    workspace = verify_workspace(workspace)
    job_dir = workspace / 'artifacts/cloud/jobs' / ('job_' + uuid.uuid4().hex)
    job_dir.mkdir(parents=True, exist_ok=False)
    out = job_dir / 'experiment'
    command = [sys.executable, '-m', 'real_video.evaluate_detail_profile',
               '--source', str(workspace / CASE / 'source/video.mp4'),
               '--baseline-dir', str(workspace / CASE / 'gaussian_cpu'),
               '--out', str(out), '--budget-seconds', str(max(10, timeout_seconds - 5)),
               '--refresh-coverage', '--device', device]
    record = dict(schema_version=1, command=command, device=device,
                  timeout_seconds=timeout_seconds, result='starting',
                  local_api_calls=0, training=False, accepted_autonomous_video=False,
                  billing_warning='This process timeout does NOT stop or terminate the RunPod Pod or its billing.')
    status = job_dir / 'job_status.json'
    status.write_text(json.dumps(record, indent=2), encoding='utf-8')
    begun = time.monotonic()
    process = None
    try:
        with (job_dir / 'stdout.log').open('wb') as stdout, (job_dir / 'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, cwd=workspace, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr,
                                       start_new_session=(os.name == 'posix'))
            try:
                record['exit_code'] = process.wait(timeout=timeout_seconds)
                record['result'] = 'completed' if record['exit_code'] == 0 else 'failed'
            except subprocess.TimeoutExpired:
                stop_child(process)
                record.update(result='timed_out', exit_code=process.returncode)
    except BaseException as error:
        if process is not None and process.poll() is None:
            stop_child(process)
        record.update(result='launcher_failed', error_type=type(error).__name__)
        raise
    finally:
        record['wall_seconds'] = time.monotonic() - begun
        record['output_directory'] = str(out)
        status.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--timeout-seconds', type=int, default=600)
    args = parser.parse_args()
    result = run(args.workspace, args.device, args.timeout_seconds)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['result'] == 'completed' else 1)
