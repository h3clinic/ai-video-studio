"""Deploy only the audited cache-admission repair before the existing evaluator starts."""
import hashlib
import json
from pathlib import Path
import time
import zipfile

from cloud.jupyter_transfer import JupyterClient
from cloud.runpod_control import api, load_key

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cloud/a14b_remote_eval_v2/cache_patch'


def main():
    session = json.loads((OUT.parent / 'guard/session.json').read_text())
    if time.time() > session['remote_deadline_epoch'] - 750:
        raise RuntimeError('Not enough guarded time to patch before evaluation')
    pod = api()
    if pod['id'] != session['pod'] or pod['desiredStatus'] != 'RUNNING':
        raise RuntimeError('Existing evaluation Pod is not running')
    if float(pod['env']['GAUSSIAN_STOP_DEADLINE']) != session['remote_deadline_epoch']:
        raise RuntimeError('Session guard mismatch')
    client = JupyterClient('https://wdq34cebh6111k-8888.proxy.runpod.net',
        token=pod['env']['JUPYTER_PASSWORD'], known_tokens=(load_key(),))
    job = json.loads(client.read_bytes('workspace/remote_eval2/job.json'))
    if job['status'] != 'download':
        raise RuntimeError('Patch only allowed while downloader runs; never during evaluation')
    with zipfile.ZipFile(ROOT / 'artifacts/cloud/a14b_eval_bundle_v2.zip') as archive:
        expected = archive.read('cloud/evaluate_a14b.py')
    old = client.read_bytes('workspace/work/eval2/cloud/evaluate_a14b.py')
    if old != expected:
        raise RuntimeError('Remote evaluator differs from exact uploaded bundle')
    OUT.mkdir(exist_ok=False)
    (OUT / 'evaluate_a14b_original.py').write_bytes(old)
    report = {'epoch': time.time(), 'job_before': job, 'patches': [],
        'scope': 'Release only verified model file cache, retain unchanged 192GiB admission gate; no weights or experiment changes.'}
    for name in ('model_cache_release.py', 'evaluate_a14b.py'):
        data = (ROOT / 'cloud' / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        (OUT / name).write_bytes(data)
        result = client.put_bytes('workspace/work/eval2/cloud/' + name, data,
            expected_sha256=digest, replace_corrupt=(name == 'evaluate_a14b.py'))
        report['patches'].append(result)
    report['job_after'] = json.loads(client.read_bytes('workspace/remote_eval2/job.json'))
    (OUT / 'deployment.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
