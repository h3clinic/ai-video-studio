"""Resume retrieval from the ONE existing evaluation; never launch/stop a Pod.

No uploads, installs, model downloads, training, or remote file writes. The
existing remote/local billing guards remain authoritative. Callers must stop
idle compute after retrieval or let those already-installed guards do so.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

from cloud.jupyter_transfer import JupyterClient, TransportError, OWNED_HOST

POD = 'wdq34cebh6111k'
ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT/'artifacts/cloud/a14b_remote_eval_v2'
REMOTE = 'workspace/remote_eval2'
MARKER = 'GAUSSIAN_RESUME_JSON '
VIDEO_NAMES = frozenset(f'{branch}_{seed}.mp4'
    for branch in ('original', 'lora_only', 'gaussian_memory')
    for seed in (103501, 103502))
LOG_NAMES = ('evaluation.log', 'preflight.log', 'dependencies.log',
             'environment.log', 'download.log')
CHUNK = 2*2**20
MAX_VIDEO = 64*2**20
MAX_LOG = 256*2**10


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def load_expected(directory=SESSION):
    directory = Path(directory)
    guard = json.loads((directory/'guard/session.json').read_text())
    control = json.loads((directory/'control.json').read_text())
    lines = (directory/'setup.log').read_text().splitlines()
    workers = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and 'worker_pid' in item:
            workers.append(item)
    if len(workers) != 1 or workers[0].get('bundle_verified') is not True:
        raise ValueError('Expected one verified existing worker record')
    pid = workers[0]['worker_pid']
    if type(pid) is not int or pid <= 1:
        raise ValueError('Invalid existing worker PID')
    keys = ('pod', 'start_epoch', 'remote_deadline_epoch', 'remote_guard_sha256', 'local_guard_pid')
    if guard.get('pod') != POD or any(guard.get(k) != control.get('guard', {}).get(k) for k in keys):
        raise ValueError('Stored controller/session identity mismatch')
    start, deadline = guard['start_epoch'], guard['remote_deadline_epoch']
    if not isinstance(start, (int, float)) or not isinstance(deadline, (int, float)) or not 0 < deadline-start <= 2400:
        raise ValueError('Invalid stored deadline; never extend a paid session')
    return dict(pod=POD, worker_pid=pid, start_epoch=start, deadline=deadline,
                remote_guard_sha256=guard['remote_guard_sha256'])


def atomic_bytes(path, data):
    """No-clobber publish; byte-identical already-retrieved files are reusable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.stat().st_size == len(data) and file_digest(path) == digest(data):
            return
        raise FileExistsError('Refusing to replace different retrieved evidence')
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.partial', delete=False) as stream:
        partial = Path(stream.name)
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(partial, path)
    except FileExistsError:
        if not path.is_file() or file_digest(path) != digest(data):
            raise
    partial.unlink()  # Only this successful temporary, never previous evidence.


def probe_code(expected):
    """Bounded read-only remote identity and completed-file manifest."""
    return '''import os,json,time,hashlib,pathlib,stat
expected = '''+repr(expected)+'''
base=pathlib.Path('/workspace/remote_eval2')
assert os.environ.get('RUNPOD_POD_ID') == expected['pod'], 'Wrong Pod'
assert abs(float(os.environ['GAUSSIAN_STOP_DEADLINE'])-expected['deadline']) < .01, 'Wrong session deadline'
events=pathlib.Path('/workspace/gaussian_shutdown_events.jsonl').read_text().splitlines()
assert any((lambda r:r.get('event')=='armed' and abs(r.get('deadline',0)-expected['deadline'])<.01)(json.loads(x)) for x in events), 'Guard identity mismatch'
def small(path, maximum):
    assert not path.is_symlink() and path.is_file() and path.stat().st_size <= maximum
    with path.open('rb') as stream: value=stream.read(maximum+1)
    assert len(value)<=maximum
    return value
raw=small(base/'job.json',65536);job=json.loads(raw)
assert job.get('model_training') is False and job.get('accepted') is False, 'Unexpected job contract'
proc=pathlib.Path('/proc')/str(expected['worker_pid'])
alive=proc.exists()
if alive:
    cmd=(proc/'cmdline').read_bytes().split(bytes([0]))
    assert b'/workspace/a14b_remote_job.py' in cmd, 'PID reused by another command'
    assert (proc/'cwd').resolve()==pathlib.Path('/workspace/work/eval2'), 'Worker directory mismatch'
else:
    assert job['status'] in ('failed','completed_visual_review_required'), 'Worker missing without terminal evidence'
result={'pod':expected['pod'],'worker_pid':expected['worker_pid'],'deadline':expected['deadline'], 'worker_alive':alive,'job':job,'job_sha256':hashlib.sha256(raw).hexdigest(),'videos':[],'errors':[]}
report_path=base/'evaluation/report.json'
if report_path.exists():
    raw=small(report_path,1048576);report=json.loads(raw)
    assert report.get('training') is False and report.get('native_gaussian_generation') is False, 'Wrong evaluation contract'
    result['report_sha256']=hashlib.sha256(raw).hexdigest()
    result['report_bytes']=len(raw)
    for value in report.get('videos',{}).values():
        name=value['file']
        assert name in '''+repr(VIDEO_NAMES)+''', 'Unexpected video path'
        try:
            path=base/'evaluation'/name
            assert not path.is_symlink() and path.is_file()
            size=path.stat().st_size
            assert 0<size<=67108864
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(2097152),b''):h.update(chunk)
            assert h.hexdigest()==value['sha256'], 'Report/video digest mismatch'
            result['videos'].append({'file':name,'bytes':size,'sha256':h.hexdigest()})
        except Exception as error:
            result['errors'].append({'file':name,'error_type':type(error).__name__})
print('''+repr(MARKER)+'''+json.dumps(result))
'''


def chunk_code(name, offset, count, expected_size):
    if name not in VIDEO_NAMES or type(offset) is not int or offset < 0 or not 0 < count <= CHUNK or not 0 < expected_size <= MAX_VIDEO:
        raise ValueError('Invalid bounded video chunk request')
    if offset+count > expected_size:
        raise ValueError('Chunk exceeds declared video')
    return '''import pathlib,hashlib,base64,json
p=pathlib.Path('/workspace/remote_eval2/evaluation')/'''+repr(name)+'''
assert not p.is_symlink() and p.is_file() and p.stat().st_size=='''+str(expected_size)+'''
with p.open('rb') as stream:
    stream.seek('''+str(offset)+''');data=stream.read('''+str(count)+''')
assert len(data)=='''+str(count)+'''
print('''+repr(MARKER)+'''+json.dumps({'offset':'''+str(offset)+''','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}))
'''


def log_code(name):
    if name not in LOG_NAMES:
        raise ValueError('Unexpected diagnostic log')
    return '''import pathlib,hashlib,base64,json
p=pathlib.Path('/workspace/remote_eval2')/'''+repr(name)+'''
result={'file':p.name,'missing':not p.exists()}
if p.exists():
    assert not p.is_symlink() and p.is_file()
    size=p.stat().st_size
    with p.open('rb') as stream:
        stream.seek(max(0,size-262144));data=stream.read(262144)
    result.update({'source_bytes':size,'tail_only':size>len(data),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()})
print('''+repr(MARKER)+'''+json.dumps(result))
'''


class Retriever:
    def __init__(self, client, expected, out, *, clock=time.time, sleep=time.sleep):
        self.client, self.expected, self.out = client, expected, Path(out)
        self.clock, self.sleep = clock, sleep
        self.cutoff = expected['deadline']-45
        self.out.mkdir(parents=True, exist_ok=True)
        self.results = self.out/'results'
        self.errors = []
        self.videos = {}
        self.latest_report = None

    def retry(self, call, attempts=3):
        for attempt in range(attempts):
            if self.clock()+20 >= self.cutoff:
                raise TimeoutError('Existing retrieval deadline reached; guard unchanged')
            try:
                return call()
            except (TransportError, OSError, ValueError) as error:
                self.errors.append({'error_type':type(error).__name__, 'attempt':attempt+1})
                if attempt+1 == attempts:
                    raise
                self.sleep(min(2**attempt, 4))

    def remote_json(self, code):
        def attempt():
            log=self.out/'probes'/('probe-'+uuid.uuid4().hex+'.log')
            log.parent.mkdir(exist_ok=True)
            timeout=min(20, self.cutoff-self.clock()-15)
            if timeout < 1:raise TimeoutError('No guarded time for inspection')
            self.client.execute(code, log, timeout_seconds=timeout, kernel_path='workspace')
            lines=log.read_text(encoding='utf-8').splitlines()
            values=[line[len(MARKER):] for line in lines if line.startswith(MARKER)]
            if len(values)!=1:raise ValueError('Missing unique read-only probe response')
            return json.loads(values[0])
        return self.retry(attempt)

    def probe(self):
        value=self.remote_json(probe_code(self.expected))
        if any(value.get(k)!=self.expected[k] for k in ('pod','worker_pid','deadline')):
            raise ValueError('Remote session identity mismatch')
        atomic_bytes(self.out/'manifests'/('manifest-'+digest(json.dumps(value,sort_keys=True).encode())+'.json'),
                     json.dumps(value,indent=2).encode())
        return value

    def fetch_video(self, item):
        name, size, expected=item['file'],item['bytes'],item['sha256']
        if name not in VIDEO_NAMES or type(size) is not int or not 0<size<=MAX_VIDEO or not isinstance(expected,str) or len(expected)!=64:
            raise ValueError('Invalid video manifest')
        target=self.results/'evaluation'/name
        if target.exists():
            if target.stat().st_size!=size or file_digest(target)!=expected:
                raise FileExistsError('Existing video differs; preserved')
        elif size<=8*2**20:
            data=self.retry(lambda:self.client.read_bytes(REMOTE+'/evaluation/'+name))
            if len(data)!=size or digest(data)!=expected:raise ValueError('Video transfer digest mismatch')
            atomic_bytes(target,data)
        else:
            chunks=[]
            for offset in range(0,size,CHUNK):
                count=min(CHUNK,size-offset)
                directory=self.out/'chunks'/name/expected
                part=directory/(str(offset)+'.bin');sidecar=directory/(str(offset)+'.json')
                if part.exists() and sidecar.exists():
                    meta=json.loads(sidecar.read_text())
                    if part.stat().st_size!=count or file_digest(part)!=meta['sha256']:
                        raise ValueError('Saved partial chunk corrupted; preserved')
                else:
                    value=self.remote_json(chunk_code(name,offset,count,size))
                    data=base64.b64decode(value['base64'],validate=True)
                    if value['offset']!=offset or value['bytes']!=count or len(data)!=count or digest(data)!=value['sha256']:
                        raise ValueError('Video chunk digest mismatch')
                    atomic_bytes(part,data)
                    atomic_bytes(sidecar,json.dumps({'sha256':digest(data),'bytes':count}).encode())
                chunks.append(part)
            # At most 64 MiB; local retrieval only, never model memory/work.
            data=b''.join(p.read_bytes() for p in chunks)
            if len(data)!=size or digest(data)!=expected:raise ValueError('Assembled video digest mismatch')
            atomic_bytes(target,data)
        self.videos[name]={'path':str(target),'bytes':size,'sha256':expected}

    def harvest(self, manifest):
        # Completed videos come before reports/logs that may be changing or huge.
        for item in manifest.get('videos',[]):
            try:self.fetch_video(item)
            except (TransportError,OSError,ValueError) as error:
                self.errors.append({'file':item.get('file'),'error_type':type(error).__name__})
        if 'report_sha256' in manifest:
            try:
                data=self.retry(lambda:self.client.read_bytes(REMOTE+'/evaluation/report.json'))
                if digest(data)!=manifest['report_sha256']:
                    # The atomic report advanced; retry on the next manifest.
                    raise ValueError('Report advanced after manifest')
                path=self.results/'evaluation'/('report-'+digest(data)+'.json')
                atomic_bytes(path,data);self.latest_report=str(path)
            except (TransportError,OSError,ValueError) as error:
                self.errors.append({'file':'report.json','error_type':type(error).__name__})
        job=json.dumps(manifest['job'],indent=2).encode()
        atomic_bytes(self.results/('job-'+digest(job)+'.json'),job)

    def logs(self):
        for name in LOG_NAMES:
            try:
                value=self.remote_json(log_code(name))
                if value['missing']:continue
                data=base64.b64decode(value['base64'],validate=True)
                if len(data)>MAX_LOG or len(data)!=value['bytes'] or digest(data)!=value['sha256']:
                    raise ValueError('Log-tail integrity mismatch')
                atomic_bytes(self.results/'logs'/(name+'.tail-'+digest(data)),data)
                atomic_bytes(self.results/'logs'/(name+'.metadata-'+digest(data)+'.json'),
                    json.dumps({k:v for k,v in value.items() if k!='base64'},indent=2).encode())
            except (TransportError,OSError,ValueError,TimeoutError) as error:
                self.errors.append({'file':name,'error_type':type(error).__name__})

    def run(self):
        terminal=False;last_job=None;last_manifest=None
        while self.clock()+35<self.cutoff:
            try:
                manifest=self.probe();last_manifest=manifest;last_job=manifest['job']['status']
                self.harvest(manifest)
                terminal=last_job in ('failed','completed_visual_review_required')
                complete=(last_job=='completed_visual_review_required' and set(self.videos)==VIDEO_NAMES and self.latest_report is not None)
                if complete or (last_job=='failed' and all(v['file'] in self.videos for v in manifest.get('videos',[]))):
                    break
            except (TransportError,OSError,ValueError,TimeoutError) as error:
                self.errors.append({'phase':'poll_or_harvest','error_type':type(error).__name__})
            self.sleep(min(15,max(0,self.cutoff-self.clock()-35)))
        self.logs()
        summary=dict(status='retrieved' if set(self.videos)==VIDEO_NAMES and self.latest_report else 'partial_or_failed',
            pod=POD,worker_pid=self.expected['worker_pid'],existing_deadline=self.expected['deadline'],
            remote_stage=last_job,terminal_observed=terminal,outputs=list(self.videos.values()),
            latest_report=self.latest_report,errors=self.errors,accepted=False,training=False,
            launched_compute=False,pod_stop_requested=False,
            next_action='Inspect results and stop the existing idle Pod; original guards remain unchanged.')
        path=self.out/('summary-'+str(time.time_ns())+'.json')
        atomic_bytes(path,json.dumps(summary,indent=2).encode())
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=SESSION/'resume_retrieval_v1')
    args=parser.parse_args()
    expected=load_expected()
    if time.time()+90>=expected['deadline']:raise RuntimeError('No time left in the existing session')
    # Only authenticated GET; no launch, restart, stop, patch, or account changes.
    from cloud.runpod_control import api, load_key, awake
    pod=api()
    if pod.get('id')!=POD or pod.get('desiredStatus')!='RUNNING':raise RuntimeError('Existing approved Pod is not running')
    if abs(float(pod.get('env',{}).get('GAUSSIAN_STOP_DEADLINE',0))-expected['deadline'])>.01:
        raise RuntimeError('Existing Pod deadline differs from stored session')
    client=JupyterClient('https://'+OWNED_HOST,token=pod['env']['JUPYTER_PASSWORD'],
        known_tokens=(load_key(),),request_timeout=8)
    with awake():result=Retriever(client,expected,args.out).run()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
