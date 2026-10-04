"""Migrate only experiment data to the existing 4090; bounded remote run."""
import os
os.environ['GAUSSIAN_RUNPOD_TARGET']='k5494yeo0hf0b7'
import hashlib
import json
import time
import zipfile
from pathlib import Path
from cloud.runpod_control import api,load_key,request,awake,POD
from cloud.runpod_guarded_session import ROOT,launch,restore
from cloud.jupyter_transfer import JupyterClient,TransportError

OUT=ROOT/'artifacts/cloud/object_edit_semantic_4090_v1'
REMOTE='workspace/object_edit_semantic_v1'
WORKER = os.environ.get('GAUSSIAN_SEMANTIC_WORKER','semantic-static-v1')
if WORKER not in ('semantic-static-v1','observed-parts-v1'):
    raise ValueError('Unknown fixed remote worker revision')
JOB=r'''
import os,sys,time,json,subprocess,zipfile,hashlib,shutil
from pathlib import Path
p=Path('/workspace/object_edit_semantic_v1');os.chdir(p)
env=dict(os.environ)
for key in ('RUNPOD_SHUTDOWN_KEY','JUPYTER_PASSWORD'):env.pop(key,None)
env.update(PYTHONPATH=str(p),HF_HOME=str(p/'model_cache'),GAUSSIAN_PRIOR_STATE=str(p/'prior_state'),HF_HUB_ENABLE_HF_TRANSFER='0')
status={'status':'preflight','stages':{}}
def save(): (p/'job.json').write_text(json.dumps(status,indent=2))
def run(name,args,limit):
    status['status']=name;save();t=time.monotonic()
    with (p/(name+'.log')).open('w') as log:
        result=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=limit)
    status['stages'][name]={'seconds':time.monotonic()-t,'exit':result.returncode};save()
    if result.returncode:raise RuntimeError(name+' failed')
try:
    import psutil
    if psutil.virtual_memory().available<8*2**30 or shutil.disk_usage(p).free<5*2**30:raise RuntimeError('Resource gate')
    # The orchestrator remains alive while the child holds the GPU lock. Do
    # not initialize Torch CUDA here: that makes our own parent look like a
    # competing compute process to the child's exclusive-GPU preflight.
    probe=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
        capture_output=True,text=True,timeout=10,check=True,env=env)
    devices=[name.strip() for name in probe.stdout.splitlines() if name.strip()]
    if len(devices)!=1:raise RuntimeError('Exactly one remote GPU required')
    status['gpu']=devices[0]
    status['gpu_preflight']='nvidia-smi subprocess; parent does not initialize Torch CUDA'
    save()
    run('dependencies',[sys.executable,'-m','pip','install','transformers==4.51.3','opencv-python-headless','imageio','imageio-ffmpeg','safetensors'],180)
    run('experiment',[sys.executable,'-u','cloud/object_edit_semantic_worker.py'],600)
    status['status']='completed_review_required'
except Exception as error:
    status.update(status='failed',error_type=type(error).__name__,error=str(error))
finally:
    save()
    with zipfile.ZipFile(p/'results.zip','w',zipfile.ZIP_STORED) as z:
        if (p/'output').exists():
            for f in (p/'output').rglob('*'):
                if f.is_file():z.write(f,f.relative_to(p))
        for f in p.glob('*.log'):z.write(f,f.name)
        z.write(p/'job.json','job.json')
    parts=[]
    with (p/'results.zip').open('rb') as stream:
        for index in range(1000):
            data=stream.read(4*2**20)
            if not data:break
            name=f'result_{index:03d}.part';(p/name).write_bytes(data)
            parts.append(dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (p/'transfer.json').write_text(json.dumps({'parts':parts}))
'''
if WORKER == 'observed-parts-v1':
    JOB = JOB.replace('cloud/object_edit_semantic_worker.py','cloud/object_edit_tracked_worker.py')

def directory_bootstrap_code():
    # The provider image sometimes serves workspace metadata/files but fails
    # its full directory listing. Use the already authenticated kernel for one
    # exclusive mkdir, NOT an overwrite-authorizing interpretation of a 404.
    return """from pathlib import Path
root=Path('/workspace')
assert root.is_dir() and root.resolve()==root, 'Unexpected workspace root'
target=root/'object_edit_semantic_v1'
assert not target.is_symlink(), 'Refuse symlink target'
target.mkdir(mode=0o700,exist_ok=False)
print({'fresh_experiment_directory_created':True})
"""


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    report={'status':'preparing','accepted':False,'worker_revision':WORKER,'balance_observed_usd':float(os.environ['GAUSSIAN_BALANCE_OBSERVED_USD']) if os.environ.get('GAUSSIAN_BALANCE_OBSERVED_USD') else None,'balance_note':'Console observation, not a guaranteed spending cap.','session_ceiling_usd':.50,'pod':POD}
    def record(**kw):
        report.update(kw);(OUT/'control.json').write_text(json.dumps(report,indent=2));print(json.dumps(kw),flush=True)
    with zipfile.ZipFile(OUT/'inputs.zip','w',zipfile.ZIP_STORED) as z:
        for name in ('cloud/object_edit_semantic_worker.py','cloud/object_edit_worker.py','real_video/gaussian3d.py','real_video/local_object_repair.py'):
            z.write(ROOT/name,name)
        if WORKER == 'observed-parts-v1':
            for name in ('cloud/object_edit_tracked_worker.py','real_video/observed_part_placement.py'):
                z.write(ROOT/name,name)
        for name in ('cloud/__init__.py','real_video/__init__.py'):z.writestr(name,'')
        z.writestr('job.py',JOB)
        z.write(ROOT/'research/prior_manifest.json','prior_manifest.json')
        z.write(ROOT/'artifacts/cloud/object_edit_remote_v2/results/output/apple_canonical.npz','input_apple.npz')
        for f in sorted((ROOT/'artifacts/cloud/object_edit_remote_v2/results/output/state').glob('*.npz')):z.write(f,'prior_state/'+f.name)
    pin=hashlib.sha256((OUT/'inputs.zip').read_bytes()).hexdigest()
    pod=api()
    host=POD+'-8888.proxy.runpod.net'
    client=JupyterClient('https://'+host,allowed_host=host,token=pod['env']['JUPYTER_PASSWORD'],known_tokens=(load_key(),))
    try:
        # Reject unrelated work; both known Pods must remain serialized.
        session=launch(OUT/'guard',1200,.50);record(status='starting',session=session,input_sha256=pin)
        for _ in range(60):
            try:
                client.verify_service()
                raw=client.read_bytes('workspace/gaussian_shutdown_events.jsonl')
                events=[json.loads(line) for line in raw.decode().splitlines()]
                if not any(e.get('event')=='armed' and abs(e.get('deadline',0)-session['remote_deadline_epoch'])<1 for e in events):
                    raise TransportError('Current watchdog not armed yet')
                break
            except TransportError:
                time.sleep(3)
        else:raise RuntimeError('Remote startup timed out')
        record(status='authenticated_jupyter_ready')
        record(operation_stage='exclusive_directory_creation')
        client.execute(directory_bootstrap_code(),OUT/'directory.log',timeout_seconds=60,kernel_path='workspace')
        record(operation_stage='verify_experiment_directory')
        client.ensure_directory(REMOTE)
        record(operation_stage='upload_verified_chunks')
        record(status='uploading_inputs',uploaded_parts=0)
        parts=[]
        with (OUT/'inputs.zip').open('rb') as stream:
            for i in range(100):
                data=stream.read(4*2**20)
                if not data:break
                name=f'input_{i:03d}.part';partpin=hashlib.sha256(data).hexdigest()
                client.put_bytes(REMOTE+'/'+name,data,expected_sha256=partpin);parts.append(name)
                if i%8==0:record(status='uploading_inputs',uploaded_parts=len(parts))
        code=f'''import pathlib,hashlib,zipfile,subprocess,sys,os
p=pathlib.Path('/{REMOTE}')
assert not (p/'output').exists() and not (p/'job.json').exists(), 'Preserve previous remote experiment'
with (p/'inputs.zip').open('xb') as out:
    for name in {parts!r}:out.write((p/name).read_bytes())
assert hashlib.sha256((p/'inputs.zip').read_bytes()).hexdigest()=={pin!r}
with zipfile.ZipFile(p/'inputs.zip') as z:
    assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
    z.extractall(p)
env=dict(os.environ)
for key in ('RUNPOD_SHUTDOWN_KEY','JUPYTER_PASSWORD'):env.pop(key,None)
with (p/'controller.log').open('w') as log:
    worker=subprocess.Popen([sys.executable,str(p/'job.py')],cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({{'pid':worker.pid}})
'''
        record(status='inputs_verified_launching',uploaded_parts=len(parts))
        record(operation_stage='launch_kernel')
        client.execute(code,OUT/'launch.log',timeout_seconds=60,kernel_path='workspace')
        previous=None
        while time.time()<session['remote_deadline_epoch']-120:
            raw=client.read_bytes(REMOTE+'/transfer.json',missing_ok=True)
            state=client.read_bytes(REMOTE+'/job.json',missing_ok=True)
            if state:
                value=json.loads(state)
                if value['status']!=previous:previous=value['status'];record(remote=value)
            if raw:break
            time.sleep(10)
        else:raise RuntimeError('Retrieval deadline')
        manifest=json.loads(raw)
        with (OUT/'results.zip').open('xb') as stream:
            for part in manifest['parts']:
                name=part['name']
                if Path(name).name!=name or not name.endswith('.part'):raise RuntimeError('Invalid result path')
                data=client.read_bytes(REMOTE+'/'+name)
                if len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:raise RuntimeError('Hash mismatch')
                stream.write(data)
        with zipfile.ZipFile(OUT/'results.zip') as z:
            if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):raise RuntimeError('Unsafe archive')
            z.extractall(OUT/'results')
        record(status='retrieved',remote_job=json.loads((OUT/'results/job.json').read_text()))
    except Exception as error:
        record(status='failed',error=client._redact(str(error)),error_type=type(error).__name__,
            transport_requests=client.request_history)
    finally:
        request('stop')
        status=request();record(provider_status=status)
        if status['desiredStatus']=='EXITED':record(cleanup=restore())

if __name__=='__main__':
    with awake():main()
