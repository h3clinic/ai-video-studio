"""One authorized remote run; no auto-retry, top-ups or local model execution."""
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
from cloud.runpod_control import api,load_key,request,awake
from cloud.runpod_guarded_session import launch,restore,ROOT
from cloud.jupyter_transfer import JupyterClient,TransportError

OUT=ROOT/'artifacts/cloud/object_edit_remote_v2'
FILES=['cloud/object_edit_worker.py','real_video/gaussian3d.py',
       'real_video/gaussian_scene_objects.py','real_video/gaussian_object_agent.py',
       'tests/test_gaussian_object_agent.py','tests/test_gaussian_scene_objects.py',
       'tests/test_gaussian3d.py']

JOB=r'''
import os,sys,subprocess,json,time,hashlib,zipfile
from pathlib import Path
p=Path('/workspace/object_edit');os.chdir(p)
env=dict(os.environ);env.pop('RUNPOD_SHUTDOWN_KEY',None);env.pop('JUPYTER_PASSWORD',None)
env['HF_HOME']=str(p/'model_cache');env['PYTHONPATH']=str(p)
status={"status":"setup","stages":{}}
def record(): (p/'job.json').write_text(json.dumps(status,indent=2))
def command(name,args,timeout):
    status['status']=name;record();start=time.monotonic()
    with (p/(name+'.log')).open('w') as log:
        result=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    status['stages'][name]={'seconds':time.monotonic()-start,'exit':result.returncode};record()
    if result.returncode:raise RuntimeError(name+' failed')
try:
    import torch,psutil,shutil
    if psutil.virtual_memory().available<8*2**30 or shutil.disk_usage(p).free<20*2**30:raise RuntimeError('Resource gate')
    if not torch.cuda.is_available():raise RuntimeError('CUDA missing')
    status['gpu']=torch.cuda.get_device_name();record()
    command('dependencies',[sys.executable,'-m','pip','install','transformers==4.51.3','opencv-python-headless',
      'imageio','imageio-ffmpeg','ipywidgets','pytest',
      'git+https://github.com/openai/shap-e.git@50131012ee11c9d2617f3886c10f000d3c7a3b43'],480)
    command('tests',[sys.executable,'-m','pytest','-q','tests'],120)
    command('experiment',[sys.executable,'-u','cloud/object_edit_worker.py'],600)
    status['status']='completed_review_required'
except Exception as error:
    status['status']='failed';status['error_type']=type(error).__name__;status['error']=str(error)
finally:
    record()
    # Include all generated state, including failures. Chunks fit authenticated transport.
    with zipfile.ZipFile(p/'results.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for root in [p/'output']:
            if root.exists():
                for f in root.rglob('*'):
                    if f.is_file():archive.write(f,f.relative_to(p))
        for f in p.glob('*.log'):archive.write(f,f.name)
        archive.write(p/'job.json','job.json')
    parts=[]
    with (p/'results.zip').open('rb') as stream:
        for index in range(1000):
            data=stream.read(4*2**20)
            if not data:break
            name=f'result_{index:03d}.part';(p/name).write_bytes(data)
            parts.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (p/'transfer.json').write_text(json.dumps({'parts':parts}))
'''


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    report={'status':'preparing','budget_ceiling_usd':8,'this_launch_max_usd':2.20,
        'balance_before_usd':11.25,'accepted':False,'outputs':[]}
    def record(**values):
        report.update(values);(OUT/'control.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(values),flush=True)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name in FILES:bundle.writestr(name,(ROOT/name).read_bytes())
        for name in ('cloud/__init__.py','real_video/__init__.py','tests/__init__.py'):bundle.writestr(name,'')
        bundle.writestr('source.mp4',(ROOT/'artifacts/real_video/runway_agents/donkey_orange_v1/source/video.mp4').read_bytes())
        bundle.writestr('job.py',JOB)
        bundle.writestr('input_apple.npz',(ROOT/'artifacts/cloud/object_edit_remote_v1/results/output/apple_canonical.npz').read_bytes())
        for name in ('depth_config.json','depth_preprocessor.json','depth_revision.json'):
            bundle.writestr(name,(ROOT/'research'/name).read_bytes())
    data=buffer.getvalue();pin=hashlib.sha256(data).hexdigest()
    (OUT/'inputs.zip').write_bytes(data)
    pod=api();client=JupyterClient('https://wdq34cebh6111k-8888.proxy.runpod.net',
        token=pod['env']['JUPYTER_PASSWORD'],known_tokens=(load_key(),))
    try:
        session=launch(OUT/'guard',2100,2.20)
        record(status='waiting_for_server',session=session,input_sha256=pin)
        for _ in range(90):
            try:
                boot=client.read_bytes('workspace/gaussian_shutdown_events.jsonl')
                if b'"event": "armed"' not in boot:raise RuntimeError('Remote guard missing')
                (OUT/'guard_remote.jsonl').write_bytes(boot);break
            except TransportError:
                if time.time()>session['remote_deadline_epoch']-1700:raise RuntimeError('Startup budget exceeded')
                time.sleep(3)
        else:raise RuntimeError('Server unavailable')
        client.ensure_directory('workspace/object_edit')
        client.put_bytes('workspace/object_edit/inputs.zip',data,expected_sha256=pin)
        code=f'''import pathlib,hashlib,zipfile,subprocess,sys,os
p=pathlib.Path('/workspace/object_edit');data=(p/'inputs.zip').read_bytes()
assert hashlib.sha256(data).hexdigest()=={pin!r}
with zipfile.ZipFile(p/'inputs.zip') as z:
    assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
    z.extractall(p)
env=dict(os.environ);env.pop('RUNPOD_SHUTDOWN_KEY',None);env.pop('JUPYTER_PASSWORD',None)
with (p/'controller.log').open('w') as log:
    worker=subprocess.Popen([sys.executable,str(p/'job.py')],cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print({{'pid':worker.pid,'started':True}})
'''
        client.execute(code,OUT/'launch.log',timeout_seconds=60,kernel_path='workspace')
        previous=None
        while time.time()<session['remote_deadline_epoch']-300:
            transfer=client.read_bytes('workspace/object_edit/transfer.json',missing_ok=True)
            status=client.read_bytes('workspace/object_edit/job.json',missing_ok=True)
            if status:
                value=json.loads(status)
                if value['status']!=previous:previous=value['status'];record(remote=value)
            if transfer:break
            time.sleep(12)
        else:raise RuntimeError('Job exceeded retrieval deadline')
        manifest=json.loads(transfer);(OUT/'transfer.json').write_bytes(transfer)
        with (OUT/'results.zip').open('xb') as stream:
            for part in manifest['parts']:
                name=part['name']
                if Path(name).name!=name or not name.endswith('.part'):raise RuntimeError('Invalid result path')
                payload=client.read_bytes('workspace/object_edit/'+name)
                if len(payload)!=part['bytes'] or hashlib.sha256(payload).hexdigest()!=part['sha256']:raise RuntimeError('Result hash mismatch')
                stream.write(payload)
        with zipfile.ZipFile(OUT/'results.zip') as archive:
            if any(Path(n).is_absolute() or '..' in Path(n).parts for n in archive.namelist()):raise RuntimeError('Unsafe result archive')
            archive.extractall(OUT/'results')
        record(status='results_retrieved',remote_job=json.loads((OUT/'results/job.json').read_text()))
    except Exception as error:
        record(status='failed',error_type=type(error).__name__,error=client._redact(str(error)))
    finally:
        request('stop')
        for _ in range(12):
            status=request()
            if status['desiredStatus']=='EXITED':break
            time.sleep(3)
        record(provider_status=status)
        if status['desiredStatus']=='EXITED':record(cleanup=restore())


if __name__=='__main__':
    with awake():main()
