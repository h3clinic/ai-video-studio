"""Submit fixed code to an already-connected GPU; never allocate or buy one."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import zipfile
from .studio_agent_tasks import validate_decision

SOURCES=('cloud/studio_gaussian_worker.py','real_video/gaussian_agent_edits.py',
         'real_video/gaussian_part_ownership.py','real_video/gaussian3d.py')


def execute_remote_edit(studio,job,asset,decision,remote):
    from cloud.jupyter_transfer import JupyterClient
    validate_decision(decision,job['agentId'],job['partId'])
    pod=studio.provider(remote['podId'])
    if pod.get('desiredStatus')!='RUNNING':raise ValueError('Remote GPU is not running')
    # This path consumes an already running session; it must have a bounded guard.
    deadline=float((pod.get('env') or {}).get('GAUSSIAN_STOP_DEADLINE',0))
    if not 120<deadline-time.time()<=1260:raise ValueError('Remote session lacks a current bounded shutdown guard')
    token=pod.get('env',{}).get('JUPYTER_PASSWORD')
    if not token:raise ValueError('Missing remote authentication')
    client=JupyterClient(remote['baseUrl'],allowed_host=remote['podId']+'-8888.proxy.runpod.net',token=token)
    events=[json.loads(line) for line in client.read_bytes('workspace/gaussian_shutdown_events.jsonl').decode().splitlines()]
    matching=[event for event in events if event.get('event')=='armed' and event.get('pod')==remote['podId'] and abs(event.get('deadline',0)-deadline)<1 and type(event.get('pid')) is int]
    if not matching:raise ValueError('Current remote watchdog identity not verified')
    watchdog_pid=matching[-1]['pid']
    directory='workspace/studio_'+job['id'];client.ensure_directory(directory)
    request=dict(agent_id=job['agentId'],part_id=job['partId'],asset_sha256=asset['sha256'],decision=decision)
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for relative in SOURCES:archive.write(studio.root/relative,relative)
        archive.writestr('cloud/__init__.py','');archive.writestr('real_video/__init__.py','')
        archive.write(studio.root/asset['path'],'asset.npz')
        archive.writestr('request.json',json.dumps(request))
    data=bundle.getvalue()
    if len(data)>8*2**20:raise ValueError('Remote edit bundle exceeds transport limit')
    pin=hashlib.sha256(data).hexdigest()
    client.put_bytes(directory+'/input.zip',data,expected_sha256=pin)
    output=studio.directory/job['id'];output.mkdir(exist_ok=True)
    limit=min(540,int(deadline-time.time())-90)
    if limit<30:raise ValueError('Not enough guarded time for execution')
    # Only server-controlled literals enter code. Gemini output is data in ZIP.
    code=f'''import pathlib,zipfile,hashlib,subprocess,sys,json,os
os.kill({watchdog_pid},0)
p=pathlib.Path('/{directory}');raw=(p/'input.zip').read_bytes()
assert hashlib.sha256(raw).hexdigest()=={pin!r}
with zipfile.ZipFile(p/'input.zip') as z:
    assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
    z.extractall(p)
env=dict(os.environ)
for k in ('RUNPOD_SHUTDOWN_KEY','JUPYTER_PASSWORD'):env.pop(k,None)
env['PYTHONPATH']=str(p)
with (p/'worker.log').open('w') as log:
    r=subprocess.run([sys.executable,'-m','cloud.studio_gaussian_worker'],cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,timeout={limit})
assert r.returncode==0,'Remote worker failed; inspect worker.log'
with zipfile.ZipFile(p/'result.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted((p/'output').iterdir()):z.write(f,f.name)
assert (p/'result.zip').stat().st_size<=8*2**20,'Result exceeds bound'
print('Remote Gaussian edit finished; quality review required')
'''
    client.execute(code,output/'remote.log',timeout_seconds=min(600,limit+30),kernel_path='workspace')
    raw=client.read_bytes(directory+'/result.zip')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(f.file_size for f in archive.infolist())>32*2**20:raise ValueError('Expanded result too large')
        if any(PurePosixPath(n).name!=n or '\\' in n or n.startswith('.') for n in archive.namelist()):raise ValueError('Unsafe result')
        result=output/'remote';result.mkdir(exist_ok=False);archive.extractall(result)
    report=json.loads((result/'report.json').read_text())
    video=result/'gaussian_motion.mp4'
    if hashlib.sha256(video.read_bytes()).hexdigest()!=report['video_sha256']:raise ValueError('Video checksum mismatch')
    return dict(output=str(video.relative_to(studio.root)),stage='Remote canonical Gaussian motion rendered; not full-scene/eating generation',
        review='Unreviewed canonical diagnostic; no quality acceptance',report=report)
