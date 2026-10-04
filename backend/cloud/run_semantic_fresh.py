"""One fresh-host 4090 allocation, existing credit, no top-ups or volume.

Starts with a remote shutdown guard already in its command. Does not delete Pods.
"""
import base64
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error
from cloud.runpod_control import load_key,NoRedirect,classify_provider_error,awake
from cloud.runpod_guarded_session import ROOT

def run_identity(run_id=None):
    """One explicit fresh artifact/pod name; no paths supplied by callers."""
    if run_id is None:
        return ROOT/'artifacts/cloud/object_edit_fresh4090_v4', 'gaussian-semantic-fresh-20261004-v3'
    if not isinstance(run_id,str) or not re.fullmatch(r'[a-z0-9_-]{1,60}',run_id):
        raise ValueError('GAUSSIAN_FRESH_RUN_ID must match [a-z0-9_-]{1,60}')
    return ROOT/'artifacts/cloud'/run_id, run_id


OUT,NAME=run_identity(os.environ.get('GAUSSIAN_FRESH_RUN_ID'))

def rest(method,path,payload=None):
    headers={'Authorization':'Bearer '+load_key(),'User-Agent':'GaussianVideoBudgetGuard/1'}
    if payload is not None:headers['Content-Type']='application/json'
    request=urllib.request.Request('https://rest.runpod.io/v1/'+path,method=method,
        data=None if payload is None else json.dumps(payload).encode(),headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(request,timeout=30) as response:
            raw=response.read(2*2**20)
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'RunPod HTTP {error.code}: '+classify_provider_error(error.read(65536))) from None

def watch():
    manifest=json.loads((OUT/'allocation.json').read_text())
    pod=manifest['pod']
    if manifest['name']!=NAME or not pod.isalnum():raise RuntimeError('Invalid owned session')
    deadline=manifest['remote_deadline_epoch']
    if deadline-time.time()>1260:raise RuntimeError('Invalid watchdog deadline')
    (OUT/'watch_armed.json').write_text(json.dumps({'pod':pod,'deadline':deadline}))
    while time.time()<deadline:
        # A completed run has already independently verified shutdown. Recheck
        # the provider before releasing our process-scoped sleep prevention.
        final_path=OUT/'final_status.json'
        if final_path.exists():
            try:
                final=json.loads(final_path.read_text())
                if (final.get('id')==pod and final.get('desiredStatus')=='EXITED'
                        and rest('GET',f'pods/{pod}').get('desiredStatus')=='EXITED'):
                    return
            except Exception:pass
        time.sleep(min(1,max(0,deadline-time.time())))
    for attempt in range(60):
        try:
            rest('POST',f'pods/{pod}/stop')
            if rest('GET',f'pods/{pod}').get('desiredStatus')=='EXITED':return
        except Exception:pass
        time.sleep(5)
    (OUT/'STOP_FAILED.json').write_text(json.dumps({'pod':pod,'requires_attention':True}))

def main():
    balance=float(os.environ.get('GAUSSIAN_BALANCE_OBSERVED_USD','nan'))
    observed=float(os.environ.get('GAUSSIAN_BALANCE_OBSERVED_EPOCH','nan'))
    if not math.isfinite(balance) or balance < .50 or not math.isfinite(observed) or not 0 <= time.time()-observed <= 1800:
        raise RuntimeError('A fresh explicit balance observation and at least $0.50 credit are required; no top-ups')
    OUT.mkdir(exist_ok=False)
    pods=rest('GET','pods')
    if not isinstance(pods,list):raise RuntimeError('Unexpected pod listing')
    if any(p.get('name')==NAME for p in pods):raise RuntimeError('Existing allocation: do not duplicate')
    if any(p.get('desiredStatus')=='RUNNING' for p in pods):raise RuntimeError('Serialize GPU use; existing running pod')
    deadline=time.time()+1200
    code=(ROOT/'cloud/runpod_boot_guard.py').read_text()
    code=code.replace("POD = os.environ.get('GAUSSIAN_GUARD_POD','wdq34cebh6111k')\nif POD not in ('wdq34cebh6111k','k5494yeo0hf0b7'):\n    raise RuntimeError('Unexpected guarded Pod')",
        "POD = os.environ['RUNPOD_POD_ID']\nif not POD.isalnum(): raise RuntimeError('Invalid provider Pod ID')")
    if "POD = os.environ['RUNPOD_POD_ID']" not in code:raise RuntimeError('Guard specialization failed')
    import secrets
    jupyter=secrets.token_urlsafe(32)
    payload=dict(name=NAME,cloudType='SECURE',computeType='GPU',gpuCount=1,
        gpuTypeIds=['NVIDIA GeForce RTX 4090'],gpuTypePriority='availability',
        imageName='runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404',
        containerDiskInGb=30,volumeInGb=0,ports=['8888/http'],minRAMPerGPU=16,
        dockerStartCmd=['python3','-u','-c',"import base64;exec(compile(base64.b64decode("+repr(base64.b64encode(code.encode()).decode())+"),'<gaussian_boot_guard>','exec'))"],
        env={'RUNPOD_SHUTDOWN_KEY':load_key(),'GAUSSIAN_STOP_DEADLINE':str(deadline),
             'JUPYTER_PASSWORD':jupyter},interruptible=False)
    # Never retry an ambiguous create; reconcile by the unique experiment name.
    try:pod=rest('POST','pods',payload)
    except Exception:
        found=[p for p in rest('GET','pods') if p.get('name')==NAME]
        if len(found)!=1:raise
        pod=found[0]
    pid=pod['id']
    allocation=dict(pod=pid,name=NAME,remote_deadline_epoch=deadline,start_epoch=time.time(),
        rate_usd_per_hour=float(pod['costPerHr']),max_authorized_cost=.50,
        balance_observed_usd=balance,balance_observed_epoch=observed,
        limitations='Watchdogs are not provider-guaranteed caps; no persistent volume; stop after retrieval.')
    (OUT/'allocation.json').write_text(json.dumps(allocation,indent=2))
    print(json.dumps(allocation),flush=True)
    try:
        if float(pod['costPerHr'])>.75:raise RuntimeError('Unexpected rate; stopping immediately')
        flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
        subprocess.Popen([sys.executable,'-m','cloud.run_semantic_fresh','--watch'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=flags)
        for _ in range(15):
            if (OUT/'watch_armed.json').exists():break
            time.sleep(1)
        else:raise RuntimeError('Local fallback failed')
        from cloud import run_semantic_alternative as runner
        runner.OUT=OUT/'experiment';runner.POD=pid
        runner.api=lambda method='GET',suffix='',payload=None:rest(method,'pods/'+pid+suffix,payload)
        def status(action='status'):
            if action=='stop':rest('POST','pods/'+pid+'/stop');return {'stop_requested':True}
            obj=rest('GET','pods/'+pid)
            return {k:obj.get(k) for k in ('id','desiredStatus','costPerHr')}
        runner.request=status
        runner.launch=lambda *args:allocation
        def restore():
            obj=rest('GET','pods/'+pid)
            if obj.get('desiredStatus')!='EXITED':raise RuntimeError('Must be stopped')
            env=dict(obj.get('env') or {})
            env.pop('RUNPOD_SHUTDOWN_KEY',None);env.pop('GAUSSIAN_STOP_DEADLINE',None)
            rest('PATCH','pods/'+pid,{'dockerStartCmd':['/start.sh'],'env':env})
            return {'temporary_shutdown_key_removed':True}
        runner.restore=restore
        runner.main()
    finally:
        rest('POST','pods/'+pid+'/stop')
        final=rest('GET','pods/'+pid)
        (OUT/'final_status.json').write_text(json.dumps({k:final.get(k) for k in ('id','desiredStatus','costPerHr')},indent=2))

if __name__=='__main__':
    with awake():
        if '--watch' in sys.argv:watch()
        else:main()
