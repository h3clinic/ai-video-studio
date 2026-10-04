"""Explicit authorized start with remote boot guard and independent local fallback.

Each invocation is bounded; no loops of paid launches. CLI action is permission
to start ONE already-existing Pod only. No new purchases or top-ups supported.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from cloud.runpod_control import api, request, load_key, POD, STORE

ROOT=Path(__file__).resolve().parents[1]


def launch(out,seconds,max_cost):
    if not 30<=seconds<=2100 or not 0<max_cost<=3.24: raise ValueError('Invalid bounded authorization')
    pod=api()
    if pod.get('desiredStatus')!='EXITED': raise RuntimeError('Pod must be stopped; no duplicate')
    rate=float(pod['costPerHr'])
    if not math.isfinite(rate) or rate<0 or rate+.03>3.53 or (rate+.03)*(seconds+120)/3600>max_cost:
        raise RuntimeError('Current rate/time exceeds explicit budget')
    if pod.get('dockerEntrypoint') not in (None,[]): raise RuntimeError('Unexpected image entrypoint')
    command=pod.get('dockerStartCmd') or []
    if command not in ([],['/start.sh']) and not (command[:3]==['python3','-u','-c'] and '<gaussian_boot_guard>' in command[-1]):
        raise RuntimeError('Unexpected existing user start command')
    out.mkdir(parents=True,exist_ok=False)
    code=(ROOT/'cloud/runpod_boot_guard.py').read_bytes()
    deadline=time.time()+seconds
    env=dict(pod.get('env') or {})
    env['RUNPOD_SHUTDOWN_KEY']=load_key()
    env['GAUSSIAN_STOP_DEADLINE']=str(deadline)
    env['GAUSSIAN_GUARD_POD']=POD
    cmd=['python3','-u','-c',"import base64;exec(compile(base64.b64decode("+repr(base64.b64encode(code).decode())+"),'<gaussian_boot_guard>','exec'))"]
    api('PATCH',payload={'dockerStartCmd':cmd,'env':env})
    configured=api()
    if configured.get('dockerStartCmd')!=cmd or configured.get('env',{}).get('GAUSSIAN_STOP_DEADLINE')!=str(deadline):
        raise RuntimeError('Remote guard readback failed; not starting')
    # This process has no ability to start compute: it can only stop it.
    flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
    watcher=subprocess.Popen([sys.executable,'-m','cloud.runpod_control','watch','--seconds',str(seconds+60),
        '--out',str(out/'local_fallback')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,creationflags=flags)
    arm=out/'local_fallback/events.jsonl'
    for _ in range(15):
        if arm.exists() and '"event": "armed"' in arm.read_text():break
        if watcher.poll() is not None: raise RuntimeError('Fallback failed; not starting')
        time.sleep(1)
    else: raise RuntimeError('Fallback not armed; not starting')
    report=dict(pod=POD,start_epoch=time.time(),remote_deadline_epoch=deadline,
        max_authorized_cost=max_cost,rate_with_storage_bound=3.53,local_guard_pid=watcher.pid,
        remote_guard_sha256=hashlib.sha256(code).hexdigest(),status='configured_not_started',
        limitations='Watchdogs require functioning host/API; not provider-guaranteed cap.')
    report_path=out/'session.json'
    report_path.write_text(json.dumps(report,indent=2))
    try:
        api('POST','/start')
        report['status']='start_requested';report['provider_status']=request()
    except Exception:
        request('stop');report['status']='start_failed_stop_requested';raise
    finally: report_path.write_text(json.dumps(report,indent=2))
    return report


def restore():
    pod=api()
    if pod.get('desiredStatus')!='EXITED': raise RuntimeError('Only restore a stopped Pod')
    env=dict(pod.get('env') or {})
    env.pop('RUNPOD_SHUTDOWN_KEY',None);env.pop('GAUSSIAN_STOP_DEADLINE',None)
    env.pop('GAUSSIAN_GUARD_POD',None)
    api('PATCH',payload={'dockerStartCmd':['/start.sh'],'env':env})
    return {'restored_default_startup':True,'shutdown_key_removed_from_pod':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('launch','restore'))
    parser.add_argument('--out',type=Path);parser.add_argument('--seconds',type=int)
    parser.add_argument('--max-cost',type=float)
    args=parser.parse_args()
    if args.action=='launch':
        if None in (args.out,args.seconds,args.max_cost):parser.error('Explicit output, seconds, cost required')
        result=launch(args.out,args.seconds,args.max_cost)
    else:result=restore()
    print(json.dumps(result,indent=2))
