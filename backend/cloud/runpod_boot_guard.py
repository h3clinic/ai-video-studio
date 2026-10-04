"""Remote startup watchdog: independent of browser, chat and laptop connection.

Starts the official image entrypoint, then asks RunPod to stop this exact Pod at
an absolute deadline. No download or GPU workload. Provider/API failures remain
possible; this is not a provider-guaranteed spending cap.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error

POD = os.environ.get('GAUSSIAN_GUARD_POD','wdq34cebh6111k')
if POD not in ('wdq34cebh6111k','k5494yeo0hf0b7'):
    raise RuntimeError('Unexpected guarded Pod')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None


def main():
    deadline = float(os.environ['GAUSSIAN_STOP_DEADLINE'])
    secret = os.environ['RUNPOD_SHUTDOWN_KEY']
    if os.environ.get('RUNPOD_POD_ID') != POD:
        raise RuntimeError('Wrong Pod for this shutdown watchdog')
    if deadline-time.time() > 2400:
        raise RuntimeError('Deadline exceeds authorized session length')
    path=Path('/workspace/gaussian_shutdown_events.jsonl')
    path.parent.mkdir(parents=True,exist_ok=True)
    def record(event, **details):
        with path.open('a') as stream:
            stream.write(json.dumps(dict(event=event,epoch=time.time(),deadline=deadline,pod=POD,**details))+'\n')
        print('GAUSSIAN_GUARD '+event,flush=True)
    record('armed',pid=os.getpid())
    try:
        # Preserve official Jupyter/SSH setup; no user startup command replaced.
        child=subprocess.Popen(['/start.sh'])
        while time.time() < deadline:
            if child.poll() is not None:
                record('official_start_exited');break
            time.sleep(min(1,max(0,deadline-time.time())))
    finally:
        record('stop_requested')
        for attempt in range(120):
            try:
                req=urllib.request.Request(f'https://rest.runpod.io/v1/pods/{POD}/stop',
                    method='POST',headers={'Authorization':'Bearer '+secret,'User-Agent':'GaussianVideoBudgetGuard/1'})
                with urllib.request.build_opener(NoRedirect).open(req,timeout=15) as response:
                    response.read(1024)
                record('stop_api_accepted')
                check=urllib.request.Request(f'https://rest.runpod.io/v1/pods/{POD}',
                    headers={'Authorization':'Bearer '+secret,'User-Agent':'GaussianVideoBudgetGuard/1'})
                with urllib.request.build_opener(NoRedirect).open(check,timeout=15) as response:
                    status=json.loads(response.read(1024*1024))
                if status.get('id')==POD and status.get('desiredStatus')=='EXITED':
                    record('stopped_verified');return
            except Exception as error:
                record('stop_retry',error_type=type(error).__name__,http_status=getattr(error,'code',None))
                time.sleep(5)
        record('FAILED_STOP_REQUIRES_ATTENTION')


if __name__=='__main__': main()
