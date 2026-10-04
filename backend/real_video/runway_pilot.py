"""Single paid Gen4 Turbo smoke test; no automatic submission retries.

Uses a previously rendered Gaussian RGB reference, not native Gaussian inputs.
Credentials only go to api.dev.runwayml.com over HTTPS, never logs/artifacts.
"""
import argparse
import base64
import ctypes
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import time
import urllib.request

from .runway_settings import CredentialStore

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/real_video/wan_temporal_memory/gaussian_rollout_v1/gaussian_000.png'
PROMPT=('A continuous side view of this orange tabby cat walking slowly to the right. '
        'Natural alternating steps, coherent four-legged anatomy, gentle tail movement. '
        'Keep its orange striped coat and body identity. Fixed camera, whole cat stays in frame, '
        'plain dark blue background. No cuts or camera orbit.')


def request(key, method, path, payload=None):
    connection=http.client.HTTPSConnection('api.dev.runwayml.com',timeout=30)
    try:
        body=None if payload is None else json.dumps(payload).encode()
        connection.request(method,path,body,{'Authorization':'Bearer '+key,
            'X-Runway-Version':'2024-11-06','Content-Type':'application/json'})
        response=connection.getresponse()
        raw=response.read(2_000_000)
        if response.status>=300:
            # Don't print untrusted API error bodies, which may echo request data.
            raise RuntimeError('Runway HTTP '+str(response.status))
        return json.loads(raw)
    finally:
        connection.close()


def save(out,record):
    pending=out/'report.pending'
    pending.write_text(json.dumps(record,indent=2),encoding='utf-8')
    os.replace(pending,out/'report.json')


def run(out,submit):
    start=time.perf_counter()
    out=out.resolve()
    if not out.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Pilot outputs must stay under project artifacts')
    key=CredentialStore().load_for_api()
    if submit:
        out.mkdir(parents=True,exist_ok=False) # Immutable attempt guards duplicate billing.
        source=SOURCE.read_bytes()
        record=dict(model='gen4_turbo',duration=5,ratio='1280:720',seed=103075,
            prompt=PROMPT,source_path=str(SOURCE),source_sha256=hashlib.sha256(source).hexdigest(),
            estimated_credits=25,estimated_usd_before_tax=.25,actual_billed_credits=None,
            pricing_source='https://docs.dev.runwayml.com/guides/pricing/',
            status='submission_started',automatic_submission_retries=0,
            gaussian_native=False,weights_modified=False,gaussian_writeback=False,
            local_neural_inference=False,cloud_gpu_compute_unknown=True,
            accepted=False,source_has_known_anatomy_defects=True,
            research=dict(source='https://arxiv.org/html/2605.30855v1',equations=[5,6,7],
                limitation='Paper requires latent features and geometry writer; RGB-only API does not implement its closed latent memory loop.'))
        save(out,record)
        payload=dict(model=record['model'],duration=5,ratio=record['ratio'],seed=record['seed'],
                     promptText=PROMPT,promptImage='data:image/png;base64,'+base64.b64encode(source).decode())
        try:
            response=request(key,'POST','/v1/image_to_video',payload)
            task=response['id']
            if not isinstance(task,str) or not re.fullmatch(r'[a-zA-Z0-9-]{10,100}',task):
                raise RuntimeError('Invalid task identifier')
            record.update(task_id=task,status='submitted',submission_seconds=time.perf_counter()-start)
            save(out,record)
        except Exception as error:
            record.update(status='submission_failed_or_unknown',error_type=type(error).__name__,
                          safe_error=str(error) if re.fullmatch(r'Runway HTTP \d+',str(error)) else 'Submission outcome may be unknown. Do not resubmit automatically.')
            save(out,record)
            print(json.dumps(record),flush=True)
            return
    else:
        record=json.loads((out/'report.json').read_text())
        if record.get('status')=='SUCCEEDED' and record.get('video_sha256') and (out/'video.mp4').is_file():
            print('Already completed; no request submitted.');return
    task=record.get('task_id')
    if not task or not re.fullmatch(r'[a-zA-Z0-9-]{10,100}',task):
        raise RuntimeError('No safe resumable task; inspect portal, do not repeat POST')
    while time.perf_counter()-start<540:
        try:
            result=request(key,'GET','/v1/tasks/'+task)
        except Exception as error:
            record.update(poll_error_type=type(error).__name__)
            save(out,record)
            print('Polling interrupted; task ID saved. Resume without submitting again.',flush=True)
            return
        status=result.get('status','UNKNOWN')
        record.update(status=status,submit_or_resume_wall_seconds=time.perf_counter()-start)
        save(out,record)
        print(json.dumps({'status':status,'elapsed_seconds':round(time.perf_counter()-start,1)}),flush=True)
        if status=='SUCCEEDED':
            url=result['output'][0]
            if not url.startswith('https://'):
                raise RuntimeError('Non-HTTPS output refused')
            download=time.perf_counter()
            # Output retrieval never includes the API Authorization header.
            with urllib.request.urlopen(url,timeout=30) as stream, (out/'video.mp4').open('xb') as target:
                total=0
                while chunk:=stream.read(1024*1024):
                    total+=len(chunk)
                    if total>100_000_000: raise RuntimeError('Output exceeds100MB budget')
                    target.write(chunk)
            record.update(download_seconds=time.perf_counter()-download,video_bytes=total,
                video_sha256=hashlib.sha256((out/'video.mp4').read_bytes()).hexdigest(),
                local_process_peak_rss_bytes=__import__('psutil').Process().memory_info().peak_wset,
                quality_review='pending',total_this_invocation_seconds=time.perf_counter()-start)
            save(out,record)
            return
        if status in ('FAILED','CANCELED','CANCELLED'):
            record['failure_code']=str(result.get('failureCode','unspecified'))[:100]
            save(out,record)
            return
        time.sleep(10)
    print('Polling budget reached. Resume saved task; do not submit another.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--submit-once',action='store_true')
    args=parser.parse_args()
    awake=ctypes.windll.kernel32.SetThreadExecutionState
    awake.argtypes=[ctypes.c_uint32];awake.restype=ctypes.c_uint32
    if not awake(0x80000001): raise RuntimeError('Sleep prevention failed')
    try: run(args.out,args.submit_once)
    finally: awake(0x80000000)
