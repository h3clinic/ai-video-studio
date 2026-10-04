"""Paid source acquisition only. One immutable attempt per stage, no POST retry.
Local Gaussian workers never receive the API key. Human selects each paid stage.
"""
import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request
import uuid
from PIL import Image
from .runway_pilot import request,save,ROOT
from .runway_settings import CredentialStore

IMAGE_PROMPT=('Photorealistic medium close side view of a healthy gray donkey in a sunny farmyard, '
    'head, long ears, neck and forelegs visible. A wooden feeding trough in front of its muzzle '
    'holds a clearly visible peeled orange split into juicy orange segments. The donkey is about '
    'to take a bite. Natural anatomy, detailed gray fur, stable camera, realistic daylight. No text.')
VIDEO_PROMPT=('A gray donkey eats the orange segments from the trough. It lowers its muzzle to a '
    'clearly visible orange segment, grasps it with its lips and chews rhythmically. The orange '
    'segment moves into its mouth as its jaw visibly chews. Natural small ear movements. '
    'Stable anatomically coherent head and legs. Fixed camera, continuous shot, no cuts.')


def run(root,stage,submit=False):
    root=root.resolve()
    if not root.is_relative_to(ROOT/'artifacts'):raise ValueError('Outputs must be in artifacts')
    folder=root/('reference' if stage=='image' else 'source')
    model='gen4_image' if stage=='image' else 'gen4_turbo'
    credits=5 if stage=='image' else 25
    key=CredentialStore().load_for_api()
    start=time.perf_counter()
    if submit:
        # Before creating a costly attempt, verify image input is available.
        payload=dict(model=model,ratio='1280:720',seed=103078,
            promptText=IMAGE_PROMPT if stage=='image' else VIDEO_PROMPT)
        if stage=='video':
            ref=root/'reference'/'output.bin'
            with Image.open(ref) as image:mime=Image.MIME[image.format]
            raw=ref.read_bytes()
            payload.update(duration=5,promptImage='data:'+mime+';base64,'+base64.b64encode(raw).decode())
        folder.mkdir(parents=True,exist_ok=False)
        record=dict(stage=stage,model=model,estimated_credits=credits,estimated_usd_before_tax=credits*.01,
            actual_billed_credits=None,pricing_source='https://docs.dev.runwayml.com/guides/pricing/',
            prompt=payload['promptText'],ratio=payload['ratio'],seed=payload['seed'],
            duration=payload.get('duration'),status='submission_started',no_automatic_retry=True,
            classification='Original externally generated RGB, not Gaussian output',local_model_loaded=False)
        if stage=='video':record['reference_sha256']=hashlib.sha256(raw).hexdigest()
        save(folder,record)
        try:
            result=request(key,'POST','/v1/text_to_image' if stage=='image' else '/v1/image_to_video',payload)
            task=result['id']
            if not isinstance(task,str) or not re.fullmatch(r'[A-Za-z0-9-]{10,100}',task):raise ValueError('Invalid task')
            record.update(task_id=task,status='submitted',submission_seconds=time.perf_counter()-start)
            save(folder,record)
        except Exception as error:
            record.update(status='submission_failed_or_unknown',error_type=type(error).__name__,
                safe_error=str(error) if re.fullmatch(r'Runway HTTP \d+',str(error)) else 'Do not automatically resubmit; outcome unknown')
            save(folder,record);print(json.dumps(record),flush=True);return
    else:record=json.loads((folder/'report.json').read_text())
    output_file=folder/('output.bin' if stage=='image' else 'video.mp4')
    if record.get('output_sha256'):
        if not output_file.is_file() or hashlib.sha256(output_file.read_bytes()).hexdigest()!=record['output_sha256']:
            raise ValueError('Saved output missing or changed; refusing to trust cached success')
        print('Already downloaded and hash verified. No new API request.');return
    task=record.get('task_id','')
    if not re.fullmatch(r'[A-Za-z0-9-]{10,100}',task):raise ValueError('No resumable task')
    while time.perf_counter()-start<540:
        result=request(key,'GET','/v1/tasks/'+task)
        status=result.get('status','UNKNOWN')
        record.update(status=status,poll_wall_seconds=time.perf_counter()-start);save(folder,record)
        print(json.dumps(dict(stage=stage,status=status,seconds=round(time.perf_counter()-start,1))),flush=True)
        if status=='SUCCEEDED':
            url=result['output'][0]
            if not url.startswith('https://'):raise ValueError('HTTPS output required')
            file=output_file
            pending=folder/('download-'+uuid.uuid4().hex+'.pending')
            # Never attach API credentials to media download requests.
            download=time.perf_counter()
            with urllib.request.urlopen(url,timeout=30) as stream,pending.open('xb') as target:
                total=0
                while chunk:=stream.read(1024*1024):
                    total+=len(chunk)
                    if total>100_000_000:raise ValueError('100MB download limit')
                    target.write(chunk)
            if file.exists():
                # Preserve any incomplete earlier retrieval; never repeat paid POST.
                file.rename(folder/(file.name+'.partial-'+uuid.uuid4().hex))
            os.replace(pending,file)
            if stage=='image':
                with Image.open(file) as image:image.save(folder/'preview.png')
            record.update(output_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),output_bytes=total,
                download_seconds=time.perf_counter()-download,wall_seconds=time.perf_counter()-start,
                process_peak_rss_bytes=__import__('psutil').Process().memory_info().peak_wset)
            save(folder,record);return
        if status in ('FAILED','CANCELED','CANCELLED'):return
        time.sleep(10)
    print('Polling paused at time budget; resume this task without another submission.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['image','video'],required=True)
    parser.add_argument('--submit-once',action='store_true');args=parser.parse_args()
    awake=ctypes.windll.kernel32.SetThreadExecutionState
    awake.argtypes=[ctypes.c_uint32];awake.restype=ctypes.c_uint32
    if not awake(0x80000001):raise RuntimeError('Sleep prevention failed')
    try:run(args.out,args.stage,args.submit_once)
    finally:awake(0x80000000)
