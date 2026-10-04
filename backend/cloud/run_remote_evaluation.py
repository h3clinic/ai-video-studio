"""Explicit one-session evaluation controller; retrieves outputs then stops Pod.

Remote and local independent deadlines are installed before startup. This client
only uploads the allowlisted bundle and experiment code. It never tops up funds.
"""
import hashlib
import json
from pathlib import Path
import time
from cloud.runpod_control import api,load_key,request,awake
from cloud.runpod_guarded_session import launch,ROOT
from cloud.jupyter_transfer import JupyterClient,TransportError

OUT=ROOT/'artifacts/cloud/a14b_remote_eval_v2'
PARTS=ROOT/'artifacts/cloud/a14b_eval_transfer_v2'
PIN='25d40fbba4fd0733a6ad1a6a8cebd5718a8de749a15a160f83ba918572720f07'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    summary={'status':'preparing','outputs':[],'accepted':False,'training':False}
    def record(**fields):
        summary.update(fields)
        tmp=OUT/'control.pending';tmp.write_text(json.dumps(summary,indent=2));tmp.replace(OUT/'control.json')
        print(json.dumps(fields),flush=True)
    pod=api()
    client=JupyterClient('https://wdq34cebh6111k-8888.proxy.runpod.net',
        token=pod['env']['JUPYTER_PASSWORD'],known_tokens=(load_key(),))
    session=launch(OUT/'guard',2100,2.20)
    deadline=session['remote_deadline_epoch']
    try:
        record(status='waiting_for_jupyter',guard=session,balance_before_usd=2.87)
        for attempt in range(30):
            try:
                boot=client.read_bytes('workspace/gaussian_shutdown_events.jsonl')
                if b'"event": "armed"' not in boot:raise RuntimeError('Remote guard not armed')
                (OUT/'remote_guard_armed.jsonl').write_bytes(boot);break
            except TransportError:
                if time.time()>deadline-1800:raise RuntimeError('Startup consumed setup allowance')
                time.sleep(3)
        else:raise RuntimeError('Jupyter unavailable')
        record(status='uploading_verified_parts')
        for attempt in range(3):
            try:
                transferred=client.upload_parts(PARTS,'workspace/eval_transfer',PIN);break
            except TransportError:
                if attempt==2 or time.time()>deadline-1600:raise
                record(transfer_retry=attempt+1);time.sleep(3)
        record(transfer=transferred)
        for name in ('chunk_transfer.py','a14b_remote_job.py'):
            data=(ROOT/'cloud'/name).read_bytes()
            client.put_bytes('workspace/'+name,data,expected_sha256=hashlib.sha256(data).hexdigest())
        # Verify transport and every ZIP entry before unpacking our own bundle.
        setup=f'''import importlib.util, pathlib, json, zipfile, hashlib, subprocess, sys
spec=importlib.util.spec_from_file_location('chunk_transfer','/workspace/chunk_transfer.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.assemble('/workspace/eval_transfer','/workspace/eval_bundle.zip',{PIN!r})
root=pathlib.Path('/workspace/work/eval2');root.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile('/workspace/eval_bundle.zip') as archive:
    manifest=json.loads(archive.read('bundle_manifest.json'))
    assert set(archive.namelist())=={{r['path'] for r in manifest}}|{{'bundle_manifest.json'}}
    for row in manifest:
        target=(root/row['path']).resolve();assert target.is_relative_to(root)
        data=archive.read(row['path']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(data)
log=open('/workspace/eval_worker.log','x')
child=subprocess.Popen([sys.executable,'/workspace/a14b_remote_job.py'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'worker_pid':child.pid,'bundle_verified':True}}))
'''
        result=client.execute(setup,OUT/'setup.log',timeout_seconds=45,kernel_path='workspace')
        if result['status']!='ok':raise RuntimeError('Remote setup failed')
        record(status='remote_worker_running')
        previous=None
        while time.time()<deadline-75:
            data=client.read_bytes('workspace/remote_eval2/job.json',missing_ok=True)
            if data is not None:
                job=json.loads(data)
                tmp=OUT/'job.pending';tmp.write_bytes(data);tmp.replace(OUT/'job.json')
                if job['status']!=previous:
                    previous=job['status'];record(remote_stage=previous)
                if previous in ('failed','completed_visual_review_required'):break
            time.sleep(15)
        else:record(status='retrieval_before_deadline')
        # All expected small reports/logs, including failures, are retained.
        for name in ('job.json','environment.log','dependencies.log','preflight.log','download.log','evaluation.log',
                     'download/download_manifest.json','evaluation/report.json'):
            data=client.read_bytes('workspace/remote_eval2/'+name,missing_ok=True)
            if data is None:continue
            target=OUT/'results'/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(data)
            summary['outputs'].append({'path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest()})
        report=OUT/'results/evaluation/report.json'
        if report.exists():
            for video in json.loads(report.read_text()).get('videos',{}).values():
                name=video['file']
                if Path(name).name!=name or not name.endswith('.mp4'):raise ValueError('Unsafe output filename')
                data=client.read_bytes('workspace/remote_eval2/evaluation/'+name)
                if hashlib.sha256(data).hexdigest()!=video['sha256']:raise ValueError('Output video hash mismatch')
                with (report.parent/name).open('xb') as stream:stream.write(data)
                summary['outputs'].append({'path':str((report.parent/name).relative_to(ROOT)),'sha256':video['sha256']})
        job_file=OUT/'results/job.json'
        failed=job_file.exists() and json.loads(job_file.read_text()).get('status')=='failed'
        record(status='remote_job_failed_evidence_retrieved' if failed else 'retrieved_visual_review_pending')
    except Exception as error:
        record(status='failed',error_type=type(error).__name__,error=client._redact(str(error)))
    finally:
        request('stop')
        record(stop_requested=True,provider_status=request())


if __name__=='__main__':
    with awake():main()
