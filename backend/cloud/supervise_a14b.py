"""One-shot bounded worker supervisor; never creates or restarts paid resources."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

BASE=Path('/workspace')
ROOT=BASE/'work/pilot'
OUT=BASE/'pilot_output'
DEADLINE=datetime.datetime(2026,10,3,23,11,tzinfo=datetime.timezone.utc).timestamp()
manifest=BASE/'download_evidence/download_manifest.json'
status={'status':'waiting_for_verified_model','heavy_attempts':0}
def save():
    (BASE/'supervisor_status.json').write_text(json.dumps(status,indent=2))
save()
try:
    while time.time()<DEADLINE-720:
        try:
            evidence=json.loads(manifest.read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            evidence={}
        if evidence.get('status')=='verified':
            break
        time.sleep(10)
    else:
        raise TimeoutError('Insufficient remaining rental time for one 600-second worker and retrieval')
    status.update(status='running',heavy_attempts=1)
    save()
    with (BASE/'experiment.log').open('w') as log:
        command=[sys.executable,'-m','cloud.wan_a14b_experiment',
            '--model',evidence['local_path'],'--source',str(ROOT/'artifacts/real_video/runway_agents/donkey_orange_v1/source/video.mp4'),
            '--out',str(OUT),'--frames','33','--steps','20','--train-steps','2','--wall-seconds','600']
        status['command']=command; save()
        completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=620)
    status.update(status='worker_finished',exit_code=completed.returncode)
except BaseException as error:
    status.update(status='failed',error_type=type(error).__name__,error=str(error))
finally:
    save()
    with zipfile.ZipFile(BASE/'pilot_results.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in [BASE/'supervisor_status.json',BASE/'experiment.log',BASE/'setup.log',BASE/'download.log',manifest]:
            if path.is_file(): archive.write(path,path.relative_to(BASE))
        if OUT.exists():
            for path in OUT.rglob('*'):
                if path.is_file(): archive.write(path,path.relative_to(BASE))
    print(json.dumps(status),flush=True)
