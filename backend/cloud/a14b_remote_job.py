"""One remote setup + frozen evaluation. No retries of a failed experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/work/eval2')
OUT=Path('/workspace/remote_eval2')


def main():
    OUT.mkdir(exist_ok=False)
    report={'status':'starting','stages':{},'model_training':False,'accepted':False}
    started=time.monotonic()
    def save():
        temp=OUT/'job.pending';temp.write_text(json.dumps(report,indent=2));temp.replace(OUT/'job.json')
    def step(name,command,maximum):
        # Leave two minutes for local retrieval before the boot guard stops us.
        left=float(os.environ['GAUSSIAN_STOP_DEADLINE'])-time.time()-120
        if left < maximum: raise RuntimeError('Insufficient guarded time for '+name)
        report['status']=name;save();t=time.monotonic()
        env=dict(os.environ);env['PYTHONUNBUFFERED']='1';env['OMP_NUM_THREADS']='8';env['TOKENIZERS_PARALLELISM']='false'
        # Workload does not need the account-control secret.
        env.pop('RUNPOD_SHUTDOWN_KEY',None)
        with (OUT/(name+'.log')).open('x') as log:
            completed=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=maximum)
        report['stages'][name]={'seconds':time.monotonic()-t,'exit_code':completed.returncode};save()
        if completed.returncode:raise RuntimeError(name+' failed')
    try:
        python=Path('/workspace/work/eval_venv/bin/python')
        step('environment',[sys.executable,'-m','venv','--system-site-packages',str(python.parent.parent)],30)
        step('dependencies',[str(python),'-m','pip','install','-r','cloud/a14b_requirements.txt'],180)
        step('preflight',[str(python),'-c',
            "from cloud.a14b_preflight import require_dependencies;from cloud.eval_preflight import validate_assets,require_host_memory;import psutil,json;print(json.dumps({'dependencies':require_dependencies(),'assets':validate_assets('asset'),'resources':require_host_memory(int(psutil.virtual_memory().available))}))"],45)
        step('download',[str(python),'-m','cloud.download_a14b','--cache','/workspace/work/hf-cache','--out',str(OUT/'download')],1100)
        manifest=json.loads((OUT/'download/download_manifest.json').read_text())
        if manifest['status']!='verified':raise RuntimeError('Download not verified')
        step('evaluation',[str(python),'-m','cloud.evaluate_a14b','--model',manifest['local_path'],
            '--asset',str(ROOT/'asset'),'--out',str(OUT/'evaluation')],605)
        report['status']='completed_visual_review_required'
    except Exception as error:
        report['status']='failed';report['error_type']=type(error).__name__;report['error']=str(error)
    finally:
        report['total_seconds']=time.monotonic()-started;save()


if __name__=='__main__':main()
