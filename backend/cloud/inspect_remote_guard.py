"""Read-only remote guard diagnostics; never prints credentials or raw errors."""
from pathlib import Path
from cloud.runpod_control import api, load_key
from cloud.jupyter_transfer import JupyterClient

if __name__=='__main__':
    pod=api()
    client=JupyterClient('https://wdq34cebh6111k-8888.proxy.runpod.net',
        token=pod['env']['JUPYTER_PASSWORD'],known_tokens=(load_key(),))
    code='''import os, requests, json
from pathlib import Path
print(Path('/workspace/gaussian_shutdown_events.jsonl').read_text())
for agent in ('Python-urllib/3.12','GaussianVideoBudgetGuard/1'):
    response=requests.get('https://rest.runpod.io/v1/pods/wdq34cebh6111k',
        headers={'Authorization':'Bearer '+os.environ['RUNPOD_SHUTDOWN_KEY'],'User-Agent':agent},
        timeout=8,allow_redirects=False)
    print(json.dumps({'agent':agent,'http':response.status_code}))
'''
    print(client.execute(code,Path('artifacts/cloud/shutdown_canary_v2/remote_diagnosis_2.log'),
        timeout_seconds=25,kernel_path='workspace'))
