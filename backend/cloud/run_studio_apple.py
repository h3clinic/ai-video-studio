"""One user-approved Studio experiment, using the same durable job dispatcher."""
import os
import time
import json
from real_video.studio_backend import Studio

def main():
    if os.environ.get('GAUSSIAN_APPLE_APPROVED')!='1':raise RuntimeError('Explicit session approval required')
    studio=Studio()
    prompt='In the existing donkey eating an orange video, replace all orange-related fruit parts with apple: detailed red apple skin, pale apple flesh slices, thin red peel and the mouth-held bite. Preserve donkey, bowl, environment and source motion. Plan separate part-agent responsibilities and identify missing Gaussian bindings.'
    plan=studio.submit(dict(kind='plan',projectId='apple-experiment',prompt=prompt))
    while next(j for j in studio.jobs if j['id']==plan['id'])['status'] in ('queued','running'):time.sleep(1)
    print(json.dumps({'planning_status':next(j for j in studio.jobs if j['id']==plan['id'])['status']}),flush=True)
    job=studio.submit(dict(kind='apple_experiment',projectId='apple-experiment'))
    print(json.dumps({'job_id':job['id']}),flush=True)
    last=None
    while True:
        value=next(j for j in studio.jobs if j['id']==job['id'])
        if value['stage']!=last:print(json.dumps({'status':value['status'],'stage':value['stage']}),flush=True);last=value['stage']
        if value['status'] not in ('queued','running'):
            print(json.dumps(value),flush=True);break
        time.sleep(2)

if __name__=='__main__':main()
