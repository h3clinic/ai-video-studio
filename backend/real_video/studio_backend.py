"""Authenticated Electron orchestration. No local GPU inference or automatic purchases."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from .runway_settings import CredentialStore
from .gemini_edit_planner import NoRedirect
from .studio_roles import ROLE_IDS, role_state
from .studio_assets import public_parts, select_asset
from .elevenlabs_client import SoundError
from cloud.runpod_client import pod_request, RunPodError
from cloud.runpod_control import credential_configured

ROOT = Path(__file__).resolve().parents[1]
PARTS = [('whole_apple','Whole apple'),('apple_slice','Apple slices'),('red_peel','Red peel'),('mouth_piece','Mouth-held bite')]
APPLE_REPLACEMENT = dict(ready=False, revision='semantic-static-v1-rejected',
    reason='The previous worker is rejected: fixed whole-apple placement and missing slice/peel replacements. Unchanged paid reruns are disabled.',
    missing=['Verified per-part 3D assets and materials', 'Source part tracks and mouth-contact correspondence', 'Reviewed scene-level replacement worker'])


def utc(): return datetime.now(timezone.utc).isoformat()


def append_job_event(job, *, at=None):
    """Observed lifecycle only; never copy prompts, responses or secrets to events."""
    events=job.setdefault('events',[])
    sequence=(events[-1]['sequence'] if events else 0)+1
    status=job.get('status','unknown')
    # Blocker descriptions may originate from a model. Keep event text generic.
    generic={'blocked':'Task blocked; inspect prerequisites','failed':'Task failed; no automatic retry',
             'interrupted':'Application restarted; previous task interrupted'}
    stage=generic.get(status,job.get('stage','Task state changed'))
    events.append(dict(sequence=sequence,at=at or utc(),status=status,stage=stage,
        agentId=job.get('agentId'),partId=job.get('partId')))
    del events[:-200]


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temp, path)


class Studio:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.directory = self.root/'artifacts/studio'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = []
        project_file=self.directory/'projects.json'
        self.projects=json.loads(project_file.read_text()) if project_file.exists() else []
        self.remote = {'status':'disconnected', 'checkedAt':None}
        path = self.directory/'jobs.json'
        if path.exists():
            self.jobs = json.loads(path.read_text())
            for job in self.jobs:
                if job['status'] in ('queued','running'):
                    job.update(status='interrupted', stage='Application restarted; no automatic retry',finishedAt=utc())
                    append_job_event(job,at=job['finishedAt'])
            self.persist()

    def persist(self): atomic_json(self.directory/'jobs.json', self.jobs)

    def project(self, project_id, title, archived=None):
        if not isinstance(project_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',project_id):raise ValueError('Invalid project')
        if not isinstance(title,str) or not 1<=len(title)<=200:raise ValueError('Invalid project title')
        if archived is not None and type(archived) is not bool:raise ValueError('Invalid archive flag')
        with self.lock:
            found=next((p for p in self.projects if p['id']==project_id),None)
            if found is None:
                found=dict(id=project_id,title=title,createdAt=utc());self.projects.append(found)
            else:found['title']=title
            if archived is not None:found['archived']=archived
            atomic_json(self.directory/'projects.json',self.projects)
            return dict(found)

    def media(self):
        paths = {
            'reference':('artifacts/cloud/gemini_apple_parts_v2/apple_parts_reference.jpg','Apple material reference','image','2D reference; not fitted to Gaussians'),
            'latest_video':('artifacts/cloud/object_edit_fresh4090_v4/experiment/results/output/gaussian_apple.mp4','Previous Gaussian edit — rejected','video','Rejected: coarse apple and contact failures'),
            'comparison':('artifacts/cloud/object_edit_fresh4090_v4/experiment/results/output/comparison_020.png','Previous comparison','image','Historical diagnostic')}
        for job in self.jobs:
            if job['kind']=='agent_swarm':
                from .studio_swarm import SOURCE, PREVIEW
                paths['latest_video_'+job['id']]=(SOURCE,'Source video','video','Runway · RGB · unchanged')
                paths['reference_'+job['id']]=(PREVIEW,'Source image','image','2D reference · not bound to splats')
                for worker in job.get('workers',[]):
                    if worker.get('execution')=='elevenlabs_sound' and worker.get('output'):
                        paths['audio_'+worker['mediaId']]=(worker['output'],worker['name'],'audio',worker.get('review','Unreviewed, separate sound layer'))
            if job['kind']=='gemini_reference' and job['status']=='completed':
                out = self.root/job['output']
                if out.is_file():
                    paths['reference_'+job['id']]=(str(out.relative_to(self.root)), 'Generated project reference','image','Generated; review required')
            if job['kind'] in ('apple_experiment','agent_task','sound_mix') and job.get('output'):
                out=self.root/job['output']
                if out.is_file():
                    if job.get('outputKind')=='audio':
                        paths['audio_'+job['id']]=(str(out.relative_to(self.root)),'Sound Agent effect','audio',job.get('review','Unreviewed audio'))
                    else:
                        title='Video with agent sound' if job['kind']=='sound_mix' else 'Canonical Gaussian task' if job['kind']=='agent_task' else 'Apple repair experiment'
                        paths['latest_video_'+job['id']]=(str(out.relative_to(self.root)),title,'video',job.get('review','Review required'))
        return {k:(self.root/v[0],v[1:]) for k,v in paths.items() if (self.root/v[0]).is_file()}

    def state(self):
        with self.lock:
            parts=[]
            latest={}
            for job in self.jobs:
                if job.get('plan'):latest[job.get('projectId')]=job['plan']
            for project_id,plan in latest.items():
                parts.extend(dict(p,projectId=project_id,ownerRoleId='task-assigner',gaussianCount=None,bindingVerified=False) for p in plan['parts'])
            for job in self.jobs:
                for worker in job.get('workers',[]):
                    if not any(p['projectId']==job['projectId'] and p['id']==worker['partId'] for p in parts):
                        parts.append(dict(id=worker['partId'],label=worker['partId'].replace('_',' '),projectId=job['projectId'],
                            ownerRoleId=worker['roleId'],task=worker['task'],gaussianCount=None,bindingVerified=False))
            for bound in public_parts(self.root):
                parts=[p for p in parts if (p['projectId'],p['id'])!=(bound['projectId'],bound['id'])]
                parts.append(bound)
            projects=[p for p in self.projects if not p.get('archived')]
            return dict(updatedAt=utc(), projects=[p for p in self.projects if not p.get('archived')],parts=parts,
                agents=role_state(projects,self.jobs,parts),
                jobs=list(self.jobs), credentials={p:{'configured':credential_configured() if p=='runpod' else CredentialStore(p).configured()} for p in ('gemini','runway','runpod','elevenlabs')},
                capabilities={'appleReplacement':dict(APPLE_REPLACEMENT),
                    'soundEffects':dict(provider='elevenlabs',maximumRequestedSeconds=4.8,maximumContainerSeconds=5.0,
                        automaticVideoSync=False,automaticVideoMux=False,requestsPerTask=1)},
                remote=dict(self.remote), media=[dict(id=k,assetId=hashlib.sha256(str(v[0].relative_to(self.root)).encode()).hexdigest(),title=v[1][0],kind=v[1][1],review=v[1][2],projectId=next((j.get('projectId') for j in self.jobs if k.endswith('_'+j['id']) or any(k=='audio_'+w.get('mediaId','') for w in j.get('workers',[]))),'apple-experiment'),url='gaussian-media://artifact/'+k) for k,v in self.media().items()],
                limitations=['Agent GIFs illustrate roles. Decision tasks use owner Gemini credentials; sound tasks use owner ElevenLabs credentials. Only verified asset IDs may be edited.',
                    'Generic Gemini plans and reference generation are connected. Part assignments are proposals, not verified Gaussian ownership.',
                    'The old static apple repair is rejected and cannot be rerun from the app. General learned 3D fitting is not integrated. No top-ups.',
                    'Stored API key is not proof of validity or credit. Historical video remains rejected.'])

    def provider(self, pod, stop=False):
        try:
            return pod_request(pod,stop=stop)
        except RunPodError as error:
            with self.lock:
                self.remote=dict(podId=pod,status=error.code,checkedAt=utc(),**error.public())
            raise

    def connect(self, data):
        pod=data.get('podId')
        expected='https://'+str(pod)+'-8888.proxy.runpod.net'
        if data.get('baseUrl','').rstrip('/') != expected: raise ValueError('Use the exact Pod HTTPS Jupyter origin')
        result=self.provider(pod)
        if result.get('id')!=pod: raise ValueError('Provider identity mismatch')
        with self.lock:
            self.remote=dict(podId=pod,baseUrl=expected,status=result.get('desiredStatus','unknown'),rate=result.get('costPerHr'),checkedAt=utc())
        return dict(self.remote)

    def submit(self, data):
        kind=data.get('kind')
        if kind not in ('plan','agent_swarm','agent_task','sound_mix','apple_experiment','gemini_reference','remote_preflight'): raise ValueError('Unsupported job')
        allowed={'kind','projectId','prompt'} | ({'agentId','partId'} if kind=='agent_task' else {'resumeJobId'} if kind=='agent_swarm' else {'sourceJobId','gains'} if kind=='sound_mix' else set())
        if set(data)-allowed:raise ValueError('Unknown job fields')
        prompt=data.get('prompt','')
        if not isinstance(prompt,str) or len(prompt)>4000 or (kind in ('plan','agent_swarm','agent_task') and not prompt.strip()):raise ValueError('Bounded prompt required')
        if kind=='agent_swarm' and not all(CredentialStore(p).configured() for p in ('gemini','elevenlabs')):
            raise ValueError('Owner Gemini and ElevenLabs credentials required; no calls made')
        if kind=='agent_task':
            if data.get('agentId') not in ROLE_IDS:raise ValueError('Unknown agent role')
            if data.get('agentId')=='sound-agents' and len(prompt)>2000:raise ValueError('Sound prompt exceeds bounded request')
            if data.get('partId') is not None and (not isinstance(data['partId'],str) or not re.fullmatch(r'[a-z0-9_-]{1,60}',data['partId'])):raise ValueError('Invalid part')
        project_id=data.get('projectId','apple-experiment' if kind=='apple_experiment' else 'default')
        if not isinstance(project_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',project_id):raise ValueError('Invalid project')
        if any(p['id']==project_id and p.get('archived') for p in self.projects):raise ValueError('Project is archived')
        if kind=='apple_experiment' and os.environ.get('GAUSSIAN_APPLE_APPROVED')!='1':raise ValueError('Remote experiment needs session spending approval')
        if kind=='apple_experiment' and not APPLE_REPLACEMENT['ready']:
            raise ValueError(APPLE_REPLACEMENT['reason'])
        with self.lock:
            if any(j['status'] in ('queued','running') for j in self.jobs): raise ValueError('A job is already active')
            if kind=='remote_preflight' and self.remote.get('status')!='RUNNING': raise ValueError('Connect a running Pod first')
            if not any(p['id']==project_id for p in self.projects):
                self.project(project_id,prompt[:80] or ('Gaussian apple experiment' if kind=='apple_experiment' else 'Untitled project'))
            created=utc()
            job=dict(id=uuid.uuid4().hex,kind=kind,projectId=project_id,prompt=prompt,status='queued',stage='Queued',createdAt=created,queuedAt=created,error=None,output=None)
            if kind=='sound_mix':
                from .studio_audio_mix import validate_gains
                gains=validate_gains(data.get('gains'))
                source=next((j for j in self.jobs if j['id']==data.get('sourceJobId') and j['projectId']==project_id and j['kind']=='agent_swarm'),None)
                if not source:raise ValueError('No matching scene team')
                sounds=[w for w in source.get('workers',[]) if w.get('roleId')=='sound-agents']
                if len(sounds)!=len(gains) or any(w.get('status')!='completed' or w.get('execution')!='elevenlabs_sound' for w in sounds):
                    raise ValueError('Generate all sound layers before export')
                job.update(sourceJobId=source['id'],gains=gains,agentId='sound-agents')
            if kind=='agent_swarm' and data.get('resumeJobId'):
                previous=next((j for j in self.jobs if j['id']==data['resumeJobId'] and j['projectId']==project_id and j['kind']==kind and j['status']=='blocked'),None)
                if not previous or not previous.get('workers'):raise ValueError('No resumable team')
                job.update(resumeJobId=previous['id'],prompt=previous['prompt'])
            if kind=='agent_task':job.update(agentId=data['agentId'],partId=data.get('partId'))
            append_job_event(job,at=created)
            self.jobs.append(job); self.persist()
            threading.Thread(target=self.run,args=(job,),daemon=True).start()
            return dict(job)

    def run(self, job):
        started=time.monotonic()
        def update(**values):
            with self.lock:
                changed=any(key in values and job.get(key)!=values[key] for key in ('status','stage'))
                job.update(values)
                if changed:append_job_event(job,at=values.get('finishedAt') or values.get('startedAt'))
                self.persist()
        try:
            update(status='running',stage='Starting requested task',startedAt=utc())
            if job['kind']=='agent_swarm':
                from .studio_swarm import run_swarm
                previous=next((j for j in self.jobs if j['id']==job.get('resumeJobId')),None)
                result=run_swarm(job['prompt'],self.directory/job['id'],root=self.root,progress=update,previous=previous)
                update(workers=result['workers'],usage=result['usage'],review=result['review'],reviewRevision=result['reviewRevision'],
                       reply=f"{result['completed']} specialist tasks completed; {result['soundLayers']} separate sound layers. Source video and Gaussian geometry are unchanged.",
                       stage='Team finished; review briefs and audio separately')
                if result['completed']!=len(result['workers']):
                    update(status='blocked',finishedAt=utc(),seconds=time.monotonic()-started)
                    return
            elif job['kind']=='sound_mix':
                from .studio_audio_mix import mix_scene_audio
                source=next(j for j in self.jobs if j['id']==job['sourceJobId'] and j['projectId']==job['projectId'])
                update(stage='Mixing existing tracks; copying original video without rerendering')
                result=mix_scene_audio(source,self.directory/job['id'],root=self.root,gains=job['gains'])
                update(output=str((self.directory/job['id']/result['output']).relative_to(self.root)),outputKind='video',
                    mix=result,usage=dict(provider=None,requests=0,costUsd=0),review=result['review'],
                    reply='Video with three independently generated sound layers is ready. Visuals are unchanged.',stage='Sound mix exported')
            elif job['kind']=='agent_task' and job.get('agentId')=='sound-agents':
                from .elevenlabs_client import generate_sound
                if not CredentialStore('elevenlabs').configured():
                    update(status='blocked',stage='Owner ElevenLabs key is not configured; no API call made',
                        error='credential_missing',finishedAt=utc(),seconds=time.monotonic()-started)
                    return
                update(stage='Generating one bounded sound effect with ElevenLabs; video is unchanged')
                result=generate_sound(job['prompt'],self.directory/job['id'])
                update(output=str((self.directory/job['id']/result['output']).relative_to(self.root)),
                    outputKind='audio',sound=result,usage=dict(provider='elevenlabs',requests=result['requests'],
                        billingCharacters=result['billing_characters'],costUsd=None),
                    review=result['review'],reply='Sound effect generated separately. Listen and align it before adding it to the video.',
                    stage='Audio retrieved — listening review and video synchronization are still required')
            elif job['kind']=='agent_task':
                from .studio_agent_tasks import plan_agent_task
                asset=select_asset(self.root,job['projectId'],job.get('partId'))
                update(stage='Gemini specialist deciding typed operations with owner credential')
                result=plan_agent_task(job['prompt'],job['agentId'],job.get('partId'),asset,self.directory/job['id'])
                decision=result['decision']
                update(reply=decision['summary'],decision=decision,usage=result['usage'])
                if decision['blockers']:
                    update(status='blocked',stage='; '.join(decision['blockers']),finishedAt=utc(),seconds=time.monotonic()-started);return
                if not decision['operations'] and not decision['motion']:
                    update(stage='Specialist response recorded; no geometry changed')
                else:
                    if not asset:raise ValueError('Unbound canonical asset')
                    with self.lock:remote=dict(self.remote)
                    if remote.get('status')!='RUNNING':
                        update(status='blocked',stage='Typed operations ready; remote GPU is not connected. No local execution.',finishedAt=utc(),seconds=time.monotonic()-started);return
                    from .studio_remote_edits import execute_remote_edit
                    update(stage='Executing scoped Gaussian operations on remote GPU')
                    result=execute_remote_edit(self,job,asset,decision,remote)
                    update(**result)
            elif job['kind']=='plan':
                from .studio_planner import plan_project
                result=plan_project(job['prompt'],self.directory/job['id'])
                update(plan=result['plan'],reply=result['plan']['summary'],usage=result['usage'],stage='Plan verified; Gaussian binding and execution still required')
            elif job['kind']=='apple_experiment':
                if not APPLE_REPLACEMENT['ready']:
                    update(status='blocked',stage=APPLE_REPLACEMENT['reason'],finishedAt=utc(),seconds=time.monotonic()-started)
                    return
                run_id='studio-apple-'+job['id'][:16]
                env=dict(os.environ,GAUSSIAN_FRESH_RUN_ID=run_id)
                update(stage='Remote donor-exclusion/exposure experiment; not new asset fitting')
                with (self.directory/(job['id']+'.log')).open('w') as log:
                    result=subprocess.run([sys.executable,'-m','cloud.run_semantic_fresh'],cwd=self.root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1380)
                out=self.root/'artifacts/cloud'/run_id/'experiment/results/output/gaussian_apple.mp4'
                if result.returncode or not out.exists():raise RuntimeError('Remote experiment failed; inspect preserved log')
                update(output=str(out.relative_to(self.root)),stage='Remote video retrieved — visual review required',review='Unreviewed; no success claim')
            elif job['kind']=='gemini_reference':
                from .gemini_image_assets import generate_reference
                out=self.directory/job['id']
                if not job['prompt'].strip():raise ValueError('A project-specific reference prompt is required')
                result=generate_reference(out,api='generate-content',prompt=job['prompt'])
                update(output=str((out/result['image']).relative_to(self.root)),usage=result.get('usage'),stage='Reference generated — review and 3D fitting required')
            else:
                from cloud.jupyter_transfer import JupyterClient
                with self.lock: remote=dict(self.remote)
                pod=self.provider(remote['podId'])
                if pod.get('desiredStatus')!='RUNNING': raise ValueError('Pod is no longer running')
                token=pod.get('env',{}).get('JUPYTER_PASSWORD')
                if not token: raise ValueError('Pod has no configured Jupyter authentication')
                client=JupyterClient(remote['baseUrl'],allowed_host=remote['podId']+'-8888.proxy.runpod.net',token=token)
                out=self.directory/(job['id']+'.jsonl')
                # Fixed code, never user or model supplied code. No GPU allocation/training.
                client.execute("import json,torch,psutil; print(json.dumps({'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'available_ram_gib':psutil.virtual_memory().available/2**30}))",out,timeout_seconds=45,kernel_path='workspace')
                update(output=str(out.relative_to(self.root)),stage='Remote preflight executed; see recorded evidence')
            update(status='completed',finishedAt=utc(),seconds=time.monotonic()-started)
        except RunPodError as error:
            update(status='failed',stage='RunPod access failed — model was not run',**error.public(),finishedAt=utc(),seconds=time.monotonic()-started)
        except SoundError as error:
            update(status='failed',stage='Sound provider request failed — no automatic retry',error=error.code,
                finishedAt=utc(),seconds=time.monotonic()-started)
        except Exception as error:
            update(status='failed',stage='Failed — no automatic retry',error=type(error).__name__,finishedAt=utc(),seconds=time.monotonic()-started)


def handler(studio, token, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def valid(self):
            return self.headers.get('Host')==f'127.0.0.1:{port}' and secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+token)
        def reply(self,code,data):
            body=json.dumps(data).encode();self.send_response(code)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if not self.valid(): return self.reply(403,{'error':'Unauthorized'})
            if self.path=='/api/state': return self.reply(200,studio.state())
            if self.path.startswith('/api/media/'):
                item=studio.media().get(self.path.removeprefix('/api/media/'))
                if not item:return self.reply(404,{'error':'Unknown artifact'})
                path=item[0];size=path.stat().st_size;start=0;end=size-1
                value=self.headers.get('Range')
                if value:
                    match=re.fullmatch(r'bytes=(\d+)-(\d*)',value)
                    if not match:return self.reply(416,{'error':'Invalid range'})
                    start=int(match[1]);end=min(int(match[2]) if match[2] else end,end)
                    if start>end:return self.reply(416,{'error':'Invalid range'})
                self.send_response(206 if value else 200)
                self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
                self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
                self.send_header('Cache-Control','no-store')
                if value:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                self.end_headers()
                with path.open('rb') as stream:
                    stream.seek(start);remaining=end-start+1
                    while remaining:
                        block=stream.read(min(65536,remaining));self.wfile.write(block);remaining-=len(block)
                return
            return self.reply(404,{'error':'Unknown route'})
        def do_POST(self):
            if not self.valid():return self.reply(403,{'error':'Unauthorized'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8192 or self.headers.get('Content-Type')!='application/json':raise ValueError('Invalid request')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):raise ValueError('Object required')
                if self.path=='/api/credentials':
                    CredentialStore(data['provider']).save(data['key']);result={'saved':True}
                elif self.path=='/api/jobs':result=studio.submit(data)
                elif self.path=='/api/projects':result=studio.project(data['id'],data['title'],data.get('archived'))
                elif self.path=='/api/remote/connect':result=studio.connect(data)
                elif self.path=='/api/remote/stop':
                    if data.get('podId')!=studio.remote.get('podId'):raise ValueError('Connect the Pod before stopping it')
                    studio.provider(data['podId'],stop=True)
                    studio.remote.update(status='STOP_REQUESTED',checkedAt=utc());result=dict(studio.remote)
                else:return self.reply(404,{'error':'Unknown route'})
                self.reply(200,result)
            except RunPodError as error: self.reply(502,error.public())
            except (ValueError,KeyError): self.reply(400,{'error':'Invalid request or unmet prerequisites'})
            except Exception: self.reply(502,{'error':'Provider or credential operation failed; no automatic retry'})
    return Handler


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8790);args=parser.parse_args()
    token=os.environ.get('GAUSSIAN_STUDIO_TOKEN','')
    if len(token)<32:raise RuntimeError('A fresh session token is required')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(Studio(),token,args.port))
    server.serve_forever()

if __name__=='__main__':main()
