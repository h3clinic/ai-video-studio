"""Bounded, observable AgentVideo teams. Plans never grant Gaussian write access.

The director chooses named workers and their dependencies. Only the sound tool
executes media generation here; other workers produce explicitly typed briefs.
No arbitrary code, paths, URLs, GPU launches or paid retries are model-controlled.
"""
from datetime import datetime, timezone
import copy
import json
import math
from pathlib import Path
import re
import time
import uuid
from .agentvideo_bridge import load_upstream
from .agentvideo_gemini import GeminiLLM, GeminiVLM, CallBudget
from .studio_roles import ROLES, ROLE_IDS

MAX_WORKERS = 20
REVIEW_REVISION = 2
SOURCE = 'artifacts/real_video/runway_agents/donkey_orange_v1/source/video.mp4'
PREVIEW = 'artifacts/real_video/runway_agents/donkey_orange_v1/reference/preview.png'
SAFE_ID = re.compile(r'[a-z0-9_-]{1,60}')


def utc():
    return datetime.now(timezone.utc).isoformat()


def plan_issues(value):
    if not isinstance(value, dict) or set(value) != {'summary', 'workers'}:
        return ['Return summary and workers only']
    if not isinstance(value['summary'], str) or not 1 <= len(value['summary']) <= 1500:
        return ['Invalid summary']
    workers = value['workers']
    if not isinstance(workers, list) or not 16 <= len(workers) <= MAX_WORKERS:
        return ['This demonstration needs 16..20 workers covering all 14 roles, with 3..4 sound workers']
    seen, roles, sounds = set(), set(), 0
    for w in workers:
        if not isinstance(w, dict) or set(w) != {'id', 'roleId', 'name', 'partId', 'task', 'dependsOn'}:
            return ['Invalid worker schema']
        if any(not isinstance(w[k], str) or not SAFE_ID.fullmatch(w[k]) for k in ('id', 'partId')) or w['id'] in seen:
            return ['Unique safe worker IDs and safe part IDs required']
        if w['roleId'] not in ROLE_IDS:
            return ['Unknown role']
        if any(not isinstance(w[k], str) or not 1 <= len(w[k]) <= limit for k, limit in [('name', 80), ('task', 1400)]):
            return ['Bounded name/task required']
        if not isinstance(w['dependsOn'], list) or len(w['dependsOn']) > 8 or any(not isinstance(d, str) or d not in seen for d in w['dependsOn']):
            return ['Dependencies must reference earlier workers; no cycles or unknown IDs']
        seen.add(w['id']); roles.add(w['roleId']); sounds += w['roleId'] == 'sound-agents'
    if roles != ROLE_IDS or not 3 <= sounds <= 4:
        return ['Cover every role with 3..4 distinct sound specialists']
    return []


def brief_issues(value):
    if not isinstance(value, dict) or set(value) != {'summary', 'requirements', 'region'}:
        return ['Return summary, requirements and region only']
    if not isinstance(value['summary'], str) or not 1 <= len(value['summary']) <= 1600:
        return ['Bounded summary required']
    if not isinstance(value['requirements'], list) or len(value['requirements']) > 8 or any(not isinstance(r, str) or len(r) > 240 for r in value['requirements']):
        return ['At most eight short unmet requirements']
    box = value['region']
    if box is not None and (not isinstance(box, list) or len(box) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in box) or
            box[0] >= box[2] or box[1] >= box[3]):
        return ['Region must be null or normalized [left,top,right,bottom]']
    return []


def run_swarm(prompt, output, *, root, progress, client=None, sound_tool=None, upstream=None, previous=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    upstream = upstream or load_upstream(root.parents[1]/'work/agentvideo-reference')
    budget = CallBudget(calls=MAX_WORKERS+1, output_tokens=86016)
    client = client or GeminiLLM('gemini-3.8-flash', budget=budget, minimum_output_tokens=4096)
    if sound_tool is None:
        from .elevenlabs_client import generate_sound
        sound_tool = generate_sound
    bus = upstream['Bus'](echo=False); bus.strict = True
    deadline = time.monotonic()+600
    director = upstream['Agent']('swarm-director', bus, client)
    workers = []
    def publish(stage):
        progress(stage=stage, workers=copy.deepcopy(workers))
        (output/'workers.json').write_text(json.dumps(workers, indent=2), encoding='utf-8')
        bus.dump(output/'trace.json')
    publish('Director deciding team size and distinct responsibilities')
    try:
        if previous is not None:
            plan = dict(summary=previous['teamSummary'],workers=[{k:w[k] for k in ('id','roleId','name','partId','task','dependsOn')} for w in previous['workers']])
            if plan_issues(plan):raise ValueError('Previous team violates current contract')
            bus.log(director.name,'info','Reusing verified team allocation; no new director API call')
        else:
            plan = director.think(
            'Design an observable team for the supplied existing donkey-and-orange video. This is an RGB Runway bootstrap, not verified dynamic Gaussians. '
            'Create 16..20 named workers in dependency/topological order covering ALL these roles: '+json.dumps(ROLES)+'. '
            'You choose count and assignments. Include 3 or 4 distinct sound-agents (e.g. environment, animal foley, fruit handling). '
            'Each sound task is the exact isolated SFX prompt sent to ElevenLabs for 4.8 seconds, no speech/music and no other layers. '
            'Other roles create concrete specialist briefs/requirements, NOT executed image edits. Vision may inspect one reference image only. '
            'Keep sound workers independent of geometry/reconstruction work; their prompts alone suffice for isolated audio. Finish with a verifier that checks actual outcomes; it cannot certify audio/animation quality without review. '
            'Return JSON {summary:string,workers:[{id:string,roleId:string,name:string,partId:string,task:string,dependsOn:[earlier_worker_id]}]}. '
            'Identifiers lowercase safe; short tasks; no code, file paths, credential requests, external URLs or invented Gaussian IDs. User brief: '+prompt,
            director.verifier('team-contract', plan_issues), fallback=lambda: {}, max_tries=1,
            max_tokens=8192, rule='\nReturn only the JSON object.', what='Team allocation')
        progress(teamSummary=plan['summary'])
        sound_failure = None
        changed = set()
        # Reviewers consume all worker outcomes, not only declared dependencies.
        pending_change = any(w.get('status')!='completed' for w in (previous or {}).get('workers',[]))
        for spec in plan['workers']:
            worker = dict(spec, mediaId=uuid.uuid4().hex, parentId='swarm-director', status='queued',
                          createdAt=utc(), queuedAt=utc(), bindingVerified=False, execution='not_started')
            workers.append(worker)
            agent = director.spawn(upstream['Agent'], spec['id'], desc=spec['task'])
            publish('Worker formed: '+spec['name'])
            reused = next((w for w in (previous or {}).get('workers',[]) if w['id']==spec['id'] and w['status']=='completed'),None)
            if reused and (set(spec['dependsOn']) & changed or (spec['roleId']=='verifiers' and
                    (pending_change or changed or (previous or {}).get('reviewRevision')!=REVIEW_REVISION))):
                reused=None
            if reused:
                worker.update(copy.deepcopy(reused), reusedFrom=previous['id'])
                publish('Reused completed result: '+spec['name']);continue
            changed.add(spec['id'])
            if spec['roleId']=='sound-agents' and sound_failure:
                worker.update(status='blocked',finishedAt=utc(),summary='Sound provider unavailable; no repeated request',error=sound_failure)
                publish('Sound provider blocked: '+spec['name']);continue
            if time.monotonic()+90 > deadline:
                worker.update(status='blocked', finishedAt=utc(), summary='Session time budget reached; no API request made')
                publish('Time budget blocked: '+spec['name']); continue
            if spec['roleId']!='verifiers' and any(next(w for w in workers if w['id']==d)['status'] != 'completed' for d in spec['dependsOn']):
                worker.update(status='blocked', finishedAt=utc(), summary='A required worker did not complete; no dependent API call made')
                publish('Dependency blocked: '+spec['name']); continue
            worker.update(status='running', startedAt=utc())
            tick = time.monotonic()
            publish('Working: '+spec['name'])
            try:
                if spec['roleId'] == 'sound-agents':
                    result = sound_tool(spec['task'], output/spec['id'])
                    worker.update(sound=result, output=str((output/spec['id']/result['output']).relative_to(root)),
                                  execution='elevenlabs_sound', summary='Separate sound layer generated; not synchronized or quality-approved',
                                  review=result['review'])
                else:
                    dependencies = [dict(id=w['id'], summary=w.get('summary'), execution=w['execution']) for w in workers if w['id'] in spec['dependsOn'] or (spec['roleId']=='verifiers' and w['id']!=spec['id'])]
                    task = ('Role: '+spec['roleId']+'; part: '+spec['partId']+'; assignment: '+spec['task']+
                        '\nUser brief: '+prompt+'\nPrior worker evidence: '+json.dumps(dependencies)+
                        '\nReturn JSON {summary:string,requirements:[string],region:null}. Summary: maximum 300 characters. Requirements: zero to FIVE strings, each maximum 180 characters, only genuinely missing prerequisites. All three fields are required; no extra fields. '
                        'Nothing changes geometry in this task. No claims of edited video, verified 3D ownership, compute savings or heard audio. '
                        'Prior briefs are proposals, NOT proof that masks, motion or boundaries were verified. '
                        'Reviewers may check task completion only; report geometry/quality as unverified without measured evidence. '
                        'If viewing an image, region may instead be normalized [left,top,right,bottom] for your assigned visible part; it is only an unverified image proposal. '
                        'Without an image, region MUST be null. No code, paths, secrets or URLs.')
                    if spec['roleId'] == 'vision-sensor' and (root/PREVIEW).is_file() and isinstance(client, GeminiLLM) and not client.transport:
                        vision = GeminiVLM(client.model_id, budget=budget, minimum_output_tokens=4096)
                        vision.usage = client.usage
                        agent.llm = vision.looking_at(root/PREVIEW)
                    else:
                        vision = None
                    result = agent.think(task, agent.verifier(spec['id']+'-contract', brief_issues), fallback=lambda: {},
                                         max_tries=1, max_tokens=2048, rule='\nReturn only the JSON object.', what='Specialist brief')
                    if not vision:
                        result['region'] = None
                    worker.update(**result, execution='specialist_brief')
                worker.update(status='completed')
            except Exception as error:
                # Never surface arbitrary provider response/error text.
                worker.update(status='failed', error=getattr(error, 'code', type(error).__name__),
                              summary='Worker failed; no automatic retry. Other independent work may continue.')
                if spec['roleId']=='sound-agents':sound_failure=worker['error']
            worker.update(finishedAt=utc(), seconds=time.monotonic()-tick)
            publish('Finished: '+spec['name']+' · '+worker['status'])
        report = dict(workers=workers, teamSummary=plan['summary'], usage=client.usage, upstream=upstream['provenance'],reviewRevision=REVIEW_REVISION,
                      sceneModified=False, gaussianOwnershipVerified=False,
                      sourceVideo=SOURCE, sourcePreview=PREVIEW,
                      completed=sum(w['status']=='completed' for w in workers),
                      soundLayers=sum(w['execution']=='elevenlabs_sound' for w in workers),
                      review='Coordination and separate audio artifacts only. No accepted Gaussian video edit.')
        (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report
    finally:
        bus.dump(output/'trace.json')
