"""Bounded Gemini decisions using the owner's saved credential and typed edits."""
import json
import math
from pathlib import Path
from .agentvideo_bridge import load_upstream
from .agentvideo_gemini import GeminiLLM, CallBudget
from .studio_roles import ROLE_IDS, ROLES


def validate_decision(value, agent_id, part_id):
    if not isinstance(value,dict) or set(value)!={'summary','operations','motion','blockers'}:
        raise ValueError('Expected summary, operations, motion, blockers')
    if not isinstance(value['summary'],str) or not 1<=len(value['summary'])<=3000:
        raise ValueError('Invalid summary')
    blockers=value['blockers']
    if not isinstance(blockers,list) or len(blockers)>16 or any(not isinstance(x,str) or not 1<=len(x)<=500 for x in blockers):
        raise ValueError('Invalid blockers')
    if not isinstance(value['operations'],list) or len(value['operations'])>8:
        raise ValueError('Bounded operations required')
    from .gaussian_agent_edits import operation_from_dict
    for operation in value['operations']:
        if not isinstance(operation,dict):raise ValueError('Operation must be an object')
        if operation.get('agent_id')!=agent_id or operation.get('part_id')!=part_id or operation.get('expected_revision')!=0:
            raise ValueError('Operation cannot change assigned scope or revision')
        operation_from_dict(operation)
        if operation['kind']=='texture_paint':raise ValueError('Texture UV correspondence is not registered')
        for field in ('translation','pivot'):
            if field in operation and any(abs(x)>4 for x in operation[field]):raise ValueError('Canonical transform exceeds bound')
    motion=value['motion']
    if motion is not None:
        fields={'duration_seconds','fps','velocity','acceleration','angular_velocity','angular_acceleration','pivot'}
        if not isinstance(motion,dict) or set(motion)!=fields: raise ValueError('Invalid motion schema')
        if type(motion['fps']) is not int or not 1<=motion['fps']<=12: raise ValueError('Invalid fps')
        duration=motion['duration_seconds']
        if type(duration) not in (int,float) or not math.isfinite(duration) or not 1<=duration<=5: raise ValueError('Invalid duration')
        frame_count=duration*motion['fps']
        if frame_count<2 or not math.isclose(frame_count,round(frame_count),abs_tol=1e-8):raise ValueError('Motion duration must have an integral frame count, at least two')
        for field in fields-{'fps','duration_seconds'}:
            row=motion[field]
            if not isinstance(row,list) or len(row)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>4 for x in row):
                raise ValueError('Invalid motion vector')
        from .gaussian_agent_edits import rigid_motion_at
        rigid_motion_at(agent_id,part_id or 'unbound',0,time_seconds=duration,
            **{k:v for k,v in motion.items() if k not in ('duration_seconds','fps')})
    if blockers and (value['operations'] or motion is not None):
        raise ValueError('Blocked decisions must not execute partial changes')
    if part_id is None and (value['operations'] or motion is not None): raise ValueError('No bound part')
    return value


def plan_agent_task(prompt, agent_id, part_id, asset, output, client=None):
    if agent_id not in ROLE_IDS: raise ValueError('Unknown role')
    upstream=load_upstream(Path(__file__).resolve().parents[3]/'work/agentvideo-reference')
    client=client or GeminiLLM('gemini-3.8-flash',budget=CallBudget(calls=2,output_tokens=8192),minimum_output_tokens=4096)
    bus=upstream['Bus'](echo=False);bus.strict=True
    controller=upstream['Agent']('video-controller',bus,client)
    worker=controller.spawn(upstream['Agent'],agent_id if agent_id!='video-controller' else 'controller-task')
    def issues(value):
        try: validate_decision(value,agent_id,part_id);return []
        except (ValueError,TypeError,KeyError):return ['Invalid typed decision or scope; use blockers if inputs are missing']
    verifier=worker.verifier('typed-task-verifier',issues)
    context=dict(agent_id=agent_id,role=next(r[2] for r in ROLES if r[0]==agent_id),part_id=part_id,
        bound=asset is not None,gaussian_count=asset.get('gaussianCount') if asset else None,
        texture_binding=False,scope='canonical object only; scene contact/placement not calibrated')
    instruction='''Return JSON only {summary:string,operations:[],motion:null,blockers:[]}.
You are a role in a Gaussian editor. Never output code, paths, URLs, invented Gaussian IDs, or claims of completed edits. Copy the supplied agent_id and part_id exactly.
Allowed operation shapes (all require agent_id exactly assigned role, part_id exactly assigned part, expected_revision:0):
{kind:"recolour",agent_id,part_id,expected_revision,rgb:[0..1,0..1,0..1],strength:0..1}
{kind:"rigid_transform",agent_id,part_id,expected_revision,rotation:[[3],[3],[3]],translation:[3],pivot:[3]}; rotation must be proper orthonormal.
{kind:"texture_paint",agent_id,part_id,expected_revision,texture_id:string,strength:0..1}; requires existing verified UV binding, currently absent.
For requested animation, motion may be {duration_seconds:1..5,fps:1..12,velocity:[3],acceleration:[3],angular_velocity:[3],angular_acceleration:[3],pivot:[3]}.
All vectors finite in [-4,4]; canonical units have longest object extent 1. Angular axes must be collinear. Do not invent physically grounded trajectories.
Only propose these mathematical controls if the request permits a canonical object motion demonstration. Eating, contact, slicing, peel creation, texture fitting, scene editing or new geometry cannot be substituted with recolouring/spinning.
If requested work is unsupported or unbound, state exact missing prerequisite in blockers, with empty operations and null motion. Blocked is a useful truthful result.
Recolouring blends stored colour with target; it does not generate texture detail. Audio is handled separately by the sound-agents role, not these geometry operations. Web crawling is not connected. No success claims from schema validation.
'''
    decision=worker.think(instruction+'\nTrusted capability metadata: '+json.dumps(context)+'\nUser task: '+prompt,
        verifier,fallback=lambda:{},max_tries=2,max_tokens=4096,what='Typed agent task')
    validate_decision(decision,agent_id,part_id)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(decision=decision,usage=client.usage,upstream=upstream['provenance'],scene_modified=False)
    (output/'decision.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    bus.dump(output/'trace.json')
    return result
