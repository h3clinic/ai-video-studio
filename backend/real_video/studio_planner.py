"""General project planning through the real AgentVideo protocol and Gemini."""
import json
import re
from pathlib import Path
from .agentvideo_bridge import load_upstream
from .agentvideo_gemini import GeminiLLM, CallBudget


def issues(value):
    if not isinstance(value,dict) or set(value)!={'summary','parts'}:return ['Return summary and parts only']
    if not isinstance(value['summary'],str) or not 1<=len(value['summary'])<=3000:return ['Invalid summary']
    if not isinstance(value['parts'],list) or not 1<=len(value['parts'])<=24:return ['Need 1..24 task parts']
    seen=set()
    for part in value['parts']:
        if not isinstance(part,dict) or set(part)!={'id','label','task','protected'}:return ['Invalid part schema']
        if not isinstance(part['id'],str) or not re.fullmatch(r'[a-z0-9_-]{1,60}',part['id']) or part['id'] in seen:return ['Unique safe part IDs required']
        seen.add(part['id'])
        if any(not isinstance(part[k],str) or not 1<=len(part[k])<=1500 for k in ('label','task')) or type(part['protected']) is not bool:return ['Invalid part instruction']
    return []


def plan_project(prompt, output, client=None):
    upstream=load_upstream(Path(__file__).resolve().parents[3]/'work/agentvideo-reference')
    client=client or GeminiLLM('gemini-3.8-flash',budget=CallBudget(calls=2,output_tokens=8192),minimum_output_tokens=4096)
    bus=upstream['Bus'](echo=False);bus.strict=True
    root=upstream['Agent']('video-controller',bus,client)
    director=root.spawn(upstream['Agent'],'idea-director')
    verifier=director.verifier('plan-verifier',issues)
    task='Plan the requested video or local Gaussian edit. Return JSON only: {summary:string,parts:[{id:lowercase_identifier,label:string,task:string,protected:boolean}]}. Identify arbitrary scene-specific objects and important subparts, detached pieces, contact and occlusion requirements. No hardcoded apple defaults. Never claim generated media, GPU work, Gaussian bindings or completion. Explain missing inputs. Protected true means no edits allowed. Do not output code, paths or invented Gaussian IDs. User request:\n'+prompt
    plan=director.think(task,verifier,fallback=lambda:{},max_tries=2,max_tokens=4096,what='Project plan')
    for part in plan['parts']:
        owner=director.spawn(upstream['Agent'],'part:'+part['id'],desc=part['task'])
        part['agentId']=owner.name
        bus.log(owner.name,'info','Assigned planning responsibility; no verified Gaussian IDs yet')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    record={'plan':plan,'usage':client.usage,'upstream':upstream['provenance'],'geometry_verified':False,'scene_modified':False}
    (output/'plan.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    bus.dump(output/'trace.json')
    return record
