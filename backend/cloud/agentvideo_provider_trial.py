"""Bounded original IdeaModel smoke test; script planning, not video generation."""
import argparse
import json
from pathlib import Path
from real_video.agentvideo_bridge import load_upstream
from real_video.agentvideo_gemini import CallBudget, GeminiLLM
from real_video.gaussian_program import ROOT, Program, evidence


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--retry-service-unavailable', action='store_true')
    parser.add_argument('--provider-budget-fix', action='store_true')
    args=parser.parse_args()
    upstream=load_upstream(args.upstream)
    output=ROOT/'artifacts/cloud'/('agentvideo_gemini_live_v1' if args.live else 'agentvideo_gemini_contract_v1')
    if args.retry_service_unavailable:
        previous=json.loads((ROOT/'artifacts/cloud/agentvideo_gemini_live_v1/result.json').read_text())
        if not args.live or previous.get('error')!='Gemini HTTP 503; no automatic network retry':
            raise ValueError('Retry only for recorded service-unavailable failure')
        output=ROOT/'artifacts/cloud/agentvideo_gemini_live_retry_v1'
    if args.provider_budget_fix:
        output=ROOT/'artifacts/cloud'/('agentvideo_gemini_budget_fix_live_v1' if args.live else 'agentvideo_gemini_budget_fix_contract_v1')
    output.mkdir(exist_ok=False)
    budget=CallBudget(calls=3,output_tokens=8192)
    fixture='SETTING: farmyard in daylight\nSUBJECT donkey 1 actor\nSUBJECT apple 1 prop\nBEAT donkey 0-1 eats the apple'
    def mocked(model,payload):
        return dict(candidates=[dict(finishReason='STOP',content=dict(parts=[dict(text=fixture)]))])
    client=GeminiLLM('gemini-3.8-flash',budget=budget,transport=None if args.live else mocked)
    bus=upstream['Bus'](echo=False)
    bus.strict=True
    idea=upstream['IdeaModel']('idea',bus,client)
    result=dict(upstream=upstream['provenance'],model=client.model_id,live=args.live,
                scope='Original upstream IdeaModel and verifier with Gemini transport. No crawler, simulation or rendering.',
                video_generated=False,weights_modified=False)
    try:
        result['script']=idea.gauss_script('a donkey eating an apple',duration=5,children=())
        result['status']='script_verified_not_video'
    except Exception as error:
        result['status']='failed'
        result['error']=str(error)
        bus.log('idea','fail',str(error))
    result['usage']=client.usage
    result['calls_reserved']=budget.used
    bus.dump(str(output/'trace.json'))
    from real_video.studio_agent_events import project_trace
    (output/'studio_events.json').write_text(json.dumps(project_trace(dict(agents=bus.agents,events=bus.events)),indent=2),encoding='utf-8')
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    program=Program()
    try:
        program.record_experiment(dict(role='generation',issue_key='generation.agentvideo_provider',
            hypothesis='Original AgentVideo IdeaModel/think/verifier can operate with Gemini instead of local MLX Qwen.',
            observation=result['status']+'; script only, no asset or video acceptance.',
            outcome='partial' if result['status']!='failed' else 'failed_execution',
            next_action='Integrate remaining specialist model interfaces and real event-driven studio; port heavy Metal tools to remote backend separately.',
            inputs=[evidence(ROOT,'real_video/agentvideo_gemini.py')],
            outputs=[evidence(ROOT,output/'result.json'),evidence(ROOT,output/'trace.json')]))
        program.export()
    finally:program.db.close()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
