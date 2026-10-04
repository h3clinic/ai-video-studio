"""Bounded decision-stage experiment, no video or GPU work."""
import argparse
import json
from real_video.agentvideo_bridge import load_upstream
from real_video.agentvideo_gemini import GeminiLLM, CallBudget
from real_video.agentvideo_decision_stage import run_sizes
from real_video.studio_agent_events import project_trace
from real_video.gaussian_program import ROOT,Program,evidence


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',required=True)
    p.add_argument('--live',action='store_true')
    args=p.parse_args()
    upstream=load_upstream(args.upstream)
    source=ROOT/'artifacts/cloud/agentvideo_gemini_budget_fix_live_v1/result.json'
    previous=json.loads(source.read_text())
    if previous['status']!='script_verified_not_video':raise ValueError('Verified director result required')
    out=ROOT/'artifacts/cloud'/('agentvideo_decision_live_v1' if args.live else 'agentvideo_decision_contract_v1')
    out.mkdir(exist_ok=False)
    answers=iter([dict(length_m=2,height_m=1.5,legs=4),dict(length_m=.08,height_m=.08,legs=0)])
    def fixture(*args):
        return dict(candidates=[dict(finishReason='STOP',content=dict(parts=[dict(text=json.dumps(next(answers)))]))])
    client=GeminiLLM('gemini-3.8-flash',budget=CallBudget(calls=6,output_tokens=12288),transport=None if args.live else fixture)
    bus=upstream['Bus'](echo=False)
    result=dict(live=args.live,upstream=upstream['provenance'],source=evidence(ROOT,source))
    try:result.update(run_sizes(upstream,previous['script'],client,bus))
    except Exception as error:
        result.update(status='failed',error=str(error),scene_modified=False)
        if 'decision' in bus.agents:bus.log('decision','fail',str(error))
    result['usage']=client.usage
    bus.dump(str(out/'trace.json'))
    (out/'studio_events.json').write_text(json.dumps(project_trace(dict(agents=bus.agents,events=bus.events)),indent=2))
    (out/'result.json').write_text(json.dumps(result,indent=2))
    program=Program()
    try:
        program.record_experiment(dict(role='generation',issue_key='generation.agentvideo_decision',
            hypothesis='Original Decision Agent can inherit a verified director brief and propose subject scale through Gemini.',
            observation=result['status']+'; estimates only, no geometry or video changed.',
            outcome='failed_execution' if result['status']=='failed' else 'partial',
            next_action='Bind verified measured geometry to size proposals and connect real job events into studio. Do not treat broad numeric checks as anatomy verification.',
            inputs=[evidence(ROOT,source),evidence(ROOT,'real_video/agentvideo_decision_stage.py')],
            outputs=[evidence(ROOT,out/'result.json'),evidence(ROOT,out/'trace.json')]))
        program.export()
    finally:program.db.close()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
