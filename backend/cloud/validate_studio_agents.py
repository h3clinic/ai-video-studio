"""One explicit live provider trial; never runs local model/render workloads."""
import argparse
import json
from pathlib import Path
import uuid
from real_video.studio_agent_tasks import plan_agent_task
from real_video.studio_assets import select_asset


def record_trial(root,output):
    from real_video.gaussian_program import Program, evidence
    from real_video.studio_backend import Studio, utc
    data=json.loads((output/'trial.json').read_text())
    program=Program(root=root)
    program.record_experiment(dict(role='generation',outcome='partial',issue_key='generation.agent_scoped_gaussian_edits',
        hypothesis='An owner-key Gemini specialist can produce verified scoped recolour and Gaussian vector controls through the original AgentVideo harness.',
        observation='Live Gemini decision passed typed validation. Tiny geometry/permission regression tests passed. Remote model/renderer execution NOT performed: RunPod returned HTTP401. No accepted new video or speedup.',
        next_action='Use an authorized running guarded GPU to execute the prepared canonical diagnostic; add grounded part/texture/contact correspondence before claiming the donkey/apple scene is fixed.',
        inputs=[evidence(root,p) for p in ('real_video/studio_agent_tasks.py','real_video/gaussian_agent_edits.py','research/sweep_2026-10-04_studio_part_execution.json')],
        outputs=[evidence(root,str((output/n).relative_to(root))) for n in ('decision.json','trial.json')]))
    program.export();program.db.close()
    # Use only while the desktop backend is stopped, keeping one jobs writer.
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8790))==0:raise RuntimeError('Stop desktop backend before importing trial')
    studio=Studio(root);identifier=output.name.removeprefix('live-agent-')
    if not any(j['id']==identifier for j in studio.jobs):
        studio.jobs.append(dict(id=identifier,kind='agent_task',projectId='apple-experiment',agentId='vector-agents',partId='whole_apple',
            prompt='Canonical five-second turn-and-return with subtle red recolour; separate from eating scene.',
            status='blocked',stage='Gemini controls validated; remote execution blocked by RunPod HTTP401',
            createdAt=utc(),finishedAt=utc(),output=None,error=None,reply=data['decision']['summary'],
            decision=data['decision'],usage=data['usage']))
        studio.persist()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');parser.add_argument('--record-existing');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.record_existing:
        import re
        if args.live or not re.fullmatch(r'live-agent-[a-f0-9]{32}',args.record_existing):raise ValueError('Exact existing trial ID required')
        record_trial(root,root/'artifacts/studio'/args.record_existing);print('Recorded trial and blocked task; no provider call');return
    if not args.live:raise RuntimeError('Explicit live Gemini trial required')
    output=root/'artifacts/studio'/('live-agent-'+uuid.uuid4().hex)
    prompt=('For the existing canonical apple only, prepare a five-second fixed-camera diagnostic. '
        'Gently turn its Gaussian object to one side and back using angular acceleration; keep translation zero. '
        'Retain shape, IDs and source colour variation, with a subtle richer red recolour if appropriate. '
        'Do not simulate eating, camera orbit, crop overlays, scene contact or new texture detail. '
        'This tests actual typed vector/colour commands, not the completed donkey/apple video.')
    result=plan_agent_task(prompt,'vector-agents','whole_apple',select_asset(root,'apple-experiment','whole_apple'),output)
    report=dict(status='provider_decision_validated',decision=result['decision'],usage=result['usage'],
        gpu_execution='not performed: current RunPod credential returned HTTP 401',
        accepted_video=False,source_asset=select_asset(root,'apple-experiment','whole_apple')['sha256'])
    (output/'trial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),**report)),flush=True)


if __name__=='__main__':main()
