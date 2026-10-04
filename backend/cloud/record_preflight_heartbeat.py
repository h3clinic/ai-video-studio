"""Bounded local tests and immutable heartbeat evidence; no pretrained loads."""
import json
from pathlib import Path
import subprocess
import sys
from real_video.gaussian_program import Program,ROOT,evidence

if __name__=='__main__':
    out=ROOT/'artifacts/cloud/a14b_preflight_repair_v1'
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','unittest','tests.test_a14b_preflight','-v']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
    report=dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
        available_ram_gib_at_start=6800148/1024**2,battery_percent=99,
        coordinator_cycle=dict(max_jobs=1,budget_seconds=30,job='e191a01fe2d1417b',ready=False),
        new_heavy_experiments=0,paid_compute=False,weights_modified=False,new_media=False,
        scope='Startup dependency preflight and honest measurement labels; not a model-quality improvement')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    p=Program()
    inputs=[evidence(ROOT,n) for n in ['cloud/a14b_preflight.py','cloud/wan_a14b_experiment.py','tests/test_a14b_preflight.py']]
    outputs=[evidence(ROOT,str((out/'report.json').relative_to(ROOT)))]
    record=dict(role='generation',issue_key='generation.part_memory_backbone',
        hypothesis='Missing prompt dependencies can be rejected before model loading, avoiding the previously observed ftfy failure after a costly load.',
        inputs=inputs,outputs=outputs,outcome='accepted_mechanism' if result.returncode==0 else 'failed_execution',
        observation='One bounded coordinator cycle retained existing visual rejection. Local RAM below 8GiB so no heavy experiment. Added CPU dependency and actual prompt-cleaning smoke checks, bundle allowlist inclusion, and timing/baseline allocation disclosures. No pretrained model execution or new video. Test result is recorded, not a quality claim.',
        next_action='Evaluate held-out actions and Gaussian-memory ablation benefit before more training; native causal 3D writer and motion remain unresolved. Future cloud setup must run preflight before loading models.')
    print(p.record_experiment(record))
    p.issue('generation.part_memory_backbone','generation','A14B integration works; generalization, native Gaussian output and efficiency not established',outputs,
        record['next_action'])
    p.export();p.db.close()
    print(json.dumps(report,indent=2))
    raise SystemExit(result.returncode)
