"""Save reproducible offline checks and coordinator evidence for safety repair."""
import json
from pathlib import Path
import subprocess
import sys
from real_video.gaussian_program import Program,ROOT,evidence

if __name__=='__main__':
    out=ROOT/'artifacts/cloud/eval_safety_repair_v1'
    out.mkdir(parents=True,exist_ok=False)
    modules=['tests.test_jupyter_transfer','tests.test_runpod_guard','tests.test_chunk_transfer',
             'tests.test_eval_preflight','tests.test_a14b_preflight']
    command=[sys.executable,'-m','unittest',*modules,'-v']
    done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
    report=dict(command=command,exit_code=done.returncode,stdout=done.stdout,stderr=done.stderr,
        scope='Transfer/integrity/resource checks and authenticated shutdown. Not model training or visual quality.',
        original_bundle_bytes=54201342,original_bundle_sha256='73c3a144c5fbbc73ad2137b6c05cc8e810b4637c4a900d705a1e00cd38044683',
        verified_local_roundtrip=True,chunk_count=13,max_chunk_bytes=4194304,
        v2_bundle_bytes=54205641,v2_manifest_sha256='25d40fbba4fd0733a6ad1a6a8cebd5718a8de749a15a160f83ba918572720f07',
        canary1='Remote API requests failed; independent local timer stopped Pod automatically.',
        canary2='Explicit User-Agent repaired HTTP403. External EXITED observed before local fallback deadline.',
        balances=dict(before_canaries=3.24,after_canaries=2.87,displayed_difference=.37,
            scope='Account balance difference, not a finalized per-stage invoice'),
        coordinator_cycle=dict(max_jobs=1,budget_seconds=30,job='490eb62d67aa4b27',ready=False),
        resource_check=dict(available_ram_gib=9899852/1024**2,battery_percent=89,local_heavy_jobs=0),
        independent_review='Subagent found manifest read/hash race and remote-stop verification omission; both repaired. Windows real symlink/reparse test coverage remains incomplete.',
        weights_changed=False,new_media_from_this_repair=False)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    p=Program()
    names=['cloud/chunk_transfer.py','cloud/eval_preflight.py','cloud/evaluate_a14b.py',
           'cloud/jupyter_transfer.py','cloud/runpod_boot_guard.py','cloud/runpod_control.py',
           'cloud/runpod_guarded_session.py','research/sweep_2026-10-03_eval_safety.json']
    outputs=[evidence(ROOT,str((out/'report.json').relative_to(ROOT))),
             evidence(ROOT,'artifacts/cloud/shutdown_canary_v2/verification.json')]
    record=dict(role='generation',issue_key='generation.remote_eval_safety',
        hypothesis='Pinned resumable uploads plus independently running remote and local stop timers avoid the previous browser-dependent transfer and cutoff failures.',
        inputs=[evidence(ROOT,n) for n in names],outputs=outputs,outcome='partial',
        observation='Local transfer roundtrip and focused tests passed with one platform symlink skip. First remote timer failed HTTP requests; local fallback stopped it. Changed explicit User-Agent after remote diagnostics showed403 versus200; second canary stopped remotely before local deadline. Temporary key approved by user; no top-ups. Evaluation safety improved, not a model quality success.',
        next_action='Run exactly one bounded frozen-weight ablation, retrieve artifacts and stop, then disable temporary key and remove injected Pod secret. Review actual videos. Do not claim guaranteed billing caps, native causal Gaussian generation, or unmeasured speedup.')
    print(p.record_experiment(record));p.issue(record['issue_key'],'generation','Remote safeguards tested; final evaluation and credential cleanup pending',outputs,record['next_action'])
    p.export();p.db.close()
    print(json.dumps({'test_exit_code':done.returncode,'report':str(out/'report.json')}))
    raise SystemExit(done.returncode)
