"""Record failed remote setup honestly; never relaunch resources."""
from real_video.gaussian_program import Program,ROOT,evidence

if __name__=='__main__':
    p=Program()
    output=[evidence(ROOT,'artifacts/cloud/a14b_eval_session_v1.json'),evidence(ROOT,'artifacts/cloud/a14b_eval_stopped_v1.jpg')]
    record=dict(role='generation',issue_key='generation.remote_eval_safety',
        hypothesis='Evaluate fixed A14B adapters with original/LoRA-only/Gaussian-reader branches on two new seeds and a new action remotely.',
        inputs=[evidence(ROOT,'cloud/evaluate_a14b.py'),evidence(ROOT,'artifacts/cloud/a14b_eval_bundle_v1.zip'),evidence(ROOT,'research/sweep_2026-10-03_a14b_ablation.json')],
        outputs=output,outcome='failed_execution',
        observation='No evaluation occurred. Partial browser upload failed and Jupyter lost connection. Next clock check was past manually supervised45minute session cutoff. Pod stopped and verified0USD/hr. Displayed balance8.20to3.24USD,4.96USD consumed, no top-ups. This exceeded the intended2.64USD session plan but stayed within user8USD remaining authorization. All prior local model artifacts preserved. Manual-only budget protection was inadequate.',
        next_action='Require verified provider-side automatic billing stop before further rental; prepare chunked resumable upload and preflight without GPU. Do not repeat failed manual-only setup.')
    print(p.record_experiment(record))
    p.issue('generation.remote_eval_safety','generation','Remote upload failure and unenforced shutdown deadline',output,record['next_action'])
    p.export();p.db.close()
