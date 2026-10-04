"""Record the primary agent's actual sampled-frame review and measured ablation."""
import json
from pathlib import Path
from statistics import mean

from real_video.articulation_quality import create_review, decide_review
from real_video.gaussian_program import Program, ROOT, evidence

OUT = ROOT / 'artifacts/cloud/a14b_remote_eval_v2'


def main():
    extraction = json.loads((OUT / 'review/extraction.json').read_text())
    report = json.loads((OUT / 'results/evaluation/report.json').read_text())
    summaries = []
    for item in extraction:
        review = create_review(item['video'], item['frames'])
        review['claim_type'] = 'conditional_motion_generation'
        review['reviewer'] = dict(kind='agent', name='Primary Codex agent: sampled-frame inspection', observed_video=False)
        review['provenance'] = dict(future_conditioned=False, test_used_for_training=False, test_used_for_selection=False)
        notes = {
            'paw_placement': ('uncertain', 'Hooves are outside the image; contact cannot be assessed.'),
            'limb_bending': ('uncertain', 'Only upper forelegs are visible; no full-body articulation test.'),
            'body_shape': ('pass', 'Visible donkey head and neck remain recognizable through the sampled head lift and turn; full body is not assessed.'),
            'tearing': ('pass', 'No obvious disconnected face or neck pieces in these five samples.'),
            'temporal_coherence': ('uncertain', 'Five sampled frames show action progression, but continuous playback/flicker and chewing contact were not verified.')}
        seed = int(item['name'].rsplit('_', 1)[1])
        observations = (
            ['Muzzle near bowl, head angled down.', 'Muzzle raised above bowl.', 'Head remains raised with changed muzzle pose.', 'Head turns more toward camera.', 'More frontal donkey face, bowl and farm remain visible.']
            if seed == 103501 else
            ['Muzzle near bowl, head angled down.', 'Muzzle near orange segment above bowl.', 'Muzzle rises; orange shapes in bowl change.', 'Head turns toward camera with mouth-region change.', 'Head raised and slightly turned; orange detail differs from initial state.'])
        for frame, observation in zip(review['frames'], observations):
            frame['observation'] = observation
        for key, (status, observation) in notes.items():
            review['dimensions'][key] = dict(status=status, observation=observation,
                evidence_frame_ids=[f['id'] for f in review['frames']])
        review['limitations'] = [
            'Five exact frames per clip inspected, not continuous playback.',
            'All three branches look closely similar; no clear Gaussian-memory quality improvement established.',
            'Seed 103502 fruit/muzzle interaction needs continuous close inspection; eating/contact fidelity uncertain.',
            'New prompt and seeds, same training subject/anchor. The two-step trained checkpoint was frozen, not retrained here.',
            'RGB diffusion video with static planar Gaussian recall, not updated 3D Gaussian state.',
            'Independent protocol audit completed, but independent visual reviewer hit its usage limit before media review.']
        review['decision'] = decide_review(review, item['video'])
        target = OUT / 'review' / (item['name'] + '_review.json')
        with target.open('x') as stream: json.dump(review, stream, indent=2)
        summaries.append(dict(name=item['name'], video_sha256=item['sha256'], decision=review['decision']))
    costs = {}
    for branch in report['branches']:
        pipelines = [report['stages'][f'{branch}_{seed}_pipeline'] for seed in report['seeds']]
        decodes = [report['stages'][f'{branch}_{seed}_decode'] for seed in report['seeds']]
        costs[branch] = dict(mean_pipeline_seconds=mean(s['seconds'] for s in pipelines),
            mean_decode_seconds=mean(s['seconds'] for s in decodes),
            max_pipeline_allocated_bytes=max(s['peak_allocated_bytes'] for s in pipelines),
            max_decode_allocated_bytes=max(s['peak_allocated_bytes'] for s in decodes))
    slowdown = 100*(costs['gaussian_memory']['mean_pipeline_seconds']/costs['original']['mean_pipeline_seconds']-1)
    summary = dict(accepted=False, videos=summaries, costs=costs,
        gaussian_pipeline_slowdown_percent=slowdown, training_this_run=False,
        model_memory_savings_demonstrated=False, gaussian_native_generation=False,
        memory=report['memory_bytes'], remote_worker_seconds=report['worker_seconds'],
        full_evaluation_subprocess_seconds=556.8822297840379,
        download_and_verification_seconds=940.3488653129898,
        additional_cache_verification_seconds=report['model_cache_release']['seconds'],
        encoding_caveat='Raw *_encode stages also include SHA hashing, paired-frame copies and pixel-difference evaluation. They are not isolated codec timings and must not be compared as pure encoding costs.',
        cleanup=dict(pod_status='EXITED', current_spend_rate_usd_per_hour=0,
            stop_observed_utc='2026-10-04T02:01:43Z', controller_stop_record_utc='2026-10-04T01:51:06Z',
            default_startup_restore_api_succeeded=True, temporary_key_ui_status='Disabled',
            autopay='Disabled', no_assistant_topup=True,
            displayed_balance_before=2.87, displayed_balance_after=11.25,
            cost_caveat='Balance increased during the session; subtraction cannot establish its charge. No new spend is authorized by the increased balance. About 27.6 minutes to controller stop at a conservative $3.53/hr is about $1.63, an estimate, not an invoice.'),
        tests=dict(cache_and_resource_tests=31, passed=30, skipped=1,
            initial_failure='Windows mocked-Linux fixture stat/fstat timestamp mismatch; test emulation repaired, production Linux identity guard unchanged.',
            independent_visual_review='Unavailable: reviewer usage limit'),
        next_action='No new paid launch. Implement and train a causal geometry/correspondence writer under a separately bounded plan; compare past-only memory, no-memory and matched non-Gaussian memory on unseen clips. Do not rerun this unchanged static-reader hypothesis.')
    target = OUT / 'review/summary.json'
    with target.open('x') as stream: json.dump(summary, stream, indent=2)
    program = Program()
    try:
        result = program.record_experiment(dict(role='generation', issue_key='generation.part_memory_backbone',
            hypothesis='A previously trained Gaussian appearance reader may improve a new action over frozen LoRA-only and original Wan on two paired seeds.',
            inputs=[evidence(ROOT, p) for p in ['artifacts/cloud/a14b_eval_bundle_v2.zip',
                'artifacts/cloud/a14b_remote_eval_v2/cache_patch/deployment.json',
                'research/sweep_2026-10-03_ablation_review.json']],
            outputs=[evidence(ROOT, p) for p in ['artifacts/cloud/a14b_remote_eval_v2/results/evaluation/report.json',
                'artifacts/cloud/a14b_remote_eval_v2/review/summary.json',
                'artifacts/cloud/a14b_remote_eval_v2/key_disabled.jpg',
                'artifacts/cloud/a14b_remote_eval_v2/billing_stopped.jpg']],
            outcome='partial', observation=f'All six 33-frame videos retrieved with matching hashes. Head lift/turn visible in samples, no clear memory quality superiority. Pipeline means original {costs["original"]["mean_pipeline_seconds"]:.3f}s, Gaussian {costs["gaussian_memory"]["mean_pipeline_seconds"]:.3f}s ({slowdown:.2f}% slower), no memory savings. Static planar reader only, no training this run or 3D writer. Evaluation556.882s; GPU stopped, key disabled. Independent media review unavailable; overall quality unaccepted.',
            next_action=summary['next_action']))
        program.export()
        print(json.dumps(dict(ledger=result, summary=str(target), costs=costs, slowdown_percent=slowdown), indent=2))
    finally: program.db.close()


if __name__ == '__main__': main()
