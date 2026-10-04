"""Record explicit sampled-frame observations, not an automatic quality pass."""
import json
from real_video.articulation_quality import create_review, decide_review
from real_video.gaussian_program import Program, ROOT, evidence

OUT=ROOT/'artifacts/cloud/a14b_pilot_v1'
RUN=OUT/'pilot_output_repaired'
if __name__=='__main__':
    video=RUN/'gaussian_adapted.mp4'
    review=create_review(video,[dict(path=OUT/f'gaussian_adapted_{i:03d}.png',frame_index=i) for i in (0,8,16,24,32)])
    review['claim_type']='conditional_motion_generation'
    review['reviewer']=dict(kind='agent',name='Codex sampled-frame review',observed_video=False)
    review['provenance']=dict(future_conditioned=False,test_used_for_training=False,test_used_for_selection=False)
    observations={
        'paw_placement':('uncertain','Hooves outside framing; cannot assess contact.'),
        'limb_bending':('uncertain','Partial forelegs only; articulated limb motion not tested.'),
        'body_shape':('pass','Head and neck remain donkey-shaped with recognizable eye, muzzle, mane and fur across samples.'),
        'tearing':('pass','No obvious head or neck fragmentation in the five samples.'),
        'temporal_coherence':('uncertain','Muzzle changes and orange segment at mouth suggest chewing; continuous playback not reviewed.')}
    for frame in review['frames']:
        frame['observation']='Compared exact frame against baseline and clock-aligned source in comparison.jpg.'
    for key,(status,observation) in observations.items():
        review['dimensions'][key]=dict(status=status,observation=observation,evidence_frame_ids=[f['id'] for f in review['frames']])
    review['limitations']=['Five sampled frames only, no continuous playback claim.',
        'Single-source fitting, no held-out generalization.',
        'Baseline and adapted look similar; no demonstrated quality improvement.',
        'RGB video with planar Gaussian appearance recall; no native Gaussian motion or 3D writer.']
    review['decision']=decide_review(review,video)
    (OUT/'visual_review.json').write_text(json.dumps(review,indent=2))
    p=Program()
    record=dict(role='generation',issue_key='generation.part_memory_backbone',
        hypothesis='A14B can train low-noise LoRA and Gaussian part-reader weights and generate paired RGB video on H100.',
        inputs=[evidence(ROOT,'cloud/wan_a14b_experiment.py'),evidence(ROOT,'research/sweep_2026-10-03_a14b_execution.json')],
        outputs=[evidence(ROOT,str((RUN/'report.json').relative_to(ROOT))),evidence(ROOT,str((OUT/'visual_review.json').relative_to(ROOT))),evidence(ROOT,str(video.relative_to(ROOT)))],
        outcome='partial',
        observation='Model downloaded and hash verified. Missing ftfy caused prompt encoding failure before training; preserved failure, installed dependency, resumed with 550-second budget. Two steps trained 14,419,072 parameters; frozen base unchanged, exact reload. Both 33-frame 480p clips generated. Baseline/adapted denoising 56.93/62.52 seconds: adapted about 9.8% slower, no savings. Sampled frames coherent but similar; no generalization acceptance. Retrieved results and stopped pod at $0/hr; balance $9.94 to $8.24, displayed difference $1.70.',
        next_action='Add prompt-cleaning dependency preflight. Evaluate held-out actions and memory ablations before scaling training. Native 3D writer and motion remain unresolved.')
    print(p.record_experiment(record));p.export();p.db.close()
