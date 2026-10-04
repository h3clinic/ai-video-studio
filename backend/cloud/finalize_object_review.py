"""Record this trial's human-readable sampled-frame judgments; no model work."""
import json
from pathlib import Path
from real_video.articulation_quality import create_review,decide_review
from real_video.gaussian_program import Program,evidence,ROOT

base=ROOT/'artifacts/cloud/object_edit_remote_v2'
output=base/'results/output'
video=output/'gaussian_apple.mp4'
review=create_review(video,[dict(path=output/f'comparison_{i:03d}.png',frame_index=i) for i in (0,20,39)])
review.update(claim_type='reconstruction',
    reviewer=dict(kind='agent',name='Primary agent sampled rendered-frame review; not independent',observed_video=False),
    provenance=dict(future_conditioned=True,test_used_for_training=False,test_used_for_selection=True))
observations=[
    'Right panel contains dark generated apple, unchanged donkey head, visible original peel and pith behind apple.',
    'Donkey muzzle moves upward while apple stays in bowl; original orange segment remains in muzzle. No new apple-eating action.',
    'Donkey head turns relative to first frame; apple persists but lighting and bottom contact are not convincingly integrated.'
]
for item,note in zip(review['frames'],observations):item['observation']=note
for name in ('paw_placement','limb_bending'):
    review['dimensions'][name]=dict(status='uncertain',observation='Not visible in this head-and-bowl shot.',evidence_frame_ids=['frame_000','frame_001','frame_002'])
review['dimensions']['body_shape']=dict(status='uncertain',observation='Head outline retained; whole-body anatomy not in view. Apple is recognizable but coarse.',evidence_frame_ids=['frame_000','frame_001','frame_002'])
review['dimensions']['tearing']=dict(status='fail',observation='Apple insertion boundary leaves orange peel/pith and poorly matched dark shading; oblique apple views have small holes.',evidence_frame_ids=['frame_000','frame_001','frame_002'])
review['dimensions']['temporal_coherence']=dict(status='uncertain',observation='Three spaced rendered frames show retained source head motion and persistent apple. Continuous decoded playback was not inspected.',evidence_frame_ids=['frame_000','frame_001','frame_002'])
review['limitations']=['Evidence images are saved pre-encoding render comparisons, not decoded MP4 extractions.',
    'Only main orange replaced; loose segments remain.', 'No complete 3D scene or learned material correspondence.',
    'No trained agent controller or new motion; frozen pretrained Shap-E asset generation followed by Gaussian conversion.',
    'Review is not independent; quality is unaccepted.']
(base/'visual_review.json').write_text(json.dumps(review,indent=2))
(base/'visual_decision.json').write_text(json.dumps(decide_review(review,video),indent=2))
r=json.loads((output/'report.json').read_text());m=r['frame_metrics']
summary=dict(accepted=False,duration_seconds=r['duration_seconds'],frames=r['frames'],
    edited_full_raster_seconds=sum(x['edited_full_raster_seconds'] for x in m),
    incremental_seconds=sum(x['incremental_seconds_including_scan_copy'] for x in m),
    scene_write_seconds=sum(x['scene_write_seconds'] for x in m),
    dirty_pixel_fraction=sum(x['dirty_pixels'] for x in m)/(len(m)*512*288),
    max_full_vs_incremental_error=max(x['full_vs_incremental_max_error'] for x in m),
    nonfruit_state_exact=all(x['nonfruit_state_exact'] for x in m),
    outside_cache_pixels_exact=all(x['outside_exact'] for x in m),
    compressed_scene_state_bytes=r['state_bytes'],apple_archive_bytes=r['apple_bytes'],
    scope='Local scan/copy/raster vs full raster only. No end-to-end Wan comparison; Gaussian state is larger than source MP4.',
    next_action='Improve semantic fruit segmentation, contact/depth calibration and relighting; cache GPU spatial indices to remove CPU scan overhead before claiming local-edit speedup.')
summary['incremental_slowdown']=summary['incremental_seconds']/summary['edited_full_raster_seconds']
(base/'summary.json').write_text(json.dumps(summary,indent=2))
program=Program()
try:
    for version,outcome,note,next_action in [
        ('v1','failed_execution','26 tests passed and learned apple asset saved; depth AutoConfig failed. Pod stopped and evidence retrieved.','Use explicit official pinned depth config and preserve/reuse generated apple.'),
        ('v2','partial','Completed 40 Gaussian-rendered frames. Frozen nonfruit fields and outside cached pixels exact. Dark apple and residual orange fail clean-edit quality. Incremental scan/raster is 2.62 times full raster.','Repair segmentation/lighting/contact and cache spatial indices. Do not repeat unchanged quality hypothesis or claim speedup.')]:
        directory=Path('artifacts/cloud/object_edit_remote_'+version)
        entry=dict(role='rendering',issue_key='rendering.object_local_memory',outcome=outcome,
            hypothesis='Generate one learned 3D apple and restrict replacement to owned Gaussian state; local raster may avoid full-frame work.',
            observation=note,next_action=next_action,
            inputs=[evidence(ROOT,directory/'inputs.zip'),evidence(ROOT,'research/sweep_2026-10-03_object_local_execution.json')],
            outputs=[evidence(ROOT,directory/'control.json'),evidence(ROOT,directory/'results/output/report.json')])
        if version=='v2':entry['outputs'] += [evidence(ROOT,base/'summary.json'),evidence(ROOT,base/'visual_review.json')]
        print(program.record_experiment(entry))
    program.export()
finally:program.db.close()
print(json.dumps(summary,indent=2))
