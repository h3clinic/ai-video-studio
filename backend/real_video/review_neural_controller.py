"""Record sampled-frame review of every retained neural-controller experiment."""
import json
import hashlib
from pathlib import Path
import imageio.v2 as imageio
from .articulation_quality import create_review,decide_review

ROOT=Path('artifacts/real_video/hunyuan_gaussian')


def run():
    cases=[('v1','walk'),('v2','walk'),('v3','walk'),('v3','trot'),('v3','idle')]
    observations={
        'v1':'Severe belly sag/stretch during limb motion; contacts pull limbs out of range.',
        'v2':'Belly shape improved, but forelegs curl and narrow unnaturally; paws float.',
        'v3':'Broad torso retained; front leg becomes unnaturally thin in some poses and ground contact is not correct.'}
    decisions=[]
    for version,gait in cases:
        root=ROOT/f'neural_controller_{version}'/gait
        video=root/'gaussian_motion_7s.mp4'
        review=create_review(video,[dict(path=root/f'frame_{i:03d}.png',frame_index=i) for i in [0,20,40,60,80,104]])
        review['claim_type']='conditional_motion_generation'
        review['provenance']=dict(future_conditioned=False,test_used_for_training=False,test_used_for_selection=False)
        review['reviewer']=dict(kind='agent',name='Codex source and six-timepoint dual-view inspection',observed_video=False)
        for frame in review['frames']:
            frame['observation']=(
                'Idle preserves the broad torso and mostly holds a raised-paw pose; this is not grounded standing.'
                if gait=='idle' else observations[version])
        judgments={
            'paw_placement':('fail','Paws do not establish natural grounded contact; no validated stance cycle.'),
            'limb_bending':('fail','Forelimb shape is too narrow/curled compared with a correct articulated cat.'),
            'body_shape':('fail' if version=='v1' else 'uncertain',observations[version]),
            'tearing':('uncertain','No large detached point cloud in sampled views; local thinning and triangle strain prevent acceptance.'),
            'temporal_coherence':('uncertain','Sampled pose changes inspected. Continuous playback not observed; timestamps alone do not establish smoothness.')}
        for key,(status,observation) in judgments.items():
            review['dimensions'][key]=dict(status=status,observation=observation,evidence_frame_ids=[f'frame_{i:03d}' for i in range(6)])
        reader=imageio.get_reader(video)
        hashes=[hashlib.sha256(frame.tobytes()).hexdigest() for frame in reader]; reader.close()
        review['numeric_metrics']=dict(decoded_frames=len(hashes),unique_decoded_frames=len(set(hashes)))
        review['limitations']=['Dog-trained pretrained model, no new training. Static guidance is not a video sequence.',
            'Manual cat rig and straight-line headless controller adaptation, not general species/topology support.',
            'Visual-development runs, not held-out generalization evaluation.',
            'No success or Wan-efficiency claim; original static asset unchanged.']
        review['decision']=decide_review(review,video)
        (root/'visual_review.json').write_text(json.dumps(review,indent=2))
        decisions.append(dict(version=version,gait=gait,accepted=review['decision']['accepted'],video=str(video),**review['numeric_metrics']))
    (ROOT/'neural_controller_selection.json').write_text(json.dumps(dict(
        candidates=decisions,quality_approved_candidate=None,default_scene_changed=False,
        next_bottleneck='Validated cat rig, anatomical skin weights and reliable contact-aware retargeting'),indent=2))
    print(json.dumps(decisions),flush=True)


if __name__=='__main__': run()
