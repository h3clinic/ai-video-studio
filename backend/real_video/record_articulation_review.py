"""Record the root agent's actual rejected visual inspection for this trial.

This is a signed-in-prose observation record, not a learned/automatic quality
metric. The observations below were written AFTER viewing the named renders.
It intentionally does not claim continuous real-time video playback.
"""
import json
from pathlib import Path
from .articulation_quality import create_review, decide_review
from .checkpoint_io import digest

ROOT=Path('artifacts/real_video/true3d/articulated_weight_loop/v2_constrained')


def main():
    video=ROOT/'joint_skinning_native.mp4'
    output=ROOT/'visual_review.json'
    if output.exists(): raise FileExistsError('Preserve completed visual judgment')
    frames=[10,14,22,32]
    review=create_review(video,[dict(path=ROOT/f'comparison_{frame:02d}.jpg',frame_index=frame-2) for frame in frames])
    review['candidate']='Rightmost image_refined panel; center pose_only also fails anatomy.'
    review['reviewer']=dict(kind='agent',name='Root agent direct view_image inspection',observed_video=False)
    review['claim_type']='reconstruction'
    review['provenance']=dict(future_conditioned=True,test_used_for_training=False,test_used_for_selection=False)
    descriptions=[
        'Rightmost candidate: rear legs are shortened/stubby with a trailing thin appendage; reference shows recognizable long rear lower limbs. The lifted front paw shape is also wrong.',
        'Rightmost candidate: front legs visually merge into a near-vertical column; reference shows a distinct lifted and bent foreleg. Rear legs remain malformed.',
        'Rightmost candidate: extended forward limb is thick and poorly shaped, not the reference paw reach; hips and rear-leg silhouette differ substantially.',
        'Rightmost candidate: an incorrect bulky shoulder/body, thick forward leg and stubby rear legs persist. Less extreme surface damage than unconstrained v1 does not make this natural articulation.'
    ]
    for record,description in zip(review['frames'],descriptions):record['observation']=description
    evidence=[record['id'] for record in review['frames']]
    observations={
        'paw_placement':('fail','The near-vertical foreleg at source14 misses the distinct lifted paw, and later forward-paw reach is incorrect.'),
        'limb_bending':('fail','Rear limbs stay stubby or folded into the pelvis; front-leg bending and separation do not match the source.'),
        'body_shape':('fail','Bulky inferred torso and shoulder proportions persist, especially in the independently viewed oblique renders.'),
        'tearing':('fail','Less severe than v1, but rear thin appendages, faceted transitions and distorted leg surfaces remain visible; cannot certify intact anatomical surfaces.'),
        'temporal_coherence':('uncertain','Sampled states show motion, but continuous real-time playback was not observed; no smooth natural-gait approval is warranted. Spatial failures already reject the model.')
    }
    for key,(status,observation) in observations.items():
        review['dimensions'][key]=dict(status=status,observation=observation,evidence_frame_ids=evidence)
    review['oblique_evidence']=[dict(path=str((ROOT/f'image_refined_oblique_{frame:02d}.png').resolve()),
        sha256=digest(ROOT/f'image_refined_oblique_{frame:02d}.png'),source_frame=frame,
        observation='Directly inspected: weak/malformed hind-limb geometry, bulky torso and incorrect foreleg shape remain.') for frame in [14,32]]
    evaluation=json.loads((ROOT/'evaluation.json').read_text())
    review['numeric_metrics']=evaluation['metrics']['image_refined']
    review['limitations']=[
        'Inspected rendered comparison frames10,14,22,32 and oblique14,32; not continuous video playback.',
        'Comparison JPGs are pre-encode source frames for the hashed native MP4, not decoded JPEG extractions; frame_index=source_frame-2.',
        'Held-out image frames are within the same observed clip, between fitting anchors; not external generation evaluation.',
        'This model changes 3D joint and skinning weights, not the canonical mesh, Gaussian appearance or Wan weights.'
    ]
    review['decision']=decide_review(review,video)
    assert not review['decision']['accepted'] and review['decision']['status']=='rejected'
    output.write_text(json.dumps(review,indent=2))
    selection=dict(accepted_model=None,quality_achieved=False,tenfold_quality_claim=False,compute_savings_claim=False,
        reviewed_artifact_sha256=review['decision']['video_sha256'],visual_review=str(output),
        candidates={'v1_pose_only':'rejected','v1_image_refined':'rejected','v2_pose_only':'rejected','v2_image_refined':'rejected'},
        reason='Anatomical shape, leg bending and paw placement fail direct inspection. Do not promote these fitted weights into a successful generator.',
        next_dependency='A validated animatable 3D animal rest shape and rig, then train the motion model through this differentiable 3D path.')
    selection_path=ROOT.parent/'selection.json'
    if selection_path.exists():raise FileExistsError('Preserve selection decision')
    selection_path.write_text(json.dumps(selection,indent=2))
    print(json.dumps(review['decision']),flush=True)


if __name__=='__main__':main()
