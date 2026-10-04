"""Record root's actual sampled-image review of the two lifting experiments."""
import json
from pathlib import Path
from .articulation_quality import create_review, decide_review
from .checkpoint_io import digest


def main():
    base = Path('artifacts/real_video/detail_lift')
    for variant in ['v3', 'v4_tangent']:
        folder = base/variant
        destination = folder/'visual_review.json'
        if destination.exists(): raise FileExistsError('Preserve reviews')
        video = folder/'camera_only_7seconds.mp4'
        review = create_review(video, [dict(path=folder/f'inspection_{i:03d}.png', frame_index=i) for i in [0, 28, 84]])
        review.update(claim_type='reconstruction',
            reviewer=dict(kind='agent', name='Root: direct inspection of three camera poses and separate three-source-pose contact sheet', observed_video=False),
            provenance=dict(future_conditioned=False, test_used_for_training=False, test_used_for_selection=False))
        observations = [
            'Reference camera: lifted cat retains source stripes, ears, raised tail, lifted foreleg and lawn; existing blur remains.',
            '+18 degree camera: recognizable cat but dark disocclusion holes and isolated splats around the head, front leg and upper boundary; background is a partial curved sheet.',
            '-18 degree camera: stretched torso appearance, missing region behind tail/back and scattered splats; far-side anatomy is absent.'
        ]
        for frame, observation in zip(review['frames'], observations): frame['observation'] = observation
        evidence = [f['id'] for f in review['frames']]
        judgments = {
            'paw_placement': ('uncertain', 'Front reference feet remain recognizable but blur and oblique holes preclude confirming ground contact.'),
            'limb_bending': ('uncertain', 'This is one static lifted pose, not a learned articulation sequence. Complete limb geometry is unverified.'),
            'body_shape': ('fail', 'Oblique views expose a partial depth surface with stretched torso and no verified hidden volume.'),
            'tearing': ('fail', 'Large dark disocclusion gaps and floating splats remain beside head/legs and behind the tail.'),
            'temporal_coherence': ('uncertain', 'Only sampled camera poses inspected, not continuous playback. No persistent identities across independently lifted source poses.')
        }
        for dimension, (status, observation) in judgments.items():
            review['dimensions'][dimension] = dict(status=status, observation=observation, evidence_frame_ids=evidence)
        review['limitations'] = ['7-second video is a camera orbit of one static pose, not animal motion.',
            'Front-view agreement is to the existing blurry Gaussian rendering at 416x240, not original Wan or real-video ground truth.',
            'Three-pose sheet inspects source frames 0,16,32 independently; it does not establish tracked motion.',
            'Fitted colours change per-asset appearance; no generalizable neural weights trained.',
            'Surface-normal variant does not visibly eliminate the principal missing-surface failure.']
        review['additional_evidence'] = dict(path=str(folder/'three_pose_comparison.jpg'),
            sha256=digest(folder/'three_pose_comparison.jpg'), observation='Reference-view matches at all three poses; translated camera reveals holes at each pose.')
        review['decision'] = decide_review(review, video)
        destination.write_text(json.dumps(review, indent=2))
        print(json.dumps(dict(variant=variant, decision=review['decision'])))


if __name__ == '__main__': main()
