"""Persist sampled-frame reviews of the exact inspected Gaussian rollout v1.

Not an automatic reviewer. The observations are valid only for the hash-pinned
media actually inspected by /root/temporal_eval, including oblique renders.
Extract decoded evidence first, inspect it, and only then write these reviews.
"""
import argparse
import json
from pathlib import Path

import cv2

if __package__:
    from real_video.articulation_quality import create_review, decide_review
    from real_video.review_temporal_wan import digest, verify_decoded_samples
else:
    from articulation_quality import create_review, decide_review
    from review_temporal_wan import digest, verify_decoded_samples


FRAMES = (0, 16, 32, 48, 56)
EXPECTED = {
    'gaussian_condition.mp4': '4a184d315db81926026d3dfeb2774dec78a3e2f3d1795f3aab58b9dedab02b24',
    'wan.mp4': '190b964ba413f671c444fb8f98179ed455b0c77212f40b0bd7f64ec8a9faf49d',
    'gaussian_vs_wan.jpg': '7b97262be1b55029131806fe620ebe50010ebe1cde05879e3e9c69a0bc6dcdd9',
    'gaussian_oblique_000.png': '248969375d497750949a459edf87d044820089fce7efb0914a994a4c9f8f8278',
    'gaussian_oblique_016.png': '2907093f271fceb177a794e3d1357408727f3907c7e2db88c03d06c74c3353eb',
    'gaussian_oblique_032.png': 'f0928d7df9b3373657fd68ed0034066189b56a06e378833fad5f823da021d42e',
    'gaussian_oblique_048.png': '5a6aa5efa5af49f9a271e9ee5a16256eb915be080a7dc93ab21b5c793e80b88f',
    'gaussian_oblique_056.png': '63cf2650e0dffa71f4c5d37e3888688bf993650585166fe825dd9bd6dcc40cfc',
}
VIDEOS = {'gaussian': 'gaussian_condition.mp4', 'wan': 'wan.mp4'}


def verify_inputs(root):
    for name, expected in EXPECTED.items():
        if digest(root / name) != expected:
            raise ValueError(f'{name} differs from the inspected artifact; a new review is required.')


def evidence(root, prefix):
    return [dict(path=str(root / 'review_evidence' / f'{prefix}_{index:03d}.png'),
                 frame_index=index) for index in FRAMES]


def extract(root):
    root = Path(root).resolve(strict=True)
    verify_inputs(root)
    directory = root / 'review_evidence'
    directory.mkdir(exist_ok=False)
    for prefix, filename in VIDEOS.items():
        wanted = set(FRAMES)
        capture = cv2.VideoCapture(str(root / filename))
        index = 0
        try:
            while wanted:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Video ended before all required evidence frames.')
                if index in wanted:
                    if not cv2.imwrite(str(directory / f'{prefix}_{index:03d}.png'), frame):
                        raise OSError('Failed to save decoded evidence.')
                    wanted.remove(index)
                index += 1
        finally:
            capture.release()
    print(f'Decoded evidence saved to {directory}; visual inspection still required.')


GAUSSIAN_OBSERVATIONS = [
    'Frame 0: the right-facing orange striped Gaussian cat is fully visible, with an upright '
    'curled tail. Legs are short and tucked under the torso, with no supporting ground visible; '
    'the oblique view also shows compressed lower-limb forms.',
    'Frame 16: the forelimb on the visible side curls into a broad rounded shape under the chest; '
    'hind paws remain tucked. The corresponding oblique view exposes a thick folded forepaw, '
    'not a clearly jointed planted or swinging cat leg.',
    'Frame 32: the front paw contour changes into another rounded folded shape, while a hind '
    'paw shifts backward. The right-facing body and upright tail remain centered; the '
    'oblique view still shows shortened limbs under the chest and abdomen.',
    'Frame 48: a hind paw extends backward and the forepaw shape changes, demonstrating some '
    'local motion. Shortened, folded limb contours remain apparent in both views, without '
    'an interpretable grounded cat walking pose.',
    'Frame 56: a rounded tucked forelimb is again prominent, and the hind paws remain close '
    'to the body. The oblique view confirms the same compact, curled-limb failure rather '
    'than resolving a normal articulated gait.',
]
WAN_OBSERVATIONS = [
    'Frame 0: Wan shows a larger left-facing orange cat on grass with an extended foreleg. '
    'Its orientation, body proportions, facial appearance and markings differ from the '
    'right-facing compact Gaussian cat at the corresponding time.',
    'Frame 16: the left-facing Wan cat changes leg pose while remaining much larger in '
    'frame than the Gaussian reference. Its extended legs do not follow the Gaussian '
    'reference forepaw curl; matching orange coloration alone is not identity retention.',
    'Frame 32: the Wan cat has moved left and changes its foreleg arrangement. The Gaussian '
    'cat remains centered and right-facing in its root-following view, so the RGB subject '
    'does not reproduce the supplied geometry projection or articulated pose.',
    'Frame 48: the Wan cat reaches the left image boundary and its face is partly cropped. '
    'The Gaussian reference remains entirely inside frame. This is a visible failure '
    'of the requested scene-relative placement and full-subject retention.',
    'Frame 56: much of the head lies beyond the left boundary as the Wan cat continues '
    'leftward. The Gaussian cat remains centered, right-facing and fully visible. Leg '
    'movement is present, but identity, direction and trajectory fidelity are not achieved.',
]


def dimensions(prefix, ids):
    if prefix == 'gaussian':
        descriptions = {
            'paw_placement': ('fail',
                'Paws remain tightly tucked or curled under the body in the sampled poses, '
                'rather than exhibiting a convincing supporting/swinging arrangement. '
                'No ground surface is rendered, so actual contact or sliding cannot be verified.'),
            'limb_bending': ('fail',
                'Side and oblique views show rounded folded forelimbs and shortened lower '
                'legs, most clearly at frames 16, 32 and 56. Shape changes are visible, '
                'but they do not form anatomically convincing cat leg articulation.'),
            'body_shape': ('fail',
                'The orange striped torso and tail persist, but the compact torso-to-limb '
                'proportions and deformed paw contours are inadequate for a realistic '
                'walking cat. The oblique samples confirm this is not only side-view occlusion.'),
            'tearing': ('uncertain',
                'No clearly detached body fragments are visible in the five inspected '
                'poses. Distorted silhouettes and blurred limb surfaces remain; sparse '
                'frames cannot establish absence of tearing between samples.'),
            'temporal_coherence': ('uncertain',
                'The canonical torso, stripes and tail persist while paw contours change. '
                'No continuous playback was viewed; sampled poses do not establish smooth '
                'gait phases, stable contacts or coherent joint motion.'),
        }
    else:
        descriptions = {
            'paw_placement': ('uncertain',
                'Several recognisable walking-like foot placements appear on the lawn, '
                'but paws are soft and partially occluded. Sampled frames cannot verify '
                'stable contact or absence of sliding; placement does not reproduce the '
                'Gaussian reference poses.'),
            'limb_bending': ('uncertain',
                'Leg configurations change in the RGB samples; this is not a frozen '
                'image. Blurred/overlapping limbs and missing intermediate playback '
                'prevent a correct-articulation judgment, and the visible limb shapes '
                'do not follow the supplied Gaussian pose shapes.'),
            'body_shape': ('fail',
                'Wan replaces the compact right-facing Gaussian cat with a larger '
                'left-facing cat having different proportions, markings and facial '
                'appearance. Generic orange-cat similarity is not retention of the '
                'stored Gaussian subject geometry or identity.'),
            'tearing': ('uncertain',
                'No unambiguous detached body fragments appear in the inspected RGB '
                'frames, but soft overlapping legs and sparse sampling prevent a '
                'pass. Later face cropping is a framing failure, not evidence of tearing.'),
            'temporal_coherence': ('uncertain',
                'The RGB cat moves left and is partly outside the image by frames 48 '
                'and 56, while the Gaussian condition remains centered and right-facing. '
                'This proves condition-following failure in the samples, but smoothness '
                'of the intervening generated motion was not continuously viewed.'),
        }
    return {key: dict(status=status, observation=observation, evidence_frame_ids=ids)
            for key, (status, observation) in descriptions.items()}


def write_reviews(root):
    root = Path(root).resolve(strict=True)
    verify_inputs(root)
    targets = [root / f'{prefix}_visual_{suffix}.json'
               for prefix in VIDEOS for suffix in ('review', 'decision')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Preserve previous reviews; do not overwrite them.')
    records = {}
    for prefix, filename in VIDEOS.items():
        video = root / filename
        frames = evidence(root, prefix)
        verify_decoded_samples(video, frames)
        review = create_review(video, frames)
        review['claim_type'] = 'conditional_motion_generation'
        review['reviewer'] = dict(kind='agent', name='/root/temporal_eval', observed_video=False)
        review['provenance'] = dict(future_conditioned=False, test_used_for_training=False,
                                   test_used_for_selection=False)
        review['provenance_scope'] = (
            'New pretrained neural controller predictions after a static initial state; '
            'no recorded source/test video future or saved motion clip supplied. Wan receives '
            'the complete newly generated Gaussian-condition sequence in advance; this is '
            'not online causal Wan generation, even though future recorded-video conditioning is false.')
        review['scope'] = (
            'New neural locomotion retargeted to persistent 3D Gaussian state; anatomical quality rejected.'
            if prefix == 'gaussian' else
            'Wan RGB generation conditioned on newly predicted Gaussian motion; Wan does not emit or update Gaussian state.')
        observations = GAUSSIAN_OBSERVATIONS if prefix == 'gaussian' else WAN_OBSERVATIONS
        for frame, observation in zip(review['frames'], observations):
            frame['observation'] = observation
        ids = [frame['id'] for frame in review['frames']]
        review['dimensions'] = dimensions(prefix, ids)
        review['comparison_evidence'] = [
            dict(path=str(root / name), sha256=expected)
            for name, expected in EXPECTED.items() if name != filename
        ]
        review['oblique_view_scope'] = (
            'The five inspected oblique images render the same generated Gaussian states '
            'from another camera. Wan has no supplied oblique RGB outputs; its 3D consistency is unverified.')
        review['numeric_metrics'] = dict(frame_count=57, fps=8,
            encoded_playback_duration_seconds=7.125, sample_time_span_seconds=7.0)
        review['additional_checks'] = (
            dict(realistic_cat_articulation='fail', ground_contact='unverified',
                 persistent_geometry_runtime='implemented_not_quality_proof')
            if prefix == 'gaussian' else
            dict(gaussian_identity_retention='fail', gaussian_pose_following='fail',
                 scene_relative_size_and_orientation='fail', full_subject_retention='fail',
                 wan_writes_gaussian_state=False))
        review['limitations'] = [
            'Only five separated frames and comparison/oblique images were visually inspected, not continuous playback.',
            'The external controller is a dog locomotion prior with an approximate manually inferred cat retarget.',
            'New controller predictions do not establish newly trained, generalizable cat-specific motion weights.',
            'Gaussian projected features condition Wan, but Wan outputs an RGB video latent, not Gaussian memory updates.',
            'No quality-matched compute or memory savings are established by these rejected videos.',
        ]
        decision = decide_review(review, video)
        if decision['accepted']:
            raise AssertionError('These explicitly rejected artifacts must not pass.')
        decision.update(promoted=False, scope=review['scope'])
        records[prefix] = (review, decision)
    for prefix, (review, decision) in records.items():
        (root / f'{prefix}_visual_review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
        (root / f'{prefix}_visual_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    return {prefix: record[1] for prefix, record in records.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/real_video/wan_temporal_memory/gaussian_rollout_v1')
    parser.add_argument('--phase', choices=('extract', 'review'), required=True)
    args = parser.parse_args()
    if args.phase == 'extract':
        extract(args.root)
    else:
        print(json.dumps(write_reviews(args.root), indent=2))


if __name__ == '__main__':
    main()
