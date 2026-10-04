"""Persist the agent's actual sampled-frame review of temporal Wan v2.

This is NOT an automatic reviewer or a reusable source of visual judgments.
The observations below apply only to the hash-pinned media inspected in this
session. A different candidate requires a fresh visual inspection and review.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from real_video.articulation_quality import create_review, decide_review
else:
    from articulation_quality import create_review, decide_review


EXPECTED = {
    'tracked.mp4': 'e5cea089430861624faa1f2e119e3306e25897516dc6aee22d856b2e34439081',
    'source.mp4': '0a07d241470c0849fa112c749d61b71a5a481162c3b70763f3a1fae8fcbdff5c',
    'static.mp4': '392f9182fc72d8aa7e4ce4ef692265874784219e401da03a89d3cbd464e7c2e3',
    'comparison.jpg': '9d459935e181baad14a8edd1120201f2d772e57321c1c9324dc3abbb0a0a05fc',
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_decoded_samples(video, evidence):
    """Check sample pixels against decoding the exact candidate, on CPU."""
    desired = {item['frame_index']: item for item in evidence}
    capture = cv2.VideoCapture(str(video))
    index = 0
    try:
        while desired:
            ok, decoded_bgr = capture.read()
            if not ok:
                raise ValueError('Video ended before the evidence frame.')
            if index in desired:
                recorded_bgr = cv2.imread(desired[index]['path'], cv2.IMREAD_COLOR)
                if not np.array_equal(recorded_bgr, decoded_bgr):
                    raise ValueError(f'Sample {index} is not the exact decoded video frame.')
                desired.pop(index)
            index += 1
    finally:
        capture.release()


def paired_difference(first, second):
    captures = [cv2.VideoCapture(str(path)) for path in (first, second)]
    differences = []
    try:
        while True:
            decoded = [capture.read() for capture in captures]
            if not any(item[0] for item in decoded):
                break
            if not all(item[0] for item in decoded):
                raise ValueError('Paired videos have different decoded lengths.')
            a, b = (item[1] for item in decoded)
            if a.shape != b.shape:
                raise ValueError('Paired video shapes differ.')
            differences.append(float(np.abs(a.astype(np.float32) - b).mean() / 255))
    finally:
        for capture in captures:
            capture.release()
    return dict(decoded_frame_count=len(differences),
                mean_rgb_absolute_difference_0_to_1=float(np.mean(differences)),
                per_frame_rgb_absolute_difference_0_to_1=differences,
                interpretation='Same-seed static/tracked pixel difference only; not a motion-following or quality score.')


def write_review(root):
    root = Path(root).resolve(strict=True)
    review_path, decision_path = root / 'visual_review.json', root / 'visual_decision.json'
    if review_path.exists() or decision_path.exists():
        raise FileExistsError('Review already exists; preserve it and review new candidates separately.')
    for name, expected_hash in EXPECTED.items():
        if digest(root / name) != expected_hash:
            raise ValueError(f'{name} differs from the actually inspected artifact; do not reuse observations.')
    frames = [dict(path=str(root / 'audit' / f'output_{i:06d}.png'), frame_index=i)
              for i in (0, 8, 16)]
    source_frames = [dict(path=str(root / 'audit' / f'source_{i:06d}.png'), frame_index=i)
                     for i in (0, 8, 16)]
    verify_decoded_samples(root / 'tracked.mp4', frames)
    verify_decoded_samples(root / 'source.mp4', source_frames)
    review = create_review(root / 'tracked.mp4', frames)
    review['claim_type'] = 'conditional_motion_generation'
    review['scope'] = 'Recorded-motion-conditioned Wan RGB generation, not novel Gaussian motion generation.'
    review['claim_type_note'] = (
        'The gate uses conditional_motion_generation as its generation label. Future recorded '
        'motion conditioning is explicitly true, so this cannot support a novel-motion claim.')
    review['reviewer'] = dict(kind='agent', name='/root/temporal_eval', observed_video=False)
    review['provenance'] = dict(future_conditioned=True, test_used_for_training=False,
                                 test_used_for_selection=False)
    review['validation_context'] = dict(sequence='cows', split='validation',
        training_sequences=['cat-girl', 'dog-agility', 'elephant', 'sheep'],
        explanation='Source and training split names checked in this experiment report; no final test set is used here.')
    observations = [
        'Frame 0: a small dark left-facing ungulate holds one foreleg folded upward on a bright lawn. '
        'The source is a much larger right-facing brown-and-white cow beside a muddy fenced path; '
        'its appearance, orientation, scale and environment are not preserved.',
        'Frame 8: the dark animal has separated forelegs and a rear leg angled forward, different '
        'from frame 0. The source cow remains right-facing and advances on the muddy path. '
        'The generated animal still lacks the reference markings and silhouette.',
        'Frame 16: the generated animal occupies a different position with changed leg pose, '
        'but remains dark and left-facing on a lawn. The white-and-brown right-facing source '
        'identity is still absent. Fine hoof contact and joint structure are not resolved.',
    ]
    for frame, observation in zip(review['frames'], observations):
        frame['observation'] = observation
    ids = [frame['id'] for frame in review['frames']]
    descriptions = {
        'paw_placement': ('uncertain',
            'Hoof endpoints and ground contact are poorly resolved at this scale. Frame 0 has '
            'one tucked-up foreleg; frames 8 and 16 show different placements. Sparse frames '
            'cannot establish stable contact or exclude foot sliding.'),
        'limb_bending': ('uncertain',
            'Leg poses do change, so the output is not literally frozen. The short sampled '
            'sequence does not establish anatomically correct articulation, and its poses '
            'are visually nearly the same as the static-conditioning ablation.'),
        'body_shape': ('fail',
            'The generated dark, slim left-facing animal does not preserve the source cow: '
            'the broad brown-and-white torso, white face, orientation and scene-relative '
            'size are all substantially different across all three samples.'),
        'tearing': ('uncertain',
            'No obvious detached body fragments appear in these three sampled frames, '
            'but fine limbs are blurred and intermediate frames were not continuously '
            'viewed. This is insufficient evidence to pass tearing.'),
        'temporal_coherence': ('uncertain',
            'Coarse dark-body appearance persists across the samples and leg poses vary. '
            'Continuous playback was not viewed, and static versus tracked samples are '
            'nearly indistinguishable; motion control and smoothness are not established.'),
    }
    for key, (status, observation) in descriptions.items():
        review['dimensions'][key] = dict(status=status, observation=observation,
                                         evidence_frame_ids=ids)
    review['comparison_evidence'] = {
        name: dict(path=str((root / name).resolve()), sha256=expected)
        for name, expected in EXPECTED.items() if name != 'tracked.mp4'
    }
    review['source_sample_evidence'] = [dict(**item, sha256=digest(item['path'])) for item in source_frames]
    review['numeric_metrics'] = {
        'static_vs_tracked': paired_difference(root / 'static.mp4', root / 'tracked.mp4'),
        'output_fps': 12, 'decoded_frame_count': 17, 'duration_seconds': 17 / 12,
    }
    review['limitations'] = [
        'Only sampled frames and comparison sheets were visually inspected; no continuous-playback attestation.',
        'The 1.417-second diagnostic is too short to establish a robust gait cycle or persistent memory.',
        'Recorded future trajectories condition RGB video generation; this is not newly predicted Gaussian motion.',
        'No oblique 3D view or 3D-ground-truth anatomy is supplied; no 3D-quality claim is made.',
        'A lower denoising loss or nonzero pixel difference does not establish identity retention or motion control.',
        'This review rejects the artifact and does not promote a checkpoint or claim compute savings.',
    ]
    decision = decide_review(review, root / 'tracked.mp4')
    if decision['accepted']:
        raise AssertionError('This explicitly rejected candidate must not be accepted.')
    decision.update(promoted=False, scope=review['scope'])
    review_path.write_text(json.dumps(review, indent=2, allow_nan=False), encoding='utf-8')
    decision_path.write_text(json.dumps(decision, indent=2, allow_nan=False), encoding='utf-8')
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/real_video/wan_temporal_memory/v2')
    args = parser.parse_args()
    print(json.dumps(write_review(args.root), indent=2))


if __name__ == '__main__':
    main()
