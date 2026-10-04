"""Record the inspected v3 background candidate; reject, never auto-promote."""
import json
from pathlib import Path
import torch
from .checkpoint_io import load_verified
from .articulation_quality import create_review, decide_review


def main():
    folder = Path('artifacts/real_video/background_completion/v3_texture')
    destination = folder/'visual_review.json'
    if destination.exists():
        raise FileExistsError('Preserve existing review')
    original = load_verified(Path('artifacts/real_video/detail_lift/v4_tangent/frame_000_opacity_0.95.pt'))
    scene = load_verified(folder/'completed_scene.pt')
    n = scene['original_count']
    audit = {key: torch.equal(original[key], scene[key][:n])
             for key in ['position', 'covariance', 'colour']}
    audit['opacity'] = torch.equal(scene['opacity'][:n], torch.full_like(scene['opacity'][:n], original['opacity']))
    audit['all_finite'] = all(torch.isfinite(scene[k]).all().item() for k in ['position', 'covariance', 'colour', 'opacity'])
    audit['positive_covariance'] = bool((torch.linalg.eigvalsh(scene['covariance']) > 0).all())
    assert all(audit.values()), audit
    video = folder/'background_completion_7seconds.mp4'
    review = create_review(video, [dict(path=folder/f'inspection_{i:03d}.png', frame_index=i) for i in [0, 28, 84]])
    review.update(claim_type='reconstruction', reviewer=dict(kind='agent', name='Root sampled-frame inspection', observed_video=False),
                  provenance=dict(future_conditioned=False, test_used_for_training=False, test_used_for_selection=False))
    observations = [
        'Reference view: original cat and background closely retained, including existing blur.',
        'Positive oblique view: large hole by head partly filled with green background; pinholes and boundary distortions remain.',
        'Negative oblique view: some missing background filled, but dark gaps, partial scene boundaries and distorted cat geometry remain.'
    ]
    for frame, observation in zip(review['frames'], observations):
        frame['observation'] = observation
    evidence = [f['id'] for f in review['frames']]
    judgments = {
        'paw_placement': ('uncertain', 'Blur and incomplete oblique geometry prevent verification of ground contact.'),
        'limb_bending': ('uncertain', 'Static cat; no articulation sequence evaluated.'),
        'body_shape': ('fail', 'Oblique cat remains an incomplete depth surface, not a complete animal volume.'),
        'tearing': ('fail', 'Background filling reduces some gaps but pinholes and discontinuities remain.'),
        'temporal_coherence': ('uncertain', 'Three camera poses inspected, not continuous playback; no generated cat motion.')
    }
    for key, (status, observation) in judgments.items():
        review['dimensions'][key] = dict(status=status, observation=observation, evidence_frame_ids=evidence)
    review['checkpoint_audit'] = audit
    review['background_semantics'] = dict(status='fail', observation='Direct inspection of completed_background.png shows an orange leg-like remnant and an invented bush-like mass; not an approved empty lawn.')
    review['limitations'] = ['Invented background, not recovered ground truth.', 'Camera-only seven-second orbit of one static pose.', 'No Wan weight changes or demonstrated generation speedup.', 'Hole counts include scene-border gaps, not only cat-shaped gaps.']
    review['decision'] = decide_review(review, video)
    destination.write_text(json.dumps(review, indent=2))
    print(json.dumps(dict(audit=audit, decision=review['decision'])))


if __name__ == '__main__':
    main()
