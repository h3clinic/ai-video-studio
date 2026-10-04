"""Persist observations made by the root agent after directly viewing renders.

This is not an automatic visual metric. Never reuse these observations for a
different experiment. Original metrics and failed artifacts are preserved.
"""
import json
from pathlib import Path
from .articulation_quality import create_review, decide_review
from .checkpoint_io import digest

ROOT = Path('artifacts/real_video/true3d/acceleration_hypothesis/v1/joint_trial')


def main():
    output = ROOT / 'visual_review.json'
    if output.exists():
        raise FileExistsError('Preserve completed visual review')
    frames = [4, 10, 14, 22, 32]
    video = ROOT / 'acceleration_native.mp4'
    review = create_review(video, [dict(path=ROOT / f'comparison_{f:02d}.jpg',
                                       frame_index=f-4) for f in frames])
    review['candidate'] = 'Top-right learned 3D acceleration panel, source top-left.'
    review['reviewer'] = dict(kind='agent', name='Root agent direct image inspection',
                              observed_video=False)
    review['claim_type'] = 'reconstruction'
    review['provenance'] = dict(future_conditioned=True, test_used_for_training=False,
                                test_used_for_selection=False)
    observations = [
        'Initial shared seed already has a short stump-like tail, shortened rear limbs, a bulky torso and incorrect foot shapes relative to the source.',
        'The source has a clearly lifted forepaw; the candidate has a short dangling far foreleg and incorrectly shaped rear feet, with a thin trailing appendage below the rear leg.',
        'The near foreleg points almost straight down rather than following the source lifted bent paw. Rear feet remain short and malformed; a faint thin appendage extends below them.',
        'The source forward paw reach is missing. A narrow sheet connects the chest to a hanging foreleg; rear limbs merge into a distorted dangling shape.',
        'The near foreleg is bent backward instead of matching the source forward extension. Another forelimb is a thin stretched sheet, and a long thin trailing strip extends behind the rear leg.'
    ]
    for item, observation in zip(review['frames'], observations):
        item['observation'] = observation
    evidence = [item['id'] for item in review['frames']]
    dimensions = {
        'paw_placement': ('fail', 'Lifted and extended reference paws are not matched in the rendered body despite lower projected-joint error.'),
        'limb_bending': ('fail', 'Malformed short hind limbs and incorrect foreleg bend remain; later poses stretch skin instead of producing anatomical articulation.'),
        'body_shape': ('fail', 'Bulky inferred torso, truncated tail and incorrect limb proportions are visible from the source camera and oblique views.'),
        'tearing': ('fail', 'Thin translucent strips/sheets extend from rear and front limbs in later poses. A shared mesh does not prevent this visible surface distortion.'),
        'temporal_coherence': ('uncertain', 'Separated frames confirm changed poses, but continuous playback was not inspected. No natural gait, fine-motion or temporal smoothness approval is given.')
    }
    for key, (status, observation) in dimensions.items():
        review['dimensions'][key] = dict(status=status, observation=observation,
                                         evidence_frame_ids=evidence)
    review['oblique_evidence'] = []
    for f, observation in [
        (14, 'Oblique render confirms bulky torso, stump-like tail, malformed rear feet and a nearly vertical front leg.'),
        (32, 'Oblique render shows a long thin trailing rear strip and a sheet-like stretched foreleg. The head is partially outside this diagnostic camera crop.')
    ]:
        path = ROOT / f'oblique_{f:02d}.png'
        review['oblique_evidence'].append(dict(path=str(path.resolve()),
            sha256=digest(path), source_frame=f, observation=observation))
    review['numeric_metrics'] = json.loads((ROOT / 'evaluation.json').read_text())['metrics']
    review['limitations'] = [
        'Inspected five rendered poses and two oblique images, not continuous video playback.',
        'JPGs are pre-encode comparison images; native video index equals source frame minus four.',
        'The seed, rig and training targets are full-clip conditioned. This is not a future-blind forecast or a new-video generator.',
        'Independent per-Gaussian physical micro-motion was not measured; all dot positions derive from the shared surface.',
        'The native clip contains 29 states at 16fps (1.8125s). No seven-second generation is claimed.',
        'Fixed-size current dynamics state excludes the asset, model, rendered frames, saved trajectories and training autograd. No quality-matched efficiency claim.'
    ]
    review['decision'] = decide_review(review, video)
    assert review['decision']['status'] == 'rejected'
    output.write_text(json.dumps(review, indent=2))
    selection = dict(accepted_model=None, quality_achieved=False,
        visual_review=str(output), model_sha256=digest(ROOT / 'model.pt'),
        video_sha256=digest(video), compute_savings_claim=False,
        reason='Changing acceleration does not repair the visually rejected rest anatomy and surface binding. Numerical improvements do not override direct rejection.')
    selection_path = ROOT / 'selection.json'
    if selection_path.exists():
        raise FileExistsError('Preserve selection')
    selection_path.write_text(json.dumps(selection, indent=2))
    print(json.dumps(review['decision']), flush=True)


if __name__ == '__main__':
    main()
