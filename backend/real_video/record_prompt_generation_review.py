"""Record root's actual sampled-frame inspection of fresh orange-cat output."""
import json
from pathlib import Path
from .articulation_quality import create_review, decide_review

ROOT = Path('artifacts/real_video/prompt_generation/orange_cat_seed_531002')


def main():
    output = ROOT/'visual_review.json'
    if output.exists():
        raise FileExistsError('Preserve previous review')
    video = ROOT/'orange_cat_gaussian.mp4'
    indices = [0, 8, 16, 24, 32]
    review = create_review(video, [dict(path=ROOT/f'frame_{i:03d}.png', frame_index=i) for i in indices])
    review.update(claim_type='conditional_motion_generation',
        reviewer=dict(kind='agent', name='Root direct inspection of five rendered PNGs', observed_video=False),
        provenance=dict(future_conditioned=False, test_used_for_training=False, test_used_for_selection=False),
        classification='Fresh text/noise-conditioned planar Gaussian hybrid generation, NOT persistent 3D',
        prompt_satisfaction='Recognizable orange tabby cat on grass with changed leg poses',
        input_source='No source image/video exists for this fresh sample; comparison is to the text request, not the previous cat video.')
    observations = [
        'Recognizable full-body orange striped cat on green grass, tail curled upward. One foreleg is lifted. Face, fur and paw boundaries are visibly soft.',
        'The lifted foreleg reaches farther forward while the near foreleg remains near vertical. Body stays recognizable, but markings and grass are oversaturated and lack fine detail.',
        'The reaching foreleg extends downward/forward toward the grass; the other foreleg stays under the chest. Tail and torso remain recognizable; face detail is blurred.',
        'The extended foreleg is farther forward and downward, with the cat farther across the view. Body outline remains coherent in this sampled pose, while fine paws and hind-leg separation are indistinct.',
        'The cat remains recognizable with a forward foreleg and upright tail. Blur persists in the face, fur and lower limbs; this is not a high-detail final-quality result.'
    ]
    for item, observation in zip(review['frames'], observations):
        item['observation'] = observation
    evidence = [item['id'] for item in review['frames']]
    judgments = {
        'paw_placement': ('uncertain', 'Paw reach visibly changes, but blur and grass occlusion prevent certifying reliable contacts and all four feet.'),
        'limb_bending': ('uncertain', 'A foreleg progresses from lifted to extended in separated poses; blurred overlapping limbs prevent full anatomical/gait approval.'),
        'body_shape': ('pass', 'The five inspected poses retain a recognizable cat torso, head, ears and raised tail. This assesses these 2D views only, not hidden 3D shape.'),
        'tearing': ('pass', 'No obvious detached limb sheets or separated torso fragments in these five frames. Does not establish continuity of persistent physical Gaussian identities.'),
        'temporal_coherence': ('uncertain', 'Separated frames show changing pose rather than a single crop translation, but continuous playback was not inspected; temporal quality is not approved.')
    }
    for name, (status, observation) in judgments.items():
        review['dimensions'][name] = dict(status=status, observation=observation, evidence_frame_ids=evidence)
    review['detail_quality'] = dict(status='fail', observation='Blurry face/fur/paws and oversaturated patterned grass. Not production-quality photorealism.')
    review['limitations'] = [
        'Native 33-frame 16fps video, 2.0625 seconds. No repetition or seven-second extension.',
        'PNG evidence is rendered before MP4 encoding; independent replay verifies correspondence.',
        'No oblique views: this model only generates image-plane Gaussian fields, not 3D geometry.',
        'Wan latent denoising is unchanged; only the trained Gaussian output decoder replaces RGB decoding.',
        'Gaussian grid indices are not persistent tracked physical cat points; latest 3D acceleration weights are not used.',
        'No matched-quality compute or memory saving is claimed.'
    ]
    review['decision'] = decide_review(review, video)
    output.write_text(json.dumps(review, indent=2))
    print(json.dumps(review['decision']), flush=True)


if __name__ == '__main__':
    main()
