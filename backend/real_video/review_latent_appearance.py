"""Candidate-specific review of the exact reader-v2 development ablation.

Not an automatic visual judge. Extraction and numerical paired comparisons
do not accept a candidate; observations are added only after direct viewing.
No GPU, neural inference, sharpening or output compositing is performed.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .articulation_quality import create_review, decide_review
from .checkpoint_io import digest
from .review_temporal_wan import verify_decoded_samples


FRAMES = (0, 8, 16)
EXPECTED = {
    'source': 'afd47b7f9ec016294ef37fcfce9d6f4d2c7ede422d62dff342e23bcecc6f4b14',
    'v3_reader_disabled': '01d0596c3459802fa5068b594ad008d28e02074f799fa51969c188b751f42c0d',
    'memory': 'cc2c2ebc0d1bed9a7cac16a576dfea419c9460ae84daec18f010fddc66b1943d',
    'flipped_features': 'bc18cd56bd120a34f763e517af528ba83ea5fd7c97330fdcbcbe715dbf1640da',
}


def verify(root):
    for name, expected in EXPECTED.items():
        if digest(root/f'{name}.mp4') != expected:
            raise ValueError(f'{name}: artifact changed; new inspection required')


def decoded(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != 17 or fps != 12:
        raise ValueError(f'Unexpected artifact clock {len(frames)} frames at {fps} fps')
    result = np.stack(frames)
    if result.shape != (17, 256, 448, 3):
        raise ValueError(f'Unexpected decoded dimensions: {result.shape}')
    return result


def paired(first, second):
    difference = (first.astype(np.float64)-second.astype(np.float64))/255.
    mse = float(np.square(difference).mean())
    return dict(decoded_frames=17, mean_absolute_rgb_difference_0_to_1=float(np.abs(difference).mean()),
        rms_rgb_difference_0_to_1=math.sqrt(mse), mse_0_to_1=mse,
        psnr_db=None if mse == 0 else -10*math.log10(mse),
        identical_decoded_frames=int(np.all(first == second, axis=(1, 2, 3)).sum()),
        per_frame_mean_absolute_rgb_difference_0_to_1=np.abs(difference).mean((1, 2, 3)).tolist())


def extract(root):
    verify(root)
    directory = root/'review_evidence'
    directory.mkdir(exist_ok=False)
    arrays = {name: decoded(root/f'{name}.mp4') for name in EXPECTED}
    sheet = Image.new('RGB', (448*3, 280*4), '#20252b')
    for row, (name, frames) in enumerate(arrays.items()):
        for col, index in enumerate(FRAMES):
            frame = Image.fromarray(frames[index])
            frame.save(directory/f'{name}_{index:03d}.png')
            sheet.paste(frame, (col*448, row*280+24))
            ImageDraw.Draw(sheet).text((col*448+8, row*280+5), f'{name} | decoded frame {index}', fill='white')
    sheet.save(directory/'decoded_comparison.png')
    metrics = dict(
        scope='All-frame decoded same-seed paired pixel differences; not perceptual quality, 3D consistency or ground-truth motion accuracy',
        videos={name: dict(path=str(root/f'{name}.mp4'), sha256=checksum) for name, checksum in EXPECTED.items()},
        memory_vs_flipped=paired(arrays['memory'], arrays['flipped_features']),
        memory_vs_v3=paired(arrays['memory'], arrays['v3_reader_disabled']),
        source_comparisons={name: paired(arrays['source'], arrays[name]) for name in EXPECTED if name != 'source'},
        continuous_playback_inspected=False,
        quality_accepted=False,
        warning='Comparisons use different generated subjects and scenes; source pixel errors do not isolate detail or validate identity. A low memory-versus-flipped difference alone does not prove that all memory is ignored.')
    (directory/'decoded_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


OBSERVATIONS = {
    'v3_reader_disabled': [
        'Decoded frame 0: a nearly solid brown animal is viewed from its rear-right side, facing left across a vivid green field. The source is a broadside white-and-brown pied cow facing right on a gray dirt farm path; coat, orientation, proportions and surroundings do not match.',
        'Decoded frame 8: the brown animal remains rear-oblique and left-facing. Tall narrow lower limbs and the dark rear-facing body differ from the source cow\'s broadside stance, white lower legs and patch pattern; green field replaces the source path and fence.',
        'Decoded frame 16: the brown animal turns its head slightly while its rump remains prominent. The source cow is still broadside and right-facing. No source-specific coat, farm layout or corresponding hoof configuration is retained.',
    ],
    'memory': [
        'Decoded frame 0: a solid brown animal with an elongated dark muzzle and a long forehead/head protrusion faces left and away. Its coat, head geometry, view direction and vivid green surroundings fail to reproduce the supplied pied cow on a farm path.',
        'Decoded frame 8: the brown animal lowers its head and the long curved protrusion remains conspicuous. Source frame 8 shows the pied cow broadside, facing right with a level head; the generated posture and identity do not follow that source.',
        'Decoded frame 16: the generated head is lowered toward the grass and an elongated curved projection extends left with a dark tip. This is absent from the reference cow. The body remains solid brown in the replacement green landscape; leg and head configurations are not the recorded cow pose.',
    ],
    'flipped_features': [
        'Decoded frame 0: despite horizontal flipping of the supplied appearance features, the brown rear-oblique animal, elongated dark muzzle and tall head projection look very similar to the unflipped-memory result. Neither result matches the source pied cow or farm setting.',
        'Decoded frame 8: a near-matching brown animal again lowers its head, with the same conspicuous curved projection. The white-and-brown markings and right-facing source identity remain absent, as in the unflipped-memory candidate.',
        'Decoded frame 16: the lowered head, curved leftward projection with dark tip, brown rump and vivid green field closely resemble the unflipped-memory output. Fine pixel differences exist, but the major incorrect subject and action persist after the negative-control feature flip.',
    ],
}


def review(root):
    verify(root)
    directory = root/'review_evidence'
    source_evidence = [dict(path=str(directory/f'source_{index:03d}.png'), frame_index=index) for index in FRAMES]
    verify_decoded_samples(root/'source.mp4', source_evidence)
    destinations = [root/f'{name}_visual_{suffix}.json'
        for name in OBSERVATIONS for suffix in ('review', 'decision')]
    if any(path.exists() for path in destinations):
        raise FileExistsError('Preserve previous reviews; never overwrite visual decisions')
    metrics = json.loads((directory/'decoded_metrics.json').read_text(encoding='utf-8'))
    results = {}
    for name, observations in OBSERVATIONS.items():
        video = root/f'{name}.mp4'
        frames = [dict(path=str(directory/f'{name}_{index:03d}.png'), frame_index=index) for index in FRAMES]
        verify_decoded_samples(video, frames)
        item = create_review(video, frames)
        item['claim_type'] = 'reconstruction'
        item['reviewer'] = dict(kind='agent', name='/root/detail_visual_audit', observed_video=False)
        item['scope'] = 'Recorded-future-track-guided development motion transfer. RGB synthesis, not autonomous Gaussian motion or 3D geometry generation.'
        item['provenance'] = dict(future_conditioned=True, test_used_for_training=False, test_used_for_selection=False)
        item['provenance_scope'] = 'DAVIS cows is the existing repeatedly inspected development validation sequence, not a fresh final test. The recorded future 2D tracks are supplied; only the first RGB frame contributes appearance memory.'
        for frame, observation in zip(item['frames'], observations):
            frame['observation'] = observation
        ids = [frame['id'] for frame in item['frames']]
        observations_by_dimension = {
            'paw_placement': ('fail', 'The depicted hoof/leg arrangement belongs to a left-facing rear-oblique replacement animal, not the source cow\'s broadside rightward walking poses. This is a source-pose fidelity failure; contact stability and sliding cannot be judged from three still samples.'),
            'limb_bending': ('uncertain', 'Some leg configurations differ across the three decoded samples, but continuous playback was not inspected. Lower-leg overlap and lack of matched source articulation prevent a correct-anatomy or gait-phase judgment.'),
            'body_shape': ('fail', 'The broad white-and-brown pied cow is replaced by a solid brown rear-oblique animal with different head, limb and torso proportions.' + (' The added long curved head projection and head-down action are absent from the source.' if name != 'v3_reader_disabled' else ' Its broadside source pose and distinctive markings are absent.')),
            'tearing': ('uncertain', 'No unequivocally detached fragments are visible in these three samples, but elongated head structures and overlapping legs prevent an anatomical pass. Sparse frames do not establish absence of tearing between samples.'),
            'temporal_coherence': ('uncertain', 'Only decoded frames 0, 8 and 16 were visually inspected, not continuous playback. The source-pose and identity mismatch is clear; intervening smoothness, stable contacts and fine-detail persistence remain unverified.'),
        }
        item['dimensions'] = {key: dict(status=status, observation=observation, evidence_frame_ids=ids)
            for key, (status, observation) in observations_by_dimension.items()}
        item['comparison_evidence'] = [dict(path=entry['path'], frame_index=entry['frame_index'],
            sha256=digest(entry['path']), source_video_sha256=EXPECTED['source']) for entry in source_evidence]
        item['numeric_metrics'] = dict(decoded_frames=17, fps=12,
            encoded_duration_seconds=17/12, sample_time_span_seconds=16/12,
            source_paired_pixels=metrics['source_comparisons'][name],
            memory_vs_flipped_pixels=metrics['memory_vs_flipped'])
        item['additional_checks'] = dict(source_identity_retention='fail', source_coat_markings='fail',
            source_background_layout='fail', source_view_direction='fail',
            fine_detail_retention='not demonstrated', wan_writes_gaussian_state=False,
            new_reader_changes_rgb_output=True,
            negative_control_interpretation='Feature flipping changes decoded pixels, but major incorrect subject and action remain visually near-matching; correctly aligned appearance use is not established.')
        item['limitations'] = [
            'This is a 17-frame, 12-fps, 1.4167-second diagnostic, not a seven-second generated demonstration.',
            'Three temporally separated frames were decoded from each exact MP4 and inspected; continuous playback was not viewed.',
            'No oblique camera render of a shared 3D state exists for these RGB outputs; 3D consistency is unverified.',
            'The disabled-reader baseline still includes the earlier trained v3 LoRA/control adapter; it is not original Wan.',
            'Paired RGB differences and latent losses measure numerical changes, not faithful geometry, animal identity or fur detail.',
            'The negative-control comparison does not prove the entire memory branch is ignored; it fails to demonstrate useful spatially aligned identity recall in this pilot.',
            'No quality-matched compute saving is supported by these rejected candidates.',
        ]
        decision = decide_review(item, video)
        if decision['accepted'] or decision['status'] != 'rejected':
            raise AssertionError('Explicitly failed identity/body review must reject this exact candidate')
        decision.update(promoted=False, continuous_playback_inspected=False)
        results[name] = (item, decision)
    for name, (item, decision) in results.items():
        (root/f'{name}_visual_review.json').write_text(json.dumps(item, indent=2), encoding='utf-8')
        (root/f'{name}_visual_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print(json.dumps({name: decision for name, (_, decision) in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/real_video/detail_memory/reader_v2'))
    parser.add_argument('--phase', choices=('extract', 'review'), required=True)
    args = parser.parse_args()
    {'extract': extract, 'review': review}[args.phase](args.root.resolve(strict=True))
