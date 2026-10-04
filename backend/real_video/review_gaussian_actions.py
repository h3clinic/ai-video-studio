"""Persist manually inspected, rejected Gaussian action candidates v1 and v3.

This is not an automatic quality evaluator. Extract exact decoded MP4 evidence,
inspect it, then run review. Observations are pinned to the inspected artifacts.
"""
import argparse
import json
from pathlib import Path

import cv2

from real_video.articulation_quality import create_review, decide_review
from real_video.review_temporal_wan import digest, verify_decoded_samples


FRAMES = (0, 15, 30, 45, 60, 75, 104)
EXPECTED = {
    'gaussian_actions_7s.mp4': '2a324e9ffe5368c0ef1a42005bd80b75cce2db94add15b69f9e17d42612f8a28',
    'scene_sheet.jpg': '5f6a951deeb49a6a8abbb83b2cccbc0716779efdd34466f6daf742e19e7907e2',
    'oblique_sheet.jpg': 'f7fc03020538b6ef950b6416e2218777a16dfa7e04a41153a2efc489a026f675',
}
EXPECTED_V3 = {
    'gaussian_actions_7s.mp4': 'c50b9a33cc018a6727c3171d6609c298bd7f50f5371d596732eca6aedd17259f',
    'scene_sheet.jpg': '283da6a671166daf90e73867befdc20601f1f20844ddc9884d79935cfd1d628f',
    'oblique_sheet.jpg': 'cb6c6ea8ecadb0eaf5fa8c3a44eacd699b0e3f37a8006d538038d61a20969825',
}
SOURCE = Path('artifacts/real_video/gaussian_dog_insertion/v1/dog/appearance_six_views.jpg')
SOURCE_HASH = '1acd8e989746dcf92e105b24d10b14402912e041d38dd0359dc43e8c488c693b'
PREVIEW_V2_HASHES = {
    'scene_sheet.jpg': 'ad40710b1170a439e70e5bf1973db5967e5bfff4b58803237df21fe560290146',
    'oblique_sheet.jpg': '3a34c8a7cc20fc226c924bbe4ec66cad7e1fb1010faa86598c536ebd9cc4ae60',
    'preview.json': 'a8e5cee5f1ef07f2e8afe2896f41b29295a7573c7715982611ce86810ce52a4b',
}


def review_preview_v2(root):
    for name, expected in PREVIEW_V2_HASHES.items():
        if digest(root / name) != expected:
            raise ValueError(f'{name} is not the inspected v2 preview.')
    target = root / 'preview_visual_rejection.json'
    if target.exists():
        raise FileExistsError('Preserve the existing preview review.')
    record = dict(
        schema_version=1, artifact_type='preview_only_no_video',
        reviewer='/root/dog_visual_review', observed_continuous_playback=False,
        preview_only=True, quality_accepted=False, promoted=False, status='rejected',
        sampled_pose_frames=list(FRAMES),
        evidence=[dict(path=str((root / name).resolve()), sha256=expected)
                  for name, expected in PREVIEW_V2_HASHES.items()],
        classification='Pretrained conditional skeletal predictions applied to stored '
                       'Gaussian animals; not Wan-native generation.',
        observations=[
            'The oblique dog views have a conspicuous swept, twisted neck-to-chest fold '
            'across the inspected poses; the head sits unnaturally low and forward relative '
            'to the bulging front chest and torso.',
            'Dog legs are shortened, folded and poorly separated, with dangling feet and '
            'no convincing supporting contact configuration.',
            'The cat retains the same curled, rounded forepaws and compressed lower limbs '
            'as candidate v1.',
            'Both subjects translate through the scene in separated previews. The source '
            'dog identity remains recognizable, but this does not rescue distorted anatomy.',
            'Later poses are similar in position; sparse preview images cannot validate '
            'smooth gait, temporal coherence, or absence of tearing between sampled times.',
        ],
        dimensions=dict(paw_placement='fail', limb_bending='fail', body_shape='fail',
                        tearing='uncertain', temporal_coherence='uncertain'),
        limitations=[
            'This is a preview-image rejection, not an articulation_quality video review.',
            'No full v2 video was generated or inspected, and no exact-MP4-frame provenance is claimed.',
            'Source assets already contain malformed anatomy and texture flaws.',
        ])
    target.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(str(target))


def verify_inputs(root):
    if root.name not in ('v1', 'v3'):
        raise ValueError('Only manually inspected v1 and v3 videos are supported.')
    expected_hashes = EXPECTED if root.name == 'v1' else EXPECTED_V3
    for name, expected in expected_hashes.items():
        if digest(root / name) != expected:
            raise ValueError(f'{name} is not the visually inspected {root.name} artifact.')
    if digest(SOURCE) != SOURCE_HASH:
        raise ValueError('The inspected source dog appearance sheet has changed.')


def evidence(root):
    return [dict(path=str(root / 'review' / f'decoded_{index:03d}.png'), frame_index=index)
            for index in FRAMES]


def extract(root):
    verify_inputs(root)
    directory = root / 'review'
    directory.mkdir(exist_ok=False)
    capture = cv2.VideoCapture(str(root / 'gaussian_actions_7s.mp4'))
    wanted = set(FRAMES)
    index = 0
    images = []
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Video ended before all evidence frames were decoded.')
            if index in wanted:
                if not cv2.imwrite(str(directory / f'decoded_{index:03d}.png'), frame):
                    raise OSError('Could not save decoded frame.')
                images.append(frame)
                wanted.remove(index)
            index += 1
    finally:
        capture.release()
    cv2.imwrite(str(directory / 'decoded_sheet.jpg'), cv2.vconcat(images))
    print(f'Exact decoded evidence saved under {directory}; inspect before review.')


OBSERVATIONS = [
    'Frame 0: the full orange cat and larger pale orange dog are inside the fixed garden view. '
    'The cat has tucked rounded forepaws, and the dog has visibly short dangling legs with no '
    'convincing supporting stance. The garden has conspicuous rows of holes and stripes.',
    'Frame 15: the cat forepaw silhouette changes but remains curled beneath its chest; '
    'the dog legs change contour while retaining a suspended appearance. Both animals retain '
    'their characteristic colors and overall identity from frame 0.',
    'Frame 30: both subjects have moved left in the fixed scene. The dog has a stretched '
    'chest-to-foreleg connection and shortened folded rear limb; the oblique diagnostic '
    'exposes this deformation. The cat paws remain compact and curled rather than well articulated.',
    'Frame 45: the two animals reach the left part of the frame. Dog paws still appear to '
    'dangle rather than make a stable planted support configuration; the cat lower legs '
    'remain shortened. The dog overlaps the cat in projection, with no validated physical interaction.',
    'Frame 60: both subjects remain near the left-side positions from frame 45. The dog '
    'retains broad torso and short hanging legs; the cat retains its orange stripes, upturned '
    'tail, and tucked forepaws. Apparent pose stability does not establish contact or gait quality.',
    'Frame 75: the later idle-like posture resembles frame 60 in location and overall shape. '
    'Source geometry and texture defects persist, including the dog broad softened surfaces '
    'and the cat curled feet. No detached major body fragments are evident in this sample.',
    'Frame 104: the animals remain fully visible near their later positions. Color identity '
    'is retained, but dog dangling legs and cat compact curled paws remain. The last frame '
    'does not resolve the anatomy or ground-contact failures seen in earlier poses.',
]
OBSERVATIONS_V3 = [
    'Frame 0: both full animals are visible in the garden. The dog has an upright head and '
    'stable neck/torso relation, unlike v2 preview, but short lower legs and dangling rear '
    'paws remain. Cat forepaws are still rounded and tightly curled. Foreground stripe gaps remain.',
    'Frame 15: the dog foreleg pose changes, showing a short separated forepaw on one side '
    'and a broad bent forelimb on the other. Rear feet remain high and close together. '
    'The cat compact paw shapes change without becoming a convincing planted walking stance.',
    'Frame 30: both animals have translated left in the fixed scene. The corresponding '
    'oblique view shows a squat dog hindquarter with partly merged/thick upper legs and '
    'short crossed or overlapping lower legs. The neck no longer has the v2 swept twist.',
    'Frame 45: the dog follows behind the cat near the left side of the garden. Both remain '
    'fully inside the image, but the dog paws still look suspended and the cat forepaws curled. '
    'Their projection overlap does not demonstrate collision-free interaction.',
    'Frame 60: both animals remain near the later stopping positions. Dog head and torso '
    'are visually more stable than v2 preview, but shortened legs and unsupported-looking '
    'rear feet remain. Cat anatomy defects persist.',
    'Frame 75: an idle-like later pose retains the same coat, tail and face identity. '
    'The dog leg silhouettes remain broad/fused near the body with short distal portions; '
    'the cat rounded paws remain tucked. No normal supporting stance is established.',
    'Frame 104: both animals remain fully framed and recognizable. Dog rear feet are still '
    'suspended-looking and cat forepaws remain curled. The final pose preserves appearance '
    'but does not resolve the incorrect articulation and unproven ground contact.',
]


def review(root):
    verify_inputs(root)
    video = root / 'gaussian_actions_7s.mp4'
    frames = evidence(root)
    verify_decoded_samples(video, frames)
    targets = [root / 'review' / f'visual_{suffix}.json' for suffix in ('review', 'decision')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Do not overwrite a preserved visual review.')
    result = create_review(video, frames)
    result['reviewer'] = dict(kind='agent', name='/root/dog_visual_review', observed_video=False)
    result['claim_type'] = 'conditional_motion_generation'
    result['provenance'] = dict(future_conditioned=False, test_used_for_training=False,
                                test_used_for_selection=False)
    result['provenance_scope'] = (
        'New predictions of a released pretrained skeletal controller, retargeted to stored '
        'Gaussian animals. No recorded future motion clip is input; no new Wan weights are '
        'trained or used. This is not a Wan-native Gaussian generator.')
    candidate_observations = OBSERVATIONS if root.name == 'v1' else OBSERVATIONS_V3
    for frame, observation in zip(result['frames'], candidate_observations):
        frame['observation'] = observation
    ids = [frame['id'] for frame in result['frames']]
    observations = {
        'paw_placement': ('fail',
            'The dog paws appear suspended and the cat forepaws remain curled under the chest '
            'across the inspected poses. Neither animal exhibits a convincing supporting '
            'stance. Numeric bounds or minimum Gaussian-center height cannot validate contact.'),
        'limb_bending': ('fail',
            'The dog forelimb-to-torso transition stretches and lower limbs remain shortened '
            'or folded; cat feet remain rounded and tucked. Oblique samples confirm these '
            'are not merely side-view occlusions. The visible bending is not a realistic gait.'),
        'body_shape': ('fail',
            'Recognizable identity persists, but the dog retains a broad softened torso and '
            'short malformed limbs while the cat has compressed limbs. The static source '
            'dog already has malformed paws and poor backside texture, so these are not '
            'all new motion-model failures. Motion adds visible limb/torso distortion.'),
        'tearing': ('uncertain',
            'No clearly detached large body parts are visible in the seven exact decoded '
            'frames. Stretched surfaces and incomplete detail remain. Sparse samples do '
            'not prove the absence of tearing between frames; background holes are a '
            'pre-existing scene-shell defect, not necessarily animal motion tearing.'),
        'temporal_coherence': ('uncertain',
            'Separated samples establish subject translation and pose changes with a fixed '
            'scene camera, followed by later near-stationary poses. Appearance is retained '
            'across these samples, but continuous playback, smooth gait phases, and stable '
            'contacts were not observed or established.'),
    }
    if root.name == 'v3':
        observations['limb_bending'] = ('fail',
            'The dog has thick partly fused upper legs and shortened, sometimes overlapping '
            'lower legs. Forelimb poses change but rear paws remain suspended-looking. '
            'The cat still has rounded curled forepaws and compressed hind limbs. These '
            'failures remain visible in the oblique views despite removal of the v2 neck twist.')
        observations['body_shape'] = ('fail',
            'Dog head and torso retain a more stable relationship than in v2 preview, '
            'without its gross twisted neck/chest fold. However, squat malformed legs '
            'and broad softened surfaces persist, and the cat lower limbs remain compressed. '
            'Some defects exist in the static source dog before motion retargeting.')
    result['dimensions'] = {
        key: dict(status=status, observation=observation, evidence_frame_ids=ids)
        for key, (status, observation) in observations.items()
    }
    expected_hashes = EXPECTED if root.name == 'v1' else EXPECTED_V3
    result['comparison_evidence'] = [
        dict(path=str((root / name).resolve()), sha256=expected)
        for name, expected in expected_hashes.items() if name != video.name
    ] + [dict(path=str(SOURCE.resolve()), sha256=SOURCE_HASH)]
    if root.name == 'v3':
        preview_root = root.parent / 'v2_preview'
        result['comparison_evidence'] += [
            dict(path=str((preview_root / name).resolve()), sha256=expected)
            for name, expected in PREVIEW_V2_HASHES.items()
            if name.endswith('.jpg')
        ]
        for record in result['comparison_evidence']:
            if digest(record['path']) != record['sha256']:
                raise ValueError('The inspected comparison evidence has changed.')
        result['changes_relative_to_v2_preview'] = dict(
            gross_neck_chest_twist='visibly_reduced',
            realistic_gait='not_achieved',
            sole_height_heuristic='not_physical_contact_validation')
    result['numeric_metrics'] = dict(frame_count=105, fps=15, duration_seconds=7)
    result['additional_checks'] = dict(
        two_animal_presence='pass_in_samples', full_body_framing='pass_in_samples',
        appearance_retention='recognizable_source_assets_not_quality_proof',
        ground_contact='fail_visually_unvalidated_physically',
        dog_cat_shoulder_height_ratio='not_measured_different_scene_depths',
        wan_native_generation=False, new_weights_trained=False,
        persistent_gaussian_update='mechanism_present_not_visual_quality_proof',
        quality_matched_compute_savings='not_established')
    result['limitations'] = [
        'Seven separated exact decoded MP4 frames, matching scene/oblique sheets, and the '
        'static dog six-view source were inspected; no continuous-playback attestation.',
        'Conditional pretrained skeletal generation applied to Gaussian assets, not a '
        'newly trained Gaussian-native/Wan-native foundation video model.',
        'The underlying dog and cat geometry already has anatomy and texture defects.',
        'No validated physical contact, collision handling, independent animal interaction, '
        'or appearance-quality-matched compute advantage is established.',
    ]
    decision = decide_review(result, video)
    if decision['accepted']:
        raise AssertionError('This visually rejected candidate must not be promoted.')
    decision.update(promoted=False, overall_quality_accepted=False,
        mechanism_scope='Stored Gaussian animals are posed by pretrained skeletal predictions; '
                        'this mechanism is separate from rejected visual/anatomical quality.')
    targets[0].write_text(json.dumps(result, indent=2), encoding='utf-8')
    targets[1].write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print(json.dumps(decision, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/real_video/gaussian_action_video/v1')
    parser.add_argument('--phase', required=True, choices=('extract', 'review', 'preview-v2'))
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    {'extract': extract, 'review': review, 'preview-v2': review_preview_v2}[args.phase](root)


if __name__ == '__main__':
    main()
