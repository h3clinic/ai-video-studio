"""Repeat learned Gaussian inference while forbidding RGB input and Wan imports."""
import builtins
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import torch

from .checkpoint_io import load_verified, keep_windows_awake, digest
from .generate_wan_gaussian import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/real_video/wan_bridge/v1'))
    args = parser.parse_args()
    base = Path('artifacts/real_video/wan_baseline/cat_seed_421001')
    run = args.run
    output = run/'generated_cat'
    checkpoint_path = (run/'gaussian_decoder.pt').resolve()
    latent_path = (base/'generated_latent.pt').resolve()
    allowed = {checkpoint_path, latent_path}
    loaded = []
    original_load, original_import = torch.load, builtins.__import__

    def guarded_load(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved not in allowed:
            raise AssertionError(f'Unexpected tensor input: {resolved.name}')
        loaded.append(resolved.name)
        return original_load(path, *args, **kwargs)

    def guarded_import(name, *args, **kwargs):
        if name == 'diffusers' or name.startswith('diffusers.'):
            raise AssertionError('Wan RGB VAE/diffusers import forbidden in Gaussian generation')
        return original_import(name, *args, **kwargs)

    torch.set_num_threads(4)
    with keep_windows_awake(), patch('torch.load', side_effect=guarded_load), \
         patch('builtins.__import__', side_effect=guarded_import), \
         patch('cv2.VideoCapture', side_effect=AssertionError('RGB video input forbidden')):
        checkpoint = load_verified(checkpoint_path)
        latent = load_verified(latent_path)
        frames, fields, _ = generate(checkpoint, latent)
    retained = load_verified(output/'gaussian_fields.pt')
    difference = (fields-retained['fields']).abs().max().item()
    assert difference < 2e-5 and len(loaded) == 2
    assert retained['checkpoint_sha256'] == digest(checkpoint_path)
    assert retained['latent_sha256'] == digest(latent_path)
    # Training data is read only after independent inference for provenance auditing.
    data_path = Path(checkpoint['config'].get('training_data_path', '../../work/real_video/wan_bridge_v1.pt'))
    assert checkpoint['training_data_sha256'] == digest(data_path)
    data = load_verified(data_path)
    metadata = data['metadata']
    counts = {split: sum(e['split']==split for e in metadata) for split in ['train','validation','test']}
    assert counts == dict(train=checkpoint['config']['train_clips'],
                          validation=checkpoint['config']['validation_clips'], test=0)
    assert not ({e['group'] for e in metadata if e['split']=='train'} &
                {e['group'] for e in metadata if e['split']=='validation'})
    assert all(e['file'].endswith('.avi') and e['file'].startswith('v_') for e in metadata)
    capture = cv2.VideoCapture(str(output/'wan_latent_gaussian.mp4'))
    shapes = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            shapes.append(list(frame.shape))
    finally:
        capture.release()
    assert len(shapes)==33 and all(s==[480,832,3] for s in shapes)
    report = dict(verified=True, only_tensor_inputs_during_generation=loaded,
                  RGB_video_input_forbidden=True, diffusers_import_forbidden=True,
                  maximum_repeated_field_difference=difference, training_split_counts=counts,
                  training_validation_groups_disjoint=True, all_video_frames_decode=True,
                  limitations='Code-path/provenance audit; not proof of broad quality, physics, novelty or efficiency at matched quality.')
    (output/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
