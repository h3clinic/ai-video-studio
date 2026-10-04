"""Verify the saved Wan baseline and its pre-RGB generated latent; no quality gate."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .checkpoint_io import digest, load_verified
from .wan_baseline import MODEL, REVISION


def audit(directory):
    directory = Path(directory)
    config = json.loads((directory / 'protocol.json').read_text())
    metrics = json.loads((directory / 'metrics.json').read_text())
    packet = load_verified(directory / 'generated_latent.pt')
    assert config == metrics['config'] == packet['config']
    assert config['model'] == MODEL and config['revision'] == REVISION
    assert metrics['no_gaussian_claim'] is True
    assert digest(directory / 'generated_latent.pt') == metrics['latent_sha256']
    assert digest(directory / 'wan_original.mp4') == metrics['video_sha256']
    z = packet['latent']
    expected = (1, 16, (config['frames'] - 1)//4 + 1, config['height']//8, config['width']//8)
    assert tuple(z.shape) == expected and torch.isfinite(z).all()
    capture = cv2.VideoCapture(str(directory / 'wan_original.mp4'))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    array = np.asarray(frames)
    assert array.shape == (config['frames'], config['height'], config['width'], 3)
    report = dict(verified=True, decoded_shape=list(array.shape), latent_shape=list(z.shape),
                  latent_tensor_bytes=z.numel()*z.element_size(),
                  mean_absolute_frame_change_uint8=float(np.abs(np.diff(array.astype(np.float32), axis=0)).mean()),
                  limitations='Integrity and frame-change check only; not proof of realism or Gaussian generation.',
                  label='Original pretrained Wan RGB baseline; not Gaussian output')
    (directory / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
