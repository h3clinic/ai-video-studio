"""Decode a saved noise-generated Wan latent directly to learned Gaussians.

No Wan RGB VAE or source video enters generation. Original MP4 is read only
after inference and saving, for a separately labeled agreement comparison.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

from .checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint
from .wan_gaussian import WanGaussianDecoder, render_gaussian_video, gaussian_state


@torch.no_grad()
def generate(checkpoint, latent_packet):
    config = latent_packet['config']
    model = WanGaussianDecoder(width=checkpoint['config']['width']).eval().to('cuda')
    model.load_state_dict(checkpoint['model'], strict=True)
    latent = latent_packet['latent'].to('cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    fields = model(latent)
    video = render_gaussian_video(fields, config['height'], config['width'])
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    if not torch.isfinite(fields).all() or not torch.isfinite(video).all():
        raise ValueError('Nonfinite Gaussian generation')
    state = gaussian_state(fields[:, :, 0], config['height'], config['width'])
    axes_error = (state['axes'].transpose(-1, -2) @ state['axes'] - torch.eye(2, device='cuda')).abs().max().item()
    frames = (video[0].permute(0, 2, 3, 1).clamp(0, 1)*255).round().byte().cpu().numpy()
    return frames, fields.cpu(), dict(decoder_and_splat_seconds=seconds, peak_cuda_allocated_bytes=peak,
                                      first_frame_axes_orthogonality_max_error=axes_error,
                                      all_frames_finite=True,
                                      timing_scope='one cold decoder+splat pass; excludes model load, transfer and encoding; no speedup claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/real_video/wan_bridge/v1/gaussian_decoder.pt'))
    parser.add_argument('--baseline', type=Path, default=Path('artifacts/real_video/wan_baseline/cat_seed_421001'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/real_video/wan_bridge/v1/generated_cat'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'gaussian_fields.pt').exists():
        raise FileExistsError('Preserve previous output; choose a new directory')
    torch.set_num_threads(4)
    checkpoint = load_verified(args.checkpoint)
    latent = load_verified(args.baseline/'generated_latent.pt')
    with keep_windows_awake():
        frames, fields, metrics = generate(checkpoint, latent)
        sha = save_inference_checkpoint(dict(fields=fields, height=latent['config']['height'],
                                            width=latent['config']['width'],
                                            checkpoint_sha256=digest(args.checkpoint),
                                            latent_sha256=digest(args.baseline/'generated_latent.pt')),
                                        args.out/'gaussian_fields.pt')
        output = args.out/'wan_latent_gaussian.mp4'
        imageio.mimwrite(output, frames, fps=16, codec='libx264', quality=8, macro_block_size=1)
        metrics.update(video_sha256=digest(output), fields_sha256=sha, field_shape=list(fields.shape),
                       frame_shape=list(frames.shape), source_rgb_used_in_generation=False,
                       decoder_training=f"{checkpoint['config']['train_clips']} real UCF training clips; "
                                        f"{checkpoint['config']['validation_clips']} validation; cat not used for training/selection",
                       scope='Wan noise-generated latent + learned planar Gaussian decoder, not 3D physics or recurrent generation')
        # Only now, after generation/saving, inspect the original RGB baseline.
        cap = cv2.VideoCapture(str(args.baseline/'wan_original.mp4'))
        original = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                original.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        original = np.asarray(original)
        if original.shape != frames.shape:
            raise ValueError('Original video shape does not match Gaussian output')
        mse = np.square((original.astype(np.float32)-frames.astype(np.float32))/255).mean()
        metrics['agreement_with_compressed_original_rgb_psnr'] = float(-10*np.log10(max(float(mse), 1e-12)))
        h, w = frames.shape[1:3]
        sheet = Image.new('RGB', (w*2, h*4))
        for row, index in enumerate([0, len(frames)//3, 2*len(frames)//3, len(frames)-1]):
            sheet.paste(Image.fromarray(original[index]), (0, row*h))
            sheet.paste(Image.fromarray(frames[index]), (w, row*h))
        sheet.save(args.out/'original_left_gaussian_right.jpg')
        (args.out/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
