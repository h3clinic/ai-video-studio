"""Paired development comparison, never a new held-out generation benchmark."""
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path('artifacts/real_video/wan_bridge')


def frames(path):
    cap = cv2.VideoCapture(str(path))
    result = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return result


def main():
    out = ROOT/'v2'
    result = json.loads((out/'results.json').read_text())
    initial = result['initial']
    selected = result['selected_validation']
    old = {e['file']: e for e in initial['cases']}
    new = {e['file']: e for e in selected['cases']}
    assert old.keys() == new.keys() and len(old) == 57
    generated = [json.loads((ROOT/v/'generated_cat/metrics.json').read_text()) for v in ['v1','v2']]
    report = dict(validation_resolution=256, validation_clips=57,
                  baseline_v1_same_validation_psnr=initial['mean_psnr'],
                  selected_v2_psnr=selected['mean_psnr'],
                  clips_with_lower_mse=sum(new[k]['mse'] < old[k]['mse'] for k in old),
                  validation_mean_mse_reduction_fraction=1-selected['mean_mse']/initial['mean_mse'],
                  original_vae_validation_psnr=float(np.mean([e['psnr'] for e in result['original_vae_validation']])),
                  selected_step=result['selected_step'], training_seconds=result['training_seconds'],
                  prepare_seconds=result['prepare_seconds'],
                  cat_v1_agreement_psnr=generated[0]['agreement_with_compressed_original_rgb_psnr'],
                  cat_v2_agreement_psnr=generated[1]['agreement_with_compressed_original_rgb_psnr'],
                  interpretation='Data, resolution and update budget changed together. Cat is development only. No matched-quality speedup claim.')
    (out/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    sources = [frames(Path('artifacts/real_video/wan_baseline/cat_seed_421001/wan_original.mp4')),
               frames(ROOT/'v1/generated_cat/wan_latent_gaussian.mp4'),
               frames(ROOT/'v2/generated_cat/wan_latent_gaussian.mp4')]
    assert all(len(s) == 33 for s in sources)
    movie = []
    for index in range(33):
        canvas = Image.new('RGB', (1248, 272), '#151922')
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(['Original Wan RGB', 'Gaussian v1: 80 clips / 128px', 'Gaussian v2: 279 clips / 256px']):
            draw.text((column*416+10, 10), title, fill='white')
            canvas.paste(Image.fromarray(sources[column][index]).resize((416, 240)), (column*416, 32))
        movie.append(np.asarray(canvas))
    imageio.mimwrite(out/'comparison.mp4', movie, fps=16, codec='libx264', quality=8, macro_block_size=1)
    Image.fromarray(movie[16]).save(out/'comparison_frame.png')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
