"""Isolated TripoSR positional-interpolation parity experiment.

This is single-image reconstruction diagnosis, not video/motion generation.
It never modifies the canonical 3D asset or the installed Transformers package.
"""
import json
import math
from pathlib import Path
import sys
import time
from types import MethodType

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .checkpoint_io import digest, keep_windows_awake, save_inference_checkpoint


WORK = Path('../../work/real_video').resolve()
OUT = Path('artifacts/real_video/true3d/parity_v1')
SOURCE = Path('artifacts/real_video/true3d/v1/conditioning.png')
OBS = Path('artifacts/real_video/learned_motion/v1/dense_seed')
LEGACY_SOURCE = ('https://raw.githubusercontent.com/huggingface/transformers/'
                 'v4.35.0/src/transformers/models/vit/modeling_vit.py')


def legacy_interpolate_pos_encoding(self, embeddings, height, width):
    """Inspected Transformers 4.35.0 ViT interpolation math, lines 73-103.

    The 0.1 offset changes sampling locations. Using size=(32,32) is not
    numerically equivalent to scale_factor=(32.1/14,32.1/14).
    """
    num_patches = embeddings.shape[1] - 1
    num_positions = self.position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return self.position_embeddings
    cls = self.position_embeddings[:, :1]
    patch = self.position_embeddings[:, 1:]
    dim = embeddings.shape[-1]
    patch_size = self.patch_size
    h0, w0 = height // patch_size + 0.1, width // patch_size + 0.1
    edge = int(math.sqrt(num_positions))
    patch = patch.reshape(1, edge, edge, dim).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, scale_factor=(h0 / edge, w0 / edge),
                          mode='bicubic', align_corners=False)
    if patch.shape[-2:] != (int(h0), int(w0)):
        raise ValueError('Legacy interpolation produced unexpected shape')
    return torch.cat((cls, patch.permute(0, 2, 3, 1).reshape(1, -1, dim)), dim=1)


def load_model():
    sys.path.insert(0, str(WORK / 'TripoSR'))
    from omegaconf import OmegaConf
    from tsr.system import TSR
    config = OmegaConf.load(WORK / 'triposr_weights/config.yaml')
    OmegaConf.resolve(config)
    allowed = {'tsr.models.tokenizers.image.DINOSingleImageTokenizer',
               'tsr.models.tokenizers.triplane.Triplane1DTokenizer',
               'tsr.models.transformer.transformer_1d.Transformer1D',
               'tsr.models.network_utils.TriplaneUpsampleNetwork',
               'tsr.models.network_utils.NeRFMLP',
               'tsr.models.nerf_renderer.TriplaneNeRFRenderer'}
    if {config[k] for k in config if k.endswith('_cls')} != allowed:
        raise ValueError('Unexpected dynamically imported model class')
    config.image_tokenizer.pretrained_model_name_or_path = str(WORK / 'dino_config')
    model = TSR(config)
    weights = torch.load(WORK / 'triposr_weights/model.ckpt',
                         map_location='cpu', weights_only=True, mmap=True)
    if 'image_tokenizer.model.layers.0.attention.q_proj.weight' in model.state_dict():
        replacements = [('.encoder.layer.', '.layers.'),
                        ('.attention.attention.query.', '.attention.q_proj.'),
                        ('.attention.attention.key.', '.attention.k_proj.'),
                        ('.attention.attention.value.', '.attention.v_proj.'),
                        ('.attention.output.dense.', '.attention.o_proj.'),
                        ('.intermediate.dense.', '.mlp.fc1.'),
                        ('.output.dense.', '.mlp.fc2.')]
        mapped = {}
        for key, value in weights.items():
            if key.startswith('image_tokenizer.model.'):
                for old, new in replacements:
                    key = key.replace(old, new)
            if key in mapped:
                raise ValueError('Checkpoint translation collision')
            mapped[key] = value
        weights = mapped
    model.load_state_dict(weights, strict=True)
    model = model.cuda().eval()
    model.renderer.set_chunk_size(8192)
    return model


def delta(first, second):
    diff = first.float() - second.float()
    return dict(max_abs=float(diff.abs().max()), rms=float(diff.square().mean().sqrt()),
                mean_abs=float(diff.abs().mean()),
                relative_rms=float(diff.square().mean().sqrt() /
                                   second.float().square().mean().sqrt().clamp_min(1e-12)))


def targets():
    """Rebuild the exact conditioning mask; use white for NeRF background."""
    image = np.asarray(Image.open(OBS / 'observed_frame_2.png').convert('RGB')) / 255.
    mask = np.asarray(Image.open(OBS / 'mask.png').convert('L')) / 255.
    yy, xx = np.where(mask > .5)
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    crop = image[y0:y1, x0:x1]
    alpha = mask[y0:y1, x0:x1]
    size = int(max(crop.shape[:2]) / .85)
    y, x = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    full_alpha = np.zeros((size, size), np.float32)
    full_alpha[y:y + crop.shape[0], x:x + crop.shape[1]] = alpha
    rgb = np.ones((size, size, 3), np.float32)
    rgb[y:y + crop.shape[0], x:x + crop.shape[1]] = crop * alpha[..., None] + 1 - alpha[..., None]
    rgb = F.interpolate(torch.from_numpy(rgb).permute(2, 0, 1)[None],
                        size=(256, 256), mode='bilinear', align_corners=False,
                        antialias=True)[0].permute(1, 2, 0).numpy()
    alpha = F.interpolate(torch.from_numpy(full_alpha)[None, None],
                          size=(256, 256), mode='bilinear', align_corners=False,
                          antialias=True)[0, 0].numpy()
    return rgb, alpha


def quality(image, target, mask):
    """Uncalibrated source-view diagnostics, not novel-view ground truth."""
    import cv2
    error = image - target
    area = max(float(mask.sum()) * 3, 1)
    mse = float((error ** 2 * mask[..., None]).sum() / area)
    gt = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    pred = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5)
    m1, m2 = blurred(gt), blurred(pred)
    v1, v2 = blurred(gt * gt) - m1 * m1, blurred(pred * pred) - m2 * m2
    cov = blurred(gt * pred) - m1 * m2
    ssim = ((2 * m1 * m2 + .01 ** 2) * (2 * cov + .03 ** 2) /
            ((m1 * m1 + m2 * m2 + .01 ** 2) * (v1 + v2 + .03 ** 2)))
    ex = cv2.Sobel(pred, cv2.CV_32F, 1, 0) - cv2.Sobel(gt, cv2.CV_32F, 1, 0)
    ey = cv2.Sobel(pred, cv2.CV_32F, 0, 1) - cv2.Sobel(gt, cv2.CV_32F, 0, 1)
    return dict(foreground_psnr=-10 * math.log10(max(mse, 1e-12)),
                foreground_mae=float((abs(error) * mask[..., None]).sum() / area),
                foreground_ssim=float((ssim * mask).sum() / mask.sum()),
                foreground_gradient_mae=float(((abs(ex) + abs(ey)) * mask).sum() / (2 * mask.sum())),
                full_image_mse=float(np.mean(error ** 2)))


@torch.no_grad()
def main():
    if (OUT / 'report.json').exists():
        raise FileExistsError('Preserve completed parity experiment')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(430602)
    start = time.perf_counter()
    model = load_model()
    import transformers
    from tsr.utils import get_spherical_cameras
    condition = Image.open(SOURCE).convert('RGB')
    emb = model.image_tokenizer.model.embeddings
    original_method = emb.interpolate_pos_encoding
    tokens = emb.position_embeddings.new_empty(1, 1025, 768)
    current_pos = original_method(tokens, 512, 512)
    legacy_pos = legacy_interpolate_pos_encoding(emb, tokens, 512, 512)
    pos_difference = delta(current_pos, legacy_pos)
    rays_o, rays_d = get_spherical_cameras(1, 0., 1.9, 40., 256, 256)
    rays_o, rays_d = rays_o[0].reshape(-1, 3).cuda(), rays_d[0].reshape(-1, 3).cuda()
    codes, pictures, timings = {}, {}, {}
    try:
        for name in ['current', 'legacy']:
            emb.interpolate_pos_encoding = original_method if name == 'current' else MethodType(legacy_interpolate_pos_encoding, emb)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.perf_counter()
            code = model(condition, device='cuda')
            torch.cuda.synchronize()
            inference = time.perf_counter() - t
            codes[name] = code.cpu()
            chunks = []
            t = time.perf_counter()
            # Ray chunking bounds memory; model's own point chunking is nested.
            for offset in range(0, len(rays_o), 1024):
                chunks.append(model.renderer(model.decoder, code[0],
                                              rays_o[offset:offset + 1024],
                                              rays_d[offset:offset + 1024]).cpu())
            torch.cuda.synchronize()
            picture = torch.cat(chunks).reshape(256, 256, 3).clamp(0, 1).numpy()
            pictures[name] = picture
            timings[name] = dict(inference_seconds=inference,
                                 render_seconds=time.perf_counter() - t,
                                 peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated())
            Image.fromarray(np.uint8(picture * 255)).save(OUT / f'{name}_front.png')
            print(json.dumps(dict(stage=name, **timings[name])), flush=True)
            del code
    finally:
        emb.interpolate_pos_encoding = original_method
    target, mask = targets()
    Image.fromarray(np.uint8(target * 255)).save(OUT / 'source_white_256.png')
    Image.fromarray(np.uint8(mask * 255)).save(OUT / 'source_mask_256.png')
    panel = Image.new('RGB', (1024, 328), '#161c24')
    draw = ImageDraw.Draw(panel)
    images = [target, pictures['current'], pictures['legacy'], np.clip(abs(pictures['current'] - pictures['legacy']) * 8, 0, 1)]
    labels = ['Observed source / white bg', 'Current interpolation', 'Legacy training-era math', 'Absolute difference x8']
    for j, (im, label) in enumerate(zip(images, labels)):
        panel.paste(Image.fromarray(np.uint8(im * 255)), (j * 256, 38))
        draw.text((j * 256 + 6, 15), label, fill='white')
    draw.text((8, 308), 'Single-image reconstruction diagnostic | Same weights, official front camera | NOT video generation', fill='white')
    panel.save(OUT / 'comparison.png')
    metrics = {name: quality(value, target, mask) for name, value in pictures.items()}
    report = dict(experiment='TripoSR positional interpolation parity, reconstruction only',
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  checkpoint_revision='5b521936b01fbe1890f6f9baed0254ab6351c04a',
                  checkpoint_sha256=digest(WORK / 'triposr_weights/model.ckpt'),
                  conditioning_sha256=digest(SOURCE), diagnostic_code_sha256=digest(Path(__file__)),
                  legacy_source=LEGACY_SOURCE, legacy_source_locator='ViTEmbeddings.interpolate_pos_encoding, lines73-103',
                  positional_difference=pos_difference, latent_difference=delta(codes['current'], codes['legacy']),
                  rendered_difference=delta(torch.from_numpy(pictures['current']), torch.from_numpy(pictures['legacy'])),
                  source_view_metrics=metrics,
                  source_view_psnr_change_legacy_minus_current=metrics['legacy']['foreground_psnr'] - metrics['current']['foreground_psnr'],
                  camera=dict(elevation_degrees=0, distance=1.9, fovy_degrees=40, target=[0, 0, 0], resolution=[256, 256]),
                  timings=timings, wall_seconds=time.perf_counter() - start,
                  limitation='Official default front camera is not calibrated to this input; metrics include geometric/camera misalignment. No ground-truth novel views. Legacy interpolation restores a training-era operation but is not full-library parity. No canonical asset changed.')
    report['latent_checkpoint_sha256'] = save_inference_checkpoint(codes, OUT / 'codes.pt')
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    with keep_windows_awake():
        main()
