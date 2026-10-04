"""Bounded VAE-only diagnostic: generated prefix vs observed-video reconstruction.

No denoiser, training, Gaussian update, or new video generation is run. A real
nine-frame source clip is encoded once without tiling. That exact code and the
saved direct-generation prefix are each decoded with and without tiling.
"""
import argparse
from contextlib import contextmanager
import gc
import inspect
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from .checkpoint_io import digest, load_verified, save_inference_checkpoint
from .gaussian_program import write_json
from .prepare_wan22 import MODEL, REPO, REVISION


FRAMES, HEIGHT, WIDTH = 9, 256, 448
EVIDENCE = (0, 4, 5, 8)
DEFAULT_SOURCE = Path('artifacts/real_video/detail_memory/wan22_i2v_v2_small')


def latent_stats(value):
    if (not isinstance(value, torch.Tensor) or value.ndim != 5 or value.shape[0] != 1
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('Finite floating B=1,C,T,H,W latent required')
    value = value.detach().float().cpu()
    return dict(shape=list(value.shape), per_frame=[dict(
        mean=x.mean().item(), std=x.std(unbiased=False).item(),
        rms=x.square().mean().sqrt().item(), minimum=x.min().item(), maximum=x.max().item())
        for x in value[0].unbind(dim=1)])


def normalizers(vae, value):
    mean = torch.as_tensor(vae.config.latents_mean, device=value.device, dtype=torch.float32)
    std = torch.as_tensor(vae.config.latents_std, device=value.device, dtype=torch.float32)
    if (mean.shape != (48,) or std.shape != (48,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or not (std > 0).all()):
        raise ValueError('Native finite 48-channel VAE normalization required')
    return mean.view(1,48,1,1,1), std.view(1,48,1,1,1)


def decode_normalized(vae, value, *, tiled):
    """Independent-cache paired decode; do not reset within temporal chunks."""
    if value.shape != (1,48,3,16,28) or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError('Exact native nine-frame prefix [1,48,3,16,28] required')
    mean, std = normalizers(vae, value)
    vae.enable_tiling() if tiled else vae.disable_tiling()
    vae.clear_cache()
    try:
        with torch.inference_mode():
            decoded = vae.decode(value.float()*std+mean, return_dict=False)[0]
        if decoded.shape != (1,3,FRAMES,HEIGHT,WIDTH) or not torch.isfinite(decoded).all():
            raise ValueError('VAE did not produce a finite native nine-frame RGB clip')
        return decoded
    finally:
        vae.clear_cache()


def encode_observed(vae, rgb):
    """Actual consecutive source RGB, not repeated first-frame latent codes."""
    if (rgb.shape != (1,3,FRAMES,HEIGHT,WIDTH) or not rgb.is_floating_point()
            or not torch.isfinite(rgb).all() or rgb.min() < -1 or rgb.max() > 1):
        raise ValueError('Nine finite observed RGB frames in [-1,1] required')
    vae.disable_tiling()
    vae.clear_cache()
    try:
        with torch.inference_mode():
            raw = vae.encode(rgb).latent_dist.mode()
            mean, std = normalizers(vae, raw)
            normalized = (raw.float()-mean)/std
        if normalized.shape != (1,48,3,16,28) or not torch.isfinite(normalized).all():
            raise ValueError('Observed video did not encode to the native latent clock')
        return normalized
    finally:
        vae.clear_cache()


def comparison_metrics(left, right):
    if left.shape != right.shape or left.shape != (1,3,FRAMES,HEIGHT,WIDTH):
        raise ValueError('Matching diagnostic RGB shapes required')
    error = (left.float()-right.float()).square().mean(dim=(0,1,3,4))
    return dict(rmse=float(error.mean().sqrt()), per_frame=[dict(
        mse=float(mse), psnr_db=None if mse == 0 else float(10*torch.log10(4/mse)),
        exact=bool(mse == 0)) for mse in error],
        units='Float decoder RGB [-1,1], before video quantization/compression; PSNR range=2')


def load_inputs(source):
    from .prepare_animal_motion import WORK
    source = Path(source).resolve(strict=True)
    report_path = source/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if ((report.get('model'), report.get('revision'), report.get('height'), report.get('width'))
            != (REPO, REVISION, HEIGHT, WIDTH)):
        raise ValueError('Exact pinned native 256x448 source experiment required')
    path = source/'cows_direct_latent.pt'
    if not path.with_suffix('.pt.sha256.json').is_file():
        raise ValueError('Saved direct latent must have its checksum sidecar')
    checkpoint = load_verified(path)
    latent = checkpoint['latent']
    if (checkpoint.get('mode') != 'cows_direct' or latent.shape != (1,48,9,16,28)
            or not latent.is_floating_point() or not torch.isfinite(latent).all()):
        raise ValueError('Unchanged 33-frame direct experiment latent required')
    root = WORK/'extracted/DAVIS/JPEGImages/480p/cows'
    paths = [root/f'{index:05d}.jpg' for index in range(FRAMES)]
    if digest(paths[0]) != report['sources']['cows']['sha256']:
        raise ValueError('Observed source frame zero changed')
    frames, records = [], []
    for frame_path in paths:
        with Image.open(frame_path) as im:
            fitted = ImageOps.fit(im.convert('RGB'), (WIDTH,HEIGHT), Image.Resampling.LANCZOS)
            frames.append(np.asarray(fitted).copy())
        records.append(dict(path=str(frame_path.resolve()), sha256=digest(frame_path)))
    rgb = torch.from_numpy(np.stack(frames)).permute(3,0,1,2)[None].float()/127.5-1
    metadata = dict(source_report=str(report_path), source_report_sha256=digest(report_path),
        generated_checkpoint=str(path), generated_checkpoint_sha256=digest(path),
        generated_full_latent_stats=latent_stats(latent),
        source_model_manifest_sha256=report['model_manifest_sha256'], observed_frames=records,
        source_preprocessing='Consecutive DAVIS frames 00000..00008; same center-fit/Lanczos as source run',
        latent_prefix='First three saved latent frames -> first nine RGB frames; causal VAE context preserved')
    return latent[:,:,:3].float().clone(), rgb, metadata


def run(out, source=DEFAULT_SOURCE):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = dict(scope=__doc__, model=REPO, revision=REVISION, frames=FRAMES,
        height=HEIGHT, width=WIDTH, fps=24, weights_modified=False,
        denoising_run=False, training_run=False, checkpoint_promoted=False,
        quality_accepted=False, generated_gaussian_writeback=False, stages={}, samples={},
        code_sha256=digest(Path(__file__)),
        metric_semantics='Stage timings are wall seconds; CUDA stages synchronize before and after. '
            'CUDA peaks are total allocated/reserved PyTorch bytes including resident VAE weights, '
            'not incremental memory or complete driver usage. Decode includes transfers to/from CUDA. '
            'Worker elapsed starts after immutable output creation; supervisor separately bounds total startup/work.',
        limitations=['Observed control is reconstruction, not generation.',
            'Tiling comparison changes spatial context, so small numerical differences are expected.',
            'VAE decode internally clamps: returned saturation is measurable, pre-clamp extremes are not.',
            'Nine-frame prefix diagnoses the early failure, not complete 33-frame motion quality.'])
    def status(message):
        report.update(stage=message, wall_elapsed_seconds=time.perf_counter()-start)
        write_json(out/'report.json', report)
        print(message, flush=True)
    @contextmanager
    def stage(name, cuda=False):
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            record = dict(seconds=time.perf_counter()-begin)
            if cuda:
                record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
            report['stages'][name] = record
            status(name)
    try:
        status('CPU input and pinned model verification')
        with stage('input_load_and_validation'):
            generated, observed, report['inputs'] = load_inputs(source)
        with stage('model_hash_verification'):
            from .sample_wan22_memory import check_manifest
            check_manifest(MODEL)
            checksum = digest(MODEL/'download_manifest.json')
            if checksum != report['inputs']['source_model_manifest_sha256']:
                raise ValueError('Model manifest differs from saved-latent source experiment')
            report['model_manifest_sha256'] = checksum
        free, total = torch.cuda.mem_get_info()
        report['cuda_preflight'] = dict(free_bytes=free,total_bytes=total,required_free_bytes=8*2**30,
            note='VAE-only entry allowance, not a proven peak-memory bound')
        if torch.cuda.memory_allocated() or free < 8*2**30:
            raise RuntimeError('VAE-only diagnostic requires idle CUDA and 8 GiB available')
        torch.set_num_threads(4)
        with stage('vae_load', cuda=True):
            import diffusers
            from diffusers import AutoencoderKLWan
            implementation = Path(inspect.getfile(AutoencoderKLWan)).resolve()
            report['software'] = dict(torch_version=str(torch.__version__),
                diffusers_version=diffusers.__version__,vae_implementation_path=str(implementation),
                vae_implementation_sha256=digest(implementation),
                provenance='Installed implementation identity; not a verified upstream source revision')
            vae = AutoencoderKLWan.from_pretrained(MODEL/'vae', torch_dtype=torch.float32,
                local_files_only=True, low_cpu_mem_usage=True, device_map={'':'cuda'}).eval().requires_grad_(False)
            report['vae_config'] = dict(patch_size=vae.config.patch_size,z_dim=vae.config.z_dim,
                sample_tile=[vae.tile_sample_min_height,vae.tile_sample_min_width],
                sample_stride=[vae.tile_sample_stride_height,vae.tile_sample_stride_width],dtype='float32')
        with stage('observed_video_encode_untiled', cuda=True):
            reconstructed_code = encode_observed(vae, observed.cuda()).cpu()
        with stage('observed_latent_save_and_statistics'):
            report['observed_normalized_latent_stats'] = latent_stats(reconstructed_code)
            report['observed_first_vs_saved_direct_anchor'] = dict(
                rmse=(reconstructed_code[:,:,:1]-generated[:,:,:1]).square().mean().sqrt().item(),
                note='Actual nine-frame untiled source encode vs original first-frame encode rounded to BF16; '
                     'causal frame zero should agree up to the recorded BF16 roundoff.',
                equal_after_bfloat16_rounding=torch.equal(
                    reconstructed_code[:,:,:1].to(torch.bfloat16).float(),generated[:,:,:1]))
            save_inference_checkpoint(dict(latent=reconstructed_code, scope='Observed video reconstruction only'),
                                      out/'observed_latent.pt')
        decoded_results = {}
        for name, value in (('generated_direct',generated), ('observed_reconstruction',reconstructed_code)):
            for tiled in (True,False):
                label = name+('_tiled' if tiled else '_untiled')
                with stage(label+'_decode', cuda=True):
                    decoded_results[label] = decode_normalized(vae,value.cuda(),tiled=tiled).cpu()
                gc.collect(); torch.cuda.empty_cache()
        del vae
        gc.collect(); torch.cuda.empty_cache()
        with stage('numeric_comparisons'):
            report['comparisons'] = {name+'_tiling_difference':comparison_metrics(
                decoded_results[name+'_tiled'],decoded_results[name+'_untiled'])
                for name in ('generated_direct','observed_reconstruction')}
            for tiled in (True,False):
                label = 'observed_reconstruction'+('_tiled' if tiled else '_untiled')
                report['comparisons'][label+'_vs_source'] = comparison_metrics(decoded_results[label],observed)
            for label,value in decoded_results.items():
                report['samples'][label] = dict(per_frame=[dict(
                    mean_rgb=x.mean(dim=(1,2)).tolist(),std_rgb=x.std(dim=(1,2),unbiased=False).tolist(),
                    lower_clamp_fraction=(x <= -.999).float().mean().item(),
                    upper_clamp_fraction=(x >= .999).float().mean().item()) for x in value[0].unbind(dim=1)])
        import imageio.v2 as imageio
        sheet = Image.new('RGB',(WIDTH*len(EVIDENCE),(HEIGHT+26)*5))
        for row,(label,value) in enumerate([('observed_source',observed),*decoded_results.items()]):
            with stage(label+'_video_encode'):
                video = ((value[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).numpy()
                video_path = out/(label+'.mp4')
                imageio.mimwrite(video_path,video,fps=24,codec='libx264',quality=8,macro_block_size=1)
            with stage(label+'_delivered_evidence_and_hash'):
                reader = imageio.get_reader(video_path)
                try:
                    for column,index in enumerate(EVIDENCE):
                        frame = Image.fromarray(reader.get_data(index))
                        frame.save(out/f'{label}_{index:03d}.png')
                        sheet.paste(frame,(column*WIDTH,row*(HEIGHT+26)+26))
                        ImageDraw.Draw(sheet).text((column*WIDTH+8,row*(HEIGHT+26)+6),
                            f'{label} | {index}',fill='white')
                finally:
                    reader.close()
                report['samples'].setdefault(label,{})['mp4_sha256'] = digest(video_path)
                report['samples'][label]['mp4_bytes'] = video_path.stat().st_size
        with stage('comparison_sheet_save'):
            sheet.save(out/'comparison.png')
        report['finished'] = True
        status('Diagnostic complete; visual interpretation pending, no quality promotion')
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc(),finished=False)
        status('Diagnostic failed; partial evidence retained, no automatic retry')
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--wall-seconds',type=int,default=600)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from .guarded_worker import require_supervised_child, run_guarded_worker
    if args.worker:
        require_supervised_child()
        return run(args.out,args.source)
    return run_guarded_worker('real_video.diagnose_wan22_decode',
        ['--worker','--out',str(args.out.resolve()),'--source',str(args.source.resolve())],
        args.out.resolve(),wall_seconds=args.wall_seconds,resource='gpu')


if __name__ == '__main__':
    main()
