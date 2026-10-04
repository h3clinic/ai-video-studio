"""Paired same-latent decoder benchmark, not matched-quality superiority.

Fresh processes in ABBA order. Each process runs one cold decode followed by
two warm repeats, with synchronized timing and no retained previous GPU output.
The original Wan denoiser is NOT rerun; combined times are labeled estimates.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch
from .checkpoint_io import digest, load_verified, keep_windows_awake

SOURCE = Path('artifacts/real_video/prompt_generation/orange_cat_seed_531002')
DECODER = Path('artifacts/real_video/wan_bridge/v2/gaussian_decoder.pt')
MODEL = Path('../../work/wan21_13b')
ROOT = SOURCE/'paired_decoder_benchmark_v1'


class RSSSample:
    def __init__(self):
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak = self.process.memory_info().rss
        self.samples = 0

    def sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.samples += 1

    def run(self):
        while not self.stop.wait(.005):
            self.sample()

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set(); self.thread.join(); self.sample()


@torch.inference_mode()
def worker(mode, out, save_media):
    if out.exists():
        raise FileExistsError('Preserve previous measurements')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    # Keep shared library-import overhead comparable for isolated process RSS.
    from diffusers import AutoencoderKLWan
    from .wan_gaussian import WanGaussianDecoder, render_gaussian_video
    packet = load_verified(SOURCE/'generated_latent.pt')
    latent = packet['latent'].to('cuda', dtype=torch.float32)
    config = packet['config']; h, w = config['height'], config['width']
    started = time.perf_counter()
    if mode == 'wan':
        model = AutoencoderKLWan.from_pretrained(MODEL/'vae', torch_dtype=torch.float32,
            local_files_only=True).eval().to('cuda')
        model.enable_tiling()
        mean = torch.tensor(model.config.latents_mean, device='cuda').view(1,-1,1,1,1)
        std = torch.tensor(model.config.latents_std, device='cuda').view(1,-1,1,1,1)
        executed_parameter_count = sum(p.numel() for p in model.decoder.parameters()) + sum(p.numel() for p in model.post_quant_conv.parameters())
        settings = dict(dtype='float32', tiling_enabled=True,
            tile_sample_min_height=model.tile_sample_min_height,
            tile_sample_min_width=model.tile_sample_min_width,
            tile_sample_stride_height=model.tile_sample_stride_height,
            tile_sample_stride_width=model.tile_sample_stride_width,
            unused_encoder_resident=True)

        def decode():
            return model.decode(latent*std+mean, return_dict=False)[0], None
    else:
        checkpoint = load_verified(DECODER)
        model = WanGaussianDecoder(width=checkpoint['config']['width']).eval().to('cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        executed_parameter_count = sum(p.numel() for p in model.parameters())
        settings = dict(dtype='float32', render='normalized finite-support anisotropic 2D splats',
                        splats_per_frame=24960, fixed_cell_grid=[120,208])

        def decode():
            fields = model(latent)
            return render_gaussian_video(fields, h, w), fields
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    report = dict(mode=mode, latent_sha256=digest(SOURCE/'generated_latent.pt'),
        config=config, settings=settings, model_load_and_transfer_seconds=load_seconds,
        all_loaded_parameter_count=sum(p.numel() for p in model.parameters()),
        executed_decoder_parameter_count=executed_parameter_count,
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        cpu_threads=4, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        battery_at_start=psutil.sensors_battery()._asdict() if psutil.sensors_battery() else None,
        trials=[], timing_scope='GPU-ready latent to decoder RGB tensor, including VAE unnormalization / Gaussian splatting; excludes load, input transfer, output transfer, quantization and encoding',
        memory_scope='Allocated and reserved CUDA tensors include loaded model and latent. CPU RSS sampled every5ms during decode; isolated process, not whole pipeline.',
        limitations=['FP32 tiled existing Wan baseline, not fastest possible Wan configuration',
                     'Same latent/dimensions, unequal image quality; not a rate-distortion matched comparison',
                     'ABBA fresh-process order reduces order bias but power/thermal state is not controlled'])
    retained = None
    for index in range(3):
        gc.collect()
        if index == 0:
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with RSSSample() as rss:
            started = time.perf_counter()
            video, fields = decode()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
        row = dict(phase='cold' if index==0 else 'warm', repeat=index, seconds=elapsed,
            resident_cuda_allocated_bytes=resident,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            incremental_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()-resident,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            sampled_process_rss_peak_bytes=rss.peak, rss_samples=rss.samples)
        report['trials'].append(row)
        print(json.dumps(dict(mode=mode, out=str(out), **row)), flush=True)
        if index==0 and save_media:
            if not torch.isfinite(video).all():
                raise ValueError('Nonfinite decoder output')
            if mode=='wan':
                retained=((video[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
            else:
                retained=(video[0].clamp(0,1)*255).round().byte().permute(0,2,3,1).cpu().numpy()
        del video, fields
    if retained is not None:
        import imageio.v2 as imageio
        from PIL import Image
        np.save(out/'uncompressed_rgb.npy', retained)
        imageio.mimwrite(out/f'{mode}.mp4', retained, fps=16, codec='libx264', quality=8, macro_block_size=1)
        for frame in [0,8,16,24,32]:
            Image.fromarray(retained[frame]).save(out/f'frame_{frame:03d}.png')
        report['video_bytes']=(out/f'{mode}.mp4').stat().st_size
        report['video_sha256']=digest(out/f'{mode}.mp4')
    report['battery_at_end']=psutil.sensors_battery()._asdict() if psutil.sensors_battery() else None
    (out/'metrics.json').write_text(json.dumps(report,indent=2))


def summarize(root):
    records=[json.loads((root/f'{index}_{mode}'/'metrics.json').read_text())
             for index,mode in enumerate(['wan','gaussian','gaussian','wan'])]
    summary={}
    for mode in ['wan','gaussian']:
        selected=[row for record in records if record['mode']==mode for row in record['trials']]
        warm=[r['seconds'] for r in selected if r['phase']=='warm']
        summary[mode]=dict(cold_seconds=[r['seconds'] for r in selected if r['phase']=='cold'],
            warm_seconds=warm, warm_median_seconds=statistics.median(warm), warm_min_seconds=min(warm), warm_max_seconds=max(warm),
            peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in selected),
            peak_cuda_reserved_bytes=max(r['peak_cuda_reserved_bytes'] for r in selected),
            sampled_process_rss_peak_bytes=max(r['sampled_process_rss_peak_bytes'] for r in selected),
            executed_decoder_parameter_count=next(r['executed_decoder_parameter_count'] for r in records if r['mode']==mode))
    prior=json.loads((SOURCE/'metrics.json').read_text())
    denoise=prior['timings']['fresh_denoise_seconds']; denoise_peak=prior['denoise_peak_cuda_allocated_bytes']
    for item in summary.values():
        item['estimated_common_denoise_plus_warm_decode_seconds']=denoise+item['warm_median_seconds']
        item['estimated_staged_pipeline_peak_cuda_allocated_bytes']=max(denoise_peak,item['peak_cuda_allocated_bytes'])
    a,b=summary['wan'],summary['gaussian']
    ratios=dict(decoder_speedup=a['warm_median_seconds']/b['warm_median_seconds'],
        decoder_cuda_peak_reduction_percent=100*(1-b['peak_cuda_allocated_bytes']/a['peak_cuda_allocated_bytes']),
        estimated_common_denoise_decode_time_reduction_percent=100*(1-b['estimated_common_denoise_plus_warm_decode_seconds']/a['estimated_common_denoise_plus_warm_decode_seconds']),
        estimated_staged_cuda_peak_reduction_percent=100*(1-b['estimated_staged_pipeline_peak_cuda_allocated_bytes']/a['estimated_staged_pipeline_peak_cuda_allocated_bytes']))
    wa=np.load(root/'0_wan'/'uncompressed_rgb.npy'); ga=np.load(root/'1_gaussian'/'uncompressed_rgb.npy')
    delta=(wa.astype(np.float32)-ga.astype(np.float32))/255
    mse=float(np.mean(delta**2)); mae=float(np.mean(np.abs(delta)))
    quality=dict(agreement_psnr_db=float(-10*np.log10(mse)), agreement_mae=mae,
        label='Agreement with original Wan decoded output, not ground-truth accuracy or generative quality')
    storage=dict(wan_latent_file_bytes=(SOURCE/'generated_latent.pt').stat().st_size,
        gaussian_fields_file_bytes=(SOURCE/'gaussian_fields.pt').stat().st_size,
        gaussian_decoder_file_bytes=DECODER.stat().st_size,
        full_wan_vae_weight_files_bytes=sum(p.stat().st_size for p in (MODEL/'vae').glob('*.safetensors')),
        wan_mp4_bytes=(root/'0_wan'/'wan.mp4').stat().st_size,
        gaussian_mp4_bytes=(root/'1_gaussian'/'gaussian.mp4').stat().st_size,
        raw_rgb_uint8_tensor_bytes=int(wa.nbytes),
        note='Gaussian fields are uncompressed framewise FP32, not a persistent3D scene or competitive video codec. MP4 size comparisons have unequal visual quality; per-clip representations require their decoder.')
    from PIL import Image,ImageDraw
    for frame in [0,16,32]:
        canvas=Image.new('RGB',(1664,520),'#171b22');draw=ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(wa[frame]),(0,28));canvas.paste(Image.fromarray(ga[frame]),(832,28))
        draw.text((8,7),'Original Wan RGB decoder | same fresh latent',fill='white')
        draw.text((840,7),'Learned 2D Gaussian decoder | lower detail',fill='white')
        canvas.save(root/f'comparison_{frame:03d}.jpg')
    report=dict(protocol=json.loads((root/'protocol.json').read_text()), measured=summary, ratios=ratios,
        shared_measured_denoiser_seconds=denoise, shared_measured_denoiser_peak_cuda_allocated_bytes=denoise_peak,
        combined_estimate_limit='Same previously measured denoising cost added to each warm decoder; not two newly timed end-to-end generations. Excludes shared text encoding, load/offload, output transfers, and MP4 encoding. GPU peaks are staged max, not sum. Total CPU RAM and energy savings not established.',
        quality=quality, storage=storage)
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(saved=str(root/'summary.json'), ratios=ratios)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['wan','gaussian'])
    parser.add_argument('--out',type=Path,default=ROOT)
    parser.add_argument('--save-media',action='store_true')
    args=parser.parse_args()
    with keep_windows_awake():
        if args.worker:
            worker(args.worker,args.out,args.save_media)
            return
        if args.out.exists():
            raise FileExistsError('Preserve prior benchmark')
        args.out.mkdir(parents=True)
        protocol=dict(order=['wan','gaussian','gaussian','wan'], fresh_process_each=True,
            cold_runs_per_mode=2, warm_runs_per_mode=4, cpu_threads=4, dtype='float32',
            latent_sha256=digest(SOURCE/'generated_latent.pt'), gaussian_decoder_sha256=digest(DECODER),
            vae_input_hashes={str(p):digest(p) for p in (MODEL/'vae').glob('*') if p.is_file()},
            format=dict(frames=33,height=480,width=832,fps=16),
            no_weight_training=True, no_final_test_selection=True, quality_matched=False,
            memory='Fresh-process CUDA allocated/reserved peaks and sampled5ms isolated-process RSS',
            conditions='Existing RTX5070Laptop, current battery/power state; not a controlled-power benchmark')
        (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        (args.out/'runner.py').write_bytes(Path(__file__).read_bytes())
        for index, mode in enumerate(protocol['order']):
            command=[sys.executable,'-m','real_video.benchmark_wan_decoders','--worker',mode,'--out',str(args.out/f'{index}_{mode}')]
            if index in [0,1]:command.append('--save-media')
            subprocess.run(command,check=True)
        summarize(args.out)


if __name__=='__main__':main()
