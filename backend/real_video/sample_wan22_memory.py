"""Stronger pretrained I2V: direct anchor vs restored Gaussian anchor.

No future RGB, tracks, mask or target is an inference input. The pretrained
Wan2.2 TI2V denoiser generates future RGB; it does NOT generate Gaussian state.
The rejected Wan2.1 adapters are not dimension-compatible and are not loaded.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from .checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint
from .gaussian_program import write_json
from .prepare_wan22 import MODEL, REPO, REVISION
from .train_latent_appearance import BASE, resource_guard


def release():
    gc.collect()
    torch.cuda.empty_cache()


def check_manifest(model_dir):
    """Revalidate the complete local pinned model before allocating a GPU.

    The fetch manifest records the official LFS hashes. Rehash every completed
    file here, including same-size modifications; the manifest is provenance,
    not a replacement for checking current bytes. This is one streaming pass
    before the worker wall budget, not a second model load into RAM.
    """
    import math
    from pathlib import PurePosixPath

    root = Path(model_dir).resolve(strict=True)
    report = json.loads((root/'download_manifest.json').read_text(encoding='utf-8'))
    if (report.get('status') != 'verified' or report.get('revision') != REVISION
            or report.get('repo') != REPO or report.get('license') != 'apache-2.0'):
        raise ValueError('Complete pinned official Apache-2.0 model download required')
    rows = report.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty complete model file manifest required')

    def checksum(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

    files = {}
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Malformed model file record')
        name = row.get('path')
        if (not isinstance(name, str) or not name or '\\' in name or ':' in name
                or any(part in ('', '.', '..') for part in name.split('/'))
                or PurePosixPath(name).is_absolute()):
            raise ValueError('Model file path must be a contained relative POSIX path')
        if name in files:
            raise ValueError('Duplicate model file record: '+name)
        target = (root/name).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError('Model file escapes its verified directory: '+name)
        size = row.get('size')
        if type(size) is not int or size <= 0 or target.stat().st_size != size:
            raise ValueError('Model file size changed or is invalid: '+name)
        recorded = row.get('sha256')
        remote = row.get('remote_lfs_sha256')
        if not checksum(recorded) or (remote is not None and not checksum(remote)):
            raise ValueError('Explicit SHA256 provenance required: '+name)
        if name.endswith('.safetensors') and remote is None:
            raise ValueError('Official LFS hash required for model weights: '+name)
        actual = digest(target)
        if actual != recorded or (remote is not None and actual != remote):
            raise ValueError('Model file SHA256 changed or mismatches official LFS: '+name)
        files[name] = target
        total += size
    if type(report.get('download_bytes')) is not int or report['download_bytes'] != total:
        raise ValueError('Manifest does not contain the complete declared download')
    required = {'model_index.json', 'transformer/config.json',
                'vae/config.json', 'scheduler/scheduler_config.json'}
    if not required <= files.keys():
        raise ValueError('Required model architecture/configuration files are missing')

    def load_json(name):
        value = json.loads(files[name].read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('Expected JSON object: '+name)
        return value

    # All shards named by the authoritative local index must themselves be
    # listed and verified. A forged "verified" partial download cannot pass.
    for component in ('transformer', 'vae'):
        single = component+'/diffusion_pytorch_model.safetensors'
        index_name = single+'.index.json'
        if single in files:
            if index_name in files:
                raise ValueError('Ambiguous single/sharded model layout: '+component)
            continue
        if index_name not in files:
            raise ValueError('Required model weights/index missing: '+component)
        mapping = load_json(index_name).get('weight_map')
        if not isinstance(mapping, dict) or not mapping or any(not isinstance(k, str) or not k for k in mapping):
            raise ValueError('Nonempty tensor-to-shard mapping required: '+component)
        for shard in mapping.values():
            if (not isinstance(shard, str) or '/' in shard or '\\' in shard or ':' in shard
                    or not shard.endswith('.safetensors') or component+'/'+shard not in files):
                raise ValueError('Unverified or unsafe model shard: '+str(shard))

    config = load_json('transformer/config.json')
    if (config.get('_class_name') != 'WanTransformer3DModel'
            or (config.get('in_channels'), config.get('out_channels'), config.get('patch_size')) != (48, 48, [1, 2, 2])
            or config.get('text_dim') != 4096 or config.get('image_dim') is not None):
        raise ValueError('Unexpected native TI2V transformer architecture')
    vae = load_json('vae/config.json')
    if (vae.get('_class_name') != 'AutoencoderKLWan' or vae.get('z_dim') != 48
            or vae.get('scale_factor_spatial') != 16 or vae.get('scale_factor_temporal', 4) != 4):
        raise ValueError('Native Wan2.2 VAE must use 48 channels and 16x16x4 compression')
    for name in ('latents_mean', 'latents_std'):
        values = vae.get(name)
        if (not isinstance(values, list) or len(values) != 48
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values)
                or (name == 'latents_std' and any(x <= 0 for x in values))):
            raise ValueError('Invalid native VAE normalization: '+name)
    scheduler = load_json('scheduler/scheduler_config.json')
    if (scheduler.get('_class_name') != 'UniPCMultistepScheduler'
            or scheduler.get('prediction_type') != 'flow_prediction'
            or scheduler.get('use_flow_sigmas') is not True
            or scheduler.get('use_dynamic_shifting', False) is not False):
        raise ValueError('Native fixed-shift flow scheduler required')
    model_index = load_json('model_index.json')
    # The official package defaults to WanPipeline (T2V). This runner uses
    # its verified native TI2V mask/timestep path directly, so the package is
    # not required to select an I2V pipeline or declare expand_timesteps.
    if model_index.get('_class_name') not in ('WanPipeline', 'WanImageToVideoPipeline'):
        raise ValueError('Unexpected official Wan pipeline class')
    return report


def validate_size(height, width, frames, steps):
    if any(type(x) is not int for x in (height, width, frames, steps)):
        raise ValueError('Integer test dimensions required')
    if height < 256 or width < 256 or height % 32 or width % 32 or height*width > 480*832:
        raise ValueError('Bounded 256..480p, 32-pixel-aligned test required')
    if frames < 17 or frames > 49 or (frames-1) % 4 or not 20 <= steps <= 50:
        raise ValueError('Bounded 4n+1-frame / 20..50-step diagnostic required')


def sampling_schedule(config, override=None):
    """Use the verified model's schedule unless an explicit ablation is named.

    The rejected v2 experiment silently replaced the downloaded shift5 with3.
    Restoring the published schedule is a controlled hypothesis, not a proven
    repair. Keep explicit overrides so the earlier setup remains reproducible.
    """
    native = config.get('flow_shift')
    for value in (native, override if override is not None else native):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 1 <= value <= 10):
            raise ValueError('Finite explicit flow shift in [1,10] required')
    return dict(flow_shift=float(native if override is None else override),
        downloaded_flow_shift=float(native),
        source='downloaded_scheduler' if override is None else 'explicit_experimental_override',
        quality_validated=False)


def gpu_memory_profile(height, width, frames):
    """Explicit feasibility exception, not a general lowering of the GPU gate.

    The pinned model was instantiated on meta by the preflight audit. The
    small profile has 1008 tokens, versus 5070 at 480p/49frames. One GiB for
    scratch is an engineering allowance, not a proven activation/workspace
    bound. All larger/unreviewed profiles retain the original11GiB floor.
    """
    small = (height, width, frames) == (256, 448, 33)
    parameters = 10021521792
    scratch = 2**30
    return dict(name='native_256p_33frame_feasibility' if small else 'original_11gib_gate',
        lower_resolution_feasibility=small,
        required_free_cuda_bytes=parameters+scratch if small else 11*2**30,
        estimated_parameter_bytes=parameters,
        estimated_scratch_allowance_bytes=scratch if small else None,
        patch_tokens=(1+(frames-1)//4)*(height//32)*(width//32),
        evidence='artifacts/real_video/detail_memory/wan22_preflight_v1/audit.json',
        limit='Estimated parameter/scratch entry gate, not guaranteed peak fit. '
            'Driver workspaces, allocator fragmentation and changing WDDM budgets remain unmeasured.')


def run(out, height=480, width=832, frames=49, steps=50, flow_shift=None):
    validate_size(height, width, frames, steps)
    verification_start = time.perf_counter()
    manifest = check_manifest(MODEL)
    schedule = sampling_schedule(json.loads((MODEL/'scheduler/scheduler_config.json').read_text()), flow_shift)
    verification_seconds = time.perf_counter()-verification_start
    if out.exists():
        raise FileExistsError('Experiment outputs are immutable')
    out.mkdir(parents=True)
    report = dict(model=REPO, revision=REVISION, model_manifest_sha256=digest(MODEL/'download_manifest.json'),
        scope='Native image-to-video first-frame recall, not 3D Gaussian motion generation',
        weights_modified=False, old_adapters_loaded=False, checkpoint_promoted=False,
        quality_accepted=False, output_domain='RGB video', future_inputs=False,
        generated_gaussian_writeback=False, frames=frames, fps=24, height=height, width=width,
        costs=dict(training_seconds=0., geometry_fitting_seconds=0.,
            generated_gaussian_rasterization_seconds=0., model_verification_seconds=verification_seconds,
            timing_semantics='Selected stages, not an exhaustive additive total. Wall elapsed includes all work '
                'since model verification, as of the last report. Denoising includes status writes; VAE decode '
                'includes latent transfer and RGB postprocessing/transfer. Bank build includes an initial recall '
                'and diagnostics; serialized recall is measured again separately. CUDA peaks are total '
                'PyTorch allocated/reserved bytes in each stage, including resident weights, not incremental '
                'memory or all driver/device usage.',
            exclusions='Pretrained model training and cached text-embedding computation are not measured here. '
                'Zero training/fitting/rasterization means those stages are absent, not free equivalents. '
                'Gaussian anchor read rasterization is included in bank build/read times, not RGB rendering.'),
        inference_steps=steps, seed=103075, guidance_scale=5., flow_shift=schedule['flow_shift'],
        sampling_schedule=schedule,
        appearance_bank='planar fixed-ID native 48-channel first-frame features; not 400k 3D cat',
        comparison='paired within Wan2.2; model/resolution/length differ from reader_v2, no cross-run savings claim',
        limitations=['Lower resolution than documented 720p model setting',
            'Small reused development case; not an independent benchmark',
            'First-frame likeness is enforced by pretrained I2V; judge future frames separately',
            'No trained 3D geometry writer, recurrent state prediction, or appearance writeback'],
        samples={}, code={str(p):digest(p) for p in (Path(__file__), Path('real_video/wan22_gaussian_anchor.py'))})
    def status(message):
        report['stage'] = message
        report['costs']['wall_elapsed_seconds'] = time.perf_counter()-verification_start
        write_json(out/'report.json', report)
        print(message, flush=True)
    done = None
    try:
        status('Guarded resource preflight')
        done, report['resources'] = resource_guard(out, 600)
        experiment_start = time.monotonic()
        profile = gpu_memory_profile(height, width, frames)
        cuda_free, cuda_total = torch.cuda.mem_get_info()
        report['cuda_preflight'] = dict(**profile, free_bytes=cuda_free, total_bytes=cuda_total,
            pytorch_allocated_bytes=torch.cuda.memory_allocated(),
            pytorch_reserved_bytes=torch.cuda.memory_reserved())
        status('Recorded CUDA resource profile')
        if torch.cuda.memory_allocated() or cuda_free < profile['required_free_cuda_bytes']:
            raise RuntimeError('Idle CUDA memory below explicit profile requirement; do not terminate other jobs')
        torch.set_num_threads(4)
        from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
        from .wan22_gaussian_anchor import build_gaussian_anchor, recall_gaussian_anchor
        from .prepare_animal_motion import WORK
        root = WORK/'extracted/DAVIS'
        assert 'cows' in (root/'ImageSets/2017/val.txt').read_text().split()
        assert 'sheep' in (root/'ImageSets/2017/train.txt').read_text().split()
        sources = {}
        for name in ('cows', 'sheep'):
            path = sorted((root/'JPEGImages/480p'/name).glob('*.jpg'))[0]
            im = ImageOps.fit(Image.open(path).convert('RGB'), (width, height), Image.Resampling.LANCZOS)
            im.save(out/f'{name}_source.png')
            sources[name] = (path, np.asarray(im).copy())
        report['sources'] = {name:dict(path=str(path), sha256=digest(path),
            preprocessing='Resize/center-fit source still to test aspect ratio; no generated frame crop/composite')
            for name,(path,_) in sources.items()}
        status('Encoding observed first frames with native 48-channel Wan2.2 VAE')
        load_start = time.perf_counter()
        vae = AutoencoderKLWan.from_pretrained(MODEL/'vae', torch_dtype=torch.float32,
            local_files_only=True, low_cpu_mem_usage=True, device_map={'':'cuda'}).eval().requires_grad_(False)
        torch.cuda.synchronize()
        report['costs']['anchor_vae_load_seconds'] = time.perf_counter()-load_start
        mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1,-1,1,1,1)
        std = torch.tensor(vae.config.latents_std, device='cuda').view(1,-1,1,1,1)
        anchors = {}
        with torch.inference_mode():
            for name,(path,rgb) in sources.items():
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                encode_start = time.perf_counter()
                x = torch.from_numpy(rgb).cuda().permute(2,0,1)[None,:,None].float()/127.5-1
                latent = ((vae.encode(x).latent_dist.mode()-mean)/std)[0,:,0].cpu()
                torch.cuda.synchronize()
                encode_seconds = time.perf_counter()-encode_start
                encode_peak = torch.cuda.max_memory_allocated()
                recalled, metrics, snapshot = build_gaussian_anchor(latent, digest(path))
                save_start = time.perf_counter()
                save_inference_checkpoint(snapshot, out/f'{name}_gaussian_bank.pt')
                save_seconds = time.perf_counter()-save_start
                # The actual conditioning path MUST use the serialized bank,
                # not the source image or the previous in-memory read result.
                read_start = time.perf_counter()
                restored, coverage = recall_gaussian_anchor(load_verified(out/f'{name}_gaussian_bank.pt'))
                read_seconds = time.perf_counter()-read_start
                if not torch.equal(restored, recalled):
                    raise ValueError('Gaussian anchor changed after serialized reload')
                anchors[name+'_gaussian'] = restored[None,:,None].to(torch.bfloat16)
                if name == 'cows':
                    anchors['cows_direct'] = latent[None,:,None].to(torch.bfloat16)
                report.setdefault('banks', {})[name] = dict(**metrics, restored_exact=True,
                    source_encode_seconds=encode_seconds, source_encode_peak_cuda_bytes=encode_peak,
                    serialize_and_verify_seconds=save_seconds, load_and_recall_seconds=read_seconds,
                    file_bytes=(out/f'{name}_gaussian_bank.pt').stat().st_size)
                status('Saved native Gaussian appearance bank: '+name)
        del vae, mean, std, x, latent, snapshot, recalled, restored, coverage
        release()
        embedded = load_verified(BASE/'embeddings.pt')
        positive = embedded['positive'].cuda().to(torch.bfloat16)
        negative = embedded['negative'].cuda().to(torch.bfloat16)
        report['prompt'] = embedded['prompt']
        report['prompt_embeddings_sha256'] = digest(BASE/'embeddings.pt')
        report['text_condition'] = dict(
            source='Wan2.1-T2V-1.3B official UMT5-XXL, cached by train_temporal_wan.py',
            encoder_config_sha256=digest(Path('../../work/wan21_13b/text_encoder/config.json')),
            expected_family='Wan2.2 TI2V uses UMT5-XXL; same Diffusers encode_prompt convention',
            precision='Previous encoder computed bfloat16 embeddings; reused identically in all branches',
            exact_wan22_encoder_weight_equivalence_verified=False,
            note='Architecture/family compatibility, not a bitwise equivalence audit of both text encoders')
        if positive.shape != (1,512,4096) or negative.shape != positive.shape:
            raise ValueError('Unexpected cached UMT5 embedding shape')
        del embedded
        save_inference_checkpoint(dict(anchors=anchors, scope='First frame only; source/direct anchor kept for paired control'), out/'anchors.pt')
        status('Loading larger 5B backbone directly in bfloat16; no old adapter')
        start = time.perf_counter()
        model = WanTransformer3DModel.from_pretrained(MODEL/'transformer', torch_dtype=torch.bfloat16,
            local_files_only=True, low_cpu_mem_usage=True, device_map={'':'cuda'}).eval().requires_grad_(False)
        torch.cuda.synchronize()
        report['model_load_seconds'] = time.perf_counter()-start
        report['model_parameter_count'] = sum(p.numel() for p in model.parameters())
        report['resident_parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters())
        loaded_free, loaded_total = torch.cuda.mem_get_info()
        report['model_loaded_cuda'] = dict(free_bytes=loaded_free, total_bytes=loaded_total,
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
            model_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
            note='Instantaneous post-load snapshot, not peak driver/device memory')
        if (profile['lower_resolution_feasibility'] and
                cuda_free < report['resident_parameter_bytes'] + profile['estimated_scratch_allowance_bytes']):
            raise RuntimeError('Measured model parameters exceed the small-profile entry allowance')
        outputs = {}
        latent_frames = 1+(frames-1)//4
        # Correct/direct comparison differs ONLY by the recalled first anchor.
        # Sheep changes subject/background/geometry. This checks conditioning
        # content sensitivity, not isolated identity or a Gaussian-memory gain.
        # Prioritize the user-requested Gaussian-memory video; a completed
        # direct-only control is not a substitute if the budget is exhausted.
        modes = ('cows_gaussian', 'cows_direct', 'sheep_gaussian')
        report['sheep_control_scope'] = 'Different source content, including background and subject; not identity-only'
        previous_denoise_seconds = 0.
        for mode in modes:
            if outputs and time.monotonic()-experiment_start+previous_denoise_seconds > 360:
                report['skipped_for_decode_budget'] = [m for m in modes if m not in outputs]
                status('Reserving remaining wall budget for decoding saved outputs')
                break
            anchor = anchors[mode].cuda()
            latents = torch.randn((1,48,latent_frames,height//16,width//16),device='cuda',
                dtype=torch.float32,generator=torch.Generator(device='cuda').manual_seed(report['seed']))
            first_mask = torch.ones((1,1,latent_frames,height//16,width//16),device='cuda')
            first_mask[:,:,0] = 0
            scheduler = UniPCMultistepScheduler.from_pretrained(MODEL/'scheduler', local_files_only=True,
                flow_shift=report['flow_shift'])
            scheduler.set_timesteps(steps, device='cuda')
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
            with torch.inference_mode():
                for index,t in enumerate(scheduler.timesteps):
                    conditioned = ((1-first_mask)*anchor+first_mask*latents).to(torch.bfloat16)
                    # Native TI2V protocol: source tokens have diffusion time0.
                    token_time = (first_mask[0,0,:,::2,::2]*t).flatten()[None]
                    pred = model(hidden_states=conditioned,timestep=token_time,
                        encoder_hidden_states=positive,return_dict=False)[0]
                    uncond = model(hidden_states=conditioned,timestep=token_time,
                        encoder_hidden_states=negative,return_dict=False)[0]
                    guided = uncond.float()+report['guidance_scale']*(pred.float()-uncond.float())
                    latents = scheduler.step(guided,t,latents,return_dict=False)[0]
                    if not torch.isfinite(latents).all():
                        raise ValueError('Nonfinite generation')
                    if index % 5 == 0:
                        status(f'{mode}: denoising {index+1}/{steps}')
                latents = (1-first_mask)*anchor+first_mask*latents
            torch.cuda.synchronize()
            report['samples'][mode] = dict(denoise_seconds=time.perf_counter()-start,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                native_first_frame_exact=bool(torch.equal(latents[:,:,:1],anchor.float())))
            previous_denoise_seconds = report['samples'][mode]['denoise_seconds']
            outputs[mode] = latents.cpu()
            save_start = time.perf_counter()
            save_inference_checkpoint(dict(latent=outputs[mode],mode=mode,seed=report['seed']),out/f'{mode}_latent.pt')
            report['samples'][mode]['latent_checkpoint_seconds'] = time.perf_counter()-save_start
            report['samples'][mode]['latent_checkpoint_bytes'] = (out/f'{mode}_latent.pt').stat().st_size
            status('Saved '+mode)
        del model, anchor, latents, first_mask, pred, uncond, guided, conditioned, token_time, positive, negative
        release()
        status('Decoding generated future frames; no overlays or interpolation')
        import imageio.v2 as imageio
        load_start = time.perf_counter()
        vae = AutoencoderKLWan.from_pretrained(MODEL/'vae',torch_dtype=torch.float32,
            local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda'}).eval().requires_grad_(False)
        vae.enable_tiling()
        torch.cuda.synchronize()
        report['costs']['decoder_vae_load_seconds'] = time.perf_counter()-load_start
        mean = torch.tensor(vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
        std = torch.tensor(vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
        sheets = Image.new('RGB',(width*3,(height+26)*4))
        for col,name in enumerate(('cows','cows','sheep')):
            sheets.paste(Image.fromarray(sources[name][1]),(col*width,26))
            ImageDraw.Draw(sheets).text((col*width+8,6),name+' observed first frame',fill='white')
        for row,(mode,value) in enumerate(outputs.items(),start=1):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            with torch.inference_mode():
                decoded = vae.decode(value.cuda()*std+mean,return_dict=False)[0]
                video = ((decoded[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
            torch.cuda.synchronize()
            report['samples'][mode]['decode_seconds'] = time.perf_counter()-start
            report['samples'][mode]['decode_peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
            report['samples'][mode]['decode_peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            encode_start = time.perf_counter()
            imageio.mimwrite(out/f'{mode}.mp4',video,fps=24,codec='libx264',quality=8,macro_block_size=1)
            report['samples'][mode]['mp4_encode_seconds'] = time.perf_counter()-encode_start
            report['samples'][mode]['mp4_bytes'] = (out/f'{mode}.mp4').stat().st_size
            # Evidence comes from the delivered encoded MP4, not pre-encode RGB.
            review_start = time.perf_counter()
            reader = imageio.get_reader(out/f'{mode}.mp4')
            for col,idx in enumerate((0,frames//2,frames-1)):
                frame = Image.fromarray(reader.get_data(idx))
                frame.save(out/f'{mode}_{idx:03d}.png')
                sheets.paste(frame,(col*width,row*(height+26)+26))
                ImageDraw.Draw(sheets).text((col*width+8,row*(height+26)+6),f'{mode} | frame {idx}',fill='white')
            reader.close()
            report['samples'][mode]['mp4_sha256'] = digest(out/f'{mode}.mp4')
            report['samples'][mode]['evidence_decode_and_hash_seconds'] = time.perf_counter()-review_start
            del decoded, video
            release()
            status('Decoded '+mode)
        sheets.save(out/'comparison.png')
        report['finished'] = True
        report['all_prespecified_modes_completed'] = set(outputs) == set(modes)
        status('Complete; visual review pending, no Gaussian-generation success claimed')
    except Exception as error:
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        status('Failed; evidence retained, no automatic bigger retry')
        raise
    finally:
        if done is not None:
            done.set()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--width',type=int,default=832)
    parser.add_argument('--frames',type=int,default=49)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--flow-shift',type=float,default=None,
        help='Explicit ablation only; omitted uses the verified downloaded scheduler')
    parser.add_argument('--wall-seconds',type=int,default=600)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    from .guarded_worker import require_supervised_child, run_guarded_worker
    if args.worker:
        require_supervised_child()
        run(args.out,args.height,args.width,args.frames,args.steps,args.flow_shift)
    else:
        validate_size(args.height,args.width,args.frames,args.steps)
        worker_args = ['--worker','--out',str(args.out.resolve()),'--height',str(args.height),
            '--width',str(args.width),'--frames',str(args.frames),'--steps',str(args.steps)]
        if args.flow_shift is not None:
            worker_args += ['--flow-shift',str(args.flow_shift)]
        run_guarded_worker('real_video.sample_wan22_memory',worker_args,
            args.out.resolve(),wall_seconds=args.wall_seconds,resource='gpu')
