"""One bounded native-resolution Wan2.2 A14B Gaussian-reader/LoRA pilot.

Run only on an explicitly approved cloud GPU. Downloads are a separate, pinned
preparation step; this script is local-files-only and never purchases compute.
The <=600 second deadline starts before model loading. It does NOT stop Pod
billing: an external billing watchdog and human-authorized budget are required.

This is single-source low-noise-expert appearance fitting, NOT generalized
training, a learned 3D writer, or native Gaussian motion generation. Generated
RGB uses only the first-frame image/latent memory and prompt; recorded future
frames enter the training target only. The static planar memory is an appearance
cue, not future tracks. No crop overlays or camera-orbit stand-ins are produced.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time


REPO = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
REVISION = '596658fd9ca6b7b71d5057529bbf319ecbc61d74'
PROMPT = ('A realistic donkey slowly eats a fresh orange, moving its muzzle and '
          'jaw naturally as it chews. Preserve the donkey identity, natural fur, '
          'anatomy and original environment. Fixed camera, continuous natural action.')
NEGATIVE = ('blurry, deformed anatomy, extra limbs, duplicate animal, disintegrating '
            'fur, camera orbit, static image, text, watermark')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_report(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def validate_profile(frames, steps, train_steps, wall_seconds):
    if frames not in (17, 33, 81):
        raise ValueError('Explicit 17,33,81-frame pilot required; no silent shortening')
    if type(steps) is not int or not 10 <= steps <= 50:
        raise ValueError('Explicit 10..50 denoising steps required')
    if type(train_steps) is not int or not 1 <= train_steps <= 4:
        raise ValueError('Bounded 1..4 real optimizer steps required')
    if type(wall_seconds) is not int or not 30 <= wall_seconds <= 600:
        raise ValueError('Model-load-through-output budget must be 30..600 seconds')


def validate_model(model):
    """Check pinned local Hub snapshot plus native I2V expert/VAE contracts.

    The downloader supplies official provenance; a matching path and config are
    not independent cryptographic proof of every shard. Report that limitation.
    """
    model = Path(model).resolve(strict=True)
    if model.name != REVISION:
        raise ValueError('Use the exact pinned Hugging Face snapshot directory')
    result = {}
    for name in ('model_index.json', 'transformer/config.json', 'transformer_2/config.json',
                 'vae/config.json', 'scheduler/scheduler_config.json'):
        file = model/name
        result[name] = json.loads(file.read_text(encoding='utf-8'))
    index = result['model_index.json']
    if (index.get('_class_name') != 'WanImageToVideoPipeline'
            or index.get('boundary_ratio') != .9 or index.get('expand_timesteps', False)):
        raise ValueError('Expected native A14B I2V boundary .9, not TI2V first-frame protocol')
    for name in ('transformer/config.json', 'transformer_2/config.json'):
        config = result[name]
        if (config.get('in_channels') != 36 or config.get('out_channels') != 16
                or tuple(config.get('patch_size', ())) != (1,2,2)
                or config.get('image_dim') is not None):
            raise ValueError('Expert configuration is not pinned A14B I2V')
    if result['vae/config.json'].get('z_dim') != 16:
        raise ValueError('A14B requires a 16-channel native VAE')
    return model, {name: digest(model/name) for name in result}


def flow_training_pair(clean, noise, sigma):
    """Diffusers Wan convention: noise at sigma=1, data at sigma=0."""
    if clean.shape != noise.shape or not 0 < float(sigma) < .9:
        raise ValueError('Matched clean/noise and strictly low-expert sigma required')
    return (1-sigma)*clean + sigma*noise, noise-clean


def source_indices(source_frames, source_fps, frames, fps=16):
    if source_frames < 1 or source_fps <= 0:
        raise ValueError('Nonempty source and positive source clock required')
    indices = [round(t*source_fps/fps) for t in range(frames)]
    if indices[-1] >= source_frames:
        raise ValueError('Source too short for requested duration; no loops or endpoint holds')
    return indices


@contextmanager
def deadline_guard(out, report, seconds):
    """Kill only this worker on elapsed deadline; preserve last report/checkpoint."""
    done = threading.Event()
    started = time.monotonic()
    def guard():
        if not done.wait(seconds):
            # Separate file avoids racing the normal stage-report atomic writer.
            try:
                write_report(out/'deadline_stop.json', dict(status='deadline_exceeded',
                    wall_seconds=time.monotonic()-started, last_stage=report.get('stage'),
                    accepted=False, billing_stopped=False))
            finally:
                os._exit(124)
    thread = threading.Thread(target=guard, daemon=True)
    thread.start()
    try:
        yield started
    finally:
        done.set()
        thread.join(timeout=1)


def run(args):
    validate_profile(args.frames, args.steps, args.train_steps, args.wall_seconds)
    from cloud.a14b_preflight import require_dependencies
    require_dependencies()
    model_path, configs = validate_model(args.model)
    if args.out.exists():
        raise FileExistsError('Immutable new output directory required')
    args.out.mkdir(parents=True)
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    import imageio.v2 as imageio
    from real_video.checkpoint_io import save_inference_checkpoint, load_verified

    report = dict(schema_version=1, model=REPO, revision=REVISION, config_sha256=configs,
        model_shard_hashes_independently_rechecked=False,
        model_provenance='Pinned official local Hugging Face snapshot; downloader owns shard verification',
        source_sha256=digest(args.source), source=str(args.source), prompt=args.prompt,
        negative_prompt=NEGATIVE, seed=args.seed, frames=args.frames, fps=16,
        duration_seconds=args.frames/16, height=480, width=832, sampling_steps=args.steps,
        guidance_scale=3.5, guidance_scale_2=3.5, training_steps_requested=args.train_steps,
        scope='One-source low-noise-expert LoRA and first-frame Gaussian reader appearance-fit pilot',
        high_noise_expert_trained=False, low_noise_expert_trained=False,
        future_tracks_at_inference=False, future_rgb_written_to_memory=False,
        gaussian_native_generation=False, geometry_scope='Static planar anchor; no measured 3D',
        quality_accepted=False, accepted_autonomous_video=False, stages={}, training=[], videos={},
        timing_scope={'denoising': 'Native pipeline call including image-condition VAE encode, latent setup and callbacks; excludes final RGB decode and MP4 encoding. Not isolated DiT time.'},
        baseline_scope='Same loaded pipeline with adapters disabled, not removed; adapter and Gaussian-memory allocations remain resident. Not a pristine-Wan memory benchmark.',
        limits=['No held-out generalization evidence', 'First-frame planar memory only; no generated-state writeback',
                'One to four fitting steps are not converged training',
                '20-step default is explicitly a quality pilot, not full convergence',
                'No claimed compute savings; base and adapted generation both pay full diffusion/VAE cost'])
    def status(stage):
        report['stage'] = stage
        write_report(args.out/'report.json', report)
        print(json.dumps(dict(stage=stage)), flush=True)
    def tidy():
        gc.collect(); torch.cuda.empty_cache()
    @contextmanager
    def stage(name):
        status(name)
        torch.cuda.synchronize(); start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            torch.cuda.synchronize()
            report['stages'][name] = dict(seconds=time.perf_counter()-start,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                process_rss_bytes=psutil.Process().memory_info().rss)
            status(name+(' completed' if succeeded else ' failed'))
    status('resource preflight')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One explicitly provisioned visible CUDA GPU required')
    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    report['resources'] = dict(gpu=props.name, gpu_total_bytes=props.total_memory,
        host_available_bytes=psutil.virtual_memory().available, worker_deadline_seconds=args.wall_seconds)
    if props.total_memory < 70*2**30 or psutil.virtual_memory().available < 90*2**30:
        raise RuntimeError('This unquantized pilot requires an 80GB-class GPU and >=90GiB free host RAM')
    if torch.cuda.memory_allocated() > 0:
        raise RuntimeError('Unexpected already-allocated worker CUDA memory')
    torch.manual_seed(args.seed)
    report['versions'] = dict(torch=torch.__version__)
    try:
        with deadline_guard(args.out, report, args.wall_seconds) as started:
            from diffusers import WanImageToVideoPipeline
            import diffusers
            from real_video.wan_part_control import install_part_adaptation
            from real_video.wan_cat_memory import LowRankLinear
            from real_video.gaussian_latent_memory import GaussianLatentMemory
            from real_video.gaussian_part_protocol import GaussianPartProtocol
            from real_video.latent_track_memory import planar_gaussian_cache
            report['versions']['diffusers'] = diffusers.__version__
            with stage('model_load'):
                pipe = WanImageToVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                    local_files_only=True, low_cpu_mem_usage=True)
                pipe.set_progress_bar_config(disable=False)
                pipe.vae.to(dtype=torch.float32)
                report['pipeline_source_sha256'] = digest(Path(__import__('inspect').getfile(type(pipe))))
            with stage('source_preparation'):
                reader = imageio.get_reader(args.source)
                try:
                    metadata = reader.get_meta_data()
                    source_count = reader.count_frames()
                    indices = source_indices(source_count, float(metadata['fps']), args.frames)
                    images = [Image.fromarray(reader.get_data(i)).convert('RGB').resize((832,480), Image.Resampling.LANCZOS)
                              for i in indices]
                finally:
                    reader.close()
                report['source_frame_indices'] = indices
                report['source_fps'] = float(metadata['fps'])
                report['training_target_clock'] = 'Nearest source samples at native16fps; repeated source samples disclosed in indices'
                anchor_image = images[0]
                anchor_image.save(args.out/'anchor.png')
            with stage('prompt_encode'):
                pipe.text_encoder.to(device)
                with torch.no_grad():
                    positive, negative = pipe.encode_prompt(prompt=args.prompt, negative_prompt=NEGATIVE,
                        do_classifier_free_guidance=True, device=device, num_videos_per_prompt=1)
                positive, negative = positive.to(torch.bfloat16), negative.to(torch.bfloat16)
                pipe.text_encoder.to('cpu'); tidy()
            with stage('training_target_and_native_anchor_encode'):
                pipe.vae.to(device)
                with torch.no_grad():
                    anchor_rgb = pipe.video_processor.preprocess(anchor_image, height=480, width=832).to(device)
                    rgb = pipe.video_processor.preprocess(images,height=480,width=832).to(device)
                    rgb = rgb.permute(1,0,2,3)[None]
                    mean = torch.tensor(pipe.vae.config.latents_mean,device=device).view(1,16,1,1,1)
                    std = torch.tensor(pipe.vae.config.latents_std,device=device).view(1,16,1,1,1)
                    # Separate single-frame encode makes future-target leakage impossible here.
                    anchor = ((pipe.vae.encode(anchor_rgb[:,:,None]).latent_dist.mode()-mean)/std)[0,:,0].cpu()
                    clean = ((pipe.vae.encode(rgb).latent_dist.mode()-mean)/std).to(torch.bfloat16)
                    _, native_condition = pipe.prepare_latents(anchor_rgb,1,16,480,832,args.frames,
                        torch.float32,device,torch.Generator(device=device).manual_seed(args.seed))
                if tuple(native_condition.shape[1:]) != (20,*clean.shape[2:]):
                    raise ValueError('Native I2V condition must contain4mask+16latentchannels')
                native_condition = native_condition.to(torch.bfloat16)
                del rgb, mean, std
                pipe.vae.to('cpu'); tidy()
            with stage('gaussian_anchor_fit_and_part_binding'):
                c,h,w = anchor.shape
                yy,xx = torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
                ids = torch.arange(h*w); gen = torch.zeros_like(ids)
                centers = torch.stack((xx,yy),-1).reshape(-1,2).float()
                asset_digest = digest(args.out/'anchor.png')
                cache = planar_gaussian_cache(centers,torch.ones(len(ids)),ids,h,w,
                    generations=gen,asset_digest=asset_digest)
                memory = GaussianLatentMemory(ids,c,asset_digest,device=device,generations=gen)
                memory.write(anchor.to(device),cache,ids=ids,confidence=torch.ones(h,w,device=device))
                if args.part_labels:
                    label_image = Image.open(args.part_labels)
                    if label_image.mode not in ('L','I','I;16','P'):
                        raise ValueError('Part labels must be an integer grayscale/palette label image')
                    labels = torch.from_numpy(np.array(label_image.resize((w,h),Image.Resampling.NEAREST)).astype(np.int64)).flatten()
                    report['part_assignment'] = dict(kind='supplied first-frame ownership labels',
                        source_sha256=digest(args.part_labels), count=int(labels.unique().numel()))
                else:
                    labels = torch.zeros_like(ids)
                    report['part_assignment'] = dict(kind='single whole-scene part; no automatic anatomy segmentation',count=1)
                protocol = GaussianPartProtocol(memory,ids=ids,generations=gen,part_ids=labels)
                report['memory'] = memory.memory_bytes()
                save_inference_checkpoint(memory.snapshot(),args.out/'gaussian_anchor.pt')
                del anchor
            with stage('install_low_expert_adaptation'):
                low = pipe.transformer_2.to(device)
                low.requires_grad_(False)
                frozen = [(p,p._version) for p in low.parameters()]
                control, handles, details = install_part_adaptation(low,intent_dim=1,
                    blocks=(0,10,20,30),hidden=64,rank=4)
                report['adapter'] = details
                t,lh,lw = clean.shape[2:]
                intents = torch.ones(1,int(labels.unique().numel()),1,device=device)
                # The same first-frame geometry is recalled at every latent time.
                # Nothing in these controls comes from later source frames.
                control.bind_memory_frames(protocol,[cache]*t,h,w,intents,(t,lh//2,lw//2))
                trainable = {name:p for name,p in low.named_parameters() if p.requires_grad}
                original = {name:p.detach().cpu().clone() for name,p in trainable.items()}
                optimizer = torch.optim.AdamW(trainable.values(),lr=1e-4,weight_decay=0.)
                low.enable_gradient_checkpointing(); low.train()
            with stage('low_expert_training'):
                for step in range(args.train_steps):
                    sigma = (.5,.7,.3,.8)[step]
                    noise = torch.randn(clean.shape,device=device,dtype=clean.dtype,
                        generator=torch.Generator(device=device).manual_seed(args.seed+100+step))
                    noisy,target = flow_training_pair(clean,noise,sigma)
                    optimizer.zero_grad(set_to_none=True)
                    model_input = torch.cat((noisy,native_condition),dim=1).requires_grad_(True)
                    prediction = low(hidden_states=model_input,
                        timestep=torch.tensor([sigma*1000],device=device),
                        encoder_hidden_states=positive,return_dict=False)[0]
                    # No first-frame enforcement pixels can dominate this loss.
                    loss = (prediction[:,:,1:].float()-target[:,:,1:].float()).square().mean()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(trainable.values(),1.,error_if_nonfinite=True)
                    optimizer.step()
                    report['training'].append(dict(step=step+1,sigma=sigma,loss=float(loss.detach()),grad_norm=float(norm)))
                    status('training step '+str(step+1))
                low.eval(); low.disable_gradient_checkpointing()
                report['base_parameters_unchanged'] = all(p._version==version for p,version in frozen)
                if not report['base_parameters_unchanged']:
                    raise ValueError('Frozen pretrained expert weights unexpectedly changed')
                report['adapter_parameter_change_l2'] = float(sum((p.detach().cpu()-original[name]).square().sum()
                    for name,p in trainable.items()).sqrt())
                if report['adapter_parameter_change_l2'] <= 0:
                    raise ValueError('Optimizer did not change actual adapter weights')
                report['low_noise_expert_trained'] = True
            with stage('adapter_checkpoint'):
                saved = {name:p.detach().cpu().clone() for name,p in trainable.items()}
                save_inference_checkpoint(dict(weights=saved,repo=REPO,revision=REVISION,
                    expert='low_noise_transformer_2',source_sha256=report['source_sha256'],
                    training=report['training'],scope=report['scope']),args.out/'low_expert_adapter.pt')
                restored = load_verified(args.out/'low_expert_adapter.pt')
                if set(restored['weights']) != set(trainable):
                    raise ValueError('Adapter checkpoint layout changed')
                with torch.no_grad():
                    for name,p in trainable.items():
                        p.copy_(restored['weights'][name])
                report['saved_adapter_reload_exact'] = all(torch.equal(p.detach().cpu(),saved[name]) for name,p in trainable.items())
                del restored,saved,original,optimizer,trainable,frozen
                del prediction,loss,model_input,noisy,target,noise,clean,native_condition
                tidy()
            with stage('high_expert_gpu_load'):
                pipe.transformer.requires_grad_(False).eval().to(device)
            def callback(pipeline,index,timestep,values):
                report['denoise_progress'] = dict(branch=report.get('active_branch'),step=index+1,timestep=float(timestep))
                write_report(args.out/'report.json',report)
                return values
            for branch,enabled in (('original_baseline',False),('gaussian_adapted',True)):
                report['active_branch'] = branch
                control.enabled = enabled
                for module in low.modules():
                    if isinstance(module,LowRankLinear):
                        module.enabled = enabled
                # Native pipeline creates its own identical image-only condition.
                pipe.vae.to(device)
                with stage(branch+'_denoising'), torch.no_grad():
                    result = pipe(image=anchor_image,prompt_embeds=positive,negative_prompt_embeds=negative,
                        height=480,width=832,num_frames=args.frames,num_inference_steps=args.steps,
                        guidance_scale=3.5,guidance_scale_2=3.5,
                        generator=torch.Generator(device=device).manual_seed(args.seed),
                        output_type='latent',callback_on_step_end=callback).frames
                with stage(branch+'_latent_checkpoint'):
                    save_inference_checkpoint(dict(latent=result.detach().cpu()),args.out/(branch+'_latent.pt'))
                with stage(branch+'_vae_decode'), torch.no_grad():
                    mean = torch.tensor(pipe.vae.config.latents_mean,device=device).view(1,16,1,1,1)
                    std = torch.tensor(pipe.vae.config.latents_std,device=device).view(1,16,1,1,1)
                    pixels = pipe.vae.decode(result.float()*std+mean,return_dict=False)[0]
                    frames = ((pixels[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
                with stage(branch+'_mp4_encoding'):
                    video = args.out/(branch+'.mp4')
                    with imageio.get_writer(video,fps=16,codec='libx264',quality=8,macro_block_size=16) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                    report['videos'][branch] = dict(path=str(video),sha256=digest(video),frames=len(frames),
                        duration_seconds=len(frames)/16,modified_weights=enabled,review='not yet independently reviewed')
                del result,pixels,frames,mean,std
                tidy()
            report['worker_wall_seconds'] = time.monotonic()-started
            status('completed; outputs require independent visual review')
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error),quality_accepted=False)
        status('failed; checkpoints and prior artifacts preserved')
        raise
    return report


def self_test():
    """Small CPU contract checks only; never loads or downloads a video model."""
    import unittest
    import torch
    class Contracts(unittest.TestCase):
        def test_flow_sign(self):
            clean,noise=torch.tensor([2.]),torch.tensor([10.])
            noisy,target=flow_training_pair(clean,noise,.5)
            torch.testing.assert_close(noisy,torch.tensor([6.]))
            torch.testing.assert_close(noisy-.5*target,clean)
        def test_low_noise_only(self):
            for sigma in (0.,.9,1.):
                with self.assertRaises(ValueError):
                    flow_training_pair(torch.ones(1),torch.zeros(1),sigma)
        def test_profiles(self):
            validate_profile(33,20,2,600)
            for args in ((32,20,2,600),(33,2,2,600),(33,20,5,600),(33,20,2,601)):
                with self.assertRaises(ValueError):validate_profile(*args)
        def test_source_clock(self):
            self.assertEqual(source_indices(40,8,33)[-1],16)
            with self.assertRaises(ValueError):source_indices(4,8,33)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--part-labels',type=Path)
    parser.add_argument('--prompt',default=PROMPT)
    parser.add_argument('--frames',type=int,default=33)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--train-steps',type=int,default=2)
    parser.add_argument('--wall-seconds',type=int,default=600)
    parser.add_argument('--seed',type=int,default=103091)
    parser.add_argument('--self-test',action='store_true')
    options=parser.parse_args()
    if options.self_test:
        self_test()
    else:
        if any(getattr(options,key) is None for key in ('model','source','out')):
            parser.error('--model, --source and --out are required')
        run(options)
