"""Remote-only frozen-checkpoint ablation; no optimizer, training, or downloads."""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import time

from cloud.wan_a14b_experiment import REVISION, REPO, NEGATIVE, digest, validate_model, write_report, deadline_guard

PROMPT = ('A realistic donkey lifts its head away from the wooden bowl while chewing an orange, '
          'then turns its muzzle slightly toward the camera. Preserve the same donkey and farm '
          'environment, natural fur and anatomy. Fixed camera, continuous natural action.')
SEEDS=(103501,103502)
BRANCHES=('original','lora_only','gaussian_memory')

def run(args):
    from cloud.a14b_preflight import require_dependencies
    from cloud.eval_preflight import validate_assets, require_host_memory, validate_adapter, effective_memory
    dependencies=require_dependencies()
    # Cheap CPU checks precede every baseline and all pretrained/GPU loading.
    assets=validate_assets(args.asset)
    import psutil
    cache_release=None
    manifest=getattr(args, 'download_manifest', None)
    manifest_pin=getattr(args, 'download_manifest_sha256', None)
    if bool(manifest) != bool(manifest_pin):
        raise ValueError('Cache release requires both verified download manifest and its SHA-256 pin')
    manifest_provenance='Caller-supplied SHA-256 pin'
    if manifest is None:
        # The owned remote coordinator writes this sibling with our pinned
        # downloader and checks status=verified before spawning this evaluator.
        # This is local pipeline provenance, NOT an independent publisher sig.
        manifest=args.out.parent/'download'/'download_manifest.json'
        manifest_pin=digest(manifest)
        manifest_provenance='Own completed downloader sibling manifest; locally bound SHA-256, not independent publisher authentication'
    if manifest is not None:
        from cloud.model_cache_release import release_model_cache
        before=effective_memory(int(psutil.virtual_memory().available))
        cache_release=release_model_cache(args.model, manifest, manifest_pin)
        cache_release['manifest_provenance']=manifest_provenance
        cache_release['effective_memory_before']=before
        # Advisory success is not admission: always take a fresh measurement.
        after=effective_memory(int(psutil.virtual_memory().available))
        cache_release['effective_memory_after']=after
        print(json.dumps(dict(model_cache_release=cache_release)),flush=True)
    resources=require_host_memory(int(psutil.virtual_memory().available))
    model,configs=validate_model(args.model)
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    import numpy as np
    from PIL import Image
    import imageio.v2 as imageio
    from diffusers import WanImageToVideoPipeline
    from real_video.checkpoint_io import load_verified
    from real_video.wan_part_control import install_part_adaptation
    from real_video.gaussian_latent_memory import GaussianLatentMemory
    from real_video.gaussian_part_protocol import GaussianPartProtocol
    from real_video.latent_track_memory import planar_gaussian_cache

    report=dict(schema_version=1,repo=REPO,revision=REVISION,config_sha256=configs,
        dependencies=dependencies,asset_preflight=assets,host_memory_preflight=resources,
        model_cache_release=cache_release,
        prompt=PROMPT,seeds=list(SEEDS),branches=list(BRANCHES),
        frames=33,fps=16,width=832,height=480,steps=20,training=False,
        anchor_sha256=digest(args.asset/'anchor.png'),
        adapter_sha256=digest(args.asset/'low_expert_adapter.pt'),
        memory_sha256=digest(args.asset/'gaussian_anchor.pt'),
        scope='New prompt and seeds; SAME training subject/anchor. Not subject-held-out evaluation.',
        baseline_scope='Original runs before adapters or Gaussian GPU memory. LoRA-only retains disabled control-head weights, but no bank/maps are bound until gaussian_memory.',
        memory_accounting_scope='GPU peaks cover the whole resident process. CPU RSS is sampled only at stage end, not peak RSS. Paired RGB comparison buffers are evaluation overhead, not model memory savings.',
        stages={},videos={},quality_accepted=False,native_gaussian_generation=False,
        limitations=['Only two seeds; fixed branch order, no statistical runtime claim.',
          'Planar first-frame latent memory, no 3D writer or generated-state update.',
          'No claim of source-independent or unseen-subject generalization.'])
    def status(name):
        report['stage']=name;write_report(args.out/'report.json',report);print(name,flush=True)
    def tidy():
        gc.collect();torch.cuda.empty_cache()
    @contextmanager
    def stage(name):
        status(name);torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        try: yield
        finally:
            torch.cuda.synchronize()
            report['stages'][name]=dict(seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                stage_end_process_rss_bytes=psutil.Process().memory_info().rss,
                retained_evaluation_rgb_bytes=sum(v.nbytes for v in paired.values()))
            write_report(args.out/'report.json',report)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('One remote CUDA GPU required')
    if torch.cuda.get_device_properties(0).total_memory<70*2**30:
        raise RuntimeError('Requires an 80GB-class GPU')
    device=torch.device('cuda'); control=None; low=None; paired={}
    try:
        with deadline_guard(args.out,report,600) as started:
            with stage('model_load'):
                pipe=WanImageToVideoPipeline.from_pretrained(model,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True)
                pipe.vae.to(dtype=torch.float32)
                for m in (pipe.transformer,pipe.transformer_2,pipe.text_encoder,pipe.vae): m.requires_grad_(False).eval()
                image=Image.open(args.asset/'anchor.png').convert('RGB')
                if image.size!=(832,480): raise ValueError('Anchor dimensions mismatch')
            with stage('prompt_encode'):
                pipe.text_encoder.to(device)
                with torch.no_grad():
                    positive,negative=pipe.encode_prompt(prompt=PROMPT,negative_prompt=NEGATIVE,do_classifier_free_guidance=True,device=device,num_videos_per_prompt=1)
                positive,negative=positive.to(torch.bfloat16),negative.to(torch.bfloat16)
                pipe.text_encoder.to('cpu');tidy()
            with stage('gpu_model_transfer'):
                pipe.transformer.to(device);pipe.transformer_2.to(device);pipe.vae.to(device)
            original_prepare=pipe.prepare_latents
            timings=[]
            def timed_prepare(*a,**k):
                torch.cuda.synchronize();t=time.perf_counter()
                value=original_prepare(*a,**k)
                torch.cuda.synchronize();timings.append(time.perf_counter()-t)
                return value
            pipe.prepare_latents=timed_prepare
            for branch in BRANCHES:
                if branch=='lora_only':
                    with stage('load_frozen_adapter'):
                        low=pipe.transformer_2
                        control,handles,details=install_part_adaptation(low,intent_dim=1,blocks=(0,10,20,30),hidden=64,rank=4)
                        saved=load_verified(args.asset/'low_expert_adapter.pt')
                        validate_adapter(saved)
                        if saved['repo']!=REPO or saved['revision']!=REVISION: raise ValueError('Adapter backbone mismatch')
                        parameters={n:p for n,p in low.named_parameters() if p.requires_grad}
                        if set(parameters)!=set(saved['weights']): raise ValueError('Adapter parameter layout mismatch')
                        if any(p.shape != saved['weights'][n].shape for n,p in parameters.items()):
                            raise ValueError('Adapter parameter shape mismatch; broadcasting forbidden')
                        with torch.no_grad():
                            for n,p in parameters.items(): p.copy_(saved['weights'][n])
                        low.requires_grad_(False).eval()
                        report['adapter']=details
                        del saved,parameters;tidy()
                if branch=='gaussian_memory':
                    with stage('bind_frozen_gaussian_memory'):
                        snapshot=load_verified(args.asset/'gaussian_anchor.pt')
                        if snapshot['asset_digest']!=report['anchor_sha256']: raise ValueError('Wrong Gaussian anchor')
                        ids=snapshot['ids'];generations=snapshot['generations'];h,w=60,104
                        if len(ids)!=h*w: raise ValueError('Unexpected memory grid')
                        memory=GaussianLatentMemory(ids,16,snapshot['asset_digest'],device=device,generations=generations,max_observation_mass=snapshot['max_observation_mass'])
                        memory.restore(snapshot)
                        yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
                        cache=planar_gaussian_cache(torch.stack((xx,yy),-1).reshape(-1,2).float(),torch.ones(len(ids)),ids,h,w,generations=generations,asset_digest=snapshot['asset_digest'])
                        protocol=GaussianPartProtocol(memory,ids=ids,generations=generations,part_ids=torch.zeros_like(ids))
                        control.bind_memory_frames(protocol,[cache]*9,h,w,torch.ones(1,1,1,device=device),(9,30,52))
                        report['memory_bytes']=memory.memory_bytes()
                        del snapshot;tidy()
                if control is not None: control.enabled=(branch=='gaussian_memory')
                for seed in SEEDS:
                    name=f'{branch}_{seed}';timings.clear()
                    with stage(name+'_pipeline'),torch.no_grad():
                        latent=pipe(image=image,prompt_embeds=positive,negative_prompt_embeds=negative,height=480,width=832,num_frames=33,num_inference_steps=20,
                            guidance_scale=3.5,guidance_scale_2=3.5,generator=torch.Generator(device=device).manual_seed(seed),output_type='latent').frames
                    report['stages'][name+'_pipeline']['native_condition_prepare_seconds']=sum(timings)
                    report['stages'][name+'_pipeline']['scope']='Includes native image-condition encoding and sampling; no RGB decode or MP4 encode.'
                    with stage(name+'_decode'),torch.no_grad():
                        mean=torch.tensor(pipe.vae.config.latents_mean,device=device).view(1,16,1,1,1)
                        std=torch.tensor(pipe.vae.config.latents_std,device=device).view(1,16,1,1,1)
                        pixels=pipe.vae.decode(latent.float()*std+mean,return_dict=False)[0]
                        frames=((pixels[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
                    with stage(name+'_encode'):
                        path=args.out/(name+'.mp4')
                        with imageio.get_writer(path,fps=16,codec='libx264',quality=8,macro_block_size=16) as writer:
                            for frame in frames: writer.append_data(frame)
                        report['videos'][name]=dict(file=path.name,sha256=digest(path),frames=len(frames),review='pending')
                        if branch=='lora_only': paired[seed]=frames.copy()
                        elif branch=='gaussian_memory':
                            report['videos'][name]['mean_abs_pixel_difference_vs_lora_only']=float(np.abs(frames.astype(np.float32)-paired.pop(seed).astype(np.float32)).mean())
                            report['videos'][name]['difference_is_not_quality_score']=True
                    del latent,pixels,frames,mean,std;tidy()
            report['worker_seconds']=time.monotonic()-started;status('completed; visual review required')
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));status('failed');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--asset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--download-manifest',type=Path)
    p.add_argument('--download-manifest-sha256')
    run(p.parse_args())
