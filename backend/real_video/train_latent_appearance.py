"""Bounded real-DAVIS Wan latent-memory reader pilot, not 3D generation.

Preserves the v3 five-channel/LoRA adapter. Only a new appearance reader learns.
The memory contains first-frame VAE features; recorded future 2D tracks guide
planar Gaussian recall. No future RGB/latent is written into conditioning.
This tests a missing model connection, NOT autonomous Gaussian-state writing.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import threading
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import psutil
import torch

from .checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint
from .gaussian_program import write_json

BASE = Path('artifacts/real_video/wan_temporal_memory/v3')
MODEL = Path('../../work/wan21_13b')


def free():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def future_metrics(prediction, noisy, clean, noise, sigma):
    """Future-only latent error; not a perceptual/detail success measure."""
    estimate=noisy.float()-sigma*prediction.float()
    error=(estimate-clean.float())[:,:,1:]
    return dict(flow_mse=float((prediction.float()-(noise-clean).float())[:,:,1:].square().mean()),
                future_latent_mse=float(error.square().mean()),
                future_edge_error=float(sum(torch.diff(error,dim=d).square().mean() for d in (3,4))))


def resource_guard(out, wall_seconds):
    """Fail safely before/while this one process consumes scarce host memory.

    GPU-resident model loading, no text encoder, eight GiB free at entry,
    seven GiB process RSS cap, three GiB free floor. This preserves the
    persistent program's conservative entry policy.
    No other application's process or power settings are changed.
    """
    initial=psutil.virtual_memory().available
    battery=psutil.sensors_battery()
    if initial<8*2**30: raise RuntimeError('Training preflight requires 8 GiB available RAM')
    if battery and not battery.power_plugged and battery.percent<=20:
        raise RuntimeError('Low battery; no training started')
    if not 30<=wall_seconds<=600: raise ValueError('Wall budget must be30..600seconds')
    done=threading.Event(); start=time.monotonic(); process=psutil.Process()
    stats=dict(initial_available_ram_bytes=initial, peak_process_rss_bytes=process.memory_info().rss,
               minimum_available_ram_bytes=initial, max_wall_seconds=wall_seconds,
               resource_profile='GPU-resident reader pilot; no text encoder;8GiBentry,7GiBRSS,3GiBfloor',
               allowance_evidence='reader_v1 stopped at5.62GBRSS during loading with7.79GBfree; v2 increases loading allowance only',
               battery_percent=None if battery is None else battery.percent,
               power_plugged=None if battery is None else battery.power_plugged)
    def monitor():
        while not done.wait(.5):
            rss=process.memory_info().rss; available=psutil.virtual_memory().available
            stats['peak_process_rss_bytes']=max(stats['peak_process_rss_bytes'],rss)
            stats['minimum_available_ram_bytes']=min(stats['minimum_available_ram_bytes'],available)
            reason=('wall budget' if time.monotonic()-start>wall_seconds else
                    'process RSS cap' if rss>7*2**30 else 'available RAM floor' if available<3*2**30 else None)
            if reason:
                try: write_json(out/'resource_stop.json',dict(reason=reason,resources=stats,checkpoint_promoted=False))
                finally:
                    # Only this worker exits. Windows releases its wake request.
                    os._exit(75)
    thread=threading.Thread(target=monitor,daemon=True); thread.start()
    return done,stats


def run(out, steps=32, sample_steps=20, wall_seconds=600):
    if out.exists(): raise FileExistsError('Immutable experiment output required')
    if not 1<=steps<=64 or not 1<=sample_steps<=28: raise ValueError('Bounded training/sampling required')
    out.mkdir(parents=True)
    report=dict(scope='Trained Wan appearance reader fed by first-frame Gaussian latent memory',
                geometry_scope='Fixed-ID planar Gaussian memory with recorded future2Dtracks; not learned3Dgeometry',
                output_domain='Wan RGB; not generated Gaussian state',quality_accepted=False,
                checkpoint_promoted=False,training_steps=steps,sample_steps=sample_steps,
                sources={str(p):digest(p) for p in (BASE/'temporal_adapter.pt',BASE/'data.pt',BASE/'embeddings.pt')},
                code={p.name:digest(p) for p in [Path(__file__),Path('real_video/wan_latent_appearance.py'),Path('real_video/latent_track_memory.py')]},
                future_rgb_written_to_memory=False,training=[],validation={},samples={},
                baseline='v3_reader_disabled keeps the earlier trained v3 adapter; NOT original Wan',
                limitations=['Same small DAVIS development split; not an independent final test',
                    'First-frame appearance plus full recorded future tracks condition the whole clip',
                    'Corruption augmentation is feature noise/dropout, not a learned deviation archive',
                    'No learned correspondence/geometry writer; no generated-state feedback',
                    'Latent edge error and sharpness do not establish identity or anatomical quality'])
    def status(stage):
        report['stage']=stage; write_json(out/'report.json',report)
        print(stage,flush=True)
    status('Resource preflight')
    try: done,resources=resource_guard(out,wall_seconds)
    except Exception as error:
        report['error']=repr(error); status('Resource preflight rejected; no model loaded'); raise
    report['resources']=resources
    try:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline, UniPCMultistepScheduler
        from .wan_temporal_control import TemporalSpatialControl, attach_temporal_control
        from .wan_latent_appearance import install_latent_appearance
        from .latent_track_memory import build_anchor_memory_condition
        from .wan_cat_memory import install_lora
        from .prepare_animal_motion import WORK
        torch.set_num_threads(4); torch.manual_seed(103070)
        records=load_verified(BASE/'data.pt')['records']
        official={s:set((WORK/f'extracted/DAVIS/ImageSets/2017/{s}.txt').read_text().split()) for s in ('train','val')}
        train_names={'cat-girl','dog-agility','elephant','sheep'}
        for record in records:
            if record['split']=='train':
                assert record['sequence'] in train_names & official['train']
            else: assert record['sequence']=='cows' and 'cows' in official['val']
        assert len(records)==9 and all(r['split']=='train' for r in records[:8])
        report['data_split']=[dict(sequence=r['sequence'],start=r['start'],split=r['split']) for r in records]
        status('Encoding first-frame anchors separately from future training targets')
        vae=AutoencoderKLWan.from_pretrained(MODEL/'vae',torch_dtype=torch.float32,
            local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda'}).eval().requires_grad_(False)
        mean=torch.tensor(vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
        std=torch.tensor(vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
        targets=[]; conditions=[]; memory_snapshots=[]
        with torch.no_grad():
            for i,record in enumerate(records):
                rgb=record['rgb'].cuda().permute(3,0,1,2)[None].float()/127.5-1
                anchor=((vae.encode(rgb[:,:,:1]).latent_dist.mode()-mean)/std)[0,:,0].cpu()
                target=((vae.encode(rgb).latent_dist.mode()-mean)/std).to(torch.bfloat16)
                cond,meta,snapshot=build_anchor_memory_condition(anchor,record['tracks'][0],record['visibility'][0],
                    record['source_hashes'][0],return_snapshot=True)
                conditions.append(cond.cuda()); targets.append(target)
                memory_snapshots.append(snapshot)
                report.setdefault('memory',[]).append(dict(sequence=record['sequence'],**meta))
                print(f'Encoded {i+1}/9',flush=True)
        save_inference_checkpoint(dict(source_data_sha256=digest(BASE/'data.pt'),
            anchors_only=memory_snapshots,conditions=[c.cpu() for c in conditions],
            supervised_training_targets=[x.cpu() for x in targets],
            scope='Appearance banks contain source frame only; separate training targets must not be used as inference conditions'),out/'prepared.pt')
        report['prepared_checkpoint_sha256']=digest(out/'prepared.pt')
        del vae,mean,std,rgb,anchor,cond,snapshot,memory_snapshots; free()
        embed=load_verified(BASE/'embeddings.pt')
        positive=embed['positive'].cuda(); negative=embed['negative'].cuda(); del embed
        status('Loading GPU-resident Wan; restoring and freezing original v3 adapter')
        model=WanTransformer3DModel.from_pretrained(MODEL/'transformer',torch_dtype=torch.bfloat16,
            local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda'}).eval().requires_grad_(False)
        install_lora(model,4)
        old=TemporalSpatialControl(1536,channels=5).cuda(); attach_temporal_control(model,old)
        previous=load_verified(BASE/'temporal_adapter.pt')
        named={n:p for n,p in model.named_parameters() if p.requires_grad}
        if set(named)!=set(previous['weights']): raise ValueError('v3 weight layout mismatch')
        with torch.no_grad():
            for n,p in named.items(): p.copy_(previous['weights'][n])
        del named,previous; model.requires_grad_(False)
        frozen=[(p,p._version) for p in model.parameters()]
        reader=install_latent_appearance(model,old).cuda()
        params=list(reader.parameters())
        report['new_trainable_parameters']=sum(p.numel() for p in params)
        report['old_trainable_parameters_now_frozen']=3588032
        original={n:p.detach().cpu().clone() for n,p in reader.named_parameters()}
        rgbs=[r['condition'].cuda() for r in records]
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        model.enable_gradient_checkpointing()
        def predict(index,noise,sigma,mode='memory',corrupt=False):
            x=targets[index]
            cond=conditions[index]
            reader.enabled=mode!='v3_reader_disabled'
            if mode=='flipped_features':
                cond=cond.clone(); cond[:,:16]=cond[:,:16].flip(-1)
            if corrupt:
                cond=cond.clone()
                cond[:,:16] += torch.randn_like(cond[:,:16])*.03
                mask=(torch.rand_like(cond[:,:1])>.1)
                cond*=mask
            grid=(x.shape[2],x.shape[3]//2,x.shape[4]//2)
            old.bind(rgbs[index],grid); reader.bind(cond,grid)
            noisy=((1-sigma)*x+sigma*noise).requires_grad_(True)
            pred=model(hidden_states=noisy,timestep=sigma.reshape(1)*1000,
                encoder_hidden_states=positive,return_dict=False)[0]
            return pred,noisy
        def evaluate():
            result={}
            with torch.no_grad():
                for mode in ('v3_reader_disabled','memory','flipped_features'):
                    rows=[]
                    for j,level in enumerate((.3,.6,.9)):
                        x=targets[-1]; noise=torch.randn(x.shape,device='cuda',dtype=x.dtype,
                            generator=torch.Generator(device='cuda').manual_seed(103071+j))
                        sigma=torch.tensor(level,device='cuda')
                        pred,noisy=predict(8,noise,sigma,mode)
                        rows.append(future_metrics(pred,noisy,x,noise,sigma))
                    result[mode]={k:sum(row[k] for row in rows)/len(rows) for k in rows[0]}
            return result
        report['validation']['before']=evaluate(); status('Training only the latent-memory appearance reader')
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
        for step in range(steps):
            i=step%8; x=targets[i]; noise=torch.randn_like(x); sigma=torch.rand((),device='cuda')*.8+.1
            optimizer.zero_grad(set_to_none=True)
            pred,noisy=predict(i,noise,sigma,corrupt=step%4==0)
            error=(pred.float()-(noise-x).float())[:,:,1:]
            flow=error.square().mean()
            # Modest target-relative spatial derivatives, not artificial sharpening.
            clean_error=(noisy.float()-sigma*pred.float()-x.float())[:,:,1:]
            edge=sum(torch.diff(clean_error,dim=d).square().mean() for d in (3,4))
            loss=flow+.05*edge
            if not torch.isfinite(loss): raise ValueError('Nonfinite appearance loss')
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); optimizer.step()
            report['training'].append(dict(step=step+1,flow=float(flow.detach()),edge=float(edge.detach()),grad=float(grad)))
            if step%8==0 or step==steps-1: status(f'Appearance training {step+1}/{steps}')
        torch.cuda.synchronize()
        report['training_seconds']=time.perf_counter()-start
        report['training_peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        report['frozen_parameters_unchanged']=all(p._version==v for p,v in frozen)
        assert report['frozen_parameters_unchanged']
        report['reader_parameter_change_l2']=float(sum((p.detach().cpu()-original[n]).square().sum() for n,p in reader.named_parameters()).sqrt())
        report['validation']['after']=evaluate()
        save_inference_checkpoint(dict(reader=reader.state_dict(),base_adapter_sha256=digest(BASE/'temporal_adapter.pt'),
            protocol=report),out/'latent_reader.pt')
        restored=load_verified(out/'latent_reader.pt')
        assert restored['base_adapter_sha256']==digest(BASE/'temporal_adapter.pt')
        assert all(torch.equal(value.detach().cpu(),restored['reader'][name].cpu()) for name,value in reader.state_dict().items())
        reader.load_state_dict(restored['reader'],strict=True)
        report['saved_reader_reload_exact']=True
        del restored
        status('Reader weights saved; generating same-seed development ablation')
        del optimizer,params,frozen,original,pred,noisy,noise,x,flow,edge,loss,clean_error,error; free()
        model.disable_gradient_checkpointing()
        samples={}
        # Cows is development validation: no cherry-picked training cat demonstration.
        for mode in ('v3_reader_disabled','memory','flipped_features'):
            reader.enabled=mode!='v3_reader_disabled'
            sample_condition=conditions[-1]
            if mode=='flipped_features':
                sample_condition=sample_condition.clone()
                sample_condition[:,:16]=sample_condition[:,:16].flip(-1)
            reader.bind(sample_condition,(5,16,28))
            old.bind(rgbs[-1],(5,16,28))
            scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,
                num_train_timesteps=1000,flow_shift=8.)
            pipe=WanPipeline(tokenizer=None,text_encoder=None,transformer=model,vae=None,scheduler=scheduler)
            torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            with torch.inference_mode():
                samples[mode]=pipe(prompt_embeds=positive,negative_prompt_embeds=negative,height=256,width=448,
                    num_frames=17,num_inference_steps=sample_steps,guidance_scale=5.,
                    generator=torch.Generator(device='cuda').manual_seed(103075),output_type='latent').frames.cpu()
            torch.cuda.synchronize()
            report['samples'][mode]=dict(denoise_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            save_inference_checkpoint(dict(latent=samples[mode]),out/f'{mode}_latent.pt')
            del pipe; status(f'Sampled {mode}')
        del model,old,reader,positive,negative,targets,conditions,rgbs; free()
        status('Decoding diagnostic videos; no overlays or sharpening')
        import imageio.v2 as imageio
        vae=AutoencoderKLWan.from_pretrained(MODEL/'vae',torch_dtype=torch.float32,
            local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda'}).eval().requires_grad_(False)
        mean=torch.tensor(vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
        std=torch.tensor(vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
        images={'source':records[-1]['rgb'].numpy()}
        with torch.inference_mode():
            for mode,latent in samples.items():
                rgb=vae.decode(latent.cuda().float()*std+mean,return_dict=False)[0]
                images[mode]=((rgb[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
        sheet=Image.new('RGB',(448*3,280*4))
        for row,(mode,frames) in enumerate(images.items()):
            imageio.mimwrite(out/f'{mode}.mp4',frames,fps=12,codec='libx264',quality=8,macro_block_size=1)
            for col,index in enumerate((0,8,16)):
                im=Image.fromarray(frames[index]); im.save(out/f'{mode}_{index:03d}.png')
                sheet.paste(im,(col*448,row*280+24))
                ImageDraw.Draw(sheet).text((col*448+8,row*280+5),f'{mode} | frame {index} | development motion transfer',fill='white')
        sheet.save(out/'comparison.png')
        report['outputs']={str(p.name):digest(p) for p in out.glob('*.mp4')}
        report['finished']=True; status('Completed; visual review pending, quality not accepted')
    except Exception as error:
        report['error']=repr(error); report['traceback']=traceback.format_exc(); status('Failed; candidate retained')
        raise
    finally:
        done.set()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--sample-steps',type=int,default=20)
    parser.add_argument('--wall-seconds',type=int,default=600)
    args=parser.parse_args()
    with keep_windows_awake(): run(args.out,args.steps,args.sample_steps,args.wall_seconds)
