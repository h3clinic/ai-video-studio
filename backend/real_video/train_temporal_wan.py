"""Real-video temporal adapter pilot; motion-guided RGB generation, not 3D recovery.

Supervised motion comes from recorded DAVIS sequences. Supplying their future
tracks at inference is motion transfer, never novel motion generation. The same
5-channel interface can consume projected moving Gaussian appearance + alpha.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from real_video.checkpoint_io import load_verified, save_inference_checkpoint, keep_windows_awake, digest
from real_video.wan_cat_memory import install_lora, LowRankLinear
from real_video.prepare_animal_motion import WORK, TRAIN, VAL


def free():
    gc.collect(); torch.cuda.empty_cache()


def temporal_flow_loss(prediction, noisy, clean, noise, sigma):
    """Flow matching plus supervised CLEAN temporal differences, not motion size.

    Diffusion time sigma is separate from video time (tensor dimension2).
    Maximizing frame differences alone would reward flicker; this penalizes
    disagreement with the observed temporal changes, including static regions.
    """
    if prediction.shape != clean.shape or noisy.shape != clean.shape or noise.shape != clean.shape:
        raise ValueError('Flow tensors must have matching shapes')
    if clean.ndim != 5 or clean.shape[2] < 2:
        raise ValueError('A video with multiple latent times is required')
    flow=(prediction.float()-(noise-clean).float()).square().mean()
    estimate=noisy.float()-sigma*prediction.float()
    motion=(torch.diff(estimate,dim=2)-torch.diff(clean.float(),dim=2)).square().mean()
    return flow+.25*motion,flow,motion


def video_conditions(frames, masks):
    from real_video.wan_temporal_control import transport_first_features, wan_temporal_average
    h, w = frames.shape[1:3]
    yy, xx = np.meshgrid(np.arange(h//8)*8+3.5, np.arange(w//8)*8+3.5, indexing='ij')
    positions = np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float32)
    tracks = [positions.copy()]; visibility = [np.ones(len(positions), np.float32)]
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flow_engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    def at(array, xy):
        return cv2.remap(array, xy[:, 0, None], xy[:, 1, None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[:, 0]
    for i in range(1, len(frames)):
        forward = flow_engine.calc(gray[i-1], gray[i], None)
        backward = flow_engine.calc(gray[i], gray[i-1], None)
        movement = at(forward, positions)
        positions = positions+movement
        error = np.linalg.norm(movement+at(backward, positions), axis=-1)
        valid = (positions[:,0]>=0)&(positions[:,0]<w)&(positions[:,1]>=0)&(positions[:,1]<h)
        visible = visibility[-1]*np.exp(-error**2/9)*valid
        tracks.append(positions.copy()); visibility.append(visible.astype(np.float32))
    canonical = torch.cat([torch.from_numpy(frames[0].copy()).permute(2,0,1).float()/255,
                           torch.from_numpy(masks[0].copy()).float()[None]], 0)[None]
    canonical = F.avg_pool2d(canonical, 8)
    # avg_pool8 pixel cell j covers centers8j..8j+7, whose center is8j+3.5.
    tracks = (torch.from_numpy(np.stack(tracks))[None]-3.5)/8
    visible = torch.from_numpy(np.stack(visibility))[None]
    volume, occupancy = transport_first_features(canonical, tracks, visibility=visible)
    return wan_temporal_average(torch.cat([volume, occupancy.clamp(0,1)], 1)), tracks, visible


def prepare(out):
    root = WORK/'extracted/DAVIS'
    # Official train/validation membership is asserted, not inferred from names.
    selected = [('cat-girl',0),('dog-agility',0),('elephant',0),('sheep',0),
                ('cat-girl',8),('dog-agility',8),('elephant',8),('sheep',8),('cows',0)]
    official = {s:set((root/f'ImageSets/2017/{s}.txt').read_text().split()) for s in ['train','val']}
    records = []
    for name, start in selected:
        split = 'train' if name in TRAIN else 'validation'
        assert name in official['train' if split=='train' else 'val']
        paths = sorted((root/'JPEGImages/480p'/name).glob('*.jpg'))[start:start+17]
        if len(paths)!=17: raise ValueError(f'Insufficient frames for {name}')
        frames = np.stack([cv2.resize(np.asarray(Image.open(p).convert('RGB')), (448,256), interpolation=cv2.INTER_AREA) for p in paths])
        masks=[]
        for p in paths:
            label=np.asarray(Image.open(root/'Annotations/480p'/name/f'{p.stem}.png'))
            mask=(label==2) if name=='cat-girl' else (label>0)
            masks.append(cv2.resize(mask.astype(np.float32),(448,256),interpolation=cv2.INTER_NEAREST))
        condition, tracks, visibility = video_conditions(frames, np.stack(masks))
        records.append(dict(sequence=name,start=start,split=split,condition=condition,
                            rgb=torch.from_numpy(frames),tracks=tracks,visibility=visibility,
                            source_hashes=[digest(p) for p in paths]))
        print(f'Prepared {name}:{start} {split} actual video sequence',flush=True)
    save_inference_checkpoint(dict(records=records,provenance=json.loads((WORK/'provenance.json').read_text())),out/'data.pt')
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--skip-samples',action='store_true')
    parser.add_argument('--control-lr',type=float,default=1e-4)
    parser.add_argument('--attention-lr',type=float,default=1e-4)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    from transformers import UMT5EncoderModel, AutoTokenizer
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline, UniPCMultistepScheduler
    from real_video.wan_temporal_control import TemporalSpatialControl, attach_temporal_control
    model_dir=Path('../../work/wan21_13b')
    torch.set_num_threads(6); cv2.setNumThreads(4); torch.manual_seed(73003)
    report=dict(scope='Recorded-motion-conditioned Wan RGB generation; NOT novel 3D Gaussian motion',
                quality_accepted=False,train_sequences=['cat-girl','dog-agility','elephant','sheep'],
                validation_sequences=['cows'],frames=17,height=256,width=448,steps=args.steps,
                output_fps=12,source_frame_stride=1,
                fps_note='12 fps diagnostic playback; DAVIS JPEG sequence timing not verified here',
                architecture='5-channel canonical RGBA feature transport plus occupancy at every latent time; attention LoRA',
                training=[],validation={},samples={},no_output_compositing=True,
                limits=['Small real-video subset, not foundation training','Future recorded tracks guide validation generation',
                        'No 3D ground truth in DAVIS','No efficiency gain established'])
    def status(message):
        print(message,flush=True)
        (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with keep_windows_awake():
        records=prepare(args.out)
        status('Encoding generic animal-motion prompt, without forcing the validation species')
        prompt='An animal moving naturally in a real environment. Continuous shot, coherent body and limb movement.'
        negative='blurry, distorted anatomy, extra limbs, flickering, static image, subtitles, watermark'
        if args.resume_from:
            cached=load_verified(args.resume_from/'embeddings.pt')
            if cached['prompt']!=prompt: raise ValueError('Resume prompt mismatch')
            positive,negative_emb=cached['positive'].cuda(),cached['negative'].cuda()
            del cached
        else:
            encoder=UMT5EncoderModel.from_pretrained(model_dir/'text_encoder',torch_dtype=torch.bfloat16,
                local_files_only=True,device_map='auto',max_memory={0:'8GiB','cpu':'4GiB'},low_cpu_mem_usage=True).eval()
            tokenizer=AutoTokenizer.from_pretrained(model_dir/'tokenizer',local_files_only=True)
            text_pipe=WanPipeline(tokenizer=tokenizer,text_encoder=encoder,transformer=None,vae=None,scheduler=None)
            with torch.no_grad():
                positive,negative_emb=text_pipe.encode_prompt(prompt=prompt,negative_prompt=negative,
                    device=torch.device('cuda'),dtype=torch.bfloat16,max_sequence_length=512)
            del encoder,tokenizer,text_pipe
        save_inference_checkpoint(dict(positive=positive.cpu(),negative=negative_emb.cpu(),prompt=prompt),args.out/'embeddings.pt')
        free()
        status('Encoding full 17-frame training sequences (not isolated stills)')
        vae=AutoencoderKLWan.from_pretrained(model_dir/'vae',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).cuda()
        mean=torch.tensor(vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
        std=torch.tensor(vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
        latents=[]
        with torch.no_grad():
            for record in records:
                rgb=record['rgb'].cuda().permute(3,0,1,2)[None].float()/127.5-1
                latents.append(((vae.encode(rgb).latent_dist.mode()-mean)/std).to(torch.bfloat16))
        del vae,mean,std,rgb
        free()
        status('Attaching new temporal control and attention LoRA to original Wan weights')
        model=WanTransformer3DModel.from_pretrained(model_dir/'transformer',torch_dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False).cuda()
        install_lora(model,4)
        control=TemporalSpatialControl(1536,channels=5).cuda()
        attach_temporal_control(model,control)
        if args.resume_from:
            path=args.resume_from/'temporal_adapter.pt'
            previous=load_verified(path)
            named={n:p for n,p in model.named_parameters() if p.requires_grad}
            if set(named)!=set(previous['weights']): raise ValueError('Resume weight names mismatch')
            with torch.no_grad():
                for n,p in named.items(): p.copy_(previous['weights'][n])
            report['initial_checkpoint_sha256']=digest(path)
            report['resume_scope']='Continue adapter weights with fresh optimizer; not exact optimizer resume'
            del previous,named
        params=[p for p in model.parameters() if p.requires_grad]
        report['trainable_parameters']=sum(p.numel() for p in params)
        frozen=[(p,p._version) for p in model.parameters() if not p.requires_grad]
        control_params=list(control.parameters())
        control_ids={id(p) for p in control_params}
        attention_params=[p for p in params if id(p) not in control_ids]
        optimizer=torch.optim.AdamW([dict(params=control_params,lr=args.control_lr),
                                     dict(params=attention_params,lr=args.attention_lr)],weight_decay=.01)
        report['learning_rates']=dict(control=args.control_lr,attention=args.attention_lr)
        model.enable_gradient_checkpointing()
        conditions=[d['condition'].cuda() for d in records]
        def predict(x,noise,sigma,condition):
            control.bind(condition,(x.shape[2],x.shape[3]//2,x.shape[4]//2))
            noisy=((1-sigma)*x+sigma*noise).requires_grad_(True)
            pred=model(hidden_states=noisy,timestep=sigma.reshape(1)*1000,encoder_hidden_states=positive,return_dict=False)[0]
            return pred,noisy
        def validation():
            x=latents[-1]
            noise=torch.randn(x.shape,device='cuda',dtype=x.dtype,generator=torch.Generator(device='cuda').manual_seed(73004))
            sigma=torch.tensor(.6,device='cuda')
            with torch.no_grad():
                pred,_=predict(x,noise,sigma,conditions[-1])
                return float((pred.float()-(noise-x).float()).square().mean())
        report['validation']['before']=validation()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
        for step in range(args.steps):
            i=step%8; x=latents[i]; noise=torch.randn_like(x); sigma=torch.rand((),device='cuda')*.9+.05
            optimizer.zero_grad(set_to_none=True)
            pred,noisy=predict(x,noise,sigma,conditions[i])
            loss,flow,motion=temporal_flow_loss(pred,noisy,x,noise,sigma)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); optimizer.step()
            report['training'].append(dict(step=step+1,flow=float(flow.detach()),temporal_difference=float(motion.detach()),gradient_norm=float(grad)))
            if step%8==0 or step==args.steps-1: status(f'Temporal sequence training {step+1}/{args.steps}: {float(loss.detach()):.5f}')
        torch.cuda.synchronize()
        report['training_seconds']=time.perf_counter()-start
        report['training_peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        report['frozen_weights_unchanged']=all(p._version==v for p,v in frozen)
        assert report['frozen_weights_unchanged']
        report['validation']['after']=validation()
        state={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        save_inference_checkpoint(dict(weights=state,report=report,control_channels=5,rank=4),args.out/'temporal_adapter.pt')
        if args.skip_samples:
            status('Weights saved; inference is a separate explicitly recorded stage')
            return
        del optimizer,params,frozen,latents,pred,noisy,noise,x,flow,motion,loss,state
        free(); model.disable_gradient_checkpointing()
        samples={}
        for mode in ['base','static','tracked']:
            control.enabled=mode!='base'
            for m in model.modules():
                if isinstance(m,LowRankLinear): m.enabled=mode!='base'
            condition=conditions[-1] if mode=='tracked' else conditions[-1][:,:,:1].expand(-1,-1,5,-1,-1)
            control.bind(condition,(5,16,28))
            scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,num_train_timesteps=1000,flow_shift=8.)
            pipe=WanPipeline(tokenizer=None,text_encoder=None,transformer=model,vae=None,scheduler=scheduler)
            status(f'Generating same-seed {mode} validation: recorded-motion guidance, not new motion')
            torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            with torch.inference_mode():
                samples[mode]=pipe(prompt_embeds=positive,negative_prompt_embeds=negative_emb,height=256,width=448,
                    num_frames=17,num_inference_steps=28,guidance_scale=5.,generator=torch.Generator(device='cuda').manual_seed(73005),output_type='latent').frames.cpu()
            torch.cuda.synchronize()
            report['samples'][mode]=dict(denoise_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
            save_inference_checkpoint(dict(latent=samples[mode]),args.out/f'{mode}_latent.pt')
            del pipe
        del model,control,positive,negative_emb,m
        free()
        status('Decoding and exporting honest source/static/tracked comparison')
        vae=AutoencoderKLWan.from_pretrained(model_dir/'vae',torch_dtype=torch.float32,local_files_only=True).eval().cuda()
        mean=torch.tensor(vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
        std=torch.tensor(vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
        import imageio.v2 as imageio
        images={'source':records[-1]['rgb'].numpy()}
        with torch.inference_mode():
            for mode,latent in samples.items():
                decoded=vae.decode(latent.cuda().float()*std+mean,return_dict=False)[0]
                images[mode]=((decoded[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
        sheet=Image.new('RGB',(448*3,256*4))
        for row,(mode,frames) in enumerate(images.items()):
            imageio.mimwrite(args.out/f'{mode}.mp4',frames,fps=12,codec='libx264',quality=8,macro_block_size=1)
            for col,i in enumerate([0,8,16]):
                image=Image.fromarray(frames[i]); image.save(args.out/f'{mode}_{i:03d}.png'); sheet.paste(image,(col*448,row*256))
        sheet.save(args.out/'comparison.jpg')
        status('Finished; no automatic acceptance of motion/identity quality')


if __name__=='__main__': main()
