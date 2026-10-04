"""Paired original-resolution inference and Gaussian-conditioning ablation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from real_video.wan_cat_memory import GaussianMemory, LowRankLinear, install_lora, free
from real_video.checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--frames',type=int,default=33)
    p.add_argument('--height',type=int,default=480)
    p.add_argument('--width',type=int,default=832)
    p.add_argument('--modes',nargs='+',choices=['base','adapted','no_memory'],default=['base','adapted','no_memory'])
    p.add_argument('--memory-guidance',choices=['shared','positive-only'],default='positive-only')
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    from diffusers import WanTransformer3DModel, WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler
    torch.set_num_threads(6)
    checkpoint=load_verified(args.adapter)
    root=Path('../../work/wan21_13b')
    text=load_verified('artifacts/real_video/wan_baseline/cat_seed_421001/prompt_embeddings.pt')
    positive=text['positive'].cuda().bfloat16(); negative=text['negative'].cuda().bfloat16()
    report=dict(adapter_sha256=digest(args.adapter),scope='RGB Wan personalization. Gaussian memory conditioning ablation, not Gaussian motion generation',
                height=args.height,width=args.width,frames=args.frames,steps=args.steps,seed=71032,
                quality_accepted=False,samples={},memory_guidance=args.memory_guidance,
                memory_note='Peak PyTorch allocated VRAM; sequential warm runs, not a controlled speed benchmark')
    def record(): (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    record()
    with keep_windows_awake(),torch.inference_mode():
        model=WanTransformer3DModel.from_pretrained(root/'transformer',torch_dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False).cuda()
        names=install_lora(model,checkpoint['report']['rank'])
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        if expected!=set(checkpoint['lora']): raise ValueError('Adapter keys do not match this model')
        for name,param in model.named_parameters():
            if name in checkpoint['lora']: param.copy_(checkpoint['lora'][name].to(param))
        memory=GaussianMemory(checkpoint['memory']['features']).cuda().eval()
        memory.load_state_dict(checkpoint['memory'])
        real_condition=memory(positive)
        negative_condition=memory(negative) if args.memory_guidance=='shared' else negative
        saved_features=memory.features.clone()
        memory.features.zero_()
        empty_condition=memory(positive)
        memory.features.copy_(saved_features)
        report['memory_residual_rms']=(real_condition.float()-positive.float()).square().mean().sqrt().item()
        report['real_vs_zero_geometry_condition_rms']=(real_condition.float()-empty_condition.float()).square().mean().sqrt().item()
        samples={}
        for mode in args.modes:
            for module in model.modules():
                if isinstance(module,LowRankLinear): module.enabled=mode!='base'
            condition=real_condition if mode=='adapted' else positive
            scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,num_train_timesteps=1000,flow_shift=8.)
            pipe=WanPipeline(tokenizer=None,text_encoder=None,transformer=model,vae=None,scheduler=scheduler)
            print(f'Generating {mode} at {args.width}x{args.height}, {args.frames} frames',flush=True)
            free(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
            neg_condition=negative_condition if mode=='adapted' else negative
            samples[mode]=pipe(prompt_embeds=condition,negative_prompt_embeds=neg_condition,height=args.height,width=args.width,
                               num_frames=args.frames,num_inference_steps=args.steps,guidance_scale=6.,
                               generator=torch.Generator(device='cuda').manual_seed(71032),output_type='latent').frames.cpu()
            torch.cuda.synchronize()
            report['samples'][mode]=dict(seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            save_inference_checkpoint(dict(latent=samples[mode]),args.out/f'{mode}_latent.pt')
            record(); del pipe
        if 'adapted' in samples and 'no_memory' in samples:
            report['adapted_vs_no_memory_latent_rmse']=(samples['adapted'].float()-samples['no_memory'].float()).square().mean().sqrt().item()
        del model, memory, module, param, real_condition, empty_condition
        free()
        vae=AutoencoderKLWan.from_pretrained(root/'vae',torch_dtype=torch.float32,local_files_only=True).eval().cuda()
        vae.enable_tiling()
        mean=torch.tensor(vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
        std=torch.tensor(vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
        import imageio.v2 as imageio
        sheet=Image.new('RGB',(args.width*3,args.height*len(samples)))
        for row,(mode,latent) in enumerate(samples.items()):
            video=vae.decode(latent.cuda().float()*std+mean,return_dict=False)[0]
            frames=((video[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
            imageio.mimwrite(args.out/f'{mode}.mp4',frames,fps=16,codec='libx264',quality=8,macro_block_size=1)
            for col,index in enumerate([0,args.frames//2,args.frames-1]):
                image=Image.fromarray(frames[index]); image.save(args.out/f'{mode}_{index:03d}.png')
                sheet.paste(image,(col*args.width,row*args.height))
            del video
        sheet.save(args.out/'comparison.jpg')
        record()
        print('Paired inference and ablation complete; visual review required',flush=True)


if __name__=='__main__': main()
