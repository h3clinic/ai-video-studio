"""Two independent Wan generations for Gaussian reuse versus whole-scene regeneration.

Generation stage only. Composition/reconstruction is measured separately.
"""
import gc
import json
from pathlib import Path
import threading
import time
import torch
import psutil
from PIL import Image
from real_video.checkpoint_io import keep_windows_awake, load_verified, save_inference_checkpoint, digest
from real_video.wan_baseline import PROMPT, NEGATIVE, MODEL, REVISION

OUT=Path('artifacts/real_video/dog_scene_comparison/v1')
ROOT=Path('../../work/wan21_13b')
SOURCE=Path('artifacts/real_video/wan_cat_memory/v1_fullres/adapted.mp4')
PROMPTS={'dog':PROMPT.replace('orange tabby cat','golden retriever dog'),
         'joint':PROMPT+' A friendly golden retriever dog walks beside the cat on its right. Both animals are fully visible.'}


class Measure:
    def __enter__(self):
        self.stop=threading.Event(); self.peak_rss=0; self.min_available=psutil.virtual_memory().available
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); self.start=time.perf_counter()
        def sample():
            while not self.stop.is_set():
                self.peak_rss=max(self.peak_rss,psutil.Process().memory_info().rss)
                self.min_available=min(self.min_available,psutil.virtual_memory().available)
                self.stop.wait(.05)
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start()
        return self
    def __exit__(self,*_):
        torch.cuda.synchronize(); self.stop.set(); self.thread.join()
        self.result=dict(seconds=time.perf_counter()-self.start,peak_process_rss_bytes=self.peak_rss,
                         minimum_system_available_bytes=self.min_available,
                         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())


def free(): gc.collect(); torch.cuda.empty_cache()


def main():
    from transformers import UMT5EncoderModel, AutoTokenizer
    from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan, UniPCMultistepScheduler
    import imageio.v2 as imageio
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'generation_report.json').exists(): raise FileExistsError('Retain completed experiment')
    torch.set_num_threads(6)
    protocol=dict(model=MODEL,revision=REVISION,source_video=str(SOURCE),source_sha256=digest(SOURCE),
                  original_prompt=PROMPT,prompts=PROMPTS,negative_prompt=NEGATIVE,
                  height=480,width=832,frames=33,fps=16,steps=40,seed=71032,guidance_scale=6.,
                  role='A: reuse exact cat clip + independently generated dog; B: full-scene prompt regeneration',
                  limitations=['Different workflows, not equal outputs or proven equivalent quality','Gaussian capture and replay are reconstruction/composition, not new motion generation'],
                  common={},cases={})
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if previous!=protocol: raise ValueError('Protocol changed')
    else: protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    def record(message):
        print(message,flush=True)
        (OUT/'progress.json').write_text(json.dumps(dict(message=message,report=protocol),indent=2),encoding='utf-8')
    with keep_windows_awake(),torch.inference_mode():
        if not all((OUT/name/'prompt_embeddings.pt').exists() for name in PROMPTS):
            record('Loading text encoder with bounded CPU/GPU placement')
            with Measure() as m:
                encoder=UMT5EncoderModel.from_pretrained(ROOT/'text_encoder',torch_dtype=torch.bfloat16,local_files_only=True,
                    device_map='auto',max_memory={0:'8GiB','cpu':'4GiB'},low_cpu_mem_usage=True).eval()
                tokenizer=AutoTokenizer.from_pretrained(ROOT/'tokenizer',local_files_only=True)
                text_pipe=WanPipeline(tokenizer=tokenizer,text_encoder=encoder,transformer=None,vae=None,scheduler=None)
            protocol['common']['text_encoder_load']=m.result
            for name,prompt in PROMPTS.items():
                target=OUT/name; target.mkdir(exist_ok=True)
                if (target/'prompt_embeddings.pt').exists(): continue
                record(f'Encoding {name} prompt')
                with Measure() as m:
                    positive,negative=text_pipe.encode_prompt(prompt=prompt,negative_prompt=NEGATIVE,device=torch.device('cuda'),dtype=torch.bfloat16,max_sequence_length=512)
                protocol['cases'][name]=dict(text_encode=m.result)
                save_inference_checkpoint(dict(positive=positive.cpu(),negative=negative.cpu(),prompt=prompt),target/'prompt_embeddings.pt')
                del positive,negative
            del text_pipe,encoder,tokenizer; free()
        record('Loading original Wan transformer; no cat adapter applied to dog or joint baseline')
        with Measure() as m:
            transformer=WanTransformer3DModel.from_pretrained(ROOT/'transformer',torch_dtype=torch.bfloat16,local_files_only=True).eval().cuda()
        protocol['common']['transformer_load']=m.result
        for name in PROMPTS:
            target=OUT/name
            if (target/'generated_latent.pt').exists(): continue
            embeddings=load_verified(target/'prompt_embeddings.pt')
            scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,num_train_timesteps=1000,flow_shift=8.)
            pipe=WanPipeline(tokenizer=None,text_encoder=None,transformer=transformer,vae=None,scheduler=scheduler)
            record(f'Generating {name}: 832x480, 33 frames, 40 steps')
            with Measure() as m:
                latent=pipe(prompt_embeds=embeddings['positive'].cuda(),negative_prompt_embeds=embeddings['negative'].cuda(),
                            height=480,width=832,num_frames=33,num_inference_steps=40,guidance_scale=6.,
                            generator=torch.Generator(device='cuda').manual_seed(71032),output_type='latent').frames
            protocol['cases'].setdefault(name,{})['denoise']=m.result
            save_inference_checkpoint(dict(latent=latent.cpu(),prompt=PROMPTS[name]),target/'generated_latent.pt')
            (target/'denoise_metrics.json').write_text(json.dumps(m.result,indent=2),encoding='utf-8')
            del pipe,embeddings,latent; free()
        del transformer; free()
        with Measure() as m:
            vae=AutoencoderKLWan.from_pretrained(ROOT/'vae',torch_dtype=torch.float32,local_files_only=True).eval().cuda()
            vae.enable_tiling()
        protocol['common']['vae_load']=m.result
        mean=torch.tensor(vae.config.latents_mean,device='cuda').view(1,-1,1,1,1)
        std=torch.tensor(vae.config.latents_std,device='cuda').view(1,-1,1,1,1)
        for name in PROMPTS:
            target=OUT/name; record(f'Decoding {name}')
            latent=load_verified(target/'generated_latent.pt')['latent'].cuda().float()
            with Measure() as m:
                video=vae.decode(latent*std+mean,return_dict=False)[0]
                frames=((video[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
            protocol['cases'].setdefault(name,{})['decode']=m.result
            imageio.mimwrite(target/'video.mp4',frames,fps=16,codec='libx264',quality=8,macro_block_size=1)
            for index in [0,16,32]: Image.fromarray(frames[index]).save(target/f'frame_{index:03d}.png')
            del latent,video,frames; free()
        (OUT/'generation_report.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        record('Generation finished; Gaussian composition and quality comparison remain')


if __name__=='__main__': main()
