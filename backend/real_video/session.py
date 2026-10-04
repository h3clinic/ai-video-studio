"""Reusable, source-free Gaussian video generation; independent clips, not recurrence."""
import argparse
import json
from pathlib import Path
import statistics
import threading
import time
import torch
from PIL import Image,ImageDraw
from .checkpoint_io import digest,load_verified,save_inference_checkpoint,keep_windows_awake
from .latent import inference_decoder
from .model import GaussianVideoDenoiser,sample
from .representation import render_fields_chunked
from .evaluate import save_video,to_frames


class GaussianVideoSession:
    """Load once, capture once, produce independent clips from fresh seeds.

    Returns CPU-owned tensors, so subsequent calls cannot overwrite old results.
    Calls and close are serialized because CUDA graphs use mutable static buffers.
    No clip history, teacher, training examples or analysis encoder is retained.
    A fixed 8-frame latent is NOT persistent physical state for a long video.
    """
    def __init__(self,checkpoint,device='cuda',cuda_graph=True,precast_weights=True):
        self._lock=threading.Lock(); self.closed=False
        self.device=torch.device(device)
        if self.device.type!='cuda' and (cuda_graph or precast_weights):
            raise ValueError('Graph/precast paths require CUDA')
        started=time.perf_counter()
        packet=load_verified(checkpoint)
        self.checkpoint_sha256=digest(checkpoint)
        self.labels=tuple(packet['labels']); self.shape=tuple(packet['latent_shape'])
        self.distilled_guidance=packet.get('distilled_guidance')
        self.guidance=1.5 if self.distilled_guidance is None else self.distilled_guidance
        self.prior=GaussianVideoDenoiser(**packet['config']).eval().requires_grad_(False)
        self.prior.load_state_dict(packet['model'])
        self.decoder=inference_decoder(packet['codec_config'],packet['codec'])
        if precast_weights:
            from .inference_precision import precast_autocast_weights
            precast_autocast_weights(self.prior); precast_autocast_weights(self.decoder)
        self.prior=self.prior.to(self.device); self.decoder=self.decoder.to(self.device)
        self.stats={key:packet[key].to(self.device) for key in ('mean','std','latent_mean','latent_std')}
        self.runner=None
        if cuda_graph:
            from .graph_sampling import GraphedVelocity
            self.runner=GraphedVelocity(self.prior,1,self.shape,self.guidance,self.distilled_guidance)
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        self.setup_seconds=time.perf_counter()-started
        self.graph_setup_seconds=self.runner.setup_seconds if self.runner else 0.
        self.cuda_graph=cuda_graph; self.precast_weights=precast_weights

    @torch.no_grad()
    def generate(self,label,seed,steps=50):
        with self._lock:
            if self.closed: raise RuntimeError('Generation session is closed')
            if label not in self.labels: raise ValueError(f'Unknown label: {label}')
            if not isinstance(seed,int) or not 0<=seed<2**63: raise ValueError('Use a nonnegative 63-bit seed')
            if not isinstance(steps,int) or not 2<=steps<=1000: raise ValueError('Use 2..1000 sampling steps')
            labels=torch.tensor([self.labels.index(label)],device=self.device)
            normalized=sample(self.prior,labels,seed,steps=steps,guidance=self.guidance,shape=self.shape,
                              distilled_guidance=self.distilled_guidance,velocity_fn=self.runner)
            z=(normalized*self.stats['latent_std']+self.stats['latent_mean']).half().float()
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                fields=self.decoder.decode(z).float()
            fields=fields*self.stats['std']+self.stats['mean']
            video=render_fields_chunked(fields,frame_chunk=1)
            return dict(video=video[0].cpu(),fields=fields[0].cpu(),latent=z[0].half().cpu(),
                        label=label,seed=seed,steps=steps,checkpoint_sha256=self.checkpoint_sha256)

    def close(self):
        with self._lock:
            self.runner=None; self.prior=None; self.decoder=None; self.stats=None; self.closed=True

    def __enter__(self): return self
    def __exit__(self,*args): self.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--first-seed',type=int,default=3010000)
    args=parser.parse_args()
    if not 1<=args.count<=200: parser.error('Use 1..200 independent clips')
    if not 0<=args.first_seed<2**63-args.count: parser.error('Invalid first seed')
    args.out.mkdir(parents=True,exist_ok=False); torch.set_num_threads(2)
    records=[]; wall=time.perf_counter()
    with GaussianVideoSession(args.checkpoint) as session:
        for i in range(2): session.generate(session.labels[0],3009000+i)
        # Latent and field packets are written per clip, never accumulated on GPU.
        setup=session.setup_seconds
        sheet=Image.new('RGB',(512,args.count*148),(18,22,29))
        for i in range(args.count):
            label=session.labels[i%len(session.labels)]
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
            result=session.generate(label,args.first_seed+i)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated(); resident=torch.cuda.memory_allocated()
            stem=f'clip_{i:03d}_{label}'
            save_video(result['video'],args.out/(stem+'.mp4'),'NOISE -> GAUSSIANS | '+label)
            packet={k:v for k,v in result.items() if k!='video'}
            save_inference_checkpoint(packet,args.out/(stem+'.pt'))
            for j,t in enumerate((0,2,4,7)):
                frame=Image.fromarray(to_frames(result['video'])[t]).resize((128,128),Image.Resampling.NEAREST)
                sheet.paste(frame,(128*j,148*i+20))
            ImageDraw.Draw(sheet).text((4,148*i+3),f'{i:02d} {label} | seed {result["seed"]}',fill='white')
            record=dict(index=i,label=label,seed=result['seed'],generation_seconds=elapsed,
                        peak_cuda_allocated_bytes=peak,between_calls_cuda_allocated_bytes=resident,
                        video=stem+'.mp4',gaussian_packet=stem+'.pt')
            records.append(record)
            # Flush progress so interrupted runs still identify completed outputs.
            (args.out/'progress.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(json.dumps(record),flush=True)
            del result,packet
        sheet.save(args.out/'contact_sheet.png')
        report=dict(checkpoint_sha256=session.checkpoint_sha256,labels=list(session.labels),count=args.count,
                    setup_seconds=setup,graph_setup_seconds=session.graph_setup_seconds,records=records,
                    median_generation_seconds=statistics.median(r['generation_seconds'] for r in records),
                    total_wall_seconds_including_setup_warmup_encoding_and_saving=time.perf_counter()-wall,
                    peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in records),
                    between_calls_memory_span_bytes=max(r['between_calls_cuda_allocated_bytes'] for r in records)-min(r['between_calls_cuda_allocated_bytes'] for r in records),
                    independent_clips=True,trained_this_run=False,
                    limits=['20 independent short clips do not form a coherent long video.',
                            'GPU memory checks exclude CPU results, metadata, driver/context and file storage.',
                            'Generation timing includes CPU field/code/video transfer, not encoding or saving.',
                            'No new quality selection or external benchmark; every scheduled case is retained.'])
        (args.out/'session.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    with keep_windows_awake(): main()
