"""SOURCE-CONDITIONED Gaussian trajectory fitting and replay; never noise generation.

The source video is used only by fit(). Replay reads Gaussian coefficients only.
Fixed temporal cosine bases compress trajectories; learned coefficients specify
positions, scales, directions and colours. Unit frames are enforced by the renderer.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import statistics
import time
import cv2
import imageio.v3 as iio
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import torch
from .representation import encode_clip,render_fields,decode_state
from .checkpoint_io import digest,save_inference_checkpoint,load_verified,keep_windows_awake


def temporal_basis(frames,terms,device='cpu'):
    return temporal_basis_at(frames,terms,torch.arange(frames,device=device),device=device)


def temporal_basis_at(frames,terms,times,device='cpu',derivative=False):
    """Continuous extension in frame-index units; not observed extra frames."""
    if not 1<=terms<=frames: raise ValueError('Require 1 <= terms <= frames')
    t=torch.as_tensor(times,device=device,dtype=torch.float32).reshape(-1,1)+0.5
    k=torch.arange(terms,device=device,dtype=torch.float32)[None]
    if derivative:
        return -torch.sin(math.pi*t*k/frames)*(math.pi*k/frames)*math.sqrt(2/frames)
    basis=torch.cos(math.pi*t*k/frames)*math.sqrt(2/frames)
    basis[:,0]=1/math.sqrt(frames)
    return basis


def expand_coefficients(coefficients,basis):
    return torch.einsum('tk,ckhw->cthw',basis,coefficients)


def physical_fields(coefficients,basis,position_limit=None):
    fields=expand_coefficients(coefficients,basis)
    if position_limit is not None:
        fields=torch.cat((position_limit*fields[:2].tanh(),fields[2:]),dim=0)
    return fields


@torch.no_grad()
def replay(packet,device='cuda',chunk=1):
    """CPU output; at most chunk frames' splat intermediates are on the GPU."""
    coefficients=packet['coefficients'].float().to(device)
    basis=temporal_basis(packet['frames'],coefficients.shape[1],device)
    frames=[]
    for begin in range(0,len(basis),chunk):
        fields=physical_fields(coefficients,basis[begin:begin+chunk],packet.get('position_limit'))[None]
        frames.append(render_fields(fields,size=packet['size'],radius=packet['radius'],
                                    min_log_scale=-2.)[0].cpu())
    return torch.cat(frames)


def read_video(path):
    # Include decoder construction and complete decode in the baseline timing.
    capture=cv2.VideoCapture(str(path)); frames=[]
    try:
        fps=capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok,frame=capture.read()
            if not ok: break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally: capture.release()
    if not frames: raise ValueError('Empty video')
    return np.stack(frames),fps


def fit(path,out,steps=600,grid=64,terms=8,anchored=False,lr=0.015):
    torch.manual_seed(58103)
    frames,fps=read_video(path)
    if frames.shape[1]!=frames.shape[2]: raise ValueError('This prototype requires square frames')
    if len(frames)>64 or frames.shape[1]>512: raise ValueError('Clip exceeds bounded pilot budget')
    device='cuda'; size=frames.shape[1]; terms=min(terms,len(frames))
    started=time.perf_counter()
    analyzed=np.concatenate([encode_clip(frame[None],grid=grid) for frame in frames],axis=1) if anchored else encode_clip(frames,grid=grid)
    raw=torch.from_numpy(analyzed).to(device)
    position_limit=2/grid if anchored else None  # Half a grid cell in displacement units.
    basis=temporal_basis(len(frames),terms,device)
    coefficients=torch.nn.Parameter(torch.einsum('tk,cthw->ckhw',basis,raw))
    target=torch.from_numpy(frames).permute(0,3,1,2).float().to(device)/255
    optimizer=torch.optim.Adam([coefficients],lr=lr)
    radius=7
    history=[]
    for step in range(1,steps+1):
        indices=torch.randint(len(frames),(2,),device=device)
        field=physical_fields(coefficients,basis[indices],position_limit)[None]
        output=render_fields(field,size=size,radius=radius,min_log_scale=-2.)[0]
        loss=(output-target[indices]).square().mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite Gaussian fit')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([coefficients],1.)
        optimizer.step()
        if step%100==0 or step in (1,steps):
            record=dict(step=step,pixel_mse=loss.item(),seconds=time.perf_counter()-started)
            history.append(record); print(json.dumps(record),flush=True)
    torch.cuda.synchronize()
    packet=dict(coefficients=coefficients.detach().half().cpu(),frames=len(frames),size=size,grid=grid,
                radius=radius,fps=fps,terms=terms,source_sha256=digest(path),
                position_limit=position_limit,dot_identity='Spatial anchors, NOT physical particle tracks' if anchored else 'Optical-flow tracks',
                fit_learning_rate=lr,
                kind='SOURCE-CONDITIONED RECONSTRUCTION; not noise generation',
                coordinate_system='Canonical 64-pixel coordinates, orthonormal local axes, normalized planar splatting',
                fit_steps=steps,fit_seconds=time.perf_counter()-started,history=history)
    save_inference_checkpoint(packet,out/'gaussian_trajectories.pt')
    return packet


@torch.no_grad()
def evaluate(packet,path,out,repeats=7):
    reference,fps=read_video(path)
    if digest(path)!=packet['source_sha256']: raise ValueError('Reference identity differs')
    reconstructed=replay(packet)
    target=torch.from_numpy(reference).permute(0,3,1,2).float()/255
    mse=(reconstructed-target).square().mean().item()
    uint8=(reconstructed.clamp(0,1)*255).round().byte().permute(0,2,3,1).numpy()
    imageio.mimsave(out/'gaussian_reconstruction.mp4',uint8,fps=fps,codec='libx264',macro_block_size=1)
    preview=[]
    for a,b in zip(reference,uint8):
        view=Image.new('RGB',(2*packet['size'],packet['size']+28),(18,22,29))
        view.paste(Image.fromarray(a),(0,28)); view.paste(Image.fromarray(b),(packet['size'],28))
        ImageDraw.Draw(view).text((5,6),'AI SOURCE | GAUSSIAN RECONSTRUCTION (NOT GENERATION)',fill='white')
        preview.append(view)
    preview[0].save(out/'comparison.gif',save_all=True,append_images=preview[1:],duration=round(1000/fps),loop=0)
    columns=[0,len(preview)//3,2*len(preview)//3,len(preview)-1]
    sheet=Image.new('RGB',(preview[0].width,preview[0].height*4))
    for j,i in enumerate(columns): sheet.paste(preview[i],(0,j*preview[0].height))
    sheet.save(out/'comparison.png')
    del reconstructed,target
    torch.cuda.empty_cache()
    replay(packet); torch.cuda.synchronize()  # Warm-up; source not available to replay.
    gaussian=[]; peaks=[]; reserved=[]
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        result=replay(packet); torch.cuda.synchronize()
        gaussian.append(time.perf_counter()-start)
        peaks.append(torch.cuda.max_memory_allocated()); reserved.append(torch.cuda.max_memory_reserved())
        del result
    ordinary=[]
    for _ in range(repeats):
        start=time.perf_counter(); decoded,_=read_video(path)
        ordinary.append(time.perf_counter()-start)
    sample_fields=physical_fields(packet['coefficients'].float(),temporal_basis(packet['frames'],packet['terms']),packet.get('position_limit'))
    rotations=decode_state(sample_fields[:,0][None],min_log_scale=-2.)['R']
    result=dict(kind=packet['kind'],source=str(path),source_sha256=digest(path),
                fit_seconds=packet['fit_seconds'],fit_steps=packet['fit_steps'],
                frames=packet['frames'],size=packet['size'],grid=packet['grid'],temporal_terms=packet['terms'],
                dot_identity=packet.get('dot_identity','Optical-flow tracks'),
                psnr_db=-10*math.log10(max(mse,1e-12)),
                raw_rgb_uint8_bytes=reference.nbytes,source_mp4_bytes=path.stat().st_size,
                fp16_coefficient_payload_bytes=packet['coefficients'].numel()*2,
                gaussian_file_bytes=(out/'gaussian_trajectories.pt').stat().st_size,
                reconstruction_mp4_bytes=(out/'gaussian_reconstruction.mp4').stat().st_size,
                gaussian_replay_seconds=gaussian,gaussian_replay_median_seconds=statistics.median(gaussian),
                ordinary_cpu_decode_seconds=ordinary,ordinary_cpu_decode_median_seconds=statistics.median(ordinary),
                gaussian_peak_cuda_allocated_bytes=max(peaks),gaussian_peak_cuda_reserved_bytes=max(reserved),
                ordinary_decode_cuda_bytes=0,
                axes_orthogonality_max=(rotations.transpose(-1,-2)@rotations-torch.eye(3)).abs().max().item(),
                limits=['Fits these source frames; not generalization or noise generation.',
                        'Gaussian fitting cost is additional to original AI video generation.',
                        'GPU Gaussian replay versus CPU H264 decode: implementation comparison, not matched hardware/FLOPs.',
                        'Peak CUDA bytes are tensor allocator measurements, not total device/process/CPU memory.',
                        'Raw byte reduction is not an advantage over compressed video.',
                        'Per-clip fitting outputs are never used to train or select the noise generator.'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,nargs='?')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--grid',type=int,default=64)
    parser.add_argument('--terms',type=int,default=8)
    parser.add_argument('--anchored',action='store_true')
    parser.add_argument('--lr',type=float,default=0.015)
    parser.add_argument('--replay',type=Path,help='Skip fitting; load saved coefficients')
    parser.add_argument('--render-only',action='store_true',help='Only saved Gaussian coefficients; never read source video')
    args=parser.parse_args()
    if not 1<=args.steps<=3000 or not 8<=args.grid<=128: parser.error('Invalid bounded fitting budget')
    if args.render_only and not args.replay: parser.error('--render-only requires --replay')
    if not args.render_only and not args.source: parser.error('Source required for fitting/evaluation')
    torch.set_num_threads(2); cv2.setNumThreads(2)
    args.out.mkdir(parents=True,exist_ok=False)
    if args.replay:
        packet=load_verified(args.replay)
        save_inference_checkpoint(packet,args.out/'gaussian_trajectories.pt')
    else: packet=fit(args.source,args.out,args.steps,args.grid,args.terms,args.anchored,args.lr)
    if args.render_only:
        video=replay(packet)
        frames=(video.clamp(0,1)*255).round().byte().permute(0,2,3,1).numpy()
        imageio.mimsave(args.out/'gaussian_reconstruction.mp4',frames,fps=packet['fps'],codec='libx264',macro_block_size=1)
        (args.out/'provenance.json').write_text(json.dumps(dict(kind=packet['kind'],
            input='Saved Gaussian trajectory coefficients only; no source RGB read',
            coefficients_sha256=digest(args.replay)),indent=2),encoding='utf-8')
        return
    # fit()'s tensors and optimizer have left scope before peak-memory measurement.
    evaluate(packet,args.source,args.out)


if __name__=='__main__':
    with keep_windows_awake(): main()
