"""Two-stage seed extraction and isolated recurrent continuation beyond Wan clip."""
import argparse
import builtins
import json
from pathlib import Path
import time
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from .persistent_gaussian import PersistentGaussianDynamics, GaussianMemorySession, wan_fields_to_state, render_state, state_bytes

OUT=Path('artifacts/real_video/persistent_gaussian/v1')
SOURCE=Path('artifacts/real_video/wan_bridge/v2/generated_cat/gaussian_fields.pt')


def seed():
    source=load_verified(SOURCE)
    raw=source['fields']
    states=torch.stack([wan_fields_to_state(raw[:,:,i],source['height'],source['width']) for i in [-2,-1]],2)
    save_inference_checkpoint(dict(observed_states=states,height=source['height'],width=source['width'],
                                   source_sha256=digest(SOURCE),dt=.5,
                                   note='Last TWO states of completed 33-frame joint Wan clip; no future beyond that clip'),OUT/'cat_seed.pt')
    print('Saved two-state-only seed packet; subsequent process does not load the full clip.')


def run():
    if (OUT/'continuation.json').exists(): raise FileExistsError('Preserve existing continuation')
    allowed={(OUT/'dynamics.pt').resolve(),(OUT/'cat_seed.pt').resolve()}
    loaded=[]; original_load=torch.load; original_import=builtins.__import__
    def guarded_load(path,*args,**kwargs):
        if Path(path).resolve() not in allowed: raise AssertionError('Unexpected tensor input during continuation')
        loaded.append(Path(path).name); return original_load(path,*args,**kwargs)
    def guarded_import(name,*args,**kwargs):
        if name=='diffusers' or name.startswith('diffusers.'):
            raise AssertionError('No diffusion/VAE during recurrent continuation')
        return original_import(name,*args,**kwargs)
    torch.set_num_threads(4)
    with keep_windows_awake(), torch.no_grad(), patch('torch.load',side_effect=guarded_load), \
         patch('builtins.__import__',side_effect=guarded_import):
        weights=load_verified(OUT/'dynamics.pt'); packet=load_verified(OUT/'cat_seed.pt')
        model=PersistentGaussianDynamics(weights['config']['hidden']).eval().to('cuda')
        model.load_state_dict(weights['model'])
        observed=packet['observed_states'].to('cuda')
        state=model.initialize(observed[:,:,0],observed[:,:,1],dt=packet['dt'])
        session=GaussianMemorySession(model,state)
        del state,observed
        height,width=packet['height'],packet['width']
        expected_bytes=state_bytes(session.state)
        del packet
        checkpoints={}; times=[]; memory_sizes=[]; samples=[]
        torch.cuda.reset_peak_memory_stats()
        writer=imageio.get_writer(OUT/'recurrent_continuation.mp4',fps=16,codec='libx264',quality=8,macro_block_size=1)
        try:
            for step in range(32):
                torch.cuda.synchronize(); start=time.perf_counter()
                fields=session.advance(dt=.5)
                rgb=render_state(fields,height,width,radius=8)
                torch.cuda.synchronize(); times.append(time.perf_counter()-start)
                memory_sizes.append(state_bytes(session.state))
                if not torch.isfinite(fields).all() or not torch.isfinite(session.state['memory']).all():
                    raise ValueError('Nonfinite continuation')
                image=(rgb[0].clamp(0,1).permute(1,2,0)*255).round().byte().cpu().numpy()
                writer.append_data(image)
                if step in [0,7,15,31]: samples.append(image)
                if step in [7,11,31]: checkpoints[step]=session.snapshot()
        finally: writer.close()
        peak=torch.cuda.max_memory_allocated()
        # Restore at frame 8 and independently reproduce frame 12, without any clip input.
        session.restore(checkpoints[7])
        for _ in range(4): session.advance(dt=.5)
        differences={k:(session.state[k].cpu()-checkpoints[11]['state'][k]).abs().max().item() for k in session.state}
        assert max(differences.values())<1e-6
        assert set(memory_sizes)=={expected_bytes}
        session.restore(checkpoints[31])
        before=state_bytes(session.state)
        # Long no-render stress test: finite fixed-shape mechanics, not a quality claim.
        for _ in range(128): session.advance(dt=.5)
        assert state_bytes(session.state)==before
        assert all(torch.isfinite(v).all() for v in session.state.values())
        frame_norm_error=(session.state['fields'][:,4:6].norm(dim=1)-1).abs().max().item()
    save_inference_checkpoint(dict(snapshot=checkpoints[31],model_sha256=digest(OUT/'dynamics.pt'),
                                   height=height,width=width,dt=.5),OUT/'cat_memory_after32.pt')
    sheet=Image.new('RGB',(width*2,height*2))
    for i,sample in enumerate(samples): sheet.paste(Image.fromarray(sample),((i%2)*width,(i//2)*height))
    sheet.save(OUT/'continuation_contact.jpg')
    report=dict(only_tensor_inputs=loaded,frames=32,seed_frames=2,frame_shape=[height,width,3],
                current_state_bytes=expected_bytes,state_bytes_constant_over_demo=True,
                extra_128_steps_finite_fixed_shape=True,unit_axis_max_error_after_stress=frame_norm_error,
                save_restore_max_difference_by_tensor=differences,
                median_step_and_render_seconds_after_4_warmup=float(np.median(times[4:])),
                peak_cuda_allocated_bytes=peak,video_sha256=digest(OUT/'recurrent_continuation.mp4'),
                no_future_frames_or_latents=True,
                limits='Continuation beyond a joint-generated seed clip. Fixed indexed slots, no verified object/occlusion memory; OOD visual quality unproven. Output files/renderer/training memory excluded from state bytes.')
    (OUT/'continuation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def compose():
    # Separate post-inference visualization. No RGB videos were input to dynamics.
    sources=[Path('artifacts/real_video/wan_bridge/v2/generated_cat/wan_latent_gaussian.mp4'),OUT/'recurrent_continuation.mp4']
    writer=imageio.get_writer(OUT/'warmup_then_memory.mp4',fps=16,codec='libx264',quality=8,macro_block_size=1)
    try:
        for segment,path in enumerate(sources):
            reader=imageio.get_reader(path)
            try:
                for frame in reader:
                    canvas=Image.new('RGB',(832,512),'#151922')
                    canvas.paste(Image.fromarray(frame),(0,32))
                    ImageDraw.Draw(canvas).text((12,10),
                        'JOINT WAN GAUSSIAN WARM-UP (not recurrent)' if segment==0 else 'RECURRENT CONTINUATION: stored state only; no Wan / future frames',fill='white')
                    writer.append_data(np.asarray(canvas))
            finally: reader.close()
    finally: writer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['seed','run','compose'])
    {'seed':seed,'run':run,'compose':compose}[parser.parse_args().mode]()
