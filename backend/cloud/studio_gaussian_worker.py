"""Fixed remote worker: persistent object edits, not a scene/motion generator."""
import hashlib
import json
from pathlib import Path
import time


def main():
    import fcntl
    import subprocess
    lock=open('/tmp/gaussian_studio_gpu.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
    if active.stdout.strip():raise RuntimeError('Existing GPU work detected; refusing concurrent execution')
    import numpy as np
    import torch
    import psutil
    import imageio.v2 as imageio
    from real_video.gaussian_agent_edits import canonical_state_from_npz, apply_agent_operations, rigid_motion_at, state_to_arrays, operation_from_dict
    from real_video.gaussian3d import render
    if not torch.cuda.is_available() or psutil.virtual_memory().available<8*2**30:
        raise RuntimeError('Remote CUDA and 8 GiB available RAM required')
    root=Path.cwd();request=json.loads((root/'request.json').read_text())
    output=root/'output';output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    started=time.perf_counter()
    state,manifest,provenance=canonical_state_from_npz(root/'asset.npz',
        expected_sha256=request['asset_sha256'],part_id=request['part_id'],agent_id=request['agent_id'])
    load_seconds=time.perf_counter()-started
    start=time.perf_counter()
    operations=[operation_from_dict(value) for value in request['decision']['operations']]
    canonical,receipts=state,()
    if operations:canonical,manifest,receipts=apply_agent_operations(state,manifest,operations)
    edit_seconds=time.perf_counter()-start
    start=time.perf_counter()
    np.savez_compressed(output/'canonical_edited.npz',**state_to_arrays(canonical))
    checkpoint_seconds=time.perf_counter()-start
    motion=request['decision']['motion']
    fps=motion['fps'] if motion else 1
    count=round(motion['duration_seconds']*fps) if motion else 1
    controls={k:v for k,v in (motion or {}).items() if k not in ('fps','duration_seconds')}
    frames=[];motion_seconds=0.;raster_seconds=0.;conversion_seconds=0.;readback_seconds=0.
    for index in range(count):
        start=time.perf_counter()
        current=canonical
        if motion:
            operation=rigid_motion_at(request['agent_id'],request['part_id'],manifest.revision,time_seconds=index/fps,**controls)
            current,_,_=apply_agent_operations(canonical,manifest,[operation])
        motion_seconds+=time.perf_counter()-start
        start=time.perf_counter()
        arrays=state_to_arrays(current)
        data=[torch.as_tensor(arrays[k],device='cuda',dtype=torch.float32) for k in ('position','covariance','colour','opacity')]
        torch.cuda.synchronize();conversion_seconds+=time.perf_counter()-start
        start=time.perf_counter()
        with torch.no_grad():
            # Fixed camera: any movement comes from Gaussian means/covariance.
            frame=render(*data,torch.tensor([0.,0.,-2.2],device='cuda'),torch.zeros(3,device='cuda'),height=288,width=384,radius=4,ground=False)[0]
        torch.cuda.synchronize();raster_seconds+=time.perf_counter()-start
        start=time.perf_counter()
        frames.append((frame.cpu().numpy().clip(0,1)*255).round().astype('uint8'))
        readback_seconds+=time.perf_counter()-start
    start=time.perf_counter()
    imageio.mimsave(output/'gaussian_motion.mp4',frames,fps=fps,codec='libx264',macro_block_size=1)
    for i in sorted(set((0,count//2,count-1))):imageio.imwrite(output/f'frame_{i:03d}.png',frames[i])
    encoding_seconds=time.perf_counter()-start
    report=dict(accepted=False,status='rendered_review_required',classification='Canonical 3D Gaussian edit/motion diagnostic; not eating or full-scene generation',
        provenance=provenance,receipts=receipts,controls=motion,ids_preserved=sorted(state)==sorted(canonical),
        frames=count,fps=fps,actual_video_seconds=count/fps,source_sha256=request['asset_sha256'],training_seconds=0,generative_model_seconds=0,
        stages=dict(asset_load_seconds=load_seconds,attribute_edit_seconds=edit_seconds,motion_seconds=motion_seconds,
            array_transfer_seconds=conversion_seconds,rasterization_seconds=raster_seconds,encoding_seconds=encoding_seconds,
            checkpoint_seconds=checkpoint_seconds,frame_readback_seconds=readback_seconds),
        worker_wall_seconds=time.perf_counter()-started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),weights_changed=False,
        limitations=['No scene contact/occlusion binding','No learned new motion','Existing asset remains coarse','No texture UV mapping supplied'],
        video_sha256=hashlib.sha256((output/'gaussian_motion.mp4').read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
