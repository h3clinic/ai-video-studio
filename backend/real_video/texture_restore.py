"""Tiled pretrained texture restoration before persistent Gaussian baking.

This is an appearance experiment, not recovered hidden ground truth or training.
Requires spandrel==0.4.2 in the project virtual environment.
"""
import gc
import json
import shutil
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .checkpoint_io import digest,keep_windows_awake
from .hunyuan_gaussian import WORK
from . import hunyuan_multiview as paint

URL='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
WEIGHT_SHA256='4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'
ROOT=Path('artifacts/real_video/hunyuan_gaussian/detail_ablation/restored')

@torch.inference_mode()
def tiled(model,x,tile=128,pad=32):
    if tile<=0 or pad<0:raise ValueError('Tile must be positive and padding nonnegative')
    _,_,h,w=x.shape;scale=model.scale
    out=x.new_empty(1,3,h*scale,w*scale)
    for y in range(0,h,tile):
        for z in range(0,w,tile):
            y1,z1=min(y+tile,h),min(z+tile,w)
            a,b=max(0,y-pad),max(0,z-pad)
            patch=model(x[:,:,a:min(y1+pad,h),b:min(z1+pad,w)])
            out[:,:,y*scale:y1*scale,z*scale:z1*scale]=patch[:,:,(y-a)*scale:(y1-a)*scale,(z-b)*scale:(z1-b)*scale]
    return out

def main():
    from spandrel import ModelLoader
    ROOT.mkdir(parents=True,exist_ok=True)
    source=ROOT.parent.parent/'v5_full_paint'
    weights=WORK/'texture_restoration/RealESRGAN_x4plus.pth'
    weights.parent.mkdir(parents=True,exist_ok=True)
    if not weights.exists():
        pending=weights.with_suffix('.download')
        torch.hub.download_url_to_file(URL,str(pending))
        if digest(pending)!=WEIGHT_SHA256:raise ValueError('Downloaded restoration weights failed SHA256 check')
        pending.replace(weights)
    if digest(weights)!=WEIGHT_SHA256:raise ValueError('Restoration weights failed SHA256 check')
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    state=torch.load(weights,map_location='cpu',weights_only=True)
    model=ModelLoader().load_from_state_dict(state).eval().cuda().half()
    for i in range(6):
        dest=ROOT/f'generated_{i}.png'
        if dest.exists():continue
        arr=np.asarray(Image.open(source/f'generated_{i}.png').convert('RGB')).copy()
        x=torch.from_numpy(arr).permute(2,0,1)[None].cuda().half()/255
        y=tiled(model,x).float().clamp(0,1)
        Image.fromarray((y[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(dest)
        print(f'restored texture {i}: {tuple(y.shape)}',flush=True)
    torch.cuda.synchronize()
    report=dict(model_url=URL,weights_sha256=digest(weights),seconds=time.perf_counter()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),input_size=512,output_size=2048,
        source=str(source),base_weights_modified=False,accepted=False,
        limitations=['Independent per-view restoration may invent inconsistent detail; must evaluate seams.',
                    'No geometry correction, no novel motion, not verified hidden appearance.'])
    (ROOT/'generation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    del model,state,x,y;gc.collect();torch.cuda.empty_cache()
    for name in ['controls.pt','controls.pt.sha256.json']:
        if not (ROOT/name).exists():shutil.copy2(source/name,ROOT/name)
    paint.ROOT=ROOT;paint.BASE=ROOT.parent.parent/'v4_full'
    if not (ROOT/'gaussian_fitted.pt').exists():paint.bake(.8,'mean',ROOT,diagnostics=False)
    if not (ROOT/'camera_only_7s.mp4').exists():paint.inspect()

if __name__=='__main__':
    torch.set_num_threads(4)
    with keep_windows_awake():main()
