"""Export actual XYZ Gaussian reconstruction and explicitly slowed inspection."""
import json
import math
import time
from pathlib import Path
import hashlib
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .checkpoint_io import load_verified,keep_windows_awake,digest
from .gaussian3d import render
from .surface_skin3d import SurfaceSkinner
from .fit_motion3d import read_frames,rotation_vectors

ROOT=Path('artifacts/real_video/true3d/v2_motion/surface_refined')
APPEARANCE=Path('artifacts/real_video/true3d/v4_appearance/cat_asset.pt')


def cuda_packet(packet):
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in packet.items()}


@torch.no_grad()
def main():
    if (ROOT/'video_report.json').exists(): raise FileExistsError('Preserve rendered comparisons')
    a=cuda_packet(load_verified(APPEARANCE)); rig=cuda_packet(load_verified(ROOT/'rig.pt')); m=load_verified(ROOT/'motion.pt')
    source,source_fps=read_frames(); cam=m['camera']; h,w=source[0].shape[:2]
    eye=cam['eye'].cuda(); target=cam['target'].cuda()
    # Undo the conditioning crop in INTRINSICS, not with an image warp/crop.
    principal=[cam['size']/2+cam['x0']-cam['pad_x'],cam['size']/2+cam['y0']-cam['pad_y']]
    fov=math.degrees(2*math.atan(h/cam['size']*math.tan(math.radians(cam['fov'])/2)))
    support=SurfaceSkinner(a,rig); rot=m['rotation_vectors'].cuda(); shift=m['root_translation'].cuda()
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18); report={}
    for mode,n,fps in [('native',33,source_fps),('inspection_7s',168,24)]:
        path=ROOT/f'motion_{mode}.mp4'; writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=1)
        times=[]; torch.cuda.reset_peak_memory_stats()
        try:
            for index in range(n):
                frame=float(index) if mode=='native' else index*(len(source)-1)/(n-1)
                lower=int(frame); upper=min(lower+1,len(source)-1); fraction=frame-lower
                torch.cuda.synchronize(); begin=time.perf_counter()
                rv=rot[lower]*(1-fraction)+rot[upper]*fraction; tr=shift[lower]*(1-fraction)+shift[upper]*fraction
                pos,cov,_,_,_=support(rotation_vectors(rv),tr)
                rgb,alpha=render(pos,cov,a['colour'],a['opacity'],eye,target,h,w,fov=fov,ground=False,principal=principal)
                rgb=rgb+(1-alpha[...,None])*(.5-rgb.new_tensor([.15,.19,.24]))
                torch.cuda.synchronize(); times.append((time.perf_counter()-begin)*1000)
                canvas=Image.new('RGB',(w*2,h+80),'#151922'); draw=ImageDraw.Draw(canvas)
                canvas.paste(Image.fromarray(source[round(frame)]),(0,32))
                canvas.paste(Image.fromarray(np.uint8(rgb.clamp(0,1).cpu().numpy()*255)),(w,32))
                draw.text((8,7),'SOURCE WAN VIDEO | '+('native speed' if mode=='native' else 'SLOWED inspection; source frames held'),font=font,fill='white')
                draw.text((w+8,7),'XYZ GAUSSIAN RECONSTRUCTION | manual-assisted motion fit',font=font,fill='white')
                draw.text((8,h+39),'Persistent appearance + shared mesh + fitted joint vectors. Not learned action generation. Paw/contact errors remain.',font=font,fill='white')
                draw.text((8,h+60),f'Source frame {frame:.2f}/32 | '+('33 source states, 2.06 seconds' if mode=='native' else '2-second source motion stretched to 7 seconds; interpolated joint controls'),font=font,fill='#c5cbd5')
                writer.append_data(np.asarray(canvas))
                if mode=='native' and index in [2,10,18,26,32]: canvas.save(ROOT/f'comparison_{index:02d}.jpg')
                if index%42==0: print(json.dumps(dict(mode=mode,frame=index,total=n)),flush=True)
        finally: writer.close()
        reader=imageio.get_reader(path); decoded=0; unique=set()
        for pic in reader:
            decoded+=1; unique.add(hashlib.sha256(pic.tobytes()).hexdigest())
        meta=reader.get_meta_data(); reader.close(); assert decoded==n and meta['fps']==fps
        report[mode]=dict(decoded_frames=decoded,fps=fps,seconds=n/fps,unique_decoded_frames=len(unique),
            median_deform_render_ms=float(np.median(times[5:])),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            sha256=digest(path),video_bytes=path.stat().st_size)
    report.update(appearance_sha256=digest(APPEARANCE),motion_sha256=digest(ROOT/'motion.pt'),rig_sha256=digest(ROOT/'rig.pt'),
        unused_degenerate_mesh_faces=support.unused_degenerate_faces,principal=principal,full_source_fov=fov,
        limitations='Manual-assisted reconstruction from33sourceframes. No new learned motion. 7second file is slowed and joint-interpolated. Foot contacts/anatomy fail; no quality-matched efficiency comparison.',
        source_code_sha256={name:digest(Path('real_video')/name) for name in ['fit_motion3d.py','surface_skin3d.py','gaussian3d.py','export_surface_motion.py']})
    (ROOT/'video_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    with keep_windows_awake(): main()
