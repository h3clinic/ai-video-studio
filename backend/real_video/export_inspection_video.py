"""Seven-second inspection playback. Does not create new motion frames."""
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .checkpoint_io import digest

ROOT=Path('artifacts/real_video/connected_surface/v3')


def main():
    out=ROOT/'inspection_7s'; out.mkdir(exist_ok=True)
    targets=[out/'cat_large_7s.mp4',out/'cat_before_after_7s.mp4']
    if any(p.exists() for p in targets): raise FileExistsError('Preserve inspection exports')
    with np.load(ROOT/'cat.npz') as data:
        new=data['connected']; old=data['independent']
    assert new.shape==old.shape and len(new)==13
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    fps=28; count=196; indices=np.minimum(np.arange(count)*13//count,12)
    for path,comparison in zip(targets,[False,True]):
        width=1664 if comparison else 832
        writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=9,macro_block_size=1)
        try:
            for source in indices:
                canvas=Image.new('RGB',(width,568),'#141820'); draw=ImageDraw.Draw(canvas)
                panels=[('Previous bindings',old[source]),('Connected Gaussian surface',new[source])] if comparison else [('Connected Gaussian surface',new[source])]
                for panel,(label,frame) in enumerate(panels):
                    x=panel*832; draw.text((x+12,9),label,font=font,fill='white'); canvas.paste(Image.fromarray(frame),(x,44))
                draw.text((12,529),f'SLOW INSPECTION | existing state {source+1}/13 | no new motion / no interpolation',font=small,fill='white')
                draw.text((12,549),'7-second playback only. Wheat: Neil Oakes, CC BY-SA 2.0. Gait remains incorrect.',font=small,fill='#c7cbd1')
                writer.append_data(np.asarray(canvas))
                if not comparison and source==12 and not (out/'preview.jpg').exists(): canvas.save(out/'preview.jpg')
        finally: writer.close()
    checks=[]
    for path in targets:
        cap=cv2.VideoCapture(str(path)); actual_fps=cap.get(cv2.CAP_PROP_FPS); n=0
        while cap.read()[0]: n+=1
        cap.release(); assert n==196 and abs(actual_fps-28)<1e-6
        checks.append(dict(path=str(path),frames=n,fps=actual_fps,seconds=n/actual_fps,sha256=digest(path)))
    record=dict(source=str(ROOT/'cat.npz'),source_sha256=digest(ROOT/'cat.npz'),unique_motion_states=13,
                new_forecast_steps=0,interpolation=False,looping=False,frame_repetition=True,outputs=checks,
                math_sweep=dict(source='https://arxiv.org/html/2308.09713v1',section='3, local rigidity and long-term isometry',
                                cache_locator='html:char-21600',purpose='Checked persistent-motion distinction; playback retiming is not additional dynamics or long-horizon validation. No architecture change.'))
    (out/'manifest.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record,indent=2))


if __name__=='__main__': main()
