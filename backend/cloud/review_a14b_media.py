"""Extract exact-source comparison frames; no perceptual scoring implied."""
from pathlib import Path
import hashlib
import json
import imageio.v2 as imageio
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cloud/a14b_pilot_v1'
RUN=OUT/'pilot_output_repaired'
if __name__=='__main__':
    report=json.loads((RUN/'report.json').read_text())
    board=Image.new('RGB',(5*416,3*266),'white')
    draw=ImageDraw.Draw(board)
    for row,(name,path) in enumerate([
        ('Source',ROOT/'artifacts/real_video/runway_agents/donkey_orange_v1/source/video.mp4'),
        ('Baseline',RUN/'original_baseline.mp4'),
        ('Gaussian adapted',RUN/'gaussian_adapted.mp4')]):
        if row:
            key=('original_baseline','gaussian_adapted')[row-1]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==report['videos'][key]['sha256']
        reader=imageio.get_reader(path)
        for col,index in enumerate((0,8,16,24,32)):
            source_index=report['source_frame_indices'][index] if row==0 else index
            frame=Image.fromarray(reader.get_data(source_index))
            if row: frame.save(OUT/f'{key}_{index:03d}.png')
            board.paste(frame.resize((416,240)),(col*416,row*266+26))
            draw.text((col*416+5,row*266+6),f'{name}: {index/16:.1f}s',fill='black')
        reader.close()
    board.save(OUT/'comparison.jpg')
    print(OUT/'comparison.jpg')
