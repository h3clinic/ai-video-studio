"""Extract actual decoded output frames and an unaccepted visual-review template."""
import argparse
import json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
from .articulation_quality import create_review


def extract(out,video_name='video.mp4',review_name='review'):
    review=out/review_name;review.mkdir(exist_ok=False)
    video=out/video_name
    cap=cv2.VideoCapture(str(video))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    if count<2 or fps<=0:raise ValueError('No readable video')
    indices=sorted(set(round(i*(count-1)/7) for i in range(8)))
    frames=[]
    sheet=Image.new('RGB',(960,4*290),(20,28,36))
    draw=ImageDraw.Draw(sheet)
    for n,index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
        if not ok:raise ValueError('Frame decode failed')
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        path=review/f'frame_{index:03d}.png';image.save(path)
        frames.append(dict(path=path,frame_index=index))
        image.thumbnail((480,270))
        x,y=(n%2)*480,(n//2)*290
        sheet.paste(image,(x,y));draw.text((x+8,y+271),f'Frame {index} / {index/fps:.2f}s',fill='white')
    cap.release();sheet.save(review/'contact.png')
    metadata=dict(frame_count=count,fps=fps,duration_seconds=count/fps,
        sampled_indices=indices,continuous_playback_inspected=False)
    (review/'extraction.json').write_text(json.dumps(metadata,indent=2))
    (review/'visual_review.json').write_text(json.dumps(create_review(video,frames),indent=2))
    print(json.dumps(metadata))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path)
    parser.add_argument('--video-name',default='video.mp4')
    parser.add_argument('--review-name',default='review')
    args=parser.parse_args()
    extract(args.out,args.video_name,args.review_name)
