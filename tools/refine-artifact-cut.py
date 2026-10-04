"""Trim transparent margins and enlarge existing artifact cutaways; preserve inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np

spec=importlib.util.spec_from_file_location('artifact',Path(__file__).with_name('render-artifact-cut.py'))
editor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(editor)
base=editor.base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',type=Path,required=True)
    args=parser.parse_args()
    root=args.pack.resolve()
    report=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    target_root=root/'trimmed-agents'
    target_root.mkdir(exist_ok=False)
    cuts=root/'edit/refined'
    cuts.mkdir(exist_ok=False)
    ffmpeg=base.encoder_path(None)
    trimmed={}
    for number in range(49,63):
        name=f'agent-{number}'
        source=root/'agents'/f'{name}.mov'
        info=base.probe(ffmpeg,source)
        w,h=info['width'],info['height']
        sw,sh=(w+3)//4,(h+3)//4
        result=subprocess.run([ffmpeg,'-v','error','-threads','2','-i',str(source),
                               '-vf',f'alphaextract,scale={sw}:{sh}',
                               '-an','-pix_fmt','gray','-f','rawvideo','-'],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=60)
        alpha=np.frombuffer(result.stdout,np.uint8).reshape(-1,sh,sw)
        ys,xs=np.nonzero(alpha.max(axis=0)>12)
        if not len(xs):
            raise ValueError('Empty alpha artifact')
        x,y=max(0,int(xs.min()*w/sw)-8),max(0,int(ys.min()*h/sh)-8)
        right,bottom=min(w,int((xs.max()+1)*w/sw)+9),min(h,int((ys.max()+1)*h/sh)+9)
        target=target_root/f'{name}.mov'
        base.execute([ffmpeg,'-v','error','-n','-threads','2','-i',source,
                      '-vf',f'crop={right-x}:{bottom-y}:{x}:{y}:exact=1',
                      '-an','-c:v','qtrle','-pix_fmt','argb','-threads','2',target])
        base.execute([ffmpeg,'-v','error','-n','-ss','1.5','-threads','2','-i',target,
                      '-frames:v','1','-update','1',target.with_suffix('.png')])
        trimmed[name]={'output':str(target),'source':str(source),'source_sha256':base.sha256(source),
                       'crop':[x,y,right-x,bottom-y],'sha256':base.sha256(target)}
        print(f'Trimmed {name}: {right-x}x{bottom-y}',flush=True)
    parts=[]
    for index,shot in enumerate(report['shots']):
        name=shot['asset']
        seconds=shot['end']-shot['start']
        if name not in trimmed:
            parts.append(root/'edit'/f'{index:02d}-{name}.mp4')
            continue
        target=cuts/f'{index:02d}-{name}.mp4'
        filters='[1:v]scale=1700:960:force_original_aspect_ratio=decrease,fps=30,setsar=1[fg];[0:v][fg]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p[v]'
        base.execute([ffmpeg,'-v','error','-n','-f','lavfi','-i',
                      f'color=c=0x101010:s=1920x1080:r=30:d={seconds}',
                      '-stream_loop','-1','-threads','2','-i',trimmed[name]['output'],
                      '-filter_complex_threads','2','-filter_complex',filters,'-map','[v]',
                      '-an','-frames:v',str(seconds*30),'-c:v','libx264','-preset','veryfast',
                      '-crf','18','-threads','4','-movflags','+faststart',target])
        parts.append(target)
    concat=cuts/'concat.txt'
    concat.write_text(''.join("file '"+str(p).replace('\\','/').replace("'","'\\''")+"'\n" for p in parts),encoding='utf-8')
    silent=cuts/'silent.mp4'
    base.execute([ffmpeg,'-v','error','-n','-f','concat','-safe','0','-i',concat,'-c','copy',silent])
    output=root/'artifact_cut_final.mp4'
    base.execute([ffmpeg,'-v','error','-n','-i',silent,'-i',root/'artifact_cut_clean.mp4',
                  '-map','0:v','-map','1:a','-c','copy','-movflags','+faststart',output])
    base.execute([ffmpeg,'-v','error','-n','-i',output,'-i',root/'narration.srt','-map','0','-map','1',
                  '-c','copy','-c:s','mov_text','-metadata:s:s:0','language=eng','-disposition:s:0','0',
                  '-movflags','+faststart',root/'artifact_cut_final_subtitles.mp4'])
    result={'clean_video':str(output),'sha256':base.sha256(output),'format':base.probe(ffmpeg,output),
            'agents':trimmed,'new_provider_calls':0,'prior_pack_preserved':True,
            'change':'Transparent motion-envelope margins trimmed; agent cutaways enlarged. No slide graphics.',
            'review':'pending'}
    (root/'refinement.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'video':str(output),'format':result['format']}),flush=True)


if __name__=='__main__':
    main()
