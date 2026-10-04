"""Extract reusable video artifacts and edit them without presentation cards.

Offline only. Inputs are an explicit JSON plan, existing app recordings, original
agent artwork and a sound-enhanced source clip. No model/provider calls occur.
Transparent MOVs use the lossless QuickTime Animation codec, not a chroma key.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy import ndimage
from PIL import Image

SPEC = importlib.util.spec_from_file_location('demo_helpers', Path(__file__).with_name('render-demo.py'))
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def opaque_foreground(rgb):
    """Only remove near-white connected to the border, retaining enclosed whites."""
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    white = (low >= 238) & ((high.astype(np.int16)-low) <= 12)
    seeds = np.zeros(white.shape, bool)
    seeds[[0, -1], :] = white[[0, -1], :]
    seeds[:, [0, -1]] = white[:, [0, -1]]
    exterior = ndimage.binary_propagation(seeds, mask=white)
    alpha = ndimage.gaussian_filter((~exterior).astype(np.float32), .45)
    alpha[alpha < .015] = 0
    alpha[alpha > .985] = 1
    # Remove the white matte from softened silhouette pixels before alpha export.
    color = np.clip((rgb.astype(np.float32)-255*(1-alpha[..., None])) /
                    np.maximum(alpha[..., None], .001), 0, 255)
    color[alpha == 0] = 0
    return np.dstack([color.astype(np.uint8), np.round(alpha*255).astype(np.uint8)])


def artwork_bounds(ffmpeg, source, info):
    result = base.execute([ffmpeg, '-hide_banner', '-threads', '2', '-i', source,
                           '-vf', 'negate,cropdetect=16:2:0', '-an', '-f', 'null', '-'])
    bounds = [tuple(map(int, m)) for m in re.findall(r'crop=(\d+):(\d+):(\d+):(\d+)', result.stderr)]
    if not bounds:
        raise ValueError('No foreground artwork found')
    x = max(0, min(b[2] for b in bounds)-16)
    y = max(0, min(b[3] for b in bounds)-16)
    right = min(info['width'], max(b[2]+b[0] for b in bounds)+16)
    bottom = min(info['height'], max(b[3]+b[1] for b in bounds)+16)
    return x, y, right-x, bottom-y


def extract_agent(ffmpeg, source, target):
    info = base.probe(ffmpeg, source)
    x, y, width, height = artwork_bounds(ffmpeg, source, info)
    raw = subprocess.run([ffmpeg, '-v', 'error', '-threads', '2', '-i', str(source),
                          '-vf', f'crop={width}:{height}:{x}:{y},fps=24', '-an',
                          '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    encoder = subprocess.Popen([ffmpeg, '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo',
                                '-pixel_format', 'rgba', '-video_size', f'{width}x{height}',
                                '-framerate', '24', '-i', '-', '-an', '-c:v', 'qtrle',
                                '-pix_fmt', 'argb', '-threads', '2', str(target)],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    alphas = []
    try:
        for index, frame in enumerate(frames):
            rgba = opaque_foreground(frame)
            encoder.stdin.write(rgba.tobytes())
            alphas.append(float((rgba[..., 3] == 0).mean()))
            if index == len(frames)//2:
                Image.fromarray(rgba).save(target.with_suffix('.png'))
        encoder.stdin.close()
        error = encoder.stderr.read()
        if encoder.wait(timeout=60):
            raise RuntimeError(error.decode(errors='replace')[-2000:])
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
    return {'source': str(source), 'source_sha256': base.sha256(source), 'crop': [x,y,width,height],
            'frames': len(frames), 'fps': 24, 'minimum_transparent_fraction': min(alphas),
            'output': str(target), 'sha256': base.sha256(target),
            'meaning': 'Original animated role illustration, not generated scene or live execution'}


def source_path(value, plan_file):
    path = (plan_file.parent / value).resolve()
    if not path.is_file():
        raise ValueError(f'Missing explicit media input: {path}')
    return path


def validate_shots(shots):
    if not isinstance(shots, list):
        raise ValueError('Shots must be an array')
    elapsed = 0
    for shot in shots:
        if not isinstance(shot, dict) or set(shot)-{'asset','seconds'}:
            raise ValueError('Shots contain only an asset ID and seconds')
        if not isinstance(shot.get('asset'), str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', shot['asset']):
            raise ValueError('Invalid asset ID')
        if type(shot.get('seconds')) is not int or shot['seconds'] <= 0:
            raise ValueError('Shot durations must be positive whole seconds')
        elapsed += shot['seconds']
    if elapsed != 180:
        raise ValueError('Artifact cut must total exactly 180 seconds')


def srt_time(value):
    value = round(value*1000)
    return f'{value//3600000:02d}:{value//60000%60:02d}:{value//1000%60:02d},{value%1000:03d}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frontend', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--ffmpeg')
    args = parser.parse_args()
    plan_file = args.plan.resolve()
    plan = json.loads(plan_file.read_text(encoding='utf-8'))
    validate_shots(plan['shots'])
    outline = base.parse_timed_script(args.script.read_text(encoding='utf-8'))
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError('Preserving existing output; choose a new directory')
    root.mkdir(parents=True)
    for name in ('agents', 'clips', 'edit'):
        (root/name).mkdir()
    ffmpeg = base.encoder_path(args.ffmpeg)
    report = {'new_provider_calls': 0, 'slides_created': 0, 'assets': {}, 'shots': [],
              'narration_recorded': False, 'editorial_notes': [
                  'Original 5-second scene repeats; no new visual generation or Gaussian replacement.',
                  'Role animations are illustrations, not evidence of live agent execution.',
                  'UI crops and held frames are editorial changes, not modified experiment results.',
                  'Subtitles have estimated timings for the separate human narration script.']}
    assets = {}
    for number in range(49, 63):
        source = args.frontend.resolve()/f'public/videos/agent-{number}.mp4'
        target = root/'agents'/f'agent-{number}.mov'
        print(f'Extracting transparent agent {number}', flush=True)
        report['assets'][f'agent-{number}'] = extract_agent(ffmpeg, source, target)
        assets[f'agent-{number}'] = target
    for name, entry in plan['clips'].items():
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
            raise ValueError('Invalid clip ID')
        source = source_path(entry['source'], plan_file)
        info = base.probe(ffmpeg, source)
        crop = base.validate_crop(entry.get('crop'), info)
        start, seconds = float(entry.get('start', 0)), float(entry['seconds'])
        if not math.isfinite(start) or not math.isfinite(seconds) or start < 0 or seconds <= 0 or start+seconds > info['seconds']+.05:
            raise ValueError('Clip exceeds source duration')
        prefix = f'crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]},' if crop else ''
        target = root/'clips'/f'{name}.mp4'
        print(f'Cropping actual artifact: {name}', flush=True)
        base.execute([ffmpeg, '-v', 'error', '-nostdin', '-n', '-ss', str(start), '-threads', '2', '-i', source,
                      '-t', str(seconds), '-vf', prefix+'pad=ceil(iw/2)*2:ceil(ih/2)*2,fps=30',
                      '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', '-threads', '2',
                      '-pix_fmt', 'yuv420p', '-movflags', '+faststart', target])
        report['assets'][name] = {'source': str(source), 'source_sha256': base.sha256(source),
                                  'start': start, 'seconds': seconds, 'crop': crop,
                                  'output': str(target), 'sha256': base.sha256(target)}
        assets[name] = target
    elapsed, parts = 0, []
    for index, shot in enumerate(plan['shots']):
        name, seconds = shot['asset'], shot['seconds']
        source = assets[name]
        agent = name.startswith('agent-')
        loop = agent or name == 'source-video'
        args_in = ['-stream_loop', '-1'] if loop else []
        filters = ('scale=850:850:force_original_aspect_ratio=decrease' if agent else
                   'scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos')
        filters += f',fps=30,tpad=stop_mode=clone:stop_duration={seconds},setsar=1[fg];'
        filters += '[0:v][fg]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p[v]'
        target = root/'edit'/f'{index:02d}-{name}.mp4'
        print(f'Editing {elapsed:03d}–{elapsed+seconds:03d}s: {name}', flush=True)
        base.execute([ffmpeg, '-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i',
                      f'color=c=0x101010:s=1920x1080:r=30:d={seconds}',
                      *args_in, '-threads', '2', '-i', source, '-filter_complex_threads', '2',
                      '-filter_complex', '[1:v]'+filters, '-map', '[v]', '-an',
                      '-frames:v', str(seconds*30), '-c:v', 'libx264', '-preset', 'veryfast',
                      '-crf', '18', '-threads', '4', '-movflags', '+faststart', target])
        report['shots'].append({'start':elapsed,'end':elapsed+seconds,'asset':name,
                                'repeated':loop,'held_final_frame':not loop and seconds>base.probe(ffmpeg,source)['seconds']})
        elapsed += seconds
        parts.append(target)
    concat = root/'edit/concat.txt'
    concat.write_text(''.join("file '"+str(p).replace('\\','/').replace("'","'\\''")+"'\n" for p in parts),encoding='utf-8')
    silent = root/'edit/silent.mp4'
    base.execute([ffmpeg,'-v','error','-n','-f','concat','-safe','0','-i',concat,'-c','copy',silent])
    mix = source_path(plan['sound_mix'], plan_file)
    clean = root/'artifact_cut_clean.mp4'
    audio = '[1:a]aresample=48000,atrim=duration=35,asetpts=PTS-STARTPTS,volume=0.8,afade=t=in:d=0.15,afade=t=out:st=34.85:d=0.15,adelay=95000:all=1,apad=whole_dur=180,atrim=duration=180[a]'
    base.execute([ffmpeg,'-v','error','-n','-i',silent,'-stream_loop','-1','-threads','2','-i',mix,
                  '-filter_complex_threads','2','-filter_complex',audio,'-map','0:v','-map','[a]',
                  '-c:v','copy','-c:a','aac','-b:a','192k','-threads','2','-t','180','-movflags','+faststart',clean])
    captions = root/'narration.srt'
    captions.write_text('\n\n'.join(f"{i}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n{c['text']}"
                                    for i,c in enumerate(base.caption_chunks(outline),1))+'\n',encoding='utf-8')
    selectable = root/'artifact_cut_subtitles.mp4'
    base.execute([ffmpeg,'-v','error','-n','-i',clean,'-i',captions,'-map','0','-map','1',
                  '-c','copy','-c:s','mov_text','-metadata:s:s:0','language=eng',
                  '-disposition:s:0','0','-movflags','+faststart',selectable])
    report.update({'clean_video':str(clean),'subtitled_video':str(selectable),'sha256':base.sha256(clean),
                   'format':base.probe(ffmpeg,clean),'sound_mix':str(mix),'sound_mix_sha256':base.sha256(mix),
                   'sound_seconds':[95,130],'visual_review':'pending'})
    (root/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'output_dir':str(root),'clean_video':str(clean),'format':report['format']}),flush=True)


if __name__ == '__main__':
    main()
