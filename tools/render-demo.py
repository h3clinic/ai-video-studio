"""Edit genuine studio footage into a 180-second, 1080p sponsor demonstration.

No capture, provider calls, image generation, or model work occurs here. Pillow
typesets editorial cards; FFmpeg scales/crops existing media and encodes <=4
threads. The guide captions are estimated timing for a human narration script.

Usage:
  python tools/render-demo.py --script docs/demo-script.txt --mix MIX.mp4 \
      --capture APP_ONLY.mp4 --output DEMO.mp4

Capture map: {"3":{"start":40,"duration":18,"crop":[0,0,1920,900]}, ...}
Keys are segment numbers or "SEGMENT_STARTSECOND" for a specific shot. An
optional "capture" path overrides the source relative to the capture-map file.
Crops are optional [x,y,width,height]. If an
excerpt is shorter than its editorial slot its final genuine frame is held.
Never pass a desktop capture or footage containing credentials to this script.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont

FRONTEND = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS, SECONDS = 1920, 1080, 30, 180
ACCENT, INK, MUTED = '#7dd3fc', '#f0f5fc', '#acbad0'
ROLE_IDS = [49, 53, 59, 60, 61, 62]
ROLE_NAMES = ['Idea Model', 'Vision Sensor', 'Task Assigner', 'Vector Agents', 'Sound Agents', 'Verifiers']


def execute(args, timeout=600):
    result = subprocess.run([str(x) for x in args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('FFmpeg failed:\n' + result.stderr[-5000:])
    return result


def encoder_path(explicit):
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise ValueError('--ffmpeg executable does not exist')
        return str(candidate)
    found = shutil.which('ffmpeg')
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError('Install imageio-ffmpeg, put FFmpeg on PATH, or pass --ffmpeg PATH.') from None


def validate_github_url(value):
    """Allow only plain HTTPS repository/branch links; never credentials or extras."""
    if value == '':
        return value
    if not isinstance(value, str) or len(value) > 2048 or any(c.isspace() for c in value):
        raise ValueError('GitHub URL must be a plain repository or branch URL')
    url = urlparse(value)
    parts = url.path.strip('/').split('/')
    safe = re.compile(r'[A-Za-z0-9_.-]+\Z')
    if (url.scheme != 'https' or url.netloc != 'github.com' or url.query or url.fragment
            or not (len(parts) == 2 or (len(parts) >= 4 and parts[2] == 'tree'))
            or any(not safe.fullmatch(part) or part in ('.', '..') for part in parts)):
        raise ValueError('GitHub URL must be a repository or branch URL on https://github.com')
    return value


def validate_outline(outline):
    """Validate complete, positive, contiguous coverage before doing any encoding."""
    segments = outline.get('segments')
    if not isinstance(segments, list) or not segments:
        raise ValueError('No timed narration sections in --script')
    previous = 0
    for index, segment in enumerate(segments, 1):
        left, right = segment.get('start_seconds'), segment.get('end_seconds')
        if (type(left) is not int or type(right) is not int or left != previous
                or right <= left or right > SECONDS):
            raise ValueError('Outline must cover exactly 0–180 seconds without gaps or overlaps')
        if not isinstance(segment.get('narration'), str) or not segment['narration'].strip():
            raise ValueError('Every timed section requires narration')
        if segment.get('index') != index:
            raise ValueError('Section indexes must be consecutive')
        previous = right
    if previous != SECONDS:
        raise ValueError('Outline must cover exactly 0–180 seconds without gaps or overlaps')
    words = sum(len(segment['narration'].split()) for segment in segments)
    if outline.get('spoken_word_count') != words:
        raise ValueError('Narration word count differs from outline')
    return outline


def parse_timed_script(source):
    """Read headings such as 00:00–00:30 — Title; introductory prose is ignored."""
    headings = list(re.finditer(r'^(\d{2}):(\d{2})[–-](\d{2}):(\d{2})\s+[—-]\s+([^\r\n]+)\r?$', source, re.M))
    segments = []
    for i, heading in enumerate(headings):
        if int(heading[2]) >= 60 or int(heading[4]) >= 60:
            raise ValueError('Time seconds must be between 00 and 59')
        end = headings[i+1].start() if i+1 < len(headings) else len(source)
        segments.append({'index': i+1, 'start_seconds': int(heading[1])*60+int(heading[2]),
                         'end_seconds': int(heading[3])*60+int(heading[4]), 'title': heading[5].strip(),
                         'narration': ' '.join(source[heading.end():end].split()),
                         'total_segments': len(headings)})
    return validate_outline({'segments': segments,
                             'spoken_word_count': sum(len(s['narration'].split()) for s in segments)})


def validate_crop(crop, capture_info):
    if crop is None:
        return None
    if not isinstance(crop, (list, tuple)) or len(crop) != 4 or any(type(v) is not int or v < 0 for v in crop):
        raise ValueError('Crop requires four nonnegative integers')
    x, y, width, height = crop
    if width < 2 or height < 2 or x+width > capture_info['width'] or y+height > capture_info['height']:
        raise ValueError('Crop exceeds app capture')
    return tuple(crop)


def load_capture_map(path):
    if path is None:
        return {}
    entries = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(entries, dict):
        raise ValueError('Capture map must be an object keyed by segment or shot')
    for key, entry in entries.items():
        if not re.fullmatch(r'[1-9]\d*(?:_\d+)?', key) or not isinstance(entry, dict):
            raise ValueError('Capture map entries must be objects keyed by segment or shot')
        if set(entry) - {'capture', 'start', 'duration', 'crop'}:
            raise ValueError('Unknown capture map field')
        if 'capture' in entry:
            if not isinstance(entry['capture'], str) or not entry['capture']:
                raise ValueError('Capture override must identify a local video')
            entry['capture'] = str((path.parent / entry['capture']).resolve())
    return entries


def probe(ffmpeg, path):
    result = subprocess.run([ffmpeg, '-hide_banner', '-i', str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    text = result.stderr
    match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', text)
    size = re.search(r'Video:.*?\b(\d{2,5})x(\d{2,5})\b', text)
    if not match or not size:
        raise ValueError(f'Cannot inspect video: {path}')
    return {'seconds': int(match[1])*3600 + int(match[2])*60 + float(match[3]),
            'width': int(size[1]), 'height': int(size[2]), 'audio': 'Audio:' in text}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def font(size, bold=False):
    name = 'segoeuib.ttf' if bold else 'segoeui.ttf'
    candidates = ([Path(os.environ['WINDIR']) / 'Fonts' / name] if os.environ.get('WINDIR') else []) + [
                  Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def lines_for(draw, text, face, max_width):
    lines, current = [], ''
    for word in text.split():
        proposed = f'{current} {word}'.strip()
        if current and draw.textlength(proposed, font=face) > max_width:
            lines.append(current)
            current = word
        else:
            current = proposed
    if current:
        lines.append(current)
    return lines


def paragraph(draw, text, xy, size=40, width=900, fill=INK, bold=False, gap=12):
    face = font(size, bold)
    x, y = xy
    for line in lines_for(draw, text, face, width):
        draw.text((x, y), line, font=face, fill=fill)
        y += size + gap
    return y


def card(path, segment, kind, github=''):
    canvas = Image.new('RGB', (WIDTH, HEIGHT), '#0b1120')
    draw = ImageDraw.Draw(canvas)
    for y in range(HEIGHT):
        value = int(8*y/HEIGHT)
        draw.line((0,y,WIDTH,y), fill=(11+value,17+value,32+value))
    draw.rectangle((48,43,54,83), fill=ACCENT)
    draw.text((76,43), 'AI VIDEO STUDIO', font=font(28, True), fill=INK)
    draw.text((1190,45), 'SpacetimeDB  +  ElevenLabs', font=font(27, True), fill=ACCENT)
    draw.line((48,102,1872,102), fill='#26354a', width=2)
    if kind == 'intro':
        paragraph(draw, 'A scene. A team.\nAn editable history.'.replace('\n',' '), (78,182), 76, 1040, bold=True, gap=18)
        paragraph(draw, 'Persistent, part-level video editing', (82,430), 38, 1060, fill=ACCENT)
        paragraph(draw, 'Shared instructions with SpacetimeDB.\nThree generated ElevenLabs sound layers.', (82,520), 34, 980, fill=MUTED)
        draw.text((1260,765), 'EXISTING SOURCE + GENERATED AUDIO', font=font(19, True), fill=MUTED)
    elif kind == 'roles':
        paragraph(draw, 'One observable team', (78,145), 62, 1650, bold=True)
        draw.text((82,237), '14 protocol roles · 17 recorded workers · 3 generated sound layers', font=font(29), fill=ACCENT)
        for i, label in enumerate(ROLE_NAMES):
            x = 80 + (i % 3)*600
            y = 302 + (i // 3)*266
            draw.rounded_rectangle((x-2,y-2,x+550,y+208), 16, fill='#182438', outline='#34455e', width=2)
            draw.text((x+2,y+214), label, font=font(24, True), fill=INK)
        draw.text((80,866), 'Original role artwork · illustrations, not generated scene output', font=font(21), fill=MUTED)
    elif kind == 'outro':
        paragraph(draw, 'Inspect a decision.\nShare an instruction.\nExport a result.'.replace('\n',' '), (78,168), 65, 1110, bold=True, gap=18)
        paragraph(draw, 'Membership · revision checks · attributed history', (82,466), 29, 1110, fill=ACCENT)
        paragraph(draw, 'Independent sound layers · local MP4 export', (82,540), 29, 1110, fill=MUTED)
        if github:
            paragraph(draw, github.removeprefix('https://'), (82,682), 27, 1120, fill=INK)
        draw.text((1260,765), 'EXPORTED OUTPUT · VISUALS UNCHANGED', font=font(18, True), fill=MUTED)
    else:
        paragraph(draw, segment['title'], (58,121), 31, 1740, fill=INK, bold=True)
        footer = 'ACTUAL APP RECORDING' if kind == 'app' else '5-SECOND SOURCE CLIP · REPEATED FOR REVIEW · ORIGINAL VISUALS + GENERATED SOUND'
        draw.text((58,883), footer, font=font(18, True), fill=MUTED)
    draw.rectangle((0,924,WIDTH,HEIGHT), fill='#080e18')
    draw.line((48,924,1872,924), fill='#26354a', width=2)
    draw.text((48,932), 'NARRATION GUIDE · estimated caption timing · no recorded voice', font=font(18), fill=MUTED)
    draw.text((1680,932), f"{segment['index']:02d} / {segment.get('total_segments', 9):02d}", font=font(18, True), fill=ACCENT)
    canvas.save(path)


def caption_chunks(outline):
    captions = []
    for segment in outline['segments']:
        words = segment['narration'].split()
        groups = max(1, math.ceil(len(words)/11))
        duration = segment['end_seconds']-segment['start_seconds']
        for i in range(groups):
            first = round(i*len(words)/groups)
            last = round((i+1)*len(words)/groups)
            group = words[first:last]
            mid = math.ceil(len(group)/2)
            captions.append({'start': segment['start_seconds']+duration*first/len(words),
                             'end': segment['start_seconds']+duration*last/len(words),
                             'text': ' '.join(group[:mid])+'\n'+' '.join(group[mid:])})
    return captions


def ass_time(seconds):
    value = max(0, round(seconds*100))
    return f'{value//360000}:{value//6000%60:02d}:{value//100%60:02d}.{value%100:02d}'


def write_ass(path, captions, start, end):
    header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Guide,Segoe UI,34,&H00F5F5F5,&H00F5F5F5,&H00180E08,&H00180E08,0,0,0,0,100,100,0,0,1,1,0,2,80,80,25,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    rows = []
    for caption in captions:
        left, right = max(start,caption['start']), min(end,caption['end'])
        if right <= left:
            continue
        text = caption['text'].replace('\\','').replace('{','(').replace('}',')').replace('\n',r'\N')
        rows.append(f'Dialogue: 0,{ass_time(left-start)},{ass_time(right-start)},Guide,,0,0,0,,{text}')
    path.write_text(header+'\n'.join(rows)+'\n', encoding='utf-8')


def filter_path(path):
    return str(path.resolve()).replace('\\','/').replace(':',r'\:').replace("'",r"\'")


def build_shots(outline):
    validate_outline(outline)
    shots = []
    for segment in outline['segments']:
        start, end, index = segment['start_seconds'], segment['end_seconds'], segment['index']
        if index == 1:
            chunks = [('intro', start, start+5), ('app',start+5,end)]
        elif index == 2:
            chunks = [('roles',start,start+5), ('app',start+5,end)]
        elif len(outline['segments']) == 6 and index == 3:
            chunks = [('app',start,start+10), ('app',start+10,start+22), ('app',start+22,start+29), ('app',start+29,end)]
        elif len(outline['segments']) == 6 and index == 4:
            chunks = [('app',start,start+15), ('output',start+15,end)]
        elif len(outline['segments']) == 6 and index == 5:
            chunks = [('app',start,start+15), ('app',start+15,end)]
        elif len(outline['segments']) == 6 and index == 6:
            chunks = [('outro',start,end)]
        elif index == 7:
            chunks = [('app',start,start+7), ('output',start+7,end)]
        elif index == 9:
            chunks = [('outro',start,end)]
        else:
            chunks = [('app',start,end)]
        for kind,left,right in chunks:
            if right <= left or left < start or right > end:
                raise ValueError('Timed sections are too short for the demo shot layout')
            shots.append({'segment':segment,'kind':kind,'start':left,'end':right})
    return shots


def render_shot(ffmpeg, shot, assets, capture, capture_info, capture_map, captions, github, mix, frontend=FRONTEND):
    segment, kind = shot['segment'], shot['kind']
    tag = f"{segment['index']:02d}_{shot['start']:03d}_{kind}"
    duration = shot['end']-shot['start']
    background, subtitles, target = assets/f'{tag}.png', assets/f'{tag}.ass', assets/f'{tag}.mp4'
    card(background,segment,kind,github)
    write_ass(subtitles,captions,shot['start'],shot['end'])
    args = [ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-threads','2',
            '-loop','1','-framerate',str(FPS),'-i',background]
    filters, previous, input_index = [], '0:v', 1
    excerpt = None
    overlays = []
    if kind == 'app':
        entry = capture_map.get(f"{segment['index']}_{shot['start']}", capture_map.get(str(segment['index']), {}))
        capture = Path(entry.get('capture', capture))
        capture_info = probe(ffmpeg, capture)
        default_start = (segment['start_seconds']/SECONDS)*max(0,capture_info['seconds']-20)
        source_start = float(entry.get('start',default_start))
        requested = float(entry.get('duration',duration))
        available = min(requested,capture_info['seconds']-source_start)
        if not math.isfinite(source_start) or not math.isfinite(requested) or source_start<0 or available<=0:
            raise ValueError(f'Capture map outside source for segment {segment["index"]}')
        args += ['-ss',str(source_start),'-t',str(min(duration,available)),'-threads','2','-i',capture]
        crop = validate_crop(entry.get('crop'), capture_info)
        prefix = ''
        if crop:
            x,y,w,h = crop
            prefix=f'crop={w}:{h}:{x}:{y},'
        overlays.append((input_index,48,172,1824,696,prefix+f'tpad=stop_mode=clone:stop_duration={duration},'))
        excerpt = {'start_seconds':source_start,'available_seconds':available,'crop':crop,
                   'held_final_frame':available<duration,'source':str(capture), 'source_sha256':sha256(capture)}
        input_index += 1
    elif kind == 'roles':
        for i,role in enumerate(ROLE_IDS):
            source=frontend/f'public/videos/agent-{role}.mp4'
            if not source.is_file():
                raise FileNotFoundError(source)
            args += ['-stream_loop','-1','-threads','2','-i',source]
            overlays.append((input_index,80+(i%3)*600,302+(i//3)*266,550,208,''))
            input_index += 1
    else:
        args += ['-stream_loop','-1','-threads','2','-i',mix]
        box = (48,172,1824,696) if kind=='output' else (1260,365,580,380)
        overlays.append((input_index,*box,''))
    for i,(source,x,y,width,height,prefix) in enumerate(overlays):
        filters.append(f'[{source}:v]{prefix}fps={FPS},scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x111c2c,setsar=1,setpts=PTS-STARTPTS[media{i}]')
        filters.append(f'[{previous}][media{i}]overlay={x}:{y}:eof_action=repeat[composite{i}]')
        previous=f'composite{i}'
    filters.append(f"[{previous}]ass=filename='{filter_path(subtitles)}',format=yuv420p[video]")
    args += ['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[video]',
             '-an','-frames:v',str(round(duration*FPS)),'-r',str(FPS),'-c:v','libx264',
             '-preset','veryfast','-crf','18','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',target]
    execute(args)
    return target, {'kind':kind,'start_seconds':shot['start'],'end_seconds':shot['end'],
                    'frames':round(duration*FPS),'capture_excerpt':excerpt}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True,help='Actual app-only recording')
    parser.add_argument('--capture-map',type=Path)
    parser.add_argument('--script',type=Path,required=True,help='Timed narration text covering exactly 0–180 seconds')
    parser.add_argument('--github',default='')
    parser.add_argument('--ffmpeg')
    parser.add_argument('--mix',type=Path,required=True,help='Existing sound-enhanced MP4; never generated by this tool')
    parser.add_argument('--output',type=Path,required=True,help='New destination MP4; existing outputs are preserved')
    parser.add_argument('--frontend',type=Path,default=FRONTEND,help='App repository containing public/videos/agent-*.mp4')
    parser.add_argument('--test-title',action='store_true',help='Render only a two-second title preview; never a full demo')
    args=parser.parse_args()
    try:
        validate_github_url(args.github)
    except ValueError as error:
        parser.error(str(error))
    for name in ('script','mix','capture'):
        source=getattr(args,name).resolve()
        if not source.is_file():
            parser.error(f'--{name} must identify an existing file')
        setattr(args,name,source)
    args.frontend=args.frontend.resolve()
    args.output=args.output.resolve()
    if args.output.suffix.lower()!='.mp4':
        parser.error('--output must name a new .mp4 file')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Preserving existing deliverable or manifest; choose another --output')
    outline=parse_timed_script(args.script.read_text(encoding='utf-8'))
    shots=build_shots(outline)
    capture_map=load_capture_map(args.capture_map.resolve() if args.capture_map else None)
    allowed_keys={str(s['segment']['index']) for s in shots if s['kind']=='app'} | {
        f"{s['segment']['index']}_{s['start']}" for s in shots if s['kind']=='app'}
    if set(capture_map)-allowed_keys:
        parser.error('Capture map contains a key that does not identify an app shot')
    ffmpeg=encoder_path(args.ffmpeg)
    segments=outline['segments']
    capture_info=probe(ffmpeg,args.capture)
    mix_info=probe(ffmpeg,args.mix)
    if not mix_info['audio']:
        parser.error('--mix must contain the existing generated sound track')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    assets=args.output.parent/f'{args.output.stem}.edit_assets_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()%1000000000:09d}'
    assets.mkdir(exist_ok=False)
    captions=caption_chunks(outline)
    if args.test_title:
        shot={'segment':segments[0],'kind':'intro','start':0,'end':2}
        target,report=render_shot(ffmpeg,shot,assets,args.capture,capture_info,{},captions,args.github,args.mix,args.frontend)
        with target.open('rb') as source, args.output.open('xb') as destination:
            shutil.copyfileobj(source,destination)
        print(json.dumps({'title_test':str(args.output),'background':str(assets/'01_000_intro.png'),'video':probe(ffmpeg,args.output),'frames_expected':60},indent=2))
        return
    if not capture_map:
        print('NOTICE: no capture map; excerpts are distributed across the recording. Review scene-to-script alignment.',flush=True)
    outputs,reports=[],[]
    for shot in shots:
        print(f'Rendering {shot["start"]:03d}–{shot["end"]:03d}s: {shot["kind"]}',flush=True)
        target,report=render_shot(ffmpeg,shot,assets,args.capture,capture_info,capture_map,captions,args.github,args.mix,args.frontend)
        outputs.append(target);reports.append(report)
    concat=assets/'shots.txt'
    concat.write_text(''.join("file '"+str(p.resolve()).replace('\\','/').replace("'","'\\''")+"'\n" for p in outputs),encoding='utf-8')
    silent=assets/'edited_silent.mp4'
    execute([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-f','concat','-safe','0','-i',concat,'-c','copy','-an',silent])
    audio_start,audio_end=(95,130) if len(segments)==6 else (100,142)
    audio_duration=audio_end-audio_start
    audio=f'[1:a]aresample=48000,atrim=duration={audio_duration},asetpts=PTS-STARTPTS,volume=0.8,afade=t=in:st=0:d=0.15,afade=t=out:st={audio_duration-.15}:d=0.15,adelay={audio_start*1000}:all=1,apad=whole_dur=180,atrim=duration=180[audio]'
    if args.output.exists():
        raise FileExistsError(f'Preserving existing deliverable; choose another --output: {args.output}')
    execute([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-i',silent,'-stream_loop','-1','-threads','2','-i',args.mix,
             '-filter_complex_threads','2','-filter_complex',audio,'-map','0:v:0','-map','[audio]','-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-threads','4','-t','180','-movflags','+faststart',args.output])
    copied=execute([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-i',args.output,'-map','0:v:0','-c:v','copy','-an','-progress','pipe:1','-nostats','-f','null','-'])
    frames=int(re.findall(r'^frame=(\d+)$',copied.stdout,re.M)[-1])
    info=probe(ffmpeg,args.output)
    if frames!=FPS*SECONDS or info['seconds']!=SECONDS or (info['width'],info['height'])!=(WIDTH,HEIGHT):
        raise RuntimeError(f'Output verification failed: {info}, {frames} frames')
    manifest={'output':str(args.output.resolve()),'sha256':sha256(args.output),'width':WIDTH,'height':HEIGHT,'fps':FPS,'frames':frames,'duration_seconds':SECONDS,
              'capture':str(args.capture.resolve()),'capture_sha256':sha256(args.capture),'capture_info':capture_info,'capture_mapping':'explicit' if capture_map else 'distributed_requires_editorial_review',
              'github_url':args.github,'mix':str(args.mix.resolve()),'mix_sha256':sha256(args.mix),'audio_active_seconds':[audio_start,audio_end],
              'narration_script':str(args.script),'narration_script_sha256':sha256(args.script),'spoken_word_count':outline['spoken_word_count'],
              'frontend':str(args.frontend),'editor_sha256':sha256(__file__),
              'narration_audio':False,'caption_timing':'Heuristic per spoken word; narration guide, not measured voice alignment',
              'role_artwork':'Original public/videos/agent-*.mp4 illustrations, not model output',
              'new_provider_calls':0,'encoding_threads_max':4,'shots':reports,'visual_quality_review':'Pending inspection of rendered frames and playback'}
    with args.output.with_suffix('.json').open('x',encoding='utf-8') as handle:
        handle.write(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
