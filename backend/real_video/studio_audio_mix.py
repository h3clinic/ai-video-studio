"""Mix existing agent audio with a source clip; no model/API calls or visual edits."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def validate_gains(gains):
    if not isinstance(gains,list) or not 3 <= len(gains) <= 4 or any(
            type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1 for x in gains):
        raise ValueError('Three or four finite gains in [0,1] required')
    return gains


def confined_file(root, relative):
    if not isinstance(relative,str):raise ValueError('Missing source artifact')
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Source artifact unavailable')
    return path


def mix_scene_audio(source_job, output, *, root, gains, source_video=None):
    from .studio_swarm import SOURCE
    from .elevenlabs_client import mp3_duration
    import imageio_ffmpeg
    root,output=Path(root),Path(output)
    validate_gains(gains)
    if source_job.get('kind')!='agent_swarm':raise ValueError('Scene team required')
    workers=[w for w in source_job.get('workers',[]) if w.get('roleId')=='sound-agents']
    if len(workers)!=len(gains) or any(w.get('status')!='completed' or w.get('execution')!='elevenlabs_sound' for w in workers):
        raise ValueError('Every sound layer must be generated before mixing')
    source=confined_file(root,source_video or SOURCE)
    tracks=[confined_file(root,w.get('output')) for w in workers]
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    track_info=[]
    for worker,path,gain in zip(workers,tracks,gains):
        content=path.read_bytes()
        sha=hashlib.sha256(content).hexdigest()
        if worker.get('sound',{}).get('sha256')!=sha:raise ValueError('Audio checksum mismatch')
        track_info.append(dict(workerId=worker['id'],name=worker['name'],partId=worker['partId'],
            source=str(path.relative_to(root)),sha256=sha,gain=gain,duration_seconds=mp3_duration(content)))
    output.mkdir(parents=True,exist_ok=False)
    target=output/'scene_with_sound.mp4'
    args=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(source)]
    for track in tracks:args+=['-i',str(track)]
    filters=[f'[{i+1}:a]volume={gain:.6f},apad[a{i}]' for i,gain in enumerate(gains)]
    # Explicit gain control, unnormalized sum, then conservative output ceiling.
    filters+=[''.join(f'[a{i}]' for i in range(len(gains)))+f'amix=inputs={len(gains)}:duration=longest:normalize=0,alimiter=limit=0.95:level=0:latency=1[mix]']
    args+=['-filter_complex',';'.join(filters),'-map','0:v:0','-map','[mix]',
        '-map_metadata','-1','-c:v','copy','-c:a','aac','-b:a','192k','-threads','1','-shortest','-t','10','-movflags','+faststart',str(target)]
    started=time.monotonic()
    result=subprocess.run(args,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=45)
    if result.returncode or not target.is_file():raise RuntimeError('Audio mix failed; no provider retry')
    elapsed=time.monotonic()-started
    report=dict(sourceJobId=source_job['id'],source_video=str(source.relative_to(root)),source_sha256=source_hash,
        output=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),tracks=track_info,
        audio_modified=True,visuals_modified=False,gaussians_modified=False,new_api_calls=0,
        model_generation_seconds=0,gaussian_fitting_seconds=0,rasterization_seconds=0,
        audio_mix_and_mux_seconds=elapsed,video_reencoded=False,
        alignment='All layers start at t=0; no event-level audiovisual synchronization verified',
        review='Generated stems mixed onto unchanged Runway source; not a Gaussian apple replacement')
    (output/'mix_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
