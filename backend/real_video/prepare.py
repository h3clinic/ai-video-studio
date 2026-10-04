"""Download a pinned real UCF101 subset and freeze group-disjoint splits before training."""
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import numpy as np
import torch
from external_validation import remote,file_hash
from .representation import SIZE,FRAMES,GRID,encode_clip

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent.parent/'work/real_video'
OUT=ROOT/'artifacts/real_video'
REV='b9984b8d2a95e4a1879e1b071e9433858d0bc24a'
SHA='e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb'


def main():
    import cv2
    import imageio.v2 as imageio
    WORK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    if (WORK/'dataset.pt').exists():
        print('Prepared dataset already exists; refusing to replace its frozen split.'); return
    source=WORK/'UCF101_subset.tar.gz'
    if not source.exists():
        url=f'https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/{REV}/UCF101_subset.tar.gz'
        with remote(url) as response,source.open('wb') as f:
            while data:=response.read(8*1024*1024): f.write(data)
    if file_hash(source)!=SHA: raise ValueError('UCF source hash mismatch')
    clips=WORK/'clips'; clips.mkdir(exist_ok=True)
    entries=[]
    with tarfile.open(source,'r:*') as archive:
        for item in archive:
            if not item.isfile() or not item.name.endswith('.avi'): continue
            name=Path(item.name).name
            match=re.fullmatch(r'v_(\w+)_g(\d+)_c(\d+)\.avi',name)
            if not match: raise ValueError(f'Unexpected video name {name}')
            path=clips/name
            if path.exists(): continue
            path.write_bytes(archive.extractfile(item).read())
    filenames=sorted(clips.glob('*.avi'))
    for path in filenames:
        match=re.fullmatch(r'v_(\w+)_g(\d+)_c(\d+)\.avi',path.name)
        entries.append(dict(file=path.name,label=match[1],group=f'{match[1]}:g{match[2]}',sha256=file_hash(path)))
    labels=sorted({e['label'] for e in entries})
    split_by_group={}
    for label in labels:
        groups=sorted({e['group'] for e in entries if e['label']==label},key=lambda g:hashlib.sha256(f'20260929:{g}'.encode()).hexdigest())
        count=max(1,round(len(groups)*0.15))
        for i,g in enumerate(groups): split_by_group[g]='test' if i<count else 'validation' if i<2*count else 'train'
    for e in entries: e['split']=split_by_group[e['group']]
    manifest=dict(dataset='sayakpaul/ucf101-subset',revision=REV,archive_sha256=SHA,labels=labels,
                  split_policy='Within each class, SHA256(20260929:class:group) order: first ~15% groups test, next ~15% validation, remainder train. Replaces archive clip split to prevent same-source group leakage. Not official full UCF splits.',
                  clip_policy='8 frames at approximately 8fps from a hash-selected start; resize to 64x64 (aspect distortion disclosed).',
                  representation='1024 flow-tracked planar anisotropic Gaussians, SO(2) unit axes embedded in SO(3). No recovered depth.',
                  entries=entries,failures=[],scope='Real recorded action videos, not ModelScope-generated training targets')
    (OUT/'data_protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    fields=[]; videos=[]; metadata=[]
    start=time.perf_counter()
    for i,entry in enumerate(entries):
        try:
            reader=imageio.get_reader(str(clips/entry['file']),format='ffmpeg')
            fps=reader.get_meta_data()['fps']
            frames=[cv2.resize(frame,(SIZE,SIZE),interpolation=cv2.INTER_AREA) for frame in reader]
            reader.close()
            stride=max(1,round(fps/8))
            maximum=max(0,len(frames)-1-(FRAMES-1)*stride)
            offset=int(hashlib.sha256(entry['file'].encode()).hexdigest()[:8],16)%(maximum+1)
            indices=np.minimum(offset+np.arange(FRAMES)*stride,len(frames)-1)
            selected=np.stack([frames[j] for j in indices])
            fields.append(torch.from_numpy(encode_clip(selected)).half())
            videos.append(torch.from_numpy(selected.copy()))
            metadata.append(dict(entry,frame_indices=indices.tolist(),source_fps=fps))
        except Exception as error:
            manifest['failures'].append(dict(file=entry['file'],error=str(error)))
        if (i+1)%25==0: print(json.dumps(dict(decoded=i+1,total=len(entries),elapsed=round(time.perf_counter()-start))),flush=True)
    tensor=torch.stack(fields)
    train=torch.tensor([e['split']=='train' for e in metadata])
    mean=tensor[train].float().mean((0,2,3,4),keepdim=True)
    std=tensor[train].float().std((0,2,3,4),keepdim=True).clamp_min(0.05)
    torch.save(dict(fields=tensor,videos=torch.stack(videos),metadata=metadata,labels=labels,mean=mean,std=std),WORK/'dataset.pt')
    manifest['decoded_counts']={s:sum(e['split']==s for e in metadata) for s in ('train','validation','test')}
    manifest['dataset_tensor_sha256']=file_hash(WORK/'dataset.pt')
    (OUT/'data_protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(labels=labels,counts=manifest['decoded_counts'],failures=len(manifest['failures'])),indent=2))


if __name__=='__main__': main()
