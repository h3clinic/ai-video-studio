"""Additional deterministic temporal windows from original training videos only."""
import json
from pathlib import Path
import time
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from .representation import encode_clip
from .checkpoint_io import digest,save_inference_checkpoint,load_verified,keep_windows_awake

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent.parent/'work/real_video'
PROTOCOL=ROOT/'artifacts/real_video/data_protocol.json'


def validate_windows(packet,metadata,labels,protocol_sha256):
    if packet['dataset_protocol_sha256']!=protocol_sha256: raise ValueError('Dataset protocol mismatch')
    allowed={m['file']:m for m in metadata if m['split']=='train'}
    if packet['labels']!=labels: raise ValueError('Label vocabulary mismatch')
    if tuple(packet['fields'].shape)!=(len(packet['metadata']),9,8,32,32): raise ValueError('Invalid window tensor shape')
    if not torch.isfinite(packet['fields']).all(): raise ValueError('Nonfinite window data')
    seen=set()
    for m in packet['metadata']:
        source=allowed.get(m['file'])
        if source is None or m['split']!='train': raise ValueError('Non-training source in extra windows')
        if any(m[key]!=source[key] for key in ('group','label','sha256')): raise ValueError('Source identity mismatch')
        indices=m['frame_indices']
        if len(indices)!=8 or indices!=sorted(indices) or min(indices)<0 or max(indices)>=m['source_frames']:
            raise ValueError('Invalid frame indices')
        identity=(m['file'],tuple(indices))
        if identity in seen or indices==source['frame_indices']: raise ValueError('Duplicate temporal window')
        seen.add(identity)
    return True


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1)
    destination=WORK/'windows_train.pt'
    data=torch.load(WORK/'dataset.pt',map_location='cpu',weights_only=True)
    protocol_hash=digest(PROTOCOL)
    if destination.exists():
        existing=load_verified(destination)
        validate_windows(existing,data['metadata'],data['labels'],protocol_hash)
        print('Verified existing training-window cache; not replacing it.'); return
    train=[m for m in data['metadata'] if m['split']=='train']
    tensors=[]; records=[]; failures=[]; start=time.perf_counter()
    for number,source in enumerate(train):
        path=WORK/'clips'/source['file']
        if digest(path)!=source['sha256']: raise ValueError(f'Changed source file: {path}')
        try:
            reader=imageio.get_reader(str(path),format='ffmpeg',input_params=['-threads','1'])
            try:
                fps=reader.get_meta_data()['fps']
                frames=[cv2.resize(frame,(64,64),interpolation=cv2.INTER_AREA) for frame in reader]
            finally: reader.close()
            stride=max(1,round(fps/8)); maximum=max(0,len(frames)-1-7*stride)
            starts=sorted(set(np.linspace(0,maximum,4).round().astype(int).tolist()))
            for offset in starts:
                indices=np.minimum(offset+np.arange(8)*stride,len(frames)-1).tolist()
                if indices==source['frame_indices']: continue
                fields=encode_clip(np.stack([frames[i] for i in indices]))
                tensors.append(torch.from_numpy(fields).half())
                records.append(dict(source,frame_indices=indices,source_frames=len(frames)))
        except Exception as error:
            failures.append(dict(file=source['file'],error=str(error)))
        if (number+1)%25==0: print(json.dumps(dict(training_sources_processed=number+1,total_sources=len(train),extra_windows=len(records))),flush=True)
        time.sleep(0.01)
    packet=dict(fields=torch.stack(tensors),metadata=records,labels=data['labels'],dataset_protocol_sha256=protocol_hash)
    validate_windows(packet,data['metadata'],data['labels'],protocol_hash)
    checksum=save_inference_checkpoint(packet,destination)
    report=dict(training_source_clips=len(train),extra_temporal_windows=len(records),unique_source_files=len({m['file'] for m in records}),
                validation_or_test_sources_used=0,failures=failures,seconds=time.perf_counter()-start,sha256=checksum,
                policy='Up to four evenly spaced one-second windows per original training clip, excluding its existing window. Same source groups, not new independent videos.')
    report_path=ROOT/'artifacts/real_video/weight_design/temporal_windows.json'
    with report_path.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    with keep_windows_awake(): main()
