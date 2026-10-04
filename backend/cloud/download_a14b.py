"""Download only pinned official A14B components; no model execution or billing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

REPO = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
REVISION = '596658fd9ca6b7b71d5057529bbf319ecbc61d74'
PARTS = ('transformer/', 'transformer_2/', 'text_encoder/', 'vae/', 'tokenizer/', 'scheduler/')


def run(cache, out):
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi, snapshot_download
    out=Path(out); out.mkdir(parents=True, exist_ok=False)
    cache=Path(cache); cache.mkdir(parents=True, exist_ok=True)
    api=HfApi(token=False)
    info=api.model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha!=REVISION: raise ValueError('Model revision mismatch')
    files=[f for f in info.siblings if f.rfilename=='model_index.json' or f.rfilename.startswith(PARTS)]
    size=sum(f.size for f in files)
    if size > 130_000_000_000: raise ValueError('Unexpected pinned model size')
    if shutil.disk_usage(cache).free < size + 20*1024**3: raise ValueError('Insufficient temporary disk')
    record=dict(repo_id=REPO, revision=REVISION, expected_bytes=size, status='downloading', files=[])
    def save(): (out/'download_manifest.json').write_text(json.dumps(record,indent=2))
    save(); started=time.monotonic()
    local=snapshot_download(REPO, revision=REVISION, cache_dir=str(cache),
        allow_patterns=[f.rfilename for f in files], max_workers=8, token=False)
    record.update(local_path=local,download_seconds=time.monotonic()-started,status='verifying')
    save()
    for f in files:
        path=Path(local)/f.rfilename
        if path.stat().st_size!=f.size: raise ValueError('Downloaded file size mismatch')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
        actual=h.hexdigest()
        expected=f.lfs.sha256 if f.lfs is not None else None
        if expected is not None and actual!=expected: raise ValueError('Downloaded model digest mismatch')
        record['files'].append(dict(path=f.rfilename,bytes=f.size,sha256=actual,lfs_verified=expected is not None))
        print(json.dumps(dict(verified=f.rfilename,bytes=f.size)),flush=True)
    record.update(status='verified',total_seconds=time.monotonic()-started)
    save(); print(json.dumps(dict(status='verified',local_path=local)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cache',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); run(a.cache,a.out)
