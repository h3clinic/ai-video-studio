"""Explicit code/source allowlist; never uploads or purchases resources."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['cloud/wan_a14b_experiment.py', 'cloud/download_a14b.py',
         'cloud/a14b_preflight.py', 'cloud/a14b_requirements.txt'] + [
    'real_video/'+name+'.py' for name in ('__init__','checkpoint_io',
    'wan_part_control','wan_cat_memory','wan_temporal_control',
    'gaussian_latent_memory','gaussian_part_protocol','latent_track_memory')
] + ['artifacts/real_video/runway_agents/donkey_orange_v1/source/video.mp4']

if __name__ == '__main__':
    out=ROOT/'artifacts/cloud/wan_a14b_bundle_v1.zip'
    out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            p=(ROOT/name).resolve(strict=True)
            assert p.is_relative_to(ROOT) and not (ROOT/name).is_symlink()
            data=p.read_bytes()
            records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
            z.writestr(name,data)
        z.writestr('bundle_manifest.json',json.dumps(records,indent=2))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for r in records:
            assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,files=records)))
