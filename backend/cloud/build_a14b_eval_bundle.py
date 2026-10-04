"""Immutable allowlisted evaluation bundle, no secrets or future source video."""
import hashlib
import argparse
import json
from pathlib import Path
import zipfile
from cloud.build_a14b_bundle import ROOT, FILES

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New immutable path inside the project')
    args=parser.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT): raise ValueError('Output must remain inside the project')
    selected=[(name,ROOT/name) for name in FILES if not name.endswith('.mp4')]
    selected.append(('cloud/evaluate_a14b.py',ROOT/'cloud/evaluate_a14b.py'))
    selected.append(('cloud/eval_preflight.py',ROOT/'cloud/eval_preflight.py'))
    base=ROOT/'artifacts/cloud/a14b_pilot_v1/pilot_output_repaired'
    for name in ('anchor.png','low_expert_adapter.pt','low_expert_adapter.pt.sha256.json','gaussian_anchor.pt','gaussian_anchor.pt.sha256.json'):
        selected.append(('asset/'+name,base/name))
    manifest=[]
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for name,path in selected:
            assert path.resolve().is_relative_to(ROOT) and not path.is_symlink()
            data=path.read_bytes()
            z.writestr(name,data)
            manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for r in manifest: assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,files=len(manifest))))
