"""Controlled texture-retention trials. No neural inference or motion training."""
import json
from pathlib import Path
import torch
from . import hunyuan_multiview as paint
from .checkpoint_io import keep_windows_awake, load_verified

ROOT=Path('artifacts/real_video/hunyuan_gaussian/detail_ablation')

def main():
    source=ROOT.parent/'v5_full_paint'
    paint.BASE=ROOT.parent/'v4_full'
    ROOT.mkdir(parents=True,exist_ok=True)
    reference=load_verified(source/'gaussian_fitted.pt')
    for name,sigma,fusion in [('narrow',.62,'mean'),('winner',.8,'winner'),('combined',.62,'winner')]:
        paint.ROOT=ROOT/name
        if not (paint.ROOT/'gaussian_fitted.pt').exists():
            paint.bake(sigma,fusion,source,diagnostics=False)
        candidate=load_verified(paint.ROOT/'gaussian_fitted.pt')
        for key in ['position','ids','face_id','barycentric']:
            if not torch.equal(candidate[key],reference[key]):raise AssertionError(f'{name}: changed {key}')
        if fusion=='mean' and not torch.equal(candidate['colour'],reference['colour']):
            raise AssertionError('Sigma-only trial changed color')
        if sigma==.8 and not torch.equal(candidate['covariance'],reference['covariance']):
            raise AssertionError('Fusion-only trial changed covariance')
        print(f'{name}: controlled state invariants pass',flush=True)
        if not (paint.ROOT/'camera_only_7s.mp4').exists():paint.inspect()
    (ROOT/'invariants.json').write_text(json.dumps({'passed':True,'variants':['narrow','winner','combined'],
        'unchanged':['position','ids','face_id','barycentric'],'quality_accepted':False,
        'scope':'Static appearance ablation; camera-only diagnostics; no model weight change.'},indent=2))

if __name__=='__main__':
    torch.set_num_threads(4)
    with keep_windows_awake():main()
