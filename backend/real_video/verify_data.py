"""Verify split isolation, source duplication and train-only normalization."""
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent.parent/'work/real_video'


def main():
    d=torch.load(WORK/'dataset.pt',weights_only=True,map_location='cpu')
    groups={s:{m['group'] for m in d['metadata'] if m['split']==s} for s in ('train','validation','test')}
    hashes={s:{m['sha256'] for m in d['metadata'] if m['split']==s} for s in groups}
    pairs=[('train','validation'),('train','test'),('validation','test')]
    group_overlap={f'{a}/{b}':sorted(groups[a]&groups[b]) for a,b in pairs}
    hash_overlap={f'{a}/{b}':sorted(hashes[a]&hashes[b]) for a,b in pairs}
    ids=[i for i,m in enumerate(d['metadata']) if m['split']=='train']
    expected_mean=d['fields'][ids].float().mean((0,2,3,4),keepdim=True)
    expected_std=d['fields'][ids].float().std((0,2,3,4),keepdim=True).clamp_min(0.05)
    torch.testing.assert_close(expected_mean,d['mean'])
    torch.testing.assert_close(expected_std,d['std'])
    passed=not any(group_overlap.values()) and not any(hash_overlap.values())
    result=dict(group_counts={k:len(v) for k,v in groups.items()},group_overlap=group_overlap,
                identical_source_overlap=hash_overlap,train_only_normalization=True,passed=passed,
                caveat='Group IDs and exact hashes checked, not semantic/perceptual duplicate detection across different source groups.')
    (ROOT/'artifacts/real_video/split_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
