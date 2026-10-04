"""All predeclared source-conditioned Gaussian replay cases; no cherry picking."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import torch
import cv2
from .checkpoint_io import digest,keep_windows_awake
from .replay_fit import fit,evaluate

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(2)
    protocol=json.loads((ROOT/'artifacts/external_validation/protocol.json').read_text())
    old=json.loads((ROOT/'artifacts/external_validation/external_results.json').read_text())
    expected={(r['model'],r['uuid']):r['sha256'] for r in old['cases'] if r['status']=='decoded'}
    args.out.mkdir(parents=True,exist_ok=False)
    sources=[]
    for index,case in enumerate(protocol['selected'],1):
        path=ROOT/'artifacts/external_validation/modelscope'/f"{case['uuid']}.mp4"
        checksum=digest(path)
        if checksum!=expected[('modelscope',case['uuid'])]: raise ValueError('Source hash differs')
        sources.append(dict(index=index,path=str(path),sha256=checksum,**case))
    tasks=[dict(case=s,grid=g,terms=16) for s in sources for g in (32,64,96)]
    tasks += [dict(case=s,grid=64,terms=k) for s in sources[:2] for k in (4,8)]
    manifest=dict(kind='Source-conditioned reconstruction development; NOT noise generation',sources=sources,
                  settings=dict(steps=1200,lr=0.003,seed=58103,anchored=True,radius=7),
                  task_count=len(tasks),protocol_sha256=digest(ROOT/'research/SCALING_PROTOCOL.md'))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    shutil.copytree(Path(__file__).parent,args.out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    results=[]
    for task in tasks:
        s=task['case']; grid=task['grid']; terms=task['terms']
        name=f"case_{s['index']:02d}_g{grid}_k{terms}"
        out=args.out/name; out.mkdir()
        record=dict(name=name,case=s['index'],uuid=s['uuid'],prompt=s['prompt'],grid=grid,terms=terms)
        print(json.dumps(dict(starting=name)),flush=True)
        try:
            packet=fit(Path(s['path']),out,steps=1200,grid=grid,terms=terms,anchored=True,lr=0.003)
            evaluate(packet,Path(s['path']),out)
            metrics=json.loads((out/'metrics.json').read_text())
            record.update(status='completed',metrics=metrics)
        except Exception as error:
            record.update(status='failed',error=repr(error))
            (out/'failure.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        results.append(record)
        (args.out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps(dict(finished=name,status=record['status'])),flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(dict(completed=sum(r['status']=='completed' for r in results),total=len(tasks))),flush=True)


if __name__=='__main__':
    with keep_windows_awake(): main()
