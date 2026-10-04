"""All validation windows, controls, allocation accounting and media verification."""
import argparse
import json
from pathlib import Path
import cv2
import torch
from .animal_control_graph import AnimalControlGraph
from .train_animal_visual import DATA,CACHE,OUT,BASE,evaluate
from .checkpoint_io import load_verified,digest,keep_windows_awake


def main():
    global DATA,CACHE,OUT,BASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA); parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--root',type=Path,default=OUT); parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--seed-root',type=Path,default=BASE/'seeds'); args=parser.parse_args()
    DATA=args.data; CACHE=args.cache; OUT=args.root; BASE=args.base
    if (OUT/'audit.json').exists(): raise FileExistsError('Preserve audit')
    torch.set_num_threads(4)
    with keep_windows_awake():
        data=load_verified(DATA); cache=load_verified(CACHE); selected=load_verified(OUT/'model.pt'); model=AnimalControlGraph().cuda().eval(); model.load_state_dict(selected['model'])
        if cache['source_tracks_sha256']!=digest(DATA) or cache['metadata']!=data['metadata']: raise ValueError('Visual/control cache provenance mismatch')
        val=[i for i,m in enumerate(data['metadata']) if m['split']=='validation']
        scores={mode:evaluate(model,data,cache,val,mode) for mode in ['learned','velocity','frozen','damped_velocity','zero_hidden']}
        seed=load_verified(args.seed_root/'cat.pt'); memory={name:seed[name].numel()*seed[name].element_size() for name in ['points','index','weight']}
        ctrl={k:v[None].cuda() for k,v in seed['controls'].items()}
        with torch.no_grad():
            state=model.initialize(ctrl['position'],ctrl['angle'],ctrl['visibility'],ctrl['colour'],ctrl['adjacency'])
            sizes=[]
            for _ in range(128):
                state=model.step(state); sizes.append(sum(v.numel()*v.element_size() for v in state.values()))
        stability=dict(steps=128,finite=all(bool(torch.isfinite(v).all()) for v in state.values()),state_size_constant=len(set(sizes))==1,
                       caveat='Shape/finiteness only, not long-horizon quality or accurate memory.')
        memory['control_state_bytes']=sizes[0]; memory['points_bindings_and_control_state_bytes']=sum(memory.values())
        memory['previous_dense_forecast_points_uv_and_state_bytes']=3828160+765632+322560
        memory['note']='Excludes model, background, rendering intermediates and outputs. Smaller control state but bigger bindings; not overall memory reduction.'
        videos=[]
        for root in [BASE,OUT]:
            for path in (root/'evaluation').glob('*.mp4'):
                cap=cv2.VideoCapture(str(path)); count=0
                while True:
                    ok,frame=cap.read()
                    if not ok: break
                    count+=1
                cap.release()
                if count!=13: raise ValueError(f'Unexpected frame count: {path} {count}')
                videos.append(dict(path=str(path),frames=count,sha256=digest(path)))
        result=dict(validation=scores,memory=memory,stability=stability,videos=videos,wan_weights_changed=False,test_sequences=0,damped_velocity_factor=.9,
                    caveat='All validation reused during development; four sequences, not independent final evidence. Conditional motion only; no unseen-surface synthesis.')
        (OUT/'audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps(dict(means={m:s['means'] for m,s in scores.items()},memory=memory,videos=len(videos))),flush=True)


if __name__=='__main__': main()
