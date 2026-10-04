"""Independent moving-region audit and explicitly labelled cat diagnostic."""
import argparse
import json
from pathlib import Path
import time
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import torch
from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake
from .learned_motion import LearnedGaussianMotion
from .motion_coherence import edges,make_context
from .train_coherent_motion import BASE,DATA
from .demo_learned_motion import predict,ROOT as OLD
from .dense_gaussian_seed import SOURCE,OUT as SEEDS
from .edit_gaussian_memory import splat_layer,array_image
from .gaussian_motion_memory import select_cat


@torch.no_grad()
def moving_audit(checkpoint,data,indices):
    model=LearnedGaussianMotion().cuda().eval(); model.load_state_dict(checkpoint['model']); cases=[]
    for i in indices:
        seq=data['fields'][i:i+1].cuda(); moving=(seq[:,:2,2]-seq[:,:2,1]).norm(dim=1)>.005
        state=model.initialize(seq[:,:,0],seq[:,:,1],seq[:,:,2]); ctx=make_context(seq[:,:,0],seq[:,:,1],seq[:,:,2]); rows=[]
        for t in range(3,9):
            state=model.step(state)
            error=((state['fields'][:,:2]-seq[:,:2,t])*64).norm(dim=1)
            displacement=((state['fields'][:,:2]-seq[:,:2,2])*64).norm(dim=1)
            edge_mse=sum(((p-q)*64).square().sum(1).mean() for p,q in zip(edges(state['fields'][:,:2]),edges(seq[:,:2,t])))/2
            rows.append(dict(moving_epe=error[moving].mean().item() if moving.any() else None,
                             moving_displacement=displacement[moving].mean().item() if moving.any() else None,
                             all_epe=error.mean().item(),unweighted_edge_mse=edge_mse.item()))
        means={k:float(np.mean([r[k] for r in rows])) if rows[0][k] is not None else None for k in rows[0]}
        cases.append(dict(file=data['metadata'][i]['file'],group=data['metadata'][i]['group'],scores=means,
                          moving_fraction=moving.float().mean().item(),
                          mean_edge_weight=float(torch.cat([v.flatten() for v in ctx['edge_weights']]).mean()),
                          positive_cell_weight_fraction=(ctx['cell_weight']>0).float().mean().item()))
    means={k:float(np.mean([r['scores'][k] for r in cases if r['scores'][k] is not None])) for k in cases[0]['scores']}
    return dict(means=means,cases=cases)


def compare(candidate,reference):
    c,r=candidate['means'],reference['means']
    severe=[a['file'] for a,b in zip(candidate['cases'],reference['cases']) if a['scores']['all_epe']>1.1*b['scores']['all_epe']]
    checks=dict(moving_epe=c['moving_epe']<=1.02*r['moving_epe'],
                moving_displacement_min=c['moving_displacement']>=.8*r['moving_displacement'],
                moving_displacement_max=c['moving_displacement']<=1.2*r['moving_displacement'],
                unweighted_edges=c['unweighted_edge_mse']<=1.02*r['unweighted_edge_mse'],
                clip_regressions=len(severe)/len(candidate['cases'])<=.1)
    # Paired source-group bootstrap; descriptive validation uncertainty, not test inference.
    groups=sorted({a['group'] for a in candidate['cases']}); differences=[]
    for group in groups:
        differences.append(np.mean([a['scores']['all_epe']-b['scores']['all_epe']
                                    for a,b in zip(candidate['cases'],reference['cases']) if a['group']==group]))
    rng=np.random.default_rng(430399); values=np.asarray(differences)
    draws=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return dict(passed=all(checks.values()),checks=checks,severe_regression_files=severe,
                group_mean_epe_difference=float(values.mean()),group_bootstrap_95_interval=np.quantile(draws,[.025,.975]).tolist(),groups=len(groups))


@torch.no_grad()
def benchmark(checkpoint,seed):
    model=LearnedGaussianMotion().cuda().eval(); model.load_state_dict(checkpoint['model'])
    observations=seed['observed_coarse'].cuda(); state=model.initialize(observations[:,:,0],observations[:,:,1],observations[:,:,2],dt=.5)
    for _ in range(8): state=model.step(state,dt=.5)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); times=[]; sizes=[]
    for _ in range(128):
        torch.cuda.synchronize(); start=time.perf_counter(); state=model.step(state,dt=.5); torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-start)); sizes.append(sum(v.numel()*v.element_size() for v in state.values()))
    return dict(median_update_only_ms=float(np.median(times)),state_bytes=sizes[0],constant_size=len(set(sizes))==1,
                finite=all(bool(torch.isfinite(v).all()) for v in state.values()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                note='One warmed pass; excludes splat rendering, fitting, encoding and Wan generation. Not quality matched.')


def cat_evaluation(root,forecast):
    if (root/'cat_diagnostic.json').exists(): raise FileExistsError('Preserve cat diagnostic')
    with np.load(forecast/'forecast_1px/outputs.npz') as f: new={k:f[k] for k in f.files}
    with np.load(OLD/'forecast_1px/outputs.npz') as f: old={k:f[k] for k in f.files}
    cap=cv2.VideoCapture(str(SOURCE)); source=[]
    try:
        for _ in range(27):
            ok,frame=cap.read()
            if not ok: raise ValueError('Reference too short')
            source.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally: cap.release()
    masks=np.stack([select_cat(frame) for frame in source[2:]])
    seed=load_verified(SEEDS/'seed_1px.pt')
    with torch.no_grad(): bg=array_image(splat_layer(seed['background'].cuda())[0]).astype(np.float32)/255
    target=np.stack(source[2:]).astype(np.float32)/255*masks[...,None]+bg[None]*(1-masks[...,None])
    variants={'previous':old['learned'],'candidate':new['learned'],'frozen':new['frozen'],'zero_hidden':new['zero_hidden']}; metrics={}
    for name,frames in variants.items():
        mse=((frames.astype(np.float32)/255-target)**2).mean((1,2,3))
        metrics[name]=dict(future24_rgb_mse=float(mse[1:].mean()),future24_psnr=float(-10*np.log10(max(mse[1:].mean(),1e-12))),
                           future12_rgb_mse=float(mse[1:13].mean()))
    writer=imageio.get_writer(root/'cat_comparison.mp4',fps=8,codec='libx264',quality=8,macro_block_size=1)
    try:
        for t in range(25):
            canvas=Image.new('RGB',(832,590),'#141820'); d=ImageDraw.Draw(canvas)
            d.text((8,7),'RESEARCH DIAGNOSTIC | 95,704 Gaussians | half-speed inspection',fill='white')
            d.text((8,24),'Not accepted as coherent gait. Wheat: Neil Oakes / CC BY-SA 2.0',fill='white')
            panels=[('Wan reference: evaluation only',source[t+2]),('Previous learned vectors',old['learned'][t]),
                    ('Coherence candidate: NOT visual success',new['learned'][t]),('Frozen control',new['frozen'][t])]
            for i,(label,arr) in enumerate(panels):
                x,y=(i%2)*416,48+(i//2)*268; d.text((x+5,y),label,fill='white'); canvas.paste(Image.fromarray(arr).resize((416,240)),(x,y+20))
            writer.append_data(np.asarray(canvas))
            if t in [0,6,12,24]: canvas.save(root/f'comparison_{t:02d}.jpg')
    finally: writer.close()
    (root/'cat_diagnostic.json').write_text(json.dumps(dict(metrics=metrics,selection_used_cat=False,
        limits='Development cat OOD, approximate reference masks; not independent final test or perceptual quality certification.'),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,default=Path('artifacts/real_video/coherent_motion/v2'))
    parser.add_argument('--mode',choices=['audit','predict','cat'],required=True); args=parser.parse_args(); root=args.root
    torch.set_num_threads(4); cv2.setNumThreads(4)
    with keep_windows_awake():
        if args.mode=='audit':
            if (root/'independent_audit.json').exists(): raise FileExistsError('Preserve audit')
            results=json.loads((root/'results.json').read_text()); history=json.loads((root/'history.json').read_text())
            chosen=results['selected'] or min([r for r in history if r['strength']>0],key=lambda r:r['validation']['means']['edge_mse_pixels64'])
            candidate_path=root/chosen['candidate']/'checkpoints'/f"step_{chosen['step']:08d}.pt"
            candidate=load_verified(candidate_path); original=load_verified(BASE)
            control=load_verified(root/'lambda_0/checkpoints'/f"step_{chosen['step']:08d}.pt")
            data=load_verified(DATA); val=[i for i,m in enumerate(data['metadata']) if m['split']=='validation']
            audited={name:moving_audit(checkpoint,data,val) for name,checkpoint in [('candidate',candidate),('original',original),('control',control)]}
            comparisons={name:compare(audited['candidate'],audited[name]) for name in ['original','control']}
            seed=load_verified(SEEDS/'seed_1px.pt')
            costs={name:benchmark(checkpoint,seed) for name,checkpoint in [('original',original),('candidate',candidate)]}
            diagnostic=root/'diagnostic'
            sha=save_inference_checkpoint(dict(model=candidate['model'],config=candidate['config'],selected_step=chosen['step'],
                                              diagnostic_only=True,primary_gate_passed=results['selected'] is not None),diagnostic/'model.pt')
            report=dict(candidate=chosen['candidate'],step=chosen['step'],primary_gate_passed=results['selected'] is not None,
                        diagnostic_model_sha256=sha,comparisons=comparisons,audited=audited,costs=costs,
                        visually_accepted=False,wan_weights_changed=False)
            (root/'independent_audit.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({k:v for k,v in report.items() if k!='audited'}),flush=True)
        elif args.mode=='predict': predict(1,root=root/'diagnostic')
        else: cat_evaluation(root,root/'diagnostic')


if __name__=='__main__': main()
