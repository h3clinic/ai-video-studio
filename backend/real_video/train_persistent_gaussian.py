"""Real-video causal state forecasting, with frozen/velocity/reset baselines."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from .checkpoint_io import load_verified, save_inference_checkpoint, save_training_checkpoint, digest, keep_windows_awake
from .representation import encode_clip
from .persistent_gaussian import PersistentGaussianDynamics, tracked_fields_to_state, render_state, state_bytes

WORK=Path('../../work/real_video')
OUT=Path('artifacts/real_video/persistent_gaussian/v1')


def prepare():
    path=WORK/'persistent_gaussian_tracks_v1.pt'
    if path.exists(): return load_verified(path)
    source=WORK/'wan_bridge_all_256.pt'
    data=load_verified(source)
    assert all(e['split'] in ['train','validation'] for e in data['metadata'])
    fields=[]; videos=[]
    for i,video in enumerate(data['videos'].numpy()):
        small=np.stack([cv2.resize(frame,(64,64),interpolation=cv2.INTER_AREA) for frame in video])
        raw=torch.from_numpy(encode_clip(small)).permute(1,0,2,3)
        field=tracked_fields_to_state(raw).permute(1,0,2,3)
        fields.append(field); videos.append(torch.from_numpy(small))
        if (i+1)%50==0: print(json.dumps(dict(stage='causal_tracks',clips=i+1)),flush=True)
    result=dict(fields=torch.stack(fields),videos=torch.stack(videos),metadata=data['metadata'],
                source_sha256=digest(source),policy='Per-frame causal optical flow; no Wan latents used')
    save_inference_checkpoint(result,path)
    return result


def field_loss(pred,target):
    position=(pred[:,:2]-target[:,:2]).square().mean()
    scale=(pred[:,2:4]-target[:,2:4]).square().mean()
    direction=(1-(pred[:,4:6]*target[:,4:6]).sum(1).square().clamp(0,1)).mean()
    colour=(pred[:,6:]-target[:,6:]).square().mean()
    return 20*position+.2*scale+.1*direction+.25*colour


@torch.no_grad()
def evaluate(model,data,indices,baselines=False):
    model.eval(); records=[]
    modes=['recurrent','frozen','velocity','reset_memory'] if baselines else ['recurrent']
    for i in indices:
        sequence=data['fields'][i:i+1].to('cuda')
        target_rgb=data['videos'][i].permute(0,3,1,2).to('cuda').float()/255
        scores={}
        for mode in modes:
            state=model.initialize(sequence[:,:,0],sequence[:,:,1])
            losses=[]; pixels=[]
            for t in range(2,9):
                if mode in ['recurrent','reset_memory']:
                    state=model.step(state,reset_memory=mode=='reset_memory')
                elif mode=='velocity':
                    state=dict(state,fields=torch.cat((state['fields'][:,:2]+state['velocity'],state['fields'][:,2:]),1))
                losses.append(field_loss(state['fields'],sequence[:,:,t]).item())
                if baselines:
                    rgb=render_state(state['fields'])
                    pixels.append((rgb[0]-target_rgb[t]).square().mean().item())
            scores[mode]=dict(field_loss=float(np.mean(losses)))
            if baselines: scores[mode]['rgb_mse']=float(np.mean(pixels))
        records.append(dict(file=data['metadata'][i]['file'],scores=scores))
    means={mode:dict(field_loss=float(np.mean([r['scores'][mode]['field_loss'] for r in records]))) for mode in modes}
    if baselines:
        for mode in modes:
            mse=float(np.mean([r['scores'][mode]['rgb_mse'] for r in records]))
            means[mode].update(rgb_mse=mse,rgb_psnr=float(-10*np.log10(max(mse,1e-12))))
    return dict(means=means,cases=records)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): raise FileExistsError('Existing experiment retained; do not overwrite')
    protocol=dict(seed=430101,steps=2000,hidden=16,batch=4,lr=.0003,
                  seed_frames=2,predicted_frames=7,teacher_forcing=False,
                  selection='validation recurrent field forecast loss every 250 updates',
                  targets='causal optical-flow Gaussian tracks from recorded clips',
                  limitations='Planar indexed slots; not verified objects, occlusion recall or physical dynamics')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4)
    with keep_windows_awake():
        data=prepare()
        train=[i for i,e in enumerate(data['metadata']) if e['split']=='train']
        validation=[i for i,e in enumerate(data['metadata']) if e['split']=='validation']
        assert len(train)==279 and len(validation)==57
        assert not ({data['metadata'][i]['group'] for i in train}&{data['metadata'][i]['group'] for i in validation})
        torch.manual_seed(430101)
        model=PersistentGaussianDynamics().to('cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
        initial=evaluate(model,data,validation)
        history=[dict(step=0,validation=initial)]; best=initial['means']['recurrent']['field_loss']
        start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        def save(step,improved):
            save_training_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=protocol,
                                          step=step,validation=history[-1]['validation'],rng=torch.get_rng_state(),
                                          cuda_rng=torch.cuda.get_rng_state()),OUT,step,improved)
        save(0,True)
        for step in range(1,2001):
            model.train()
            indices=[train[i] for i in torch.randint(len(train),(4,)).tolist()]
            sequence=data['fields'][indices].to('cuda')
            state=model.initialize(sequence[:,:,0],sequence[:,:,1]); loss=0
            for t in range(2,9):
                state=model.step(state)
                loss=loss+field_loss(state['fields'],sequence[:,:,t])/7
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            if step%50==0:
                progress=dict(stage='train_recurrent',step=step,total=2000,loss=loss.item(),elapsed=time.perf_counter()-start)
                (OUT/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                print(json.dumps(progress),flush=True)
            if step%250==0:
                score=evaluate(model,data,validation)
                value=score['means']['recurrent']['field_loss']; improved=value<best; best=min(best,value)
                history.append(dict(step=step,validation=score)); save(step,improved)
                (OUT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        training_seconds=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated()
        selected=load_verified(OUT/'best.pt'); model.load_state_dict(selected['model'])
        final=evaluate(model,data,validation,baselines=True)
        checkpoint=dict(model=selected['model'],config=protocol,selected_step=selected['step'],
                        data_sha256=digest(WORK/'persistent_gaussian_tracks_v1.pt'))
        sha=save_inference_checkpoint(checkpoint,OUT/'dynamics.pt')
        sample=model.initialize(data['fields'][:1,:,0].to('cuda'),data['fields'][:1,:,1].to('cuda'))
        report=dict(selected_step=selected['step'],validation=final,initial=initial,
                    training_seconds=training_seconds,training_peak_cuda_bytes=peak,
                    current_state_bytes_32x32=state_bytes(sample),parameters=sum(p.numel() for p in model.parameters()),
                    checkpoint_sha256=sha,train_clips=279,validation_clips=57,test_clips=0)
        (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(stage='complete',selected_step=selected['step'],validation=final['means'])),flush=True)


if __name__=='__main__': main()
