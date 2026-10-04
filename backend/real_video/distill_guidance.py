"""Learn one conditional Gaussian-latent prediction instead of two CFG branches.

This only distills the existing (limited-quality) generator. Training examples
are generated teacher trajectories, never source-conditioned reconstructions.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import torch
from .checkpoint_io import load_verified,save_training_checkpoint,save_inference_checkpoint,digest,keep_windows_awake
from .model import GaussianVideoDenoiser,sample,schedule,guided_velocity

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent.parent/'work/real_video'


def forward_noised(z,t,noise,alpha):
    a=alpha[t].sqrt().reshape(-1,1,1,1,1)
    s=(1-alpha[t]).sqrt().reshape(-1,1,1,1,1)
    return a*z+s*noise


@torch.no_grad()
def real_training_codes(checkpoint):
    from .latent import GaussianCodec
    from .representation import mirror_fields
    data=load_verified(WORK/'dataset.pt')
    ids=[i for i,m in enumerate(data['metadata']) if m['split']=='train']
    if len(ids)!=279 or data['labels']!=checkpoint['labels']: raise ValueError('Unexpected training protocol')
    codec=GaussianCodec(**checkpoint['codec_config']).cuda().eval().requires_grad_(False)
    codec.load_state_dict(checkpoint['codec'])
    fields=data['fields'][ids]; values=[]
    mean=checkpoint['mean'].cuda(); std=checkpoint['std'].cuda()
    labels=torch.tensor([data['labels'].index(data['metadata'][i]['label']) for i in ids],device='cuda')
    for mirror in (False,True):
        for start in range(0,len(fields),8):
            raw=fields[start:start+8].float().cuda()
            if mirror: raw=mirror_fields(raw)
            with torch.autocast('cuda',dtype=torch.bfloat16): z,_=codec.encode((raw-mean)/std)
            values.append((z.float()-checkpoint['latent_mean'].cuda())/checkpoint['latent_std'].cuda())
    return torch.cat(values),labels.repeat(2),[data['metadata'][i] for i in ids]


@torch.no_grad()
def collect(model,shape,first_seed,batches):
    parts={k:[] for k in ('x','t','target','labels')}
    labels=torch.arange(model.classes,device='cuda')
    def observe(x,t,v):
        parts['x'].append(x.half().cpu()); parts['t'].append(t.cpu())
        parts['target'].append(v.half().cpu()); parts['labels'].append(labels.cpu())
    for index in range(batches):
        sample(model,labels,seed=first_seed+index,steps=50,guidance=1.5,shape=shape,observer=observe)
        print(json.dumps(dict(cache_seed=first_seed+index,complete=index+1,total=batches)),flush=True)
    return {k:torch.cat(v) for k,v in parts.items()}


@torch.no_grad()
def validation(model,data):
    model.eval(); error=0.; count=len(data['x'])
    for start in range(0,count,32):
        x=data['x'][start:start+32].float().cuda()
        t=data['t'][start:start+32].cuda(); labels=data['labels'][start:start+32].cuda()
        target=data['target'][start:start+32].float().cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16): predicted=model(x,t,labels)
        error+=(predicted.float()-target).square().mean().item()*len(x)
    return error/count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache',type=Path,help='Reuse verified teacher trajectories')
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--ema',type=float,default=0.,help='EMA decay; zero selects raw student')
    parser.add_argument('--fresh-states',action='store_true',help='Mix real-training forward noise and jittered rollout states')
    parser.add_argument('--initialize',type=Path)
    parser.add_argument('--steps',type=int,default=4000)
    args=parser.parse_args()
    if not args.run.replace('_','').isalnum(): parser.error('Invalid run name')
    if not 0<args.lr<=0.001 or not 0<=args.ema<1: parser.error('Invalid optimization settings')
    if not 500<=args.steps<=10000 or args.steps%500: parser.error('Use a multiple of 500 updates, at most 10000')
    if args.fresh_states and not args.cache: parser.error('Fresh-state protocol requires its existing training cache')
    if args.out.exists(): raise FileExistsError(args.out)
    torch.set_num_threads(2); torch.manual_seed(61729)
    teacher=load_verified(args.teacher)
    if teacher.get('distilled_guidance') is not None: raise ValueError('Expected original CFG teacher')
    expected='a7f5251544fc5120c5ce32f97bc3e4ac6710c7febd1848efa552b0f5f9886cff'
    if digest(args.teacher)!=expected: raise ValueError('Protocol teacher mismatch')
    out=WORK/'runs'/args.run; out.mkdir(parents=True,exist_ok=False)
    (out/'source').mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copyfile(path,out/'source'/path.name)
    shutil.copyfile(ROOT/'research/COMPUTE_LEARNING_PROTOCOL.md',out/'protocol.md')
    manifest=dict(teacher_sha256=expected,train_seed_range=[710000,710019],validation_seed_range=[810000,810003],
                  train_trajectories=200,validation_trajectories=40,steps_per_trajectory=50,
                  final_test_used=False,external_clips_used=False,guidance=1.5,updates=args.steps,batch=16,
                  lr=args.lr,ema=args.ema,seed=61729,selection='Validation guided-velocity MSE',
                  targets='Teacher generated states; not additional real video',
                  fresh_states=args.fresh_states,initialize_sha256=digest(args.initialize) if args.initialize else None,
                  source_sha256={p.name:digest(p) for p in (out/'source').glob('*.py')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    model=GaussianVideoDenoiser(**teacher['config']).cuda().eval()
    model.load_state_dict(teacher['model'])
    torch.cuda.synchronize(); started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    if args.cache:
        cached=load_verified(args.cache)
        if cached['manifest']['teacher_sha256']!=expected: raise ValueError('Wrong cache teacher')
        if cached['manifest']['train_seed_range']!=[710000,710019] or cached['manifest']['validation_seed_range']!=[810000,810003]:
            raise ValueError('Wrong cache split')
        train=cached['train']; valid=cached['validation']
        manifest['cache_reused_sha256']=digest(args.cache)
    else:
        train=collect(model,tuple(teacher['latent_shape']),710000,20)
        valid=collect(model,tuple(teacher['latent_shape']),810000,4)
    cache_seconds=time.perf_counter()-started
    (out/'cache_timing.json').write_text(json.dumps(dict(seconds=cache_seconds,reused=bool(args.cache)),indent=2),encoding='utf-8')
    if not args.cache: save_inference_checkpoint(dict(train=train,validation=valid,manifest=manifest),out/'trajectories.pt')
    # Identical weights initially, but the student must be deterministic during fitting.
    config=dict(teacher['config'],dropout=0.)
    student=GaussianVideoDenoiser(**config).cuda()
    if args.initialize:
        initial=load_verified(args.initialize)
        if initial['distillation']['teacher_sha256']!=expected: raise ValueError('Wrong initialization teacher')
        student.load_state_dict(initial['model'])
    else: student.load_state_dict(model.state_dict())
    if args.fresh_states:
        real_codes,real_labels,training_metadata=real_training_codes(teacher)
        model.eval().requires_grad_(False); alpha=schedule('cuda')
        manifest.update(real_training_sources=len(training_metadata),mirrored_real_codes=len(real_codes),
                        targets='Fresh original-teacher targets on mixed real forward-noise and jittered TRAIN rollout states')
        (out/'training_sources.json').write_text(json.dumps(training_metadata,indent=2),encoding='utf-8')
    else: del model
    preparation_seconds=time.perf_counter()-started
    optimizer=torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=1e-4)
    averaged=deepcopy(student).eval().requires_grad_(False) if args.ema else student
    baseline=validation(student,valid); best=baseline
    torch.cuda.synchronize(); train_start=time.perf_counter()
    history=[dict(step=0,validation_mse=baseline)]
    # Retain a valid no-improvement outcome instead of requiring a best.pt to appear.
    save_training_checkpoint(dict(config=config,model={k:v.detach().cpu() for k,v in student.state_dict().items()},
                                  step=0,validation_mse=baseline,distilled_guidance=1.5,teacher_sha256=expected),out,0,True)
    print(json.dumps(history[-1]),flush=True)
    for step in range(1,args.steps+1):
        student.train(); index=torch.randint(len(train['x']),(16,))
        x=train['x'][index].float().cuda(); target=train['target'][index].float().cuda()
        t=train['t'][index].cuda(); labels=train['labels'][index].cuda()
        if args.fresh_states:
            with torch.no_grad():
                real_index=torch.randint(len(real_codes),(8,),device='cuda')
                real_time=torch.randint(1000,(8,),device='cuda')
                real_x=forward_noised(real_codes[real_index],real_time,torch.randn_like(real_codes[real_index]),alpha)
                x=torch.cat((real_x,x[8:]+.15*torch.randn_like(x[8:])))
                t=torch.cat((real_time,t[8:])); labels=torch.cat((real_labels[real_index],labels[8:]))
                target=guided_velocity(model,x,t,labels,1.5)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): predicted=student(x,t,labels)
        loss=(predicted.float()-target).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite student loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
        optimizer.step()
        if args.ema:
            with torch.no_grad():
                for a,p in zip(averaged.parameters(),student.parameters()): a.lerp_(p,1-args.ema)
        if step%100==0: print(json.dumps(dict(step=step,train_mse=loss.item())),flush=True)
        if step%500==0:
            error=validation(averaged,valid); improved=error<best; best=min(best,error)
            record=dict(step=step,train_mse=loss.item(),validation_mse=error,elapsed=time.perf_counter()-train_start)
            history.append(record); print(json.dumps(record),flush=True)
            checkpoint=dict(config=config,model={k:v.detach().cpu() for k,v in averaged.state_dict().items()},
                            training_model={k:v.detach().cpu() for k,v in student.state_dict().items()},
                            optimizer=deepcopy(optimizer.state_dict()),step=step,validation_mse=error,
                            distilled_guidance=1.5,teacher_sha256=expected,rng_cpu=torch.get_rng_state(),
                            rng_cuda=torch.cuda.get_rng_state(),training_options=manifest)
            save_training_checkpoint(checkpoint,out,step,improved)
            (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        time.sleep(.01)
    torch.cuda.synchronize(); train_seconds=time.perf_counter()-train_start
    selected=load_verified(out/'best.pt')
    export=dict(teacher)
    export.update(model=selected['model'],config=config,step=selected['step'],distilled_guidance=1.5,
                  distillation=dict(teacher_sha256=expected,validation_mse=selected['validation_mse'],
                                    training=manifest['targets'],train_trajectories=200,
                                    fresh_states=args.fresh_states,initialize_sha256=manifest['initialize_sha256']))
    checksum=save_inference_checkpoint(export,args.out)
    report=dict(manifest=manifest,initial_validation_mse=baseline,best_validation_mse=best,
                selected_step=selected['step'],cache_seconds=cache_seconds,training_seconds=train_seconds,
                preparation_seconds=preparation_seconds,total_extra_compute_seconds=preparation_seconds+train_seconds,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=checksum,
                history=history,limits=['Teacher fidelity, not better real-video semantics or novel physics.',
                                       'Shared decoder unchanged; only guidance evaluation distilled.'])
    report['learned_weights_selected']=selected['step']>0
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    args.out.with_suffix('.training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    with keep_windows_awake(): main()
