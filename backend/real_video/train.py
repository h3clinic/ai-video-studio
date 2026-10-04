"""Train only on real-video training groups. Keep test data out of model selection."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import shutil
from pathlib import Path
import time
import torch
from .model import GaussianVideoDenoiser,schedule
from .objectives import field_loss,appearance_loss

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent.parent/'work/real_video'


@torch.no_grad()
def validate(model,x,labels,alpha):
    gen=torch.Generator(device=x.device).manual_seed(90117)
    losses=[]; baselines=[]
    for begin in range(0,len(x),8):
        clean=x[begin:begin+8]
        y=labels[begin:begin+8]
        t=torch.randint(50,950,(len(clean),),device=x.device,generator=gen)
        noise=torch.randn(clean.shape,device=x.device,generator=gen)
        a=alpha[t,None,None,None,None].sqrt(); s=(1-alpha[t,None,None,None,None]).sqrt()
        noisy=a*clean+s*noise
        target=a*noise-s*clean
        with torch.autocast('cuda',dtype=torch.bfloat16): prediction=model(noisy,t,y).float()
        losses.append((prediction-target).square().mean().item()*len(clean))
        # If clean fields were standard Gaussian, the optimal v prediction is zero.
        baselines.append(target.square().mean().item()*len(clean))
    return sum(losses)/len(x),sum(baselines)/len(x)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--steps',type=int,default=2500)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--lr',type=float,default=0.0002)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--rgb-weight',type=float,default=1.)
    parser.add_argument('--render-weight',type=float,default=0.)
    parser.add_argument('--render-every',type=int,default=4)
    parser.add_argument('--refinement-blocks',type=int)
    args=parser.parse_args()
    if not args.run.replace('_','').isalnum() or not 1<=args.steps<=20000: parser.error('Invalid run or step budget')
    if args.rgb_weight<=0 or args.render_weight<0 or args.render_every<1: parser.error('Invalid objective weights')
    if args.refinement_blocks is not None and not 0<=args.refinement_blocks<=4: parser.error('Use 0..4 refinement blocks')
    torch.set_num_threads(4); torch.manual_seed(7129)
    if not torch.cuda.is_available(): raise RuntimeError('This training run requires CUDA')
    device='cuda'
    data=torch.load(WORK/'dataset.pt',map_location='cpu',weights_only=True)
    label_ids=torch.tensor([data['labels'].index(m['label']) for m in data['metadata']],device=device)
    train_ids=torch.tensor([i for i,m in enumerate(data['metadata']) if m['split']=='train'])
    val_ids=torch.tensor([i for i,m in enumerate(data['metadata']) if m['split']=='validation'])
    x=((data['fields'][train_ids].float()-data['mean'])/data['std']).to(device)
    xv=((data['fields'][val_ids].float()-data['mean'])/data['std']).to(device)
    y=label_ids[train_ids]; yv=label_ids[val_ids]
    mean=data['mean'].to(device); std=data['std'].to(device)
    training_rgb=None
    if args.render_weight:
        training_rgb=data['videos'][train_ids].float().permute(0,1,4,2,3).to(device)/255
    model=GaussianVideoDenoiser(len(data['labels']),args.width,refinement_blocks=args.refinement_blocks or 0).to(device)
    if args.resume:
        ckpt=torch.load(args.resume,map_location=device,weights_only=True)
        config=dict(ckpt['config'])
        if args.refinement_blocks is not None:
            if args.refinement_blocks<config.get('refinement_blocks',0): parser.error('Cannot discard trained refinement blocks')
            config['refinement_blocks']=args.refinement_blocks
        model=GaussianVideoDenoiser(**config).to(device)
        compatibility=model.load_state_dict(ckpt['model'],strict=False)
        if compatibility.unexpected_keys or any(not k.startswith(('refinements.','coordinates.')) for k in compatibility.missing_keys):
            raise ValueError(f'Unexpected checkpoint mismatch: {compatibility}')
    ema=deepcopy(model).eval().requires_grad_(False)
    alpha=schedule(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    out=WORK/'runs'/args.run; out.mkdir(parents=True,exist_ok=False)
    source=out/'source'; source.mkdir()
    for module in Path(__file__).parent.glob('*.py'): shutil.copyfile(module,source/module.name)
    manifest=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  started=datetime.now(timezone.utc).isoformat(),train_count=len(x),validation_count=len(xv),test_used=False,
                  parameters=sum(p.numel() for p in model.parameters()),labels=data['labels'],
                  parent_checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
                  objective_selection='Original unweighted validation v-MSE, fixed seed 90117; not a perceptual generation score',
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    log=(out/'training.jsonl').open('w',encoding='utf-8')
    start=time.perf_counter(); best=float('inf'); torch.cuda.reset_peak_memory_stats()
    for step in range(1,args.steps+1):
        model.train()
        idx=torch.randint(len(x),(args.batch,),device=device)
        clean=x[idx]; labels=y[idx].clone()
        labels[torch.rand(len(labels),device=device)<0.1]=model.classes
        t=torch.randint(0,1000,(len(clean),),device=device)
        noise=torch.randn_like(clean)
        a=alpha[t,None,None,None,None].sqrt(); s=(1-alpha[t,None,None,None,None]).sqrt()
        noisy=a*clean+s*noise; target=a*noise-s*clean
        with torch.autocast('cuda',dtype=torch.bfloat16): prediction=model(noisy,t,labels).float()
        parameter_loss=field_loss(prediction,target,args.rgb_weight)
        image_loss=prediction.new_zeros(())
        if args.render_weight and step%args.render_every==0:
            image_loss=appearance_loss(prediction,noisy,alpha[t,None,None,None,None],mean,std,training_rgb[idx])
        loss=parameter_loss+args.render_weight*image_loss
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        with torch.no_grad():
            for p,q in zip(ema.parameters(),model.parameters()): p.lerp_(q,0.005)
        if step%250==0 or step==1 or step==args.steps:
            val,baseline=validate(ema,xv,yv,alpha)
            record=dict(step=step,loss=loss.item(),validation=val,zero_velocity_baseline=baseline,
                        parameter_loss=parameter_loss.item(),image_loss=image_loss.item(),
                        elapsed_s=round(time.perf_counter()-start,2),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            checkpoint=dict(model=ema.state_dict(),config=ema.config,mean=data['mean'],std=data['std'],labels=data['labels'],
                            training_objective=dict(rgb_weight=args.rgb_weight,render_weight=args.render_weight,render_every=args.render_every),
                            step=step,validation=val,scope='Real UCF101 subset; planar Gaussian-field diffusion, no pretrained video model',
                            dataset_sha256=hashlib.sha256((ROOT/'artifacts/real_video/data_protocol.json').read_bytes()).hexdigest())
            torch.save(checkpoint,out/'last.pt')
            if val<best: best=val; torch.save(checkpoint,out/'best.pt')
            print(json.dumps(record),flush=True); log.write(json.dumps(record)+'\n'); log.flush()
    log.close()


if __name__=='__main__': main()
