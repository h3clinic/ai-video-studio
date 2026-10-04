"""Two-stage Gaussian latent training on development data; test split is excluded."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from .latent import GaussianCodec,kl_loss,rendered_loss
from .model import GaussianVideoDenoiser,schedule
from .objectives import field_loss
from .representation import render_fields,mirror_fields
from .train import validate
from .evaluate import save_video
from .checkpoint_io import load_verified,save_training_checkpoint,keep_windows_awake,digest
from .prepare_windows import validate_windows

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent.parent/'work/real_video'


def load_development():
    data=torch.load(WORK/'dataset.pt',map_location='cpu',weights_only=True)
    ids={split:[i for i,m in enumerate(data['metadata']) if m['split']==split] for split in ('train','validation')}
    result={}
    mean,std=data['mean'].cuda(),data['std'].cuda()
    for split,index in ids.items():
        result[split]=dict(fields=(data['fields'][index].float().cuda()-mean)/std,
                           rgb=data['videos'][index].float().permute(0,1,4,2,3).cuda()/255,
                           metadata=[data['metadata'][i] for i in index],
                           labels=torch.tensor([data['labels'].index(data['metadata'][i]['label']) for i in index],device='cuda'))
    return result,mean,std,data['labels']


@torch.no_grad()
def codec_validate(model,data,mean,std):
    model.eval(); total=0.; pixels=0.; count=len(data['fields'])
    for i in range(0,count,4):
        x=data['fields'][i:i+4]; rgb=data['rgb'][i:i+4]
        with torch.autocast('cuda',dtype=torch.bfloat16): prediction,_,_=model(x,stochastic=False)
        prediction=prediction.float()
        rendered=render_fields(prediction*std+mean)
        total+=(field_loss(prediction,x,3)+0.5*rendered_loss(rendered,rgb)).item()*len(x)
        pixels+=(rendered-rgb).square().mean().item()*len(x)
    return total/count,-10*torch.log10(torch.tensor(pixels/count)).item()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['codec','prior'])
    parser.add_argument('--run',required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--codec',type=Path)
    parser.add_argument('--dropout',type=float,default=0.)
    parser.add_argument('--mirror',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--lr',type=float,default=0.0002)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--pause-ms',type=float,default=0.)
    parser.add_argument('--extra-windows',type=Path)
    args=parser.parse_args()
    if not args.run.replace('_','').isalnum() or not 1<=args.steps<=20000 or args.batch<1: parser.error('Invalid run budget')
    if args.stage=='prior' and not args.codec: parser.error('Prior training requires frozen --codec')
    if not 0<=args.dropout<0.5: parser.error('Dropout must be in [0,0.5)')
    if args.stage=='codec' and (args.dropout or args.mirror): parser.error('These options currently apply to the prior only')
    if args.resume and args.stage!='prior': parser.error('Resume currently supports the prior stage')
    if args.extra_windows and args.stage!='prior': parser.error('Extra windows currently apply to the prior only')
    if args.lr<=0 or not 1<=args.threads<=8 or not 0<=args.pause_ms<=1000: parser.error('Invalid resource/training options')
    torch.set_num_threads(args.threads); torch.manual_seed(31729 if args.stage=='codec' else 41729)
    resumed=load_verified(args.resume) if args.resume else None
    options=dict(stage=args.stage,batch=args.batch,lr=args.lr,dropout=args.dropout,mirror=args.mirror,
                 extra_windows_sha256=digest(args.extra_windows) if args.extra_windows else None)
    previous_options=dict(resumed.get('training_options',{})) if resumed else {}
    previous_options.setdefault('extra_windows_sha256',None)
    exact_resume=bool(resumed and previous_options==options and 'optimizer' in resumed and 'training_model' in resumed)
    previous_updates=resumed.get('completed_updates',resumed['step']) if resumed else 0
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    data,mean,std,labels=load_development()
    out=WORK/'runs'/args.run; out.mkdir(parents=True,exist_ok=False)
    (out/'source').mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p,out/'source'/p.name)
    manifest=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  started=datetime.now(timezone.utc).isoformat(),train_clips=len(data['train']['fields']),
                  validation_clips=len(data['validation']['fields']),test_used=False,
                  dataset_protocol_sha256=hashlib.sha256((ROOT/'artifacts/real_video/data_protocol.json').read_bytes()).hexdigest(),
                  parent_checkpoint_sha256=digest(args.resume) if args.resume else None,
                  resume_mode=('optimizer_and_rng_restored' if exact_resume else 'warm_restart_from_verified_EMA_weights') if resumed else 'from_scratch',
                  previous_updates=previous_updates,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    codec=GaussianCodec().cuda()
    if args.stage=='codec':
        model=codec; model.train()
        manifest['parameters']=sum(p.numel() for p in model.parameters())
        manifest['selection']='Validation weighted field MSE plus 0.5 multiscale rendered/motion L1; not generative quality'
    else:
        codec_checkpoint=load_verified(args.codec)
        torch.testing.assert_close(mean,codec_checkpoint['mean'].cuda())
        torch.testing.assert_close(std,codec_checkpoint['std'].cuda())
        codec=GaussianCodec(**codec_checkpoint['codec_config']).cuda().eval().requires_grad_(False)
        codec.load_state_dict(codec_checkpoint['codec'])
        with torch.no_grad():
            latents={}
            for split in ('train','validation'):
                chunks=[]
                for i in range(0,len(data[split]['fields']),8):
                    with torch.autocast('cuda',dtype=torch.bfloat16): z,_=codec.encode(data[split]['fields'][i:i+8])
                    chunks.append(z.float())
                latents[split]=torch.cat(chunks)
            latent_mean=latents['train'].mean((0,2,3,4),keepdim=True)
            latent_std=latents['train'].std((0,2,3,4),keepdim=True).clamp_min(0.05)
            prior_labels=data['train']['labels']
            training_parts=[latents['train']]; label_parts=[prior_labels]
            if args.extra_windows:
                extra=load_verified(args.extra_windows)
                validate_windows(extra,data['train']['metadata'],labels,manifest['dataset_protocol_sha256'])
                extra_labels=torch.tensor([labels.index(m['label']) for m in extra['metadata']],device='cuda')
                for mirrored in ([False,True] if args.mirror else [False]):
                    chunks=[]
                    for i in range(0,len(extra['fields']),8):
                        raw=extra['fields'][i:i+8].float().cuda()
                        if mirrored: raw=mirror_fields(raw)
                        with torch.autocast('cuda',dtype=torch.bfloat16): z,_=codec.encode((raw-mean)/std)
                        chunks.append(z.float())
                    training_parts.append(torch.cat(chunks)); label_parts.append(extra_labels)
                manifest['extra_temporal_windows']=len(extra['fields'])
                manifest['extra_windows_sha256']=options['extra_windows_sha256']
            if args.mirror:
                chunks=[]
                for i in range(0,len(data['train']['fields']),8):
                    raw=data['train']['fields'][i:i+8]*std+mean
                    transformed=(mirror_fields(raw)-mean)/std
                    with torch.autocast('cuda',dtype=torch.bfloat16): z,_=codec.encode(transformed)
                    chunks.append(z.float())
                training_parts.append(torch.cat(chunks)); label_parts.append(prior_labels)
            latents['train']=torch.cat(training_parts); prior_labels=torch.cat(label_parts)
            latents={k:(v-latent_mean)/latent_std for k,v in latents.items()}
        model=GaussianVideoDenoiser(classes=len(labels),width=48,channels=16,dropout=args.dropout).cuda()
        ema=deepcopy(model).eval().requires_grad_(False)
        if resumed:
            if resumed['labels']!=labels: raise ValueError('Resume labels differ')
            torch.testing.assert_close(latent_mean,resumed['latent_mean'].cuda())
            torch.testing.assert_close(latent_std,resumed['latent_std'].cuda())
            for name,value in codec_checkpoint['codec'].items():
                torch.testing.assert_close(value,resumed['codec'][name],rtol=0,atol=0)
            model.load_state_dict(resumed['training_model'] if exact_resume else resumed['model'])
            ema.load_state_dict(resumed['model'])
        alpha=schedule('cuda')
        manifest.update(parameters=sum(p.numel() for p in model.parameters()),codec_parameters=sum(p.numel() for p in codec.parameters()),
                        codec_sha256=hashlib.sha256(args.codec.read_bytes()).hexdigest(),latent_shape=list(latents['train'].shape[1:]),
                        training_latent_count=len(latents['train']),
                        latent_normalization_source='original unaugmented training posterior means only',selection='Fixed-noise validation latent v-MSE')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    if exact_resume:
        optimizer.load_state_dict(resumed['optimizer'])
        torch.set_rng_state(resumed['rng_cpu'])
        torch.cuda.set_rng_state_all(resumed['rng_cuda'])
    best=float('inf'); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    if resumed:
        best,_=validate(ema,latents['validation'],data['validation']['labels'],schedule('cuda'))
        initial=dict(resumed)
        initial.update(model=ema.state_dict(),config=ema.config,step=0,completed_updates=previous_updates,validation=best,
                       training_model=model.state_dict(),optimizer=optimizer.state_dict(),training_options=options,
                       rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
        save_training_checkpoint(initial,out,0,True)
        print(json.dumps(dict(resumed_validation=best,resume_mode=manifest['resume_mode'])),flush=True)
    with (out/'training.jsonl').open('w',encoding='utf-8') as log:
        for step in range(1,args.steps+1):
            model.train()
            count=len(data['train']['fields']) if args.stage=='codec' else len(latents['train'])
            idx=torch.randint(count,(args.batch,),device='cuda')
            if args.stage=='codec':
                x=data['train']['fields'][idx]
                with torch.autocast('cuda',dtype=torch.bfloat16): predicted,mu,logvar=model(x)
                predicted=predicted.float()
                loss=field_loss(predicted,x,3)+0.0001*min(step/1000,1)*kl_loss(mu.float(),logvar.float())
                if step%4==0:
                    rendered=render_fields(predicted[:2]*std+mean)
                    loss=loss+0.5*rendered_loss(rendered,data['train']['rgb'][idx[:2]])
            else:
                x=latents['train'][idx]; condition=prior_labels[idx].clone()
                condition[torch.rand(len(x),device='cuda')<0.1]=model.classes
                t=torch.randint(0,1000,(len(x),),device='cuda'); noise=torch.randn_like(x)
                a=alpha[t,None,None,None,None].sqrt(); s=(1-alpha[t,None,None,None,None]).sqrt()
                with torch.autocast('cuda',dtype=torch.bfloat16): predicted=model(a*x+s*noise,t,condition).float()
                loss=(predicted-(a*noise-s*x)).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if args.stage=='prior':
                with torch.no_grad():
                    for p,q in zip(ema.parameters(),model.parameters()): p.lerp_(q,0.005)
            if step%250==0 or step==1 or step==args.steps:
                if args.stage=='codec':
                    val,psnr=codec_validate(model,data['validation'],mean,std)
                    checkpoint=dict(codec=model.state_dict(),codec_config=model.config,mean=mean.cpu(),std=std.cpu(),labels=labels,
                                    step=step,validation=val,validation_reconstruction_psnr_db=psnr,
                                    scope='Reconstruction codec only, not a video generator')
                    extra=dict(reconstruction_psnr_db=psnr)
                else:
                    val,baseline=validate(ema,latents['validation'],data['validation']['labels'],alpha)
                    checkpoint=dict(codec=codec.state_dict(),codec_config=codec.config,model=ema.state_dict(),config=ema.config,
                                    mean=mean.cpu(),std=std.cpu(),labels=labels,latent_mean=latent_mean.cpu(),latent_std=latent_std.cpu(),
                                    latent_shape=list(latents['train'].shape[2:]),step=step,validation=val,
                                    scope='Noise -> learned Gaussian latent -> Gaussian fields -> splatting; no reference input')
                    extra=dict(zero_velocity_mse=baseline)
                checkpoint.update(training_model=model.state_dict(),optimizer=optimizer.state_dict(),training_options=options,
                                  completed_updates=previous_updates+step,parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'],
                                  rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
                improved=val<best
                save_training_checkpoint(checkpoint,out,step,improved)
                if improved: best=val
                record=dict(step=step,loss=loss.item(),validation=val,elapsed_s=round(time.perf_counter()-start,2),
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),**extra)
                print(json.dumps(record),flush=True); log.write(json.dumps(record)+'\n'); log.flush()
            if args.pause_ms: time.sleep(args.pause_ms/1000)
    if args.stage=='codec':
        selected=torch.load(out/'best.pt',map_location='cuda',weights_only=True)
        model.load_state_dict(selected['codec']); model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): reconstruction,_,_=model(data['validation']['fields'][:1],stochastic=False)
        with torch.no_grad(): video=render_fields(reconstruction.float()*std+mean)[0]
        save_video(video,out/'codec_reconstruction.mp4','CODEC RECONSTRUCTION - NOT GENERATION')
        save_video(data['validation']['rgb'][0],out/'real_reference.mp4','REAL VALIDATION REFERENCE')


if __name__=='__main__':
    with keep_windows_awake(): main()
