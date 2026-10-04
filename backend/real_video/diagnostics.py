"""Validation-only weight diagnostics. Never a generative-video quality score."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .model import GaussianVideoDenoiser,schedule
from .representation import render_fields

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent.parent/'work/real_video'


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=GaussianVideoDenoiser(**ckpt['config']).cuda().eval()
    model.load_state_dict(ckpt['model'])
    data=torch.load(WORK/'dataset.pt',map_location='cpu',weights_only=True)
    ids=[i for i,m in enumerate(data['metadata']) if m['split']=='validation']
    mean,std=ckpt['mean'].cuda(),ckpt['std'].cuda()
    fields=data['fields'][ids].float().cuda()
    clean=(fields-mean)/std
    labels=torch.tensor([data['labels'].index(data['metadata'][i]['label']) for i in ids],device='cuda')
    reference=data['videos'][ids].float().permute(0,1,4,2,3).cuda()/255
    alpha=schedule('cuda')
    records=[]
    for noise_step in (150,500,850):
        rng=torch.Generator(device='cuda').manual_seed(21703)
        channels=torch.zeros(9,device='cuda'); wrong_loss=0.; pixels=0.; temporal=0.
        for start in range(0,len(ids),4):
            x0=clean[start:start+4]; y=labels[start:start+4]; n=len(x0)
            noise=torch.randn(x0.shape,device='cuda',generator=rng)
            a=alpha[noise_step].sqrt(); s=(1-alpha[noise_step]).sqrt()
            x=a*x0+s*noise; target=a*noise-s*x0
            tt=torch.full_like(y,noise_step)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(x,tt,y).float()
                wrong=model(x,tt,(y+1)%model.classes).float()
            channels+=(prediction-target).square().mean((0,2,3,4))*n
            wrong_loss+=(wrong-target).square().mean().item()*n
            reconstructed=render_fields((a*x-s*prediction).clamp(-6,6)*std+mean)
            rgb=reference[start:start+4]
            pixels+=(reconstructed-rgb).abs().mean().item()*n
            temporal+=((reconstructed[:,1:]-reconstructed[:,:-1])-(rgb[:,1:]-rgb[:,:-1])).abs().mean().item()*n
        channels/=len(ids)
        records.append(dict(noise_step=noise_step,per_channel_v_mse=channels.tolist(),
                            all_channel_v_mse=channels.mean().item(),rgb_parameter_v_mse=channels[6:].mean().item(),
                            wrong_label_v_mse=wrong_loss/len(ids),
                            label_mismatch_penalty=wrong_loss/len(ids)-channels.mean().item(),
                            target_conditioned_image_l1=pixels/len(ids),
                            target_conditioned_temporal_difference_l1=temporal/len(ids)))
    report=dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                split='validation',clips=len(ids),noise_seed=21703,metrics=records,
                limitation='Teacher-conditioned denoising diagnostics, not unconditional generated video quality. Test split not evaluated.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
