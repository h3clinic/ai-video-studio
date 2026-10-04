"""Compare real frames, Gaussian analysis, and learned codec reconstruction on validation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from .latent import GaussianCodec
from .representation import render_fields
from .train_latent import load_development
from .evaluate import to_frames


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=GaussianCodec(**ckpt['codec_config']).cuda().eval()
    model.load_state_dict(ckpt['codec'])
    data,mean,std,labels=load_development(); val=data['validation']
    mse=0.; quantization=0.; reconstructions=[]; originals=[]
    for i in range(0,len(val['fields']),4):
        x=val['fields'][i:i+4]; z,_=model.encode(x)
        decoded=render_fields(model.decode(z)*std+mean)
        quantized=render_fields(model.decode(z.half().float())*std+mean)
        mse+=(quantized-val['rgb'][i:i+4]).square().mean().item()*len(x)
        quantization+=(quantized-decoded).abs().mean().item()*len(x)
        reconstructions.append(quantized.cpu()); originals.append(render_fields(x*std+mean).cpu())
    decoded=torch.cat(reconstructions); original=torch.cat(originals)
    sheet=Image.new('RGB',(192*3,212*len(labels)+24),(18,22,29)); draw=ImageDraw.Draw(sheet)
    for j,title in enumerate(('REAL VALIDATION','GAUSSIAN ANALYSIS','CODEC RECONSTRUCTION')): draw.text((192*j+5,6),title,fill='white')
    for label_id,label in enumerate(labels):
        first=torch.where(val['labels']==label_id)[0][0].item()
        for j,video in enumerate((val['rgb'][first].cpu(),original[first],decoded[first])):
            frame=Image.fromarray(to_frames(video)[3]).resize((192,192),Image.Resampling.NEAREST)
            sheet.paste(frame,(192*j,label_id*212+44))
        draw.text((5,label_id*212+27),label+' | NOT GENERATION',fill='white')
    sheet.save(args.out/'reconstruction_comparison.png')
    report=dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                validation_clips=len(decoded),validation_rgb_psnr_db=-10*torch.log10(torch.tensor(mse/len(decoded))).item(),
                float16_quantization_mean_absolute_pixel_change=quantization/len(decoded),
                code_values_per_clip=16*4*8*8,code_fp16_bytes_per_clip=16*4*8*8*2,
                gaussian_field_values_per_clip=9*8*32*32,raw_uint8_rgb_bytes_per_clip=8*64*64*3,
                total_codec_parameters=sum(p.numel() for p in model.parameters()),
                decoder_parameters=sum(p.numel() for name,p in model.named_parameters() if not name.startswith('encoder.')),
                limitation='Full source clip is encoded. Reconstruction/compression diagnostic only, not generation, not H264 superiority.')
    (args.out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
