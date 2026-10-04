"""Frozen same-noise teacher-fidelity audit; not an external video-quality score."""
import argparse
import json
import math
from pathlib import Path
import statistics
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .checkpoint_io import load_verified,digest,keep_windows_awake
from .model import GaussianVideoDenoiser,sample
from .latent import inference_decoder
from .representation import render_fields_chunked,decode_state
from .evaluate import to_frames


def metrics(actual,reference):
    mse=(actual-reference).square().mean().item()
    difference=actual[:,1:]-actual[:,:-1]-(reference[:,1:]-reference[:,:-1])
    return dict(pixel_mse=mse,teacher_agreement_psnr=-10*math.log10(max(mse,1e-12)),
                temporal_difference_mae=difference.abs().mean().item(),
                max_abs_pixel_error=(actual-reference).abs().max().item())


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--student',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--first-seed',type=int,default=910000)
    parser.add_argument('--previous-student',type=Path,help='Optional unchanged prior student for paired progress comparison')
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    teacher=load_verified(args.teacher); student=load_verified(args.student)
    if student['distillation']['teacher_sha256']!=digest(args.teacher): raise ValueError('Wrong teacher')
    if student.get('distilled_guidance')!=1.5: raise ValueError('Wrong guidance')
    for key in teacher['codec']:
        torch.testing.assert_close(teacher['codec'][key],student['codec'][key],rtol=0,atol=0)
    for key in ('mean','std','latent_mean','latent_std'):
        torch.testing.assert_close(teacher[key],student[key],rtol=0,atol=0)
    tm=GaussianVideoDenoiser(**teacher['config']).cuda().eval(); tm.load_state_dict(teacher['model'])
    sm=GaussianVideoDenoiser(**student['config']).cuda().eval(); sm.load_state_dict(student['model'])
    decoder=inference_decoder(teacher['codec_config'],teacher['codec']).cuda()
    mean=teacher['mean'].cuda(); std=teacher['std'].cuda()
    zmean=teacher['latent_mean'].cuda(); zstd=teacher['latent_std'].cuda()
    cases=[]; axes_error=0.
    variants=[('teacher_50',tm,50,None),('untrained_single_50',tm,50,1.5),
              ('student_50',sm,50,1.5),('teacher_25',tm,25,None),('student_25',sm,25,1.5)]
    names=['Teacher: 50 steps','No learning: single pass','Learned: 50 steps','Teacher: 25 steps','Learned: 25 steps']
    if args.previous_student:
        previous=load_verified(args.previous_student)
        if previous['distillation']['teacher_sha256']!=digest(args.teacher): raise ValueError('Wrong previous student teacher')
        pm=GaussianVideoDenoiser(**previous['config']).cuda().eval(); pm.load_state_dict(previous['model'])
        variants.append(('previous_student_50',pm,50,1.5)); names.append('Previous student: 50')
    # No source clips or training caches are loaded anywhere in this audit.
    gallery=Image.new('RGB',(len(variants)*192,20*158),(18,22,29))
    for index in range(20):
        seed=args.first_seed+index; label_id=index%len(teacher['labels']); label=teacher['labels'][label_id]
        y=torch.tensor([label_id],device='cuda'); videos={}; codes={}; images={}
        for name,model,steps,distilled in variants:
            z=sample(model,y,seed=seed,steps=steps,guidance=1.5,shape=tuple(teacher['latent_shape']),distilled_guidance=distilled)
            codes[name]=z.cpu()
            code=(z*zstd+zmean).half().float()
            with torch.autocast('cuda',dtype=torch.bfloat16): field=decoder.decode(code).float()
            fields=field*std+mean
            video=render_fields_chunked(fields,frame_chunk=1).cpu(); videos[name]=video
            if name=='student_50':
                state=decode_state(fields[:,:,0]); r=state['R']
                axes_error=max(axes_error,(r.transpose(-1,-2)@r-torch.eye(3,device='cuda')).abs().max().item())
            images[name]=to_frames(video[0])
        result=dict(seed=seed,label=label,variants={})
        for name,_,_,_ in variants[1:]:
            result['variants'][name]=dict(metrics(videos[name],videos['teacher_50']),
                                         normalized_latent_mse=(codes[name]-codes['teacher_50']).square().mean().item())
        cases.append(result)
        frames=[]
        for t in range(8):
            canvas=Image.new('RGB',(len(variants)*192,256),(18,22,29)); draw=ImageDraw.Draw(canvas)
            for column,((name,_,_,_),title) in enumerate(zip(variants,names)):
                draw.text((column*192+5,7),title,fill='white')
                canvas.paste(Image.fromarray(images[name][t]).resize((192,192),Image.Resampling.NEAREST),(column*192,32))
            draw.text((5,230),f'{label} | seed {seed} | NOISE-GENERATED GAUSSIANS | native 64x64, no detail enhancement',fill='white')
            frames.append(np.asarray(canvas))
        imageio.mimsave(args.out/f'comparison_{index:02d}.mp4',frames,fps=8,codec='libx264',macro_block_size=1)
        imageio.mimsave(args.out/f'student_{index:02d}.mp4',images['student_50'],fps=8,codec='libx264',macro_block_size=1)
        for column,(name,_,_,_) in enumerate(variants):
            gallery.paste(Image.fromarray(images[name][4]).resize((144,144),Image.Resampling.NEAREST),(column*192,index*158+14))
            ImageDraw.Draw(gallery).text((column*192,index*158),f'{index:02d}: {name}',fill='white')
        if index==0: Image.fromarray(frames[4]).save(args.out/'comparison_preview.png')
        print(json.dumps(result),flush=True)
    gallery.save(args.out/'all_cases.png')
    aggregates={}
    for name,_,_,_ in variants[1:]:
        records=[c['variants'][name] for c in cases]
        values={f'mean_{key}':statistics.mean(r[key] for r in records) for key in records[0]}
        values['worst_teacher_agreement_psnr']=min(r['teacher_agreement_psnr'] for r in records)
        aggregates[name]=values
    baseline=aggregates['untrained_single_50']['mean_pixel_mse']
    for name in ('student_50','student_25'):
        m=aggregates[name]
        m['approximate_fidelity_gate_passed']=bool(m['mean_teacher_agreement_psnr']>=30 and m['worst_teacher_agreement_psnr']>=25 and m['mean_pixel_mse']<baseline)
    report=dict(teacher_sha256=digest(args.teacher),student_sha256=digest(args.student),cases=cases,
                previous_student_sha256=digest(args.previous_student) if args.previous_student else None,
                aggregates=aggregates,axes_orthogonality_max=axes_error,source_video_access=False,
                frozen_seeds=[args.first_seed,args.first_seed+19],native_shape=[8,64,64],
                limits=['Agreement with an imperfect teacher is not real-video quality or physical correctness.',
                        'The 25-step sampler uses a different schedule; no further learning or audit-seed tuning.',
                        'The untrained-single-pass control uses original conditional weights without CFG, not random weights.',
                        'All metrics precede lossy MP4 encoding. The audit is not a speed or memory benchmark.'])
    (args.out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(aggregates=aggregates,axes_orthogonality_max=axes_error),indent=2),flush=True)


if __name__=='__main__':
    with keep_windows_awake(): main()
