"""Summarize every predeclared run; static figures and no model-based scoring."""
import argparse
import json
from pathlib import Path
import statistics
import cv2
from PIL import Image,ImageDraw,ImageFont


def font(size): return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args(); root=args.root
    results=json.loads((root/'results.json').read_text())
    if len(results)!=28 or any(r['status']!='completed' for r in results):
        raise ValueError('Expected every predeclared run, including failures; inspect incomplete results before summarizing')
    names=['Animal / cat','Person / park','Face-like texture','Fantasy flowers','Scarecrow motion',
           'Neuron imagery','Text / lettering','Stylized dancing']
    aggregate=[]
    for g in (32,64,96):
        rows=[r['metrics'] for r in results if r['grid']==g and r['terms']==16]
        quality=[r['psnr_db'] for r in rows]
        latency=[r['gaussian_replay_median_seconds']*1000 for r in rows]
        aggregate.append(dict(grid=g,dots=g*g,cases=len(rows),mean_psnr_db=statistics.mean(quality),
                              min_psnr_db=min(quality),max_psnr_db=max(quality),
                              median_replay_ms=statistics.median(latency),min_replay_ms=min(latency),max_replay_ms=max(latency),
                              peak_cuda_mib=max(r['gaussian_peak_cuda_allocated_bytes'] for r in rows)/2**20,
                              coefficient_payload_bytes=rows[0]['fp16_coefficient_payload_bytes'],
                              median_fit_seconds=statistics.median(r['fit_seconds'] for r in rows),
                              min_gaussian_to_mp4_ratio=min(r['gaussian_file_bytes']/r['source_mp4_bytes'] for r in rows),
                              max_gaussian_to_mp4_ratio=max(r['gaussian_file_bytes']/r['source_mp4_bytes'] for r in rows)))
    report=dict(completed=len(results),aggregate=aggregate,category_labels='Descriptive labels after visual inspection; prompts retained separately in manifest',
                temporal=[r for r in results if r['grid']==64 and r['case']<=2],
                limits=['Source-conditioned fits, not noise generation.',
                        'One fitting seed, no statistical significance or large-population claim.',
                        'Same optimization updates, not matched per-parameter budget or equal quality.',
                        'Timing range is across clips, not a confidence interval.'])
    (root/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # A data-driven scientific plot, rendered without additional dependencies.
    W,H=1440,570; bg=(15,22,31); fg=(235,242,250); muted=(162,181,201); accent=(255,214,101)
    figure=Image.new('RGB',(W,H),bg); draw=ImageDraw.Draw(figure)
    draw.text((26,17),'GAUSSIAN SCALING | 8 fixed AI-video sources, all retained',font=font(29),fill=fg)
    draw.text((26,61),'256 x 256 | 16 frames | 16 temporal terms | 1,200 fitting updates per run',font=font(20),fill=muted)
    panels=[('Reconstruction quality','PSNR (dB)','mean_psnr_db',(20,42)),
            ('Rendering memory','Allocated GPU tensors (MiB)','peak_cuda_mib',(0,160)),
            ('Replay time','Milliseconds per 16-frame clip','median_replay_ms',(0,100))]
    for j,(title,unit,key,domain) in enumerate(panels):
        left=65+j*475; top=163; bottom=438; width=360
        draw.text((left-25,107),title,font=font(23),fill=fg)
        draw.text((left-25,140),unit,font=font(16),fill=muted)
        def point(n,y): return (left+(n-1024)/(9216-1024)*width,bottom-(y-domain[0])/(domain[1]-domain[0])*(bottom-top))
        for k in range(5):
            value=domain[0]+(domain[1]-domain[0])*k/4
            py=point(1024,value)[1]
            draw.line((left,py,left+width,py),fill=(43,58,75))
            draw.text((left-39,py-9),f'{value:g}',font=font(15),fill=muted)
        draw.line((left,top,left,bottom,left+width,bottom),fill=muted,width=1)
        if j==0:
            for case in range(1,9):
                rows=sorted([r for r in results if r['case']==case and r['terms']==16],key=lambda r:r['grid'])
                points=[point(r['grid']**2,r['metrics']['psnr_db']) for r in rows]
                draw.line(points,fill=(99,121,145),width=2)
        points=[point(a['dots'],a[key]) for a in aggregate]
        draw.line(points,fill=accent,width=3)
        for a,(px,py) in zip(aggregate,points):
            if j==2:
                low=point(a['dots'],a['min_replay_ms'])[1]; high=point(a['dots'],a['max_replay_ms'])[1]
                draw.line((px,low,px,high),fill=muted,width=2)
            draw.ellipse((px-4,py-4,px+4,py+4),fill=accent)
            draw.text((max(left,min(px-18,left+width-48)),py-28),f"{a[key]:.1f}",font=font(17),fill=fg)
            draw.text((max(left-12,min(px-20,left+width-46)),bottom+10),f"{a['dots']:,}",font=font(16),fill=muted)
        draw.text((left+91,bottom+42),'Gaussian dots per frame',font=font(17),fill=fg)
    draw.text((26,520),'Quality: thin lines = all 8 sources; yellow = mean. Time: median of clip medians; bars = range across sources.',font=font(18),fill=muted)
    figure.save(root/'scaling.png')
    # One fixed frame per source: original source and every dot count, no best frames.
    sheet=Image.new('RGB',(4*256,8*282+44),bg); ink=ImageDraw.Draw(sheet)
    for j,title in enumerate(('Released AI source','1,024 dots','4,096 dots','9,216 dots')):
        ink.text((j*256+8,10),title,font=font(20),fill=fg)
    for case in range(1,9):
        rows=sorted([r for r in results if r['case']==case and r['terms']==16],key=lambda r:r['grid'])
        paths=[Path(rows[0]['metrics']['source'])]+[root/r['name']/'gaussian_reconstruction.mp4' for r in rows]
        y=44+(case-1)*282
        ink.text((8,y),f'{case:02d}  {names[case-1]}',font=font(18),fill=fg)
        for j,path in enumerate(paths):
            capture=cv2.VideoCapture(str(path)); capture.set(cv2.CAP_PROP_POS_FRAMES,8)
            ok,frame=capture.read(); capture.release()
            if not ok: raise ValueError(f'Cannot decode {path}')
            sheet.paste(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),(j*256,y+26))
    sheet.save(root/'all_cases.png')
    # Generate only the measured tables; interpretation is kept in the authored report.
    lines=['# Measured tables','', '| Dots/frame | Mean PSNR (range), dB | Peak allocated MiB | Median replay ms | fp16 payload bytes |',
           '|---:|---:|---:|---:|---:|']
    for a in aggregate:
        lines.append(f"| {a['dots']:,} | {a['mean_psnr_db']:.2f} ({a['min_psnr_db']:.2f}-{a['max_psnr_db']:.2f}) | {a['peak_cuda_mib']:.2f} | {a['median_replay_ms']:.2f} | {a['coefficient_payload_bytes']:,} |")
    lines += ['', '| Visually described source | 1,024 dots PSNR | 4,096 dots PSNR | 9,216 dots PSNR |','|---|---:|---:|---:|']
    for case,name in enumerate(names,1):
        rows=sorted([r for r in results if r['case']==case and r['terms']==16],key=lambda r:r['grid'])
        lines.append('| '+name+' | '+' | '.join(f"{r['metrics']['psnr_db']:.2f}" for r in rows)+' |')
    lines += ['', '| Case | Terms | PSNR dB | fp16 payload bytes | GPU peak MiB |','|---|---:|---:|---:|---:|']
    for r in sorted(report['temporal'],key=lambda r:(r['case'],r['terms'])):
        m=r['metrics']; lines.append(f"| {r['case']} | {r['terms']} | {m['psnr_db']:.2f} | {m['fp16_coefficient_payload_bytes']:,} | {m['gaussian_peak_cuda_allocated_bytes']/2**20:.2f} |")
    (root/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__': main()
