"""Auditable 2D Gaussian asset edits. NOT new semantic video generation.

One saved visible state, approximate user-independent GrabCut selection, imported
wheat photo, deterministic controls. No learned dynamics or diffusion at edit time.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import urllib.request

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F

from .checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint
from .persistent_gaussian import render_state

OUT = Path('artifacts/real_video/gaussian_edits/v1')
SEED = Path('artifacts/real_video/persistent_gaussian/v1/cat_seed.pt')
WHEAT_URL = 'https://upload.wikimedia.org/wikipedia/commons/e/e3/Wheat_field_%28geograph_3021195%29.jpg'
WHEAT_PAGE = 'https://commons.wikimedia.org/wiki/File:Wheat_field_(geograph_3021195).jpg'
LABELS = ['1 Wheat field: imported background', '2 Shift right: same visible cat',
          '3 Enlarge: no new fur detail', '4 Mirror: NOT unseen opposite side',
          '5 Grey coat: colour edit only', '6 Upright attempt: rotation, NOT standing']
HEIGHT, WIDTH = 480, 832


def array_image(rgb):
    return (rgb[0].permute(1,2,0).clamp(0,1)*255).round().byte().cpu().numpy()


def splat_layer(points, height=HEIGHT, width=WIDTH, radius=10):
    """N,10 = xy,log widths,unit axis,RGB[-1,1],selection alpha.

    Normalized planar colour and clamped coverage, not depth-sorted 3D opacity.
    Finite support is identical in cached and recomputed benchmarks.
    """
    centre = points[:,:2]*height-.5
    scale = points[:,2:4].exp()*height
    axis = F.normalize(points[:,4:6],dim=-1)
    yy,xx = torch.meshgrid(torch.arange(-radius,radius+1,device=points.device),
                           torch.arange(-radius,radius+1,device=points.device),indexing='ij')
    offsets=torch.stack((xx.flatten(),yy.flatten()),-1)
    pixels=centre.floor().long()[:,None]+offsets[None]
    delta=pixels.to(points.dtype)-centre[:,None]
    local=torch.stack(((delta*axis[:,None]).sum(-1),
                       -delta[...,0]*axis[:,None,1]+delta[...,1]*axis[:,None,0]),-1)
    weights=(-.5*(local/scale[:,None]).square().sum(-1)).exp()*points[:,None,9]
    valid=(pixels[...,0]>=0)&(pixels[...,0]<width)&(pixels[...,1]>=0)&(pixels[...,1]<height)
    weights=weights*valid
    index=(pixels[...,1].clamp(0,height-1)*width+pixels[...,0].clamp(0,width-1)).flatten()
    denom=points.new_zeros(height*width).scatter_add(0,index,weights.flatten())
    values=((points[:,6:9]+1)/2)[:,None]*weights[...,None]
    numerator=points.new_zeros(height*width,3).scatter_add(0,index[:,None].expand(-1,3),values.reshape(-1,3))
    rgb=(numerator/denom.clamp_min(1e-6)[:,None]).T.reshape(1,3,height,width)
    alpha=denom.clamp(0,1).reshape(1,1,height,width)
    return rgb,alpha


def controls(case, fraction):
    # dx,dy in height units; uniform scale; angle; mirror flag; greyscale mix.
    c=[0.,0.,1.,0.,0.,0.]
    if case==1: c[0]=.28*fraction
    elif case==2: c[2]=1+.25*fraction
    elif case==3: c[4]=1.
    elif case==4: c[5]=fraction
    elif case==5: c[2]=1-.22*fraction; c[3]=-math.pi/2*fraction; c[0]=.17*fraction; c[1]=-.08*fraction
    return c


def affine(control, reference):
    dx,dy,scale,angle,mirror,grey=control
    cs,sn=math.cos(angle),math.sin(angle)
    matrix=reference.new_tensor([[cs,-sn],[sn,cs]])
    matrix[:,0]*=1-2*mirror
    return scale*matrix,reference.new_tensor([dx,dy])


def transform(points, pivot, control):
    matrix,offset=affine(control,points)
    result=points.clone()
    result[:,:2]=(points[:,:2]-pivot)@matrix.T+pivot+offset
    result[:,2:4]=points[:,2:4]+math.log(control[2])
    result[:,4:6]=F.normalize(points[:,4:6]@matrix.T,dim=-1)
    colour=(points[:,6:9]+1)/2
    luminance=(colour*points.new_tensor([.2126,.7152,.0722])).sum(-1,keepdim=True)
    result[:,6:9]=((1-control[5])*colour+control[5]*luminance)*2-1
    return result


def photo_points(image, gh=120, gw=208):
    colour=F.interpolate(image,size=(gh,gw),mode='area')[0].permute(1,2,0).reshape(-1,3)
    yy,xx=torch.meshgrid(torch.arange(gh),torch.arange(gw),indexing='ij')
    pos=torch.stack(((xx+.5)*WIDTH/gw/HEIGHT,(yy+.5)/gh),-1).reshape(-1,2)
    logs=torch.full_like(pos,math.log(.65/gh))
    axes=torch.zeros_like(pos); axes[:,0]=1
    return torch.cat((pos,logs,axes,colour*2-1,torch.ones(gh*gw,1)),1)


@torch.no_grad()
def preview():
    OUT.mkdir(parents=True,exist_ok=True)
    packet=load_verified(SEED)
    rgb=render_state(packet['observed_states'][:,:,1].cuda(),HEIGHT,WIDTH,radius=8)
    Image.fromarray(array_image(rgb)).save(OUT/'source_state.png')
    print('Saved source_state.png for segmentation inspection',flush=True)


@torch.no_grad()
def prepare():
    if (OUT/'assets.pt').exists(): raise FileExistsError('Preserve existing assets')
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    packet=load_verified(SEED)
    fields=packet['observed_states'][:,:,1].contiguous()
    source=array_image(render_state(fields.cuda(),HEIGHT,WIDTH,radius=8))
    Image.fromarray(source).save(OUT/'source_state.png')
    # Hand-specified region and colour seeds, explicitly not learned object memory.
    mask=np.full((HEIGHT,WIDTH),cv2.GC_BGD,np.uint8)
    mask[100:470,0:580]=cv2.GC_PR_BGD
    hsv=cv2.cvtColor(source,cv2.COLOR_RGB2HSV)
    foreground=(hsv[:,:,0]<28)&(hsv[:,:,1]>55)&(source[:,:,0]>80)
    foreground[:100]=False; foreground[:,580:]=False; foreground[470:]=False
    mask[foreground]=cv2.GC_PR_FGD
    sure=cv2.erode(foreground.astype(np.uint8),np.ones((9,9),np.uint8))>0
    mask[sure]=cv2.GC_FGD
    cv2.setRNGSeed(430102)
    cv2.grabCut(source,mask,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
    selection=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(selection,8)
    if count<2: raise ValueError('No selected foreground')
    selection=(labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.float32)
    Image.fromarray((selection*255).astype(np.uint8)).save(OUT/'selection_mask.png')
    flat=fields[0].flatten(1).T
    xy=(flat[:,:2]*HEIGHT-.5).round().long()
    selected=torch.from_numpy(selection)[xy[:,1].clamp(0,HEIGHT-1),xy[:,0].clamp(0,WIDTH-1)]
    cat=torch.cat((flat[selected>.5],selected[selected>.5,None]),1)
    pivot=(cat[:,:2].amin(0)+cat[:,:2].amax(0))/2
    stage=time.perf_counter()
    photo=OUT/'wheat_original.jpg'
    if not photo.exists():
        request=urllib.request.Request(WHEAT_URL,headers={'User-Agent':'GaussianResearchPilot/1.0 (local research; source attribution retained)'})
        with urllib.request.urlopen(request,timeout=60) as response: data=response.read()
        photo.write_bytes(data)
    download=time.perf_counter()-stage
    wheat=ImageOps.fit(Image.open(photo).convert('RGB'),(WIDTH,HEIGHT),method=Image.Resampling.LANCZOS)
    wheat.save(OUT/'wheat_crop.png')
    wheat_tensor=torch.from_numpy(np.array(wheat).copy()).permute(2,0,1)[None].float()/255
    background=photo_points(wheat_tensor)
    catrgb,catalpha=splat_layer(cat.cuda())
    bgrgb,_=splat_layer(background.cuda())
    torch.cuda.synchronize()
    save_inference_checkpoint(dict(cat=cat,pivot=pivot,background=background,
                                   source_sha256=digest(SEED),wheat_sha256=digest(photo),
                                   source='Approximate selected visible planar cat; not learned semantic identity'),OUT/'assets.pt')
    Image.fromarray(array_image(catrgb*catalpha+.15*(1-catalpha))).save(OUT/'cat_layer.png')
    Image.fromarray(array_image(catrgb*catalpha+bgrgb*(1-catalpha))).save(OUT/'wheat_composite.png')
    record=dict(source_seed=str(SEED),source_sha256=digest(SEED),cat_splats=len(cat),background_splats=len(background),
                cat_tensor_bytes=cat.numel()*4,background_tensor_bytes=background.numel()*4,
                pivot_tensor_bytes=pivot.numel()*4,preparation_seconds=time.perf_counter()-start,
                photo_download_seconds=download,source_wheat_url=WHEAT_URL,source_wheat_page=WHEAT_PAGE,
                wheat_author='Neil Oakes',wheat_license='CC BY-SA 2.0',
                license_url='https://creativecommons.org/licenses/by-sa/2.0/',
                derivative_license='CC BY-SA 2.0 for wheat composites/videos',
                changes='crop/resize, Gaussian resampling, foreground compositing, annotation',
                limitations='One visible 2D cutout selected using manual region/colour heuristics and GrabCut. Not segmentation learned by the dynamics. No new pose, viewpoint, fur, disoccluded content, shadows or physical interaction.')
    (OUT/'preparation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)


class Editor:
    def __init__(self,packet,mode):
        self.mode=mode
        self.cat=packet['cat'].cuda(); self.pivot=packet['pivot'].cuda()
        background=packet['background'].cuda()
        if mode=='fresh_gaussians': self.background=background
        else: self.bg,_=splat_layer(background)
        if mode=='cached_raster':
            rgb,alpha=splat_layer(self.cat)
            self.sprite=torch.cat((rgb*alpha,alpha),1)
            yy,xx=torch.meshgrid(torch.arange(HEIGHT,device='cuda'),torch.arange(WIDTH,device='cuda'),indexing='ij')
            self.output_xy=torch.stack(((xx+.5)/HEIGHT,(yy+.5)/HEIGHT),-1).float()
            del self.cat

    def __call__(self,control):
        if self.mode=='cached_raster':
            matrix,offset=affine(control,self.sprite)
            xy=(self.output_xy-self.pivot-offset)@torch.linalg.inv(matrix).T+self.pivot
            grid=torch.stack((2*xy[...,0]*HEIGHT/WIDTH-1,2*xy[...,1]-1),-1)[None]
            rgba=F.grid_sample(self.sprite,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            rgb,alpha=rgba[:,:3],rgba[:,3:]
            lum=(rgb*rgb.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
            rgb=(1-control[5])*rgb+control[5]*lum
            return rgb+self.bg*(1-alpha)
        cat=transform(self.cat,self.pivot,control)
        rgb,alpha=splat_layer(cat)
        bg=splat_layer(self.background)[0] if self.mode=='fresh_gaussians' else self.bg
        return rgb*alpha+bg*(1-alpha)


@torch.no_grad()
def benchmark(mode,trial=0):
    suffix='' if trial==0 else f'_trial{trial}'
    path=OUT/f'benchmark_{mode}{suffix}.json'
    if path.exists(): raise FileExistsError(path)
    packet=load_verified(OUT/'assets.pt')
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start=time.perf_counter(); editor=Editor(packet,mode); torch.cuda.synchronize()
    setup_seconds=time.perf_counter()-start; setup_peak=torch.cuda.max_memory_allocated()
    for _ in range(4): result=editor(controls(2,.5)); del result
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    resident=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    records=[]
    for repeat in range(10):
        for case in range(6):
            control=controls(case,(repeat%5+1)/5)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); wall=time.perf_counter(); begin.record()
            result=editor(control); end.record(); torch.cuda.synchronize()
            records.append(dict(case=case+1,repeat=repeat,wall_ms=(time.perf_counter()-wall)*1000,gpu_ms=begin.elapsed_time(end)))
            del result
    report=dict(mode=mode,trial=trial,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                setup_seconds=setup_seconds,setup_peak_allocated_bytes=setup_peak,
                resident_allocated_bytes=resident,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),records=records,
                median_wall_ms=float(np.median([r['wall_ms'] for r in records])),
                median_gpu_ms=float(np.median([r['gpu_ms'] for r in records])),
                by_case={str(case+1):dict(median_wall_ms=float(np.median([r['wall_ms'] for r in records if r['case']==case+1]))) for case in range(6)},
                scope='60 warmed edit+render calls, 832x480, 10/case; no download, extraction, load, video encoding or diffusion. Separate process per mode. CUDA allocated tensors, not whole-device or CPU RAM. Fixed source view, deterministic edits only.')
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


@torch.no_grad()
def demonstrate():
    if (OUT/'six_examples.mp4').exists(): raise FileExistsError('Preserve existing demo')
    packet=load_verified(OUT/'assets.pt')
    editor=Editor(packet,'cached_gaussians')
    fresh=Editor(packet,'fresh_gaussians'); raster=Editor(packet,'cached_raster')
    audits=[]
    for case in range(6):
        control=controls(case,1)
        a,b,c=editor(control),fresh(control),raster(control)
        diff=(a-b).abs().max().item()
        if diff>1e-6: raise AssertionError(f'Cache changed output: {diff}')
        audits.append(dict(case=case+1,cached_vs_fresh_max_abs=diff,raster_vs_gaussian_mse=(a-c).square().mean().item()))
    del fresh,raster,a,b,c
    source_sha=digest(OUT/'assets.pt')
    writers=[imageio.get_writer(OUT/f'example_{i+1}.mp4',fps=16,codec='libx264',quality=8,macro_block_size=1) for i in range(6)]
    overview=imageio.get_writer(OUT/'six_examples.mp4',fps=16,codec='libx264',quality=8,macro_block_size=1)
    try:
        for frame in range(33):
            sheet=Image.new('RGB',(1248,592),'#141820')
            draw=ImageDraw.Draw(sheet)
            draw.text((12,8),'SAVED 2D GAUSSIAN EDITS | deterministic controls, NOT six new AI-generated scenes',fill='white')
            draw.text((12,26),'Wheat: Neil Oakes / CC BY-SA 2.0 | Imported photo, cropped + splatted | No new cat pose generated',fill='white')
            for case in range(6):
                rgb=editor(controls(case,frame/32))
                arr=array_image(rgb)
                canvas=Image.new('RGB',(WIDTH,HEIGHT+48),'#141820'); canvas.paste(Image.fromarray(arr),(0,48))
                d=ImageDraw.Draw(canvas); d.text((10,6),LABELS[case],fill='white')
                d.text((10,25),'2D asset edit | Wheat: Neil Oakes, CC BY-SA 2.0 | no diffusion or learned pose update',fill='white')
                writers[case].append_data(np.asarray(canvas))
                col,row=case%3,case//3
                sheet.paste(Image.fromarray(arr).resize((416,240)),(col*416,64+row*264))
                draw.text((col*416+6,48+row*264),LABELS[case],fill='white')
                if frame==32: canvas.save(OUT/f'example_{case+1}.png')
            overview.append_data(np.asarray(sheet))
            if frame==32: sheet.save(OUT/'six_examples.jpg')
    finally:
        for writer in writers: writer.close()
        overview.close()
    assert digest(OUT/'assets.pt')==source_sha
    report=dict(assets_unchanged=True,source_assets_sha256=source_sha,frames_per_example=33,
                edits=LABELS,cache_equivalence=audits,
                no_wan_no_diffusion_no_learned_recurrent_model=True,
                semantic_upright_pose_achieved=False,
                video_sha256={p.name:digest(p) for p in OUT.glob('*.mp4')})
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def summarize():
    if (OUT/'REPORT.md').exists(): raise FileExistsError('Preserve completed report')
    groups={}
    for mode in ['fresh_gaussians','cached_gaussians','cached_raster']:
        records=[json.loads(p.read_text()) for p in sorted(OUT.glob(f'benchmark_{mode}*.json'))]
        if len(records)!=3: raise ValueError('Expected three independent processes per mode')
        values=[r['median_wall_ms'] for r in records]
        groups[mode]=dict(median_of_process_medians_ms=float(np.median(values)),
                          process_medians_ms=values,calls=180,
                          warmed_peak_bytes=max(r['peak_allocated_bytes'] for r in records),
                          setup_inclusive_peak_bytes=max(max(r['setup_peak_allocated_bytes'],r['peak_allocated_bytes']) for r in records),
                          resident_bytes=max(r['resident_allocated_bytes'] for r in records),
                          setup_seconds=[r['setup_seconds'] for r in records],
                          per_case_ms={str(c):float(np.median([r['by_case'][str(c)]['median_wall_ms'] for r in records])) for c in range(1,7)})
    fresh,cached,raster=(groups[m] for m in ['fresh_gaussians','cached_gaussians','cached_raster'])
    prep=json.loads((OUT/'preparation.json').read_text()); audit=json.loads((OUT/'audit.json').read_text())
    decoded={}
    for path in sorted(OUT.glob('*.mp4')):
        cap=cv2.VideoCapture(str(path)); count=0; shape=None
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                count+=1; shape=list(frame.shape)
        finally: cap.release()
        if count!=33: raise AssertionError(f'Bad frame count {path}: {count}')
        if digest(path)!=audit['video_sha256'][path.name]: raise AssertionError('Video changed')
        decoded[path.name]=dict(frames=count,shape=shape,bytes=path.stat().st_size)
    summary=dict(groups=groups,decoded_videos=decoded,
                  warmed_latency_ratio=fresh['median_of_process_medians_ms']/cached['median_of_process_medians_ms'],
                  warmed_latency_reduction_percent=100*(1-cached['median_of_process_medians_ms']/fresh['median_of_process_medians_ms']),
                  warmed_peak_reduction_percent=100*(1-cached['warmed_peak_bytes']/fresh['warmed_peak_bytes']),
                  setup_inclusive_peak_reduction_percent=100*(1-cached['setup_inclusive_peak_bytes']/fresh['setup_inclusive_peak_bytes']),
                  foreground_samples_per_frame=prep['cat_splats']*441,
                  fresh_samples_per_frame=(prep['cat_splats']+prep['background_splats'])*441,
                  asset_file_bytes=(OUT/'assets.pt').stat().st_size,
                  max_cache_pixel_error=max(v['cached_vs_fresh_max_abs'] for v in audit['cache_equivalence']))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    rows=[]
    for mode,g in groups.items():
        rows.append(f"| {mode} | {g['median_of_process_medians_ms']:.2f} | {g['warmed_peak_bytes']/2**20:.2f} | {g['setup_inclusive_peak_bytes']/2**20:.2f} | {g['resident_bytes']/2**20:.2f} |")
    percase=[]
    for i,label in enumerate(LABELS,1):
        percase.append(f"| {label} | {fresh['per_case_ms'][str(i)]:.2f} | {cached['per_case_ms'][str(i)]:.2f} | {raster['per_case_ms'][str(i)]:.2f} |")
    report=f'''# Six Gaussian edit tests — reuse works; semantic cat posing does not

## What is actually stored

The previous model stored a planar Gaussian representation of visible scene
appearance, not a separable, rigged or complete 3D cat. For this experiment we
selected ONE last visible cat state from the existing two-state packet. An
explicitly hand-specified region/colour heuristic plus seeded OpenCV GrabCut
made an approximate foreground mask. This selection was not learned by the
persistent dynamics. No new learned weights were trained in this experiment.

The resulting asset has {prep['cat_splats']:,} foreground Gaussians and
{prep['background_splats']:,} background Gaussians. The latter come from an imported
photograph, not a generated wheat world. Geometry, axes, colour and selection
opacity are retained; velocity and learned hidden memory are NOT used by this
editor. Controls are deterministic. It is not evidence of learned object recall.

## Six delivered tests

1. Replace background with a wheat-field photograph converted to Gaussians.
2. Translate the selected visible cat to the right.
3. Enlarge it; this does not recover fur detail.
4. Mirror it; this does not reveal its actual unseen opposite side.
5. Desaturate the coat; no new fur or lighting is synthesized.
6. Attempt upright orientation by rigid 2D rotation. **Standing-pose request
   fails:** limbs and anatomy are not reposed, and it visibly floats/rotates.

`six_examples.mp4` contains all six labeled panels; `example_1.mp4` through
`example_6.mp4` are individual 33-frame, 16fps previews. All use the SAME frozen
visible pose with varying controls, not new learned gait or continuation.
Visual inspection: inherited blur, imperfect cutout edge/green halo, mismatched
scale and ground contact, clipped source tail, no wheat occlusion or shadows.
These are compositing tests, not six successful realistic AI video generations.

## Measured processing evidence

832x480, RTX 5070 Laptop, PyTorch 2.11.0+cu128. Three independent processes
per method, 60 warmed calls/process, ten calls/case/process. Orders were
fresh/cached/raster, raster/cached/fresh, cached/fresh/raster. Reported latency
is the median of three process medians, with synchronization around each call.
Raw per-call wall and CUDA-event times are in `benchmark_*.json`. Clocks and
thermals were not locked; these are local pilot measurements, not universal rates.

| Method | Edit + render ms | Warmed peak MiB | Setup-inclusive peak MiB | Warm resident MiB |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Fresh renders cat AND wheat every call. Cached Gaussians render the same cat
but reuse a rasterized wheat buffer. Raster baseline also caches a premultiplied
cat image and transforms it with bilinear sampling, conventional layer editing.

Caching the background reduces warmed latency by
**{summary['warmed_latency_reduction_percent']:.1f}% ({summary['warmed_latency_ratio']:.2f}x)** and warmed peak
allocated tensors by **{summary['warmed_peak_reduction_percent']:.1f}%** against fresh rendering.
But setup renders the background once: **setup-inclusive peak falls only
{summary['setup_inclusive_peak_reduction_percent']:.1f}%**. Resident GPU allocation increases because of the cached pixels.
Do not describe warmed-peak reduction as equivalent end-to-end memory savings.

Maximum normalized RGB discrepancy between fresh and cached Gaussians over six
final edits is {summary['max_cache_pixel_error']:.3g}, below 1e-6. This supports an output-matched
cache optimization. Original assets remain byte-identical. Raster comparison
is not bit-identical under resampling; its MSE is at most
{max(v['raster_vs_gaussian_mse'] for v in audit['cache_equivalence']):.3g} against our Gaussian editor on those endpoints.
**Raster editing is faster and has lower warmed peak allocation in this test.**
Therefore these results do NOT establish an advantage unique to Gaussian vectors.

| Case | Fresh Gaussian ms | Cached Gaussian ms | Cached raster ms |
|---|---:|---:|---:|
{chr(10).join(percase)}

## Why the cache saves work

This renderer evaluates a 21x21 window per splat. Per-frame candidate samples
fall from ({prep['cat_splats']} + {prep['background_splats']}) x 441 =
{summary['fresh_samples_per_frame']:,} to {prep['cat_splats']} x 441 =
{summary['foreground_samples_per_frame']:,}, after the one-time background render.
It still composites a full image each frame. This is counted renderer work,
not measured total FLOPs, electrical energy or a GPU-instruction count.

Latency excludes downloading, mask creation, asset extraction, model/source
generation, loading, output transfer and MP4 encoding. Preparation took
{prep['preparation_seconds']:.2f}s including {prep['photo_download_seconds']:.2f}s download;
per-method setup times are retained in the raw reports. Whole-device VRAM,
driver allocations and CPU RAM are not covered by PyTorch allocated-byte peaks.
No Wan regeneration baseline was run because it would solve a different,
unmatched-quality task; previous Wan wall time includes a battery pause and
cannot be used to claim a speedup here.

## Storage and memory truth

Cat attributes: {prep['cat_tensor_bytes']:,} bytes. Wheat attributes:
{prep['background_tensor_bytes']:,} bytes. Pivot: 8 bytes. Exported asset file:
{summary['asset_file_bytes']:,} bytes. Six-float controls are small and reuse the
same base asset rather than duplicating it. This does NOT store all poses or
past frames. The Gaussian wheat attributes alone exceed the original JPEG's
{(OUT/'wheat_original.jpg').stat().st_size:,} bytes. There is no matched codec/compression superiority claim.

## Verification and research conclusion

74 unit tests pass, including transform covariance, source ownership, mirror
inversion and alpha bounds. All seven MP4s decode to 33 frames and hashes match.
Raw results and failed pose attempt are retained. No original checkpoint is changed.

What is supported: a visible Gaussian asset can be reused for cheap constrained
edits, and avoiding static-background rerendering measurably saves work.
What is NOT supported: stored anatomical cat, novel realistic poses, hidden
surfaces, learned controlled dynamics, or cheaper quality-matched video generation.
The missing research step is object/part-aware canonical state plus trained
pose/deformation and visibility/completion, with a raster baseline retained.

## Background attribution

[Wheat field, Neil Oakes]({WHEAT_PAGE}),
[CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/).
Changes: crop/resize, Gaussian resampling, foreground composite and annotation.
The wheat composites and their videos are shared under CC BY-SA 2.0.
Source photograph and metadata are retained; no endorsement is implied.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preview','prepare','fresh_gaussians','cached_gaussians','cached_raster','demo','report'])
    parser.add_argument('--trial',type=int,default=0)
    args=parser.parse_args(); torch.set_num_threads(4)
    with keep_windows_awake():
        if args.mode=='preview': preview()
        elif args.mode=='prepare': prepare()
        elif args.mode=='demo': demonstrate()
        elif args.mode=='report': summarize()
        else: benchmark(args.mode,args.trial)


if __name__=='__main__': main()
