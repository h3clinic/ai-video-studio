"""Remote diagnostic: text-grounded fruit masks -> local Gaussian state edits.

Frozen DINO/SAM, classical boundary inpainting, cached generated apple.
Not trained motion, complete 3D reconstruction, or verified semantic ownership.
"""
import hashlib
import json
import math
import time
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from huggingface_hub import HfApi, snapshot_download
from cloud.object_edit_worker import raster, FIELDS, FOV, W, H
from real_video.local_object_repair import repair_selected,fit_bounded_exposure

ROOT=Path('/workspace/object_edit_semantic_v1')
OUT=ROOT/'output'
PRIOR=Path(os.environ.get('GAUSSIAN_PRIOR_STATE','/workspace/object_edit/output/state'))
TEXT='orange fruit. orange peel. orange slice. orange segment.'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def clock():
    torch.cuda.synchronize()
    return time.perf_counter()

def main():
    if not torch.cuda.is_available(): raise RuntimeError('Remote CUDA required')
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(6)
    cv2.setNumThreads(2)
    report=dict(accepted=False,status='loading',classification='Gaussian object editing of recorded motion',
        training_seconds=0,new_motion_generated=False,stages={},frames=[],prompt=TEXT,
        limitations=['Per-frame semantic predictions are not persistent material tracks.',
        'Classical local inpainting cannot recover unseen geometry or natural new bites.',
        'Apple appearance and placement reused from rejected v2 for controlled comparison.',
        'No complete ownership proof; masks require visual review; no speedup claim.'])
    def save(): (OUT/'report.json').write_text(json.dumps(report,indent=2))
    save()
    manifest=json.loads((ROOT/'prior_manifest.json').read_text())
    for name,pin in manifest.items():
        if digest(PRIOR/name)!=pin: raise RuntimeError('Prior Gaussian state hash mismatch')
    start=clock()
    ids=['IDEA-Research/grounding-dino-tiny','facebook/sam-vit-base']
    revisions={name:HfApi().model_info(name).sha for name in ids}
    report['model_revisions']=revisions;save()
    # Resolve pinned files explicitly before Transformers auto-class dispatch.
    # This avoids the remote AutoProcessor missing-file resolution observed in v1.
    paths=[]
    for name in ids:
        path=snapshot_download(name,revision=revisions[name],allow_patterns=['*.json','*.txt','*.safetensors'],max_workers=2)
        if not (Path(path)/'preprocessor_config.json').is_file():raise RuntimeError('Pinned processor file unavailable')
        paths.append(path)
    dp=AutoProcessor.from_pretrained(paths[0],local_files_only=True)
    dm=AutoModelForZeroShotObjectDetection.from_pretrained(paths[0],local_files_only=True,use_safetensors=True).cuda().eval()
    sp=SamProcessor.from_pretrained(paths[1],local_files_only=True)
    sm=SamModel.from_pretrained(paths[1],local_files_only=True,use_safetensors=True).cuda().eval()
    report['stages']['detector_segmenter_load_seconds']=clock()-start
    apple=np.load(ROOT/'input_apple.npz',allow_pickle=False)
    pos=apple['position'].astype('float32');rgb=apple['colour'].astype('float32')
    focal=.5*H/math.tan(math.radians(FOV)*.5)
    yy,xx=np.mgrid[:H,:W]
    roi=(xx-350.)**2/33.**2+(yy-214.)**2/30.**2<1
    frames=[];maskframes=[];before=[]
    (OUT/'deltas').mkdir()
    torch.cuda.reset_peak_memory_stats()
    for fi in range(40):
        t=clock()
        prior=np.load(PRIOR/f'{fi:03d}.npz',allow_pickle=False)
        a={k:torch.from_numpy(prior[k].copy()) for k in FIELDS}
        im=(a['colour'].numpy().reshape(H,W,3)*255).round().clip(0,255).astype('uint8')
        inputs=dp(images=Image.fromarray(im),text=TEXT,return_tensors='pt').to('cuda')
        pred=dm(**inputs)
        det=dp.post_process_grounded_object_detection(pred,inputs.input_ids,box_threshold=.25,text_threshold=.25,target_sizes=[(H,W)])[0]
        boxes=det['boxes'].cpu().numpy();scores=det['scores'].cpu().numpy()
        # Fail-closed size gate excludes implausibly scene-sized detections.
        valid=(scores>=.25)&((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])<W*H*.12)
        boxes=boxes[valid][:12]
        mask=np.zeros((H,W),bool); mask_scores=[]
        if len(boxes):
            si=sp(images=Image.fromarray(im),input_boxes=[boxes.tolist()],return_tensors='pt').to('cuda')
            so=sm(**si,multimask_output=False)
            masks=sp.image_processor.post_process_masks(so.pred_masks.cpu(),si['original_sizes'].cpu(),si['reshaped_input_sizes'].cpu())[0].numpy()
            for m,q in zip(masks,so.iou_scores[0].cpu().numpy()):
                m=m.reshape(H,W).astype(bool);quality=float(q.reshape(-1)[0])
                if quality>=.8 and 3<int(m.sum())<W*H*.08:
                    mask|=m;mask_scores.append(quality)
        semantic_seconds=clock()-t
        # Every point retains its source-frame ID. Only selected colour fields repaired.
        expanded=cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)
        residual=expanded & ~roi
        if residual.sum()>W*H*.08: raise RuntimeError('Edit area guard')
        t=clock()
        repaired=repair_selected(im,residual,expanded|roi)
        if fi==0:
            ring=cv2.dilate((expanded|roi).astype('uint8'),np.ones((15,15),np.uint8)).astype(bool)&~(expanded|roi)
            rgb,gain=fit_bounded_exposure(rgb,im[ring].astype('float32')/255)
            report['apple_display_exposure_gain']=gain
            report['limitations'].append('Frozen bounded exposure adjustment is not physical relighting; may still be mismatched.')
        colours=a['colour'].clone()
        colours[torch.from_numpy(residual.reshape(-1))]=torch.from_numpy(repaired.reshape(-1,3)[residual.reshape(-1)].astype('float32')/255)
        keep=~torch.from_numpy(roi.reshape(-1))
        edited={k:a[k][keep].clone() for k in FIELDS};edited['colour']=colours[keep]
        z=a['position'][:,2].numpy().reshape(H,W)
        cz=float(np.median(z[roi]));scale=59*cz/focal
        center=np.array([-(350.5-W/2)*cz/focal,-(211.5-H/2)*cz/focal,cz],dtype='float32')
        obj=dict(position=torch.from_numpy(pos*scale+center),
            covariance=torch.from_numpy(np.tile(np.eye(3,dtype='float32')[None]*(scale*.010)**2,(len(pos),1,1))),
            colour=torch.from_numpy(rgb),opacity=torch.ones(len(pos))*.9)
        edited={k:torch.cat([edited[k],obj[k]]) for k in FIELDS}
        repair_seconds=clock()-t
        unchanged=~torch.from_numpy((roi|residual).reshape(-1))
        exact=bool(torch.equal(a['colour'][unchanged],colours[unchanged]))
        np.savez_compressed(OUT/'deltas'/f'{fi:03d}.npz',source_sha256=manifest[f'{fi:03d}.npz'],
            removed_ids=np.flatnonzero(roi.reshape(-1)),changed_ids=np.flatnonzero(residual.reshape(-1)),
            changed_colours=colours.numpy()[residual.reshape(-1)],semantic_mask=mask,**{'apple_'+k:v.numpy() for k,v in obj.items()})
        t=clock();pic=raster(edited);raster_seconds=clock()-t
        frames.append((pic.cpu().numpy().clip(0,1)*255).round().astype('uint8'));before.append(im)
        overlay=im.copy();overlay[expanded]=(overlay[expanded]*.35+np.array([255,0,255])*.65).astype('uint8')
        maskframes.append(overlay)
        report['frames'].append(dict(frame=fi,boxes=boxes.tolist(),sam_scores=mask_scores,
            semantic_pixels=int(mask.sum()),residual_pixels=int(residual.sum()),outside_colour_exact=exact,
            semantic_seconds=semantic_seconds,repair_seconds=repair_seconds,raster_seconds=raster_seconds))
        report['status']='rendering';save()
    t=clock()
    imageio.mimsave(OUT/'gaussian_apple.mp4',frames,fps=40/(121/24),codec='libx264',macro_block_size=1)
    imageio.mimsave(OUT/'semantic_masks.mp4',maskframes,fps=40/(121/24),codec='libx264',macro_block_size=1)
    report['stages']['encoding_seconds']=clock()-t
    for i in (0,10,20,30,39): Image.fromarray(np.concatenate([before[i],maskframes[i],frames[i]],axis=1)).save(OUT/f'comparison_{i:03d}.png')
    report.update(status='completed_review_required',peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        output_sha256=digest(OUT/'gaussian_apple.mp4'))
    save()

if __name__=='__main__':
    with torch.inference_mode(): main()
