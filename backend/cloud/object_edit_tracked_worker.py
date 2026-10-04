"""Remote source-observation repair candidate; explicitly partial/unaccepted.

Whole canonical fruit can replace a separately detected fruit body. Missing
slice/peel assets are preserved, not erased. Source motion, inferred depth and
optical-flow mask associations are observations, NOT generated action/contact.
Previous worker and all prior artifacts remain unchanged.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time
from contextlib import contextmanager

ROOT = Path('/workspace/object_edit_semantic_v1')
TEXT = 'orange fruit. orange peel. orange slice. orange segment. donkey. wooden bowl.'
GROUNDING_QUERIES = ('orange fruit.', 'orange peel.', 'orange slice.',
                     'orange segment.', 'donkey.', 'wooden bowl.')


def independent_grounding(image, processor, model, height, width):
    """One semantic phrase per query; keep provenance and fail closed.

    Joint captions produced merged slice/segment labels and broad fruit masks
    in v6. Isolating captions avoids cross-phrase token assignment, but does
    NOT prove SAM masks are correct. Unknown/partial labels remain protected;
    no change to the occluder or projected coverage gates.
    """
    records = []
    for query in GROUNDING_QUERIES:
        inputs = processor(images=image, text=query, return_tensors='pt').to('cuda')
        predicted = model(**inputs)
        found = processor.post_process_grounded_object_detection(
            predicted, inputs.input_ids, box_threshold=.25, text_threshold=.25,
            target_sizes=[(height, width)])[0]
        boxes = found['boxes'].cpu().numpy(); scores = found['scores'].cpu().numpy()
        labels = found.get('text_labels', found.get('labels', []))
        if len(labels) != len(boxes) or len(scores) != len(boxes) or any(not isinstance(label,str) for label in labels):
            raise RuntimeError('Detector did not return aligned textual instance labels')
        for box, score, label in zip(boxes, scores, labels):
            if len(box)!=4 or not all(math.isfinite(float(x)) for x in box) or not math.isfinite(float(score)):
                raise RuntimeError('Invalid detector geometry or confidence')
            area = float((box[2]-box[0])*(box[3]-box[1]))
            kind = label_kind(label)
            # A returned label from a different query is never promoted to an
            # editable body, even if its words happen to name one.
            if ' '.join(label.lower().strip(' .').split()) != query.rstrip('.'):
                kind = 'unknown'
            limit = .9 if kind == 'protected' else .12
            if float(score)>=.25 and box[2]>box[0] and box[3]>box[1] and 4<area<width*height*limit:
                records.append(dict(box=box.tolist(), score=float(score), label=label,
                                    kind=kind, grounding_query=query))
            if len(records)>16:
                raise RuntimeError('Instance cap exceeded; no silent mask truncation')
    return records


def label_kind(label):
    value = ' '.join(str(label).lower().strip(' .').split())
    if value == 'orange fruit': return 'whole_fruit'
    if value in ('orange peel','orange slice','orange segment'): return 'unsupported_piece'
    if value in ('donkey','wooden bowl','bowl'): return 'protected'
    return 'unknown'


@contextmanager
def exclusive_gpu():
    """Serialize our remote workers, then reject any pre-existing GPU process.

    Linux remote-only gate, before Torch/model loading. A cooperative lock does
    not control unrelated software; the Pod-level bounded watchdog remains
    required. Closing the descriptor releases the advisory lock on all exits.
    """
    import fcntl
    import subprocess
    with open('/tmp/gaussian_studio_gpu.lock','a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Gaussian GPU worker owns the shared lock') from None
        active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
            capture_output=True,text=True,timeout=10,check=True)
        if active.stdout.strip():raise RuntimeError('Existing GPU work detected; refusing concurrent execution')
        yield


def _run():
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import psutil
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
    from huggingface_hub import HfApi, snapshot_download
    from cloud.object_edit_worker import raster, FIELDS, FOV, W, H
    from real_video.gaussian3d import render
    from real_video.local_object_repair import fit_bounded_exposure
    from real_video.observed_part_placement import (mask_iou, associate_observations,
        partition_replacement_masks, fit_observed_placement, evaluate_replacement_coverage)

    if not torch.cuda.is_available() or psutil.virtual_memory().available < 8*2**30:
        raise RuntimeError('Remote CUDA and 8 GiB available RAM required')
    torch.set_num_threads(6); cv2.setNumThreads(2)
    out = ROOT/'output'; out.mkdir(exist_ok=False)
    prior_path = Path(os.environ.get('GAUSSIAN_PRIOR_STATE','/workspace/object_edit/output/state'))
    manifest = json.loads((ROOT/'prior_manifest.json').read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, pin in manifest.items():
        if digest(prior_path/name) != pin: raise RuntimeError('Prior state checksum mismatch')
    started = time.perf_counter()
    def clock():
        torch.cuda.synchronize(); return time.perf_counter()
    report = dict(accepted=False, task_complete=False, status='loading', frames=[], stages={},
        classification='Partial observed-instance Gaussian replacement; not new motion generation',
        new_motion_generated=False, training_seconds=0, model_weights_changed=False,
        source_state_manifest_sha256=digest(ROOT/'prior_manifest.json'),
        canonical_asset_sha256=digest(ROOT/'input_apple.npz'), prompt=TEXT,
        grounding_queries=list(GROUNDING_QUERIES), semantic_revision='independent-phrases-v2-unvalidated',
        limitations=['Whole fruit has a replacement asset; slices/peel/held bites do not and are preserved.',
            'DINO/SAM instances and DIS mask associations are predictions, not verified ownership.',
            'Single-view foreground depth and bbox alignment do not prove 3D support/contact.',
            'Alpha coverage/spill gates are diagnostics, not pixel-perfect protected-region proofs.',
            'Insufficiently covered original body samples remain; residual orange boundaries require review.',
            'No unseen background inpainting or newly generated eating action.',
            'Canonical Shap-E appearance remains coarse; no texture/weight fitting performed.'])
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2))
    save()
    start = clock()
    names = ('IDEA-Research/grounding-dino-tiny','facebook/sam-vit-base')
    revisions = {name:HfApi().model_info(name).sha for name in names}
    report['model_revisions'] = revisions
    paths = [snapshot_download(name,revision=revisions[name],allow_patterns=['*.json','*.txt','*.safetensors'],max_workers=2) for name in names]
    dp = AutoProcessor.from_pretrained(paths[0],local_files_only=True)
    dm = AutoModelForZeroShotObjectDetection.from_pretrained(paths[0],local_files_only=True,use_safetensors=True).cuda().eval()
    sp = SamProcessor.from_pretrained(paths[1],local_files_only=True)
    sm = SamModel.from_pretrained(paths[1],local_files_only=True,use_safetensors=True).cuda().eval()
    report['stages']['detector_segmenter_load_seconds'] = clock()-start
    report['status']='models_loaded';save()
    with np.load(ROOT/'input_apple.npz',allow_pickle=False) as data:
        position = data['position'].astype('float32'); colour = data['colour'].astype('float32')
    if (position.shape != (20000,3) or colour.shape != position.shape or not np.isfinite(position).all()
            or not np.isfinite(colour).all() or (colour < 0).any() or (colour > 1).any()):
        raise ValueError('Unexpected canonical asset')
    canonical_ids = np.arange(len(position),dtype=np.int64)
    np.savez_compressed(out/'canonical_memory.npz',ids=canonical_ids,position=position,colour=colour,
                        source_sha256=report['canonical_asset_sha256'])
    focal = .5*H/math.tan(math.radians(FOV)*.5)
    yy,xx = np.mgrid[:H,:W]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    previous_gray = None; previous_tracks = []; track_counter = 0; initialized=False
    frames=[]; overlays=[]; sources=[]; gain=None
    for name in ('instances','deltas'): (out/name).mkdir()
    torch.cuda.reset_peak_memory_stats()
    for frame_index in range(40):
        frame_clock=clock();t=clock()
        with np.load(prior_path/f'{frame_index:03d}.npz',allow_pickle=False) as prior:
            scene={key:torch.from_numpy(prior[key].copy()) for key in FIELDS}
        if (tuple(scene['position'].shape)!=(H*W,3) or tuple(scene['colour'].shape)!=(H*W,3)
                or tuple(scene['covariance'].shape)!=(H*W,3,3) or tuple(scene['opacity'].shape)!=(H*W,)):
            raise ValueError('Pinned source must contain one row-major Gaussian per image pixel')
        image=(scene['colour'].numpy().reshape(H,W,3)*255).round().clip(0,255).astype('uint8')
        load_seconds=clock()-t;t=clock()
        records=independent_grounding(Image.fromarray(image),dp,dm,H,W)
        for index,record in enumerate(records):
            record['observation_id']=f'frame_{frame_index:03d}_instance_{index:03d}'
        # Preserve the source on over-cap ambiguity instead of silently omitting masks.
        if len(records)>16:raise RuntimeError('Instance cap exceeded; no silent mask truncation')
        if records:
            si=sp(images=Image.fromarray(image),input_boxes=[[row['box'] for row in records]],return_tensors='pt').to('cuda')
            so=sm(**si,multimask_output=False)
            masks=sp.image_processor.post_process_masks(so.pred_masks.cpu(),si['original_sizes'].cpu(),si['reshaped_input_sizes'].cpu())[0].numpy()
            for record,mask,quality in zip(records,masks,so.iou_scores[0].cpu().numpy()):
                record['mask']=mask.reshape(H,W).astype(bool);record['sam_score']=float(quality.reshape(-1)[0])
        for record in records:
            record['mask_eligible']=record['sam_score']>=.8 and 3<int(record['mask'].sum())<W*H*(.9 if record['kind']=='protected' else .08)
        semantic_seconds=clock()-t;t=clock()
        # Persist raw predicted instance observations BEFORE fitting can fail.
        # IDs here identify observations in one frame, NOT tracked material.
        mask_data=np.stack([row['mask'] for row in records]) if records else np.zeros((0,H,W),bool)
        np.savez_compressed(out/'instances'/f'{frame_index:03d}.npz',masks=mask_data,
            observation_ids=np.array([row['observation_id'] for row in records],dtype='U80'),
            labels=np.array([row['label'] for row in records],dtype='U80'),
            scores=np.array([row['score'] for row in records]),sam_scores=np.array([row['sam_score'] for row in records]),
            boxes=np.array([row['box'] for row in records]),source_sha256=manifest[f'{frame_index:03d}.npz'])
        instance_persistence_seconds=clock()-t;t=clock()
        # Even low-confidence detections protect their boxes, but never authorize writes.
        protected=np.zeros((H,W),bool)
        for record in records:
            if record['kind']!='whole_fruit' or not record['mask_eligible']:
                if record['mask_eligible']: protected|=record['mask']
                else:
                    x0,y0,x1,y1=record['box'];protected[max(0,int(y0)):min(H,math.ceil(y1)),max(0,int(x0)):min(W,math.ceil(x1))]=True
        whole=[];duplicate_count=0
        for record in sorted(records,key=lambda row:-row['score']):
            if record['kind']!='whole_fruit' or not record['mask_eligible']:continue
            if any(mask_iou(record['mask'],kept['mask'])>.8 for kept in whole):
                duplicate_count+=1;continue
            whole.append(record)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        old=previous_tracks
        if old and previous_gray is not None:
            flow=dis.calc(gray,previous_gray,None)
            old=[dict(row,mask=cv2.remap(row['mask'].astype('uint8'),(xx+flow[:,:,0]).astype('float32'),
                (yy+flow[:,:,1]).astype('float32'),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).astype(bool)) for row in old]
        matches=associate_observations(old,whole) if old else {}
        # Initialize on the first qualified observation, not blindly at frame 0.
        # Once initialized, lost identities are NOT silently reallocated later.
        if not initialized and whole:
            for index in range(len(whole)):
                matches[index]=f'whole_fruit_{track_counter:03d}';track_counter+=1
            initialized=True;report['track_initialization_frame']=frame_index
        tracks=[dict(part_id=matches[i],observation_id=row['observation_id'],label=row['label'],mask=row['mask'])
                for i,row in enumerate(whole) if i in matches]
        lost_track_ids=sorted({row['part_id'] for row in previous_tracks}-{row['part_id'] for row in tracks})
        masks_by_id={row['part_id']:row['mask'] for row in tracks}
        # Unmatched whole bodies must also protect their samples from another replacement.
        for index,row in enumerate(whole):
            if index not in matches:protected|=row['mask']
        partition=partition_replacement_masks(masks_by_id,set(masks_by_id),protected)
        tracking_seconds=clock()-t;t=clock()
        observation_evidence=dict(frame=frame_index,source_sha256=manifest[f'{frame_index:03d}.npz'],
            instances=[{key:value for key,value in row.items() if key!='mask'} for row in records],
            track_observations=[{key:value for key,value in row.items() if key!='mask'} for row in tracks],
            instance_pixel_counts=partition['instance_pixel_counts'],protected_pixels=int(protected.sum()),
            ambiguous_pixels=int(partition['ambiguous'].sum()),lost_track_ids=lost_track_ids,
            scene_depth_contract='position[:,2].reshape(H,W), row-major observed relative camera depth; not metric ground truth')
        (out/'instances'/f'{frame_index:03d}.json').write_text(json.dumps(observation_evidence,indent=2))
        report['status']='fitting_observed_parts';report['pending_frame']=observation_evidence;save()
        depth=scene['position'][:,2].numpy().reshape(H,W)
        candidates=[]; deferred=[]; removal=np.zeros((H,W),bool)
        for part_id,mask in partition['eligible'].items():
            try:
                # Strong overlap rejects a body whose source support is mostly unknown.
                original=masks_by_id[part_id]
                if not mask.any():raise ValueError('No eligible pixels after protected/ambiguous mask subtraction')
                if mask.sum()<.7*original.sum():raise ValueError('Protected overlap obscures body support')
                fit=fit_observed_placement(position,mask,depth,focal=focal,principal=(W/2,H/2))
                candidates.append(dict(part_id=part_id,fit=fit,mask=mask))
            except ValueError as error:deferred.append(dict(part_id=part_id,reason=str(error)))
        placement_seconds=clock()-t;t=clock()
        fits=[];objects=[];coverage_diagnostics=[]
        for row in candidates:
            fit=row['fit'];scale=fit['scale']
            obj=dict(position=torch.from_numpy(position*scale+np.array(fit['translation'],dtype='float32')),
                covariance=torch.from_numpy(np.tile(np.eye(3,dtype='float32')[None]*(scale*.010)**2,(len(position),1,1))),
                colour=torch.from_numpy(colour),opacity=torch.full((len(position),),.9))
            _,alpha=render(*(obj[key].cuda() for key in FIELDS),torch.zeros(3,device='cuda'),
                torch.tensor([0.,0.,1.],device='cuda'),height=H,width=W,fov=FOV,radius=2,ground=False)
            coverage=evaluate_replacement_coverage(alpha.cpu().numpy(),row['mask'],protected|partition['ambiguous'])
            coverage_diagnostics.append(dict(part_id=row['part_id'],**{key:value for key,value in coverage.items() if key!='remove_mask'}))
            if coverage['accepted_for_insertion']:
                fits.append(dict(part_id=row['part_id'],fit=fit));objects.append(obj);removal|=coverage['remove_mask']
            else:deferred.append(dict(part_id=row['part_id'],reason='; '.join(coverage['rejection_reasons'])))
        coverage_seconds=clock()-t;t=clock()
        if gain is None and fits:
            all_targets=np.zeros((H,W),bool)
            for record in records:all_targets|=record['mask']
            ring=cv2.dilate(removal.astype('uint8'),np.ones((15,15),np.uint8)).astype(bool)&~all_targets
            if ring.any():colour,gain=fit_bounded_exposure(colour,image[ring].astype('float32')/255)
            else:gain=1.
            report['apple_display_exposure_gain']=gain
            np.savez_compressed(out/'canonical_display_colour.npz',ids=canonical_ids,colour=colour,
                gain=gain,source_sha256=report['canonical_asset_sha256'])
            report['canonical_display_colour_sha256']=digest(out/'canonical_display_colour.npz')
        keep=~torch.from_numpy(removal.reshape(-1))
        edited={key:scene[key][keep].clone() for key in FIELDS}
        for obj in objects:
            obj['colour']=torch.from_numpy(colour)
            edited={key:torch.cat((edited[key],obj[key])) for key in FIELDS}
        assembly_seconds=clock()-t;t=clock()
        np.savez_compressed(out/'deltas'/f'{frame_index:03d}.npz',removed_ids=np.flatnonzero(removal.reshape(-1)),
            canonical_ids=canonical_ids,protected_mask=protected,source_sha256=manifest[f'{frame_index:03d}.npz'],
            part_ids=np.array([row['part_id'] for row in fits],dtype='U80'),
            scales=np.array([row['fit']['scale'] for row in fits],dtype=np.float64),
            translations=np.array([row['fit']['translation'] for row in fits],dtype=np.float64).reshape(-1,3))
        persistence_seconds=clock()-t;t=clock()
        pic=raster(edited);raster_seconds=clock()-t;t=clock()
        frames.append((pic.cpu().numpy().clip(0,1)*255).round().astype('uint8'));sources.append(image)
        overlay=image.copy();overlay[protected]=(overlay[protected]*.5+np.array([40,130,255])*.5).astype('uint8')
        overlay[removal]=(overlay[removal]*.4+np.array([255,0,255])*.6).astype('uint8');overlays.append(overlay)
        readback_seconds=clock()-t
        report['frames'].append(dict(frame=frame_index,instances=[{k:v for k,v in row.items() if k!='mask'} for row in records],
            fits=fits,deferred=deferred,coverage_diagnostics=coverage_diagnostics,
            lost_track_ids=lost_track_ids,track_initialization_complete=initialized,
            track_observations=observation_evidence['track_observations'],instance_pixel_counts=partition['instance_pixel_counts'],
            unmatched_whole_instances=len(whole)-len(matches),duplicate_whole_masks=duplicate_count,
            unsupported_parts_preserved=sum(row['kind']=='unsupported_piece' for row in records),
            removed_pixels=int(removal.sum()),source_gaussians_outside_removal_exact=all(torch.equal(scene[key][keep],edited[key][:int(keep.sum())]) for key in FIELDS),
            association_verified=False,stages=dict(source_load_seconds=load_seconds,semantic_seconds=semantic_seconds,
                instance_persistence_seconds=instance_persistence_seconds,
                mask_tracking_seconds=tracking_seconds,placement_seconds=placement_seconds,
                alpha_coverage_render_seconds=coverage_seconds,scene_assembly_seconds=assembly_seconds,persistence_seconds=persistence_seconds,
                rasterization_seconds=raster_seconds,readback_seconds=readback_seconds),wall_seconds=clock()-frame_clock))
        previous_tracks=tracks;previous_gray=gray;report.pop('pending_frame',None);report['status']='rendering';save()
        if frame_index==4 and not any(row['fits'] for row in report['frames']):
            report['early_stop']='First five observed frames had no safe insertion. Preserved source; no full-length result claimed.'
            break
    t=clock()
    fps=40/(121/24)
    imageio.mimsave(out/'gaussian_apple.mp4',frames,fps=fps,codec='libx264',macro_block_size=1)
    imageio.mimsave(out/'semantic_masks.mp4',overlays,fps=fps,codec='libx264',macro_block_size=1)
    review_indices=sorted({0,len(frames)//2,len(frames)-1}|{index for index in (10,20,30,39) if index<len(frames)})
    for index in review_indices:Image.fromarray(np.concatenate((sources[index],overlays[index],frames[index]),axis=1)).save(out/f'comparison_{index:03d}.png')
    report['stages']['encoding_seconds']=clock()-t
    report.update(status='blocked_no_safe_insertion' if 'early_stop' in report else 'completed_partial_review_required',wall_seconds=clock()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),output_sha256=digest(out/'gaussian_apple.mp4'),
        output_frames=len(frames),output_fps=fps,output_seconds=len(frames)/fps,review_frame_indices=review_indices,
        replaced_frames=sum(bool(row['fits']) for row in report['frames']),
        unresolved='Separate replacement assets for apple slices/red peel/mouth-held bite and verified source contact are absent.')
    save()


def main():
    with exclusive_gpu():
        import torch
        with torch.inference_mode():_run()


if __name__=='__main__':main()
