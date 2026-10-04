"""Five-case fixed-weight attachment intervention; forecast and reference separate."""
import argparse
import builtins
import json
import time
from pathlib import Path
from unittest.mock import patch
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image,ImageDraw
from .connected_gaussian import build_surface,ConnectedSurface,deform_surface
from .animal_control_graph import AnimalControlGraph,bind_points,deform_points
from .demo_animal_graph import SEED_ROOT,WORK,HEIGHT,VAL
from .dense_gaussian_seed import FixedGeometrySplat,SOURCE
from .edit_gaussian_memory import array_image,splat_layer
from .gaussian_motion_memory import select_cat
from .checkpoint_io import load_verified,save_inference_checkpoint,digest,keep_windows_awake

MODEL=Path('artifacts/real_video/animal_graph/visual_v2/model.pt')
OUT=Path('artifacts/real_video/connected_surface/v3')
NAMES=['cat',*VAL]
MODES=['independent','fixed_opacity','connected','frozen']


def tensor_bytes(packet):
    return sum(v.numel()*v.element_size() for v in packet.values() if isinstance(v,torch.Tensor))


@torch.no_grad()
def predict():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists(): raise FileExistsError('Preserve experiment')
    protocol=dict(spacing_normalized=1/60,stiffness=4,iterations=8,guard='local with rigid root-motion fallback',min_determinant=.2,min_stretch=.45,max_stretch=2.2,
                  cases=NAMES,checkpoint=str(MODEL),model_sha256=digest(MODEL),weights_changed=False,
                  scope='Fixed-weight causal planar decoder intervention. Not recovered anatomical articulation or text-to-video.',
                  sweep_sources=['https://arxiv.org/html/2402.04796v1','https://arxiv.org/html/2312.14937v1'],
                  decision='Parameters fixed before outputs. Report every case, all guard slowdowns and foreground metrics.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    allowed={MODEL.resolve(),*((SEED_ROOT/f'{name}.pt').resolve() for name in NAMES)}
    original_load=torch.load; imported=builtins.__import__; reads=[]
    def guard(path,*args,**kwargs):
        if Path(path).resolve() not in allowed: raise AssertionError('Unexpected tensor input')
        reads.append(str(path)); return original_load(path,*args,**kwargs)
    def forbidden(*args,**kwargs): raise AssertionError('No source frames or future tracks in forecast')
    def import_guard(name,*args,**kwargs):
        if name=='diffusers' or name.startswith('diffusers.'): raise AssertionError('No diffusion input')
        return imported(name,*args,**kwargs)
    report={}
    with patch('torch.load',side_effect=guard),patch('cv2.VideoCapture',side_effect=forbidden),patch('PIL.Image.open',side_effect=forbidden),patch('numpy.load',side_effect=forbidden),patch('builtins.__import__',side_effect=import_guard):
        model=AnimalControlGraph().cuda().eval(); model.load_state_dict(load_verified(MODEL)['model'])
        for name in NAMES:
            seed=load_verified(SEED_ROOT/f'{name}.pt'); ctrl={k:v[None].cuda() for k,v in seed['controls'].items()}
            points=seed['points'].cuda(); index=seed['index'].cuda(); weight=seed['weight'].cuda()
            if name=='cat': h,w=480,832; background=splat_layer(seed['background'].cuda())[0]
            else: background=seed['background'].cuda(); h,w=background.shape[-2:]
            start=time.perf_counter(); cpu=build_surface(seed['points'],protocol['spacing_normalized'])
            solver=ConnectedSurface(cpu,protocol['stiffness'],protocol['iterations'])
            asset={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
            vertex_points=points.new_zeros(len(cpu['rest']),10); vertex_points[:,:2]=asset['rest']; vertex_points[:,4]=1; vertex_points[:,9]=1
            vi,vw=bind_points(asset['rest'],ctrl['position'][0,2])
            torch.cuda.synchronize(); setup_seconds=time.perf_counter()-start
            # Save persistent attachment bank; validation is performed outside the input guard.
            results={}; stats={}; vertex_trace=[]; snapshot=None
            for mode in MODES:
                state=model.initialize(ctrl['position'],ctrl['angle'],ctrl['visibility'],ctrl['colour'],ctrl['adjacency'])
                solver.previous=solver.rest.copy(); frames=[]; masks=[]; timing=[]; movement=[]; guards=[]
                torch.cuda.reset_peak_memory_stats()
                for t in range(13):
                    torch.cuda.synchronize(); start=time.perf_counter()
                    if t and mode!='frozen': state=model.step(state)
                    moved=deform_points(points,state['reference'][0],state['position'][0],state['angle'][0]-state['reference_angle'][0],state['visibility'][0],index,weight)
                    if mode=='connected':
                        proposal=deform_points(vertex_points,state['reference'][0],state['position'][0],state['angle'][0]-state['reference_angle'][0],torch.ones_like(state['visibility'][0]),vi,vw)[:,:2].cpu().numpy()
                        vertices,quality=solver.step(proposal); guards.append(quality)
                        if t==4: snapshot=(solver.previous.copy(),{k:v.clone() for k,v in state.items()})
                        if t==8:
                            replay=ConnectedSurface(cpu,protocol['stiffness'],protocol['iterations']); replay.previous=snapshot[0].copy(); rs=snapshot[1]
                            for _ in range(4):
                                rs=model.step(rs)
                                rp=deform_points(vertex_points,rs['reference'][0],rs['position'][0],rs['angle'][0]-rs['reference_angle'][0],torch.ones_like(rs['visibility'][0]),vi,vw)[:,:2].cpu().numpy()
                                restored,_=replay.step(rp)
                            assert np.array_equal(restored,vertices)
                        moved=deform_surface(asset,torch.tensor(vertices,device='cuda',dtype=points.dtype))
                        vertex_trace.append(vertices.astype(np.float32))
                    elif mode in ['fixed_opacity','frozen']: moved[:,9]=points[:,9]
                    # Common wider stencil: upper allowed stretch requires support beyond old radius=2.
                    renderer=FixedGeometrySplat(moved,height=h,width=w,radius=4)
                    rgb,alpha=renderer.render((moved[:,6:9]+1)/2); composite=rgb*alpha+background*(1-alpha)
                    torch.cuda.synchronize(); timing.append((time.perf_counter()-start)*1000)
                    frames.append(array_image(composite)); masks.append(alpha[0,0].cpu().numpy())
                    movement.append(float((moved[:,:2]-points[:,:2]).norm(dim=1).mean()*h))
                    assert torch.equal(moved[:,6:9],points[:,6:9])
                    del renderer,rgb,alpha,composite,moved
                stats[mode]=dict(median_ms=float(np.median([timing[i] for i in range(3,13) if i!=8])),times_ms=timing,
                                 peak_cuda_bytes=torch.cuda.max_memory_allocated(),mean_displacement_pixels=movement,guards=guards)
                results[mode]=np.stack(frames); results[mode+'_alpha']=np.stack(masks)
            results['vertices']=np.stack(vertex_trace)
            np.savez_compressed(OUT/f'{name}.npz',**results)
            # Keep CPU bank for a verified export after the guarded prediction phase.
            cpu['vertex_control_index']=vi.cpu(); cpu['vertex_control_weight']=vw.cpu()
            report[name]=dict(gaussians=len(points),vertices=len(cpu['rest']),triangles=len(cpu['faces']),setup_seconds=setup_seconds,
                              stored_asset_tensor_bytes=tensor_bytes(cpu),control_state_bytes=tensor_bytes(state),
                              previous_vertex_state_bytes=solver.previous.nbytes,statistics=stats,seed_sha256=digest(SEED_ROOT/f'{name}.pt'))
            # Direct serialization is output-only; the verified loader is forbidden for new inputs here.
            torch.save(cpu,OUT/f'{name}_asset.pt')
            print(json.dumps(dict(name=name,connected_ms=stats['connected']['median_ms'],last_guard=stats['connected']['guards'][-1])),flush=True)
    for name in NAMES:
        load_verified(OUT/f'{name}_asset.pt')
        report[name]['asset_sha256']=digest(OUT/f'{name}_asset.pt')
    report['reads']=reads; report['future_inputs']=0
    report['memory_note']='Asset tensor counts exclude solver factors, model, GPU/CPU duplicate buffers, output frames and transient tensors. Not total RAM savings.'
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))


def reference(name,seed):
    frames=[]; masks=[]
    if name=='cat':
        cap=cv2.VideoCapture(str(SOURCE))
        try:
            for i in range(15):
                ok,bgr=cap.read()
                if not ok: raise ValueError('Missing reference')
                if i>=2:
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); frames.append(rgb); masks.append(select_cat(rgb))
        finally: cap.release()
        with torch.no_grad(): bg=array_image(splat_layer(seed['background'].cuda())[0]).astype(np.float32)/255
    else:
        root=WORK/'extracted/DAVIS'; paths=sorted((root/'JPEGImages/480p'/name).glob('*.jpg'))
        for i in range(4,29,2):
            rgb=np.array(Image.open(paths[i]).convert('RGB')); width=round(rgb.shape[1]*HEIGHT/rgb.shape[0]); frames.append(cv2.resize(rgb,(width,HEIGHT),interpolation=cv2.INTER_AREA))
            mask=np.array(Image.open(root/'Annotations/480p'/name/(paths[i].stem+'.png')))==seed['object']
            masks.append(cv2.resize(mask.astype(np.float32),(width,HEIGHT),interpolation=cv2.INTER_NEAREST))
        bg=seed['background'][0].permute(1,2,0).numpy()
    masks=np.stack(masks); truth=np.stack(frames).astype(np.float32)/255*masks[...,None]+bg*(1-masks[...,None])
    return truth,masks>.5


def evaluate():
    folder=OUT/'evaluation'; folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists(): raise FileExistsError('Preserve evaluation')
    report={}
    for name in NAMES:
        seed=load_verified(SEED_ROOT/f'{name}.pt'); truth,gt=reference(name,seed)
        with np.load(OUT/f'{name}.npz') as data: forecast={k:data[k] for k in data.files}
        asset=load_verified(OUT/f'{name}_asset.pt'); scores={}
        for mode in MODES:
            output=forecast[mode].astype(np.float32)/255; pred=forecast[mode+'_alpha']>.5; union=pred|gt
            error=(output-truth)**2
            regional=(error.sum(-1)*union).sum((1,2))/(3*np.maximum(union.sum((1,2)),1))
            scores[mode]=dict(future12_mse=float(error[1:].mean()),mask_iou=float(((pred&gt).sum((1,2))/np.maximum(union.sum((1,2)),1))[1:].mean()),
                              foreground_union_mse=float(regional[1:].mean()))
        report[name]=scores
        writer=imageio.get_writer(folder/f'{name}_comparison.mp4',fps=8,codec='libx264',quality=8,macro_block_size=1)
        try:
            for t in range(13):
                canvas=Image.new('RGB',(832,600),'#141820'); draw=ImageDraw.Draw(canvas)
                draw.text((8,6),f'{name.upper()} | fixed Gaussian attachments | same learned motion weights',fill='white')
                draw.text((8,22),'Planar forecast, NOT recovered anatomy. Reference is evaluation-only.',fill='white')
                wire=Image.fromarray(forecast['connected'][t]).copy(); pen=ImageDraw.Draw(wire)
                h,w=forecast['connected'][t].shape[:2]; vertices=forecast['vertices'][t]*h
                edges=asset['edges'].numpy()
                for a,b in edges[::max(1,len(edges)//350)]: pen.line([tuple(vertices[a]),tuple(vertices[b])],fill=(60,255,120),width=1)
                panels=[('Reference',np.uint8(np.clip(truth[t],0,1)*255)),('Previous independent bindings',forecast['independent'][t]),
                        ('NEW: connected + shape transport',forecast['connected'][t]),('NEW: actual moving support (not bones)',np.asarray(wire))]
                for j,(label,img) in enumerate(panels):
                    x,y=j%2*416,42+j//2*266; draw.text((x+5,y),label,fill='white'); canvas.paste(Image.fromarray(img).resize((416,240)),(x,y+20))
                draw.text((8,583),'DAVIS: Pont-Tuset et al. 2017. Cat wheat: Neil Oakes, CC BY-SA 2.0.',fill='white')
                writer.append_data(np.asarray(canvas))
                if t in [0,6,12]: canvas.save(folder/f'{name}_{t:02d}.jpg')
        finally: writer.close()
        cap=cv2.VideoCapture(str(folder/f'{name}_comparison.mp4')); count=0
        while cap.read()[0]: count+=1
        cap.release(); assert count==13
        report[name]['video_frames']=count
        print(json.dumps(dict(name=name,scores=scores)),flush=True)
    (folder/'results.json').write_text(json.dumps(report,indent=2))


def audit():
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    repeat=OUT.parent/'v3_repeat'; report={}
    first=json.loads((OUT/'audit.json').read_text()); second=json.loads((repeat/'audit.json').read_text())
    scores=json.loads((OUT/'evaluation/results.json').read_text())
    for name in NAMES:
        asset=load_verified(OUT/f'{name}_asset.pt'); solver=ConnectedSurface(asset)
        with np.load(OUT/f'{name}.npz') as data, np.load(repeat/f'{name}.npz') as again:
            exact=np.array_equal(data['vertices'],again['vertices']); assert exact
            quality=[solver.quality(v.astype(np.float64)) for v in data['vertices']]
            assert min(q['min_determinant'] for q in quality)>.1999
            assert min(q['min_stretch'] for q in quality)>.4499
            assert max(q['max_stretch'] for q in quality)<2.2001
            pixel_difference=int(np.abs(data['connected'].astype(np.int16)-again['connected'].astype(np.int16)).max())
        edges=asset['edges'].long().T.numpy(); n=len(asset['rest'])
        components=connected_components(sparse.coo_matrix((np.ones(edges.shape[1]),edges),shape=(n,n)),directed=False,return_labels=False)
        assert components==1
        delta=100*(scores[name]['connected']['future12_mse']/scores[name]['independent']['future12_mse']-1)
        report[name]=dict(connected_components=int(components),repeat_vertices_exact=exact,repeat_max_pixel_difference=pixel_difference,
                          min_determinant=min(q['min_determinant'] for q in quality),rgb_mse_change_percent=delta,
                          isolated_repeat_independent_ms=second[name]['statistics']['independent']['median_ms'],
                          isolated_repeat_connected_ms=second[name]['statistics']['connected']['median_ms'],
                          asset_plus_current_tensor_bytes=first[name]['stored_asset_tensor_bytes']+first[name]['control_state_bytes']+first[name]['previous_vertex_state_bytes'],
                          video_sha256=digest(OUT/'evaluation'/f'{name}_comparison.mp4'))
    report['decision']='Do not promote as a quality or efficiency win: cat/camel errors increase; correct gait not learned. Structural attachment implemented.'
    report['scope']='Five reused diagnostic windows, not independent test benchmark. Model weights unchanged. 149 unit tests passed separately.'
    report['timing']='Single warmed repeats, no concurrent project tests in second pass. Not a statistically controlled benchmark or comparison to Wan.'
    report['memory']='Counts exclude solver factorization, runtime copies, rendering buffers, model and backgrounds; not total RAM.'
    report['source_hashes']={str(p):digest(p) for p in [Path('real_video/connected_gaussian.py'),Path('real_video/demo_connected_gaussian.py'),MODEL]}
    (OUT/'final_audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=['predict','evaluate','audit']); parser.add_argument('--root',type=Path,default=OUT); args=parser.parse_args(); OUT=args.root
    torch.set_num_threads(4); cv2.setNumThreads(4)
    with keep_windows_awake(): {'predict':predict,'evaluate':evaluate,'audit':audit}[args.mode]()


if __name__=='__main__': main()
