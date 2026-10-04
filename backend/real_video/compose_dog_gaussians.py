"""Gaussian-only replay/composition with honest full-bank memory accounting."""
import gzip
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import psutil
import torch
from real_video.checkpoint_io import load_verified, keep_windows_awake, digest
from real_video.edit_gaussian_memory import splat_layer
from real_video.dog_scene_experiment import Measure, OUT, SOURCE


def tensor_bytes(value):
    if isinstance(value,torch.Tensor): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,list): return sum(tensor_bytes(v) for v in value)
    return 0


class GaussianReplay:
    def __init__(self,packet):
        self.packet=packet
    def at(self,index,device='cuda'):
        if index<0 or index>=self.packet['frames']: raise IndexError(index)
        segment=max((s for s in self.packet['segments'] if s['first_frame']<=index),key=lambda s:s['first_frame'])
        state=segment['states'][index-segment['first_frame']]
        points=segment['base'].to(device=device,dtype=torch.float32).clone()
        delta=state['delta'].to(device=device,dtype=torch.float32)
        points[:,:2]+=delta[:,:2]; points[:,2:4]+=delta[:,2:4]
        c,s=delta[:,4].cos(),delta[:,4].sin(); u=points[:,4:6].clone()
        points[:,4:6]=torch.stack((c*u[:,0]-s*u[:,1],s*u[:,0]+c*u[:,1]),1)
        points[:,9]=state['visibility'].to(device=device,dtype=torch.float32)/255
        return points


def dog_transform(points,base,scale=.48):
    # Explicit screen-space edit, not inferred physical scale or a 3D placement.
    lower=base[:,:2].amin(0).to(points); upper=base[:,:2].amax(0).to(points)
    pivot=torch.stack(((lower[0]+upper[0])*.5,upper[1]))
    target=points.new_tensor([650/480,447/480])
    result=points.clone(); result[:,:2]=(points[:,:2]-pivot)*scale+target
    result[:,2:4]+=math.log(scale)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dog-capture',default='dog_capture')
    parser.add_argument('--output-name',default='composition')
    args=parser.parse_args()
    target=OUT/args.output_name; target.mkdir(exist_ok=True)
    if (target/'metrics.json').exists(): raise FileExistsError('Existing result retained')
    torch.set_num_threads(6)
    report=dict(scope='Planar Gaussian reconstruction + independent recorded dog motion; no joint interaction generation',
                quality_accepted=False,dog_screen_scale=.48,dog_target_bottom_xy=[650,447],
                stages={},storage={},limitations=['Not 3D environment memory','All future motion states stored; banks refresh every 8 frames','RGB outputs and Gaussian replay are not quality-matched until reviewed'])
    with keep_windows_awake(),torch.inference_mode():
        with Measure() as m:
            cat_packet=load_verified(OUT/'cat_capture/gaussian_memory.pt')
            dog_packet=load_verified(OUT/args.dog_capture/'gaussian_memory.pt')
            cat=GaussianReplay(cat_packet); dog=GaussianReplay(dog_packet)
        report['stages']['load_gaussian_banks']=m.result
        report['storage']['all_cpu_gaussian_tensor_bytes']=tensor_bytes(cat_packet)+tensor_bytes(dog_packet)
        report['storage']['cat_bank_file_bytes']=(OUT/'cat_capture/gaussian_memory.pt').stat().st_size
        report['storage']['dog_bank_file_bytes']=(OUT/args.dog_capture/'gaussian_memory.pt').stat().st_size
        report['dog_capture_directory']=args.dog_capture
        for name in ['cat','dog']:
            path=OUT/('cat_capture' if name=='cat' else args.dog_capture)/'gaussian_memory.pt'; zipped=path.with_suffix('.pt.gz')
            compression_start=time.perf_counter()
            if not zipped.exists():
                with path.open('rb') as src,gzip.open(zipped,'wb',compresslevel=6) as dst: shutil.copyfileobj(src,dst)
            report['stages'][name+'_lossless_compression_seconds']=time.perf_counter()-compression_start
            report['storage'][f'{name}_lossless_gzip_bytes']=zipped.stat().st_size
        # Warm renderer and synchronize before measured replay. No RGB input here.
        points=cat.at(0); splat_layer(points,height=480,width=832,radius=3); del points
        frames=[]; cat_frames=[]; alphas=[]; live_state=[]
        with Measure() as m:
            for i in range(33):
                a=cat.at(i); b=dog_transform(dog.at(i),dog_packet['segments'][0]['base'])
                rgb,_=splat_layer(a,height=480,width=832,radius=3)
                dog_rgb,alpha=splat_layer(b,height=480,width=832,radius=3)
                composite=dog_rgb*alpha+rgb*(1-alpha)
                live_state.append(a.numel()*a.element_size()+b.numel()*b.element_size())
                frames.append((composite[0].permute(1,2,0).clamp(0,1)*255).round().byte().cpu().numpy())
                cat_frames.append((rgb[0].permute(1,2,0).clamp(0,1)*255).round().byte().cpu().numpy())
                alphas.append(alpha[0,0].cpu().numpy())
                del a,b,rgb,dog_rgb,alpha,composite
        report['stages']['replay_with_output_copy']=m.result
        report['storage']['maximum_live_gaussian_state_bytes']=max(live_state)
        report['storage']['retained_validation_rgb_bytes']=sum(f.nbytes for f in frames+cat_frames)
        report['storage']['retained_validation_alpha_bytes']=sum(f.nbytes for f in alphas)
        export_start=time.perf_counter()
        imageio.mimwrite(target/'gaussian_cat_plus_dog.mp4',frames,fps=16,codec='libx264',quality=8,macro_block_size=1)
        report['stages']['mp4_export_seconds']=time.perf_counter()-export_start
        for i in [0,7,15,16,23,31,32]:
            Image.fromarray(frames[i]).save(target/f'frame_{i:03d}.png')
            Image.fromarray(cat_frames[i]).save(target/f'cat_replay_{i:03d}.png')
        # Validation phase only: read source RGB after Gaussian-only rendering ends.
        cap=cv2.VideoCapture(str(SOURCE)); errors=[]
        for i in range(33):
            ok,bgr=cap.read()
            if not ok: raise ValueError('Source changed')
            original=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
            errors.append(float(np.square(cat_frames[i].astype(np.float32)/255-original).mean()))
        cap.release()
        report['reconstruction_validation']=dict(cat_replay_psnr_db=-10*math.log10(np.mean(errors)),per_frame_mse=errors,
                                                 meaning='Reconstruction agreement with exact source, not generative quality')
        # Fair replay comparison: an already generated RGB MP4 also needs no Wan.
        for name,path in [('A_encoded_video',target/'gaussian_cat_plus_dog.mp4'),('B_encoded_video',OUT/'joint/video.mp4')]:
            durations=[]
            for repeat in range(3):
                start=time.perf_counter(); cap=cv2.VideoCapture(str(path)); count=0
                while True:
                    ok,frame=cap.read()
                    if not ok: break
                    if name=='B_encoded_video' and repeat==0 and count in [7,15,23,31]:
                        Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(OUT/f'joint/frame_{count:03d}.png')
                    count+=1
                cap.release(); durations.append(time.perf_counter()-start)
            report['stages'][name+'_cpu_decode']=dict(seconds_each=durations,decoded_frames=count,
                retained_rgb_frame_bytes=480*832*3,no_diffusion=True,no_cuda_decode=True)
            report['storage'][name+'_file_bytes']=path.stat().st_size
        report['source_cat_sha256']=digest(SOURCE)
        sheet=Image.new('RGB',(832*4,480*2))
        for col,i in enumerate([0,7,23,32]):
            sheet.paste(Image.fromarray(frames[i]),(col*832,0))
            sheet.paste(Image.open(OUT/f'joint/frame_{i:03d}.png'),(col*832,480))
        sheet.save(target/'A_vs_B.jpg')
        del frames,cat_frames,alphas,sheet
        import gc
        gc.collect(); torch.cuda.empty_cache()
        with Measure() as m:
            for i in range(33):
                a=cat.at(i); b=dog_transform(dog.at(i),dog_packet['segments'][0]['base'])
                rgb,_=splat_layer(a,height=480,width=832,radius=3)
                dog_rgb,alpha=splat_layer(b,height=480,width=832,radius=3)
                output=((dog_rgb*alpha+rgb*(1-alpha))[0].permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy()
                del a,b,rgb,dog_rgb,alpha,output
        report['stages']['streaming_replay_without_rgb_history']=m.result
        report['thermal_caveat']='During joint Wan inference GPU was 87C and NVIDIA reported active software thermal slowdown. Sequential wall times do not establish an algorithmic speedup.'
        (target/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__=='__main__': main()
