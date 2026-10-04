"""Animated inspection of actual fitted Gaussian states, not invented vector art."""
import argparse
import json
import math
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .checkpoint_io import load_verified,digest,keep_windows_awake
from .replay_fit import temporal_basis_at,physical_fields,expand_coefficients
from .representation import decode_state,render_fields

BG=(15,22,31); TEXT=(235,242,250); MUTED=(170,185,201)
U=(39,222,241); V=(255,115,197); SELECTED=(255,214,101)


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',size)


def arrow(draw,start,end,color,width=2):
    draw.line([tuple(start),tuple(end)],fill=color,width=width)
    delta=np.asarray(end)-start; length=np.linalg.norm(delta)
    if length<1e-5: return
    d=delta/length; normal=np.array([-d[1],d[0]])
    tip=np.asarray(end)
    draw.polygon([tuple(tip),tuple(tip-6*d+3*normal),tuple(tip-6*d-3*normal)],fill=color)


def ellipse(draw,p,Q,sigma,multiplier=1.,origin=(0,0),color=(190,205,220),width=1):
    angles=np.linspace(0,2*np.pi,65)
    points=p[None]+2*((np.stack((np.cos(angles),np.sin(angles)),1)*sigma)@Q.T)
    points=points*multiplier+np.asarray(origin)
    draw.line([tuple(point) for point in points],fill=color,width=width)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coefficients',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--label',default='Gaussian coefficient replay')
    args=parser.parse_args()
    torch.set_num_threads(2)
    packet=load_verified(args.coefficients)
    if packet.get('position_limit') is None: parser.error('This close-up uses the anchored representation')
    args.out.mkdir(parents=True,exist_ok=False)
    device='cuda'; coefficients=packet['coefficients'].float().to(device)
    S=packet['size']; g=packet['grid']; T=packet['frames']; K=packet['terms']
    row=int(0.625*g); col=g//2; selected=row*g+col
    roi_side=S//4; roi_left=S//2-roi_side//2; roi_top=int(0.625*S)-roi_side//2
    roi=(roi_left,roi_top,roi_left+roi_side,roi_top+roi_side)
    count=96; times=torch.linspace(0,T-1,count,device=device)
    bases=temporal_basis_at(T,K,times,device)
    derivatives=temporal_basis_at(T,K,times,device,derivative=True)
    records=[]; max_orthogonality=0.
    writer=imageio.get_writer(args.out/'vectors_live.mp4',fps=12,codec='libx264',quality=8,macro_block_size=1)
    try:
        for index,t in enumerate(times):
            basis=bases[index:index+1]
            fields=physical_fields(coefficients,basis,packet['position_limit'])[None]
            video=render_fields(fields,size=S,radius=packet['radius'],min_log_scale=-2.)[0,0]
            state=decode_state(fields[:,:,0],min_log_scale=-2.)
            p=((state['p'][0,:,:2]+0.5)*(S/64)-0.5).cpu().numpy()
            Q=state['R'][0,:,:2,:2].cpu().numpy()
            sigma=(state['scale'][0,:,:2]*(S/64)).cpu().numpy()
            color=state['color'][0].cpu().numpy()
            raw=expand_coefficients(coefficients,basis)[:2,0].flatten(1).T
            raw_derivative=expand_coefficients(coefficients,derivatives[index:index+1])[:2,0].flatten(1).T
            velocity=(packet['fps']*(S/4)*packet['position_limit']*(1-raw.tanh().square())*raw_derivative).cpu().numpy()
            orth=float(np.max(np.abs(Q.transpose(0,2,1)@Q-np.eye(2))))
            max_orthogonality=max(max_orthogonality,orth)
            rgb=(video.clamp(0,1)*255).round().byte().permute(1,2,0).cpu().numpy()
            source=Image.fromarray(rgb)
            canvas=Image.new('RGB',(1280,800),BG); draw=ImageDraw.Draw(canvas)
            draw.text((24,14),'INSIDE THE GAUSSIAN VIDEO',font=font(30),fill=TEXT)
            draw.text((24,55),args.label+' | saved coefficients only | reconstruction, NOT noise generation',font=font(18),fill=MUTED)
            draw.text((24,91),'Reconstructed view',font=font(22),fill=TEXT)
            draw.text((432,91),'4x crop: real dots and axes',font=font(22),fill=TEXT)
            draw.text((848,91),f'One indexed dot: #{selected}',font=font(22),fill=TEXT)
            top=128; panel=384
            canvas.paste(source.resize((panel,panel),Image.Resampling.BILINEAR),(24,top))
            full_scale=panel/S
            draw.rectangle((24+roi[0]*full_scale,top+roi[1]*full_scale,24+roi[2]*full_scale,top+roi[3]*full_scale),outline=SELECTED,width=3)
            canvas.paste(source.crop(roi).resize((panel,panel),Image.Resampling.NEAREST),(432,top))
            scale=panel/roi_side
            origin=np.array([432-roi_left*scale,top-roi_top*scale])
            crop_overlay=Image.new('RGBA',(panel,panel),(0,0,0,0)); marks=ImageDraw.Draw(crop_overlay)
            crop_origin=np.array([-roi_left*scale,-roi_top*scale])
            stride=max(1,g//32)
            for i in range(g*g):
                rr,cc=divmod(i,g)
                if i!=selected and (rr%stride or cc%stride): continue
                if not (roi[0]<=p[i,0]<roi[2] and roi[1]<=p[i,1]<roi[3]): continue
                location=p[i]*scale+crop_origin
                ellipse(marks,p[i],Q[i],sigma[i],scale,crop_origin,color=(*SELECTED,220) if i==selected else (205,218,231,140),width=2 if i==selected else 1)
                length=14 if i!=selected else 23
                arrow(marks,location,location+Q[i,:,0]*length,U,2)
                arrow(marks,location,location+Q[i,:,1]*length,V,2)
                marks.ellipse((location[0]-2,location[1]-2,location[0]+2,location[1]+2),fill=SELECTED if i==selected else TEXT)
            canvas.paste(crop_overlay,(432,top),crop_overlay)
            draw=ImageDraw.Draw(canvas)
            # Show the selected dot's actual covariance kernel at a labelled display scale.
            q=Q[selected]; sig=sigma[selected]; magnification=66/max(sig)
            center=np.array([1042.,306.]); xy=np.stack(np.meshgrid(np.arange(380),np.arange(354)),axis=-1)
            delta=(xy-np.array([190.,178.]))/magnification
            local=delta@q
            weight=np.exp(-0.5*((local/sig)**2).sum(-1))[...,None]
            heat=(np.asarray(BG)+(color[selected]*255-np.asarray(BG))*weight).clip(0,255).astype(np.uint8)
            canvas.paste(Image.fromarray(heat),(852,128))
            draw=ImageDraw.Draw(canvas)
            ellipse(draw,np.zeros(2),q,sig,magnification,center,color=SELECTED,width=2)
            arrow(draw,center,center+q[:,0]*96,U,4); arrow(draw,center,center+q[:,1]*96,V,4)
            draw.ellipse((center[0]-4,center[1]-4,center[0]+4,center[1]+4),fill=TEXT)
            draw.text((858,480),'Isolated footprint, enlarged',font=font(19),fill=MUTED)
            draw.text((24,536),f'Source time: {float(t)/packet["fps"]:.3f} s   |   fitted frame coordinate {float(t):.2f}',font=font(22),fill=TEXT)
            draw.text((24,575),f'{g*g:,} Gaussian dots/frame | {K} temporal terms | {S} x {S} pixels',font=font(20),fill=MUTED)
            arrow(draw,np.array([28,629]),np.array([66,629]),U,3)
            draw.text((78,611),'u: first unit axis',font=font(21),fill=TEXT)
            arrow(draw,np.array([332,629]),np.array([370,629]),V,3)
            draw.text((382,611),'v: perpendicular unit axis',font=font(21),fill=TEXT)
            draw.text((24,661),'Ellipse = 2 sigma. Arrow lengths enlarged for clarity; they are NOT forces.',font=font(19),fill=MUTED)
            draw.text((24,697),'Slow-motion cosine interpolation between fitted frames; no new source frames.',font=font(19),fill=MUTED)
            draw.text((24,733),'Spatial anchors can change colour as objects pass. Image y increases downward.',font=font(18),fill=MUTED)
            values=[f'centre = ({p[selected,0]:.2f}, {p[selected,1]:.2f}) px',
                    f'scales = ({sig[0]:.3f}, {sig[1]:.3f}) px',
                    f'||u|| = {np.linalg.norm(q[:,0]):.6f}',
                    f'||v|| = {np.linalg.norm(q[:,1]):.6f}',
                    f'u dot v = {np.dot(q[:,0],q[:,1]):.2e}',
                    f'det(Q) = {np.linalg.det(q):.6f}',
                    f'fitted centre speed = {np.linalg.norm(velocity[selected]):.2f} px/s']
            for line,text in enumerate(values): draw.text((848,532+line*31),text,font=font(19),fill=TEXT)
            draw.line((24,787,1256,787),fill=(53,67,83),width=4)
            draw.line((24,787,24+1232*index/(count-1),787),fill=SELECTED,width=4)
            writer.append_data(np.asarray(canvas))
            if index in (0,count//2,count-1): canvas.save(args.out/f'frame_{index:03d}.png')
            records.append(dict(frame=index,source_frame=float(t),source_seconds=float(t)/packet['fps'],
                                dot_index=selected,position=p[selected].tolist(),unit_axes=q.tolist(),
                                sigma_pixels=sig.tolist(),fitted_velocity_pixels_per_second=velocity[selected].tolist()))
    finally: writer.close()
    report=dict(coefficients_sha256=digest(args.coefficients),input='Saved Gaussian coefficients only',
                type='Diagnostic overlays on source-conditioned reconstruction',frames=count,fps=12,
                unit_arrow_display='Fixed length for legibility, not a magnitude or force vector',
                ellipse='Actual covariance 2-sigma contour; ideal ellipse, finite renderer support still applies',
                time='Fractional cosine-basis interpolation, slowed playback; not extra observed frames',
                dot_identity=packet['dot_identity'],max_orthogonality_error=max_orthogonality,states=records)
    (args.out/'vector_states.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.out),frames=count,max_orthogonality_error=max_orthogonality)),flush=True)


if __name__=='__main__':
    with keep_windows_awake(): main()
