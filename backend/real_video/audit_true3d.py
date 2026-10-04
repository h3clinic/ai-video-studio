"""Audit saved actual-3D diagnostic assets, motion packets and video decodes."""
import json
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
import imageio.v2 as imageio
from .checkpoint_io import load_verified,digest
from .demo_true3d import ROOT,covariance
from .gaussian3d import rotation,skin


def main():
    torch.set_num_threads(4)
    asset=load_verified(ROOT/'cat_asset.pt'); rig=load_verified(ROOT/'rig.pt')
    p=asset['position']; original=p.clone(); frame=asset['frame']; eye=torch.eye(3)
    assert len(asset['ids'].unique())==len(p)==45000
    assert torch.allclose(frame.transpose(-1,-2)@frame,eye.expand_as(frame),atol=1e-5)
    assert torch.all(torch.det(frame)>.9999)
    assert torch.allclose(rig['weight'].sum(1),torch.ones(len(p)),atol=1e-6)
    assert torch.all(rig['weight']>=0)
    bary=asset['barycentric']; verts=asset['mesh_vertices'][asset['mesh_faces'][asset['face_id'].long()].long()]
    assert torch.allclose((verts*bary[...,None]).sum(1),p,atol=2e-6)
    results={}; cov=covariance(asset)
    for mode in ['orbit','rig_test']:
        path=ROOT/f'cat_3d_{mode}_7s.mp4'; reader=imageio.get_reader(path)
        meta=reader.get_meta_data(); hashes=set(); n=0; previous=None; diffs=[]
        import hashlib
        for pic in reader:
            n+=1; hashes.add(hashlib.sha256(pic.tobytes()).hexdigest())
            if previous is not None: diffs.append(float(np.abs(pic.astype(np.float32)-previous).mean()))
            previous=pic.astype(np.float32)
        reader.close(); assert n==168 and meta['fps']==24
        controls=load_verified(ROOT/f'{mode}_controls.pt')
        assert len(controls['angles'])==n and len(controls['camera'])==n
        results[mode]=dict(decoded_frames=n,fps=meta['fps'],duration=n/meta['fps'],unique_decoded_frames=len(hashes),
                           mean_adjacent_pixel_difference=float(np.mean(diffs)),video_bytes=path.stat().st_size)
        if mode=='rig_test':
            min_eig=float('inf'); max_extent=0.
            for angles in controls['angles']:
                local=rotation(torch.tensor([0.,0.,1.]).expand(len(angles),3),angles)
                moved,co,_=skin(p,cov,rig['joints'],rig['parents'],local,rig['index'].long(),rig['weight'])
                min_eig=min(min_eig,float(torch.linalg.eigvalsh(co).min()))
                max_extent=max(max_extent,float((moved.amax(0)-moved.amin(0)).max()))
            assert min_eig>0
            results[mode].update(min_covariance_eigenvalue=min_eig,max_xyz_extent=max_extent)
        results[mode]['control_packet_bytes']=(ROOT/f'{mode}_controls.pt').stat().st_size
    assert torch.equal(p,original)
    results['asset_bytes']=(ROOT/'cat_asset.pt').stat().st_size
    results['rig_bytes']=(ROOT/'rig.pt').stat().st_size
    results['asset_sha256']=digest(ROOT/'cat_asset.pt')
    results['provenance']={str(path):digest(path) for path in [Path('real_video/gaussian3d.py'),Path('real_video/infer_3d_cat.py'),Path('real_video/demo_true3d.py')]}
    results['packages']={name:importlib.metadata.version(name) for name in ['torch','transformers','omegaconf','einops','trimesh','scikit-image']}
    results['meaning']='File sizes are not runtime memory or matched-quality savings. Saved appearance/IDs persist; controls are scripted, not learned. Positive covariance and fixed bones do not prove valid anatomy.'
    (ROOT/'audit.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))


if __name__=='__main__': main()
