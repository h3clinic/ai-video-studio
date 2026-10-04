"""Explicit cat-specific thin-protrusion cleanup; not learned anatomy repair."""
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import marching_cubes
import torch
import trimesh
from .checkpoint_io import load_verified,save_inference_checkpoint,digest

BASE=Path('artifacts/real_video/hunyuan_gaussian/v4_full')
OUT=BASE.parent/'v6_clean_shape'
RADIUS=3


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    a=load_verified(BASE/'shape.pt');mesh=trimesh.Trimesh(a['vertices'].numpy(),a['faces'].numpy(),process=True)
    pitch=.005
    grid=mesh.voxelized(pitch).fill(); volume=np.pad(grid.matrix,4)
    if RADIUS==3:
        axis=np.arange(-3,4);ball=sum(c*c for c in np.meshgrid(axis,axis,axis,indexing='ij'))<=9
        opened=ndimage.binary_opening(volume,structure=ball)
    else:
        eroded=ndimage.distance_transform_edt(volume)>RADIUS
        opened=ndimage.distance_transform_edt(~eroded)<=RADIUS
    # This asset faces +X. Restrict removal to upper head, preserving torso/legs.
    xyz=np.indices(volume.shape).astype(np.float32)
    world_x=(xyz[0]-4)*pitch+grid.transform[0,3];world_y=(xyz[1]-4)*pitch+grid.transform[1,3]
    bounds=mesh.bounds;head=(world_x>bounds[0,0]+.76*np.ptp(bounds,axis=0)[0])&(world_y>bounds[0,1]+.52*np.ptp(bounds,axis=0)[1])
    if RADIUS>3:head&=world_y<bounds[0,1]+.81*np.ptp(bounds,axis=0)[1]
    cleaned=np.where(head,opened,volume)
    vertices,faces,_,_=marching_cubes(cleaned.astype(np.float32),.5,spacing=(pitch,pitch,pitch))
    vertices+=grid.transform[:3,3]-4*pitch
    result=trimesh.Trimesh(vertices,faces,process=True);result.fix_normals()
    result.export(OUT/'shape.ply')
    save_inference_checkpoint(dict(vertices=torch.tensor(result.vertices,dtype=torch.float32),faces=torch.tensor(result.faces,dtype=torch.long),
        scope='Full Hunyuan inferred shape plus cat-specific voxel opening in upper-head region; not learned geometry refinement.'),OUT/'shape.pt')
    for name in ['conditioning.png','reference.png','mask.png']:Image.open(BASE/name).save(OUT/name)
    report=dict(source_sha256=digest(BASE/'shape.pt'),pitch=pitch,opening_radius_voxels=RADIUS,removed_head_voxels=int((volume&~cleaned).sum()),
        occupied_voxels=int(volume.sum()),watertight=bool(result.is_watertight),accepted=False,
        limitations=['Global surface revoxelized at 0.005 scene units; detail loss possible.','Localized morphological prior may remove desirable details or alter facial anatomy.','This is a diagnostic engineering cleanup, not model training or verified anatomy.'])
    (OUT/'repair_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--radius',type=int,default=3);parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args();RADIUS=args.radius;OUT=args.out
    if not 1<=RADIUS<=10:raise ValueError('Opening radius outside diagnostic bounds')
    main()
