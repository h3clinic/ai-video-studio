"""Export a verified static Gaussian checkpoint for the local orbit inspector."""
import json
import argparse
from pathlib import Path
import numpy as np
from .checkpoint_io import load_verified, digest

def main(source=Path('artifacts/real_video/background_completion/v3_texture/completed_scene.pt'), name='scene'):
    out = Path('artifacts/real_video/orbit_viewer')
    out.mkdir(exist_ok=True, parents=True)
    scene = load_verified(source)
    cov = scene['covariance'].numpy()
    packed = np.concatenate([scene['position'].numpy(), cov[:,0,:], cov[:,1,1:3], cov[:,2,2:3], scene['colour'].numpy(), scene['opacity'].numpy()[:,None]], axis=1).astype('<f4')
    assert packed.shape[1] == 13 and np.isfinite(packed).all()
    packed.tofile(out/f'{name}.bin')
    (out/f'{name}.json').write_text(json.dumps(dict(count=len(packed), original_count=scene['original_count'], source_sha256=digest(source), binary_sha256=digest(out/f'{name}.bin'), pivot=scene.get('camera_pivot',[0,0,float(np.median(packed[:scene['original_count'],2]))]), distance=scene.get('camera_distance'), scope=scene.get('scope','Static incomplete inferred geometry'))))
    print(json.dumps(dict(count=len(packed), bytes=packed.nbytes, output=str(out.resolve()))))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=Path('artifacts/real_video/background_completion/v3_texture/completed_scene.pt'));parser.add_argument('--name',default='scene');args=parser.parse_args()
    if not args.name.replace('_','').isalnum(): raise ValueError('Invalid export name')
    main(args.source,args.name)
