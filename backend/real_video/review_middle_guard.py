"""Persist inspected guard-repair results and source-preservation assertions."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from .checkpoint_io import load_verified,digest
from .review_multiview_candidate import record

def main():
    base=Path('artifacts/real_video/hunyuan_gaussian')
    old=load_verified(base/'v5_full_paint/gaussian_fitted.pt');n=len(old['position'])
    tree=cKDTree(old['position'].numpy())
    for name in ['v14_middle_guarded','v15_middle_wide_guarded']:
        root=base/name
        out=load_verified(root/'gaussian_fitted.pt')
        for key in ['position','covariance','normal','colour','ids']:
            assert torch.equal(old[key],out[key][:n]),key
        distance,_=tree.query(out['position'][n:].numpy())
        assert float(distance.max())<.025
        record(root,[
            'Side: body silhouette retained; coarse paws and slab-like whiskers persist.',
            'Front: large orange sheet no longer occludes the eye region; eyes still poorly defined and forehead texture over-smoothed.',
            'Opposite side: face remains connected at sampled scale; malformed whiskers persist. No new articulation.',
            'Rear: unchanged green underside and cyan tail; static camera-only sequence.'
        ],{
            'paw_placement':('fail','Original coarse paws and unverified ground contact remain.'),
            'limb_bending':('uncertain','No generated animal motion in this diagnostic.'),
            'body_shape':('fail','Large sheet regression repaired, but original malformed whiskers and poor facial detail remain.'),
            'tearing':('uncertain','Gross sheet/gap defect is absent in sampled views; watertightness and deformation are not established.'),
            'temporal_coherence':('uncertain','Only four separated orbit frames inspected, not continuous playback.')
        })
        result=dict(accepted=False,viewer_replaced=False,
            local_defect_improved=True,full_facial_quality_pass=False,
            improvement_scope='Relative to v13 middle replacement, not demonstrated improvement over the original v5 face.',
            original_attribute_preservation=['position','covariance','normal','colour','ids'],
            checkpoint_sha256=digest(root/'gaussian_fitted.pt'),
            new_point_distance_to_source_quantiles=np.quantile(distance,[0,.5,.9,.99,1]).tolist(),
            comparison='v13 unguarded wide fill max distance 0.12861; guarded fill hard support limit 0.025 model units. Geometry bound is not a quality metric.',
            base_weights_modified=False,
            limitations=['Uses original middle surface as a prior; not missing anatomy generation.',
                        'Reuses source appearance; not a learned inpainting model.',
                        'No motion generation or quality-matched compute savings established.'])
        (root/'outcome.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    main()
