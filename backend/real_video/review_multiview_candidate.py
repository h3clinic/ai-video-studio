"""Create fail-closed review from explicitly supplied inspected observations."""
import argparse
import json
from pathlib import Path
from .articulation_quality import create_review,decide_review
from .checkpoint_io import digest


def record(root,observations,judgments):
    path=root/'visual_review.json'
    if path.exists():raise FileExistsError('Preserve prior review')
    video=root/'camera_only_7s.mp4'
    review=create_review(video,[dict(path=root/f'orbit_{i:03d}.png',frame_index=i) for i in [0,28,56,84]])
    review.update(claim_type='reconstruction',reviewer=dict(kind='agent',name='Root sampled-frame and six-view inspection',observed_video=False),
        provenance=dict(future_conditioned=False,test_used_for_training=False,test_used_for_selection=False))
    for f,observation in zip(review['frames'],observations):f['observation']=observation
    for key,(status,observation) in judgments.items():review['dimensions'][key]=dict(status=status,observation=observation,evidence_frame_ids=[f['id'] for f in review['frames']])
    review['additional_evidence']=[dict(path=str(root/'appearance_six_views.jpg'),sha256=digest(root/'appearance_six_views.jpg'))]
    review['limitations']=['Camera-only diagnostic, not cat motion. Sampled frames, not continuous playback.',
        'Generated hidden texture has no ground truth. Base neural weights unchanged.',
        'Color coverage is not image quality. Geometry remains single-image inferred.']
    review['decision']=decide_review(review,video);path.write_text(json.dumps(review,indent=2));print(json.dumps(review['decision']))
