"""Record explicit sampled-frame pilot observations; never auto-accept training."""
import json
from pathlib import Path
from real_video.articulation_quality import create_review, decide_review


def record(folder, mode, indices, observations):
    folder=Path(folder)
    video=folder/f'{mode}.mp4'
    review=create_review(video,[dict(path=folder/f'{mode}_{i:03d}.png',frame_index=i) for i in indices])
    review['claim_type']='conditional_motion_generation'
    review['reviewer']=dict(kind='agent',name='Codex sampled-frame visual inspection',observed_video=False)
    review['provenance']=dict(future_conditioned=False,test_used_for_training=False,test_used_for_selection=False)
    for frame in review['frames']: frame['observation']=observations['frames']
    for key,(status,observation) in observations['dimensions'].items():
        review['dimensions'][key]=dict(status=status,observation=observation,evidence_frame_ids=[f['id'] for f in review['frames']])
    review['limitations']=['Sampled images only; continuous playback not observed','RGB generation, no 3D motion or persistent per-Gaussian trajectory claim',observations['appearance']]
    review['decision']=decide_review(review,video)
    (folder/f'{mode}_visual_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    return review['decision']


if __name__=='__main__':
    print(record('artifacts/real_video/wan_cat_memory/v1','adapted',[0,4,8],dict(
        frames='Orange-striped quadruped faces left; later frames lose definition, with visible ghosting at head/legs.',
        appearance='Reference stripe pattern is more apparent than in base, but saturated orange/white and dark contour artifacts are not faithful fur detail.',
        dimensions=dict(paw_placement=('fail','Front paw contours merge into grass; contact and anatomy not reliably resolved.'),
                        limb_bending=('fail','Front limbs are shortened/ambiguous; later frames muddy articulation.'),
                        body_shape=('fail','Broad torso and oversized contrasting stripes do not establish reference identity.'),
                        tearing=('fail','Ghost-like duplicated contours around head and legs in later sampled frames.'),
                        temporal_coherence=('fail','Substantial loss of detail between first and later samples.')))))
