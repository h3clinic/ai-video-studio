"""Record inspected sampled-frame evidence, never claim continuous playback."""
import json
from .original_cat_motion import OUT
from .articulation_quality import create_review, decide_review


def run():
    video=OUT/'motion_transfer_7s.mp4'
    review=create_review(video,[dict(path=OUT/f'frame_{i:03d}.png',frame_index=i) for i in [0,28,56,84,111]])
    review['claim_type']='reconstruction'
    review['reviewer']=dict(kind='agent',name='Codex sampled-frame inspection',observed_video=False)
    review['provenance']=dict(future_conditioned=True,test_used_for_training=False,test_used_for_selection=False)
    observations=[
        'Original static silhouette retained; raised far forepaw and blurry facial texture already present.',
        'Small limb change visible in both fixed cameras; no detached cloud visible.',
        'Far foreleg extends forward; shoulder and belly start distorting.',
        'Foreleg advances but shoulder skin stretches into a diagonal ridge; paws do not form a grounded gait.',
        'Distorted belly and shoulder persist; no coherent contact cycle established.']
    for frame,observation in zip(review['frames'],observations): frame['observation']=observation
    values={
        'paw_placement':('fail','Floating paws; no contact-consistent gait.'),
        'limb_bending':('fail','Foreleg movement occurs but limb proportions and articulation are not reliable.'),
        'body_shape':('fail','Late poses develop diagonal shoulder/belly stretching absent at rest.'),
        'tearing':('uncertain','No separated Gaussian cloud in sampled views; extreme face-area changes prevent a no-tearing claim.'),
        'temporal_coherence':('uncertain','Five sampled times show pose evolution; continuous playback not inspected.')}
    for key,(status,observation) in values.items():
        review['dimensions'][key]=dict(status=status,observation=observation,evidence_frame_ids=[f'frame_{i:03d}' for i in range(5)])
    review['limitations']=['Retargeted fit of a different observed cat clip; NOT new conditional generation.',
        '1.875 seconds of learned source motion slowed to seven seconds.',
        'Manual inferred rig; original source image and static six-view sheet inspected for comparison.',
        'No quality-matched compute comparison; no source asset overwritten.']
    review['decision']=decide_review(review,video)
    (OUT/'visual_review.json').write_text(json.dumps(review,indent=2))
    print(json.dumps(review['decision']))


if __name__=='__main__': run()
