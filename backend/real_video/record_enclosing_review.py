"""Record directly inspected six-side and sampled-orbit evidence."""
import json
from .enclosing_scene import ROOT
from .articulation_quality import create_review,decide_review
from .checkpoint_io import digest

def main():
    folder=ROOT/'inspection';out=folder/'visual_review.json'
    if out.exists():raise FileExistsError('Preserve review')
    video=folder/'generated_surfaces_camera_only_7s.mp4'
    review=create_review(video,[dict(path=folder/f'orbit_{i:03d}.png',frame_index=i) for i in [0,56,84]])
    review.update(claim_type='reconstruction',reviewer=dict(kind='agent',name='Root direct sampled-frame, six-face and live front/back viewer inspection',observed_video=False),
                  provenance=dict(future_conditioned=False,test_used_for_training=False,test_used_for_selection=False))
    observations=['Visible cat side retains orange stripes; paws blunted; background has rectangular sampling artifacts and source-anchor seam.',
                  'Opposite side contains generated coloured body surface, limbs and tail, not an empty sheet. Body too rounded; feet do not establish convincing ground contact.',
                  'End-on oblique pose shows actual torso thickness, but face and limb anatomy remain poorly defined.']
    for frame,observation in zip(review['frames'],observations):frame['observation']=observation
    evidence=[f['id'] for f in review['frames']]
    judgments={'paw_placement':('fail','Feet look suspended or blunted; no convincing grounding or contact shadow.'),
        'limb_bending':('fail','Poorly formed static limb geometry; no learned articulated motion.'),
        'body_shape':('fail','Full inferred volume exists, but body too rounded and face poorly formed.'),
        'tearing':('fail','Environment anchor seam and finite-stencil background grid remain in orbit video.'),
        'temporal_coherence':('uncertain','Static geometry persists, but only sampled camera poses inspected; no continuous playback or new cat motion.')}
    for key,(status,observation) in judgments.items():review['dimensions'][key]=dict(status=status,observation=observation,evidence_frame_ids=evidence)
    review['additional_evidence']=[dict(path=str(folder/name),sha256=digest(folder/name)) for name in ['cat_six_views.jpg','environment_six_views.jpg']]
    review['limitations']=['Existing TripoSR weights infer hidden cat geometry; Wan and Gaussian-native weights were not trained.',
        'Environment uses six overlapping diffusion generations and a shared inferred radial depth surface plus imposed ground plane.',
        'Full angular coverage is not complete free-roaming geometry or anatomical correctness.',
        'Cat six-view up/down labels denote viewing direction; up looks from below, down from above.',
        'Interactive WebGL has a different footprint approximation from the finite-stencil Python video renderer.']
    review['decision']=decide_review(review,video)
    out.write_text(json.dumps(review,indent=2));print(json.dumps(review['decision']))

if __name__=='__main__':main()
