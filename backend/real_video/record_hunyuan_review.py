"""Explicit root-agent judgments after inspecting rendered source and orbit samples."""
import json
from .hunyuan_gaussian import ROOT
from .checkpoint_io import digest
from .articulation_quality import create_review,decide_review


def main():
    path=ROOT/'visual_review.json'
    if path.exists(): raise FileExistsError('Preserve review')
    video=ROOT/'camera_only_7s.mp4'
    review=create_review(video,[dict(path=ROOT/f'orbit_{i:03d}.png',frame_index=i) for i in [0,28,56,84]])
    review.update(claim_type='reconstruction',reviewer=dict(kind='agent',name='Root direct source, six-view, orbit-sample and sidebar inspection',observed_video=False),
        provenance=dict(future_conditioned=False,test_used_for_training=False,test_used_for_selection=False))
    observations=['Source-facing fur stripes and eye visible; gray foot and ear areas, rough face, incomplete paws.',
        'End-on face shows gray unobserved side and discontinuous appearance; real torso thickness exists.',
        'Opposite side geometry complete but untextured gray, with scattered front-color leakage through surface sampling gaps.',
        'Rear view has real thickness, but strong appearance boundary and poorly defined limb structure.']
    for frame,observation in zip(review['frames'],observations): frame['observation']=observation
    evidence=[f['id'] for f in review['frames']]
    judgments={'paw_placement':('fail','Rough incomplete paws, no verified ground contact.'),
        'limb_bending':('fail','Legs separated but anatomy imperfect; static pose only, no learned articulation.'),
        'body_shape':('fail','Recognizable closed cat volume; facial protrusions and uneven anatomy still fail finished-asset quality.'),
        'tearing':('fail','Untextured appearance boundary and small color leakage remain; no geometry tearing claim from static inspection.'),
        'temporal_coherence':('uncertain','Only sampled camera poses inspected, not continuous playback; cat itself does not move.')}
    for key,(status,observation) in judgments.items(): review['dimensions'][key]=dict(status=status,observation=observation,evidence_frame_ids=evidence)
    review['additional_evidence']=[dict(path=str(ROOT/name),sha256=digest(ROOT/name)) for name in ['reference.png','fit_result.png','geometry_six_views.jpg','appearance_six_views.jpg']]
    review['numeric_metrics']=json.loads((ROOT/'appearance_report.json').read_text())
    review['limitations']=['Unchanged Hunyuan base weights; Gaussian RGB parameters fitted per asset only.',
        'No unseen texture generator was run. Full Hunyuan Paint remains unintegrated on this Windows/12GB system.',
        'Sharper visible detail is not complete blur elimination or validated hidden anatomy.',
        'Shape timer includes loading, inference, extraction and checkpoint saves; not a matched baseline comparison.',
        '314 unit tests pass, but visual quality is rejected.']
    review['decision']=decide_review(review,video)
    path.write_text(json.dumps(review,indent=2));print(json.dumps(review['decision']))


if __name__=='__main__':main()
