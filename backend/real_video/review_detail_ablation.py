"""Record explicitly inspected failures, not automatic quality approval."""
import json
from pathlib import Path
from .articulation_quality import create_review,decide_review
from .review_multiview_candidate import record

ROOT=Path('artifacts/real_video/hunyuan_gaussian/detail_ablation')

def main():
    observations=[
        'Side view: striped body and four limbs visible, but muzzle is malformed and paw shapes coarse. Static pose.',
        'Front view: solid whisker slab across muzzle and weak eye definition; forepaws lack clear toes. Static pose.',
        'Opposite side: generated stripes cover backside; head remains malformed and texture has baked-in color casts.',
        'Rear view: tail and hind legs retained, green underside and cyan tail bands remain. Static pose; not motion evidence.'
    ]
    judgments={
        'paw_placement':('fail','Coarse paws and suspended inferred pose do not establish correct ground contact.'),
        'limb_bending':('uncertain','Static single pose; no articulated motion generated or evaluated.'),
        'body_shape':('fail','Whiskers are solid slabs and facial anatomy remains malformed in front and oblique views.'),
        'tearing':('uncertain','Sampled views show speckling; no continuous playback reviewed.'),
        'temporal_coherence':('uncertain','Camera-only orbit of a static asset; no evidence of learned coherent animal motion.')
    }
    if not (ROOT/'restored/visual_review.json').exists():record(ROOT/'restored',observations,judgments)
    for name in ['narrow','winner','combined']:
        root=ROOT/name;path=root/'visual_review.json'
        if path.exists():continue
        video=root/'camera_only_7s.mp4'
        review=create_review(video,[dict(path=root/f'orbit_{i:03d}.png',frame_index=i) for i in [0,28,56,84]])
        review['limitations']=['Not fully visually reviewed. Narrow close-up shows holes; winner close-up still has malformed anatomy. No promotion.']
        review['decision']=decide_review(review,video)
        path.write_text(json.dumps(review,indent=2))
    outcome=dict(accepted=False,viewer_replaced=False,weights_trained=False,tests_passed=24,
        variants=['narrow','winner','combined','restored'],
        result='No candidate solves facial quality. Narrow footprints introduce holes; fusion change insufficient; learned SR does not repair anatomy.',
        completed='Parameterized baking, controlled fixed-state trials, 2048px pretrained texture restoration baked into 400000 persistent Gaussians.',
        unresolved='Jointly consistent high-detail facial generation and corrected facial geometry; no measured quality-matched efficiency win.')
    (ROOT/'outcome.json').write_text(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
