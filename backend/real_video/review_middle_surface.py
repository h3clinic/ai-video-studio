"""Record actual failed middle-patch inspection; never auto-promote."""
import json
from pathlib import Path
from .review_multiview_candidate import record

def main():
    for name in ['v12_middle_surface','v13_middle_solid']:
        root=Path('artifacts/real_video/hunyuan_gaussian')/name
        record(root,[
            'Side view retains body but coarse paws and slab-like whiskers remain.',
            'Front view shows new flat orange sheets covering the forehead and eye region; facial structure is worse than the source.',
            'Opposite side shows abrupt facial patch transitions and malformed muzzle; no new animal motion.',
            'Rear retains green underside and cyan tail artifacts. Static shape only.'
        ],{
            'paw_placement':('fail','Coarse original paws persist; source has finer distinct toes.'),
            'limb_bending':('uncertain','No generated articulation in this static reconstruction experiment.'),
            'body_shape':('fail','New middle surface damages forehead and eyes instead of restoring plausible facial anatomy.'),
            'tearing':('fail','Visible abrupt patch boundaries and discontinuous facial surfaces.'),
            'temporal_coherence':('uncertain','Four separated camera frames inspected, not continuous playback or animal motion.')
        })
        (root/'outcome.json').write_text(json.dumps(dict(
            accepted=False,viewer_replaced=False,base_weights_modified=False,
            result='New middle splats were constructed, but facial quality regressed. Rejected.',
            source_comparison='Compared against original Wan frame_000 and six rendered views, close-up and four orbit frames.',
            scope='Controlled middle replacement on current Hunyuan asset, not a recovered historic mirrored-halves checkpoint.',
            required_next='Anatomy-conditioned local 3D patch generator with fixed side boundaries; interpolated geometry and colors are insufficient.'
        ),indent=2))

if __name__=='__main__':main()
