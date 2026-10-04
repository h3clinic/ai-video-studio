"""Explicit inspected observations, not automatic approval from smoothness."""
import json
from pathlib import Path
from .review_multiview_candidate import record

def main():
    root=Path('artifacts/real_video/hunyuan_gaussian/v11_weld_band')
    record(root,[
        'Side: body shape and coarse paw placement persist; muzzle protrusions remain; no new animal motion.',
        'Front: eyes poorly defined and solid whisker slabs remain. No satisfactory facial correction.',
        'Opposite side: back texture retained; coarse limb shapes and malformed muzzle persist.',
        'Rear: hind legs and tail present; green/cyan baked colors persist. Camera-only static geometry.'
    ],{
        'paw_placement':('fail','Original coarse paws and unverified contact remain; band edit does not address them.'),
        'limb_bending':('uncertain','No articulation generated or tested.'),
        'body_shape':('fail','Central band fairing does not correct the malformed muzzle, whiskers, or eye definition.'),
        'tearing':('uncertain','No local flipped triangles, but no continuous deformation or global self-intersection review.'),
        'temporal_coherence':('uncertain','Four sampled camera-only views; no continuous playback or learned motion.')
    })
    (root/'outcome.json').write_text(json.dumps(dict(accepted=False,viewer_replaced=False,
        result='Bounded geometry edit works numerically; facial visual quality still fails.',
        base_weights_modified=False,another_agent_role='Read-only pipeline/seam audit, not a trained filling model.',
        required_next='Reference- or prior-conditioned patch geometry generation with fixed surrounding boundary; smoothing alone is insufficient.'),indent=2))

if __name__=='__main__':main()
