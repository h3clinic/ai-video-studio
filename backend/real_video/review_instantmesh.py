"""Explicit sampled-view review of the two completed InstantMesh experiments."""
import json
from pathlib import Path
from .checkpoint_io import digest,load_verified
from .review_multiview_candidate import record

def main():
    for name in ['v1','v2_head']:
        root=Path('artifacts/real_video/instantmesh')/name;head=name=='v2_head'
        observations=(
            ['Rear: truncated slab-like back of head and neck, with incorrect green bands.',
             'Side: eyes and nose are present, but ear/head shape is stretched and differs from source.',
             'Face: asymmetric, broadened/stretched ear region and poorly defined eyes; not a satisfactory replacement.',
             'Opposite side: flat cut surface and misshapen head persist; isolated crop not attached to body.']
            if head else
            ['Rear: generated back and tail are present; green/cyan texture contamination persists.',
             'Side: full torso is present, but coarse paws and fine-feature loss remain.',
             'Face: narrow simplified muzzle and poorly resolved eyes; original thin whiskers are not recovered.',
             'Opposite side: static body is present, but paws and face still lack source detail.']
        )
        record(root,observations,{
            'paw_placement':('uncertain' if head else 'fail','Not present in head-only crop.' if head else 'Coarse paws and unverified contact; possible small disconnected flecks.'),
            'limb_bending':('uncertain','Static reconstruction, no generated articulation.'),
            'body_shape':('fail','Head is distorted and cut off; not integrated.' if head else 'Connected-looking body/back but facial shape and detail remain inadequate.'),
            'tearing':('uncertain','Sampled views cannot certify watertight joins or deformation behavior.'),
            'temporal_coherence':('uncertain','Four orbit frames inspected, not continuous playback; no animal motion.')
        })
        views=json.loads((root/'views_report.json').read_text());mesh=json.loads((root/'mesh_report.json').read_text())
        output=dict(accepted=False,viewer_default_replaced=False,scope='Frozen InstantMesh -> mesh -> persistent surface-bound Gaussian asset.',
            result='Pipeline executed successfully; visual facial quality failed.',
            gaussians=mesh['gaussians'],views_inference_seconds=views['inference_seconds'],mesh_inference_seconds=mesh['inference_seconds'],
            maximum_stage_peak_gpu_allocated_bytes=max(views['peak_gpu_allocated_bytes'],mesh['peak_gpu_allocated_bytes']),
            source_conditioning_sha256=digest(root/'conditioning.png'),manual_head_roi=[460,75,650,290] if head else None,
            base_weights_trained=False,source_comparison='Original Wan reference frame and conditioning crop, six orthographic/perspective inspection directions and four orbit frames.',
            timing_limits='Inference excludes loading, download, export and preview rendering. No quality-matched efficiency comparison.',
            remaining='Full-context high-detail reconstruction/refinement is still needed; naive isolated head crop is rejected.')
        (root/'outcome.json').write_text(json.dumps(output,indent=2));print(json.dumps(output),flush=True)

if __name__=='__main__':main()
