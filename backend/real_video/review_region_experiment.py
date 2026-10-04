"""Explicit sampled-frame observations for the first region experiment."""
import json
from pathlib import Path
from .review_multiview_candidate import record

def main():
    root=Path('artifacts/real_video/hunyuan_gaussian/v10_region_head')
    observations=[
        'Three-quarter head: visible green eyes and red nose but enlarged eye appearance, forehead spikes and clumped whiskers; flat bottom boundary.',
        'Opposite three-quarter view: whiskers remain thick white clumps; neck cropped into a slab, ear asymmetry and baked green color remain.',
        'Rear-oblique: large flat crop boundary, distorted neck/back completion; not a seamless whole-character part.',
        'Other rear-oblique: flat crop boundary and coarse surface persist. Same static head, camera changed only.'
    ]
    judgments={
        'paw_placement':('uncertain','Isolated head experiment; no paws present or evaluated.'),
        'limb_bending':('uncertain','No limbs or generated articulation in this experiment.'),
        'body_shape':('fail','Head proportions distorted; solid whisker clumps and flat crop boundary persist; no body integration.'),
        'tearing':('uncertain','No deformation tested; crop boundary is not an evaluated body seam.'),
        'temporal_coherence':('uncertain','Only sampled camera-orbit frames inspected, not continuous playback or animal motion.')
    }
    record(root/'paint',observations,judgments)
    shape=json.loads((root/'shape/shape_report.json').read_text())
    paint=json.loads((root/'paint/generation_report.json').read_text())
    result=dict(accepted=False,viewer_replaced=False,base_weights_modified=False,
        mechanism='Category-independent explicit-ROI rerun of pretrained shape and multiview paint; not a trained region refiner.',
        observation='Eyes and nose more discernible, but face identity/proportions degraded; whiskers remain clumps. No overall quality pass.',
        generalization='One manually selected cat crop only. Reusable code does not establish cross-category model generalization.',
        next_hypothesis='Condition local geometry updates on the whole-object 3D state, preserve boundary constraints, and learn bounded regional Gaussian residuals from diverse multiview data.',
        isolated_part=True,gaussians=400000,shape_seconds=shape['seconds'],paint_seconds=paint['seconds'],
        peak_stage_cuda_allocated_bytes=max(shape['peak_cuda_allocated_bytes'],paint['peak_cuda_allocated_bytes']),
        timing_limit='Stage timings include their model loading; exclude other pipeline stages. Not a quality-matched efficiency comparison.',
        validation='9 targeted unit tests passed; visual candidate rejected.')
    (root/'outcome.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
