"""Explicit observations of the full-resolution ablation and shared-CFG pilot."""
import json
from pathlib import Path
from real_video.review_wan_cat_memory import record


def main():
    observations=dict(
        frames='Full-resolution samples 0,16,32 show a cat stepping forward with a curved tail. Gaussian-conditioned output has intensely clipped orange/white stripes and cyan-black facial shadows, unlike the reference and the no-memory control.',
        appearance='FAIL: severe saturation and blown highlights destroy fur detail. The no-memory control is visually less distorted. Source frame 000 and base/adapted/no-memory contact sheet were inspected.',
        dimensions=dict(paw_placement=('uncertain','Foreground paw is visible, but grass occludes other contacts and no continuous playback was observed.'),
                        limb_bending=('uncertain','Different front-leg extensions visible across sampled poses; full gait cycle not verified.'),
                        body_shape=('uncertain','Recognizable quadruped silhouette; exact reference anatomy and identity not established.'),
                        tearing=('uncertain','No disconnected surface patches apparent in sampled RGB frames; continuous sequence not inspected.'),
                        temporal_coherence=('uncertain','Sampled poses change consistently at a coarse level; temporal stability not established.')))
    full=record('artifacts/real_video/wan_cat_memory/v1_fullres','adapted',[0,16,32],observations)
    shared=record('artifacts/real_video/wan_cat_memory/v1_shared_pilot','adapted',[0,4,8],dict(
        frames='Shared-memory positive/negative guidance still yields strong stripe contrast and later-frame ghosting. Body stays approximately in the same location while legs change.',
        appearance='FAIL: sharing memory across guidance branches is insufficient to remove saturation and detail degradation.',
        dimensions=dict(paw_placement=('fail','Later paw boundaries lose definition in grass.'),
                        limb_bending=('uncertain','Limb silhouette changes; physically valid articulation unverified.'),
                        body_shape=('uncertain','Cat silhouette retained, but exact identity not established.'),
                        tearing=('fail','Later samples have visibly blurred/ghosted outlines around head and limbs.'),
                        temporal_coherence=('fail','Later frames lose sharp detail compared to the first.'))))
    result=dict(quality_accepted=False,promoted=False,
                conclusion='Actual Wan effective attention weights trained successfully; Gaussian conditioning visibly changes output but degrades appearance. Neither tested conditioning scheme meets quality target.',
                full_resolution_articulation_gate=full,shared_pilot_gate=shared,
                appearance_gate=dict(status='fail',reason=observations['appearance']),
                verified_base_checkpoint_files_unchanged=True,
                scope='RGB personalization; no per-Gaussian motion output or compute savings',
                baseline_denoise_seconds=331.40820040000835,adapted_denoise_seconds=366.9094978000212,
                peak_cuda_allocated_bytes_both=3603765760,
                timing_limit='One sequential sample per method, no replicated benchmark; decode/loading excluded',
                ablation_latent_rmse=0.654071569442749,
                next_required_work=['Align geometry conditioning with pretrained feature space rather than treating arbitrary Gaussian descriptors as text residuals.',
                                    'Foreground/face multiscale supervision and identity preservation across views; separate appearance from background.',
                                    'Evaluate multiple conditioning assets and held-out motions before claiming geometry-aware generalization.'])
    Path('artifacts/real_video/wan_cat_memory/selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
