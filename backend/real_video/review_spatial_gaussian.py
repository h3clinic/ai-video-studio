"""Record observed failures; retain experimental weights without promotion."""
import json
from pathlib import Path
from real_video.review_wan_cat_memory import record


def main():
    root = Path('artifacts/real_video/wan_spatial_gaussian')
    folder = root/'v1_isolated'
    decisions = {}
    for mode in ['center', 'shift_only', 'scale_only']:
        decisions[mode] = record(folder, mode, [0, 4, 8], dict(
            frames='Right-facing orange cat with elevated tail. Later samples are visibly softer; feet and facial features lack reference detail.',
            appearance='Clipped orange/white coat; supplied shape influences orientation but exact identity, size, and placement are not preserved.',
            dimensions=dict(
                paw_placement=('uncertain', 'Feet/contact are not sufficiently resolved to verify natural stepping.'),
                limb_bending=('uncertain', 'A front leg changes position, but three samples cannot validate articulation.'),
                body_shape=('fail', 'Generated silhouette and dimensions do not match the corresponding Gaussian condition.'),
                tearing=('uncertain', 'Soft or ghosted contours in later frames; continuous playback not inspected.'),
                temporal_coherence=('fail', 'Visible sharpness loss from first to later sampled frames.'))))
    metrics = json.loads((folder/'report.json').read_text())['outputs']
    center, shift, small = [metrics[k] for k in ['center', 'shift_only', 'scale_only']]
    findings = dict(
        quality_accepted=False, promoted=False, visual_decisions=decisions,
        scope='Actual trained spatial residuals inside Wan; RGB output, not Gaussian trajectories',
        supervision='Six rendered views of one inferred asset; no external real-video training/validation',
        expected_orange_centroid_shift_px=shift['condition_extent']['center_x']-center['condition_extent']['center_x'],
        observed_orange_centroid_shift_px=shift['first_frame_extent']['center_x']-center['first_frame_extent']['center_x'],
        expected_orange_width_ratio=small['condition_extent']['x_span_90']/center['condition_extent']['x_span_90'],
        observed_orange_width_ratio=small['first_frame_extent']['x_span_90']/center['first_frame_extent']['x_span_90'],
        metric_limit='Color-threshold heuristic; not semantic segmentation or proof of geometry fidelity',
        result='Partial position response, failed scale control. Appearance and temporal detail remain unacceptable.',
        v1_protocol_correction='Original v1 translated sample changed BOTH dx and scale. v1_isolated separates these without retraining.',
        efficiency='No savings established. v1 control-off vs on includes same frozen appearance LoRA, NOT original untouched Wan.',
        next_requirement='Broader calibrated multi-view/multi-scale paired training and explicit spatial consistency supervision; temporal training needed for sustained detail.',
        frozen_evaluation=True, test_used_for_training=False)
    (folder/'evaluation.json').write_text(json.dumps(findings, indent=2), encoding='utf-8')
    (root/'selection.json').write_text(json.dumps(dict(accepted=False, candidate='v1',
        reason=findings['result'], active_checkpoint=None), indent=2), encoding='utf-8')
    print(json.dumps(findings, indent=2))


if __name__ == '__main__': main()
