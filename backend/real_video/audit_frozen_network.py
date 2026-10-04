"""Read-only model diagnosis; writes a new audit, never changes model weights.

Only the saved three-observation cat seed and the frozen v3 checkpoint enter
predictions. No source video, future target tracks or fitted mesh motion is read.
"""
import json
from pathlib import Path

import torch

from .checkpoint_io import digest, load_verified
from .vector_motion_network import VectorMotionNetwork


ROOT = Path('artifacts/real_video/true3d/frozen_motion_diagnosis/v1')
MODEL = Path('artifacts/real_video/true3d/learned_motion_loop/v3/model.pt')
SEED = Path('artifacts/real_video/animal_graph/v1/seeds/cat.pt')


def stats(values):
    values = values.detach().double().flatten()
    return dict(mean=float(values.mean()), median=float(values.median()),
                minimum=float(values.min()), maximum=float(values.max()),
                p90=float(torch.quantile(values, .9)))


def motion_partition(reference, current):
    """Translation, global planar rotation and residual, all in source pixels."""
    x = reference.double() * 480
    y = current.double() * 480
    cx, cy = x.mean(0), y.mean(0)
    xc, yc = x - cx, y - cy
    u, _, vh = torch.linalg.svd(xc.T @ yc)
    correction = torch.eye(2, dtype=x.dtype)
    correction[-1, -1] = torch.linalg.det(u @ vh)
    row_rotation = u @ correction @ vh
    remainder = yc - xc @ row_rotation
    displacement = y - x
    return dict(centroid_translation_px=(cy - cx).tolist(),
                centroid_translation_norm_px=float((cy - cx).norm()),
                best_rigid_rotation_degrees=float(torch.rad2deg(torch.atan2(row_rotation[0, 1], row_rotation[0, 0]))),
                total_displacement_norm_px=stats(displacement.norm(dim=-1)),
                centered_displacement_rms_px=float((displacement - displacement.mean(0)).square().sum(-1).mean().sqrt()),
                nonrigid_after_best_SE2_rms_px=float(remainder.square().sum(-1).mean().sqrt()),
                nonrigid_after_best_SE2_norm_px=stats(remainder.norm(dim=-1)))


@torch.no_grad()
def run():
    torch.set_num_threads(4)
    destination = ROOT / 'network_audit.json'
    if destination.exists():
        raise FileExistsError('Preserve prior read-only diagnosis')
    checkpoint = load_verified(MODEL)
    seed = load_verified(SEED)
    observed = seed['controls']['position'][None]
    adjacency = seed['controls']['adjacency'][None]
    confidence = seed['controls']['confidence'][2][None]
    model = VectorMotionNetwork(**checkpoint['config']).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    disabled = VectorMotionNetwork(**checkpoint['config']).eval()
    for parameter in disabled.parameters():
        parameter.zero_()
    initial = model.initialize(observed, adjacency, confidence)
    reference = observed[0, 2]
    v01 = (observed[0, 1] - observed[0, 0]) * 480
    v12 = (observed[0, 2] - observed[0, 1]) * 480
    features = model.features(initial)
    names = [('displacement', 0, 2), ('reference_relative_to_center', 2, 4),
             ('velocity_times20', 4, 6), ('acceleration_times20', 6, 8),
             ('initial_velocity_times20', 8, 10), ('global_velocity_times20', 10, 12),
             ('observed_confidence', 12, 13), ('elapsed_step_over12', 13, 14)]
    modes = {}
    for mode in ['trained_v3', 'zero_weights', 'constant_velocity']:
        active = disabled if mode == 'zero_weights' else model
        state = active.initialize(observed, adjacency, confidence)
        captured = []
        handle = active.head.register_forward_hook(lambda _module, _args, output: captured.append(output.detach().clone()))
        rows = []
        try:
            for index in range(12):
                previous = state
                if mode == 'constant_velocity':
                    state = dict(previous, position=previous['position'] + previous['velocity'],
                                 time=previous['time'] + 1)
                    damping = torch.ones_like(previous['velocity'][..., :1])
                    residual = torch.zeros_like(previous['velocity'])
                else:
                    state = active.step(previous)
                    action = captured[-1]
                    damping = .5 + .5 * action[..., 2:3].sigmoid()
                    residual = active.residual_scale * previous['scale'] * action[..., :2].tanh()
                damped = damping * previous['velocity']
                reconstructed = damped + residual
                denominator = previous['velocity'].norm(dim=-1) * residual.norm(dim=-1)
                valid_direction = denominator > 1e-12
                cosine = (previous['velocity'] * residual).sum(-1) / denominator.clamp_min(1e-12)
                step_pixels = state['velocity'][0] * 480
                lower = reference[:, 1] * 480 > 300
                rows.append(dict(step=index + 1,
                    damping=stats(damping),
                    previous_speed_px_per_step=stats(previous['velocity'].norm(dim=-1) * 480),
                    damped_speed_px_per_step=stats(damped.norm(dim=-1) * 480),
                    learned_residual_norm_px_per_step=stats(residual.norm(dim=-1) * 480),
                    learned_residual_mean_vector_px_per_step=(residual[0].mean(0) * 480).tolist(),
                    residual_opposes_previous_velocity_fraction=float((cosine[valid_direction] < 0).float().mean()) if bool(valid_direction.any()) else None,
                    net_speed_px_per_step=stats(step_pixels.norm(dim=-1)),
                    net_lower_region_speed_mean_px=float(step_pixels[lower].norm(dim=-1).mean()),
                    net_upper_region_speed_mean_px=float(step_pixels[~lower].norm(dim=-1).mean()),
                    node_count_step_motion_under_half_pixel=int((step_pixels.norm(dim=-1) < .5).sum()),
                    velocity_formula_max_abs_error=float((reconstructed - state['velocity']).abs().max()),
                    recurrent_memory_norm=float(state['memory'].norm()),
                    cumulative_motion=motion_partition(reference, state['position'][0])))
        finally:
            handle.remove()
        modes[mode] = dict(per_step=rows, final_motion=motion_partition(reference, state['position'][0]),
                           mean_speed_across_steps_px=sum(row['net_speed_px_per_step']['mean'] for row in rows) / 12)
    report = dict(
        classification='Read-only diagnosis of the projected-motion network, before XYZ lifting or rendering.',
        model_path=str(MODEL), model_sha256=digest(MODEL), seed_path=str(SEED), seed_sha256=digest(SEED),
        checkpoint_selected_step=checkpoint['selected_step'], model_configuration=checkpoint['config'],
        parameters=sum(p.numel() for p in model.parameters()), final_head_weight_norm=float(model.head[-1].weight.norm()),
        strict_load=True, future_rgb_reads=0, future_target_reads=0, mesh_motion_reads=0,
        observed_shape=list(observed.shape), controls=observed.shape[2],
        units='Seed XY is image-height-normalized. Multiply by original 480-pixel height for source-image pixels. One step is one supplied sampled frame; DAVIS/Wan physical-time equivalence remains unverified.',
        observed=dict(frame0_to1_speed_px=stats(v01.norm(dim=-1)), frame1_to2_speed_px=stats(v12.norm(dim=-1)),
                      frame1_to2_centroid_step_px=v12.mean(0).tolist(), frame1_to2_motion_partition=motion_partition(observed[0, 1], observed[0, 2]),
                      frame0_to2_motion_partition=motion_partition(observed[0, 0], observed[0, 2]),
                      frame1_to2_subhalf_pixel_control_count=int((v12.norm(dim=-1) < .5).sum()),
                      confidence=stats(confidence), observed_object_span_pixels=float(initial['scale'].squeeze() * 480),
                      adjacency_row_sum=stats(initial['adjacency'].sum(-1)),
                      raw_zero_adjacency_rows=int((adjacency.sum(-1) == 0).sum())),
        initializer_checks=dict(position_is_observed_frame2=bool(torch.equal(initial['position'], observed[:, 2])),
            velocity_is_frame2_minus_frame1=bool(torch.equal(initial['velocity'], observed[:, 2] - observed[:, 1])),
            acceleration_is_difference_of_observed_velocities=bool(torch.equal(initial['acceleration'], observed[:, 2] - 2 * observed[:, 1] + observed[:, 0])),
            feature_width=int(features.shape[-1]), features_finite=bool(torch.isfinite(features).all()),
            initial_feature_components={name: stats(features[..., start:end]) for name, start, end in names}),
        recurrence='v_next=(0.5+0.5*sigmoid(a_damp))*v + residual_scale*observed_span*tanh(a_xy); p_next=p+v_next',
        outputs=modes,
        objective_review=dict(v3='Uniform-window, globally weighted squared positional loss plus velocity and edge error; weak nonzero supervision remains for low-confidence/hidden pseudo-targets.',
            v4='Sequence-balanced robust visibility-confidence-normalized loss was subsequently tried. Its best trained candidate still lost to the step-zero damping fallback on validation.',
            limitation='No action/gait/phase conditioning, 3D motion supervision or multi-hypothesis stochastic motion objective. Predicting from only three flow-derived XY states can regress toward a damped low-motion conditional estimate. This is a plausible explanatory mechanism, not an isolated causal proof of the training objective.'),
        source_sha256={name: digest(Path('real_video') / name) for name in ['vector_motion_network.py', 'audit_frozen_network.py']},
        mutation_scope='Only this new JSON audit is written. No model, seed, geometry, source video or earlier result is modified.'
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(destination), observed=report['observed'],
                         final={name: values['final_motion'] for name, values in modes.items()},
                         first_last={name: [values['per_step'][0], values['per_step'][-1]] for name, values in modes.items()}), indent=2))


if __name__ == '__main__':
    run()
