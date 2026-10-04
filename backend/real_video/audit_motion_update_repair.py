"""Independent CPU audit; no training, fitting, or mutation of prediction inputs.

Metric arithmetic is reproduced here rather than calling either trainer's
evaluator. Future targets are used only for scoring after immutable rollouts.
"""
import json
from pathlib import Path
import numpy as np
import torch

from .checkpoint_io import load_verified, digest
from .motion_model_io import load_motion_model
from .attach_vector_weights import ASSET, CAMERA, SEED, control_binding
from .dense_surface_fit import ARAPMesh

ROOT = Path('artifacts/real_video/true3d/motion_update_repair/v1')
TRIAL = Path('artifacts/real_video/true3d/learned_motion_loop/residual_velocity_v1')
OLD = Path('artifacts/real_video/true3d/learned_motion_loop/v3/model.pt')
DATA = Path('../../work/real_video/davis_animals/control_tracks_animal_only_v2.pt')


def basis(triangles):
    a, b = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    normal = torch.linalg.cross(a, b)
    norm = normal.norm(dim=-1)
    result = torch.stack((a, b, normal/norm.clamp_min(1e-12)[:, None]), -1)
    result[norm < 1e-12] = torch.eye(3)
    return result


@torch.no_grad()
def validation(models, protocol):
    data = load_verified(DATA)
    indices = [i for i, m in enumerate(data['metadata']) if m['split'] == 'validation']
    sequences = sorted({data['metadata'][i]['sequence'] for i in indices})
    truth = data['position'][indices]
    adjacency = data['adjacency'][indices]
    confidence = data['confidence'][indices]
    weight = confidence[:, 3:].clamp(0, 1)*data['visibility'][indices, 3:].clamp(0, 1)
    observed = truth[:, 2]
    scale = (observed.amax(1, keepdim=True)-observed.amin(1, keepdim=True)).amax(-1, keepdim=True).clamp_min(1e-5)
    horizon = torch.arange(1, 13)[None, :, None, None]
    initial_velocity = truth[:, 2]-truth[:, 1]
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.rollout(truth[:, :3], adjacency, 12, confidence[:, 2])[0]
    predictions['frozen'] = observed[:, None].expand(-1, 12, -1, -1)
    predictions['velocity'] = observed[:, None]+horizon*initial_velocity[:, None]
    predictions['average_velocity'] = observed[:, None]+horizon*((truth[:, 2]-truth[:, 0])/2)[:, None]
    for d in [.75, .95]:
        predictions[f'damped{round(d*100)}'] = observed[:, None]+(d*(1-d**horizon)/(1-d))*initial_velocity[:, None]
    true_velocity = torch.diff(truth[:, 2:], dim=1)
    denominator = weight.sum(2, keepdim=True)[..., None].clamp_min(1e-8)
    true_relative = true_velocity-(true_velocity*weight[..., None]).sum(2, keepdim=True)/denominator
    moving = true_relative.norm(dim=-1)/scale[:, None, :, 0] > protocol['evaluation_moving_threshold']
    mw = weight*moving
    reports = {}
    for mode, prediction in predictions.items():
        error = ((prediction-truth[:, 3:])/scale[:, None]*32).norm(dim=-1)
        pv = torch.diff(torch.cat((observed[:, None], prediction), 1), dim=1)
        pr = pv-(pv*weight[..., None]).sum(2, keepdim=True)/denominator
        ve = ((pv-true_velocity)/scale[:, None]*32).norm(dim=-1)
        re = ((pr-true_relative)/scale[:, None]*32).norm(dim=-1)
        cosine = (pr*true_relative).sum(-1)/(pr.norm(dim=-1)*true_relative.norm(dim=-1)).clamp_min(1e-8)
        ade = (error*weight).sum((1, 2))/weight.sum((1, 2)).clamp_min(1e-8)
        legacy = ((error*weight).sum(2)/weight.sum(2).clamp_min(1e-8)).mean(1)
        fde = (error[:, -1]*weight[:, -1]).sum(1)/weight[:, -1].sum(1).clamp_min(1e-8)
        groups = {}
        for sequence in sequences:
            ids = [j for j, i in enumerate(indices) if data['metadata'][i]['sequence'] == sequence]
            mass = mw[ids].sum()
            groups[sequence] = dict(ade=float(ade[ids].mean()), legacy_ade=float(legacy[ids].mean()),
                fde=float(fde[ids].mean()), moving_velocity_epe=float((ve*mw)[ids].sum()/mass),
                moving_relative_velocity_epe=float((re*mw)[ids].sum()/mass),
                moving_signed_direction_cosine=float((cosine*mw)[ids].sum()/mass),
                moving_relative_amplitude_ratio=float((pr.norm(dim=-1)*mw/scale[:, None, :, 0])[ids].sum() /
                    (true_relative.norm(dim=-1)*mw/scale[:, None, :, 0])[ids].sum()))
        reports[mode] = dict(means={k:float(np.mean([g[k] for g in groups.values()])) for k in next(iter(groups.values()))}, per_sequence=groups)
    return dict(data_sha256=digest(DATA), validation_windows=len(indices), sequence_macro=True,
        validation_reused_for_adaptation=True, final_test_count=0, metrics=reports,
        definitions='New ADE normalizes all valid horizons jointly per window. Legacy ADE normalizes each horizon separately before averaging. Scores use object span times 32 units, not raw pixel units.')


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    output = ROOT/'independent_audit.json'
    if output.exists():
        raise FileExistsError('Preserve independent audit')
    paths = dict(residual_learned=TRIAL/'model.pt', previous_learned=OLD)
    checkpoints = {k:load_verified(v) for k, v in paths.items()}
    models = {k:load_motion_model(v) for k, v in checkpoints.items()}
    initial = load_verified(TRIAL/'checkpoints/step_00000000.pt')
    differences = {k:float((v-initial['model'][k]).abs().max()) for k, v in checkpoints['residual_learned']['model'].items()}
    packet = load_verified(ROOT/'forecast.pt')
    old_packet = load_verified(Path('artifacts/real_video/true3d/attached_vector_weights/v2/forecast.pt'))
    asset, camera, seed = load_verified(ASSET), load_verified(CAMERA), load_verified(SEED)
    observed = seed['controls']['position'][None]
    adjacency = seed['controls']['adjacency'][None]
    confidence = seed['controls']['confidence'][2][None]
    replay = {}
    for name, model in models.items():
        state = model.initialize(observed, adjacency, confidence)
        path = [state['position'][0].clone()]
        for _ in range(30):
            state = model.step(state)
            path.append(state['position'][0].clone())
        path = torch.stack(path)
        replay[name] = dict(control_replay_max_error=float((path-packet['controls'][name]).abs().max()),
            first12_endpoint_motion_mean_px=float(((path[12]-path[0])*480).norm(dim=-1).mean()),
            final_state_max_error=max(float((v-packet['final_state'][name][k]).abs().max()) for k,v in state.items()))
    ids, depth, focal, rotation = control_binding(asset['mesh_vertices'], camera, observed[0, 2])
    rest = asset['mesh_vertices'].numpy()
    solver = ARAPMesh(rest, asset['mesh_faces'].numpy())
    node_weight = np.maximum(confidence[0].numpy(), .05)*10
    weight = np.zeros(len(rest), np.float64)
    np.add.at(weight, ids, node_weight)
    first_step = {}
    for mode in packet['controls']:
        displacement = (packet['controls'][mode][1]-observed[0, 2]).numpy()*480
        world = np.column_stack((displacement*depth[:, None]/focal, np.zeros(len(ids))))@rotation
        accum = np.zeros_like(rest, dtype=np.float64)
        np.add.at(accum, ids, (rest[ids]+world)*node_weight[:, None])
        target = rest.copy()
        target[weight > 0] = accum[weight > 0]/weight[weight > 0, None]
        result = solver.solve(target, weight, stiffness=6., iterations=8, prior=.002, initial=rest.copy())
        first_step[mode] = float(np.abs(result-packet['vertices'][mode][1].numpy()).max())
    old_identity = dict(mesh_first13_bitwise_equal=torch.equal(packet['vertices']['previous_learned'][:13], old_packet['vertices']['learned']),
        controls_first13_bitwise_equal=torch.equal(packet['controls']['previous_learned'][:13], old_packet['controls']['learned']),
        mesh_first13_max_error=float((packet['vertices']['previous_learned'][:13]-old_packet['vertices']['learned']).abs().max()),
        same_asset_hash=packet['asset_sha256']==old_packet['asset_sha256'], same_seed_hash=packet['seed_sha256']==old_packet['seed_sha256'],
        same_old_model_hash=packet['old_model_sha256']==old_packet['model_sha256'],
        same_control_bindings=torch.equal(packet['control_vertex_index'], old_packet['control_vertex_index']))
    print(json.dumps(dict(control_replay=replay, unchanged_old_branch=old_identity)), flush=True)
    # Independent full-state material decode. No GaussianSurfaceMemory calls.
    record = load_verified(ROOT/'gaussian_vectors.pt')
    face = asset['mesh_faces'].long()
    fi = asset['face_id'].long()
    inv = torch.linalg.inv(basis(asset['mesh_vertices'][face]))
    cov = (asset['frame']*asset['scale'][:, None].square())@asset['frame'].transpose(-1, -2)
    pe, ce, min_eigen = 0., 0., float('inf')
    for t, vertices in enumerate(packet['vertices']['residual_learned']):
        triangles = vertices[face]
        centers = (triangles[fi]*asset['barycentric'][..., None]).sum(1)
        deformation = (basis(triangles)@inv)[fi]
        covariance = deformation@cov@deformation.transpose(-1, -2)
        pe = max(pe, float((centers-record['position'][t]).abs().max()))
        ce = max(ce, float((covariance-record['covariance'][t]).abs().max()))
        min_eigen = min(min_eigen, float(torch.linalg.eigvalsh(record['covariance'][t]).min()))
    # Independent exponential integration; do not use the producer's SciPy inverse.
    angular = record['angular_velocity']/record['fps']
    skew = torch.zeros(*angular.shape[:-1], 3, 3)
    skew[..., 0, 1], skew[..., 0, 2] = -angular[..., 2], angular[..., 1]
    skew[..., 1, 0], skew[..., 1, 2] = angular[..., 2], -angular[..., 0]
    skew[..., 2, 0], skew[..., 2, 1] = -angular[..., 1], angular[..., 0]
    theta = angular.norm(dim=-1)[..., None, None]
    exponential = torch.eye(3)+torch.sinc(theta/torch.pi)*skew+.5*torch.sinc(theta/(2*torch.pi)).square()*(skew@skew)
    rotation_error = float((exponential@record['frame'][:-1]-record['frame'][1:]).abs().max())
    material = dict(record_sha256=digest(ROOT/'gaussian_vectors.pt'), states=len(record['position']), gaussians=len(record['ids']),
        ids_match=torch.equal(record['ids'], asset['ids']), asset_hash_matches=record['asset_sha256']==digest(ASSET),
        forecast_hash_matches=record['motion_sha256']==digest(ROOT/'forecast.pt'), all_state_center_max_error=pe,
        all_state_covariance_max_error=ce, minimum_covariance_eigenvalue=min_eigen,
        rotation_step_rodrigues_max_error=rotation_error,
        position_delta_max_error=float((torch.diff(record['position'], dim=0)-record['delta']).abs().max()),
        velocity_max_error=float((record['delta']*record['fps']-record['velocity']).abs().max()),
        orthogonality_max_error=float((record['frame'].transpose(-1, -2)@record['frame']-torch.eye(3)).abs().max()),
        motion_mode_identified_numerically='residual_learned; export does not include an explicit mode field')
    protocol = json.loads((TRIAL/'protocol.json').read_text())
    report = dict(scope='Independent CPU implementation and conditional-motion audit, not an unbiased final test or proof of physical motion.',
        forecast_sha256=digest(ROOT/'forecast.pt'), helper_sha256=digest(Path(__file__)),
        model_sha256={k:digest(v) for k,v in paths.items()}, architecture={k:type(v).__name__ for k,v in models.items()},
        new_checkpoint_step=checkpoints['residual_learned']['selected_step'], new_checkpoint_trained=checkpoints['residual_learned']['trained'],
        new_parameter_count=sum(v.numel() for v in models['residual_learned'].parameters()),
        changed_parameter_tensors=sum(v>0 for v in differences.values()), total_parameter_tensors=len(differences),
        max_parameter_change_from_step0=max(differences.values()), selected_equals_best_trained=digest(TRIAL/'model.pt')==digest(TRIAL/'best_trained.pt'),
        control_replay=replay, unchanged_old_branch=old_identity, independent_first_arap_step_max_error=first_step,
        vector_record=material, validation=validation(models, protocol),
        code_review=dict(prediction_future_inputs=False, three_observed_frames=True, no_teacher_forcing=True, same_solver=True,
            posthoc_motion_gain=1., same_camera_and_normalization=True, depth_is_fixed_inferred_prior=True,
            training_gradients_through_surface_or_gaussian_render=False, prediction_dimensions='48 XY controls, not all Gaussian 3D/rotation targets'),
        visual_review=dict(accepted_natural_articulation=False, inspected_steps=[12,15,30], trained_horizon_steps=12,
            within_horizon_failure='At step12/source14 the reference front paw is bent and lifted forward, while the prediction retains a downward extended front leg. Rear limbs overlap and torso shape is distorted.',
            beyond_horizon_failure='Steps15 and30 progressively slump/compress the torso and cross/fold rear limbs. More motion does not establish a correct gait.',
            frame12_path=str(ROOT/'evaluation/independent_frame_12.png')),
        tenfold_quality_achieved=False, quality_matched_cost_saving_demonstrated=False,
        limits=['Validation was reused adaptively; no untouched test.', 'Deterministic video-conditioned sparse XY forecast, not text-to-video or learned physical 3D.',
            'Steps13..30 are outside the trained 12-step horizon.', 'All-state Gaussian records are expanded diagnostic storage, not inference memory/compression evidence.',
            'Small validation trajectory improvement is not natural articulation. Physical sampling-rate transfer is uncalibrated.'])
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), vector_record=material, validation_means={k:v['means'] for k,v in report['validation']['metrics'].items()})), flush=True)


if __name__ == '__main__':
    main()
