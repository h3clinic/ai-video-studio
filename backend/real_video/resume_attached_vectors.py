"""Save/resume actual recurrent Gaussian-support inference state, not replay.

Checkpoint construction can read the observed seed to recover an earlier run's
unsaved recurrent state. Resume itself reads ONLY the checkpoint, fixed asset
and learned weights. This is an engineering smoke test, not a quality claim.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .attach_vector_weights import control_binding, SEED
from .checkpoint_io import load_verified, save_inference_checkpoint, digest, keep_windows_awake
from .dense_surface_fit import ARAPMesh
from .motion_model_io import load_motion_model

ROOT = Path('artifacts/real_video/true3d/attached_vector_weights/v2')


def load_model(checkpoint):
    """Compatibility alias; explicit factory prevents cross-architecture loading."""
    return load_motion_model(checkpoint)


def forecast_model_provenance(forecast, mode):
    if mode not in forecast['controls'] or mode not in forecast['vertices']:
        raise ValueError('Forecast mode has no saved control/geometry trajectory')
    if 'model_paths' in forecast:
        hash_key = {'residual_learned':'new_model_sha256', 'previous_learned':'old_model_sha256'}.get(mode)
        if hash_key is None or mode not in forecast['model_paths']:
            raise ValueError('Selected forecast mode is not an explicitly identified learned model')
        return Path(forecast['model_paths'][mode]).resolve(), forecast[hash_key]
    if mode != 'learned':
        raise ValueError('Legacy forecast supports learned continuation only')
    return Path(forecast['model_path']).resolve(), forecast['model_sha256']


def continuation_tensor_bytes(packet):
    return sum(value.numel()*value.element_size() for key,value in packet.items()
               if isinstance(value,torch.Tensor) and key not in ['generated_vertices','generated_controls']) + \
           sum(value.numel()*value.element_size() for value in packet['state'].values())


@contextmanager
def resume_input_guard(checkpoint_path):
    allowed = {Path(checkpoint_path).resolve()}
    reads = []
    original_load = torch.load

    def guarded(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved not in allowed:
            raise AssertionError('Forbidden resume input: ' + str(resolved))
        reads.append(str(resolved))
        return original_load(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Image/video/NumPy source reads forbidden during resume')

    with patch('torch.load', side_effect=guarded), patch('cv2.VideoCapture', side_effect=forbidden), \
         patch('cv2.imread', side_effect=forbidden), patch('PIL.Image.open', side_effect=forbidden), \
         patch('numpy.load', side_effect=forbidden):
        yield allowed, reads


def validate_state(packet, asset):
    required = ['state', 'current_vertices', 'canonical_control_reference', 'bound_vertex_index', 'bound_depth',
                'focal', 'camera_rotation', 'constraint_weight', 'steps_completed', 'asset_path', 'asset_sha256',
                'model_path', 'model_sha256', 'format_version']
    if any(key not in packet for key in required) or packet['format_version'] != 1:
        raise ValueError('Not a complete supported continuation checkpoint')
    n = len(packet['canonical_control_reference'])
    vertices = packet['current_vertices']
    if vertices.shape != asset['mesh_vertices'].shape or packet['state']['position'].shape != (1, n, 2):
        raise ValueError('Continuation shape mismatch')
    for key in ['bound_vertex_index', 'bound_depth', 'constraint_weight']:
        if packet[key].shape != (n,):
            raise ValueError('Binding shape mismatch')
    index = packet['bound_vertex_index']
    if index.dtype not in [torch.int32, torch.int64] or (index < 0).any() or (index >= len(vertices)).any():
        raise ValueError('Invalid bound vertex indices')
    if packet['focal'] <= 0 or not np.isfinite(packet['focal']) or packet['camera_rotation'].shape != (3, 3):
        raise ValueError('Invalid camera lift')
    if (packet['bound_depth'] <= 0).any() or (packet['constraint_weight'] < 0).any():
        raise ValueError('Invalid constraint depth/weights')
    for value in list(packet['state'].values()) + [vertices, packet['bound_depth'], packet['constraint_weight'], packet['camera_rotation']]:
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite persistent inference state')


@torch.no_grad()
def advance(model, asset, packet, steps):
    """No file or image input. Advance the retained state and connected support."""
    if steps < 1:
        raise ValueError('Positive continuation horizon required')
    validate_state(packet, asset)
    state = {key: value.clone() for key, value in packet['state'].items()}
    reference = packet['canonical_control_reference']
    rest = asset['mesh_vertices'].numpy()
    solver = ARAPMesh(rest, asset['mesh_faces'].numpy())
    previous = packet['current_vertices'].numpy().copy()
    index = packet['bound_vertex_index'].numpy()
    depth = packet['bound_depth'].numpy()
    rotation = packet['camera_rotation'].numpy()
    node_weight = packet['constraint_weight'].numpy()
    vertices, controls = [], []
    for _ in range(steps):
        state = model.step(state)
        displacement = (state['position'][0] - reference).numpy() * 480
        world = np.column_stack((displacement * depth[:, None] / packet['focal'], np.zeros(len(index)))) @ rotation
        target = rest.copy()
        weight = np.zeros(len(rest), np.float64)
        accum = np.zeros_like(rest, dtype=np.float64)
        np.add.at(accum, index, (rest[index] + world) * node_weight[:, None])
        np.add.at(weight, index, node_weight)
        active = weight > 0
        target[active] = accum[active] / weight[active, None]
        previous = solver.solve(target, weight, stiffness=6., iterations=8, prior=.002, initial=previous)
        vertices.append(torch.from_numpy(previous.copy()))
        controls.append(state['position'][0].clone())
    # Drop optional smoke trajectories before constructing a new fixed-size state.
    updated = {key: value for key, value in packet.items() if key not in ['generated_vertices', 'generated_controls', 'resume_input_reads']}
    updated.update(state=state, current_vertices=vertices[-1].clone(), steps_completed=packet['steps_completed'] + steps)
    return updated, torch.stack(vertices), torch.stack(controls)


@torch.no_grad()
def resume(checkpoint_path, steps=2):
    """Only three allowlisted tensor files; no observed seed or past trajectory."""
    with resume_input_guard(checkpoint_path) as (allowed, reads):
        packet = load_verified(checkpoint_path)
        asset_path, model_path = Path(packet['asset_path']), Path(packet['model_path'])
        allowed.update([asset_path.resolve(), model_path.resolve()])
        if digest(asset_path) != packet['asset_sha256'] or digest(model_path) != packet['model_sha256']:
            raise ValueError('Resume fixed asset/model hash mismatch')
        asset, weights = load_verified(asset_path), load_verified(model_path)
        model = load_model(weights)
        updated, vertices, controls = advance(model, asset, packet, steps)
    return updated, vertices, controls, reads


@torch.no_grad()
def build_and_smoke(root=ROOT, steps=2, forecast_mode='learned'):
    checkpoint_path, smoke_path = root / 'continuation.pt', root / 'resume_smoke.pt'
    if checkpoint_path.exists() or smoke_path.exists():
        raise FileExistsError('Preserve immutable continuation artifacts')
    forecast_path = root / 'forecast.pt'
    forecast = load_verified(forecast_path)
    original_forecast_sha = digest(forecast_path)
    seed = load_verified(SEED)
    if digest(SEED) != forecast['seed_sha256']:
        raise ValueError('Observed seed changed since forecast')
    asset_path = Path(forecast['asset_path']).resolve()
    model_path, expected_model_sha = forecast_model_provenance(forecast, forecast_mode)
    if digest(asset_path) != forecast['asset_sha256'] or digest(model_path) != expected_model_sha:
        raise ValueError('Forecast dependencies changed')
    asset, weights = load_verified(asset_path), load_verified(model_path)
    model = load_model(weights)
    observed = seed['controls']['position'][None]
    adjacency = seed['controls']['adjacency'][None]
    confidence = seed['controls']['confidence'][2][None]
    state = model.initialize(observed, adjacency, confidence)
    completed = len(forecast['controls'][forecast_mode]) - 1
    for _ in range(completed):
        state = model.step(state)
    control_error = float((state['position'][0] - forecast['controls'][forecast_mode][-1]).abs().max())
    if not torch.equal(state['position'][0], forecast['controls'][forecast_mode][-1]):
        raise AssertionError(f'Rebuilt recurrent state disagrees with saved controls: {control_error}')
    stored_state_error = None
    if forecast_mode in forecast.get('final_state', {}):
        stored = forecast['final_state'][forecast_mode]
        if set(stored) != set(state):
            raise AssertionError('Saved recurrent state keys differ from independently rebuilt state')
        stored_state_error = max(float((stored[key]-state[key]).abs().max()) for key in state)
        if stored_state_error != 0:
            raise AssertionError('Saved recurrent state differs from observed-seed replay')
        state = {key:value.clone() for key,value in stored.items()}
    index, depth, focal, rotation = control_binding(asset['mesh_vertices'], forecast['camera'], observed[0, 2])
    if not torch.equal(torch.from_numpy(index), forecast['control_vertex_index']):
        raise AssertionError('Canonical control binding changed')
    packet = dict(format_version=1, state={key:value.clone() for key,value in state.items()},
                  current_vertices=forecast['vertices'][forecast_mode][-1].clone(), canonical_control_reference=observed[0,2].clone(),
                  bound_vertex_index=torch.from_numpy(index), bound_depth=torch.from_numpy(depth), focal=focal,
                  camera_rotation=torch.from_numpy(rotation), constraint_weight=confidence[0].clamp_min(.05) * 10,
                  steps_completed=completed, fps=float(forecast['fps']),
                  asset_path=str(asset_path), model_path=str(model_path), asset_sha256=digest(asset_path), model_sha256=digest(model_path),
                  source_forecast_sha256=original_forecast_sha, forecast_mode=forecast_mode,
                  model_architecture=weights.get('architecture','damped_velocity_v1'),
                  classification='Persistent conditional projected-motion recurrent state with inferred XYZ support; quality gate FAILED; no learned depth.')
    # Reference continuation starts from the same saved endpoint state/mesh.
    # The resumed branch below has no access to the seed or reference result.
    reference, reference_vertices, reference_controls = advance(model, asset, packet, steps)
    checkpoint_sha = save_inference_checkpoint(packet, checkpoint_path)
    resumed, resumed_vertices, resumed_controls, reads = resume(checkpoint_path, steps)
    state_error = max(float((resumed['state'][key] - reference['state'][key]).abs().max()) for key in reference['state'])
    vertex_error = float((resumed_vertices - reference_vertices).abs().max())
    continued_control_error = float((resumed_controls - reference_controls).abs().max())
    if max(state_error, vertex_error, continued_control_error) != 0:
        raise AssertionError('Persistent continuation differs from in-memory reference')
    resumed.update(generated_vertices=resumed_vertices, generated_controls=resumed_controls, resume_input_reads=reads)
    smoke_sha = save_inference_checkpoint(resumed, smoke_path)
    if digest(forecast_path) != original_forecast_sha:
        raise AssertionError('Original forecast modified')
    report = dict(continuation_sha256=checkpoint_sha, smoke_sha256=smoke_sha, forecast_sha256=original_forecast_sha,
                  forecast_mode=forecast_mode, model_architecture=packet['model_architecture'],
                  steps_completed_before=completed, smoke_new_steps=steps, steps_completed_after=resumed['steps_completed'],
                  reconstructed_seed_to_forecast_control_max_error=control_error, resume_state_max_error=state_error,
                  saved_final_state_seed_replay_max_error=stored_state_error,
                  resume_vertices_max_error=vertex_error, resume_controls_max_error=continued_control_error,
                  resume_inputs=reads, resume_tensor_file_count=len(reads), resume_seed_reads=0, resume_image_reads=0,
                  resume_past_trajectory_reads=0, future_reference_reads=0, original_forecast_immutable=True,
                  checkpoint_bytes=checkpoint_path.stat().st_size,
                  recurrent_state_tensor_bytes=sum(value.numel()*value.element_size() for value in packet['state'].values()),
                  checkpoint_tensor_bytes=continuation_tensor_bytes(packet),
                  continued_state_tensor_bytes=continuation_tensor_bytes(resumed),
                  continuation_state_size_constant=continuation_tensor_bytes(packet)==continuation_tensor_bytes(resumed),
                  source_sha256=digest(Path(__file__)),
                  limits='Exact continuation on this software/hardware. No quality, physical accuracy, indefinite stable rollout or memory-compression claim. ARAP workspace/factorization and fixed asset/model excluded from packet tensor bytes. New steps are smoke-only, not rendered success.')
    (root / 'resume_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--forecast-mode', default='learned')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with keep_windows_awake():
        build_and_smoke(args.root, args.steps, args.forecast_mode)
