"""Bounded 3-candidate training/validation loop for causal projected motion.

Recorded DAVIS optical-flow targets are supervision, not true material tracks.
No final test set or Wan-cat diagnostic enters training/checkpoint selection.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .vector_motion_network import VectorMotionNetwork
from .checkpoint_io import load_verified, save_training_checkpoint, save_inference_checkpoint, digest, keep_windows_awake
from .prepare_animal_motion import WORK

OUT = Path('artifacts/real_video/true3d/learned_motion_loop/v2')
DATA = WORK / 'control_tracks_animal_only_v2.pt'


def sample(data, index, device):
    return {key: data[key][index].to(device) for key in ['position', 'adjacency', 'confidence', 'visibility']}


def initialize(model, values):
    return model.initialize(values['position'][:, :3], values['adjacency'], values['confidence'][:, 2])


def loss_for(prediction, values, regularization):
    target = values['position'][:, 3:]
    ref = values['position'][:, 2]
    scale = (ref.amax(1, keepdim=True) - ref.amin(1, keepdim=True)).amax(-1, keepdim=True).clamp_min(1e-5)
    weight = values['confidence'][:, 3:].clamp_min(.02) * (.1 + .9 * values['visibility'][:, 3:])
    residual = (prediction - target) / scale[:, None] * 32
    position = (residual.square().sum(-1) * weight).sum() / weight.sum()
    predicted_velocity = torch.diff(torch.cat((ref[:, None], prediction), 1), dim=1)
    target_velocity = torch.diff(values['position'][:, 2:], dim=1)
    velocity = (((predicted_velocity - target_velocity) / scale[:, None] * 32).square().sum(-1) * weight).sum() / weight.sum()
    # Supervised edge errors, not an invalid hard 3D-rigidity constraint on 2D.
    edges = residual[:, :, None] - residual[:, :, :, None]
    edge_weight = values['adjacency'][:, None] * weight[:, :, :, None] * weight[:, :, None, :]
    edge = (edges.square().sum(-1) * edge_weight).sum() / edge_weight.sum().clamp_min(1e-8)
    return position + .2 * velocity + regularization * edge


@torch.no_grad()
def evaluate(model, data, indices, device, baselines=False):
    model.eval()
    values = sample(data, indices, device)
    state = initialize(model, values)
    modes = ['learned', 'frozen', 'velocity', 'average_velocity', 'damped_velocity'] if baselines else ['learned']
    target = values['position'][:, 3:]
    confidence = values['confidence'][:, 3:] * values['visibility'][:, 3:]
    all_predictions = {}
    for mode in modes:
        current = initialize(model, values)
        predictions = []
        velocity = current['velocity']
        if mode == 'average_velocity':
            velocity = (values['position'][:, 2] - values['position'][:, 0]) / 2
        for _ in range(12):
            if mode == 'learned':
                current = model.step(current)
            elif mode != 'frozen':
                if mode == 'damped_velocity':
                    velocity = velocity * .95
                current = dict(current, position=current['position'] + velocity)
            predictions.append(current['position'])
        all_predictions[mode] = torch.stack(predictions, 1)
    if baselines:
        disabled = VectorMotionNetwork(**model.config).to(device)
        for parameter in disabled.parameters():
            parameter.zero_()
        all_predictions['zero_weights'], _ = disabled.rollout(values['position'][:, :3], values['adjacency'], 12, values['confidence'][:, 2])
        modes.append('zero_weights')
    metrics = {}
    for mode, prediction in all_predictions.items():
        error = ((prediction - target) / state['scale'][:, None] * 32).norm(dim=-1)
        visible_error = (error * confidence).sum(-1) / confidence.sum(-1).clamp_min(1e-8)
        # Edge error relative to observed length is a diagnostic, not 3D strain.
        current_edges = prediction[:, :, None] - prediction[:, :, :, None]
        target_edges = target[:, :, None] - target[:, :, :, None]
        edge_error = ((current_edges.norm(dim=-1) - target_edges.norm(dim=-1)) / state['scale'][:, None]).abs()
        edge_error = (edge_error * values['adjacency'][:, None]).sum((-1, -2)) / values['adjacency'].sum((-1, -2))[:, None]
        metrics[mode] = dict(ade=visible_error.mean(-1).cpu().tolist(), fde=visible_error[:, -1].cpu().tolist(),
                             all_ade=error.mean((-1, -2)).cpu().tolist(), edge_error=edge_error.mean(-1).cpu().tolist(),
                             horizon_epe=visible_error.cpu().tolist())
    sequences = sorted({data['metadata'][i]['sequence'] for i in indices})
    groups = {}
    for sequence in sequences:
        ids = [j for j, i in enumerate(indices) if data['metadata'][i]['sequence'] == sequence]
        groups[sequence] = {mode: {key: np.mean(np.array(value)[ids], axis=0).tolist() for key, value in metrics[mode].items()} for mode in modes}
    means = {mode: {key: np.mean([groups[seq][mode][key] for seq in sequences], axis=0).tolist() for key in metrics[mode]} for mode in modes}
    return dict(means=means, sequence_scores=groups, windows=len(indices))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--preset', choices=['original', 'adaptive_damping'], default='original')
    args = parser.parse_args()
    out = args.output or (OUT if args.preset == 'original' else OUT.parent / 'v3')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise FileExistsError('Preserve bounded-loop experiment; use a fresh output path')
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_verified(DATA)
    train = [i for i, item in enumerate(data['metadata']) if item['split'] == 'train']
    val = [i for i, item in enumerate(data['metadata']) if item['split'] == 'validation']
    train_sequences = sorted({data['metadata'][i]['sequence'] for i in train})
    val_sequences = sorted({data['metadata'][i]['sequence'] for i in val})
    assert train and val and not set(train_sequences).intersection(val_sequences)
    assert all(item['split'] in ['train', 'validation'] for item in data['metadata'])
    candidates = [dict(name='graph_residual64', hidden=64, residual_scale=.01, damping_init=.975, regularization=.05, seed=431001),
                  dict(name='graph_smooth64', hidden=64, residual_scale=.007, damping_init=.95, regularization=.3, seed=431002),
                  dict(name='graph_compact32', hidden=32, residual_scale=.005, damping_init=.98, regularization=.15, seed=431003)]
    if args.preset == 'adaptive_damping':
        candidates = [dict(name='damped_compact32', hidden=32, residual_scale=.002, damping_init=.75, regularization=.15, seed=431004),
                      dict(name='damped_residual64', hidden=64, residual_scale=.005, damping_init=.75, regularization=.15, seed=431005)]
    protocol = dict(candidates=candidates, steps_per_candidate=args.steps, batch=16, learning_rate=.0005,
                    evaluate_every=100, selected_by='minimum sequence-macro visible-confidence ADE on validation only',
                    seed_frames=3, predicted_frames=12, teacher_forcing=False, train_windows=len(train), validation_windows=len(val),
                    train_sequences=train_sequences, validation_sequences=val_sequences, data_sha256=digest(DATA),
                    source='DAVIS animal-only optical-flow pseudo-trajectories; not material or 3D ground truth',
                    normalization='observed frame2 bounding span only; adjacency from observed mask/frame2 only; isolated rows get self-edge',
                    augmentation='train-only x reflection and small random-walk noise on three observed positions',
                    gates=dict(ordinary='ADE <= 0.90 * best of frozen, velocity, average velocity, damped velocity; every sequence <= 1.10 * its best baseline; FDE <= best baseline FDE; edge_error <= 1.10 * best baseline edge_error',
                               tenfold='ADE <= 0.10 * strongest macro baseline ADE'),
                    test_sequences=0, no_cat_video_selection=True, learned_dimension=2,
                    claim='conditional future projected-motion generation; deterministic, not noise/text-to-video or learned hidden depth',
                    research_source='https://arxiv.org/html/2002.09405v1 section3 Euler updates and appendix rollout noise; engineering adaptation, not reproduction')
    if args.preset == 'adaptive_damping':
        protocol.update(preset='adaptive_damping', adaptation='After v2 failed, initialize at the stronger 0.75 damping baseline and learn smaller residuals. Only v2 validation feedback used; no cat diagnostic or final-test feedback.',
                        validation_reused_for_adaptation=True, validation_is_unbiased_final_test=False)
        protocol['gates']['ordinary'] += '; additionally learned ADE and FDE must be strictly lower than all-zero-weights (0.75 damping)'
        protocol['gates']['tenfold'] = 'ADE <= 0.10 * strongest macro baseline including all-zero-weights 0.75 damping'
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    loop_start = time.perf_counter()
    results = []
    with keep_windows_awake():
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for config in candidates:
            path = out / config['name']
            path.mkdir()
            torch.manual_seed(config['seed'])
            model = VectorMotionNetwork(**{key: config[key] for key in ['hidden', 'residual_scale', 'damping_init']}).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=.01)
            initial = evaluate(model, data, val, device, True)
            (path / 'initial.json').write_text(json.dumps(initial, indent=2))
            best = float('inf')
            history = []
            start = time.perf_counter()
            for step in range(1, args.steps + 1):
                model.train()
                indices = [train[i] for i in torch.randint(len(train), (protocol['batch'],)).tolist()]
                values = sample(data, indices, device)
                mirror = torch.rand(len(indices), device=device) < .5
                values['position'][mirror, :, :, 0] *= -1
                # Only observation inputs are corrupted; all targets stay exact.
                observed = values['position'][:, :3].clone()
                span = (observed[:, 2].amax(1) - observed[:, 2].amin(1)).amax(-1)
                noise = torch.randn_like(observed) * span[:, None, None, None] * .0005
                observed += noise.cumsum(1)
                prediction, _ = model.rollout(observed, values['adjacency'], 12, values['confidence'][:, 2])
                objective = loss_for(prediction, values, config['regularization'])
                if not torch.isfinite(objective):
                    raise ValueError('Nonfinite vector motion training')
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                if step % 100 == 0 or step == args.steps:
                    score = evaluate(model, data, val, device)
                    value = score['means']['learned']['ade']
                    improved = value < best
                    best = min(best, value)
                    save_training_checkpoint(dict(model={key: val.cpu() for key, val in model.state_dict().items()},
                                                  optimizer=optimizer.state_dict(), config=model.config, step=step, score=score,
                                                  torch_rng=torch.get_rng_state(), candidate=config), path, step, improved)
                    history.append(dict(step=step, loss=float(objective.detach()), validation=score, seconds=time.perf_counter() - start))
                    (path / 'history.json').write_text(json.dumps(history, indent=2))
                    progress = dict(candidate=config['name'], step=step, total=args.steps, validation_ade=value, best=best, seconds=time.perf_counter() - loop_start)
                    (out / 'progress.json').write_text(json.dumps(progress, indent=2))
                    print(json.dumps(progress), flush=True)
            selected = load_verified(path / 'best.pt')
            model.load_state_dict(selected['model'])
            scores = evaluate(model, data, val, device, True)
            checkpoint = dict(model=selected['model'], config=model.config, model_config=model.config, candidate=config, selected_step=selected['step'],
                              data_sha256=digest(DATA), protocol_sha256=digest(out / 'protocol.json'), learned_dimension=2,
                              inference_contract='initialize(three observed XY control positions, observed adjacency, optional observed confidence); then step(state) without targets')
            sha = save_inference_checkpoint(checkpoint, path / 'model.pt')
            results.append(dict(name=config['name'], selected_step=selected['step'], scores=scores, sha256=sha,
                                seconds=time.perf_counter() - start, parameters=sum(p.numel() for p in model.parameters())))
            (path / 'results.json').write_text(json.dumps(results[-1], indent=2))
        selected_result = min(results, key=lambda item: item['scores']['means']['learned']['ade'])
        selected = load_verified(out / selected_result['name'] / 'model.pt')
        selected_sha = save_inference_checkpoint(selected, out / 'model.pt')
        means = selected_result['scores']['means']
        baseline_modes = ['frozen', 'velocity', 'average_velocity', 'damped_velocity']
        if args.preset == 'adaptive_damping':
            baseline_modes.append('zero_weights')
        baseline = min(baseline_modes, key=lambda mode: means[mode]['ade'])
        ratio = means['learned']['ade'] / means[baseline]['ade']
        sequence_ratios = {seq: item['learned']['ade'] / min(item[mode]['ade'] for mode in baseline_modes)
                           for seq, item in selected_result['scores']['sequence_scores'].items()}
        gates = dict(macro_ade_10percent=ratio <= .9, no_sequence_10percent_regression=max(sequence_ratios.values()) <= 1.1,
                     fde_no_regression=means['learned']['fde'] <= min(means[m]['fde'] for m in baseline_modes),
                     edge_error_no_10percent_regression=means['learned']['edge_error'] <= 1.1 * min(means[m]['edge_error'] for m in baseline_modes),
                     tenfold_ade=ratio <= .1)
        if args.preset == 'adaptive_damping':
            gates['learned_beats_disabled_ade'] = means['learned']['ade'] < means['zero_weights']['ade']
            gates['learned_beats_disabled_fde'] = means['learned']['fde'] < means['zero_weights']['fde']
        report = dict(selected_candidate=selected_result['name'], selected_step=selected_result['selected_step'],
                      selected_weights_sha256=selected_sha, best_macro_baseline=baseline, ade_ratio_to_best_baseline=ratio,
                      per_sequence_ratio_to_best_baseline=sequence_ratios, gates=gates,
                      ordinary_gates_pass=all(v for k, v in gates.items() if k != 'tenfold_ade'),
                      loop_seconds=time.perf_counter() - loop_start,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                      candidates=results, learned_depth=False, text_to_video=False, test_sequences=0,
                      quality_claim='Validation-control forecast only; no RGB/3D quality or 10x improvement implied',
                      source_files_sha256={name: digest(Path('real_video') / name) for name in ['vector_motion_network.py', 'train_vector_motion_loop.py']})
        (out / 'selection.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != 'candidates'}), flush=True)


if __name__ == '__main__':
    main()
