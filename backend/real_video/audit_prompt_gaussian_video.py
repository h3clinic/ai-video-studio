"""Independent post-generation provenance/replay audit; never runs Wan.

CPU provenance checks are default. Pass --repeat-decoder only after the active
generation process has released the GPU. The old latent is read exclusively
after generation to test non-identity, never supplied to the learned decoder.
"""
import argparse
import builtins
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

from .checkpoint_io import digest, load_verified
from .generate_wan_gaussian import generate
from .prompt_gaussian_video import inference_input_guard, validate_text_packet
from .wan_baseline import MODEL, REVISION, PROMPT, NEGATIVE


EXPECTED_DECODER_SHA = 'b481441b7875fcba2456a22556ba7a0aa6452d13990bbc36326ad497c147a148'
EXPECTED_TEXT_SHA = 'cad060ad14116ee451b381a8dc3ce0014feebb8baa41b17352f887305f56dc6f'


def verified_packet(path):
    path = Path(path)
    if not path.with_suffix(path.suffix + '.sha256.json').is_file():
        raise AssertionError(f'Missing checkpoint digest sidecar: {path}')
    return load_verified(path)


def decode_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f'Cannot decode exported video: {path}')
    frames = []
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return np.stack(frames), fps


def main(args):
    torch.set_num_threads(4)
    folder = args.run_dir.resolve()
    output = args.out or folder / 'independent_generation_audit.json'
    if output.exists():
        raise FileExistsError(f'Preserve prior audit: {output}')
    metrics = json.loads((folder / 'metrics.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    config = metrics['config']
    assert config == protocol['config']
    assert metrics['inputs'] == protocol['inputs']
    assert config['model'] == MODEL and config['revision'] == REVISION
    assert config['seed'] == 531002 and config['steps'] == 50
    assert config['prompt'] == PROMPT and config['negative_prompt'] == NEGATIVE
    assert (config['height'], config['width'], config['frames'], config['fps']) == (480, 832, 33, 16)
    for name in ('input_image', 'input_video', 'old_scene_latent_reused', 'original_RGB_VAE_used'):
        assert config[name] is False
    decoder_path = Path(metrics['inputs']['gaussian_decoder']['path']).resolve()
    text_path = Path(metrics['inputs']['text_embeddings']['path']).resolve()
    assert digest(decoder_path) == metrics['inputs']['gaussian_decoder']['sha256'] == EXPECTED_DECODER_SHA
    assert digest(text_path) == metrics['inputs']['text_embeddings']['sha256'] == EXPECTED_TEXT_SHA
    verified_transformer_files = {}
    for recorded_path, expected_hash in metrics['inputs']['transformer_files'].items():
        weight_path = Path(recorded_path).resolve()
        assert weight_path.is_relative_to(args.model_dir.resolve() / 'transformer')
        actual_hash = digest(weight_path)
        assert actual_hash == expected_hash
        verified_transformer_files[str(weight_path)] = actual_hash
    assert sum(path.endswith('.safetensors') for path in verified_transformer_files) == 2
    text = verified_packet(text_path)
    validate_text_packet(text, PROMPT, NEGATIVE)
    del text
    checkpoint = verified_packet(decoder_path)
    assert checkpoint['model_revision'] == REVISION and checkpoint['selected_step'] == 5500
    assert checkpoint['status'] == 'trained_small_real_video_decoder_pilot_not_quality_matched'
    assert sum(value.numel() for value in checkpoint['model'].values()) == 286793
    latent_path = folder / 'generated_latent.pt'
    new_latent = verified_packet(latent_path)
    old_latent_path = args.old_latent.resolve()
    old_latent = verified_packet(old_latent_path)
    assert new_latent['config'] == config
    assert old_latent['config']['seed'] == 421001
    assert tuple(new_latent['latent'].shape) == (1, 16, 9, 60, 104)
    assert new_latent['latent'].shape == old_latent['latent'].shape
    assert not torch.equal(new_latent['latent'], old_latent['latent'])
    delta = new_latent['latent'].float() - old_latent['latent'].float()
    latent_comparison = dict(new_path=str(latent_path), new_sha256=digest(latent_path),
        old_path=str(old_latent_path), old_sha256=digest(old_latent_path),
        new_seed=531002, old_seed=421001, tensor_exactly_equal=False,
        mean_absolute_difference=delta.abs().mean().item(),
        maximum_absolute_difference=delta.abs().max().item(),
        purpose='POSTHOC non-identity check only; old latent never supplied to decoder replay')
    assert latent_comparison['new_sha256'] == metrics['latent_sha256']
    del old_latent, delta
    fields_path = folder / 'gaussian_fields.pt'
    saved_fields = verified_packet(fields_path)
    assert saved_fields['config'] == config
    assert saved_fields['checkpoint_sha256'] == EXPECTED_DECODER_SHA
    assert saved_fields['latent_sha256'] == metrics['latent_sha256']
    assert digest(fields_path) == metrics['gaussian_fields_sha256']
    assert tuple(saved_fields['fields'].shape) == (1, 9, 33, 120, 208)
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines() if line]
    steps = [event['step'] for event in events if event['stage'] == 'fresh_noise_denoising']
    assert steps == list(range(1, 51))
    assert events[-1]['stage'] == 'complete'
    executed_runner_path = folder / 'runner.py'
    current_runner_path = Path(__file__).with_name('prompt_gaussian_video.py')
    executed = executed_runner_path.read_text()
    current = current_runner_path.read_text()
    fixes = dict(
        executed_snapshot_has_read_write_guard_fix="if '+' not in str(mode)" in executed,
        current_runner_has_read_write_guard_fix="if '+' not in str(mode)" in current,
        executed_snapshot_checks_decoder_revision="checkpoint.get('model_revision')" in executed,
        current_runner_checks_decoder_revision="checkpoint.get('model_revision')" in current,
        selected_decoder_revision_independently_verified=True,
        explanation='Fresh run began before these two defensive checks were added. Its inspected code path reads approved files in read-only mode; no read-write bypass is used. The selected checkpoint revision matches. Current-runner tests are not misattributed to the executed snapshot.')
    report = dict(audit_scope='Fresh-noise provenance and optional deterministic Gaussian decoder replay; no generative quality or efficiency claim',
        cpu_provenance_pass=True, config=config, decoder_sha256=EXPECTED_DECODER_SHA,
        cached_text_sha256=EXPECTED_TEXT_SHA, latent_comparison=latent_comparison,
        transformer_manifest_independently_rehashed=verified_transformer_files,
        fifty_denoising_events_verified=True, decoder_parameters=286793,
        selected_decoder_step=5500, decoder_training_scope='UCF reconstruction-trained planar decoder; no new Wan denoiser training',
        gaussian_fields_shape=list(saved_fields['fields'].shape), gaussians_per_frame=24960,
        source_input_guard_limit='Python API guard plus code-path review, not OS sandbox or proof against arbitrary native IO',
        snapshot=dict(executed_path=str(executed_runner_path), executed_sha256=digest(executed_runner_path),
                      current_path=str(current_runner_path), current_sha256=digest(current_runner_path), **fixes),
        helper_sha256=digest(__file__),
        limitations=['Not 3D; centers have only XY coordinates.',
                     'Each fixed 4-pixel cell confines its center to +/-1.8 pixels; indices are not tracked physical cat points.',
                     'No recurrent persistent scene dynamics or acceleration checkpoint is attached.',
                     'Entire video latent and Gaussian field sequence are decoded jointly.',
                     'No matched-quality computation or memory savings demonstrated.'])

    if args.repeat_decoder:
        assert torch.cuda.is_available()
        original_import = builtins.__import__

        def guarded_import(name, *rest, **kwargs):
            if name == 'diffusers' or name.startswith('diffusers.'):
                raise AssertionError('Diffusers / RGB VAE forbidden during decoder replay')
            return original_import(name, *rest, **kwargs)

        def forbidden_media(*unused, **kwargs):
            raise AssertionError('Source media/array input forbidden during decoder replay')

        loaded_before = sorted(name for name in sys.modules if name == 'diffusers' or name.startswith('diffusers.'))
        assert not loaded_before
        with inference_input_guard(args.model_dir, latent_path, decoder_path) as observed, ExitStack() as stack:
            stack.enter_context(patch('builtins.__import__', side_effect=guarded_import))
            for name in ('cv2.VideoCapture', 'PIL.Image.open', 'imageio.v2.imread',
                         'imageio.v2.mimread', 'imageio.v2.get_reader', 'numpy.load'):
                stack.enter_context(patch(name, side_effect=forbidden_media))
            # Only these two exact, hash-verified tensor packets enter replay.
            repeated_checkpoint = verified_packet(decoder_path)
            repeated_latent = verified_packet(latent_path)
            frames, fields, decoder_metrics = generate(repeated_checkpoint, repeated_latent)
        assert set(observed) == {str(latent_path), str(decoder_path)}
        field_max_difference = (fields - saved_fields['fields']).abs().max().item()
        torch.testing.assert_close(fields, saved_fields['fields'], atol=2e-5, rtol=0)
        assert tuple(frames.shape) == (33, 480, 832, 3)
        replay = dict(passed=True, forbidden_diffusers_import=True,
            forbidden_video_image_array_reads=True, observed_tensor_paths=sorted(set(observed)),
            maximum_repeated_field_difference=field_max_difference,
            raw_frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
            decoder_metrics=decoder_metrics,
            numerical_policy='Fields atol=2e-5; PNGs at most 1 code value and <=0.01% differing channels. GPU scatter accumulation may change rounding at 8-bit boundaries; bitwise replay is not assumed.')
        # These RGB reads occur AFTER generation, for output verification only.
        png_comparisons = []
        for index in (0, 8, 16, 24, 32):
            png = np.asarray(Image.open(folder / f'frame_{index:03d}.png').convert('RGB'))
            difference = np.abs(png.astype(np.int16) - frames[index].astype(np.int16))
            maximum = int(difference.max())
            fraction = float((difference != 0).mean())
            assert maximum <= 1 and fraction <= 1e-4
            png_comparisons.append(dict(index=index, maximum_code_difference=maximum,
                differing_channels=int((difference != 0).sum()), differing_fraction=fraction))
        replay['saved_png_comparisons'] = png_comparisons
        replay['five_saved_png_frames_match_replay_exactly'] = all(item['maximum_code_difference'] == 0 for item in png_comparisons)
        with tempfile.TemporaryDirectory(prefix='gaussian-replay-') as temporary:
            repeated_video = Path(temporary) / 'repeat.mp4'
            imageio.mimwrite(repeated_video, frames, fps=16, codec='libx264', quality=8, macro_block_size=1)
            encoded_frames, repeated_fps = decode_video(repeated_video)
            output_frames, output_fps = decode_video(folder / 'orange_cat_gaussian.mp4')
            assert encoded_frames.shape == output_frames.shape == (33, 480, 832, 3)
            difference = encoded_frames.astype(np.float32) - output_frames.astype(np.float32)
            rms = float(np.sqrt(np.square(difference).mean()))
            # This tolerance tests near-identity of two lossy encodings, not quality.
            encoded_tight_gate = rms <= .25
            assert repeated_fps == output_fps == 16.0
            replay['encoded_video_frame_pixels_match_replay_exactly'] = bool(np.array_equal(encoded_frames, output_frames))
            replay['encoded_replay_comparison'] = dict(rms_code_difference=rms,
                maximum_code_difference=float(np.abs(difference).max()),
                differing_fraction=float((difference != 0).mean()),
                tight_encoding_gate_pass=encoded_tight_gate,
                acceptance='RMS <=0.25 8-bit code values, same frame shape/count/FPS; numerical replay check only')
            replay['encoded_frame_count'] = len(output_frames)
            replay['encoded_video_bytewise_equal'] = digest(repeated_video) == digest(folder / 'orange_cat_gaussian.mp4')
        replay['fields_and_saved_png_numerical_gate_pass'] = True
        replay['passed'] = bool(encoded_tight_gate)
        replay['conclusion'] = ('Gaussian fields and saved PNGs replay within explicit numerical tolerances; '
            'exact pixel identity is separately recorded. The lossy-encoding tight numerical gate ' +
            ('passed.' if encoded_tight_gate else 'FAILED; no exact or fully gate-passing replay is claimed.'))
        report['independent_decoder_replay'] = replay
        torch.cuda.empty_cache()
    else:
        report['independent_decoder_replay'] = dict(passed=None, reason='Not requested; no GPU loaded')
    video_path = folder / 'orange_cat_gaussian.mp4'
    assert digest(video_path) == metrics['video_sha256']
    report['video'] = dict(path=str(video_path), sha256=digest(video_path),
                          size_bytes=video_path.stat().st_size, duration_seconds=33 / 16)
    first_attempt = folder / 'independent_audit_first_attempt.json'
    if first_attempt.exists():
        report['preserved_first_attempt'] = json.loads(first_attempt.read_text())
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(audit=str(output.resolve()), cpu_provenance_pass=True,
        decoder_replay=report['independent_decoder_replay']['passed'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/real_video/prompt_generation/orange_cat_seed_531002'))
    parser.add_argument('--old-latent', type=Path, default=Path('artifacts/real_video/wan_baseline/cat_seed_421001/generated_latent.pt'))
    parser.add_argument('--model-dir', type=Path, default=Path('../../work/wan21_13b'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--repeat-decoder', action='store_true')
    main(parser.parse_args())
