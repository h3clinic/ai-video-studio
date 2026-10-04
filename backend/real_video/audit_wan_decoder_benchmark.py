"""CPU-only arithmetic/provenance audit of the fixed paired decoder benchmark."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, atol=1e-9):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=atol), (actual, expected)


def main(args):
    root = args.root.resolve()
    out = root / 'independent_result_audit.json'
    if out.exists():
        raise FileExistsError('Preserve previous independent audit')
    summary = read(root / 'summary.json')
    protocol = read(root / 'protocol.json')
    original = read(root.parent / 'metrics.json')
    assert summary['protocol'] == protocol
    assert protocol['order'] == ['wan', 'gaussian', 'gaussian', 'wan']
    assert protocol['fresh_process_each'] and not protocol['quality_matched']
    assert protocol['dtype'] == 'float32'
    assert protocol['latent_sha256'] == original['latent_sha256'] == digest(root.parent / 'generated_latent.pt')
    vae_hashes = {str(Path(path).resolve()): digest(path) for path in protocol['vae_input_hashes']}
    for path, expected in protocol['vae_input_hashes'].items():
        assert vae_hashes[str(Path(path).resolve())] == expected
    records = [read(root / f'{index}_{mode}' / 'metrics.json')
               for index, mode in enumerate(protocol['order'])]
    for index, record in enumerate(records):
        assert record['mode'] == protocol['order'][index]
        assert record['latent_sha256'] == protocol['latent_sha256']
        assert record['config'] == original['config']
        assert record['settings']['dtype'] == 'float32'
        assert record['cpu_threads'] == 4
        assert [row['phase'] for row in record['trials']] == ['cold', 'warm', 'warm']
        assert [row['repeat'] for row in record['trials']] == [0, 1, 2]
        expected_parameters = 73295603 if record['mode'] == 'wan' else 286793
        expected_loaded_parameters = 126892531 if record['mode'] == 'wan' else 286793
        assert record['executed_decoder_parameter_count'] == expected_parameters
        assert record['all_loaded_parameter_count'] == expected_loaded_parameters
        assert record['parameter_bytes'] == expected_loaded_parameters * 4
        for row in record['trials']:
            assert row['seconds'] > 0
            assert row['peak_cuda_reserved_bytes'] >= row['peak_cuda_allocated_bytes']
            assert row['peak_cuda_allocated_bytes'] >= row['resident_cuda_allocated_bytes']
            assert row['incremental_peak_cuda_allocated_bytes'] == row['peak_cuda_allocated_bytes'] - row['resident_cuda_allocated_bytes']
        residents = [row['resident_cuda_allocated_bytes'] for row in record['trials']]
        assert residents[1] == residents[2], ('GPU resident allocation grows across warm repeats', residents)
    assert len({record['gpu'] for record in records}) == 1
    assert len({record['tf32_matmul'] for record in records}) == 1
    assert len({record['cudnn_benchmark'] for record in records}) == 1
    independent = {}
    denoise = original['timings']['fresh_denoise_seconds']
    denoise_peak = original['denoise_peak_cuda_allocated_bytes']
    for mode in ('wan', 'gaussian'):
        rows = [row for record in records if record['mode'] == mode for row in record['trials']]
        warm = [row['seconds'] for row in rows if row['phase'] == 'warm']
        assert len(warm) == 4
        median = statistics.median(warm)
        peak = max(row['peak_cuda_allocated_bytes'] for row in rows)
        expected = summary['measured'][mode]
        close(expected['warm_median_seconds'], median)
        close(expected['warm_min_seconds'], min(warm))
        close(expected['warm_max_seconds'], max(warm))
        assert expected['peak_cuda_allocated_bytes'] == peak
        close(expected['estimated_common_denoise_plus_warm_decode_seconds'], denoise + median)
        assert expected['estimated_staged_pipeline_peak_cuda_allocated_bytes'] == max(denoise_peak, peak)
        independent[mode] = dict(warm_seconds=warm, warm_median_seconds=median,
            peak_cuda_allocated_bytes=peak, common_denoise_plus_decode_seconds=denoise + median,
            staged_peak_cuda_allocated_bytes=max(denoise_peak, peak))
    a, b = independent['wan'], independent['gaussian']
    ratios = dict(decoder_speedup=a['warm_median_seconds'] / b['warm_median_seconds'],
        decoder_cuda_peak_reduction_percent=100 * (1 - b['peak_cuda_allocated_bytes'] / a['peak_cuda_allocated_bytes']),
        estimated_common_denoise_decode_time_reduction_percent=100 * (1 - b['common_denoise_plus_decode_seconds'] / a['common_denoise_plus_decode_seconds']),
        estimated_staged_cuda_peak_reduction_percent=100 * (1 - b['staged_peak_cuda_allocated_bytes'] / a['staged_peak_cuda_allocated_bytes']))
    for name, value in ratios.items():
        close(summary['ratios'][name], value)
    wa = np.load(root / '0_wan' / 'uncompressed_rgb.npy', mmap_mode='r')
    ga = np.load(root / '1_gaussian' / 'uncompressed_rgb.npy', mmap_mode='r')
    assert wa.shape == ga.shape == (33, 480, 832, 3)
    assert wa.dtype == ga.dtype == np.uint8
    mse_sum = absolute_sum = 0.0
    for index in range(len(wa)):
        delta = (wa[index].astype(np.float64) - ga[index].astype(np.float64)) / 255
        mse_sum += float(np.square(delta).sum())
        absolute_sum += float(np.abs(delta).sum())
    psnr = -10 * math.log10(mse_sum / wa.size)
    mae = absolute_sum / wa.size
    close(summary['quality']['agreement_psnr_db'], psnr, atol=1e-4)
    close(summary['quality']['agreement_mae'], mae, atol=1e-6)
    storage = summary['storage']
    assert storage['wan_latent_file_bytes'] == (root.parent / 'generated_latent.pt').stat().st_size
    assert storage['gaussian_fields_file_bytes'] == (root.parent / 'gaussian_fields.pt').stat().st_size
    assert storage['raw_rgb_uint8_tensor_bytes'] == wa.nbytes
    for index, mode in ((0, 'wan'), (1, 'gaussian')):
        video = root / f'{index}_{mode}' / f'{mode}.mp4'
        assert digest(video) == records[index]['video_sha256']
        assert video.stat().st_size == records[index]['video_bytes'] == storage[f'{mode}_mp4_bytes']
    audit = dict(status='paired_arithmetic_and_provenance_pass', cpu_only=True,
        summary_sha256=digest(root / 'summary.json'), executed_runner_sha256=digest(root / 'runner.py'),
        verified_vae_hashes=vae_hashes, independent_measurements=independent, independent_ratios=ratios,
        uncompressed_decoder_agreement=dict(psnr_db=psnr, mae=mae, interpretation='Not ground-truth quality'),
        memory_lifetime_check='Source deletes each previous GPU output; resident allocation is stable across the two warm trials per worker. Cold-to-warm changes are preserved separately and may reflect lazy runtime initialization.',
        resident_allocation_by_worker=[dict(mode=record['mode'],
            bytes=[row['resident_cuda_allocated_bytes'] for row in record['trials']],
            cold_to_warm_increase_bytes=record['trials'][1]['resident_cuda_allocated_bytes'] - record['trials'][0]['resident_cuda_allocated_bytes']) for record in records],
        rss_sampling=dict(requested_interval_seconds=.005,
            realized_average_interval_estimates_seconds=[row['seconds'] / max(row['rss_samples'] - 1, 1)
                for record in records for row in record['trials']],
            note='Elapsed decode time divided by sample intervals is approximate; OS scheduling can make actual intervals longer than 5ms. Observed RSS maxima are not guaranteed process peaks.'),
        required_qualifications=[
            'FPS/latency is this laptop/configuration wall-time, not measured FLOPs or energy.',
            'Denoising remains unchanged; combined times are component-based estimates, not paired complete generations.',
            'Pipeline GPU peak is staged maximum, not sum. Decoder allocation savings need not lower pipeline peak.',
            'Original FP32 tiled VAE includes an unused resident encoder; not fastest/lowest-memory achievable Wan.',
            'Image detail/quality is unequal; no quality-matched superiority claim.',
            'CPU RSS includes libraries and the retained exported RGB array during first-worker warm trials; not isolated model-only or whole-pipeline RAM. Sampling requests5ms but actual OS-dependent intervals are longer and peaks can be missed.',
            'Windows working-set residency and system memory pressure can vary substantially between fresh processes; no host-RAM saving is established.',
            'Warm reserved CUDA can include cached allocations from excluded postprocessing; allocated peaks are the cleaner comparison.',
            'Power/thermal state is recorded but uncontrolled; four warm observations per mode are descriptive, not statistical generalization.',
            'Stored Gaussian fields remain larger than original latent or compressed video, and are not persistent 3D memory.',
            'Worker config is copied input-latent provenance; its original_RGB_VAE_used=false refers to earlier Gaussian generation, not the Wan benchmark branch. Worker mode/settings identify the decoder actually executed.'
        ])
    out.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(dict(audit=str(out), status=audit['status'], ratios=ratios)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/real_video/prompt_generation/orange_cat_seed_531002/paired_decoder_benchmark_v1'))
    main(parser.parse_args())
