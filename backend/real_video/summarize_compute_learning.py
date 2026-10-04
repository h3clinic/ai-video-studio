"""Summarize measured distillation costs without claiming generative-quality gains."""
import argparse
import json
import math
from pathlib import Path
import shutil
from .checkpoint_io import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out
    training=json.loads((out/'guidance_student.training.json').read_text(encoding='utf-8'))
    audit=json.loads((out/'audit/audit.json').read_text(encoding='utf-8'))
    benchmarks={name:json.loads((out/'benchmarks'/f'{name}.json').read_text(encoding='utf-8'))
                for name in ('teacher_50','student_50','teacher_25','student_25','teacher_graph_50','student_graph_50')}
    teacher=benchmarks['teacher_50']; rows=[]; comparisons={}
    # V1's final timing is measured, but its initial cache generation was not persisted.
    previous_training=164.01248959999975
    experiment_lower_bound=training['total_extra_compute_seconds']+previous_training
    for name,b in benchmarks.items():
        delta=teacher['median_seconds']-b['median_seconds']
        q=audit['aggregates'].get(name.replace('_graph',''))
        comparison=dict(seconds=b['median_seconds'],speedup=teacher['median_seconds']/b['median_seconds'],
                        time_reduction_percent=100*delta/teacher['median_seconds'],
                        peak_mib=b['peak_cuda_allocated_bytes']/2**20,
                        peak_reduction_percent=100*(1-b['peak_cuda_allocated_bytes']/teacher['peak_cuda_allocated_bytes']),
                        gmac=b['module_macs']/1e9,mac_reduction_percent=100*(1-b['module_macs']/teacher['module_macs']),
                        quality=q)
        if name.startswith('student') and delta>0:
            comparison['v2_only_break_even_clips_excluding_original_cache']=math.ceil(training['total_extra_compute_seconds']/delta)
            comparison['whole_experiment_break_even_clips_lower_bound']=math.ceil(experiment_lower_bound/delta)
        comparisons[name]=comparison
        psnr=f"{q['mean_teacher_agreement_psnr']:.2f}" if q else 'reference'
        rows.append(f"| {name} | {b['median_seconds']*1000:.1f} | {comparison['peak_mib']:.2f} | {comparison['gmac']:.3f} | {psnr} |")
    summary=dict(training=training,audit_aggregates=audit['aggregates'],comparisons=comparisons,
                 additional_experiment_seconds_lower_bound=experiment_lower_bound,
                 missing_cost='V1 teacher-trajectory cache generation time; imports and experiment administration excluded')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    primary=comparisons['student_50']; quality=primary['quality']; baseline=audit['aggregates']['untrained_single_50']
    graph=comparisons['teacher_graph_50']; graph_benchmark=benchmarks['teacher_graph_50']
    graph_setup=graph_benchmark['graph_setup_seconds']
    graph_payback=math.ceil(graph_setup/(teacher['median_seconds']-graph['seconds']))
    lines=[
        '# Learned Gaussian generation: computation experiment',
        '',
        '## Outcome',
        '',
        f"The useful same-output improvement is CUDA-graph execution of the ORIGINAL generator: "
        f"{teacher['median_seconds']*1000:.1f} -> {graph['seconds']*1000:.1f} ms/clip "
        f"({graph['speedup']:.2f}x faster), with unchanged weights and arithmetic. "
        f"It raises peak allocated tensor memory from {comparisons['teacher_50']['peak_mib']:.2f} "
        f"to {graph['peak_mib']:.2f} MiB. Setup took {graph_setup:.3f} s, amortized after about "
        f"{graph_payback} repeated clips, excluding common model loading. A one-shot CLI pays setup each run.",
        '',
        'Graph execution caches GPU operations, NOT generated video, source frames or diffusion results. '
        'Fresh noise, timestep and class labels are copied into the graph on each evaluation. '
        'Three benchmark cases (including two changed seed/label inputs) matched eager rendered pixels '
        'within 2.4e-7. Graph setup is excluded from warmed times but its persistent buffers are retained '
        'in memory measurements. Reserved CUDA memory also rises; see the JSON results.',
        '',
        'The learned student reduces counted arithmetic but FAILED the predeclared worst-case fidelity '
        'gate. Keep it experimental; do not replace the original checkpoint or claim unchanged video quality. '
        'No-training execution acceleration and learned-model approximation are separate findings.',
        '',
        f"Selected student: update {training['selected_step']}; validation guided-velocity MSE "
        f"{training['initial_validation_mse']:.8f} -> {training['best_validation_mse']:.8f}.",
        '',
        'This is actual learned noise-to-Gaussian generation, not replay of a source clip. '
        'A student learns the original generator\'s fixed guidance=1.5 velocity output. '
        'It uses one conditional evaluation per denoising step instead of a batched conditional/unconditional pair. '
        'All pixels still come from Gaussian attributes and the unchanged vector-aware splatter.',
        '',
        '## Matched implementation measurements',
        '',
        '| Variant | Median ms/clip | Peak tensor MiB | Counted GMAC | Teacher-agreement dB |',
        '|---|---:|---:|---:|---:|',*rows,
        '',
        f"Primary eager 50-step student: signed time reduction {primary['time_reduction_percent']:.1f}% "
        f"({primary['speedup']:.2f}x speed), {primary['mac_reduction_percent']:.1f}% fewer counted MACs, "
        f"and {primary['peak_reduction_percent']:.1f}% lower allocated tensor peak.",
        '',
        'All variants: batch 1, eight 64x64 frames, FP32 weights/BF16 networks, frame-chunk 1, '
        'same GPU, two warmups/seven timings in isolated processes. Timings include sampling, decoding, '
        'splatting and CPU transfer, not loading or MP4 encoding. CUDA context/driver/other apps are not '
        'included in tensor-memory numbers. MAC counts include conv/linear and attention matrix products, '
        'not every operation or splat FLOPs. No matched-quality external video-model claim.',
        'The eager student did not improve latency. Comparing both graph variants gives only a small '
        'point-estimate difference with overlapping timing ranges; these seven repeats do not establish '
        'a robust additional wall-time improvement from learning alone.',
        '',
        '## Fidelity and what actually learned',
        '',
        f"Twenty frozen noise/label cases: student-50 agreement with teacher averages "
        f"{quality['mean_teacher_agreement_psnr']:.2f} dB, worst {quality['worst_teacher_agreement_psnr']:.2f} dB. "
        f"The predeclared approximate-fidelity gate passed: {quality['approximate_fidelity_gate_passed']}.",
        f"The no-learning, single-pass original conditional weights average "
        f"{baseline['mean_teacher_agreement_psnr']:.2f} dB. Student pixel MSE "
        f"{quality['mean_pixel_mse']:.7f}, no-learning control {baseline['mean_pixel_mse']:.7f}.",
        'Large localized errors remain in some clips even when average agreement looks good; '
        'the complete per-case metrics and all videos are retained. Neither 25-step variant fixes this.',
        '',
        'PSNR here measures agreement with our own imperfect generator, NOT fidelity to real video. '
        'The original teacher and the new student remain blurry and semantically unreliable. '
        'The 25-step variants are separately reported approximations, not automatically accepted improvements. '
        'No final UCF test clips or frozen external reconstruction cases were used for weight updates or selection.',
        '',
        '## Up-front learning cost',
        '',
        f"V2 training: {training['training_seconds']:.2f} s; reused-cache loading: {training['cache_seconds']:.2f} s. "
        f"V1 failed generalization and spent at least {previous_training:.2f} s in training. "
        'V1 cache-generation time was not persisted, so total experiment overhead is a lower bound, not an exact total.',
        f"Recorded additional time across both runs is at least {experiment_lower_bound:.2f} s. "
        'The eager learned 50-step model has no positive measured latency saving, so no wall-time '
        'training-payback claim is justified for that comparison. Graph acceleration requires no student '
        'training and must not be credited to it. This is GPU wall-clock '
        'amortization, not measured energy, money or total FLOPs.',
        '',
        'V1 weights/checkpoints remain in work/real_video/runs/guidance_distill_v1. V2 retains raw optimizer, '
        'EMA weights, RNG states, immutable checksummed checkpoints and original source snapshots. '
        'V1 also exposed an export assumption when no checkpoint beat initialization; the trainer now handles '
        'step-zero selection explicitly. See V1_FAILURE.md.',
        '',
        '## Limits and artifacts',
        '',
        '- Planar Gaussian vectors, eight-frame joint clips, ten class labels; not 3D physical dynamics or long-video recurrence.',
        '- This reduces generator calls, not the number of Gaussian dots, latent payload or network parameters.',
        '- Guidance is fixed at 1.5 in the student; incompatible guidance requests fail explicitly.',
        '- Decoder and Gaussian orientation equations are unchanged. No new physical law or novelty claim.',
        '- All 20 paired comparisons: audit/comparison_00.mp4 through comparison_19.mp4; no sample filtering.',
        '- Full metrics: audit/audit.json and benchmarks/*.json; deployable weights: guidance_student.pt.',
        '- Algorithmic motivation: [Meng et al., On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142). '
        'This project adapts fixed-guidance distillation to its Gaussian-latent generator, not a reproduction of all paper stages.',
        ''
    ]
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    source=out/'source'; source.mkdir(exist_ok=True)
    for name in ('model.py','graph_sampling.py','distill_guidance.py','audit_distillation.py','benchmark.py','latent_generate.py','summarize_compute_learning.py','audit_inference.py'):
        shutil.copyfile(Path(__file__).parent/name,source/name)
    root=Path(__file__).resolve().parents[1]
    shutil.copyfile(root/'research/COMPUTE_LEARNING_PROTOCOL.md',out/'PROTOCOL.md')
    shutil.copyfile(root/'research/DISTILLATION_NOTE.md',out/'MATH_NOTE.md')
    (out/'source_hashes.json').write_text(json.dumps({p.name:digest(p) for p in source.glob('*.py')},indent=2),encoding='utf-8')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__': main()
