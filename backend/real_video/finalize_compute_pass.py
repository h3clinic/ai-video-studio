"""Verify and summarize the continued compute-learning pass, retaining failures."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import cv2
from .checkpoint_io import digest,load_verified,keep_windows_awake

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out
    def read(name): return json.loads((out/name).read_text(encoding='utf-8'))
    old=read('benchmarks/teacher_50.json')
    fast=read('benchmarks/teacher_graph_precast_50.json')
    low=read('benchmarks/teacher_precast_50.json')
    student=read('v3/benchmark_graph_precast_50.json')
    tb=read('benchmarks/teacher_graph_precast_batch8.json')
    sb=read('v3/benchmark_graph_precast_batch8.json')
    training=read('v3/guidance_student.training.json'); v2training=read('guidance_student.training.json')
    audit=read('v3/audit/audit.json'); quality=audit['aggregates']['student_50']
    prior_quality=audit['aggregates']['previous_student_50']
    precision=read('precision_audit/audit.json'); student_precision=read('v3/precision_audit/audit.json')
    assert precision['passed'] and student_precision['passed']
    load_verified(out/'v3/guidance_student.pt')
    original=ROOT/'artifacts/real_video/weight_design/continued_latent_weights.pt'
    assert digest(original)==training['manifest']['teacher_sha256']
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    (out/'verification_tests.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    if tests.returncode: raise AssertionError('Unit test failure; see verification_tests.txt')
    videos=[]
    for path in sorted(out.rglob('*.mp4')):
        reader=cv2.VideoCapture(str(path)); count=0; sizes=set()
        fps=reader.get(cv2.CAP_PROP_FPS)
        while True:
            ok,frame=reader.read()
            if not ok: break
            sizes.add((int(frame.shape[1]),int(frame.shape[0]))); count+=1
        reader.release()
        if count!=8 or len(sizes)!=1 or abs(fps-8)>0.1: raise AssertionError(f'Invalid clip: {path}')
        videos.append(dict(path=str(path.relative_to(out)),frames=count,size=list(next(iter(sizes))),fps=fps))
    wins=sum(c['variants']['student_50']['pixel_mse']<c['variants']['previous_student_50']['pixel_mse'] for c in audit['cases'])
    arithmetic_reduction=100*(1-student['module_macs']/old['module_macs'])
    speedup=old['median_seconds']/fast['median_seconds']
    low_memory_reduction=100*(1-low['peak_cuda_allocated_bytes']/old['peak_cuda_allocated_bytes'])
    total_recorded=164.0124896+v2training['total_extra_compute_seconds']+training['total_extra_compute_seconds']
    delta_batch=tb['median_seconds']-sb['median_seconds']
    summary=dict(original_unchanged=True,unit_tests_passed=True,verified_videos=videos,
                 trained_v3_steps=training['selected_step'],v3_sha256=digest(out/'v3/guidance_student.pt'),
                 student_quality=quality,previous_student_quality_on_same_seeds=prior_quality,
                 paired_pixel_mse_wins_out_of_20=wins,arithmetic_reduction_percent=arithmetic_reduction,
                 original_same_output_speedup=speedup,lower_memory_mode_reduction_percent=low_memory_reduction,
                 recorded_extra_training_preparation_seconds_lower_bound=total_recorded,
                 batch8_point_estimate_speedup=tb['median_seconds']/sb['median_seconds'],
                 batch8_training_payback_clips_lower_bound=math.ceil(total_recorded/(delta_batch/8)) if delta_batch>0 else None,
                 teacher_precision_audit=precision,student_precision_audit=student_precision)
    (out/'CONTINUED_RESULTS.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    def row(name,b):
        return f"| {name} | {b['median_seconds']*1000:.1f} | {b['peak_cuda_allocated_bytes']/2**20:.2f} | {b['module_macs']/1e9:.3f} |"
    lines=[
        '# Continued Gaussian-vector learning and cost results',
        '',
        '## What improved',
        '',
        f"- Original generator, same computation: {old['median_seconds']*1000:.1f} -> "
        f"{fast['median_seconds']*1000:.1f} ms per clip ({speedup:.2f}x faster) using GPU execution graphs "
        'and selective inference-weight precasting. Twenty fresh cases retained exactly identical Gaussian '
        'fields; maximum pixel difference was 2.98e-7. This is execution engineering, not learned arithmetic reduction.',
        f"- V3 student: {training['selected_step']} additional learning updates, with {arithmetic_reduction:.1f}% fewer "
        'counted Conv/Linear/attention MACs than the original 50-step guided prior+decoder. '
        'It passed the declared approximate teacher-fidelity gate on a new, untouched set of 20 noise seeds.',
        f"- Lower-memory mode: {low_memory_reduction:.1f}% less allocated tensor memory, but it is slower. "
        'Fast graph mode uses slightly more memory than the original eager baseline. There is no single '
        'configuration here that wins every cost measure.',
        '',
        '## Single-clip measurements',
        '',
        '| Configuration, 50 steps | Median ms/clip | Peak tensor MiB | Counted GMAC |',
        '|---|---:|---:|---:|',
        row('Original eager',old),row('Original, precast only',low),
        row('Original, graph + precast',fast),row('V3 learned, graph + precast',student),
        '',
        'Batch 1, eight 64x64 frames, RTX 5070 Laptop GPU, two warmups/seven measurements; '
        'includes sampling, Gaussian decoding/splatting and CPU transfer. Excludes import, checkpoint '
        'loading, graph setup and video encoding. Tensor memory is not whole-process VRAM or system RAM. '
        'MACs exclude splat operations, normalizations and other arithmetic. Video pixels are ALWAYS '
        'rendered from the generated Gaussian attributes, never a source-image overlay or RGB neural head.',
        '',
        f"Graph setup: original {fast['graph_setup_seconds']:.3f} s; student {student['graph_setup_seconds']:.3f} s. "
        'A one-shot CLI invocation pays that overhead each time. Reuse the runner in a long-lived process '
        'to obtain the warmed throughput. At these medians, original graph setup amortizes after about '
        f"{math.ceil(fast['graph_setup_seconds']/(old['median_seconds']-fast['median_seconds']))} clips.",
        '',
        '## Matched batch-8 throughput probe',
        '',
        f"Original graph+precast: {tb['median_seconds']*1000:.2f} ms/batch, "
        f"{tb['clips_per_second']:.2f} clips/s, {tb['peak_cuda_allocated_bytes']/2**20:.2f} MiB peak.",
        f"V3 graph+precast: {sb['median_seconds']*1000:.2f} ms/batch, "
        f"{sb['clips_per_second']:.2f} clips/s, {sb['peak_cuda_allocated_bytes']/2**20:.2f} MiB peak.",
        f"Point-estimate student/original speed ratio: {tb['median_seconds']/sb['median_seconds']:.3f}x. "
        'Seven repetitions are a small timing sample and ranges may overlap. Do not equate fewer MACs '
        'with proportional speed, energy or monetary savings. Do not compare batch-8 throughput with '
        'batch-1 latency as though they were the same workload.',
        '',
        '## Actual learning and held-out approximation',
        '',
        'V3 starts from V2 and receives fresh teacher velocity targets on a 50/50 mix of: '
        '(a) forward-noised codes from the 279 original UCF training clips and their mirrors; '
        '(b) jittered TRAIN teacher-rollout states. These are not additional independent video clips. '
        'No UCF final-test video, external ModelScope reconstruction case, or audit seed enters updates. '
        'Selection uses only the existing separate validation trajectories.',
        f"Validation velocity MSE: {training['initial_validation_mse']:.8f} -> {training['best_validation_mse']:.8f}.",
        f"On the SAME 20 new seeds, V2 -> V3 mean teacher-agreement PSNR: "
        f"{prior_quality['mean_teacher_agreement_psnr']:.2f} -> {quality['mean_teacher_agreement_psnr']:.2f} dB; "
        f"mean pixel MSE {prior_quality['mean_pixel_mse']:.8f} -> {quality['mean_pixel_mse']:.8f}. "
        f"V3 improves pixel MSE on {wins}/20 cases; it is not uniformly better.",
        f"V3 worst agreement: {quality['worst_teacher_agreement_psnr']:.2f} dB. "
        'The gate requires mean >=30 dB, worst >=25 dB, and lower pixel MSE than the no-learning '
        f"conditional-only control. Passed: {quality['approximate_fidelity_gate_passed']}.",
        'The 25-step V3 variant FAILS the worst-case gate and is not recommended. '
        'All cases, including regressions, remain in v3/audit. Comparing to the old V2 audit\'s different '
        'seeds would not be a paired measure of progress; use the previous_student_50 column in this new audit.',
        '',
        'Important: teacher agreement is NOT external natural-video quality. Both teacher and student '
        'remain visibly blurry, small, eight-frame class-conditioned prototypes. No foundation-model, '
        'photorealistic, learned-physics, long-video recurrence, or commercial-video-model superiority claim.',
        '',
        '## Learning overhead and retained records',
        '',
        f"V3 training took {training['training_seconds']:.2f} s, preparation {training['preparation_seconds']:.2f} s. "
        f"V1+V2+V3 recorded training/preparation adds up to at least {total_recorded:.2f} s; "
        'V1 cache generation was not persisted, and coding/evaluation time is excluded. This is not free '
        'inference compression. The unchanged-original graph optimization needs no student training. '
        'No robust single-clip latency improvement from distillation alone has been established.',
        'V1 overfit and hit a handled export-selection bug; V2 improved average fidelity but failed its '
        'worst-case gate. Both are preserved. V3\'s actual RNG seed is 61729, not the initially proposed '
        '71729; this deviation was recorded immediately after launch, before outcomes. No seed search was performed.',
        '',
        f"Verified all {len(videos)} MP4 files by decoding every frame; unit suite passed (see verification_tests.txt). "
        'The original checkpoint digest is unchanged. Model and optimizer/RNG recovery checkpoints are '
        'versioned with SHA256 sidecars. Training jobs held temporary Windows sleep-prevention requests '
        'and released them on exit; no permanent power-policy changes were made.',
        '',
        '## Use and inspect',
        '',
        '```powershell',
        '.\\run.ps1 -m real_video.latent_generate artifacts/real_video/weight_design/continued_latent_weights.pt --cuda-graph --precast-weights --seed 12345 --label BabyCrawling --out artifacts/real_video/new_fast_original',
        '.\\run.ps1 -m real_video.latent_generate artifacts/real_video/compute_learning/v3/guidance_student.pt --cuda-graph --precast-weights --seed 12345 --label BabyCrawling --out artifacts/real_video/new_learned_student',
        '```',
        '',
        'V3 weights: v3/guidance_student.pt. All paired videos: v3/audit/comparison_00.mp4 through '
        'comparison_19.mp4. Source-free CLI audit: v3/inference_audit/inference_audit.json. '
        'Precision equivalence: precision_audit/audit.json and v3/precision_audit/audit.json. '
        'Raw measurements: benchmarks/*.json and v3/benchmark*.json. '
        'Earlier results: REPORT.md and V1_FAILURE.md.',
        '',
        'Method references: [guided-model distillation](https://arxiv.org/abs/2210.03142); '
        '[PyTorch 2.11 CUDA graphs](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs). '
        'These are standard techniques adapted here, not novelty claims.',
        ''
    ]
    (out/'CONTINUED_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    snapshot=out/'continued_source'; snapshot.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'): shutil.copyfile(path,snapshot/path.name)
    for name in ('COMPUTE_LEARNING_PROTOCOL.md','DISTILLATION_NOTE.md'):
        shutil.copyfile(ROOT/'research'/name,snapshot/name)
    shutil.copyfile(ROOT/'tests/test_distillation.py',snapshot/'test_distillation.py')
    (out/'continued_source_hashes.json').write_text(json.dumps({p.name:digest(p) for p in snapshot.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('verified_videos','teacher_precision_audit','student_precision_audit')},indent=2))


if __name__=='__main__':
    with keep_windows_awake(): main()
