"""Measure motion controls, NOT rendering or accepted animal video quality."""
import argparse
import json
from pathlib import Path
import statistics
import time
import psutil
import torch
from .checkpoint_io import keep_windows_awake,save_inference_checkpoint
from .motion_only_session import MotionOnlySession,tensor_bytes


def run(out):
    if psutil.virtual_memory().available<8*1024**3:
        raise RuntimeError('Need8GiB available RAM')
    battery=psutil.sensors_battery()
    if battery and not battery.power_plugged and battery.percent<=20:
        raise RuntimeError('Low battery')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    start=time.perf_counter();session=MotionOnlySession()
    initialization=time.perf_counter()-start
    initial=session.snapshot()
    save_inference_checkpoint(initial,out/'initial_session.pt')
    commands=[dict(action='idle',blocks=10),dict(action='walk',blocks=40),dict(action='trot',blocks=20)]
    rows=[];timings=[];models=[];retarget=[];controls=[];states=[]
    for item in commands:
        session.set_action(item['action'])
        for _ in range(item['blocks']):
            if time.perf_counter()-start>60:raise RuntimeError('CPU pilot budget reached')
            begin=time.perf_counter();block=session.advance();elapsed=time.perf_counter()-begin
            model=session.controller.latencies[-1]
            timings.append(elapsed);models.append(model);retarget.append(elapsed-model)
            controls.append({k:v for k,v in block.items() if k in ('local_rotation','root_translation')})
            states.append(tensor_bytes(session.snapshot()))
            rows.append(dict(first_frame=block['first_frame'],action=item['action'],
                max_unreachable_target_error=float(block['ik_reach_error'].max())))
    final=session.snapshot();save_inference_checkpoint(final,out/'final_session.pt')
    expected=session.advance();session.restore(final);actual=session.advance()
    exact=all(torch.equal(expected[k],actual[k]) for k in ('local_rotation','root_translation','ik_reach_error'))
    rotations=torch.cat([c['local_rotation'] for c in controls])
    roots=torch.cat([c['root_translation'] for c in controls])
    save_inference_checkpoint(dict(local_rotation=rotations,root_translation=roots,fps=30,
        bindings=session.bindings,classification='Generated skeletal controls, not rendered accepted video',
        future_history_input=False),out/'diagnostic_controls.pt')
    result=dict(scope='CPU motion-only session architecture validation; cat anatomy remains rejected',
        commands=commands,frames=len(rotations),fps=30,duration_seconds=len(rotations)/30,
        initialization_seconds=initialization,motion_control_seconds=sum(timings),
        neural_prediction_seconds=sum(models),retarget_and_pack_seconds=sum(retarget),
        block_median_ms=1000*statistics.median(timings),
        recurrent_state_tensor_bytes=tensor_bytes(initial),state_tensor_bytes_min=min(states),
        state_tensor_bytes_max=max(states),checkpoint_resume_exact=exact,
        static_model_file_bytes=Path(session.bindings['controller']['path']).stat().st_size,
        static_asset_file_bytes=Path(session.bindings['asset']['path']).stat().st_size,
        static_mesh_file_bytes=Path(session.bindings['shape']['path']).stat().st_size,
        static_rig_file_bytes=Path(session.bindings['rig']['path']).stat().st_size,
        retained_diagnostic_control_tensor_bytes=rotations.numel()*rotations.element_size()+roots.numel()*roots.element_size(),
        local_process_peak_rss_bytes=psutil.Process().memory_info().peak_wset,
        rotation_orthogonality_max_error=float((rotations.transpose(-1,-2)@rotations-torch.eye(3)).abs().max()),
        training_seconds=None,gaussian_deformation_seconds=None,rasterization_seconds=None,encoding_seconds=None,
        new_training=False,api_calls=0,per_frame_language_model_calls=0,paid_credits=0,
        quality_accepted=False,rendered_video_produced=False,
        limitations=['Known rejected canonical shape/manual rig. No anatomy improvement claimed.',
            'Commands select released dog locomotion guidance; no free-text action understanding.',
            'State tensor bytes exclude weights/static guidance/rig/Python/temporary prediction buffers.',
            'Saved diagnostic controls grow with duration but are not inference history.',
            'No trained nonrigid residual, ground-contact guarantee, or quality-matched full-generator speedup.',
            'Upstream controller license CC-BY-NC-4.0; research-only use.'],
        bindings=session.bindings,diagnostics=rows)
    (out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('diagnostics','bindings')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    with keep_windows_awake():run(args.out)
