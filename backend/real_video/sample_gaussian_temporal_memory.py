"""Fresh learned locomotion -> persistent 3D Gaussian memory -> temporal Wan.

The released AI4AnimationPy dog controller generates new skeletal predictions;
an approximate manual cat rig retargets them. No previous trajectory or source
video is read. The Gaussian condition and generated Wan RGB video are separate
outputs: Wan does NOT emit updated Gaussian state or guarantee exact identity.

This combines a real-DAVIS-trained experimental residual adapter with a different
conditioning domain (dense 3D renders). It is not a replication of Wan-Move or
an accepted anatomy/efficiency result. The persistent memory runtime is bounded
for a fixed asset, but this script retains one finite conditioning clip for Wan.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from real_video.checkpoint_io import (
    digest, keep_windows_awake, load_verified, save_inference_checkpoint,
)
from real_video.wan_temporal_control import (
    TemporalSpatialControl, attach_temporal_control, wan_temporal_average,
)


BASE = Path('artifacts/real_video')
ASSET = BASE/'hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt'
SHAPE = BASE/'hunyuan_gaussian/v4_full/shape.pt'
BUNDLE = BASE/'neural_motion_controller/v1'
MODEL = Path('../../work/wan21_13b')
EMBEDDINGS = BASE/'wan_baseline/cat_seed_421001/prompt_embeddings.pt'
HEIGHT, WIDTH, FRAMES, FPS = 256, 448, 57, 8
EVIDENCE = (0, 16, 32, 48, 56)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def save_report(out, report, message=None):
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if message:
        print(message, flush=True)


def tensor_bytes(state):
    return sum(v.numel()*v.element_size() for v in state.values() if isinstance(v, torch.Tensor))


def image_bytes(rgb):
    return (rgb.clamp(0, 1)*255).round().byte().cpu().numpy()


@torch.inference_mode()
def prepare_condition(out, report, bundle, retarget_mode):
    from real_video.quadruped_controller import QuadrupedController
    from real_video.controller_rig import guarded_rig, CatRetargeter, DualSurfaceSkinner
    from real_video.gaussian_video_memory import GaussianVideoMemory

    if (out/'temporal_condition.pt').exists():
        raise FileExistsError('Preserve prepared condition')
    stage_start = time.perf_counter()
    controller = QuadrupedController(device='cpu', bundle=bundle)
    state0 = controller.snapshot()
    export = json.loads((bundle/'export_report.json').read_text())
    save_inference_checkpoint(state0, out/'controller_initial_state.pt')
    source = load_verified(ASSET)
    mesh = load_verified(SHAPE)
    a = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in source.items()}
    a['mesh_vertices'], a['mesh_faces'] = mesh['vertices'].cuda(), mesh['faces'].cuda()
    del source, mesh
    pieces = [torch.linalg.eigh(x) for x in a['covariance'].split(16000)]
    values = torch.cat([p[0] for p in pieces])
    frame = torch.cat([p[1] for p in pieces])
    frame[:, :, 0] *= torch.linalg.det(frame).sign()[:, None]
    a['frame'], a['scale'] = frame, values.clamp_min(1e-12).sqrt()
    del pieces, values, frame
    rig = guarded_rig(a['mesh_vertices'])
    skinner = DualSurfaceSkinner(a, rig)
    targeter = CatRetargeter(rig, controller.guidances['Walk'].cuda(), mode=retarget_mode)
    memory = GaussianVideoMemory(a, device='cuda')
    initial_bytes = memory.memory_bytes()
    identity = torch.eye(3, device='cuda').repeat(17, 1, 1)
    rest_position, rest_covariance, *_ = skinner(identity)
    rest_error = float((rest_position-a['position']).abs().max())
    rest_cov_error = float((rest_covariance-a['covariance']).abs().max())
    if rest_error >= 1e-5 or rest_cov_error >= 1e-5:
        raise ValueError('Skinner does not reproduce the canonical asset at rest')
    del rest_position, rest_covariance, identity
    center = (a['position'].amin(0)+a['position'].amax(0))/2
    direction = center.new_tensor([0., .13, 1.])
    direction = direction/direction.norm()
    selected_indices = np.rint(np.linspace(0, 210, FRAMES)).astype(int).tolist()
    selected_lookup = {value: index for index, value in enumerate(selected_indices)}
    conditions, diagnostics, pose_records = [], [], []
    torch.cuda.reset_peak_memory_stats()
    previous_memory_frame = None
    controller_seconds = 0.
    render_seconds = []
    Gaussian_path = out/'gaussian_condition.mp4'
    with imageio.get_writer(Gaussian_path, fps=FPS, codec='libx264', quality=8,
                            macro_block_size=1) as writer:
        def consume(frame_index, relative, world):
            nonlocal previous_memory_frame
            # Retarget EVERY controller sample so contact state is not skipped.
            local, shift, feet, error, contacts = targeter.step(
                relative.cuda(), (world[0]-relative[0]).cuda())
            if frame_index not in selected_lookup:
                return
            output_index = selected_lookup[frame_index]
            torch.cuda.synchronize()
            start = time.perf_counter()
            p, covariance, _, vertices, areas = skinner(local, shift)
            owned = vertices[a['mesh_faces'].long()[a['face_id'].long()]]
            normals = F.normalize(torch.linalg.cross(owned[:, 1]-owned[:, 0],
                                                     owned[:, 2]-owned[:, 0]), dim=-1)
            # Remove only floating-point antisymmetry, not shape distortion.
            covariance = .5*(covariance+covariance.transpose(-1, -2))
            dt = 1/30 if previous_memory_frame is None else (frame_index-previous_memory_frame)/30
            memory.update_covariance(p, covariance, ids=a['ids'], dt=dt, normal=normals)
            previous_memory_frame = frame_index
            target = center+shift
            camera = dict(eye=target+direction*3.5, target=target, fov=42.)
            emitted = memory.project_current(camera, HEIGHT, WIDTH, downsample=8, radius=2)
            rgb, alpha = emitted['rgb'], emitted['alpha']
            # This is a dense rendered observation, so every rendered pixel is
            # present. Alpha separately identifies Gaussian foreground coverage.
            rgba = torch.cat((rgb.permute(2, 0, 1), alpha[None]), dim=0)[None]
            rgba = F.avg_pool2d(rgba, 8)
            five = torch.cat((rgba, torch.ones_like(rgba[:, :1])), dim=1).cpu()
            conditions.append(five)
            writer.append_data(image_bytes(rgb))
            if output_index in EVIDENCE:
                Image.fromarray(image_bytes(rgb)).save(out/f'gaussian_{output_index:03d}.png')
                oblique_direction = center.new_tensor([.7, .23, 1.])
                oblique_direction /= oblique_direction.norm()
                oblique = memory.project_current(
                    dict(eye=target+oblique_direction*3.5, target=target, fov=42.),
                    HEIGHT, WIDTH, downsample=8, radius=2)
                Image.fromarray(image_bytes(oblique['rgb'])).save(out/f'gaussian_oblique_{output_index:03d}.png')
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            render_seconds.append(seconds)
            diagnostics.append(dict(frame=output_index, controller_frame=frame_index,
                time_seconds=frame_index/30, root_translation=shift.cpu().tolist(),
                ik_max_error=float(error.max()), contacts=contacts,
                area_min=float(areas.min()), area_max=float(areas.max()),
                bad_area_fraction=float(((areas < .2) | (areas > 5)).float().mean()),
                mean_displacement=float((p-a['position']).norm(dim=-1).mean()),
                seconds=seconds, memory_bytes=memory.memory_bytes()['resident_tensor_bytes']))
            pose_records.append(dict(rotation=local.cpu(), root_translation=shift.cpu()))
            if output_index % 8 == 0:
                save_report(out, report, f'Fresh neural Gaussian motion projection {output_index+1}/{FRAMES}')

        # t=0 is the stored released static initialization; all 210 later poses
        # are new model predictions, never loaded from a prior motion packet.
        consume(0, controller.position, controller.position+controller.root)
        for call in range(70):
            start = time.perf_counter()
            world, relative, _ = controller.step('Walk', .7)
            controller_seconds += time.perf_counter()-start
            for j in range(3):
                consume(1+call*3+j, relative[j], world[j])
    if len(conditions) != FRAMES:
        raise AssertionError('Incomplete generated condition')
    frame_volume = torch.stack(conditions, dim=2)
    condition = wan_temporal_average(frame_volume)
    assert condition.shape == (1, 5, 15, HEIGHT//8, WIDTH//8)
    final_bytes = memory.memory_bytes()
    if final_bytes != initial_bytes:
        raise AssertionError('Gaussian runtime state grew during rollout')
    state1 = controller.snapshot()
    # Include enough state to resume this external motion/retarget pipeline;
    # the Gaussian snapshot alone does not encode the controller or IK latches.
    resume = dict(controller=state1, anchors=[None if x is None else x.cpu() for x in targeter.anchors],
                  rig={k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in rig.items()},
                  gait='Walk', speed=.7, retarget_mode=retarget_mode,
                  gaussian_memory_snapshot='gaussian_memory_final.pt', next_controller_frame=211)
    save_inference_checkpoint(resume, out/'motion_resume_state.pt')
    memory.save(out/'gaussian_memory_final.pt')
    save_inference_checkpoint(dict(condition=condition, frame_volume=frame_volume,
        sampled_controller_frames=selected_indices, fps=FPS, gaussian_sha256=digest(ASSET),
        scope='New learned skeletal rollout, approximate manual cat retarget, dense Gaussian render conditioning'),
        out/'temporal_condition.pt')
    save_inference_checkpoint(dict(poses=pose_records, fps=FPS,
        sampled_controller_frames=selected_indices, diagnostic_only=True,
        scope='Newly generated pose diagnostics, not an input to the inference run'), out/'generated_pose_diagnostics.pt')
    report['gaussian_preparation'] = dict(seconds=time.perf_counter()-stage_start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), gaussian_count=memory.count,
        controller_seconds=controller_seconds, controller_calls=70, controller_device='cpu',
        source_controller_sha256=export['source_weights_sha256'],
        controller_export_sha256=export['export_sha256'], controller_license=export['license'],
        initial_controller_state_bytes=tensor_bytes(state0), final_controller_state_bytes=tensor_bytes(state1),
        gaussian_runtime_before=initial_bytes, gaussian_runtime_after=final_bytes,
        gaussian_memory_time_origin_note='Initial t=0 pose is committed with dt1/30; runtime elapsed clock offset1/30 relative to video time',
        immutable_gaussian_ids_enforced_on_every_update=True, generated_poses=210,
        initial_static_poses=1, source_video_reads=0, previous_motion_packet_reads=0,
        rest_position_max_error=rest_error, rest_covariance_max_error=rest_cov_error,
        render_and_skin_median_seconds=float(np.median(render_seconds)),
        frame_condition_tensor_bytes=frame_volume.numel()*frame_volume.element_size(),
        compressed_condition_tensor_bytes=condition.numel()*condition.element_size(),
        runtime_exclusions='Controller/rig/skinner copies, Wan clip tensors, model weights, projection temporaries and saved diagnostic output are outside bounded Gaussian-runtime count',
        diagnostics=diagnostics)
    save_report(out, report, 'Prepared 57 frames from fresh neural motion; anatomical acceptance remains pending')


@torch.inference_mode()
def generate_wan(out, report, checkpoint, steps):
    from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline, UniPCMultistepScheduler
    from real_video.wan_cat_memory import install_lora

    if (out/'wan.mp4').exists() or (out/'wan_latent.pt').exists():
        raise FileExistsError('Preserve existing Wan inference artifacts')
    packet = load_verified(checkpoint)
    if packet.get('rank') != 4 or packet.get('control_channels') != 5:
        raise ValueError('Expected rank4/5-channel temporal adapter')
    saved = load_verified(out/'temporal_condition.pt')
    if saved['gaussian_sha256'] != digest(ASSET):
        raise ValueError('Prepared condition references a different Gaussian asset')
    condition = saved['condition'].cuda()
    if tuple(condition.shape) != (1, 5, 15, 32, 56):
        raise ValueError('Unexpected temporal condition shape')
    del saved
    start = time.perf_counter()
    model = WanTransformer3DModel.from_pretrained(MODEL/'transformer', torch_dtype=torch.bfloat16,
        local_files_only=True).eval().requires_grad_(False).cuda()
    install_lora(model, 4)
    control = TemporalSpatialControl(1536, channels=5).cuda()
    attach_temporal_control(model, control)
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(expected) != set(packet['weights']):
        missing, extra = sorted(set(expected)-set(packet['weights'])), sorted(set(packet['weights'])-set(expected))
        raise ValueError(f'Temporal checkpoint schema mismatch: missing={missing[:4]}, extra={extra[:4]}')
    for name, parameter in expected.items():
        value = packet['weights'][name]
        if value.shape != parameter.shape:
            raise ValueError(f'Parameter shape mismatch: {name}')
        parameter.copy_(value)
    trained_parameters = sum(p.numel() for p in expected.values())
    model.requires_grad_(False)
    control.bind(condition, (15, HEIGHT//16, WIDTH//16))
    embeddings = load_verified(EMBEDDINGS)
    positive, negative = [embeddings[k].cuda().to(torch.bfloat16) for k in ('positive', 'negative')]
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True,
        num_train_timesteps=1000, flow_shift=8.)
    pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=model, vae=None, scheduler=scheduler)
    del packet, expected, embeddings, value, parameter
    free()
    report['wan_load_seconds'] = time.perf_counter()-start
    save_report(out, report, 'Generating 57-frame Wan video from learned moving Gaussian condition; no output compositing')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    latent = pipe(prompt_embeds=positive, negative_prompt_embeds=negative,
        height=HEIGHT, width=WIDTH, num_frames=FRAMES, num_inference_steps=steps,
        guidance_scale=5., generator=torch.Generator(device='cuda').manual_seed(73007),
        output_type='latent').frames.cpu()
    torch.cuda.synchronize()
    report['wan_generation'] = dict(denoise_seconds=time.perf_counter()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), steps=steps, seed=73007,
        guidance_scale=5., trained_adapter_parameters=trained_parameters,
        checkpoint_sha256=digest(checkpoint), prompt_embeddings_sha256=digest(EMBEDDINGS),
        actual_input_condition='Every latent time; same Gaussian geometry conditioning on positive and negative text branches',
        output_representation='Wan RGB video latent, NOT emitted Gaussian state')
    save_inference_checkpoint(dict(latent=latent, checkpoint_sha256=digest(checkpoint)), out/'wan_latent.pt')
    control.condition = None
    del pipe, model, control, condition, positive, negative
    free()
    start = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(MODEL/'vae', torch_dtype=torch.float32,
        local_files_only=True).eval().requires_grad_(False).cuda()
    vae.enable_tiling()
    mean = torch.tensor(vae.config.latents_mean, device='cuda').view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device='cuda').view(1, -1, 1, 1, 1)
    report['vae_load_seconds'] = time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    decoded = vae.decode(latent.cuda().float()*std+mean, return_dict=False)[0]
    frames = ((decoded[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
    torch.cuda.synchronize()
    report['vae_decode'] = dict(seconds=time.perf_counter()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated())
    if frames.shape != (FRAMES, HEIGHT, WIDTH, 3):
        raise ValueError('Unexpected decoded video shape')
    imageio.mimwrite(out/'wan.mp4', frames, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    sheet = Image.new('RGB', (WIDTH*len(EVIDENCE), HEIGHT*2))
    for col, index in enumerate(EVIDENCE):
        image = Image.fromarray(frames[index])
        image.save(out/f'wan_{index:03d}.png')
        sheet.paste(Image.open(out/f'gaussian_{index:03d}.png').convert('RGB'), (col*WIDTH, 0))
        sheet.paste(image, (col*WIDTH, HEIGHT))
    sheet.save(out/'gaussian_vs_wan.jpg')
    report['finished'] = True
    save_report(out, report, 'Generated Gaussian-condition and Wan videos separately; visual quality review required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'generate', 'all'), default='all')
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--retarget-mode', choices=('ik', 'relative'), default='ik')
    parser.add_argument('--steps', type=int, default=28)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError('Positive denoising step count required')
    if args.phase == 'generate':
        if not (args.out/'temporal_condition.pt').exists():
            raise FileNotFoundError('Prepare the Gaussian condition first')
        report = json.loads((args.out/'report.json').read_text())
    else:
        if args.out.exists():
            raise FileExistsError('Use a new immutable experiment directory')
        args.out.mkdir(parents=True)
        report = dict(scope='New pretrained neural locomotion projected from persistent 3D Gaussians into experimentally adapted Wan',
            quality_accepted=False, finished=False, frames=FRAMES, fps=FPS,
            sample_time_span_seconds=7., encoded_playback_duration_seconds=FRAMES/FPS,
            height=HEIGHT, width=WIDTH, gaussian_asset_sha256=digest(ASSET),
            geometry_sha256=digest(SHAPE), retarget_mode=args.retarget_mode,
            conditioning_channels=['rendered_rgb_r', 'rendered_rgb_g', 'rendered_rgb_b', 'gaussian_alpha', 'dense_render_occupancy'],
            camera='Root-following fixed direction; global translation partly canceled to retain full animal in frame',
            output_compositing=False, validation_or_test_clips_read=False,
            limitations=['Dog locomotion prior with manually inferred cat rig; prior retarget quality was rejected',
                'No newly trained or validated cat-specific 3D dynamics',
                'New controller predictions are not a scripted sine gait or a loaded motion clip',
                'Dense Gaussian renders differ from sparse transported DAVIS training features',
                'RGB output is newly decoded by Wan and may not preserve exact Gaussian geometry or appearance',
                'Gaussian runtime state is bounded but Wan conditioning/output and denoising memory are clip-sized',
                'No quality-matched compute or memory saving established',
                'Generated Gaussian pose diagnostics are saved outputs, not inference inputs'])
    torch.set_num_threads(4)
    with keep_windows_awake():
        if args.phase in ('prepare', 'all'):
            prepare_condition(args.out, report, args.bundle, args.retarget_mode)
            free()
            report['cuda_allocated_after_gaussian_stage_release'] = torch.cuda.memory_allocated()
            save_report(args.out, report)
        if args.phase in ('generate', 'all'):
            generate_wan(args.out, report, args.checkpoint, args.steps)


if __name__ == '__main__':
    main()
