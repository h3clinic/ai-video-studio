"""Fresh neural locomotion, retained Gaussian animals, and a fixed-camera scene.

This is a pretrained skeletal controller with heuristic mesh rigging, NOT Wan
emitting a Gaussian movie. No RGB clip, image crop or camera orbit drives motion.
The source poses retained here are evaluation output, not model input history.
"""
import argparse
import json
import math
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .checkpoint_io import digest, keep_windows_awake, load_verified, save_inference_checkpoint
from .controller_rig import CatRetargeter, DualSurfaceSkinner
from .dog_articulation import DogRetargeter, PARENTS as DOG_PARENTS
from .fit_motion3d import PARENTS as CAT_PARENTS
from .gaussian3d import render as legacy_render, project
from .asset_readiness import file_identity, require_motion_ready
from .gaussian_scene_objects import GaussianSceneObject, merge_scene_objects
from .gaussian_video_memory import GaussianVideoMemory
from .quadruped_controller import QuadrupedController
from .surface_skin3d import triangle_basis

BASE = Path('artifacts/real_video')
BUNDLE = BASE/'neural_motion_controller/v1'
ASSETS = {'cat': BASE/'hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt',
          'dog': BASE/'gaussian_dog_insertion/v1/dog/cat_asset.pt'}
CAT_MESH = BASE/'hunyuan_gaussian/v4_full/shape.pt'
DEFAULT_PREFLIGHT = {name: BASE/f'program_audits/geometry_v1/{name}_readiness.json'
                     for name in ('cat', 'dog')}
ADAPTIVE_LIMITS = dict(fragment_budget=2_000_000, max_candidate_pixels=300_000_000,
                       wall_seconds=30, slope_guard=1.3)


def renderer_metadata(renderer, scene_radius=3):
    if renderer not in ('legacy', 'adaptive'):
        raise ValueError('Renderer must be legacy or adaptive')
    module = 'gaussian3d.py' if renderer == 'legacy' else 'adaptive_gaussian_renderer.py'
    return dict(name=renderer, source=file_identity(Path(__file__).parent/module),
                limits={'scene_radius': scene_radius, 'diagnostic_radius': 3}
                       if renderer == 'legacy' else dict(ADAPTIVE_LIMITS),
                scope='Inference rasterizer only; not geometry repair, learned motion, or quality approval.',
                timeout_scope='Adaptive wall budget is per render call, checked between chunks; not a hard process deadline.')


def render_action_frame(position, covariance, colour, opacity, eye, target, *,
                        height, width, ground, renderer='legacy', radius=3,
                        fov=42., principal=None):
    """Dispatch without silently falling back after a renderer budget failure."""
    common = dict(height=height, width=width, ground=ground, fov=fov, principal=principal)
    if renderer == 'legacy':
        return legacy_render(position, covariance, colour, opacity, eye, target,
                             radius=radius, **common)
    if renderer == 'adaptive':
        from .adaptive_gaussian_renderer import render as adaptive_render
        return adaptive_render(position, covariance, colour, opacity, eye, target,
                               **common, **ADAPTIVE_LIMITS)
    raise ValueError('Renderer must be legacy or adaptive')


def preflight_assets(cat_report=None, dog_report=None):
    """Return approved CPU rigs, not newly rebuilt unreviewed replacements.

    Explicit report paths may select refreshed reviews, never waive approval.
    The current defaults intentionally block the rejected cat and dog. Even a
    static cat must have accepted geometry: a static blob is not a valid scene.
    """
    approved_rigs, provenance = {}, {}
    reports = {'cat': cat_report or DEFAULT_PREFLIGHT['cat'],
               'dog': dog_report or DEFAULT_PREFLIGHT['dog']}
    for name, path in reports.items():
        try:
            identity = file_identity(path)
            report = json.loads(Path(path).read_text(encoding='utf-8'))
            if file_identity(path) != identity:
                raise RuntimeError('Preflight report changed while reading it')
            require_motion_ready(report)
            expected_mesh = CAT_MESH if name == 'cat' else ASSETS[name]
            if (report['inputs']['asset'] != file_identity(ASSETS[name])
                    or report['inputs']['mesh'] != file_identity(expected_mesh)):
                raise RuntimeError('The approved asset/mesh is not the one this action runner uses')
            rig_identity = report['inputs']['rig']
            rig = load_verified(rig_identity['path'])
            if file_identity(rig_identity['path']) != rig_identity:
                raise RuntimeError('Approved rig changed during loading')
            expected_parents = CAT_PARENTS if name == 'cat' else DOG_PARENTS
            if (tuple(rig.get('parents', ())) != tuple(expected_parents)
                    or tuple(rig['joints'].shape) != (len(expected_parents), 3)):
                raise RuntimeError('Approved rig topology is incompatible with this retarget adapter')
            if name == 'dog' and any(k not in rig for k in ('source_to_asset', 'paw_sole_anchors', 'floor')):
                raise RuntimeError('Approved dog rig is missing retarget binding fields')
            if name == 'dog':
                basis, soles = rig['source_to_asset'], rig['paw_sole_anchors']
                if (not isinstance(basis, torch.Tensor) or basis.shape != (3, 3)
                        or not torch.isfinite(basis).all() or not basis.is_floating_point()
                        or not torch.allclose(basis.T@basis, torch.eye(3,dtype=basis.dtype),atol=1e-5)
                        or float(torch.linalg.det(basis)) <= 0):
                    raise RuntimeError('Approved dog source-to-asset basis is invalid')
                if (not isinstance(soles, torch.Tensor) or soles.shape != (4, 3)
                        or not torch.isfinite(soles).all()
                        or not math.isfinite(float(rig['floor']))):
                    raise RuntimeError('Approved dog sole/floor binding is invalid')
            approved_rigs[name] = rig
            provenance[name] = dict(report=identity, inputs=report['inputs'],
                                    preflight_implementation=report['implementation'])
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(f'{name} motion preflight blocked: {error}') from error
    return approved_rigs, provenance


def moved_to(value, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}


def make_asset(name):
    asset = load_verified(ASSETS[name])
    if name == 'cat':
        mesh = load_verified(CAT_MESH)
        asset.update(mesh_vertices=mesh['vertices'], mesh_faces=mesh['faces'])
        eigenvalues, eigenvectors = torch.linalg.eigh(asset['covariance'])
        eigenvectors[:, :, 0] *= torch.linalg.det(eigenvectors).sign()[:, None]
        asset.update(frame=eigenvectors, scale=eigenvalues.clamp_min(1e-12).sqrt())
    else:
        asset['covariance'] = (asset['frame']*asset['scale'][:, None].square())@asset['frame'].transpose(-1, -2)
        asset['normal'] = asset['frame'][:, :, 2]
    return asset


def generate_motion(speed, frames=210):
    controller = QuadrupedController(bundle=BUNDLE)
    start_state = controller.snapshot()
    local, world, controls = [], [], []
    begin = time.perf_counter()
    for index in range(math.ceil(frames/3)):
        # Action controls are explicit, not synthetic per-joint animation.
        # The learned controller generates every pose, including deceleration.
        gait, requested = action_control(index, speed)
        w, p, _ = controller.step(gait, requested)
        controls.append(dict(time_seconds=index/10, gait=gait, speed_guidance=requested))
        local.append(p); world.append(w)
    seconds = time.perf_counter()-begin
    p, w = torch.cat(local)[:frames], torch.cat(world)[:frames]
    state = controller.snapshot()
    expected = controller.step('Walk', speed)[1]
    controller.restore(state)
    assert torch.equal(expected, controller.step('Walk', speed)[1])
    return controller, dict(position=p, root=w[:, 0]-p[:, 0], initial_state=start_state,
        final_state=state, fps=30, seconds=seconds, resume_exact=True,
        recurrent_tensor_bytes=sum(v.numel()*v.element_size() for v in state.values()),
        controls=controls, source='Fresh deterministic conditional neural rollout; no recorded motion input')


def action_control(index, speed):
    if type(index) is not int or index < 0 or not 0 <= speed <= 4:
        raise ValueError('Nonnegative action index and speed in [0,4] required')
    return ('Walk' if index < 30 else 'Idle',
            speed if index < 20 else speed*max(0., (29-index)/10))


@torch.inference_mode()
def run(out, preview=False, speed=.5, cat_mode='rest_ik', animate_cat=True, dog_mode='relative', ground_paws=False, scene_radius=3,
        renderer='legacy', cat_readiness=None, dog_readiness=None):
    # No controller rollout, output directory, or CUDA allocations before this
    # guard. A numeric/rendering pass cannot override failed anatomy evidence.
    renderer_info = renderer_metadata(renderer, scene_radius)
    approved_rigs, preflight = preflight_assets(cat_readiness, dog_readiness)
    out.mkdir(parents=True, exist_ok=True)
    marker = out/('preview.json' if preview else 'metrics.json')
    if marker.exists():
        raise FileExistsError('Preserve completed experiments; select a fresh output path')
    torch.set_num_threads(4)
    total_start = time.perf_counter()
    controller, motion = generate_motion(speed)
    actors = {}
    placed = []
    for oid, name in enumerate(('cat', 'dog')):
        a = make_asset(name)
        rig = approved_rigs[name]
        # Make both characters face generally left. This is object placement,
        # not learned motion or an invented physical scale.
        rot = torch.diag(torch.tensor([-1., 1., -1.])) if name == 'cat' else torch.eye(3)
        obj = GaussianSceneObject(oid, a, digest(ASSETS[name]))
        placed_obj = obj.grounded(.75 if name == 'cat' else 1.10,
            center_xz=(-.6, .9) if name == 'cat' else (1.15, -.65), rotation=rot)
        placed.append({k: v for k, v in placed_obj.items() if k in
            ('position','covariance','colour','opacity','ids','normal','object_id','asset_sha256')})
        transform = moved_to(placed_obj['transform'], 'cuda')
        a, rig = moved_to(a, 'cuda'), moved_to(rig, 'cuda')
        skinner = DualSurfaceSkinner(a, rig)
        identity = torch.eye(3, device='cuda').repeat(len(rig['joints']), 1, 1)
        p0, c0, *_ = skinner(identity)
        rest_error = float((p0-a['position']).abs().max())
        assert rest_error < 1e-5
        targeter = (CatRetargeter(rig, controller.guidances['Walk'].cuda(), mode=cat_mode)
                    if name == 'cat' else DogRetargeter(rig, controller.guidances['Walk'].cuda(), mode=dog_mode, ground_paws=ground_paws))
        memory = GaussianVideoMemory(moved_to(placed[-1], 'cuda'), device='cuda')
        actors[name] = dict(asset=a, rig=rig, skin=skinner, targeter=targeter,
            transform=transform, memory=memory, rest_error=rest_error, poses=[], roots=[])
        save_inference_checkpoint(moved_to(rig, 'cpu'), out/f'{name}_rig.pt')
    env_source = BASE/'enclosing_scene/v1/scene.pt'
    env = load_verified(env_source); offset = env['original_count']
    env = {k: env[k][offset:] for k in ('position','covariance','colour','opacity')}
    env['ids'] = torch.arange(offset, offset+len(env['position']))
    env.update(object_id=2, asset_sha256=digest(env_source))
    scene = moved_to(merge_scene_objects(placed+[env]), 'cuda')
    # Fixed buffers: no frame-dependent re-colouring, IDs, Gaussian count, or
    # screen-space cutout. The whole scene enters a single global depth sort.
    immutable_colour = scene['colour'].clone(); immutable_ids = scene['identity_pairs'].clone()
    spans = {'cat': slice(0, 400000), 'dog': slice(400000, 600000)}
    assert len(actors['cat']['asset']['position']) == 400000
    assert len(actors['dog']['asset']['position']) == 200000
    indices = [0, 30, 60, 90, 120, 150, 208] if preview else list(range(0, 210, 2))
    # Select ONE camera from the neural root path, then hold it fixed for the
    # entire clip. This offline diagnostic framing is not bounded online state.
    trajectory_points = []
    for name, actor in actors.items():
        for index in (0, 60, 90, 209):
            result = actor['targeter'].step(motion['position'][index].cuda(), motion['root'][index].cuda())
            shift = result[1]
            t = actor['transform']
            if name == 'cat' and not animate_cat: shift = torch.zeros_like(shift)
            trajectory_points.append(t['scale']*(t['rotation']@shift)+t['translation'])
    points = torch.stack(trajectory_points)
    target = points.mean(0); target[1] = .60; target[2] = -.1
    horizontal = float(points[:, 0].max()-points[:, 0].min())+2.8
    distance = max(4.5, horizontal/(2*math.tan(math.radians(42)/2)*(960/540))+.3)
    eye = target+target.new_tensor([0., .55, distance])
    frames_out, diagnostics, deformation_times, render_times = [], [], [], []
    rotations = {name: [] for name in actors}
    root_states = {name: [] for name in actors}
    torch.cuda.reset_peak_memory_stats()
    writer = None if preview else imageio.get_writer(out/'gaussian_actions_7s.mp4', fps=15,
        codec='libx264', quality=8, macro_block_size=1)
    diagnostic_writer = None if preview else imageio.get_writer(out/'articulation_two_views_7s.mp4',
        fps=15, codec='libx264', quality=8, macro_block_size=1)
    previous_index = -2
    try:
        for index in indices:
            torch.cuda.synchronize(); start = time.perf_counter()
            row = {'pose_frame': index, 'time_seconds': index/30, 'actors': {}}
            for name, actor in actors.items():
                a, t = actor['asset'], actor['transform']
                result = actor['targeter'].step(motion['position'][index].cuda(), motion['root'][index].cuda())
                local, shift = result[:2]
                if name == 'cat' and not animate_cat:
                    local = torch.eye(3, device='cuda').repeat(len(local), 1, 1); shift = shift*0
                p, c, _, vertices, area = actor['skin'](local, shift)
                p = t['scale']*(p@t['rotation'].T)+t['translation']
                c = t['scale']**2*(t['rotation']@c@t['rotation'].T)
                # Surface normal, unlike covariance alone, carries orientation.
                triangles = vertices[actor['skin'].faces]
                normals = F.normalize(torch.linalg.cross(triangles[:,1]-triangles[:,0],
                    triangles[:,2]-triangles[:,0]), dim=-1)[actor['skin'].face_id]@t['rotation'].T
                actor['memory'].update_covariance(p, c, ids=a['ids'], dt=(index-previous_index)/30, normal=normals)
                scene['position'][spans[name]].copy_(p); scene['covariance'][spans[name]].copy_(c)
                rotations[name].append(local.cpu()); root_states[name].append(shift.cpu())
                row['actors'][name] = dict(area_min=float(area.min()), area_max=float(area.max()),
                    area_bad_fraction=float(((area<.2)|(area>5)).float().mean()),
                    below_floor_center_fraction=float((p[:,1]<-.015).float().mean()),
                    lower=p.amin(0).tolist(), upper=p.amax(0).tolist())
            previous_index = index
            torch.cuda.synchronize(); deformation_times.append(time.perf_counter()-start)
            start = time.perf_counter()
            rgb, _ = render_action_frame(**{k: scene[k] for k in ('position','covariance','colour','opacity')},
                eye=eye, target=target, height=540, width=960, ground=False, radius=scene_radius, renderer=renderer)
            pixels = (rgb*255).round().byte().cpu().numpy()
            frame = Image.fromarray(pixels)
            if writer is not None: writer.append_data(pixels)
            diagnostic_panels = []
            for name in ('cat', 'dog'):
                row_slice = spans[name]
                p = scene['position'][row_slice]
                pivot = (p.amin(0)+p.amax(0))*.5
                direction = pivot.new_tensor([.45,.16,1.])
                cam = pivot+F.normalize(direction,dim=0)*2.3
                oblique, _ = render_action_frame(p,scene['covariance'][row_slice],scene['colour'][row_slice],
                    scene['opacity'][row_slice],cam,pivot,height=384,width=512,ground=True,radius=3,renderer=renderer)
                diagnostic_panels.append((oblique*255).round().byte().cpu().numpy())
            diagnostic = Image.fromarray(np.concatenate(diagnostic_panels,axis=1))
            if diagnostic_writer is not None: diagnostic_writer.append_data(np.asarray(diagnostic))
            mean, _, depth, *_ = project(scene['position'][:600000],scene['covariance'][:600000],eye,target,540,960)
            row['frustum_fraction'] = float(((depth>.02)&(mean[:,0]>=0)&(mean[:,0]<960)&(mean[:,1]>=0)&(mean[:,1]<540)).float().mean())
            torch.cuda.synchronize(); render_times.append(time.perf_counter()-start)
            if preview or index in (0,30,60,90,120,150,208):
                frame.save(out/f'frame_{index//2:03d}.png')
                diagnostic.save(out/f'oblique_{index//2:03d}.png')
                frames_out.append((index, frame, diagnostic))
                print(json.dumps(row), flush=True)
            diagnostics.append(row)
    finally:
        if writer is not None: writer.close()
        if diagnostic_writer is not None: diagnostic_writer.close()
    assert torch.equal(scene['colour'], immutable_colour)
    assert torch.equal(scene['identity_pairs'], immutable_ids)
    for label, column in (('scene',1),('oblique',2)):
        images = [x[column] for x in frames_out]
        sheet = Image.new('RGB',(images[0].width, (images[0].height+24)*len(images)))
        for j, im in enumerate(images):
            y = j*(im.height+24); sheet.paste(im,(0,y+24))
            ImageDraw.Draw(sheet).text((8,y+5),f'{frames_out[j][0]/30:.2f}s | neural walk then idle | fixed scene camera',fill='white')
        sheet.save(out/f'{label}_sheet.jpg')
    save_inference_checkpoint(motion, out/'neural_motion.pt')
    save_inference_checkpoint(dict(rotations={k:torch.stack(v) for k,v in rotations.items()},
        root_translations={k:torch.stack(v) for k,v in root_states.items()}, pose_indices=indices),out/'retargeted.pt')
    metadata=json.loads((BUNDLE/'export_report.json').read_text())
    report = dict(classification='Fresh pretrained conditional skeletal motion applied to persistent 3D Gaussian animals',
        preview_only=preview, cat_animated=animate_cat, cat_mode=cat_mode, dog_mode=dog_mode, paw_grounding_heuristic=ground_paws,
        scene_radius=scene_radius if renderer == 'legacy' else None,
        renderer=renderer_info, asset_preflight=preflight, approved_rigs_loaded_without_rebuilding=True,
        action='walk, decelerate, idle', speed_guidance=speed,
        source_weights_sha256=metadata['source_weights_sha256'], source_assets={k:digest(v) for k,v in ASSETS.items()},
        source_code_sha256={name:digest(Path(__file__).parent/name) for name in
            ('generate_gaussian_actions.py','controller_rig.py','dog_articulation.py','quadruped_controller.py')},
        upstream='https://github.com/facebookresearch/ai4animationpy', pretrained_motion_license='CC-BY-NC-4.0',
        frames=len(indices), fps=15, duration_seconds=None if preview else 7,
        camera=dict(eye=eye.tolist(),target=target.tolist(),fixed=True),
        gaussian_count=len(scene['position']), persistent_ids_and_colours_exact=True,
        motion_generation_seconds=motion['seconds'], controller_recurrent_bytes=motion['recurrent_tensor_bytes'],
        deformation_and_memory_update_median_seconds=float(np.median(deformation_times)),
        scene_plus_two_diagnostic_renders_median_seconds=float(np.median(render_times)),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), total_wall_seconds=time.perf_counter()-total_start,
        per_actor_memory={k:v['memory'].memory_bytes() for k,v in actors.items()},
        rest_position_max_error={k:v['rest_error'] for k,v in actors.items()},
        wan_calls=0, new_weights_trained=False, input_video_motion_used=False, camera_orbit=False,
        quality_accepted=False, quality_matched_wan_speedup=None,
        limitations=['Heuristic rigs; source mesh has malformed and partly fused paws',
            'No validated contact, collision, fur dynamics or physically grounded animal anatomy',
            'Headless straight-line controller omits native root yaw, PID and contact postprocessing',
            'Cat and dog share this demonstration gait sequence; not independently planned interaction',
            'Gaussian memory excludes mesh/rig/model/render buffers and this evaluation trajectory archive',
            'Static garden depth-shell seams and asset texture flaws remain',
            'Asset preflight only permits a trial; this rendered motion still requires independent quality review',
            'Adaptive rendering changes footprint coverage, not learned anatomy, missing details, or generative capability',
            'Not a trained Wan-native Gaussian video model; no quality-matched speedup claim'], diagnostics=diagnostics)
    marker.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='diagnostics'},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=BASE/'gaussian_action_video/v1')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--speed',type=float,default=.5)
    parser.add_argument('--cat-mode',choices=('rest_ik','relative'),default='rest_ik')
    parser.add_argument('--static-cat',action='store_true')
    parser.add_argument('--dog-mode',choices=('relative','aligned','aligned_legs'),default='relative')
    parser.add_argument('--ground-paws',action='store_true')
    parser.add_argument('--scene-radius',type=int,choices=(3,5),default=3)
    parser.add_argument('--renderer',choices=('legacy','adaptive'),default='legacy')
    parser.add_argument('--cat-readiness',type=Path,default=DEFAULT_PREFLIGHT['cat'],
                        help='Fresh accepted asset/mesh/rig preflight; current default is blocked')
    parser.add_argument('--dog-readiness',type=Path,default=DEFAULT_PREFLIGHT['dog'],
                        help='Fresh accepted asset/mesh/rig preflight; no bypass option')
    args=parser.parse_args()
    with keep_windows_awake():
        run(args.out,args.preview,args.speed,args.cat_mode,not args.static_cat,args.dog_mode,args.ground_paws,args.scene_radius,
            renderer=args.renderer,cat_readiness=args.cat_readiness,dog_readiness=args.dog_readiness)
