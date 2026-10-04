"""Independent geometric audit of source-conditioned Gaussian surface motion.

This is a failure-oriented diagnostic, not an anatomical or generative-quality
metric. World-space thresholds are in the inferred asset's arbitrary units.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from .checkpoint_io import digest, load_verified
from .surface_skin3d import SurfaceSkinner, triangle_basis


DEFAULT_ROOT = Path('artifacts/real_video/true3d/v2_motion/surface_refined')
DEFAULT_ASSET = Path('artifacts/real_video/true3d/v1/cat_asset.pt')


def quantiles(values):
    points = [0., .001, .01, .5, .99, .999, 1.]
    return dict(zip(map(str, points), torch.quantile(values.double(),
                    torch.tensor(points, dtype=torch.float64)).tolist()))


@torch.no_grad()
def audit(asset_path=DEFAULT_ASSET, root=DEFAULT_ROOT):
    torch.set_num_threads(4)
    started = time.perf_counter()
    asset = load_verified(asset_path)
    rig = load_verified(root / 'rig.pt')
    motion = load_verified(root / 'motion.pt')
    if not torch.equal(asset['ids'], rig['ids']):
        raise ValueError('Asset and motion rig Gaussian IDs do not match')
    support = SurfaceSkinner(asset, rig)
    rest = asset['position']
    nearest_distance, nearest_ids = cKDTree(rest.numpy()).query(rest.numpy(), k=2)
    rest_distance = torch.from_numpy(nearest_distance[:, 1])
    neighbor = torch.from_numpy(nearest_ids[:, 1])
    close = (rest_distance > 1e-7) & (rest_distance < .005)
    if not bool(close.any()):
        raise ValueError('No close rest-neighbor pairs to audit')
    active_area = support.rest_area[support.active_faces].double()
    rest_vertices = asset['mesh_vertices']
    faces = asset['mesh_faces'].long()
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = torch.unique(edges.sort(dim=-1).values, dim=0)
    rest_edges = (rest_vertices[edges[:, 0]] - rest_vertices[edges[:, 1]]).norm(dim=-1)
    scale = float((rest.amax(0) - rest.amin(0)).max())
    robust_edges = rest_edges > scale * 1e-5
    rows = []
    for frame, (rotation, translation) in enumerate(zip(motion['local_rotation'], motion['root_translation'])):
        position, covariance, _, vertices, area_ratio = support(rotation, translation)
        gap = (position - position[neighbor]).norm(dim=-1)
        stretch = gap[close] / rest_distance[close]
        basis, _ = triangle_basis(vertices[faces])
        gradient = (basis @ support.inverse_basis)[support.active_faces]
        singular = torch.linalg.svdvals(gradient)
        eig = torch.linalg.eigvalsh(covariance)
        bad_area = (area_ratio < .1) | (area_ratio > 10.)
        bad_stretch = (singular[:, -1] < .25) | (singular[:, 0] > 4.)
        moved_edges = (vertices[edges[:, 0]] - vertices[edges[:, 1]]).norm(dim=-1)
        edge_ratio = moved_edges[robust_edges] / rest_edges[robust_edges]
        rows.append({
            'frame': frame,
            'close_neighbor_stretch_quantiles': quantiles(stretch),
            'close_neighbor_max_gap': float(gap[close].max()),
            'close_neighbor_gap_over_0_05_count': int((close & (gap > .05)).sum()),
            'active_face_area_ratio_quantiles': quantiles(area_ratio),
            'active_face_area_ratio_outside_0_1_to_10_count': int(bad_area.sum()),
            'active_face_area_ratio_outside_0_1_to_10_fraction': float(bad_area.float().mean()),
            'active_face_area_ratio_outside_0_1_to_10_rest_area_fraction': float(active_area[bad_area].sum()/active_area.sum()),
            'deformation_singular_value_min': float(singular.min()),
            'deformation_singular_value_max': float(singular.max()),
            'active_face_stretch_outside_0_25_to_4_count': int(bad_stretch.sum()),
            'active_face_stretch_outside_0_25_to_4_rest_area_fraction': float(active_area[bad_stretch].sum()/active_area.sum()),
            'robust_mesh_edge_ratio_quantiles': quantiles(edge_ratio),
            'gaussian_covariance_eigenvalue_min': float(eig.min()),
            'gaussian_covariance_eigenvalue_max': float(eig.max()),
            'gaussian_max_standard_deviation_over_0_1_count': int((eig[:, -1].clamp_min(0).sqrt() > .1).sum()),
            'gaussian_center_min_y': float(position[:, 1].min()),
            'gaussian_centers_below_ground_count': int((position[:, 1] < -1e-6).sum()),
        })
    feet = motion['joint_positions'][:, [7, 10, 13, 16]]
    horizontal_step = (feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]]).norm(dim=-1)
    near_ground = (feet[1:, :, 1] < .05) & (feet[:-1, :, 1] < .05)
    report = {
        'purpose': 'Independent failure-oriented audit, not a quality-pass certificate.',
        'classification': 'Single-Wan-video-conditioned, manually assisted 3D motion reconstruction; NOT new motion generation or training a video foundation model.',
        'appearance_scope': 'Geometry audit uses original XYZ asset. It does not evaluate fitted appearance PSNR or unseen-view texture accuracy.',
        'units': 'Arbitrary inferred-asset world units, not calibrated meters.',
        'asset_path': str(asset_path), 'asset_sha256': digest(asset_path),
        'rig_sha256': digest(root / 'rig.pt'), 'motion_sha256': digest(root / 'motion.pt'),
        'source_video_sha256': motion['source_video_sha256'],
        'frames': len(rows), 'fps': float(motion['fps']),
        'native_duration_seconds': len(rows) / float(motion['fps']),
        'persistent_gaussian_ids_match': True,
        'definitions': {
            'close_neighbor': 'For each Gaussian, select its nearest other Gaussian in canonical XYZ. Count directed pairs with rest distance >1e-7 and <0.005. Reciprocal pairs are counted twice. This tests local drift but not physical anatomy.',
            'close_neighbor_count': int(close.sum()),
            'excluded_near_zero_neighbor_distance_count': int((rest_distance <= 1e-7).sum()),
            'large_gap': 'Deformed distance >0.05 for a close rest-neighbor pair; threshold is diagnostic, not perceptual calibration.',
            'active_faces': 'Mesh triangles carrying at least one stored Gaussian; diagnostics exclude unused faces.',
            'active_face_count': int(support.active_faces.sum()),
            'face_area_threshold': 'Area ratio outside [0.1,10] is flagged; no claim this is an anatomical threshold.',
            'gradient': 'Face basis [edge1,edge2,unit normal] times inverse canonical basis; normal thickness kept constant. Singular-value range [0.25,4] is a diagnostic tolerance.',
            'edge_ratio': 'Unique undirected mesh edges with canonical length >1e-5 times maximum asset extent; excludes near-degenerate edges whose relative errors are ill-conditioned.',
            'robust_edge_count': int(robust_edges.sum()),
            'excluded_tiny_edge_count': int((~robust_edges).sum()),
            'ground': 'Nominal world y=0 plane; centers below it indicate penetration, but positive center height alone does not ensure a Gaussian footprint clears the plane.',
            'near_ground_step': 'Horizontal endpoint displacement when the same foot joint is below y=0.05 in both neighboring frames. It is not verified stance/foot-sliding ground truth.',
        },
        'visual_review': {
            'reviewed': str(root / 'motion_preview.jpg'),
            'sampled_source_frames': [2, 6, 10, 18, 26, 32],
            'improvements': ['Catastrophic head twist removed compared with manual_refined.', 'Detached belly fragments substantially resolved by shared-surface deformation.'],
            'remaining_failures': ['Feet curl and shrink unnaturally; this does not reproduce a convincing cat gait.', 'Rear far leg remains stub-like rather than anatomically verified.', 'Foot-ground contact is not established; the preview hides ground with ground=False.', 'Head exits the fixed preview crop in source frame32.', 'Appearance still has green contamination near underside/feet and does not validate unseen geometry.'],
            'verdict': 'Improved continuity; FAIL natural-motion and physical-contact quality gates.',
        },
        'summary': {
            'maximum_large_gap_pair_count_in_any_frame': max(row['close_neighbor_gap_over_0_05_count'] for row in rows),
            'maximum_close_neighbor_gap': max(row['close_neighbor_max_gap'] for row in rows),
            'maximum_close_neighbor_p99_9_stretch': max(row['close_neighbor_stretch_quantiles']['0.999'] for row in rows),
            'minimum_active_face_area_ratio': min(row['active_face_area_ratio_quantiles']['0.0'] for row in rows),
            'maximum_active_face_area_ratio': max(row['active_face_area_ratio_quantiles']['1.0'] for row in rows),
            'minimum_deformation_singular_value': min(row['deformation_singular_value_min'] for row in rows),
            'maximum_deformation_singular_value': max(row['deformation_singular_value_max'] for row in rows),
            'maximum_bad_area_face_fraction': max(row['active_face_area_ratio_outside_0_1_to_10_fraction'] for row in rows),
            'maximum_bad_area_rest_surface_fraction': max(row['active_face_area_ratio_outside_0_1_to_10_rest_area_fraction'] for row in rows),
            'minimum_gaussian_center_y': min(row['gaussian_center_min_y'] for row in rows),
            'foot_joint_y_minimum_each': feet[:, :, 1].amin(0).tolist(),
            'foot_joint_y_maximum_each': feet[:, :, 1].amax(0).tolist(),
            'near_ground_horizontal_step_maximum_each': (horizontal_step * near_ground).amax(0).tolist(),
        },
        'limitations': ['No multi-view anatomical ground truth.', 'No self-intersection/collision guarantee or topology-quality certification.', 'Face-basis orientation uses each deformed normal, so positive covariance/area does NOT prove no folds.', 'Continuity does not establish correct part ownership or natural gait.', 'No equivalent-quality generation compute or memory saving established.'],
        'per_frame': rows,
        'elapsed_cpu_seconds': time.perf_counter() - started,
    }
    target = root / 'independent_audit.json'
    if target.exists():
        raise FileExistsError(f'Preserve prior audit: {target}')
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'path': str(target), 'summary': report['summary'], 'elapsed_cpu_seconds': report['elapsed_cpu_seconds']}, indent=2))
    return report


@torch.no_grad()
def audit_feet(asset_path=DEFAULT_ASSET, root=DEFAULT_ROOT):
    """Separate endpoint fitting errors from errors introduced by skinning."""
    from .fit_motion3d import NAMES, project_points
    torch.set_num_threads(4)
    asset = load_verified(asset_path)
    rig = load_verified(root / 'rig.pt')
    motion = load_verified(root / 'motion.pt')
    camera = motion['camera']
    support = SurfaceSkinner(asset, rig)
    frame = len(motion['local_rotation']) - 1
    _, _, _, vertices, _ = support(motion['local_rotation'][frame], motion['root_translation'][frame])
    feet = torch.tensor([7, 10, 13, 16])
    distance = (asset['mesh_vertices'][:, None] - rig['joints'][feet][None]).norm(dim=-1)
    nearest = distance.argmin(0)

    def source_pixels(points):
        return project_points(points, camera) * camera['size'] / camera['width'] - points.new_tensor([
            camera['pad_x'] - camera['x0'], camera['pad_y'] - camera['y0']])

    predicted = source_pixels(motion['joint_positions'])
    error = (predicted - motion['landmarks_source']).norm(dim=-1)
    confidence = motion['landmark_confidence']
    foot_rows = []
    for slot, joint in enumerate(feet.tolist()):
        vertex = int(nearest[slot])
        foot_rows.append({
            'joint': joint, 'name': NAMES[joint],
            'closest_canonical_mesh_vertex': vertex,
            'canonical_joint_to_mesh_gap': float(distance[vertex, slot]),
            'vertex_skin_weights': dict(zip(NAMES, rig['vertex_weights'][vertex].tolist())),
            'fitted_final_joint_source_pixels': predicted[frame, joint].tolist(),
            'final_manual_or_tracked_target_source_pixels': motion['landmarks_source'][frame, joint].tolist(),
            'actual_final_vertex_source_pixels': source_pixels(vertices[vertex:vertex+1])[0].tolist(),
            'final_vertex_to_fitted_joint_world_gap': float((vertices[vertex] - motion['joint_positions'][frame, joint]).norm()),
        })
    record = {
        'classification': 'Manually assisted single-video reconstruction diagnostic, not generated motion.',
        'frame': frame,
        'asset_sha256': digest(asset_path), 'rig_sha256': digest(root / 'rig.pt'),
        'motion_sha256': digest(root / 'motion.pt'),
        'selection': 'One nearest canonical mesh vertex per inferred foot joint; tracks the same material vertex at the final frame. Nearest surface is not verified sole anatomy.',
        'dominant_vertex_weights_count': dict(zip(NAMES, torch.bincount(rig['vertex_weights'].argmax(-1), minlength=len(NAMES)).tolist())),
        'dominant_gaussian_weights_count': dict(zip(NAMES, torch.bincount(rig['weight'].argmax(-1), minlength=len(NAMES)).tolist())),
        'weighted_mean_source_pixel_error_each_joint': dict(zip(NAMES, ((error * confidence).sum(0)/confidence.sum(0).clamp_min(1e-8)).tolist())),
        'maximum_accepted_source_pixel_error_each_joint': dict(zip(NAMES, error.masked_fill(confidence <= .1, 0).amax(0).tolist())),
        'foot_details': foot_rows,
        'front_far_rest_segment_lengths': [float((rig['joints'][j] - rig['joints'][p]).norm()) for j, p in [(15, 14), (16, 15)]],
        'front_far_final_chain_fitted_source_pixels': predicted[frame, [14, 15, 16]].tolist(),
        'front_far_final_chain_target_source_pixels': motion['landmarks_source'][frame, [14, 15, 16]].tolist(),
        'findings': [
            'Front-far mesh paw follows the fitted endpoint within about5sourcepixels, but fitted endpoint misses the manually marked sourcefoot by over100pixels. This is primarily an IK/rest-anatomy discrepancy, not missing distal bone binding.',
            'Rear-far canonical footjoint has no nearby matching mesh sole; closest material vertex is over0.10worldunits away and mixes near/far leg weights. Both inferred geometry and attachment are unreliable there.',
            'Endpoint indices have zero dominant skin weights by this FK convention: incoming segments move with their parent transform. Zero endpoint weights are not by themselves a bug.',
        ],
    }
    target = root / 'independent_foot_audit.json'
    if target.exists():
        raise FileExistsError(f'Preserve prior foot audit: {target}')
    target.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'path': str(target), 'foot_details': foot_rows}, indent=2))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', type=Path, default=DEFAULT_ASSET)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--foot-details', action='store_true')
    args = parser.parse_args()
    (audit_feet if args.foot_details else audit)(args.asset, args.root)
