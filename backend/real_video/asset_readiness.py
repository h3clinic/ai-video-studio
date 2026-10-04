"""CPU-only, fail-closed prerequisites for articulating persistent splat assets.

Renderability is not anatomical readiness. Connected lower-mesh components are
measurements, NOT limb labels or an anatomy score. A separate, hash-bound review
must establish anatomy, bind pose, skinning ownership and contact readiness.
The gate does not establish that any motion generator is trained or generalizes.
Only load trusted local checkpoint files: torch checkpoints may contain pickle.
"""
from collections.abc import Mapping
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


REVIEW_DIMENSIONS = ('anatomical_parts', 'bind_pose', 'skinning_ownership',
                     'contact_readiness', 'detail_identity')


def file_identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def _tensor(data, key, shape=None, integer=False):
    value = data.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f'{key}: missing tensor')
    value = value.detach().cpu()
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f'{key}: expected shape {tuple(shape)}, got {tuple(value.shape)}')
    if integer and value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f'{key}: integer tensor required')
    if not torch.isfinite(value).all():
        raise ValueError(f'{key}: nonfinite values')
    return value


def lower_mesh_components(vertices, faces, fractions=(.1, .2, .3, .4), up_axis=1):
    """Induced vertex-edge graph below fractions of total mesh height.

    Edges come from all original triangle edges with both endpoints retained.
    The horizontal plane is NOT capped. Raised paws, tails, pose, triangulation,
    and disconnected fragments alter these counts; four components is not proof
    of four anatomically correct limbs. These diagnostics never grant approval.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0 or not np.isfinite(v).all():
        raise ValueError('Expected nonempty finite V x 3 vertices')
    if f.ndim != 2 or f.shape[1] != 3 or (len(f) and (f.min() < 0 or f.max() >= len(v))):
        raise ValueError('Invalid triangular face indices')
    if up_axis not in (0, 1, 2) or any(not 0 < x < 1 for x in fractions):
        raise ValueError('Invalid up axis or section fractions')
    low, high = v.min(0), v.max(0)
    if high[up_axis] <= low[up_axis]:
        raise ValueError('Zero mesh height')
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    bands = []
    for fraction in fractions:
        cut = float(low[up_axis] + fraction * (high[up_axis] - low[up_axis]))
        ids = np.flatnonzero(v[:, up_axis] <= cut)
        lookup = np.full(len(v), -1, dtype=np.int64)
        lookup[ids] = np.arange(len(ids))
        selected = edges[(lookup[edges] >= 0).all(1)]
        selected = lookup[selected]
        graph = coo_matrix((np.ones(len(selected), dtype=np.uint8),
                            (selected[:, 0], selected[:, 1])), shape=(len(ids), len(ids)))
        count, labels = connected_components(graph.tocsr(), directed=False)
        counts = np.bincount(labels, minlength=count)
        largest = np.argsort(-counts)[:12]
        parts = []
        for index in largest:
            points = v[ids[labels == index]]
            parts.append({'vertex_count': int(len(points)), 'minimum': points.min(0).tolist(),
                          'maximum': points.max(0).tolist(), 'centroid': points.mean(0).tolist()})
        bands.append({'height_fraction': fraction, 'cut_coordinate': cut,
                      'retained_vertices': int(len(ids)), 'component_count': int(count),
                      'components_with_at_least_10_vertices': int((counts >= 10).sum()),
                      'largest_components': parts})
    return {'up_axis': up_axis, 'bands': bands, 'semantic_limb_count': None,
            'anatomical_pass': None, 'interpretation': 'Evidence only; counts do not establish anatomy.'}


def review_template(asset_path, mesh_path=None, rig_path=None, evidence_paths=()):
    """Unaccepted template, requiring real observations, never auto-filled passes."""
    inputs = {'asset': file_identity(asset_path),
              'mesh': file_identity(mesh_path or asset_path),
              'rig': file_identity(rig_path) if rig_path else None}
    return {'schema_version': 1, 'inputs': inputs,
            'reviewer': {'name': '', 'kind': None},
            'evidence': [dict(file_identity(p), id=f'view_{i}', view=None, observation='')
                         for i, p in enumerate(evidence_paths)],
            'dimensions': {k: {'status': None, 'observation': '', 'evidence_ids': []}
                           for k in REVIEW_DIMENSIONS}, 'limitations': []}


def _review_decision(review, inputs):
    pending, failed = [], []
    if not isinstance(review, Mapping):
        return {'accepted': False, 'status': 'pending', 'reasons': ['An explicit anatomy review is required.']}
    if review.get('schema_version') != 1 or review.get('inputs') != inputs:
        failed.append('Anatomy review is stale, for different inputs, or has an unsupported schema.')
    reviewer = review.get('reviewer', {})
    if not isinstance(reviewer, Mapping) or reviewer.get('kind') not in ('human', 'agent') or not reviewer.get('name'):
        pending.append('Explicit reviewer identity is required.')
    valid, views, paths = set(), set(), set()
    evidence = review.get('evidence', [])
    if not isinstance(evidence, list):
        evidence = []
    for entry in evidence:
        if not isinstance(entry, Mapping):
            failed.append('Malformed visual evidence.')
            continue
        try:
            actual = file_identity(entry['path'])
            if actual['sha256'] != entry.get('sha256'):
                raise ValueError('Stale visual evidence')
            if (not entry.get('id') or entry['id'] in valid or actual['path'] in paths
                    or not entry.get('observation') or entry.get('view') not in ('front', 'side', 'back', 'oblique', 'above')):
                raise ValueError('Missing or duplicate visual evidence')
            valid.add(entry['id']); paths.add(actual['path']); views.add(entry['view'])
        except (OSError, KeyError, TypeError, ValueError):
            failed.append('Unreadable, changed, duplicate, or unobserved visual evidence.')
    if len(valid) < 2 or 'oblique' not in views or not views.intersection(('front', 'side', 'back')):
        pending.append('At least two distinct observed views, including oblique and front/side/back, are required.')
    dimensions = review.get('dimensions', {})
    if not isinstance(dimensions, Mapping):
        dimensions = {}
    for key in REVIEW_DIMENSIONS:
        entry = dimensions.get(key, {})
        if not isinstance(entry, Mapping):
            entry = {}
        status = entry.get('status')
        if status in ('fail', 'uncertain'):
            failed.append(f'{key}: reviewer marked {status}.')
        elif status != 'pass':
            pending.append(f'{key}: explicit review missing.')
        cited = entry.get('evidence_ids')
        if (not entry.get('observation') or not isinstance(cited, list) or not cited
                or any(not isinstance(x, str) or x not in valid for x in cited)):
            pending.append(f'{key}: observed, verified visual evidence must be cited.')
    accepted = not failed and not pending
    return {'accepted': accepted, 'status': 'accepted' if accepted else ('rejected' if failed else 'pending'),
            'reasons': failed + pending}


def audit_asset(asset_path, mesh_path=None, semantic_review=None, rig_path=None, up_axis=1):
    """Inspect a trusted local asset on CPU; never schedule a render or motion job.

    Accepts either a semantic review dict or its JSON path. A renderer-ready
    asset still fails motion preflight until shared surface, rig, and semantic
    prerequisites pass. No threshold here is a perceptual quality metric.
    """
    inputs = {'asset': file_identity(asset_path), 'mesh': file_identity(mesh_path or asset_path),
              'rig': file_identity(rig_path) if rig_path else None}
    asset = torch.load(asset_path, map_location='cpu', weights_only=False)
    mesh = torch.load(mesh_path, map_location='cpu', weights_only=False) if mesh_path else asset
    checks, measurements = {}, {}

    def check(name, function):
        try:
            result = function()
            checks[name] = {'status': 'pass', 'reason': 'Structural check passed, not visual approval.'}
            return result
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
            checks[name] = {'status': 'fail', 'reason': str(exc)}
            return None

    def gaussians():
        position = _tensor(asset, 'position')
        if position.ndim != 2 or position.shape[1] != 3 or not len(position):
            raise ValueError('position: nonempty N x 3 required')
        n = len(position)
        colour = _tensor(asset, 'colour', (n, 3))
        alpha = _tensor(asset, 'opacity', (n,))
        ids = _tensor(asset, 'ids', (n,), integer=True)
        if len(torch.unique(ids)) != n:
            raise ValueError('Duplicate persistent Gaussian identities')
        if (colour < 0).any() or (colour > 1).any() or (alpha < 0).any() or (alpha > 1).any():
            raise ValueError('Colour/opacity outside [0,1]')
        if 'covariance' in asset:
            cov = _tensor(asset, 'covariance', (n, 3, 3)).double()
            if not torch.allclose(cov, cov.transpose(-1, -2), atol=1e-8, rtol=1e-5):
                raise ValueError('Covariance not symmetric')
            eig = torch.linalg.eigvalsh(cov)
            if not (eig > 0).all():
                raise ValueError('Covariance not positive definite')
            scale = eig.sqrt()
        else:
            frame = _tensor(asset, 'frame', (n, 3, 3))
            scale = _tensor(asset, 'scale', (n, 3))
            if not (scale > 0).all():
                raise ValueError('Nonpositive Gaussian scale')
            if not torch.allclose(frame.transpose(-1, -2) @ frame, torch.eye(3).expand(n, 3, 3), atol=2e-4, rtol=1e-4) or not (torch.linalg.det(frame) > 0).all():
                raise ValueError('Gaussian frame is not a proper orthonormal basis')
        measurements['gaussians'] = {'count': n, 'minimum': position.amin(0).tolist(),
                                    'maximum': position.amax(0).tolist(),
                                    'radius_sigma_quantiles': torch.quantile(scale.double().amax(-1), torch.tensor([.5,.9,.99,1.], dtype=torch.double)).tolist(),
                                    'identity_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
        return position

    position = check('gaussian_storage', gaussians)

    def support():
        if position is None:
            raise ValueError('Cannot bind invalid Gaussian storage to a surface')
        vertices = _tensor(mesh, 'vertices' if 'vertices' in mesh else 'mesh_vertices')
        faces = _tensor(mesh, 'faces' if 'faces' in mesh else 'mesh_faces', integer=True).long()
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError('Invalid support vertices')
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError('Invalid support faces')
        extent = vertices.amax(0) - vertices.amin(0)
        if not (extent > 0).all():
            raise ValueError('Degenerate support extent')
        fid = _tensor(asset, 'face_id', (len(asset['position']),), integer=True).long()
        bary = _tensor(asset, 'barycentric', (len(fid), 3))
        if fid.min() < 0 or fid.max() >= len(faces) or (bary < -1e-6).any() or not torch.allclose(bary.sum(1), torch.ones(len(fid)), atol=1e-5):
            raise ValueError('Invalid fixed surface attachments')
        triangles = vertices[faces]
        area2 = torch.linalg.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]).norm(dim=-1)
        if (area2[torch.unique(fid)] <= float(extent.max())**2 * 1e-12).any():
            raise ValueError('Active Gaussian support face is collapsed')
        expected = (triangles[fid] * bary[..., None]).sum(1)
        error = (expected - asset['position']).norm(dim=-1)
        max_error = float(error.max())
        measurements['support'] = {'vertex_count': len(vertices), 'face_count': len(faces),
                                   'extent': extent.tolist(), 'up_axis': up_axis,
                                   'max_attachment_error': max_error,
                                   'zero_area_faces': int((area2 <= 1e-12).sum()),
                                   'ground_coordinate': float(vertices[:, up_axis].min()),
                                   'physical_units': 'Not calibrated; source-coordinate extent only'}
        if max_error > max(1e-5, float(extent.max()) * 1e-4):
            raise ValueError('Gaussian positions do not match barycentric support (zero-offset schema)')
        measurements['lower_limb_cross_sections'] = lower_mesh_components(vertices.numpy(), faces.numpy(), up_axis=up_axis)
        return vertices

    vertices = check('surface_attachment', support)

    def rig_check():
        if not rig_path:
            raise ValueError('A bound rig is required before motion generation')
        if vertices is None:
            raise ValueError('Cannot check rig without valid surface')
        rig = torch.load(rig_path, map_location='cpu', weights_only=False)
        joints = _tensor(rig, 'joints')
        if joints.ndim != 2 or joints.shape[1] != 3 or not len(joints):
            raise ValueError('Invalid rig joints')
        parents = rig.get('parents')
        if not isinstance(parents, (list, tuple)) or len(parents) != len(joints) or parents[0] != -1 or any(type(p) is not int or p < 0 or p >= i for i, p in enumerate(parents) if i):
            raise ValueError('Invalid rooted topological parent hierarchy')
        weights = _tensor(rig, 'vertex_weights', (len(vertices), len(joints)))
        if (weights < 0).any() or not torch.allclose(weights.sum(1), torch.ones(len(vertices)), atol=1e-4):
            raise ValueError('Skinning weights must be nonnegative and normalized')
        length = torch.stack([(joints[i] - joints[p]).norm() for i, p in enumerate(parents) if i])
        if not (length > 1e-8).all():
            raise ValueError('Zero length rig bone')
        info = {'joint_count': len(joints), 'min_bone_length': float(length.min()),
                'max_bone_length': float(length.max()), 'bind_pose_semantics_verified': False,
                'contact_validated': False}
        if 'paw_sole_anchors' in rig:
            anchors = _tensor(rig, 'paw_sole_anchors')
            if anchors.ndim != 2 or anchors.shape[1] != 3:
                raise ValueError('Invalid paw anchors')
            info['sole_anchor_ground_offsets'] = (anchors[:, up_axis]-vertices[:, up_axis].min()).tolist()
            info['sole_anchor_scope'] = 'Manual anchor heights only, not a solved contact constraint.'
        measurements['rig'] = info

    check('rig_structure', rig_check)
    if isinstance(semantic_review, (str, Path)):
        semantic_review = json.loads(Path(semantic_review).read_text(encoding='utf-8'))
    semantic = _review_decision(semantic_review, inputs)
    ready = all(value['status'] == 'pass' for value in checks.values()) and semantic['accepted']
    reasons = [f'{key}: {value["reason"]}' for key, value in checks.items() if value['status'] != 'pass'] + semantic['reasons']
    return {'schema_version': 1, 'inputs': inputs, 'implementation': file_identity(__file__),
            'device': 'cpu', 'checks': checks,
            'measurements': measurements, 'semantic_review': semantic_review,
            'semantic_decision': semantic,
            'decision': {'renderable': position is not None, 'surface_bound': vertices is not None,
                         'rigged_generation_ready': ready, 'motion_jobs_allowed': ready,
                         'status': 'ready' if ready else 'blocked', 'reasons': reasons},
            'claim_scope': 'Preflight prerequisites only; not proof of autonomous or Gaussian-native generation.',
            'next_action': 'Motion trial permitted, followed by artifact-linked quality review.' if ready else
                           'Repair geometry/rig or supply missing evidence. Do not repeat motion tuning on unchanged rejected assets.'}


def require_motion_ready(report):
    """Scheduling guard: reject incomplete, stale, or failed preflight reports."""
    if not isinstance(report, Mapping) or report.get('schema_version') != 1:
        raise RuntimeError('Motion blocked: missing or incompatible asset preflight')
    inputs = report.get('inputs')
    if not isinstance(inputs, Mapping) or set(inputs) != {'asset', 'mesh', 'rig'}:
        raise RuntimeError('Motion blocked: incomplete input binding')
    try:
        if report.get('implementation') != file_identity(__file__):
            raise RuntimeError('Motion blocked: preflight implementation changed; rerun the audit')
        for key, identity in inputs.items():
            if not isinstance(identity, Mapping) or file_identity(identity['path']) != identity:
                raise RuntimeError(f'Motion blocked: missing or changed {key} input')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('Motion blocked: unreadable or malformed input binding') from exc
    semantic = _review_decision(report.get('semantic_review'), inputs)
    checks = report.get('checks', {})
    if (not isinstance(checks, Mapping) or set(checks) != {'gaussian_storage', 'surface_attachment', 'rig_structure'}
            or any(not isinstance(v, Mapping) or v.get('status') != 'pass' for v in checks.values())
            or not semantic['accepted'] or report.get('decision', {}).get('motion_jobs_allowed') is not True):
        raise RuntimeError('Motion blocked: asset anatomy/rig prerequisites are not accepted')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', required=True)
    parser.add_argument('--mesh')
    parser.add_argument('--rig')
    parser.add_argument('--review')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = audit_asset(args.asset, args.mesh, args.review, args.rig)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['decision'], indent=2))
    return 0 if report['decision']['motion_jobs_allowed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
