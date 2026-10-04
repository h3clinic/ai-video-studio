"""Persistent object-local Gaussian assets in one genuinely 3D scene.

Insertion is a proper similarity transform, not an RGB overlay. Stable identity
is the pair (object_id, local Gaussian id); raster row numbers are NOT identity.
Units and target heights are explicitly user/experiment chosen, not recovered
metres. This module neither generates geometry nor learns articulated motion.
The shared renderer uses global center-depth sorting and finite splat stencils.
"""
import math
import numbers
import re

import torch


def _float_tensor(value, shape, name, *, like=None):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name}: expected tensor shape {tuple(shape)}')
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name}: finite floating tensor required')
    if like is not None and (value.dtype != like.dtype or value.device != like.device):
        raise ValueError(f'{name}: dtype/device must match positions')
    return value


def _rotation(value, like):
    value = torch.eye(3, dtype=like.dtype, device=like.device) if value is None else value
    _float_tensor(value, (3, 3), 'rotation', like=like)
    eye = torch.eye(3, dtype=like.dtype, device=like.device)
    if not torch.allclose(value.T @ value, eye, atol=2e-5, rtol=2e-5) or not torch.allclose(torch.linalg.det(value), like.new_tensor(1.), atol=2e-5, rtol=2e-5):
        raise ValueError('Rotation must be orthonormal with determinant +1')
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _validate_asset(asset):
    p = asset.get('position')
    if not isinstance(p, torch.Tensor) or p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError('Nonempty N by 3 Gaussian positions required')
    n = len(p)
    _float_tensor(p, (n, 3), 'position')
    if p.dtype not in (torch.float32, torch.float64):
        raise ValueError('Geometry requires float32 or float64')
    for name, shape in [('covariance', (n, 3, 3)), ('colour', (n, 3)), ('opacity', (n,))]:
        _float_tensor(asset.get(name), shape, name, like=p)
    ids = asset.get('ids')
    if not isinstance(ids, torch.Tensor) or ids.shape != (n,) or ids.dtype != torch.int64 or ids.device != p.device:
        raise ValueError('IDs must be N int64 values on the geometry device')
    if bool((ids < 0).any()) or len(torch.unique(ids)) != n:
        raise ValueError('Local Gaussian IDs must be unique and nonnegative')
    cov = asset['covariance']
    if not torch.allclose(cov, cov.transpose(-1, -2), atol=1e-8, rtol=1e-5):
        raise ValueError('Covariance must be symmetric')
    if bool((torch.linalg.eigvalsh(cov) <= 0).any()):
        raise ValueError('Covariance must be positive definite')
    for name in ('colour', 'opacity'):
        if bool(((asset[name] < 0) | (asset[name] > 1)).any()):
            raise ValueError(f'{name} must lie in [0,1]')
    if 'normal' in asset:
        _float_tensor(asset['normal'], (n, 3), 'normal', like=p)
        norm = asset['normal'].norm(dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=2e-4, rtol=2e-4):
            raise ValueError('Normals must be unit vectors')


class GaussianSceneObject:
    """Own an immutable copy of a canonical asset, bound to its source hash."""
    def __init__(self, object_id, asset, asset_sha256):
        if isinstance(object_id, bool) or not isinstance(object_id, numbers.Integral) or not 0 <= object_id < 2**63:
            raise ValueError('Object ID must be a nonnegative int64 value')
        if not isinstance(asset_sha256, str) or re.fullmatch(r'[0-9a-f]{64}', asset_sha256) is None:
            raise ValueError('Verified source asset SHA256 is required')
        values = {k: v.detach().clone() for k, v in asset.items() if isinstance(v, torch.Tensor)}
        if 'covariance' not in values:
            p = values.get('position')
            if p is None:
                raise ValueError('Positions required')
            frame = _float_tensor(values.get('frame'), (len(p), 3, 3), 'frame', like=p)
            scale = _float_tensor(values.get('scale'), (len(p), 3), 'scale', like=p)
            if bool((scale <= 0).any()):
                raise ValueError('Gaussian scales must be positive')
            eye = torch.eye(3, dtype=p.dtype, device=p.device).expand(len(p), 3, 3)
            if not torch.allclose(frame.transpose(-1, -2) @ frame, eye, atol=2e-4, rtol=2e-4) or bool((torch.linalg.det(frame) <= 0).any()):
                raise ValueError('Gaussian frames must be proper rotations')
            values['covariance'] = (frame * scale.square()[:, None]) @ frame.transpose(-1, -2)
        _validate_asset(values)
        self.object_id, self.asset_sha256 = int(object_id), asset_sha256
        self._asset = values

    @property
    def count(self):
        return len(self._asset['position'])

    def transform(self, *, scale=1., rotation=None, translation=None):
        """mu'=s R mu+t; covariance'=s^2 R covariance R^T; normal'=R normal."""
        scale = _positive(scale, 'Similarity scale')
        p = self._asset['position']
        rotation = _rotation(rotation, p)
        translation = p.new_zeros(3) if translation is None else translation
        _float_tensor(translation, (3,), 'translation', like=p)
        result = {k: v.clone() for k, v in self._asset.items()}
        result['position'] = scale * (p @ rotation.T) + translation
        cov = self._asset['covariance']
        result['covariance'] = scale**2 * (rotation @ cov @ rotation.T)
        if 'normal' in result:
            result['normal'] = self._asset['normal'] @ rotation.T
        if 'frame' in result:
            result['frame'] = rotation @ self._asset['frame']
        if 'scale' in result:
            result['scale'] = scale * self._asset['scale']
        if 'mesh_vertices' in result:
            result['mesh_vertices'] = scale * (self._asset['mesh_vertices'] @ rotation.T) + translation
        result['object_id'] = self.object_id
        result['asset_sha256'] = self.asset_sha256
        result['transform'] = {'scale': scale, 'rotation': rotation.clone(), 'translation': translation.clone()}
        result['units'] = 'Arbitrary common scene units; not inferred metres'
        return result

    def grounded(self, target_height, *, center_xz=(0., 0.), ground_y=0., rotation=None):
        """Normalize FULL Gaussian-center height, not anatomical shoulder height.

        This puts the lowest center on ground_y. Gaussian tails may extend below
        it; the common renderer performs floor occlusion when ground=True.
        """
        target_height = _positive(target_height, 'Target full height')
        p = self._asset['position']
        rotation = _rotation(rotation, p)
        center_xz = torch.as_tensor(center_xz, dtype=p.dtype, device=p.device)
        _float_tensor(center_xz, (2,), 'center_xz', like=p)
        if not isinstance(ground_y, numbers.Real) or not math.isfinite(ground_y):
            raise ValueError('Finite ground height required')
        rotated = p @ rotation.T
        lo, hi = rotated.amin(0), rotated.amax(0)
        height = float(hi[1] - lo[1])
        if height <= 1e-8:
            raise ValueError('Cannot normalize a zero-height asset')
        scale = target_height / height
        center = .5 * (lo + hi)
        translation = p.new_tensor([float(center_xz[0]) - scale*float(center[0]),
                                    float(ground_y) - scale*float(lo[1]),
                                    float(center_xz[1]) - scale*float(center[2])])
        result = self.transform(scale=scale, rotation=rotation, translation=translation)
        result['placement'] = dict(target_full_height=target_height, ground_y=float(ground_y),
            canonical_full_height=height, height_type='Gaussian-center AABB extent, NOT shoulder height',
            physical_scale_recovered=False)
        return result


def object_bounds(obj, sigma=0.):
    """Axis-aligned world bounds; sigma=0 uses centers, >0 Gaussian ellipsoids."""
    if not isinstance(sigma, numbers.Real) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError('Nonnegative finite sigma multiplier required')
    p, cov = obj['position'], obj['covariance']
    radius = sigma * cov.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    return (p-radius).amin(0), (p+radius).amax(0)


def overlap_report(first, second, *, sigma=0.):
    """Conservative AABB overlap, not surface collision or anatomical contact."""
    alo, ahi = object_bounds(first, sigma)
    blo, bhi = object_bounds(second, sigma)
    if alo.device != blo.device or alo.dtype != blo.dtype:
        raise ValueError('Objects must share dtype and device')
    extent = torch.minimum(ahi, bhi) - torch.maximum(alo, blo)
    gap = (-extent).clamp_min(0)
    return dict(interior_aabb_overlap=bool((extent > 0).all()),
        aabb_separation=float(gap.norm()), intersection_extent=extent.clamp_min(0).tolist(),
        sigma=float(sigma), meaning='Conservative broad-phase bounds; no exact collision inference')


def placement_report(obj, scene_lower, scene_upper, *, height_range=None, ground_y=0., tolerance=1e-5):
    p = obj['position']
    lo, hi = object_bounds(obj)
    lower, upper = [torch.as_tensor(x, dtype=p.dtype, device=p.device) for x in (scene_lower, scene_upper)]
    _float_tensor(lower, (3,), 'scene_lower', like=p)
    _float_tensor(upper, (3,), 'scene_upper', like=p)
    if not bool((upper > lower).all()) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Valid scene bounds and tolerance required')
    height = float(hi[1]-lo[1])
    if height_range is not None and (len(height_range) != 2 or not 0 < height_range[0] <= height_range[1]):
        raise ValueError('Positive ordered full-height bounds required')
    checks = dict(within_scene_bounds=bool(((lo >= lower-tolerance) & (hi <= upper+tolerance)).all()),
        centers_grounded=abs(float(lo[1])-ground_y) <= tolerance,
        height_within_bounds=height_range is None or height_range[0]-tolerance <= height <= height_range[1]+tolerance)
    return dict(object_id=obj['object_id'], asset_sha256=obj['asset_sha256'],
        lower=lo.tolist(), upper=hi.tolist(), full_height=height, checks=checks,
        accepted=all(checks.values()), physical_scale_recovered=False)


def merge_scene_objects(objects):
    """Concatenate XYZ state before a SINGLE renderer/depth sort.

    Identity pairs stay stable regardless of object insertion order. Local IDs
    may coincide across different objects. They must not collide within one.
    """
    objects = list(objects)
    if not objects:
        raise ValueError('At least one object required')
    object_ids = [x['object_id'] for x in objects]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError('Duplicate object IDs')
    objects.sort(key=lambda x: x['object_id'])
    for obj in objects:
        _validate_asset(obj)
        if obj['position'].device != objects[0]['position'].device or obj['position'].dtype != objects[0]['position'].dtype:
            raise ValueError('All objects must share geometry dtype/device')
    keys = ['position', 'covariance', 'colour', 'opacity']
    result = {k: torch.cat([x[k] for x in objects]) for k in keys}
    result['local_ids'] = torch.cat([x['ids'] for x in objects])
    result['object_ids'] = torch.cat([torch.full_like(x['ids'], x['object_id']) for x in objects])
    result['identity_pairs'] = torch.stack((result['object_ids'], result['local_ids']), -1)
    result['bindings'] = {str(x['object_id']): x['asset_sha256'] for x in objects}
    if all('normal' in x for x in objects):
        result['normal'] = torch.cat([x['normal'] for x in objects])
    return result


def render_scene(objects, eye, target, **kwargs):
    """Return shared-depth RGB/alpha, optionally stable-ID raster associations."""
    from .gaussian3d import render
    merged = merge_scene_objects(objects)
    result = render(merged['position'], merged['covariance'], merged['colour'], merged['opacity'], eye, target, **kwargs)
    if kwargs.get('return_cache', False):
        rgb, alpha, cache = result
        cache['identity_pairs'] = merged['identity_pairs'][cache['ids']]
        cache['asset_bindings'] = merged['bindings']
        return rgb, alpha, cache
    return result
