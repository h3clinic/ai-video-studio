"""Bounded persistent 3D Gaussian state plumbing, NOT a motion generator.

An external, separately evaluated model supplies absolute positions and either
canonical-to-current deformation gradients or covariances. No target video,
future trajectory archive, Wan invocation or appearance regeneration is hidden
inside this runtime. Rendering still executes for every requested view/frame.
"""
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint_io import load_verified, save_inference_checkpoint
from .gaussian3d import project, render


def _finite(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} must be a tensor with shape {tuple(shape)}')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')


def _covariance(value, count):
    _finite(value, (count, 3, 3), 'covariance')
    if not torch.allclose(value, value.transpose(-1, -2), atol=1e-9, rtol=1e-4):
        raise ValueError('Covariance must be symmetric')
    if bool((torch.linalg.cholesky_ex(value).info != 0).any()):
        raise ValueError('Covariance must be positive definite, not singular')


def _bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


class GaussianVideoMemory:
    """One immutable canonical asset plus constant-count mutable state buffers.

    ``update`` gradients map the CANONICAL asset to the requested current pose;
    they are not incremental gradients from the previous pose. IDs must remain
    in exactly the canonical row order. Position buffers are swapped/reused,
    and appearance/IDs are copied only at construction, never once per step.

    Public snapshots are independent copies; private tensors are not exposed as
    writable views. Full snapshots intentionally cost an additional state copy.
    """
    schema = 'persistent_gaussian_video_memory_v1'

    def __init__(self, asset, device='cpu'):
        required = ('position', 'covariance', 'colour', 'opacity', 'ids')
        if any(key not in asset for key in required):
            raise ValueError(f'Asset requires {required}')
        n = len(asset['position'])
        if n < 1:
            raise ValueError('An empty Gaussian asset is not supported')
        self.device = torch.device(device)
        canonical = {}
        for key in required:
            value = asset[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f'{key} must be a tensor')
            canonical[key] = value.detach().to(self.device).clone()
        ids = canonical['ids']
        if ids.dtype not in (torch.int32, torch.int64) or ids.shape != (n,):
            raise ValueError('IDs must be an integer N-vector')
        if ids.unique().numel() != n:
            raise ValueError('Every Gaussian must have a unique persistent ID')
        for key in ('position', 'covariance', 'colour', 'opacity'):
            canonical[key] = canonical[key].float()
        _finite(canonical['position'], (n, 3), 'position')
        _finite(canonical['colour'], (n, 3), 'colour')
        if canonical['opacity'].shape == (n, 1):
            canonical['opacity'] = canonical['opacity'].reshape(n)
        _finite(canonical['opacity'], (n,), 'opacity')
        for key in ('colour', 'opacity'):
            if bool(((canonical[key] < 0) | (canonical[key] > 1)).any()):
                raise ValueError(f'{key} must be in [0,1]')
        _covariance(canonical['covariance'], n)
        normal = asset.get('normal', torch.zeros_like(canonical['position']))
        _finite(normal, (n, 3), 'normal')
        canonical['normal'] = F.normalize(normal.detach().to(self.device).float(), dim=-1).clone()
        digest = hashlib.sha256()
        for key in sorted(canonical):
            value = canonical[key].contiguous().cpu()
            digest.update(f'{key}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.numpy().tobytes())
        self.asset_digest = digest.hexdigest()
        self._asset = canonical
        self.count = n
        self._position = canonical['position'].clone()
        self._previous_position = canonical['position'].clone()
        self._covariance = canonical['covariance'].clone()
        self._normal = canonical['normal'].clone()
        self.step_index = 0
        self.elapsed_seconds = 0.
        self.last_dt = 0.
        self.normals_valid = bool(canonical['normal'].norm(dim=-1).gt(0).all())

    def _identity(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('Explicit integer IDs are required for every update')
        if ids.shape != (self.count,) or not torch.equal(ids.to(self.device), self._asset['ids']):
            raise ValueError('Gaussian IDs/order do not match the canonical memory')

    def _input(self, value, shape, name):
        _finite(value, shape, name)
        return value.detach().to(device=self.device, dtype=torch.float32)

    def _validate_step(self, position, ids, dt):
        self._identity(ids)
        if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        return self._input(position, (self.count, 3), 'position')

    def _commit(self, position, covariance, normal, dt, normals_valid):
        # Do not mutate any state before validation/transport succeeds.
        self._previous_position, self._position = self._position, self._previous_position
        self._position.copy_(position)
        self._covariance.copy_(covariance)
        self._normal.copy_(normal)
        self.step_index += 1
        self.elapsed_seconds += float(dt)
        self.last_dt = float(dt)
        self.normals_valid = bool(normals_valid)

    @torch.no_grad()
    def update(self, position, *, ids, dt, deformation_gradient=None, rotations=None):
        """Accept verified external pose; transport Sigma = F Sigma0 F^T.

        Normal transport is F^-T n0. Reflections/singular gradients and improper
        rotations are rejected. These checks do not prove correct anatomy.
        """
        p = self._validate_step(position, ids, dt)
        if (deformation_gradient is None) == (rotations is None):
            raise ValueError('Provide exactly one of deformation_gradient or rotations')
        gradient = self._input(deformation_gradient if rotations is None else rotations,
                               (self.count, 3, 3), 'canonical-to-current gradient')
        if bool((torch.linalg.det(gradient) <= 0).any()):
            raise ValueError('Gradient must preserve orientation and be nonsingular')
        if bool((torch.linalg.svdvals(gradient)[..., -1] <= 1e-8).any()):
            raise ValueError('Gradient is numerically singular')
        if rotations is not None:
            eye = torch.eye(3, device=self.device).expand_as(gradient)
            if not torch.allclose(gradient @ gradient.transpose(-1, -2), eye, atol=1e-5, rtol=1e-5):
                raise ValueError('Rotations must be proper orthonormal frames')
        covariance = gradient @ self._asset['covariance'] @ gradient.transpose(-1, -2)
        covariance = (covariance + covariance.transpose(-1, -2)) * .5
        _covariance(covariance, self.count)
        normal = torch.linalg.solve(gradient.transpose(-1, -2), self._asset['normal'][..., None])[..., 0]
        normal = F.normalize(normal, dim=-1)
        self._commit(p, covariance, normal, dt, bool(self._asset['normal'].norm(dim=-1).gt(0).all()))

    @torch.no_grad()
    def update_covariance(self, position, covariance, *, ids, dt, normal=None):
        """Accept a surface skinner's absolute position/covariance output.

        A covariance does not determine a material normal. When normals are not
        supplied the normal feature is explicitly zero/invalid, not frozen at
        its old pose. The caller must verify its external deformation provenance.
        """
        p = self._validate_step(position, ids, dt)
        covariance = self._input(covariance, (self.count, 3, 3), 'covariance')
        _covariance(covariance, self.count)
        if normal is None:
            normals = torch.zeros_like(self._normal)
        else:
            normals = self._input(normal, (self.count, 3), 'normal')
            if bool((normals.norm(dim=-1) < 1e-8).any()):
                raise ValueError('Supplied normals must have nonzero length')
            normals = F.normalize(normals, dim=-1)
        self._commit(p, covariance, normals, dt, normal is not None)

    @torch.no_grad()
    def snapshot(self):
        """Independent CPU copy of bounded state; excludes canonical asset."""
        return dict(schema=self.schema, asset_digest=self.asset_digest,
                    step_index=self.step_index, elapsed_seconds=self.elapsed_seconds,
                    last_dt=self.last_dt, normals_valid=self.normals_valid,
                    position=self._position.detach().cpu().clone(),
                    previous_position=self._previous_position.detach().cpu().clone(),
                    covariance=self._covariance.detach().cpu().clone(),
                    normal=self._normal.detach().cpu().clone())

    @torch.no_grad()
    def restore(self, snapshot):
        if snapshot.get('schema') != self.schema or snapshot.get('asset_digest') != self.asset_digest:
            raise ValueError('Snapshot schema or immutable asset digest mismatch')
        step = snapshot.get('step_index')
        elapsed, dt = snapshot.get('elapsed_seconds'), snapshot.get('last_dt')
        if type(step) is not int or step < 0 or type(snapshot.get('normals_valid')) is not bool:
            raise ValueError('Invalid snapshot step/normal metadata')
        if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in (elapsed, dt)):
            raise ValueError('Invalid snapshot timing')
        if (step == 0 and (elapsed != 0 or dt != 0)) or (step > 0 and (dt <= 0 or elapsed < dt)):
            raise ValueError('Inconsistent snapshot timing')
        values = {key: self._input(snapshot.get(key), shape, key) for key, shape in
                  [('position', (self.count, 3)), ('previous_position', (self.count, 3)),
                   ('covariance', (self.count, 3, 3)), ('normal', (self.count, 3))]}
        _covariance(values['covariance'], self.count)
        if snapshot['normals_valid'] and not torch.allclose(values['normal'].norm(dim=-1),
                                                          torch.ones(self.count, device=self.device), atol=1e-5):
            raise ValueError('Valid snapshot normals must be unit length')
        for key, value in values.items():
            getattr(self, '_' + key).copy_(value)
        self.step_index, self.elapsed_seconds, self.last_dt = step, float(elapsed), float(dt)
        self.normals_valid = snapshot['normals_valid']

    def save(self, path):
        return save_inference_checkpoint(self.snapshot(), Path(path))

    def load(self, path):
        self.restore(load_verified(path))

    def memory_bytes(self):
        asset = _bytes(self._asset.values())
        current = _bytes((self._position, self._covariance, self._normal))
        previous = _bytes((self._previous_position,))
        return dict(asset_tensor_bytes=asset, current_state_tensor_bytes=current,
                    previous_state_tensor_bytes=previous,
                    resident_tensor_bytes=asset + current + previous,
                    snapshot_tensor_bytes=current + previous,
                    frame_history_tensor_bytes=0, stored_future_trajectory_tensor_bytes=0,
                    excluded='Python metadata, caller tensors, model weights, renderer/projection temporaries, emitted features and any caller-retained history')

    @torch.no_grad()
    def project_current(self, camera, height, width, *, downsample=8, radius=2,
                        include_velocity=False, depth_scale=4., world_scale=1.):
        """Emit current visibility-weighted features, without retaining a frame.

        Default channels match SpatialControl: RGB3, depth1, normal3, XYZ3,
        projected covariance3, alpha1. Optional world-velocity3 follows alpha.
        Output [1,C,H/downsample,W/downsample] is one explicit time sample;
        temporal alignment/resampling is the caller's responsibility. It is NOT
        a Wan VAE latent, and repeated static calls cannot invent motion.
        """
        if any(type(x) is not int or x < 1 for x in (height, width, downsample)):
            raise ValueError('Image dimensions/downsample must be positive integers')
        if height % downsample or width % downsample:
            raise ValueError('Image dimensions must be divisible by downsample')
        if type(radius) is not int or radius < 0:
            raise ValueError('Radius must be a nonnegative integer')
        if any(not math.isfinite(x) or x <= 0 for x in (depth_scale, world_scale)):
            raise ValueError('Feature scales must be positive and finite')
        eye = self._input(torch.as_tensor(camera['eye']), (3,), 'camera eye')
        target = self._input(torch.as_tensor(camera['target']), (3,), 'camera target')
        fov = float(camera.get('fov', 42.))
        if not math.isfinite(fov) or not 0 < fov < 179:
            raise ValueError('Invalid camera field of view')
        rgb, alpha, cache = render(self._position, self._covariance,
                                   self._asset['colour'], self._asset['opacity'], eye, target,
                                   height, width, fov, radius=radius, ground=False, return_cache=True)
        _, cov2, depth, _, _ = project(self._position, self._covariance, eye, target, height, width, fov)
        projected_covariance = torch.stack((cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]), -1) / height**2
        values = torch.cat((self._asset['colour'], depth[:, None]/depth_scale, self._normal,
                            self._position/world_scale, projected_covariance), -1)
        channels = values.shape[-1] + (3 if include_velocity else 0)
        if include_velocity:
            velocity = ((self._position-self._previous_position)/self.last_dt if self.last_dt > 0
                        else torch.zeros_like(self._position))
            values = torch.cat((values, velocity/world_scale), -1)
        summed = values.new_zeros(height*width, channels)
        for start in range(0, len(cache['ids']), 100000):
            end = start + 100000
            summed.index_add_(0, cache['pixel'][start:end],
                              cache['weight'][start:end, None]*values[cache['ids'][start:end]])
        summed = summed.reshape(height, width, channels)
        parts = (summed[..., :13], alpha[..., None])
        if include_velocity:
            parts += (summed[..., 13:],)
        features = torch.cat(parts, -1).permute(2, 0, 1)[None]
        features = F.avg_pool2d(features, downsample)
        return dict(rgb=rgb, alpha=alpha, features=features,
                    step_index=self.step_index, elapsed_seconds=self.elapsed_seconds,
                    normals_valid=self.normals_valid, asset_digest=self.asset_digest)
