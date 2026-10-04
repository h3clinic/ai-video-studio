"""Object-scoped agent memory and incremental XYZ Gaussian rendering.

Agents supply actual Gaussian geometry, not RGB patches. This broker is NOT a
trained semantic generator. It does not turn a text noun into an asset. A frame
cache must originate from this exact renderer, never an unrelated Wan MP4.
"""
import hashlib
import math
import secrets
import threading

import torch

from .gaussian_scene_objects import GaussianSceneObject, merge_scene_objects
from .gaussian3d import project, render

FIELDS = ('position', 'covariance', 'colour', 'opacity', 'ids')


def asset_digest(asset):
    h = hashlib.sha256()
    for key in FIELDS:
        value = asset[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def diagonal_curvature_step(gradient, diagonal, *, damping=1e-3, max_step=.01):
    """Clipped damped diagonal Gauss-Newton update; caller supplies curvature.

    Not a Hessian estimator, semantic model, or reproduction of 3DGS2-TR.
    """
    if (gradient.shape != diagonal.shape or gradient.dtype != diagonal.dtype
            or gradient.device != diagonal.device or not gradient.is_floating_point()
            or not torch.isfinite(gradient).all() or not torch.isfinite(diagonal).all()
            or (diagonal < 0).any() or not math.isfinite(damping) or damping <= 0
            or not math.isfinite(max_step) or max_step <= 0):
        raise ValueError('Finite matching gradient, nonnegative curvature, positive bounds required')
    return (-gradient / (diagonal + damping)).clamp(-max_step, max_step)


class GaussianObjectAgentMemory:
    """A trusted broker owns scene state; workers receive one object's copy.

    Versioned leases prevent stale/cross-object writes through this API. This
    is not a Python sandbox: untrusted code must run in a separate process.
    Material identity is (object_id, revision, local_id), invalidating old
    learned features on replacement. No scene-wide latent cache is reused.
    """
    def __init__(self, assets):
        self._objects, self._versions, self._leases = {}, {}, {}
        self._lock = threading.RLock()
        for oid, asset in assets.items():
            self._objects[oid] = self._own(oid, asset)
            self._versions[oid] = 0
        if not self._objects:
            raise ValueError('Nonempty XYZ scene required')

    @staticmethod
    def _own(oid, asset):
        values = {k: asset[k] for k in FIELDS}
        if any(v.device.type != 'cpu' for v in values.values()):
            raise ValueError('Broker currently supports CPU state only')
        if values['position'].dtype != torch.float32:
            raise ValueError('CPU renderer contract requires float32')
        return GaussianSceneObject(oid, values, asset_digest(values))

    def lease(self, object_id, *, max_points, lower, upper):
        with self._lock:
            obj = self._objects[object_id]
            lower, upper = torch.tensor(lower, dtype=torch.float32), torch.tensor(upper, dtype=torch.float32)
            if (type(max_points) is not int or not 1 <= max_points <= 100000
                    or lower.shape != (3,) or upper.shape != (3,)
                    or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
                    or not (upper > lower).all()):
                raise ValueError('Explicit point budget and finite XYZ bounds required')
            token = secrets.token_hex(24)
            self._leases[token] = (object_id, self._versions[object_id], max_points, lower, upper)
            state = obj.transform()
            return dict(token=token, object_id=object_id, revision=self._versions[object_id],
                asset={k: state[k] for k in FIELDS},
                bytes=sum(state[k].numel()*state[k].element_size() for k in FIELDS))

    def replace(self, token, replacement):
        with self._lock:
            if token not in self._leases:
                raise ValueError('Unknown or consumed object capability')
            oid, version, maximum, lower, upper = self._leases[token]
            if self._versions[oid] != version:
                raise ValueError('Stale object revision')
            if len(replacement['position']) > maximum:
                raise ValueError('Object point budget exceeded')
            new = self._own(oid, replacement)
            proposed = new.transform()
            # Bound ellipsoid extent as well as centers, not just a point mask.
            spread = 3 * proposed['covariance'].diagonal(dim1=-2, dim2=-1).sqrt()
            if not ((proposed['position']-spread >= lower).all()
                    and (proposed['position']+spread <= upper).all()):
                raise ValueError('Replacement exceeds authorized spatial extent')
            old = self._objects[oid]
            self._objects[oid] = new
            self._versions[oid] += 1
            del self._leases[token]
            return dict(object_id=oid, old_digest=old.asset_sha256, new_digest=new.asset_sha256,
                revision=self._versions[oid], invalidated_material_revision=version,
                changed_points=new.count, training=False, semantic_generation=False)

    def snapshot(self):
        with self._lock:
            return ({oid: obj.transform() for oid, obj in self._objects.items()}, dict(self._versions))


class IncrementalGaussianFrame:
    """Cache a genuine Gaussian-rendered frame, rerasterize changed footprints.

    Camera is fixed per cache. Movement requires applying transforms to object
    state before replacement. A video uses one cache per existing frame. This
    trades cached RGB memory for avoided raster work; it is not free memory.
    All overlapping occluders are included, not only the changed object.
    """
    def __init__(self, memory, eye, target, *, height=64, width=96, fov=42., radius=4):
        if not (type(height) is int and type(width) is int and 1 <= height <= 1080 and 1 <= width <= 1920
                and type(radius) is int and 1 <= radius <= 8 and math.isfinite(fov) and 1 < fov < 175):
            raise ValueError('Bounded camera/raster configuration required')
        self.memory = memory
        self.eye, self.target = eye.detach().clone(), target.detach().clone()
        if any(x.shape != (3,) or x.dtype != torch.float32 or x.device.type != 'cpu'
               or not torch.isfinite(x).all() for x in (self.eye, self.target)):
            raise ValueError('Finite CPU float32 camera required')
        self.height, self.width, self.fov, self.radius = height, width, fov, radius
        self.objects, self.versions = memory.snapshot()
        merged = merge_scene_objects(self.objects.values())
        self._rgb, _ = self._render(merged, height, width, fov, None)

    def _render(self, values, height, width, fov, principal):
        return render(*(values[k] for k in FIELDS[:4]), self.eye, self.target,
            height=height, width=width, fov=fov, radius=self.radius, ground=False, principal=principal)

    def image(self):
        return self._rgb.clone()

    def update(self):
        current, versions = self.memory.snapshot()
        changed = [oid for oid in versions if versions[oid] != self.versions[oid]]
        if not changed:
            return dict(changed_objects=[], rasterized_pixels=0, rasterized_gaussians=0)
        bounds = []
        for objects in (self.objects, current):
            for oid in changed:
                obj = objects[oid]
                means, _, depth, _, _ = project(obj['position'], obj['covariance'], self.eye, self.target,
                    self.height, self.width, self.fov)
                means = means[depth > .02].floor()
                if len(means):
                    bounds.append((means.amin(0)-self.radius, means.amax(0)+self.radius+1))
        pixels = candidates = 0
        if bounds:
            low = torch.stack([x[0] for x in bounds]).amin(0)
            high = torch.stack([x[1] for x in bounds]).amax(0)
            x0, y0 = max(0, int(low[0])), max(0, int(low[1]))
            x1, y1 = min(self.width, int(high[0])), min(self.height, int(high[1]))
            if x1 > x0 and y1 > y0:
                merged = merge_scene_objects(current.values())
                centers, _, depth, _, focal = project(merged['position'], merged['covariance'], self.eye,
                    self.target, self.height, self.width, self.fov)
                centers = centers.floor()
                keep = ((depth > .02) & (centers[:, 0]+self.radius >= x0)
                    & (centers[:, 0]-self.radius < x1) & (centers[:, 1]+self.radius >= y0)
                    & (centers[:, 1]-self.radius < y1))
                selected = {k: merged[k][keep] for k in FIELDS[:4]}
                # Preserve the original focal length and principal point.
                patch_fov = math.degrees(2*math.atan((y1-y0)/(2*focal)))
                patch, _ = self._render(selected, y1-y0, x1-x0, patch_fov,
                    (self.width/2-x0, self.height/2-y0))
                self._rgb[y0:y1, x0:x1] = patch
                pixels, candidates = (x1-x0)*(y1-y0), int(keep.sum())
        self.objects, self.versions = current, versions
        return dict(changed_objects=changed, rasterized_pixels=pixels, rasterized_gaussians=candidates,
            total_pixels=self.height*self.width, cached_rgb_bytes=self._rgb.numel()*self._rgb.element_size(),
            scope='Finite-stencil Gaussian raster only; scene scan, copies and cache memory are extra costs')
