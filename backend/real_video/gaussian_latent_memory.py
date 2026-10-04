"""Explicit latent read/write on supplied Gaussian visibility; no learned writer.

Stores N x C latent features keyed by immutable Gaussian IDs. A caller supplies
the known geometry's alpha-compositing cache at LATENT resolution. It must also
explicitly supply confidence for every write; generated content is not trusted
automatically. No RGB decoding/re-encoding or Gaussian geometry prediction is
performed. This is not a Robust Dreamer reproduction or a trained Wan adapter.
"""
import math
from pathlib import Path

import torch

from .checkpoint_io import load_verified, save_inference_checkpoint


def generation_vector(value, count, device='cpu'):
    """Validate caller-recorded lifetimes; this does not infer correspondence."""
    if (not isinstance(value, torch.Tensor) or value.shape != (count,)
            or value.dtype not in (torch.int32, torch.int64) or bool((value < 0).any())):
        raise ValueError('Track generations must be a nonnegative integer N-vector')
    return value.detach().to(device=device, dtype=torch.int64).clone()


def bind_alpha_cache(cache, *, ids, generations, asset_digest, height, width):
    """Bind row order/lifetimes at cache construction, not at memory consumption.

    Metadata is copied. Fragment tensors are shared, so callers must not mutate
    them after construction. Binding checks consistency, not provenance or truth.
    """
    if (not isinstance(cache, dict) or any(k not in cache for k in ('pixel', 'ids', 'weight'))
            or not isinstance(ids, torch.Tensor) or ids.ndim != 1
            or ids.dtype not in (torch.int32, torch.int64) or ids.numel() < 1
            or ids.unique().numel() != ids.numel()):
        raise ValueError('Cache and unique integer row identities required')
    if (not isinstance(asset_digest, str) or len(asset_digest) != 64
            or any(c not in '0123456789abcdef' for c in asset_digest)
            or any(type(v) is not int or v < 1 for v in (height, width))):
        raise ValueError('Asset digest and positive image dimensions required')
    return dict(cache, identity_binding=dict(schema='gaussian_cache_identity_v1',
        asset_digest=asset_digest, ids=ids.detach().cpu().long().clone(),
        generations=generation_vector(generations, ids.numel()), height=height, width=width))


class GaussianLatentMemory:
    schema = 'gaussian_latent_feature_memory_v1'

    def __init__(self, ids, channels, asset_digest, *, device='cpu', initial_features=None,
                 initial_mass=0., max_observation_mass=32., generations=None):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('IDs must be an integer vector')
        if ids.numel() < 1 or ids.unique().numel() != ids.numel():
            raise ValueError('Nonempty unique Gaussian IDs required')
        if type(channels) is not int or channels < 1:
            raise ValueError('Positive latent channel count required')
        if not isinstance(asset_digest, str) or len(asset_digest) != 64 or any(c not in '0123456789abcdef' for c in asset_digest):
            raise ValueError('A canonical asset SHA-256 digest is required')
        if not math.isfinite(max_observation_mass) or max_observation_mass <= 0:
            raise ValueError('Positive finite observation-mass cap required')
        if not math.isfinite(initial_mass) or not 0 <= initial_mass <= max_observation_mass:
            raise ValueError('Initial observation mass must lie in [0, cap]')
        self.device, self.channels = torch.device(device), channels
        self.count, self.asset_digest = ids.numel(), asset_digest
        self.max_observation_mass = float(max_observation_mass)
        self._ids = ids.detach().to(self.device).clone()
        # Legacy banks are explicitly unbound. Supplying generations opts into
        # a distinct snapshot schema and mandatory cache-identity validation.
        self._generations = (None if generations is None else
                             generation_vector(generations, self.count, self.device))
        if self._generations is not None:
            self.schema = 'gaussian_latent_feature_memory_lifetimes_v2'
        self._features = torch.zeros((self.count, channels), device=self.device)
        if initial_features is not None:
            self._features.copy_(self._float(initial_features, (self.count, channels), 'initial features'))
        elif initial_mass != 0:
            raise ValueError('Nonzero initial mass requires initial features')
        self._mass = torch.full((self.count,), float(initial_mass), device=self.device)
        self.writes = 0
        # Runtime invalidation counter, deliberately never restored from a file.
        self._revision = 0

    @property
    def revision(self):
        return self._revision

    def _float(self, value, shape, name):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or not value.is_floating_point():
            raise ValueError(f'{name} must be a floating tensor with shape {shape}')
        if not torch.isfinite(value).all():
            raise ValueError(f'{name} must be finite')
        converted = value.detach().to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(converted).all():
            raise ValueError(f'{name} overflows float32 storage')
        return converted

    def _identity(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('Explicit integer Gaussian IDs required')
        if ids.shape != self._ids.shape or not torch.equal(ids.to(self.device), self._ids):
            raise ValueError('Gaussian ID order mismatch')

    def _cache(self, cache, height, width, ids):
        self._identity(ids)
        if any(type(n) is not int or n < 1 for n in (height, width)):
            raise ValueError('Positive latent-map dimensions required')
        if not isinstance(cache, dict) or any(key not in cache for key in ('pixel', 'ids', 'weight')):
            raise ValueError('Expected renderer pixel/ids/weight cache')
        binding = cache.get('identity_binding')
        if self._generations is None:
            if binding is not None:
                raise ValueError('A bound cache requires a lifetime-aware memory bank')
        else:
            if (not isinstance(binding, dict) or binding.get('schema') != 'gaussian_cache_identity_v1'
                    or binding.get('asset_digest') != self.asset_digest
                    or type(binding.get('height')) is not int or binding['height'] != height
                    or type(binding.get('width')) is not int or binding['width'] != width):
                raise ValueError('Missing or mismatched cache identity binding')
            self._identity(binding.get('ids'))
            cached_generations = generation_vector(binding.get('generations'), self.count, self.device)
            if not torch.equal(cached_generations, self._generations):
                raise ValueError('Stale cache track generations')
        pixel, rows, weight = (cache[key] for key in ('pixel', 'ids', 'weight'))
        if any(not isinstance(v, torch.Tensor) or v.ndim != 1 for v in (pixel, rows, weight)):
            raise ValueError('Cache arrays must be vectors')
        if pixel.shape != rows.shape or pixel.shape != weight.shape:
            raise ValueError('Cache arrays must have matching shapes')
        if any(v.dtype not in (torch.int32, torch.int64) for v in (pixel, rows)):
            raise ValueError('Cache pixel and original-row IDs must be integer')
        pixel, rows = pixel.to(self.device).long(), rows.to(self.device).long()
        weight = self._float(weight, tuple(weight.shape), 'cache weight')
        if bool(((pixel < 0) | (pixel >= height*width) | (rows < 0) | (rows >= self.count)).any()):
            raise ValueError('Cache index out of range; IDs must be original row indices')
        if bool(((weight < 0) | (weight > 1)).any()):
            raise ValueError('Alpha-compositing weights must lie in [0,1]')
        coverage = self._features.new_zeros(height*width).index_add_(0, pixel, weight)
        if bool((coverage > 1.00001).any()):
            raise ValueError('Cache coverage exceeds alpha-compositing bound')
        return pixel, rows, weight, coverage

    @torch.no_grad()
    def advance_generations(self, generations, *, ids):
        """Retire replaced slots before any read/write; preserve occluded IDs.

        Caller must distinguish a new material lifetime from mere invisibility.
        No appearance is invented for a new lifetime. Explicit restore supports
        rollback; this forward update rejects decreasing generation counters.
        """
        self._identity(ids)
        if self._generations is None:
            raise ValueError('Lifetime-aware memory bank required')
        current = generation_vector(generations, self.count, self.device)
        if bool((current < self._generations).any()):
            raise ValueError('Track generations cannot decrease during forward updates')
        changed = current != self._generations
        self._features[changed] = 0
        self._mass[changed] = 0
        self._generations.copy_(current)
        if bool(changed.any()):
            self._revision += 1
        return dict(retired_slots=int(changed.sum()), features_generated=False)

    @torch.no_grad()
    def read(self, cache, height, width, *, ids):
        """Return sum_i A[p,i] f[i], not coverage-normalized RGB or VAE output."""
        pixel, rows, weight, coverage = self._cache(cache, height, width, ids)
        result = self._features.new_zeros(height*width, self.channels)
        for start in range(0, len(rows), 100000):
            end = start+100000
            result.index_add_(0, pixel[start:end], weight[start:end, None]*self._features[rows[start:end]])
        return dict(features=result.reshape(height, width, self.channels).permute(2, 0, 1),
                    coverage=coverage.reshape(height, width))

    @torch.no_grad()
    def write(self, latent, cache, *, ids, confidence, validity=None):
        """Accumulate confidence-weighted visible latent observations per ID.

        For cache coefficient A[p,i] and caller confidence q[p], evidence is
        m[i]=sum_p A[p,i]q[p], s[i]=sum_p A[p,i]q[p]z[p]. Update with
        f'[i]=(M[i]f[i]+s[i])/(M[i]+m[i]); M'=min(cap,M+m).
        After saturation this is an adaptive weighted mean, NOT the exact mean
        of unlimited history. It is also NOT an inverse of alpha blending:
        mixed pixels can mix features across surfaces. Confidence/visibility
        errors can contaminate memory. Zero-evidence IDs remain exactly intact.
        """
        if not isinstance(latent, torch.Tensor) or latent.ndim != 3 or latent.shape[0] != self.channels:
            raise ValueError('Observed latent must have shape C,H,W')
        _, height, width = latent.shape
        latent = self._float(latent, (self.channels, height, width), 'observed latent')
        conf = self._float(confidence, (height, width), 'explicit write confidence')
        if bool(((conf < 0) | (conf > 1)).any()):
            raise ValueError('Write confidence must lie in [0,1]')
        if validity is not None:
            if not isinstance(validity, torch.Tensor) or validity.dtype != torch.bool or validity.shape != (height, width):
                raise ValueError('Validity must be a boolean H,W map')
            conf = conf*validity.to(self.device)
        pixel, rows, weight, _ = self._cache(cache, height, width, ids)
        evidence = weight*conf.reshape(-1)[pixel]
        mass = torch.zeros_like(self._mass).index_add_(0, rows, evidence)
        total = torch.zeros_like(self._features)
        observed = latent.permute(1, 2, 0).reshape(-1, self.channels)
        for start in range(0, len(rows), 100000):
            end = start+100000
            total.index_add_(0, rows[start:end], evidence[start:end, None]*observed[pixel[start:end]])
        active = mass > 0
        denominator = self._mass[active]+mass[active]
        updated = (self._mass[active, None]*self._features[active]+total[active])/denominator[:, None]
        if not torch.isfinite(updated).all() or not torch.isfinite(mass).all():
            raise ValueError('Nonfinite latent update rejected before state mutation')
        self._features[active] = updated
        self._mass[active] = denominator.clamp_max(self.max_observation_mass)
        self.writes += 1
        self._revision += 1
        return dict(updated_gaussians=int(active.sum()), incoming_evidence_mass=float(mass.sum()),
                    writes=self.writes, geometry_updated=False)

    def memory_bytes(self):
        features = self._features.numel()*self._features.element_size()
        mass = self._mass.numel()*self._mass.element_size()
        ids = self._ids.numel()*self._ids.element_size()
        result = dict(latent_feature_bytes=features, observation_mass_bytes=mass, identity_bytes=ids,
                    resident_tensor_bytes=features+mass+ids, snapshot_tensor_bytes=features+mass+ids,
                    frame_history_bytes=0,
                    exclusions='Geometry asset, raster cache, observed/emitted maps, temporary accumulation, Wan/model weights and Python metadata')
        if self._generations is not None:
            size = self._generations.numel()*self._generations.element_size()
            result.update(generation_bytes=size,
                          resident_tensor_bytes=result['resident_tensor_bytes']+size,
                          snapshot_tensor_bytes=result['snapshot_tensor_bytes']+size)
        return result

    def snapshot(self):
        result = dict(schema=self.schema, asset_digest=self.asset_digest, channels=self.channels,
                    max_observation_mass=self.max_observation_mass, writes=self.writes,
                    ids=self._ids.detach().cpu().clone(), features=self._features.detach().cpu().clone(),
                    observation_mass=self._mass.detach().cpu().clone())
        if self._generations is not None:
            result['generations'] = self._generations.detach().cpu().clone()
        return result

    @torch.no_grad()
    def restore(self, snapshot):
        if snapshot.get('schema') != self.schema or snapshot.get('asset_digest') != self.asset_digest:
            raise ValueError('Snapshot schema or geometry-asset digest mismatch')
        self._identity(snapshot.get('ids'))
        if snapshot.get('channels') != self.channels or snapshot.get('max_observation_mass') != self.max_observation_mass:
            raise ValueError('Snapshot configuration mismatch')
        writes = snapshot.get('writes')
        if type(writes) is not int or writes < 0:
            raise ValueError('Invalid snapshot write count')
        features = self._float(snapshot.get('features'), (self.count, self.channels), 'snapshot features')
        mass = self._float(snapshot.get('observation_mass'), (self.count,), 'snapshot mass')
        if bool(((mass < 0) | (mass > self.max_observation_mass)).any()):
            raise ValueError('Snapshot mass outside configured cap')
        generations = (generation_vector(snapshot.get('generations'), self.count, self.device)
                       if self._generations is not None else None)
        self._features.copy_(features)
        self._mass.copy_(mass)
        if generations is not None:
            self._generations.copy_(generations)
        self.writes = writes
        self._revision += 1

    def save(self, path):
        return save_inference_checkpoint(self.snapshot(), Path(path))

    def load(self, path):
        self.restore(load_verified(path))
