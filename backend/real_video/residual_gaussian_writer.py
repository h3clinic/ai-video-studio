"""Opt-in CPU residual fitting of persistent Gaussian appearance parameters.

Geometry is FIXED and supplied through genuine alpha-compositing coefficients
A[p,i]=alpha_i(p)*transmittance_i(p). This module predicts neither geometry nor
motion and trains no generator/network weights. In particular, a planar cache
does not become XYZ geometry by passing through this writer.

The old weighted writer is a backprojection, not an inverse: from an empty bank
it computes f=D^-1 A.T Q z, so recall A f generally mixes/attenuates features.
Instead, with prior f0, observed feature z, explicit background b, confidence Q,
and a caller-supplied writable-ID set U, we fit

  min_delta 0.5 ||sqrt(Q)(A(f0+delta)+b-z)||_F^2
            + 0.5 lambda sum_i (1+old_mass_i)||delta_i||^2,
  delta_i=0 for i outside U; max_i ||delta_i||_2 <= trust_radius.

The sparse normal operator is H(v)=A_U.T Q A_U v+lambda D v. We use bounded
Jacobi-preconditioned conjugate-gradient iterations; each candidate is scaled
back to the trust radius and accepted only if its FINITE objective, evaluated
with the actual FP32 read operator, is nonincreasing. No dense P-by-N or N-by-N
matrix is constructed. A zero read residual is an exact no-op, including mass
and write count. Unobserved/nonwritable features and evidence stay bitwise fixed.

This is standard regularized least squares, our engineering adaptation, not a
new theorem or Robust Dreamer reproduction. Its depth-aware latent read is
motivated by arXiv:2605.30855v1 Eq5-7 and 2308.04079v1 Eq2-5. Unlike those full
systems, this does not infer correspondences, handle unknown hidden geometry,
or train a generator to tolerate recalled memory. Low-opacity/ambiguous A is
ill-conditioned; regularization and trust limits trade fit against amplification.
Pixel confidence is explicit, not inferred from coverage or from low residual.
Only the supplied current observation enters; causal provenance is the caller's
responsibility. Generated hallucinations must not silently become observations.
"""
import math

import torch

from .gaussian_latent_memory import GaussianLatentMemory


class SparseAlphaOperator:
    """Validated CPU cache adapter, with exact adjoint and bounded fragments."""
    def __init__(self, cache, count, pixels, *, chunk_size=16384):
        # Full alpha/identity validation is done by public memory.read below.
        self.rows = cache['ids'].long().cpu()
        self.pixel = cache['pixel'].long().cpu()
        self.weight = cache['weight'].double().cpu()
        self.count, self.pixels, self.chunk_size = count, pixels, chunk_size
        # Coalesce duplicate (pixel,ID) fragments so the Jacobi diagonal is exact.
        key, inverse = torch.unique(self.pixel*count+self.rows, return_inverse=True)
        weight = torch.zeros(len(key), dtype=torch.float64).index_add_(0, inverse, self.weight)
        self.diag_pixel, self.diag_rows, self.diag_weight = key//count, key%count, weight

    def forward(self, features):
        result = features.new_zeros(self.pixels, features.shape[1])
        for start in range(0, len(self.rows), self.chunk_size):
            end = start+self.chunk_size
            result.index_add_(0, self.pixel[start:end],
                self.weight[start:end,None].to(features.dtype)*features[self.rows[start:end]])
        return result

    def adjoint(self, values):
        result = values.new_zeros(self.count, values.shape[1])
        for start in range(0, len(self.rows), self.chunk_size):
            end = start+self.chunk_size
            result.index_add_(0, self.rows[start:end],
                self.weight[start:end,None].to(values.dtype)*values[self.pixel[start:end]])
        return result

    def diagonal(self, confidence):
        return self.weight.new_zeros(self.count).index_add_(0, self.diag_rows,
            self.diag_weight.square()*confidence[self.diag_pixel])


@torch.no_grad()
def write_residual(memory, observed, cache, *, ids, confidence, validity=None,
                   writable=None, background=None, regularization=.001,
                   max_iterations=24, trust_radius=8., tolerance=1e-8):
    """Fit one CURRENT observed C,H,W map; preserve existing default writer.

    Confidence must be provided in[0,1]. Writable is an optional Boolean N-vector
    for geometry/semantic exclusions; a contributing pixel is not automatically
    trusted. Background is explicit C,H,W; omitted means zero latent background.
    Only snapshots restored through GaussianLatentMemory's public API commit.
    This CPU reference is intentionally bounded and returns JSON-safe diagnostics.
    """
    if not isinstance(memory, GaussianLatentMemory) or memory.device.type != 'cpu':
        raise ValueError('CPU GaussianLatentMemory required')
    if type(max_iterations) is not int or not 1 <= max_iterations <= 64:
        raise ValueError('max_iterations must be in[1,64]')
    for name, value in (('regularization',regularization), ('trust_radius',trust_radius), ('tolerance',tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if trust_radius > 1e4 or not 1e-12 <= regularization <= 1e6 or tolerance > 1:
        raise ValueError('Bounded trust/regularization/tolerance required')
    if (not isinstance(observed,torch.Tensor) or observed.ndim != 3
            or observed.shape[0] != memory.channels or min(observed.shape) < 1
            or not observed.is_floating_point() or observed.device.type != 'cpu'
            or not bool(torch.isfinite(observed).all())):
        raise ValueError('Finite CPU observed C,H,W features required')
    channels,height,width = observed.shape
    if (not isinstance(confidence,torch.Tensor) or confidence.shape != (height,width)
            or not confidence.is_floating_point() or confidence.device.type != 'cpu'
            or not bool(torch.isfinite(confidence).all())
            or bool(((confidence < 0)|(confidence > 1)).any())):
        raise ValueError('Explicit finite CPU confidence in[0,1] required')
    if validity is not None and (not isinstance(validity,torch.Tensor) or validity.dtype != torch.bool
            or validity.shape != (height,width) or validity.device.type != 'cpu'):
        raise ValueError('CPU Boolean H,W validity required')
    if writable is not None and (not isinstance(writable,torch.Tensor) or writable.dtype != torch.bool
            or writable.shape != (memory.count,) or writable.device.type != 'cpu'):
        raise ValueError('CPU Boolean N writable gate required')
    if background is None:
        background = torch.zeros_like(observed)
    if (not isinstance(background,torch.Tensor) or background.shape != observed.shape
            or not background.is_floating_point() or background.device.type != 'cpu'
            or not bool(torch.isfinite(background).all())):
        raise ValueError('Finite CPU C,H,W background required')
    if not isinstance(cache,dict) or any(not isinstance(cache.get(k),torch.Tensor)
            or cache[k].device.type != 'cpu' for k in ('pixel','ids','weight')):
        raise ValueError('CPU alpha cache required')
    if len(cache['weight']) > 8_000_000 or memory.count*height*width >= 2**63:
        raise ValueError('Cache exceeds bounded CPU reference workload')
    # Public validation checks identity order, finite weights and alpha coverage.
    memory.read(cache,height,width,ids=ids)
    old = memory.snapshot()
    prior = old['features']
    op = SparseAlphaOperator(cache,memory.count,height*width)
    q = confidence.double().reshape(-1).clone()
    if validity is not None:
        q *= validity.reshape(-1)
    evidence = op.adjoint(q[:,None])[:,0]
    active = evidence > 0
    if writable is not None:
        active &= writable
    prior_diag = 1.+old['observation_mass'].double()
    target = observed.double().permute(1,2,0).reshape(-1,channels)
    bg = background.to(observed.dtype).permute(1,2,0).reshape(-1,channels)
    # Compose in observation precision BEFORE promotion. Subtracting a separately
    # promoted background can undo FP32 rounding and invent evidence on readback.
    def prediction(features):
        return (op.forward(features).to(observed.dtype)+bg).double()
    residual = target-prediction(prior)
    initial = float(.5*(q[:,None]*residual.square()).sum())
    stats = dict(method='fixed_geometry_residual_feature_fit', geometry_updated=False,
        generator_weights_trained=False, geometry_dimension_inferred=False,
        regularization=float(regularization), max_iterations=max_iterations,
        trust_radius=float(trust_radius), initial_objective=initial,
        final_objective=initial, objective_history=[initial], iterations=0,
        updated_gaussians=0, committed=False, writes_before=memory.writes,
        writes_after=memory.writes, active_gaussians=int(active.sum()),
        hidden_or_excluded_unchanged=True, max_feature_update_norm=0.,
        limits='Fixed supplied geometry only; no learned correspondence, no motion generation, no automatic confidence. Ill-conditioned alpha maps may amplify noise.')
    if not math.isfinite(initial):
        raise ValueError('Nonfinite initial objective; no state mutation')
    if not bool(active.any()) or not bool((q[:,None]*residual).any()):
        stats['stop_reason'] = 'zero_evidence_or_exact_readback'
        return stats

    def normal(value):
        masked = value*active[:,None]
        return (op.adjoint(q[:,None]*op.forward(masked))+
                regularization*prior_diag[:,None]*masked)*active[:,None]

    diagonal = op.diagonal(q)+regularization*prior_diag
    rhs = op.adjoint(q[:,None]*residual)*active[:,None]
    delta = torch.zeros_like(rhs)
    r = rhs.clone(); preconditioned = r/diagonal[:,None]
    direction = preconditioned.clone()
    rz = (r*preconditioned).sum()
    rhs_norm = float(rhs.norm())
    best, best_objective = prior.clone(), initial
    for iteration in range(max_iterations):
        if float(r.norm()) <= tolerance*max(rhs_norm,1e-30):
            stats['stop_reason'] = 'normal_residual_tolerance'
            break
        hd = normal(direction)
        curvature = (direction*hd).sum()
        if not bool(torch.isfinite(curvature)) or float(curvature) <= 0:
            stats['stop_reason'] = 'nonpositive_or_nonfinite_curvature'
            break
        alpha = rz/curvature
        delta += alpha*direction
        max_norm = float(delta.norm(dim=1).max())
        scale = min(1.,trust_radius/max(max_norm,1e-30))
        candidate = (prior.double()+scale*delta).float()
        candidate[~active] = prior[~active]
        change = candidate.double()-prior.double()
        fit_error = prediction(candidate)-target
        objective = float(.5*(q[:,None]*fit_error.square()).sum()+
                          .5*regularization*(prior_diag[:,None]*change.square()).sum())
        stats['iterations'] = iteration+1
        if (bool(torch.isfinite(candidate).all()) and math.isfinite(objective)
                and objective <= best_objective
                and float(change.norm(dim=1).max()) <= trust_radius*(1.+1e-6)):
            best, best_objective = candidate, objective
            stats['objective_history'].append(objective)
        r -= alpha*hd
        new_preconditioned = r/diagonal[:,None]
        new_rz = (r*new_preconditioned).sum()
        if not bool(torch.isfinite(new_rz)):
            stats['stop_reason'] = 'nonfinite_solver_residual'
            break
        direction = new_preconditioned+(new_rz/rz)*direction
        rz = new_rz
    else:
        stats['stop_reason'] = 'iteration_budget'

    changed = (best != prior).any(dim=1)
    if bool(changed.any()):
        updated = dict(old)
        updated['features'] = best
        mass = old['observation_mass'].clone()
        mass[active] = (mass[active].double()+evidence[active]).clamp_max(memory.max_observation_mass).float()
        updated['observation_mass'] = mass
        updated['writes'] = old['writes']+1
        # restore validates complete state before its public, atomic-ish commit.
        memory.restore(updated)
        stats.update(committed=True,updated_gaussians=int(changed.sum()),
            final_objective=best_objective,writes_after=memory.writes,
            max_feature_update_norm=float((best.double()-prior.double()).norm(dim=1).max()))
    return stats
