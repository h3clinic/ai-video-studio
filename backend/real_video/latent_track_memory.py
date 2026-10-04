"""First-frame latent memory recalled along supplied, recorded 2D tracks.

This is a *planar motion-transfer condition*, not 3D recovery, learned motion,
or a causal future-trajectory predictor. The only appearance input is C,H,W
from the first frame. It is written once through an actual alpha-compositing
cache into ``GaussianLatentMemory``; every output time is read from that bank.

Math: alpha_i(p) = .99 v_i exp(-||p-x_i||^2/(2 sigma^2));
A_i(p) = alpha_i(p) product_{j<i}(1-alpha_j(p)). This is the standard latent
alpha recall in Robust Dreamer, arXiv:2605.30855v1 section 3.2, equation 7
(https://arxiv.org/html/2605.30855v1), specialized by us to isotropic planar
tracks. There is NO measured depth: increasing immutable numeric ID supplies
an arbitrary, deterministic front-to-back order at collisions. This is not
that paper's trained 3D writer, 4D memory, or deviation-learning method.

The existing memory writer is a weighted observation mean, NOT an inverse of
alpha blending, so overlapping source footprints can mix appearance. Recall
is an alpha-weighted sum, never divided by coverage. Memory confidence and
coverage are not evidence of correct correspondence, anatomy, or detail.
"""

import hashlib
import math
import numbers

import torch

from .gaussian_latent_memory import GaussianLatentMemory, bind_alpha_cache, generation_vector
from .wan_temporal_control import wan_temporal_average


OPACITY = .99
ALPHA_MIN = 1e-4
MAX_CACHE_CANDIDATES = 8_000_000


def _floating_cpu(value, name, ndim):
    if (not isinstance(value, torch.Tensor) or value.ndim != ndim
            or not value.is_floating_point() or value.device.type != 'cpu'):
        raise ValueError(f'{name} must be a floating CPU tensor with {ndim} dimensions')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')
    converted = value.detach().float()
    if not bool(torch.isfinite(converted).all()):
        raise ValueError(f'{name} overflows float32 storage')
    return converted


def _visibility(value, shape):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.device.type != 'cpu'
            or (value.dtype != torch.bool and not value.is_floating_point())):
        raise ValueError(f'visibility must be a floating or boolean CPU tensor with shape {shape}')
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError('visibility must be finite and lie in [0,1]')
    return value.detach().float()


def _identity(ids, count):
    if (not isinstance(ids, torch.Tensor) or ids.shape != (count,)
            or ids.device.type != 'cpu' or ids.dtype not in (torch.int32, torch.int64)
            or count < 1 or ids.unique().numel() != count):
        raise ValueError('ids must be a nonempty unique integer CPU vector with one ID per track')
    return ids.detach().long().clone()


def _sigma(value):
    if (not isinstance(value, numbers.Real) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        raise ValueError('sigma must be positive and finite, in latent pixels')
    extent = float(value) * math.sqrt(2 * math.log(OPACITY / ALPHA_MIN))
    if not math.isfinite(extent) or extent > MAX_CACHE_CANDIDATES:
        raise ValueError('sigma would exceed the bounded cache workload')
    return float(value), math.ceil(extent)


def _inside(points, height, width):
    return ((points[..., 0] >= 0) & (points[..., 0] <= width-1)
            & (points[..., 1] >= 0) & (points[..., 1] <= height-1))


def _empty_cache():
    return dict(pixel=torch.empty(0, dtype=torch.long),
                ids=torch.empty(0, dtype=torch.long), weight=torch.empty(0))


@torch.no_grad()
def planar_gaussian_cache(centers, visibility, ids, height, width, *, sigma=.35,
                          generations=None, asset_digest=None):
    """Build CPU pixel/original-row/alpha-transmittance weights for one time.

    Centers are N,2 continuous x,y coordinates: integers denote pixel centers,
    not pixel edges. Out-of-domain *centers* contribute nothing, even if a
    footprint would overlap the border. Source-row keys remain distinct from
    arbitrary external immutable IDs. Each pixel is composited in increasing
    numeric-ID order, with no inference of depth or physical occlusion.
    """
    centers = _floating_cpu(centers, 'centers', 2)
    if centers.shape[1] != 2 or centers.shape[0] < 1:
        raise ValueError('centers must have shape N,2 with N positive')
    if any(type(value) is not int or value < 1 for value in (height, width)):
        raise ValueError('height and width must be positive integers')
    count = centers.shape[0]
    ids = _identity(ids, count)
    if (generations is None) != (asset_digest is None):
        raise ValueError('Cache binding requires both generations and asset_digest')
    def finish(cache):
        return (cache if generations is None else bind_alpha_cache(cache, ids=ids,
            generations=generations, asset_digest=asset_digest, height=height, width=width))
    # Validate even when no Gaussian contributes to the view.
    if generations is not None:
        generations = generation_vector(generations, count)
        finish(_empty_cache())
    visibility = _visibility(visibility, (count,))
    sigma, radius = _sigma(sigma)
    active_rows = torch.nonzero(_inside(centers, height, width) & (visibility > 0), as_tuple=False).flatten()
    if active_rows.numel() == 0:
        return finish(_empty_cache())

    # ceil(r_alpha) around floor(center) covers every above-threshold sample.
    # Clip the offset range to the map extent before bounded allocation.
    rx, ry = min(radius, width), min(radius, height)
    candidate_count = active_rows.numel() * (2*rx+1) * (2*ry+1)
    if candidate_count > MAX_CACHE_CANDIDATES:
        raise ValueError(f'Gaussian cache needs {candidate_count} candidate fragments; bounded limit is {MAX_CACHE_CANDIDATES}')
    oy, ox = torch.meshgrid(torch.arange(-ry, ry+1), torch.arange(-rx, rx+1), indexing='ij')
    offsets = torch.stack((ox.flatten(), oy.flatten()), dim=-1)
    selected = centers[active_rows]
    pixels_xy = selected.floor().long()[:, None] + offsets[None]
    scaled_distance2 = ((pixels_xy.double() - selected.double()[:, None]) / sigma).square().sum(-1)
    alpha = OPACITY * visibility[active_rows, None].double() * torch.exp(-.5*scaled_distance2)
    valid = ((pixels_xy[..., 0] >= 0) & (pixels_xy[..., 0] < width)
             & (pixels_xy[..., 1] >= 0) & (pixels_xy[..., 1] < height)
             & (alpha >= ALPHA_MIN))
    if not bool(valid.any()):
        return finish(_empty_cache())
    pixel = (pixels_xy[..., 1]*width + pixels_xy[..., 0])[valid]
    rows = active_rows[:, None].expand_as(alpha)[valid]
    alpha = alpha[valid]

    # Two stable sorts avoid multiplication/overflow of arbitrary ID values.
    order = torch.argsort(ids[rows], stable=True)
    order = order[torch.argsort(pixel[order], stable=True)]
    pixel, rows, alpha = pixel[order], rows[order], alpha[order]
    first = torch.ones_like(pixel, dtype=torch.bool)
    first[1:] = pixel[1:] != pixel[:-1]
    starts = torch.nonzero(first, as_tuple=False).flatten()
    lengths = torch.diff(torch.cat((starts, starts.new_tensor([len(pixel)]))))
    # FP64 log-prefix subtraction keeps long, highly colliding segments stable.
    # Opacity is strictly below one, so log1p never receives -1.
    log_prefix = torch.cat((torch.zeros(1, dtype=torch.float64), torch.log1p(-alpha).cumsum(0)))
    local_exclusive = log_prefix[:-1] - torch.repeat_interleave(log_prefix[starts], lengths)
    weight = (alpha * local_exclusive.clamp_max(0).exp()).float()
    return finish(dict(pixel=pixel, ids=rows, weight=weight))


@torch.no_grad()
def build_anchor_memory_condition(anchor, tracks, visibility, asset_digest, *, sigma=.35, ids=None,
                                  return_snapshot=False, track_generations=None):
    """Return ``(condition, metrics)`` with shape 1,C+1,1+(T-1)/4,H,W.

    ``anchor`` is ONLY the first frame's C,H,W latent. ``tracks`` is T,N,2,
    supplied at image-frame rate in latent-map pixel-center coordinates, and
    ``visibility`` is T,N. T must be 1+4k. ``ids`` defaults to immutable row IDs
    0..N-1; provide explicit IDs to retain identity under track permutation.
    For recycled slots supply nondecreasing integer T,N ``track_generations``.
    Replaced lifetimes lose their old features and contribute no anchor recall.
    Omitting this retains the legacy assumption that IDs are never reassigned.

    Source visibility/bounds and positive written evidence gate future reads.
    All first-frame pixels are supplied with explicit write confidence one;
    the source cache determines which Gaussian IDs receive evidence. Source
    fractional visibility also scales future opacity. An unseen source ID
    never gains a remembered feature from a future frame.

    Every image-rate latent recall plus its alpha-coverage channel is reduced
    by ``wan_temporal_average``. This arithmetic temporal reduction is not
    Wan's nonlinear VAE and does not produce a causal motion model. In
    particular, recorded FUTURE tracks are external motion-transfer controls.
    This function has no future RGB/features input and performs exactly one
    write, not online generated-geometry/appearance writeback.

    With ``return_snapshot=True``, a third return value contains the restorable
    Gaussian memory plus first-frame centers/visibility and render convention.
    It contains neither future tracks nor a target feature/video history, and
    can be recalled without re-encoding the source through a VAE. New motion
    still requires independently supplied tracks; the snapshot predicts none.
    """
    if type(return_snapshot) is not bool:
        raise ValueError('return_snapshot must be boolean')
    anchor = _floating_cpu(anchor, 'anchor', 3)
    tracks = _floating_cpu(tracks, 'tracks', 3)
    if any(dimension < 1 for dimension in anchor.shape):
        raise ValueError('anchor must be a nonempty C,H,W tensor')
    channels, height, width = anchor.shape
    if tracks.shape[-1] != 2 or tracks.shape[0] < 1 or tracks.shape[1] < 1:
        raise ValueError('tracks must have shape T,N,2 with T and N positive')
    frames, count, _ = tracks.shape
    if (frames-1) % 4:
        raise ValueError('Track frame count must equal 1 + 4*k')
    visible = _visibility(visibility, (frames, count))
    sigma, radius = _sigma(sigma)
    stable_ids = _identity(torch.arange(count) if ids is None else ids, count)
    generations = None
    if track_generations is not None:
        if (not isinstance(track_generations, torch.Tensor)
                or track_generations.shape != (frames, count)
                or track_generations.device.type != 'cpu'):
            raise ValueError('track_generations must be a CPU T,N integer tensor')
        generations = generation_vector(track_generations.reshape(-1), frames*count).reshape(frames, count)
        if bool((generations[1:] < generations[:-1]).any()):
            raise ValueError('Track generations must be nondecreasing')
    def cache_at(time, visibility):
        kwargs = {} if generations is None else dict(generations=generations[time], asset_digest=asset_digest)
        return planar_gaussian_cache(tracks[time], visibility, stable_ids, height, width, sigma=sigma, **kwargs)
    memory = GaussianLatentMemory(stable_ids, channels, asset_digest,
                                  generations=None if generations is None else generations[0])
    source_cache = cache_at(0, visible[0])
    write = memory.write(anchor, source_cache, ids=stable_ids, confidence=torch.ones(height, width))
    canonical = memory.snapshot()
    known = canonical['observation_mass'] > 0
    source_visibility = visible[0] * _inside(tracks[0], height, width) * known
    recalled, coverage_means, coverage_maxima = [], [], []
    cache_fragments = []
    for time in range(frames):
        retained = torch.ones(count, dtype=torch.bool)
        if generations is not None:
            memory.advance_generations(generations[time], ids=stable_ids)
            retained = generations[time] == generations[0]
        # Frame zero uses exactly the cache that supplied its sole write.
        cache = source_cache if time == 0 else cache_at(time, visible[time]*source_visibility*retained)
        result = memory.read(cache, height, width, ids=stable_ids)
        recalled.append(torch.cat((result['features'], result['coverage'][None]), dim=0))
        coverage_means.append(float(result['coverage'].mean()))
        coverage_maxima.append(float(result['coverage'].max()))
        cache_fragments.append(int(cache['weight'].numel()))
    frame_volume = torch.stack(recalled, dim=1)[None]
    condition = wan_temporal_average(frame_volume)
    metrics = dict(
        schema='planar_anchor_gaussian_latent_condition_v1',
        scope='First-frame latent appearance recall along supplied recorded future 2D tracks; motion transfer only',
        asset_digest=asset_digest,
        identity_sha256=hashlib.sha256(stable_ids.numpy().tobytes()).hexdigest(),
        gaussian_count=count, source_writes=memory.writes,
        source_updated_gaussians=write['updated_gaussians'],
        source_evidence_mass=write['incoming_evidence_mass'],
        source_frame_only=True, future_feature_inputs=False,
        material_lifetimes_checked=generations is not None,
        predicted_motion=False, generated_geometry=False, measured_depth=False,
        collision_order='Increasing immutable numeric ID; arbitrary planar approximation, NOT measured depth',
        opacity=OPACITY, alpha_min=ALPHA_MIN, sigma_latent_pixels=sigma,
        footprint_radius_latent_pixels=radius,
        source_frames=frames, condition_shape=list(condition.shape),
        recall_is_coverage_normalized=False,
        temporal_reduction='Keep frame zero, then arithmetic mean of each four recalled frames; not a VAE encoding',
        coverage_mean_by_frame=coverage_means, coverage_max_by_frame=coverage_maxima,
        cache_fragments_by_frame=cache_fragments,
        memory=memory.memory_bytes(),
        condition_tensor_bytes=condition.numel()*condition.element_size(),
        dense_recall_tensor_bytes=frame_volume.numel()*frame_volume.element_size(),
        supplied_track_tensor_bytes=tracks.numel()*tracks.element_size(),
        supplied_visibility_tensor_bytes=visible.numel()*visible.element_size(),
        supplied_generation_tensor_bytes=(0 if generations is None else generations.numel()*generations.element_size()),
        limits='No hidden geometry, new identity, predicted dynamics, learned writer, appearance refinement, or claimed compute savings. Input tracks, dense condition, caches, model weights and transient allocations are not resident Gaussian memory.',
    )
    if return_snapshot:
        canonical.update(
            condition_snapshot_schema=('planar_anchor_gaussian_latent_snapshot_v1' if generations is None
                                       else 'planar_anchor_gaussian_latent_snapshot_lifetimes_v2'),
            canonical_first_centers=tracks[0].clone(),
            canonical_visibility=visible[0].clone(),
            sigma=sigma, opacity=OPACITY, alpha_min=ALPHA_MIN,
            latent_height=height, latent_width=width,
            coordinate_convention='planar latentpixelcenter indices',
            source_image_digest=asset_digest,
            collision_order='Increasing immutable numeric ID; no measured depth',
        )
        return condition, metrics, canonical
    return condition, metrics
