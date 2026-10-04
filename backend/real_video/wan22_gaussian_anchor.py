"""Native-latent first-frame Gaussian memory for the Wan2.2 TI2V protocol.

One planar Gaussian is anchored at each integer latent pixel center, with
fixed identity and one write of an already normalized C,H,W source latent.
Wan2.2-TI2V uses C=48; the algebra here is channel-independent. This is not
the 400,000-point XYZ cat, a learned geometry writer, or generated writeback.
No future images, tracks, training targets, or encoder are accepted.

Alpha read/write uses GaussianLatentMemory and the explicit planar Gaussian
cache, not a renamed copy of the source. The default sigma=.35 pixels retains
small footprints. Reads are raw alpha-weighted sums, NOT coverage normalized;
the dense integer grid ensures >=.99 coverage. The writer and reader can blur
features even at full coverage, so error and gradient-energy metrics are
reported separately. A bank is not a compressed replacement for the VAE.

The optional denoising helpers reproduce the installed Diffusers 0.40
WanImageToVideoPipeline expand_timesteps path: first-frame mask zero, future
mask one; blend only the first source latent; patch (1,2,2) time tokens have
time zero on first-frame tokens. This is distinct from Wan2.1 concatenated
I2V conditioning. Reference: pipelines/wan/pipeline_wan_i2v.py, prepare_latents
and __call__. No model weights or GPU are loaded by this module.
"""

import hashlib
import math
import numbers
import time

import torch

from .gaussian_latent_memory import GaussianLatentMemory
from .latent_track_memory import ALPHA_MIN, OPACITY, planar_gaussian_cache


SNAPSHOT_SCHEMA = 'wan22_planar_gaussian_anchor_v1'
COORDINATES = 'planar latentpixelcenter indices'


def _finite_float(value, ndim, name, *, cpu=False):
    if (not isinstance(value, torch.Tensor) or value.ndim != ndim
            or not value.is_floating_point() or any(size < 1 for size in value.shape)):
        raise ValueError(f'{name} must be a nonempty floating {ndim}D tensor')
    if cpu and value.device.type != 'cpu':
        raise ValueError(f'{name} must be on CPU')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')
    return value


def _grid(height, width):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    return torch.stack((x, y), -1).reshape(-1, 2).float()


def _id_digest(ids):
    return hashlib.sha256(ids.detach().cpu().long().contiguous().numpy().tobytes()).hexdigest()


def _ratio(numerator, denominator):
    return numerator/denominator if denominator > 0 else (0. if numerator == 0 else None)


def _gradient_energy(value):
    pieces = []
    if value.shape[-1] > 1:
        pieces.append((value[..., 1:]-value[..., :-1]).square().reshape(-1))
    if value.shape[-2] > 1:
        pieces.append((value[:, 1:, :]-value[:, :-1, :]).square().reshape(-1))
    return float(torch.cat(pieces).mean()) if pieces else 0.


@torch.no_grad()
def recall_gaussian_anchor(snapshot):
    """Restore a saved bank and return raw C,H,W recall plus H,W coverage.

    Source RGB and VAE/encoder are neither needed nor accepted. Only the
    canonical first-frame grid can be read by this helper; novel geometry
    would require a separate, explicitly justified correspondence mechanism.
    """
    if not isinstance(snapshot, dict) or snapshot.get('anchor_snapshot_schema') != SNAPSHOT_SCHEMA:
        raise ValueError('Expected a native planar first-frame Gaussian snapshot')
    height, width = snapshot.get('latent_height'), snapshot.get('latent_width')
    if any(type(size) is not int or size < 1 for size in (height, width)):
        raise ValueError('Positive latent grid dimensions required')
    if (snapshot.get('coordinate_convention') != COORDINATES
            or snapshot.get('opacity') != OPACITY or snapshot.get('alpha_min') != ALPHA_MIN):
        raise ValueError('Snapshot rendering convention does not match this reader')
    if snapshot.get('source_image_digest') != snapshot.get('asset_digest'):
        raise ValueError('Source image digest and memory asset binding disagree')
    centers, visible = snapshot.get('canonical_first_centers'), snapshot.get('canonical_visibility')
    if (not isinstance(centers, torch.Tensor) or centers.device.type != 'cpu'
            or not torch.equal(centers, _grid(height, width))):
        raise ValueError('This snapshot requires the unchanged first-frame pixel-center grid')
    if (not isinstance(visible, torch.Tensor) or visible.device.type != 'cpu'
            or not torch.equal(visible, torch.ones(height*width))):
        raise ValueError('Dense first-frame visibility must remain one at each Gaussian')
    memory = GaussianLatentMemory(snapshot.get('ids'), snapshot.get('channels'), snapshot.get('asset_digest'))
    if memory.count != height*width or _id_digest(snapshot['ids']) != snapshot.get('identity_sha256'):
        raise ValueError('Gaussian identities or native grid size changed')
    memory.restore(snapshot)
    if memory.writes != 1 or not bool((snapshot['observation_mass'] > 0).all()):
        raise ValueError('Expected exactly one observed source write with evidence for every Gaussian')
    cache = planar_gaussian_cache(centers, visible, snapshot['ids'], height, width, sigma=snapshot.get('sigma'))
    result = memory.read(cache, height, width, ids=snapshot['ids'])
    if not bool(torch.isfinite(result['features']).all()) or float(result['coverage'].min()) < OPACITY-1e-5:
        raise ValueError('Nonfinite or unexpectedly attenuated canonical anchor recall')
    return result['features'], result['coverage']


@torch.no_grad()
def build_gaussian_anchor(anchor, source_image_digest, *, sigma=.35, ids=None):
    """Return ``(recalled_C_HW, metrics, restorable_snapshot)`` on CPU.

    Supply the native VAE-normalized FIRST FRAME latent only. Its channels are
    neither resized nor bridged to Wan2.1's 16-channel space. Gaussian IDs are
    canonical row indices by default; explicit IDs must be a unique N-vector.
    No value is rounded to integer precision; FP32 cast loss is measured.
    """
    start = time.perf_counter()
    original = _finite_float(anchor, 3, 'anchor', cpu=True).detach()
    source = original.float()
    if not bool(torch.isfinite(source).all()):
        raise ValueError('Anchor overflows native FP32 Gaussian memory')
    channels, height, width = source.shape
    count = height*width
    stable_ids = torch.arange(count) if ids is None else ids
    memory = GaussianLatentMemory(stable_ids, channels, source_image_digest)
    if memory.count != count:
        raise ValueError('Exactly one Gaussian identity per native latent pixel required')
    stable_ids = memory.snapshot()['ids']
    centers, visible = _grid(height, width), torch.ones(count)
    cache = planar_gaussian_cache(centers, visible, stable_ids, height, width, sigma=sigma)
    written = memory.write(source, cache, ids=stable_ids, confidence=torch.ones(height, width))
    snapshot = memory.snapshot()
    snapshot.update(anchor_snapshot_schema=SNAPSHOT_SCHEMA,
        source_image_digest=source_image_digest, identity_sha256=_id_digest(stable_ids),
        canonical_first_centers=centers.clone(), canonical_visibility=visible.clone(),
        sigma=float(sigma), opacity=OPACITY, alpha_min=ALPHA_MIN,
        latent_height=height, latent_width=width, coordinate_convention=COORDINATES,
        collision_order='Increasing immutable numeric ID; planar approximation, not measured depth')
    recalled, coverage = recall_gaussian_anchor(snapshot)
    before, after = source.double(), recalled.double()
    error = after-before
    rms = float(before.square().mean().sqrt())
    rmse = float(error.square().mean().sqrt())
    cast_rmse = float((original.double()-before).square().mean().sqrt())
    direct_bf16 = source.to(torch.bfloat16).float().double()
    recall_bf16 = recalled.to(torch.bfloat16).float().double()
    metrics = dict(schema=SNAPSHOT_SCHEMA,
        scope='One-time planar native-latent first-frame Gaussian appearance memory; not a 3D cat',
        source_image_digest=source_image_digest, channels=channels, latent_height=height, latent_width=width,
        gaussian_count=count, source_writes=memory.writes,
        source_updated_gaussians=written['updated_gaussians'],
        fixed_ids=True, first_frame_only=True, future_input=False, generated_writeback=False,
        weights_modified=False, geometry_dimension=2, predicted_motion=False,
        coverage_min=float(coverage.min()), coverage_mean=float(coverage.mean()), coverage_max=float(coverage.max()),
        coverage_normalized=False, sigma_latent_pixels=float(sigma), opacity=OPACITY, alpha_min=ALPHA_MIN,
        source_dtype=str(original.dtype), storage_dtype='torch.float32',
        fp32_cast_rmse=cast_rmse, latent_rmse=rmse,
        relative_latent_rmse=_ratio(rmse, rms), latent_max_abs_error=float(error.abs().max()),
        recalled_to_source_rms_ratio=_ratio(float(after.square().mean().sqrt()), rms),
        spatial_gradient_energy_ratio=_ratio(_gradient_energy(after), _gradient_energy(before)),
        direct_bf16_cast_rmse=float((direct_bf16-before).square().mean().sqrt()),
        recalled_bf16_cast_rmse=float((recall_bf16-after).square().mean().sqrt()),
        paired_bf16_latent_rmse=float((recall_bf16-direct_bf16).square().mean().sqrt()),
        memory=memory.memory_bytes(),
        canonical_geometry_tensor_bytes=centers.numel()*centers.element_size()+visible.numel()*visible.element_size(),
        source_latent_tensor_bytes=source.numel()*source.element_size(),
        build_and_restore_seconds=time.perf_counter()-start,
        exclusions='Memory bytes exclude cache, temporary maps, source/condition latents, model weights, VAE and Python metadata',
        interpretation='Coverage is not detail quality. Weighted observation writing and alpha recall can mix neighbors; latent errors are not decoded-video quality or compute savings.')
    return recalled, metrics, snapshot


def ti2v_first_frame_mask(latents):
    """Native expand_timesteps mask: zero only at t=0, one everywhere later."""
    _finite_float(latents, 5, 'latents')
    _, _, frames, height, width = latents.shape
    mask = latents.new_ones(1, 1, frames, height, width)
    mask[:, :, 0] = 0
    return mask


def apply_ti2v_anchor(latents, anchor):
    """Replace only the first latent frame; preserve every future noise value."""
    mask = ti2v_first_frame_mask(latents)
    if isinstance(anchor, torch.Tensor) and anchor.ndim == 3:
        anchor = anchor[None, :, None]
    _finite_float(anchor, 5, 'anchor')
    if (anchor.shape[0] not in (1, latents.shape[0]) or anchor.shape[1] != latents.shape[1]
            or anchor.shape[2] != 1 or anchor.shape[3:] != latents.shape[3:]
            or anchor.device != latents.device):
        raise ValueError('Anchor must match latent batch/channels/spatial shape and contain ONLY one frame on the same device')
    return (1-mask)*anchor + mask*latents


def ti2v_denoising_inputs(latents, anchor, timestep, *, patch_size=(1, 2, 2)):
    """Return native model input, B,token-count time map, and first-frame mask.

    This explicitly supports only the inspected TI2V patch-(1,2,2) protocol.
    It neither concatenates channels nor forces any future frame to a copy.
    """
    conditioned = apply_ti2v_anchor(latents, anchor)
    if tuple(patch_size) != (1, 2, 2) or latents.shape[-1] % 2 or latents.shape[-2] % 2:
        raise ValueError('Inspected TI2V protocol requires patch (1,2,2) and even latent spatial dimensions')
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() != 1 or not bool(torch.isfinite(timestep).all()) or float(timestep) < 0:
            raise ValueError('One finite nonnegative diffusion timestep required')
        step = timestep.to(device=latents.device).reshape(())
    elif isinstance(timestep, numbers.Real) and not isinstance(timestep, bool) and math.isfinite(timestep) and timestep >= 0:
        step = timestep
    else:
        raise ValueError('One finite nonnegative diffusion timestep required')
    mask = ti2v_first_frame_mask(latents)
    tokens = (mask[0, 0, :, ::2, ::2]*step).flatten()[None].expand(latents.shape[0], -1)
    return conditioned, tokens, mask
