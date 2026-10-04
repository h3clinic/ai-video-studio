"""Experimental time-varying Gaussian-memory conditioning inside Wan.

Motivation: Wan-Move, arXiv:2512.08765v1, sections 3.2--3.3, equations 2--5:
https://arxiv.org/html/2512.08765v1 . That paper transports first-frame VAE
features along visible trajectories and fine-tunes an existing I2V backbone.
Its coordinate compression averages groups of four future frame coordinates;
its feature collisions use random selection. This module instead implements
our zero-initialized residual port for the local T2V backbone, with bilinear,
weighted-mean collisions. It is NOT a Wan-Move implementation or checkpoint.

All trajectories are explicit inputs. Recorded future tracks demonstrate
motion transfer; this module alone does not predict new Gaussian dynamics,
guarantee 3D identity, or remove Wan's RGB VAE/denoising computation.
"""

import numbers

import torch
from torch import nn
import torch.nn.functional as F


def _positive_int(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value > 0


def _finite_float(value, name, ndim):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating-point {ndim}D tensor')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} contains nonfinite values')


def wan_temporal_average(volume, factor=4):
    """Preserve frame zero; average each subsequent group of ``factor`` frames.

    Input/output: B,C,T,H,W. T must be 1 + k*factor: no silent dropping or
    endpoint interpolation. This aligns the control clock with Wan's temporal
    stride; it is NOT the nonlinear causal VAE encoding. Averaging splatted
    feature volumes is also not Wan-Move's Eq. 2 coordinate-then-splat order.
    Apply to occupancy separately when it is a distinct control channel.
    """
    _finite_float(volume, 'volume', 5)
    if not _positive_int(factor) or any(n < 1 for n in volume.shape):
        raise ValueError('Positive factor and nonempty volume required')
    b, c, t, h, w = volume.shape
    if (t - 1) % factor:
        raise ValueError('Frame count must equal 1 + k*factor')
    if t == 1:
        return volume
    future = volume[:, :, 1:].reshape(b, c, (t-1)//factor, factor, h, w).mean(3)
    return torch.cat((volume[:, :, :1], future), dim=2)


def transport_first_features(first, tracks, visibility=None, weights=None):
    """Bilinearly carry first-frame features along explicitly supplied tracks.

    ``first``: B,C,H,W, ``tracks``: B,T,N,2 in this map's pixel-center x,y
    coordinates. ``visibility``: optional B,T,N boolean/float in [0,1].
    ``weights``: optional nonnegative B,N or B,T,N confidence. Invisible at
    t=0 means no trustworthy source feature, so that track is excluded later.
    Out-of-domain track centers contribute zero, never clipped border copies.

    Returns feature volume B,C,T,H,W and occupancy B,1,T,H,W. The complete
    first map is retained exactly, with occupancy one. Future empty locations
    are zero. Future occupancy is summed bilinear confidence mass (can exceed
    one at collisions); features there are the weighted mean. No depth test is
    inferred: callers must supply surface visibility/occlusion information.
    Gradients reach sampled features, noninteger coordinates and confidence;
    integer bin selection and the boundary visibility decision are discrete.
    Accumulation uses FP32 unless the feature input is FP64.
    """
    _finite_float(first, 'first', 4)
    _finite_float(tracks, 'tracks', 4)
    b, c, h, w = first.shape
    if any(n < 1 for n in first.shape) or tracks.shape[0] != b or tracks.shape[-1] != 2 or tracks.shape[1] < 1:
        raise ValueError('Invalid first map or B,T,N,2 tracks')
    if first.device != tracks.device:
        raise ValueError('Feature and trajectory devices must match')
    _, t, n, _ = tracks.shape
    dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
    source, points = first.to(dtype), tracks.to(dtype)

    def confidence(value, name, allow_static=False):
        if value is None:
            return source.new_ones(b, t, n)
        if not isinstance(value, torch.Tensor) or value.device != first.device:
            raise ValueError(f'{name} must be a tensor on the feature device')
        if value.dtype != torch.bool and not value.is_floating_point():
            raise ValueError(f'{name} must be boolean or floating point')
        if allow_static and value.shape == (b, n):
            value = value[:, None].expand(-1, t, -1)
        if value.shape != (b, t, n) or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f'Invalid {name} shape or values')
        if name == 'visibility' and bool((value > 1).any()):
            raise ValueError('visibility must lie in [0,1]')
        return value.to(dtype)

    visible = confidence(visibility, 'visibility')
    weight = confidence(weights, 'weights', allow_static=True)
    if n == 0 or t == 1:
        future = source.new_zeros(b, c, t-1, h, w)
        mass = source.new_zeros(b, 1, t-1, h, w)
        return torch.cat((source[:, :, None], future), 2), torch.cat((source.new_ones(b, 1, 1, h, w), mass), 2)

    inside = ((points[..., 0] >= 0) & (points[..., 0] <= w-1)
              & (points[..., 1] >= 0) & (points[..., 1] <= h-1))
    initial = points[:, 0]
    gx = initial[..., 0]*2/max(w-1, 1)-1 if w > 1 else torch.zeros_like(initial[..., 0])
    gy = initial[..., 1]*2/max(h-1, 1)-1 if h > 1 else torch.zeros_like(initial[..., 1])
    grid = torch.stack((gx, gy), -1)[:, None]
    sampled = F.grid_sample(source, grid, align_corners=True, mode='bilinear', padding_mode='zeros')[:, :, 0].transpose(1, 2)
    active = weight[:, 1:] * visible[:, 1:] * visible[:, :1] * inside[:, 1:] * inside[:, :1]
    # Clamp only for safe integer conversion; the independent inside mask
    # ensures an out-of-bounds center never appears on the image boundary.
    x = points[:, 1:, :, 0].clamp(-1, w)
    y = points[:, 1:, :, 1].clamp(-1, h)
    x0, y0 = x.floor(), y.floor()
    dx, dy = x-x0, y-y0
    values = sampled[:, None].expand(-1, t-1, -1, -1)
    summed = source.new_zeros(b*(t-1)*h*w, c)
    mass = source.new_zeros(b*(t-1)*h*w, 1)
    offset = torch.arange(b*(t-1), device=first.device).reshape(b, t-1, 1)*(h*w)
    for ox, oy, fraction in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                             (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        xi, yi = x0.long()+ox, y0.long()+oy
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        contribution = active * fraction * valid
        index = (offset + yi.clamp(0, h-1)*w + xi.clamp(0, w-1)).reshape(-1)
        summed.index_add_(0, index, (values*contribution[..., None]).reshape(-1, c))
        mass.index_add_(0, index, contribution.reshape(-1, 1))
    # Exact zero handling avoids amplifying empty cells, without clipping small
    # but legitimate confidence and thereby changing their feature values.
    denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
    future = (summed/denominator).reshape(b, t-1, h, w, c).permute(0, 4, 1, 2, 3)
    occupancy = mass.reshape(b, t-1, h, w, 1).permute(0, 4, 1, 2, 3)
    return torch.cat((source[:, :, None], future), 2), torch.cat((source.new_ones(b, 1, 1, h, w), occupancy), 2)


class TemporalSpatialControl(nn.Module):
    """Per-time-slot spatial residuals: every supplied time can affect Wan.

    Wan's patch tokens flatten T,H,W in that order. A shared 2D encoder acts
    independently at every time, preserving that order; zero heads give exact
    base parity initially. The backbone, not this encoder, mixes time. Bind a
    volume already at Wan's latent temporal rate. Keep the bound tensor/grid
    unchanged until backward finishes when checkpointing transformer blocks.
    """
    def __init__(self, dim, blocks=(0, 5, 10, 15, 20, 25), channels=14, hidden=64):
        super().__init__()
        if not all(_positive_int(v) for v in (dim, channels, hidden)):
            raise ValueError('Positive integer channel dimensions required')
        if not blocks or any(not isinstance(i, numbers.Integral) or isinstance(i, bool) or i < 0 for i in blocks) or len(set(blocks)) != len(blocks):
            raise ValueError('Unique nonnegative block indices required')
        self.dim, self.channels = dim, channels
        self.blocks = tuple(blocks)
        self.encoder = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, dim) for _ in self.blocks])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.condition = None
        self.grid = None
        self.enabled = True

    def bind(self, condition, grid):
        _finite_float(condition, 'condition', 5)
        if len(grid) != 3 or not all(_positive_int(n) for n in grid):
            raise ValueError('Expected positive integer T,H,W token grid')
        if any(n < 1 for n in condition.shape) or condition.shape[1] != self.channels or condition.shape[2] != grid[0]:
            raise ValueError('Condition channels/time must match control/token grid')
        self.condition, self.grid = condition, tuple(grid)

    def forward(self, hidden, head_index):
        if not self.enabled or self.condition is None:
            return hidden
        if not isinstance(head_index, numbers.Integral) or not 0 <= head_index < len(self.heads):
            raise ValueError('Invalid control head index')
        t, h, w = self.grid
        if hidden.ndim != 3 or hidden.shape[1:] != (t*h*w, self.dim):
            raise ValueError('Temporal condition/token ordering mismatch')
        condition = self.condition
        if hidden.device != condition.device or condition.shape[0] not in (1, hidden.shape[0]):
            raise ValueError('Condition batch/device mismatch')
        b, c, _, fh, fw = condition.shape
        maps = condition.permute(0, 2, 1, 3, 4).reshape(b*t, c, fh, fw)
        maps = F.adaptive_avg_pool2d(maps.to(self.heads[head_index].weight.dtype), (h, w))
        features = self.encoder(maps).flatten(2).transpose(1, 2)
        tokens = features.reshape(b, t*h*w, features.shape[-1])
        delta = self.heads[head_index](tokens)
        if b == 1:
            delta = delta.expand(hidden.shape[0], -1, -1)
        return hidden + delta.to(hidden.dtype)


def attach_temporal_control(model, control):
    """Register trainable heads and hooks; return removable hook handles."""
    if not isinstance(control, TemporalSpatialControl):
        raise TypeError('Expected TemporalSpatialControl')
    if hasattr(model, 'gaussian_temporal_control'):
        raise ValueError('Temporal control already attached')
    if max(control.blocks) >= len(model.blocks):
        raise ValueError('Requested block index outside transformer')
    handles = []
    for head_index, block_index in enumerate(control.blocks):
        def hook(module, inputs, output, j=head_index):
            return control(output, j)
        handles.append(model.blocks[block_index].register_forward_hook(hook))
    model.add_module('gaussian_temporal_control', control)
    return handles
