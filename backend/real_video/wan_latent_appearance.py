"""Trainable appearance recall alongside the existing five-channel Wan control.

This is a small local hypothesis, not a Robust Dreamer or Wan-Move reproduction.
Robust Dreamer 2605.30855v1 section 3.2 equations 5--7 motivates Gaussian latent
readback; Wan-Move 2512.08765v1 section 3.2 motivates native VAE features instead
of RGB-only control. Neither paper establishes this particular residual branch.

The Gaussian geometry/correspondences are supplied, not predicted. The writer
below is the existing frozen normalized-transpose accumulation, not a trained
geometry writer. Training this module learns how to USE recalled appearance.
Recorded future correspondences remain motion transfer, not new motion.
"""
import numbers

import torch
from torch import nn
import torch.nn.functional as F

from .gaussian_latent_memory import GaussianLatentMemory
from .wan_temporal_control import TemporalSpatialControl


def _positive(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value > 0


class LatentAppearanceControl(nn.Module):
    """Six zero-initialized residual heads consuming latent16 + coverage1.

Pixel-unshuffle preserves all four latent cells belonging to each 2x2 Wan
patch, unlike averaging them before the learned encoder. It cannot recover
detail absent from the anchor or undo erroneous Gaussian correspondences.
Bind one clip-level condition until every checkpointed backward has completed.
"""
    channels = 17

    def __init__(self, dim, blocks=(0, 5, 10, 15, 20, 25), hidden=64):
        super().__init__()
        if not _positive(dim) or not _positive(hidden):
            raise ValueError('Positive model and hidden dimensions required')
        if not blocks or any(not isinstance(i, numbers.Integral) or isinstance(i, bool) or i < 0 for i in blocks) or len(set(blocks)) != len(blocks):
            raise ValueError('Unique nonnegative block indices required')
        self.dim, self.blocks = dim, tuple(blocks)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels*4, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, dim) for _ in self.blocks])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.enabled, self.condition, self.grid = True, None, None

    def bind(self, condition, grid):
        if (not isinstance(condition, torch.Tensor) or condition.ndim != 5
                or not condition.is_floating_point() or not bool(torch.isfinite(condition).all())):
            raise ValueError('Finite floating B,17,T,H,W appearance condition required')
        if len(grid) != 3 or not all(_positive(n) for n in grid):
            raise ValueError('Positive T,H,W token grid required')
        t, h, w = grid
        if condition.shape[0] < 1 or tuple(condition.shape[1:]) != (17, t, h*2, w*2):
            raise ValueError('Appearance must retain the complete 2x2 latent patch per Wan token')
        coverage = condition[:, 16:17]
        if bool(((coverage < 0) | (coverage > 1.00001)).any()):
            raise ValueError('Alpha-composited coverage must lie in [0,1]')
        self.condition, self.grid = condition, tuple(grid)

    def clear(self):
        self.condition, self.grid = None, None

    def forward(self, hidden, head_index):
        if not self.enabled or self.condition is None:
            return hidden
        if not isinstance(head_index, numbers.Integral) or isinstance(head_index, bool) or not 0 <= head_index < len(self.heads):
            raise ValueError('Invalid appearance head index')
        t, h, w = self.grid
        condition = self.condition
        if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (t*h*w, self.dim):
            raise ValueError('Appearance and Wan token grids disagree')
        if condition.device != hidden.device or condition.shape[0] not in (1, hidden.shape[0]):
            raise ValueError('Appearance and hidden batch/device mismatch')
        b = condition.shape[0]
        maps = condition.permute(0, 2, 1, 3, 4).reshape(b*t, 17, h*2, w*2)
        maps = F.pixel_unshuffle(maps.to(self.heads[head_index].weight.dtype), 2)
        features = self.encoder(maps).flatten(2).transpose(1, 2)
        features = features.reshape(b, t*h*w, features.shape[-1])
        delta = self.heads[head_index](features)
        if b == 1:
            delta = delta.expand(hidden.shape[0], -1, -1)
        return hidden + delta.to(hidden.dtype)


def install_latent_appearance(model, base_control, *, hidden=64):
    """Freeze the existing model/adapter and add exactly one trainable reader.

Existing Wan block hooks already call ``base_control(output, head_index)``.
We hook that control MODULE's result, not the Wan blocks a second time. The
output is therefore hidden + old_RGB_residual + new_latent_residual. Return
the registered reader for ``bind`` and ``optimizer(reader.parameters())``.
Install after loading old LoRA/control weights; keep the old control enabled.
"""
    if not isinstance(base_control, TemporalSpatialControl):
        raise TypeError('An attached TemporalSpatialControl is required')
    if getattr(model, 'gaussian_temporal_control', None) is not base_control:
        raise ValueError('Base control must already belong to this model')
    if hasattr(base_control, 'latent_appearance'):
        raise ValueError('Latent appearance is already installed')
    if base_control.channels != 5:
        raise ValueError('This bridge preserves the established five-channel branch')
    reader = LatentAppearanceControl(base_control.dim, base_control.blocks, hidden=hidden)
    reference = next(base_control.parameters())
    reader.to(device=reference.device, dtype=reference.dtype)
    # Freeze before registration, so only the new reader remains trainable.
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    base_control.add_module('latent_appearance', reader)

    def append_appearance(module, inputs, output):
        if not module.enabled:
            return output
        if len(inputs) != 2:
            raise ValueError('Temporal control must receive hidden and head index positionally')
        if reader.condition is not None and reader.grid != module.grid:
            raise ValueError('RGB and latent branches must share the same Wan token grid')
        return reader(output, inputs[1])

    reader._control_hook_handle = base_control.register_forward_hook(append_appearance)
    return reader


def init_anchor_and_recall(anchor_latent, anchor_cache, recall_caches, *, ids,
                           asset_digest, confidence, validity=None):
    """Write ONLY a supplied anchor, then recall immutable per-ID features.

``anchor_latent`` is normalized Wan VAE C=16,H,W from the known first image.
All caches must be genuine alpha-compositing caches at that latent resolution,
with ``ids`` ORIGINAL Gaussian row indices, not arbitrary track labels. This
function does not estimate visibility, a camera, depth, or future positions.
Future clean latents are deliberately absent from this API. Planar tracked
Gaussians are permitted but must be labelled planar recorded-motion guidance.

    Channel 17 measures coverage by Gaussians with actual anchor evidence, not
    merely geometric coverage: unseen zero-valued memory is not a valid colour.
    Returned B,17,T,H,W is at the supplied cache clock. The caller is responsible
for causal clock alignment; do not average encoded targets into conditioning.
The frozen write can blur mixed observations; it is not an inverse renderer.
"""
    if not isinstance(anchor_latent, torch.Tensor) or anchor_latent.ndim != 3 or anchor_latent.shape[0] != 16:
        raise ValueError('Known anchor must be a normalized 16-channel Wan latent')
    if not isinstance(recall_caches, (list, tuple)) or not recall_caches:
        raise ValueError('Nonempty explicit recall caches required')
    _, height, width = anchor_latent.shape
    memory = GaussianLatentMemory(ids, 16, asset_digest, device=anchor_latent.device)
    write_report = memory.write(anchor_latent, anchor_cache, ids=ids,
                                confidence=confidence, validity=validity)
    observed = memory.snapshot()['observation_mass'].to(anchor_latent.device) > 0
    conditions = []
    for cache in recall_caches:
        # Validate the original unmodified correspondence first, including
        # original row indices, so masking cannot conceal an invalid cache.
        result = memory.read(cache, height, width, ids=ids)
        known_weight = cache['weight'].to(anchor_latent.device) * observed[cache['ids'].to(anchor_latent.device).long()]
        known_cache = dict(pixel=cache['pixel'], ids=cache['ids'], weight=known_weight)
        known = memory.read(known_cache, height, width, ids=ids)
        conditions.append(torch.cat((result['features'], known['coverage'][None]), dim=0))
    return dict(memory=memory, condition=torch.stack(conditions, dim=1)[None],
                write_report=write_report, anchor_writes=1,
                scope='Frozen fixed-geometry anchor write/read; trainable appearance reader only',
                geometry_updated=False, future_latent_reads=0)
