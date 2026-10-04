"""Wan-latent -> planar Gaussian decoder research architecture.

No pretrained Wan weights are transplanted here. This module has no RGB VAE,
source video encoder, or pixel-output head. Gaussian colours are attributes;
the explicit splatter alone produces pixels. Instances start untrained; the
separate v1 pilot checkpoint has limited real-video reconstruction training.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class WanGaussianDecoder(nn.Module):
    """Joint-clip decoder, not recurrent memory or 3D physical dynamics.

    Input is normalized Wan diffusion latent (before original VAE unnormalizing).
    Output B,9,T,Gh,Gw: offsets, scale logits, orientation pair, colour logits.
    Unlike the earlier UCF codec, this supports rectangular grids and 4k+1 frames.
    """
    def __init__(self, width=64):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError('width must be a positive multiple of eight')
        self.features = nn.Sequential(
            nn.Conv3d(16, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1), nn.SiLU())
        self.attributes = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                        nn.Conv2d(width, 9, 1))
        with torch.no_grad():
            self.attributes[-1].bias.zero_()
            self.attributes[-1].bias[4] = 1.0

    def forward(self, latent, frames=None, grid=None):
        if latent.ndim != 5 or latent.shape[1] != 16:
            raise ValueError('Expected normalized Wan latent B,16,T,H,W')
        b, _, t, h, w = latent.shape
        frames = (t - 1)*4 + 1 if frames is None else frames
        grid = (h*2, w*2) if grid is None else grid
        if frames < 1 or len(grid) != 2 or min(grid) < 1:
            raise ValueError('Invalid target time/grid size')
        features = F.interpolate(self.features(latent), size=(frames, *grid),
                                 mode='trilinear', align_corners=False)
        features = features.permute(0, 2, 1, 3, 4).reshape(b*frames, -1, *grid)
        raw = self.attributes(features)
        return raw.reshape(b, frames, 9, *grid).permute(0, 2, 1, 3, 4)


def gaussian_state(raw, height, width):
    """Raw B,9,Gh,Gw -> explicit centres, positive scales, orthonormal axes, RGB."""
    if raw.ndim != 4 or raw.shape[1] != 9 or min(height, width) < 1:
        raise ValueError('Expected B,9,Gh,Gw and positive image dimensions')
    b, _, gh, gw = raw.shape
    value = raw.flatten(2).transpose(1, 2)
    cells = raw.new_tensor([width/gw, height/gh])
    yy, xx = torch.meshgrid(torch.arange(gh, device=raw.device, dtype=raw.dtype),
                            torch.arange(gw, device=raw.device, dtype=raw.dtype), indexing='ij')
    base = (torch.stack((xx, yy), -1).reshape(1, -1, 2) + .5)*cells - .5
    centre = base + .45*cells*value[..., :2].tanh()
    scale = cells*(.35 + .65*value[..., 2:4].sigmoid())
    direction = value[..., 4:6]
    norm = direction.norm(dim=-1, keepdim=True)
    canonical = torch.zeros_like(direction)
    canonical[..., 0] = 1
    u = torch.where(norm > 1e-6, direction/norm.clamp_min(1e-6), canonical)
    v = torch.stack((-u[..., 1], u[..., 0]), -1)
    axes = torch.stack((u, v), -1)
    return dict(centre=centre, scale=scale, axes=axes, colour=value[..., 6:9].sigmoid())


def splat_frame(raw, height, width, radius=None):
    """Differentiable normalized 2D anisotropic splats with finite local support.

    Not standard depth-sorted 3DGS. Support selection is discrete; weights,
    centres, scales, orientation and colours have piecewise derivatives.
    """
    state = gaussian_state(raw, height, width)
    b, _, gh, gw = raw.shape
    radius = math.ceil(2*max(width/gw, height/gh)) if radius is None else radius
    if radius < 1:
        raise ValueError('radius must be positive')
    centre, scale, axes = state['centre'], state['scale'], state['axes']
    yy, xx = torch.meshgrid(torch.arange(-radius, radius+1, device=raw.device),
                            torch.arange(-radius, radius+1, device=raw.device), indexing='ij')
    offsets = torch.stack((xx.flatten(), yy.flatten()), -1)
    pixels = centre.floor().long()[:, :, None, :] + offsets[None, None]
    delta = pixels.to(raw.dtype) - centre[:, :, None]
    local = torch.einsum('bnki,bnij->bnkj', delta, axes)
    weights = torch.exp(-.5*(local/scale[:, :, None]).square().sum(-1))
    valid = (pixels[..., 0] >= 0) & (pixels[..., 0] < width) & (pixels[..., 1] >= 0) & (pixels[..., 1] < height)
    weights = weights*valid
    indices = (pixels[..., 1].clamp(0, height-1)*width + pixels[..., 0].clamp(0, width-1)).reshape(b, -1)
    weights_flat = weights.reshape(b, -1)
    denominator = raw.new_zeros(b, height*width).scatter_add(1, indices, weights_flat)
    colour = (state['colour'][:, :, None, :]*weights[..., None]).reshape(b, -1, 3)
    numerator = raw.new_zeros(b, height*width, 3).scatter_add(1, indices[..., None].expand(-1, -1, 3), colour)
    image = (numerator/denominator.clamp_min(1e-8)[..., None]).reshape(b, height, width, 3)
    return image.permute(0, 3, 1, 2), denominator.reshape(b, height, width)


def render_gaussian_video(raw, height, width, radius=None):
    if raw.ndim != 5 or raw.shape[1] != 9:
        raise ValueError('Expected B,9,T,Gh,Gw')
    # Bounded per-frame splat intermediates at inference, not horizon-free memory.
    return torch.stack([splat_frame(raw[:, :, t], height, width, radius)[0]
                        for t in range(raw.shape[2])], dim=1)
