"""Inference-only Gaussian renderer with bounded, covariance-sized footprints.

Uses the existing camera and +0.16 pixel covariance convention. The default
guards off-axis covariance Jacobians as in standard 3DGS and removes the
fixed-radius square truncation. This is NOT
Mip-Splatting training, a geometry/texture repair, or a learned generator.
Memory is O(N + H*W + fragment_budget), not O(N * largest_footprint_area).
Center-depth order and first-order perspective projection remain approximations.
"""
import math
import time
import torch

from .gaussian3d import project


def project_guarded(position, covariance, eye, target, height, width, fov=42., principal=None,
                    slope_guard=1.3):
    """Limit covariance-Jacobian slopes, never move projected means or geometry.

    A first-order perspective Gaussian is invalid arbitrarily far off-axis near
    z=0. The 3DGS reference computeCov2D similarly clamps x/z and y/z at 1.3
    times each viewport tangent before forming its covariance Jacobian.
    None reproduces the unguarded legacy projection for failure diagnostics.
    This is an approximation, not exact perspective integration of an ellipsoid.
    """
    mean, screen, depth, camera, focal = project(position, covariance, eye, target, height, width, fov, principal)
    if slope_guard is None:
        return mean, screen, depth, camera, focal
    if slope_guard < 1 or not math.isfinite(slope_guard):
        raise ValueError('slope_guard must be finite and at least 1, or None')
    p = (position-eye) @ camera.T
    safe = depth.clamp_min(.01)
    cx, cy = (width/2, height/2) if principal is None else principal
    low = position.new_tensor([-cx/focal, -cy/focal])*slope_guard
    high = position.new_tensor([(width-cx)/focal, (height-cy)/focal])*slope_guard
    slope = torch.minimum(torch.maximum(p[:, :2]/safe[:, None], low), high)
    jac = position.new_zeros(len(position), 2, 3)
    jac[:, 0, 0] = focal/safe; jac[:, 1, 1] = focal/safe
    jac[:, :, 2] = -focal*slope/safe[:, None]
    camera_cov = camera @ covariance @ camera.T
    screen = jac @ camera_cov @ jac.transpose(-1, -2)+torch.eye(2, device=p.device, dtype=p.dtype)[None]*.16
    return mean, screen, depth, camera, focal


def footprint_bounds(mean, covariance, opacity, height, width, alpha_threshold=1e-4,
                     fixed_radius=None):
    """Inclusive pixel-index boxes enclosing all samples above alpha_threshold.

    For delta.T inv(Sigma) delta <= q, max |delta_axis| is
    sqrt(q * Sigma_axis,axis), including tilted ellipses. Bounds use pixel
    centers (integer + 0.5). A fixed_radius is diagnostic compatibility only.
    """
    if not (0 < alpha_threshold < .995):
        raise ValueError('alpha_threshold must be between 0 and .995')
    if height < 1 or width < 1:
        raise ValueError('Positive image dimensions required')
    if fixed_radius is not None:
        if not isinstance(fixed_radius, int) or fixed_radius < 0:
            raise ValueError('fixed_radius must be a nonnegative integer')
        center = mean.floor().long()
        lower, upper = center-fixed_radius, center+fixed_radius
    else:
        q = 2 * torch.log((opacity/alpha_threshold).clamp_min(1))
        extent = torch.sqrt(q[:, None] * covariance.diagonal(dim1=-2, dim2=-1))
        lower = torch.ceil(mean-extent-.5).long()
        upper = torch.floor(mean+extent-.5).long()
    lower = torch.maximum(lower, lower.new_tensor([0, 0]))
    upper = torch.minimum(upper, upper.new_tensor([width-1, height-1]))
    size = (upper-lower+1).clamp_min(0)
    size = torch.where((opacity > alpha_threshold)[:, None], size, 0)
    return lower, upper, size


def _background(eye, target, camera, focal, height, width, ground, principal):
    yy, xx = torch.meshgrid(torch.arange(height, device=eye.device, dtype=eye.dtype)+.5,
                           torch.arange(width, device=eye.device, dtype=eye.dtype)+.5,
                           indexing='ij')
    cx, cy = (width/2, height/2) if principal is None else principal
    rays = torch.stack(((xx-cx)/focal, (yy-cy)/focal, torch.ones_like(xx)), -1) @ camera
    denominator = torch.where(rays[..., 1].abs() > 1e-7, rays[..., 1], 1e-7)
    floor_depth = -eye[1] / denominator
    floor_valid = (floor_depth > 0) & (rays[..., 1] < -1e-7) & ground
    hit = eye + floor_depth[..., None]*rays
    checker = ((torch.floor(hit[..., 0]*4)+torch.floor(hit[..., 2]*4)).long() % 2).to(eye.dtype)
    background = eye.new_tensor([.15, .19, .24]).expand(height, width, 3).clone()
    floor_colour = (.28+.12*checker)[..., None] * eye.new_tensor([.9, 1., .85])
    return torch.where(floor_valid[..., None], floor_colour, background), floor_valid, floor_depth


@torch.inference_mode()
def render(position, covariance, colour, opacity, eye, target, height=480, width=640,
           fov=42., ground=True, principal=None, fragment_budget=500_000,
           alpha_threshold=1e-4, fixed_radius=None, max_candidate_pixels=300_000_000,
           return_stats=False, wall_seconds=None, slope_guard=1.3):
    """Composite complete thresholded ellipses in bounded depth-ordered chunks.

    Splitting a single splat between chunks is safe: each split contains disjoint
    pixels, and chunks follow Gaussian center depth. Inputs are never modified.
    Bounds and budget are implementation choices, not a Mip-Splatting reproduction.
    """
    n = len(position)
    begin = time.perf_counter()
    if wall_seconds is not None and wall_seconds <= 0:
        raise ValueError('wall_seconds must be positive')
    if position.shape != (n, 3) or covariance.shape != (n, 3, 3) or colour.shape != (n, 3) or opacity.shape != (n,):
        raise ValueError('Gaussian tensor shapes do not match')
    if not isinstance(fragment_budget, int) or fragment_budget < 1:
        raise ValueError('fragment_budget must be a positive integer')
    if not (0 < fov < 180) or not math.isfinite(fov):
        raise ValueError('Invalid field of view')
    for tensor in (position, covariance, colour, opacity, eye, target):
        if tensor.device != position.device or not torch.isfinite(tensor).all():
            raise ValueError('Inputs must be finite and share a device')
    if torch.any(opacity < 0) or torch.any(opacity > 1):
        raise ValueError('Opacity must lie in [0,1]')
    mean, screen, depth, camera, focal = project_guarded(position, covariance, eye, target, height, width, fov, principal, slope_guard)
    det = screen[:, 0, 0]*screen[:, 1, 1]-screen[:, 0, 1].square()
    if torch.any(screen.diagonal(dim1=-2, dim2=-1) <= 0) or torch.any(det <= 0):
        raise ValueError('Projected covariance must be positive definite')
    background, floor_valid, floor_depth = _background(eye, target, camera, focal, height, width, ground, principal)
    # Stable ties make chunk-size invariance deterministic.
    order = depth.argsort(stable=True)
    mean, screen, depth = mean[order], screen[order], depth[order]
    colour, opacity = colour[order], opacity[order]
    lower, _, size = footprint_bounds(mean, screen, opacity, height, width, alpha_threshold, fixed_radius)
    count = size[:, 0]*size[:, 1]
    count = torch.where(depth > .02, count, 0)
    cumulative = count.cumsum(0)
    total = int(cumulative[-1]) if n else 0
    if total > max_candidate_pixels:
        raise ValueError(f'Projected work {total} exceeds explicit candidate budget {max_candidate_pixels}; no hidden radius clamp applied')
    starts = cumulative-count
    image = position.new_zeros(height*width, 3)
    transmittance = torch.ones(height*width, dtype=torch.float64, device=position.device)
    chunks, accepted, largest_chunk = 0, 0, 0
    for offset in range(0, total, fragment_budget):
        if wall_seconds is not None and time.perf_counter()-begin > wall_seconds:
            raise TimeoutError('Renderer wall budget expired between bounded fragment batches; no partial image promoted')
        stream = torch.arange(offset, min(offset+fragment_budget, total), device=position.device)
        # This ragged stream never allocates all Gaussian/pixel pairs at once.
        ids = torch.searchsorted(cumulative, stream, right=True)
        local = stream-starts[ids]
        x = lower[ids, 0]+local % size[ids, 0]
        y = lower[ids, 1]+torch.div(local, size[ids, 0], rounding_mode='floor')
        pix = y*width+x
        delta = torch.stack((x, y), -1).to(mean.dtype)+.5-mean[ids]
        c = screen[ids]
        a, b, d = c[:, 0, 0], c[:, 0, 1], c[:, 1, 1]
        power = (d*delta[:, 0].square()-2*b*delta[:, 0]*delta[:, 1]+a*delta[:, 1].square())/(a*d-b*b)
        alpha = (opacity[ids]*torch.exp(-.5*power)).clamp(0, .995)
        valid = alpha > alpha_threshold
        if ground:
            valid &= (~floor_valid.flatten()[pix]) | (depth[ids] <= floor_depth.flatten()[pix])
        chunks += 1
        largest_chunk = max(largest_chunk, len(stream))
        pix, ids, alpha = pix[valid], ids[valid], alpha[valid]
        accepted += len(pix)
        if not len(pix):
            continue
        rank = (pix*(n+1)+ids).argsort()
        pix, ids, alpha = pix[rank], ids[rank], alpha[rank]
        logs = torch.log1p(-alpha.double())
        prefix = torch.cat((logs.new_zeros(1), logs.cumsum(0)))
        first = torch.ones_like(pix, dtype=torch.bool)
        first[1:] = pix[1:] != pix[:-1]
        bases = logs.new_zeros(height*width)
        bases[pix[first]] = prefix[:-1][first]
        local_trans = torch.exp(prefix[:-1]-bases[pix])
        weights = (alpha*local_trans*transmittance[pix]).to(position.dtype)
        image.index_add_(0, pix, weights[:, None]*colour[ids])
        # Keep the exact product in log space rather than subtracting rounded A.
        log_sum = logs.new_zeros(height*width).index_add_(0, pix, logs)
        transmittance *= log_sum.exp()
    image = image.reshape(height, width, 3) + transmittance.to(position.dtype).reshape(height, width, 1)*background
    alpha = (1-transmittance).to(position.dtype).reshape(height, width)
    result = (image.clamp(0, 1), alpha)
    if return_stats:
        result += (dict(input_gaussians=n, candidate_pixels=total, accepted_fragments=accepted,
                        chunks=chunks, fragment_budget=fragment_budget, largest_candidate_chunk=largest_chunk,
                        alpha_threshold=alpha_threshold, fixed_radius=fixed_radius,
                        wall_seconds=wall_seconds,
                        covariance_jacobian_slope_guard=slope_guard,
                        bounded_memory_scope='O(input Gaussian count + framebuffer pixels + fragment_budget); inference only',
                        source_state_modified=False, geometry_or_texture_generated=False),)
    return result
