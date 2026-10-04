"""Reference-view covariance lift. Geometry is inferred, not identified from 2D."""
import torch


def lift(centre, covariance, depth, focal, principal, thickness=.01, blur=.16, normals=None):
    """Match gaussian3d.project's local covariance, including its blur floor.

    Input centres use its half-integer pixel convention. Returned coordinates
    use world x-right/y-up/z-back, with reference camera at zero looking -Z.
    This is a local footprint identity, NOT a rendered-image identity.
    """
    n = len(centre)
    if centre.shape != (n, 2) or covariance.shape != (n, 2, 2) or depth.shape != (n,):
        raise ValueError('Invalid shapes')
    if focal <= 0 or thickness <= 0 or blur < 0:
        raise ValueError('Invalid camera/thickness/blur')
    if not all(torch.isfinite(x).all() for x in (centre, covariance, depth)) or (depth <= 0).any():
        raise ValueError('Require finite inputs and positive depth')
    tolerance = 32*torch.finfo(covariance.dtype).eps*covariance.abs().amax().clamp_min(1)
    if ((covariance-covariance.transpose(-1, -2)).abs() > tolerance).any():
        raise ValueError('Covariance must be symmetric')
    covariance = (covariance+covariance.transpose(-1, -2))*.5
    inner = covariance - torch.eye(2, device=centre.device, dtype=centre.dtype)*blur
    if (torch.linalg.eigvalsh(inner) <= 0).any():
        raise ValueError('Screen covariance must exceed renderer blur floor')
    xy = (centre-centre.new_tensor(principal))/focal
    camera_position = torch.cat((xy, torch.ones_like(depth[:, None])), -1)*depth[:, None]
    j = centre.new_zeros(n, 2, 3)
    j[:, 0, 0] = focal/depth
    j[:, 1, 1] = focal/depth
    j[:, :, 2] = -focal*xy/depth[:, None]
    b = j.transpose(-1, -2) @ torch.linalg.inv(j @ j.transpose(-1, -2))
    ray = torch.nn.functional.normalize(camera_position, dim=-1)
    if normals is not None:
        if normals.shape != (n, 3) or not torch.isfinite(normals).all():
            raise ValueError('Require finite camera-space normals')
        normals = torch.nn.functional.normalize(normals, dim=-1)
        cosine = (normals*ray).sum(-1)
        if (cosine.abs() < .05).any():
            raise ValueError('Grazing tangent planes have unstable inverse projection')
        # Move the right inverse along J's nullspace to lie in the tangent plane.
        b = b-ray[:, :, None]*(normals[:, None, :]@b)/cosine[:, None, None]
    cov = b @ inner @ b.transpose(-1, -2) + thickness**2*ray[:, :, None]*ray[:, None, :]
    flip = torch.diag(centre.new_tensor([1., -1., -1.]))
    return camera_position @ flip, flip @ cov @ flip
