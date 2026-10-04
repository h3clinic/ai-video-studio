"""Observed-mask Gaussian replacement controls, not semantic/physical inference.

Uses the existing source camera convention u=cx-f*x/z, v=cy-f*y/z.
GaussianEditor arXiv:2311.14521v2 Sec.4.3 describes least-squares alignment of
estimated depth with scene depth. This bounded bbox/front-depth fit is our
engineering approximation, NOT that least-squares algorithm or a quoted paper
equation. Covariance transport is s^2*Sigma for this similarity.
"""
import hashlib
import math
import numpy as np


def boolean_mask(value, shape=None):
    value = np.asarray(value)
    if (value.dtype != np.bool_ or value.ndim != 2 or value.size > 1920*1080
            or shape is not None and value.shape != shape):
        raise ValueError('Bounded, explicit boolean HxW mask required')
    return value


def mask_bbox(mask):
    mask = boolean_mask(mask)
    yy, xx = np.nonzero(mask)
    if len(xx) < 4 or np.ptp(xx) < 1 or np.ptp(yy) < 1:
        raise ValueError('Insufficient observed mask extent')
    # Pixel centers, matching the source depth lifting convention.
    return np.array([xx.min()+.5, yy.min()+.5, xx.max()+.5, yy.max()+.5], dtype=np.float64)


def mask_iou(first, second):
    first = boolean_mask(first)
    second = boolean_mask(second, first.shape)
    union = np.count_nonzero(first | second)
    return 0. if union == 0 else float(np.count_nonzero(first & second)/union)


def associate_observations(previous, current, *, minimum_iou=.35, margin=.12):
    """Mutual-unique IoU association, rejecting ambiguity; no ID invention.

    Previous entries are {part_id,label,mask}, optionally warped by an external
    tracker into this frame. Current entries are {label,mask}. Returns current
    index -> prior part ID. Neither confidence nor association is ground truth.
    Missing/occluded observations are not extrapolated or silently recoloured.
    """
    if len(previous) > 32 or len(current) > 32 or not 0 < minimum_iou <= 1 or not 0 < margin < 1:
        raise ValueError('Bounded instances and conservative thresholds required')
    if len({p['part_id'] for p in previous}) != len(previous):
        raise ValueError('Previous part IDs must be unique')
    scores = np.zeros((len(previous), len(current)), dtype=np.float64)
    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            if old['label'] == new['label']:
                scores[i, j] = mask_iou(old['mask'], new['mask'])
    matched = {}
    for i in range(len(previous)):
        if not len(current): break
        j = int(scores[i].argmax())
        best = scores[i, j]
        row_other = np.delete(scores[i], j).max(initial=0.)
        column_other = np.delete(scores[:, j], i).max(initial=0.)
        if best >= minimum_iou and best-row_other >= margin and best-column_other >= margin:
            matched[j] = previous[i]['part_id']
    return matched


def partition_replacement_masks(instance_masks, ready_ids, protected_mask):
    """Delete only parts with ready assets; preserve every unsupported part.

    Overlap with another instance or protected object is withheld, not assigned
    by precedence. This avoids erasing peel/held slices because one body exists.
    """
    protected = boolean_mask(protected_mask)
    if not set(ready_ids) <= set(instance_masks):
        raise ValueError('Replacement references an absent observed part')
    masks = {key: boolean_mask(value, protected.shape) for key, value in instance_masks.items()}
    coverage = np.zeros(protected.shape, dtype=np.uint16)
    if len(masks) > 32: raise ValueError('Too many part instances')
    for mask in masks.values(): coverage += mask
    ambiguous = coverage > 1
    eligible = {key: mask & ~protected & ~ambiguous for key, mask in masks.items() if key in ready_ids}
    remove = np.zeros_like(protected)
    for mask in eligible.values(): remove |= mask
    observed = coverage > 0
    counts={key:dict(observed_pixels=int(mask.sum()), protected_overlap_pixels=int((mask&protected).sum()),
                    ambiguous_overlap_pixels=int((mask&ambiguous).sum()),
                    eligible_pixels=int(eligible[key].sum()) if key in eligible else 0) for key,mask in masks.items()}
    return dict(eligible=eligible, remove=remove, preserved=observed & ~remove,
                ambiguous=ambiguous, deferred_ids=tuple(key for key in masks if key not in ready_ids),
                instance_pixel_counts=counts)


def evaluate_replacement_coverage(alpha, target_mask, protected_mask, *,
                                  opaque_threshold=.95, minimum_coverage=.9,
                                  maximum_outside_alpha_fraction=.05,
                                  maximum_protected_alpha=.05):
    """Reject unsafe projected support before deleting any observed Gaussians.

    Alpha must come from the SAME Gaussian renderer/camera as the final frame.
    This is a conservative geometric diagnostic, not a quality judgement or a
    proof of visible occlusion/contact. Only sufficiently opaque target pixels
    may be deleted, even when the 90% diagnostic threshold passes. Uncovered
    source samples are retained, so this cannot silently create holes or claim
    complete recolouring; an orange boundary can remain and must be reviewed.
    """
    target=boolean_mask(target_mask);protected=boolean_mask(protected_mask,target.shape)
    alpha=np.asarray(alpha)
    settings=(opaque_threshold,minimum_coverage,maximum_outside_alpha_fraction,maximum_protected_alpha)
    if (alpha.shape!=target.shape or alpha.dtype.kind!='f' or not np.isfinite(alpha).all()
            or (alpha < -1e-6).any() or (alpha > 1+1e-6).any() or not target.any()
            or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<x<=1 for x in settings)):
        raise ValueError('Valid renderer alpha, observed target and explicit unit thresholds required')
    alpha=np.clip(alpha,0,1)
    removable=target & ~protected & (alpha>=opaque_threshold)
    coverage=float(removable.sum()/target.sum())
    total=float(alpha.sum(dtype=np.float64))
    outside=float(alpha[~target].sum(dtype=np.float64)/total) if total>0 else 0.
    protected_peak=float(alpha[protected].max(initial=0.))
    reasons=[]
    if coverage<minimum_coverage:reasons.append('Replacement does not opaquely cover observed body')
    if outside>maximum_outside_alpha_fraction:reasons.append('Replacement spills outside observed body')
    if protected_peak>maximum_protected_alpha:reasons.append('Replacement overlaps protected or unsupported part')
    accepted=not reasons
    return dict(accepted_for_insertion=accepted,rejection_reasons=reasons,
        opaque_target_coverage=coverage,outside_alpha_fraction=outside,maximum_protected_alpha=protected_peak,
        thresholds=dict(opaque_threshold=opaque_threshold,minimum_coverage=minimum_coverage,
            maximum_outside_alpha_fraction=maximum_outside_alpha_fraction,maximum_protected_alpha=maximum_protected_alpha),
        retained_target_pixels=int(target.sum()-removable.sum()),
        remove_mask=removable if accepted else np.zeros_like(target),
        quality_accepted=False,protected_rendered_pixels_proven_unchanged=False)


def fit_observed_placement(canonical_position, mask, depth, *, focal, principal,
                           maximum_depth_spread=.35, iterations=20):
    """Fit a translated/uniformly-scaled canonical asset to observed support.

    Matches bounding width/height conservatively and the mask's bottom edge.
    Median source foreground depth aligns to the near canonical surface, not
    its center (the previous worker placed its center on an observed surface).
    Rotation is retained. This does NOT recover hidden support/contact, a rig,
    a silhouette fit, absolute metric depth or motion of occluded parts.
    """
    points = np.asarray(canonical_position, dtype=np.float64)
    mask = boolean_mask(mask)
    depth = np.asarray(depth)
    if (points.ndim != 2 or points.shape[1] != 3 or not 4 <= len(points) <= 100_000
            or not np.isfinite(points).all() or np.abs(points).max() > 10_000):
        raise ValueError('Bounded finite canonical XYZ points required')
    if depth.shape != mask.shape or depth.dtype.kind != 'f':
        raise ValueError('Matching floating depth required')
    if (type(focal) not in (int, float) or not math.isfinite(focal) or focal <= 0
            or len(principal) != 2 or not all(math.isfinite(x) for x in principal)
            or not 1 <= iterations <= 40 or not 0 < maximum_depth_spread <= 1):
        raise ValueError('Finite camera and bounded fitting settings required')
    # Validate observed support BEFORE any reduction of depth samples. Empty
    # protected/clashing masks otherwise pass np.isfinite(...).all() vacuously,
    # then median warns and quantile raises IndexError instead of safe deferral.
    target = mask_bbox(mask)
    samples = depth[mask]
    if not np.isfinite(samples).all() or (samples <= .02).any():
        raise ValueError('Observed mask contains invalid depth; do not guess')
    foreground_depth = float(np.median(samples))
    if float(np.quantile(samples,.9)-np.quantile(samples,.1))/foreground_depth > maximum_depth_spread:
        raise ValueError('Mask crosses inconsistent depth layers')
    target_extent = target[2:]-target[:2]
    target_center = .5*(target[:2]+target[2:])
    origin = .5*(points.min(0)+points.max(0))
    local = points-origin
    extent = np.ptp(local, axis=0)
    if (extent[:2] <= 1e-8).any(): raise ValueError('Canonical asset has degenerate projected extent')
    scale = float(min(target_extent/extent[:2])*foreground_depth/focal)
    def placement(s, x=None, y=None):
        # Near geometry maps to observed foreground surface (approximate).
        z = foreground_depth-s*local[:,2].min()
        return np.array([-(target_center[0]-principal[0])*z/focal if x is None else x,
                         -(target_center[1]-principal[1])*z/focal if y is None else y,z])
    center = placement(scale)
    def projected(s, c):
        xyz = local*s+c
        if not np.isfinite(xyz).all() or (xyz[:,2] <= .02).any(): raise ValueError('Placement crosses near plane')
        uv = np.array(principal)-focal*xyz[:,:2]/xyz[:,2,None]
        return np.r_[uv.min(0), uv.max(0)]
    for _ in range(iterations):
        box = projected(scale, center)
        extent_now = box[2:]-box[:2]
        ratio = float(min(target_extent/extent_now))
        scale *= float(np.clip(ratio,.5,2.))
        center = placement(scale, center[0], center[1])
        box = projected(scale, center)
        # Use observed bottom support rather than an image-centered floating asset.
        delta = np.array([target_center[0]-.5*(box[0]+box[2]), target[3]-box[3]])
        center[:2] -= delta*center[2]/focal
    box = projected(scale, center)
    if not math.isfinite(scale) or not 1e-8 <= scale <= 10_000:
        raise ValueError('Invalid fitted scale')
    center_error = abs(.5*(box[0]+box[2])-target_center[0])
    bottom_error = abs(box[3]-target[3])
    if max(center_error,bottom_error) > .75:
        raise ValueError('Observed-support fit did not converge')
    return dict(scale=scale, translation=(center-scale*origin).tolist(),
                target_bbox_pixels=target.tolist(), fitted_bbox_pixels=box.tolist(),
                bottom_error_pixels=float(bottom_error), horizontal_center_error_pixels=float(center_error),
                mask_sha256=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
                median_foreground_depth=foreground_depth, mask_pixels=int(mask.sum()),
                physical_contact_verified=False, depth_metric=False,
                covariance_rule='Sigma_world = scale^2 * Sigma_canonical',
                source='observed mask bbox and foreground relative depth; not generated motion')
