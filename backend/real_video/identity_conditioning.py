"""Identity-sensitive supervision diagnostics, not an identity recognizer.

The reader_v2 pilot used a generic animal prompt, denoised frame zero freely,
excluded it from training loss, and averaged errors across object/background.
Its lower average latent loss did not retain the source animal or scene.

Wan 2503.20314v1 section 5.1 uses pretrained first-frame image conditioning;
Wan-Move 2512.08765v1 sections 3.2--3.3 use native image features and controlled
motion. Those sources motivate a genuine I2V baseline, not inference-time
clamping of an untrained T2V model. The balanced losses below are our simple
engineering diagnostic, not equations claimed by either paper.

These utilities do not attach any model, change a scheduler, alter a video,
train a geometry writer, or prove semantic identity. Masks are supervision
only; never put held-out future RGB/latents/masks into inference appearance.
"""
import math

import torch


def _validate(prediction, target, object_mask):
    for name, value in (('prediction', prediction), ('target', target)):
        if (not isinstance(value, torch.Tensor) or value.ndim != 5
                or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
            raise ValueError(f'{name} must be finite floating B,C,T,H,W')
    if prediction.shape != target.shape or prediction.device != target.device:
        raise ValueError('Matching prediction/target shape and device required')
    b, _, t, h, w = prediction.shape
    if any(n < 1 for n in prediction.shape) or t < 2:
        raise ValueError('Nonempty clip with anchor and at least one future slot required')
    if (not isinstance(object_mask, torch.Tensor) or tuple(object_mask.shape) != (b, 1, t, h, w)
            or object_mask.device != prediction.device
            or (object_mask.dtype != torch.bool and not object_mask.is_floating_point())
            or not bool(torch.isfinite(object_mask).all())):
        raise ValueError('Explicit finite B,1,T,H,W object mask on the model device required')
    if bool(((object_mask < 0) | (object_mask > 1)).any()):
        raise ValueError('Object mask must lie in [0,1]')
    # Train only the prediction: the target and support may not move to make
    # their own supervision easier. Soft masks are accepted as fixed weights.
    return prediction.float(), target.detach().float(), object_mask.detach().float()


def _region_errors(squared_error, mask):
    channels = squared_error.shape[1]
    mass = mask.sum(dim=(1, 3, 4))
    numerator = (squared_error*mask).sum(dim=(1, 3, 4))
    supported = mass > 0
    safe_mass = torch.where(supported, mass, torch.ones_like(mass))
    values = numerator/(channels*safe_mass)
    return values, supported


def identity_region_flow_loss(prediction, target_velocity, object_mask, *, anchor_weight=.25):
    """Area-balanced object/background flow loss with explicit anchor weight.

At every (batch,time), average each supported region independently and then
average the supported region losses. A small foreground therefore does not
vanish under a large background. Average all future times equally, and blend
that future loss with the first-slot loss using ``anchor_weight``.

For a model that natively keeps the anchor clean and does not predict its
velocity, set ``anchor_weight=0``; do NOT train an artificial anchor target.
For the existing Wan flow convention target_velocity is ``noise - clean``.
This criterion is untrained infrastructure, not evidence it fixes identity.
"""
    if isinstance(anchor_weight, bool) or not isinstance(anchor_weight, (int, float)) or not math.isfinite(anchor_weight) or not 0 <= anchor_weight <= 1:
        raise ValueError('Anchor weight must be finite and in [0,1]')
    prediction, target, mask = _validate(prediction, target_velocity, object_mask)
    error = (prediction-target).square()
    foreground, fg_valid = _region_errors(error, mask)
    background, bg_valid = _region_errors(error, 1-mask)
    per_time = (foreground+background)/(fg_valid.float()+bg_valid.float())
    anchor = per_time[:, 0].mean()
    future = per_time[:, 1:].mean()
    return dict(loss=anchor_weight*anchor+(1-anchor_weight)*future,
                anchor_loss=anchor, future_loss=future,
                foreground_per_time=foreground, background_per_time=background,
                foreground_supported=fg_valid, background_supported=bg_valid,
                identity_proven=False)


@torch.no_grad()
def foreground_condition_sensitivity(matched_prediction, mismatched_prediction,
                                     target, object_mask):
    """Compare correct/wrong-memory predictions over supported FUTURE objects.

Use identical noise, prompt, motion, model weights and target for the pair;
only the memory condition may change. The function cannot verify this causal
experimental setup, so the caller must record those hashes/seeds. Targets can
be clean latents or flow velocities, but all three tensors must use the SAME
domain. A positive wrong-minus-right gap means lower foreground error for the
matched condition; this is not a semantic identity or video-quality pass.

The anchor slot is excluded: a natively clamped first frame cannot manufacture
an identity success. No foreground evidence fails closed instead of scoring0.
"""
    matched, target, mask = _validate(matched_prediction, target, object_mask)
    mismatched, _, _ = _validate(mismatched_prediction, target, mask)
    good, supported = _region_errors((matched-target).square(), mask)
    bad, _ = _region_errors((mismatched-target).square(), mask)
    supported = supported[:, 1:]
    if not bool(supported.any()):
        raise ValueError('No future foreground evidence for memory sensitivity')
    right = good[:, 1:][supported].mean()
    wrong = bad[:, 1:][supported].mean()
    return dict(matched_foreground_mse=float(right), mismatched_foreground_mse=float(wrong),
                wrong_minus_right=float(wrong-right),
                evaluated_future_object_slots=int(supported.sum()),
                anchor_excluded=True, identity_proven=False)
