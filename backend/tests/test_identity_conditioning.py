import torch

from real_video.identity_conditioning import identity_region_flow_loss, foreground_condition_sensitivity


def rejects(call):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError('Invalid identity evidence accepted')


def tensors():
    prediction = torch.zeros(1, 2, 3, 2, 2)
    target = torch.zeros_like(prediction)
    mask = torch.zeros(1, 1, 3, 2, 2)
    mask[:, :, :, 0, 0] = 1
    return prediction, target, mask


def test_small_object_error_is_not_diluted_by_background():
    prediction, target, mask = tensors()
    prediction[:, :, 1:, 0, 0] = 2
    result = identity_region_flow_loss(prediction, target, mask, anchor_weight=0)
    assert float(result['loss']) == 2  # .5*foreground4 + .5*background0
    assert float((prediction-target)[:, :, 1:].square().mean()) == 1
    assert result['identity_proven'] is False


def test_anchor_supervision_is_explicit_not_silently_dropped():
    prediction, target, mask = tensors()
    prediction[:, :, 0] = 2
    excluded = identity_region_flow_loss(prediction, target, mask, anchor_weight=0)
    included = identity_region_flow_loss(prediction, target, mask, anchor_weight=.25)
    assert float(excluded['loss']) == 0
    assert float(included['loss']) == 1
    assert float(included['anchor_loss']) == 4


def test_gradients_reach_predictions_but_not_targets_or_masks():
    prediction, target, mask = tensors()
    prediction.add_(1).requires_grad_(); target.requires_grad_(); mask.requires_grad_()
    identity_region_flow_loss(prediction, target, mask)['loss'].backward()
    assert prediction.grad.abs().sum() > 0
    assert target.grad is None and mask.grad is None


def test_absent_region_does_not_halve_supported_region_loss():
    prediction, target, mask = tensors()
    prediction.add_(2)
    for value in (0., 1.):
        mask.fill_(value)
        result = identity_region_flow_loss(prediction, target, mask)
        assert float(result['loss']) == 4
        assert bool(result['foreground_supported'].all()) == bool(value)


def test_memory_sensitivity_excludes_even_enormous_anchor_changes():
    correct, target, mask = tensors()
    wrong = correct.clone()
    wrong[:, :, 0] = 100
    result = foreground_condition_sensitivity(correct, wrong, target, mask)
    assert result['wrong_minus_right'] == 0 and result['anchor_excluded'] is True
    wrong[:, :, 1:, 0, 0] = 2
    result = foreground_condition_sensitivity(correct, wrong, target, mask)
    assert result['wrong_minus_right'] == 4
    assert result['evaluated_future_object_slots'] == 2


def test_correct_memory_can_be_worse_without_relabeling_success():
    correct, target, mask = tensors()
    wrong = correct.clone(); correct[:, :, 1:, 0, 0] = 1
    result = foreground_condition_sensitivity(correct, wrong, target, mask)
    assert result['wrong_minus_right'] == -1
    assert result['identity_proven'] is False


def test_missing_object_evidence_cannot_pass_sensitivity():
    prediction, target, mask = tensors(); mask[:, :, 1:] = 0
    rejects(lambda: foreground_condition_sensitivity(prediction, prediction, target, mask))


def test_input_validation():
    prediction, target, mask = tensors()
    rejects(lambda: identity_region_flow_loss(prediction, target, mask, anchor_weight=float('nan')))
    rejects(lambda: identity_region_flow_loss(prediction, target, mask, anchor_weight=True))
    rejects(lambda: identity_region_flow_loss(prediction[:, :, :1], target[:, :, :1], mask[:, :, :1]))
    rejects(lambda: identity_region_flow_loss(prediction, target, mask+2))
    prediction[0, 0, 0, 0, 0] = float('nan')
    rejects(lambda: identity_region_flow_loss(prediction, target, mask))


if __name__ == '__main__':
    tests = [f for n, f in list(globals().items()) if n.startswith('test_') and callable(f)]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print(f'{len(tests)} identity-conditioning tests passed')
