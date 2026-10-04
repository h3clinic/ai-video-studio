import torch
from torch import nn
from real_video.repeat_temporal_dog import configure_case
from real_video.wan_cat_memory import LowRankLinear
from real_video.wan_temporal_control import TemporalSpatialControl


def test_case_switches_separate_weight_and_memory_effects():
    model = nn.Sequential(LowRankLinear(nn.Linear(2, 2), 1))
    control = TemporalSpatialControl(2, channels=5, blocks=(0,))
    for name, lora, memory in [('A_gaussian_memory', True, True),
                              ('B_original_wan', False, False),
                              ('C_adapted_without_memory', True, False)]:
        result = configure_case(model, control, name)
        assert model[0].enabled is lora and control.enabled is memory
        assert result == dict(lora_enabled=lora, gaussian_control_enabled=memory)


def test_baseline_is_exact_with_adapters_disabled():
    layer = LowRankLinear(nn.Linear(2, 2), 1)
    model = nn.Sequential(layer)
    with torch.no_grad():
        layer.up.fill_(1.)
        layer.down.fill_(1.)
    x = torch.ones(1, 2)
    configure_case(model, None, 'B_original_wan')
    assert torch.equal(model(x), layer.base(x))
    configure_case(model, None, 'C_adapted_without_memory')
    assert not torch.equal(model(x), layer.base(x))


def test_unknown_case_rejected():
    try:
        configure_case(nn.Identity(), None, 'guess')
    except ValueError:
        return
    raise AssertionError('Unknown case accepted')
