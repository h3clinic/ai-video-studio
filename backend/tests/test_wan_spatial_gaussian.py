import torch
from real_video.wan_spatial_gaussian import SpatialControl, attach_control


def test_zero_initialization_and_disable_parity():
    control = SpatialControl(8, blocks=(0,))
    control.bind(torch.randn(1, 14, 8, 12), (3, 4, 6))
    hidden = torch.randn(2, 72, 8)
    assert torch.equal(control(hidden, 0), hidden)
    with torch.no_grad(): control.heads[0].weight.fill_(.1)
    control.enabled = False
    assert torch.equal(control(hidden, 0), hidden)


def test_first_time_slot_and_gradient():
    control = SpatialControl(8, blocks=(0,))
    control.bind(torch.randn(1, 14, 8, 12), (3, 4, 6))
    hidden = torch.randn(1, 72, 8, requires_grad=True)
    control(hidden, 0).square().mean().backward()
    assert control.heads[0].weight.grad.abs().sum() > 0
    with torch.no_grad(): control.heads[0].bias.fill_(1.)
    out = control(hidden, 0)
    assert torch.equal(out[:, 24:], hidden[:, 24:])
    assert torch.allclose(out[:, :24], hidden[:, :24]+1.)


def test_grid_mismatch_fails():
    control = SpatialControl(8, blocks=(0,))
    control.bind(torch.randn(1, 14, 8, 12), (1, 4, 6))
    try: control(torch.randn(1, 23, 8), 0)
    except ValueError: return
    raise AssertionError('Wrong grid accepted')


def test_checkpointed_hook_gradients():
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Linear(8, 8).requires_grad_(False)])
    control = SpatialControl(8, blocks=(0,))
    attach_control(model, control)
    control.bind(torch.randn(1, 14, 8, 12), (1, 4, 6))
    hidden = torch.randn(1, 24, 8, requires_grad=True)
    checkpoint(model.blocks[0], hidden, use_reentrant=False).square().mean().backward()
    assert control.heads[0].weight.grad.abs().sum() > 0
    assert model.blocks[0].weight.grad is None
    assert 'gaussian_spatial_control.heads.0.weight' in model.state_dict()
