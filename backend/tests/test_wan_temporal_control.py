import torch
from torch import nn

from real_video.wan_temporal_control import (
    TemporalSpatialControl, attach_temporal_control,
    transport_first_features, wan_temporal_average,
)


def must_reject(call):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError('Invalid input accepted')


def test_initialization_and_disabled_base_parity():
    control = TemporalSpatialControl(8, blocks=(0,), channels=2)
    control.bind(torch.randn(1, 2, 3, 4, 6), (3, 2, 3))
    hidden = torch.randn(2, 18, 8)
    assert torch.equal(control(hidden, 0), hidden)
    with torch.no_grad():
        control.heads[0].bias.fill_(1)
    assert torch.allclose(control(hidden, 0), hidden+1)
    control.enabled = False
    assert torch.equal(control(hidden, 0), hidden)


def test_time_height_width_token_order():
    control = TemporalSpatialControl(1, blocks=(0,), channels=1, hidden=1)
    control.encoder = nn.Identity()
    with torch.no_grad():
        control.heads[0].weight.fill_(1)
    volume = torch.arange(24).reshape(1, 1, 3, 2, 4).float()
    control.bind(volume, (3, 2, 4))
    output = control(torch.zeros(1, 24, 1), 0)
    assert torch.equal(output.flatten(), torch.arange(24).float())


def test_last_time_slot_gradients_and_no_cross_time_mixing():
    control = TemporalSpatialControl(1, blocks=(0,), channels=1, hidden=1)
    control.encoder = nn.Identity()
    with torch.no_grad():
        control.heads[0].weight.fill_(1)
    condition = torch.randn(1, 1, 3, 2, 2, requires_grad=True)
    control.bind(condition, (3, 2, 2))
    control(torch.zeros(1, 12, 1), 0)[:, -4:].sum().backward()
    assert torch.equal(condition.grad[:, :, :2], torch.zeros_like(condition.grad[:, :, :2]))
    assert torch.equal(condition.grad[:, :, 2], torch.ones_like(condition.grad[:, :, 2]))
    assert control.heads[0].weight.grad is not None


def test_checkpointed_hook_trains_future_head():
    from torch.utils.checkpoint import checkpoint
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Linear(8, 8).requires_grad_(False)])
    control = TemporalSpatialControl(8, blocks=(0,), channels=2)
    attach_temporal_control(model, control)
    control.bind(torch.randn(1, 2, 3, 2, 2), (3, 2, 2))
    hidden = torch.randn(1, 12, 8, requires_grad=True)
    checkpoint(model.blocks[0], hidden, use_reentrant=False)[:, -4:].square().mean().backward()
    assert control.heads[0].weight.grad.abs().sum() > 0
    assert model.blocks[0].weight.grad is None
    assert 'gaussian_temporal_control.heads.0.weight' in model.state_dict()


def test_temporal_average_uses_all_four_future_frames():
    volume = torch.arange(9).float().reshape(1, 1, 9, 1, 1).requires_grad_()
    result = wan_temporal_average(volume)
    assert torch.equal(result.flatten(), torch.tensor([0., 2.5, 6.5]))
    result.sum().backward()
    assert torch.equal(volume.grad.flatten(), torch.tensor([1.] + [.25]*8))
    must_reject(lambda: wan_temporal_average(volume[:, :, :8]))
    assert torch.equal(wan_temporal_average(volume[:, :, :1]), volume[:, :, :1])


def test_transport_preserves_first_and_moves_feature():
    first = torch.arange(9).float().reshape(1, 1, 3, 3)
    tracks = torch.tensor([[[[1., 1.]], [[2., 0.]], [[0., 2.]]]])
    volume, occupancy = transport_first_features(first, tracks)
    assert torch.equal(volume[:, :, 0], first)
    assert torch.equal(occupancy[:, :, 0], torch.ones_like(first))
    assert volume[0, 0, 1, 0, 2] == 4
    assert volume[0, 0, 2, 2, 0] == 4
    assert occupancy[:, :, 1:].sum() == 2


def test_transport_visibility_and_bounds():
    first = torch.ones(1, 1, 3, 3)
    tracks = torch.tensor([[[[1., 1.]], [[2., 1.]], [[3., 1.]]]])
    visible = torch.tensor([[[1.], [0.], [1.]]])
    volume, occupancy = transport_first_features(first, tracks, visible)
    assert volume[:, :, 1:].count_nonzero() == 0
    assert occupancy[:, :, 1:].count_nonzero() == 0
    visible = torch.tensor([[[0.], [1.], [1.]]])
    assert transport_first_features(first, tracks, visible)[1][:, :, 1:].count_nonzero() == 0


def test_weighted_collisions_and_permutation_invariance():
    first = torch.tensor([[[[2., 0., 6.]]]])
    tracks = torch.tensor([[[[0., 0.], [2., 0.]], [[1., 0.], [1., 0.]]]])
    weight = torch.tensor([[1., 3.]])
    volume, occupancy = transport_first_features(first, tracks, weights=weight)
    assert volume[0, 0, 1, 0, 1] == 5
    assert occupancy[0, 0, 1, 0, 1] == 4
    other = transport_first_features(first, tracks.flip(2), weights=weight.flip(1))
    assert torch.equal(volume, other[0]) and torch.equal(occupancy, other[1])


def test_bilinear_transport_gradients_and_mass():
    first = torch.arange(9).double().reshape(1, 1, 3, 3).requires_grad_()
    tracks = torch.tensor([[[[.5, .5]], [[1.25, 1.5]]]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[.7]], dtype=torch.float64, requires_grad=True)
    volume, occupancy = transport_first_features(first, tracks, weights=weight)
    assert torch.allclose(occupancy[:, :, 1].sum(), weight.sum())
    assert torch.allclose(volume[0, 0, 1, 1:, 1:], torch.full((2, 2), 2., dtype=torch.float64))
    loss = volume[:, :, 1].sum() + occupancy[0, 0, 1, 1, 1]
    loss.backward()
    assert first.grad.abs().sum() > 0
    assert tracks.grad[:, 0].abs().sum() > 0
    assert tracks.grad[:, 1].abs().sum() > 0
    assert weight.grad.abs().sum() > 0


def test_empty_tracks_and_singleton_grid():
    first = torch.tensor([[[[3.]]]])
    volume, occupancy = transport_first_features(first, torch.empty(1, 3, 0, 2))
    assert torch.equal(volume.flatten(), torch.tensor([3., 0., 0.]))
    assert torch.equal(occupancy.flatten(), torch.tensor([1., 0., 0.]))
    volume, occupancy = transport_first_features(first, torch.zeros(1, 2, 1, 2))
    assert torch.equal(volume.flatten(), torch.tensor([3., 3.]))
    assert occupancy.sum() == 2


def test_explicit_validation():
    control = TemporalSpatialControl(8, blocks=(0,), channels=2)
    must_reject(lambda: control.bind(torch.ones(1, 2, 3, 2, 2), (2, 2, 2)))
    must_reject(lambda: control.bind(torch.ones(1, 2, 3, 2, 2), (3., 2, 2)))
    control.bind(torch.ones(1, 2, 3, 2, 2), (3, 2, 2))
    must_reject(lambda: control(torch.ones(1, 11, 8), 0))
    must_reject(lambda: TemporalSpatialControl(8, blocks=(0, 0)))
    first, tracks = torch.ones(1, 2, 3, 3), torch.ones(1, 2, 1, 2)
    must_reject(lambda: transport_first_features(first, tracks, visibility=torch.ones(1, 2, 1)*2))
    must_reject(lambda: transport_first_features(first, tracks, weights=-torch.ones(1, 1)))
    must_reject(lambda: transport_first_features(first, tracks*float('nan')))
