import torch
from torch import nn

from real_video.wan_latent_appearance import (
    LatentAppearanceControl, install_latent_appearance, init_anchor_and_recall,
)
from real_video.wan_temporal_control import TemporalSpatialControl, attach_temporal_control


def rejects(call):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError('Invalid condition accepted')


def fixture():
    torch.manual_seed(309)
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Linear(8, 8)])
    base = TemporalSpatialControl(8, blocks=(0,), channels=5, hidden=4)
    attach_temporal_control(model, base)
    base.bind(torch.randn(1, 5, 2, 4, 4), (2, 2, 2))
    with torch.no_grad():
        base.heads[0].weight.normal_()
        base.heads[0].bias.normal_()
    return model, base


def condition():
    value = torch.randn(1, 17, 2, 4, 4)
    value[:, 16] = .8
    return value


def test_exact_zero_initialization_preserves_trained_branch():
    model, base = fixture()
    hidden = torch.randn(2, 8, 8)
    before = model.blocks[0](hidden).detach()
    hooks = len(model.blocks[0]._forward_hooks)
    reader = install_latent_appearance(model, base, hidden=4)
    reader.bind(condition(), (2, 2, 2))
    assert torch.equal(before, model.blocks[0](hidden))
    assert len(model.blocks[0]._forward_hooks) == hooks
    assert len(base._forward_hooks) == 1
    rejects(lambda: install_latent_appearance(model, base))


def test_only_new_reader_trains_through_existing_checkpointed_hooks():
    from torch.utils.checkpoint import checkpoint
    model, base = fixture()
    old = {name: p.detach().clone() for name, p in model.named_parameters()}
    reader = install_latent_appearance(model, base, hidden=4)
    reader.bind(condition(), (2, 2, 2))
    optimizer = torch.optim.SGD(reader.parameters(), lr=.1)
    hidden = torch.randn(1, 8, 8, requires_grad=True)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        checkpoint(model.blocks[0], hidden, use_reentrant=False).square().mean().backward()
        optimizer.step()
    assert reader.heads[0].weight.grad.abs().sum() > 0
    assert reader.encoder[0].weight.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        if name in old:
            assert not p.requires_grad and p.grad is None and torch.equal(old[name], p)
        else:
            assert 'latent_appearance' in name and p.requires_grad


def test_residual_adds_once_and_disable_returns_old_output():
    model, base = fixture()
    hidden = torch.randn(1, 8, 8)
    before = model.blocks[0](hidden).detach()
    reader = install_latent_appearance(model, base, hidden=4)
    reader.bind(condition(), (2, 2, 2))
    with torch.no_grad():
        reader.heads[0].bias.fill_(2)
    assert torch.allclose(model.blocks[0](hidden), before+2)
    reader.enabled = False
    assert torch.equal(model.blocks[0](hidden), before)
    reader.enabled = True
    base.enabled = False
    assert torch.equal(model.blocks[0](hidden), nn.functional.linear(hidden, model.blocks[0].weight, model.blocks[0].bias))


def test_patch_unshuffle_retains_different_subpixel_positions():
    reader = LatentAppearanceControl(1, blocks=(0,), hidden=1)
    reader.encoder = nn.Identity()
    reader.heads = nn.ModuleList([nn.Linear(68, 1, bias=False)])
    with torch.no_grad():
        reader.heads[0].weight.zero_()
        reader.heads[0].weight[0, 0] = 1
    a = torch.zeros(1, 17, 1, 2, 2); a[0, 0, 0, 0, 0] = 1
    b = torch.zeros_like(a); b[0, 0, 0, 0, 1] = 1
    reader.bind(a, (1, 1, 1)); first = reader(torch.zeros(1, 1, 1), 0)
    reader.bind(b, (1, 1, 1)); second = reader(torch.zeros(1, 1, 1), 0)
    assert float(first.detach()) == 1 and float(second.detach()) == 0
    assert torch.equal(a.mean((-1, -2)), b.mean((-1, -2)))


def test_time_order_and_condition_gradients():
    reader = LatentAppearanceControl(1, blocks=(0,), hidden=1)
    reader.encoder = nn.Identity()
    reader.heads = nn.ModuleList([nn.Linear(68, 1, bias=False)])
    with torch.no_grad():
        reader.heads[0].weight.fill_(1)
    maps = condition().requires_grad_()
    reader.bind(maps, (2, 2, 2))
    reader(torch.zeros(1, 8, 1), 0)[:, 4:].sum().backward()
    assert torch.equal(maps.grad[:, :, 0], torch.zeros_like(maps.grad[:, :, 0]))
    assert bool((maps.grad[:, :, 1] == 1).all())


def test_invalid_shape_coverage_and_grid_fail_closed():
    reader = LatentAppearanceControl(8, blocks=(0,), hidden=4)
    rejects(lambda: reader.bind(torch.zeros(1, 17, 2, 2, 2), (2, 2, 2)))
    bad = condition(); bad[:, 16] = 2
    rejects(lambda: reader.bind(bad, (2, 2, 2)))
    bad = condition(); bad[0, 0, 0, 0, 0] = float('nan')
    rejects(lambda: reader.bind(bad, (2, 2, 2)))
    model, base = fixture(); reader = install_latent_appearance(model, base, hidden=4)
    reader.bind(torch.zeros(1, 17, 1, 4, 4), (1, 2, 2))
    rejects(lambda: model.blocks[0](torch.zeros(1, 8, 8)))


def test_anchor_only_memory_read_is_detached_and_ids_persist():
    ids = torch.tensor([11, 25, 80, 101])
    cache = dict(pixel=torch.arange(4), ids=torch.arange(4), weight=torch.ones(4))
    moved = dict(pixel=torch.tensor([1, 0, 3, 2]), ids=torch.arange(4), weight=torch.ones(4))
    anchor = torch.arange(64).float().reshape(16, 2, 2).requires_grad_()
    packet = init_anchor_and_recall(anchor, cache, [cache, moved], ids=ids,
        asset_digest='a'*64, confidence=torch.ones(2, 2))
    assert packet['condition'].shape == (1, 17, 2, 2, 2)
    assert not packet['condition'].requires_grad
    assert torch.equal(packet['condition'][0, :16, 0], anchor)
    assert torch.equal(packet['condition'][0, :16, 1], anchor.flip(-1))
    snapshot = packet['memory'].snapshot()
    assert torch.equal(snapshot['ids'], ids) and snapshot['writes'] == 1
    assert packet['future_latent_reads'] == 0 and packet['geometry_updated'] is False


def test_zero_confidence_does_not_invent_latent_features():
    cache = dict(pixel=torch.arange(4), ids=torch.arange(4), weight=torch.ones(4))
    packet = init_anchor_and_recall(torch.ones(16, 2, 2), cache, [cache],
        ids=torch.arange(4), asset_digest='b'*64, confidence=torch.zeros(2, 2))
    assert torch.count_nonzero(packet['condition'][:, :16]) == 0
    assert torch.count_nonzero(packet['condition'][:, 16:]) == 0
    assert packet['write_report']['updated_gaussians'] == 0
    assert packet['memory'].snapshot()['writes'] == 1


def test_unseen_gaussian_is_not_valid_appearance_coverage():
    anchor_cache = dict(pixel=torch.tensor([0]), ids=torch.tensor([0]), weight=torch.ones(1))
    recall_cache = dict(pixel=torch.tensor([0, 1]), ids=torch.tensor([0, 1]), weight=torch.ones(2))
    packet = init_anchor_and_recall(torch.ones(16, 2, 2), anchor_cache, [recall_cache],
        ids=torch.tensor([40, 80]), asset_digest='c'*64, confidence=torch.ones(2, 2))
    assert torch.equal(packet['condition'][0, 16, 0].flatten(), torch.tensor([1., 0., 0., 0.]))


def test_invalid_correspondence_is_not_silently_accepted():
    cache = dict(pixel=torch.arange(4), ids=torch.tensor([0, 1, 2, 4]), weight=torch.ones(4))
    rejects(lambda: init_anchor_and_recall(torch.ones(16, 2, 2), cache, [cache],
        ids=torch.arange(4), asset_digest='b'*64, confidence=torch.ones(2, 2)))


if __name__ == '__main__':
    tests = [value for name, value in list(globals().items()) if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print(f'{len(tests)} latent appearance tests passed')
