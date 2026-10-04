import torch
from real_video.wan_cat_memory import LowRankLinear, GaussianMemory, gaussian_tokens, merge_lora_for_inference


def test_lora_base_parity_and_gradient():
    torch.manual_seed(2)
    base=torch.nn.Linear(12,8)
    layer=LowRankLinear(base,rank=3)
    x=torch.randn(2,4,12)
    assert torch.equal(layer(x),base(x))
    layer(x).square().mean().backward()
    assert base.weight.grad is None
    assert layer.up.grad.abs().sum()>0
    with torch.no_grad(): layer.up.add_(.1)
    assert not torch.equal(layer(x),base(x))
    layer.enabled=False
    assert torch.equal(layer(x),base(x))


def test_gaussian_memory_uses_all_points_and_is_permutation_invariant():
    torch.manual_seed(4)
    n=100
    asset=dict(position=torch.rand(n,3),covariance=torch.eye(3)[None].repeat(n,1,1),
               normal=torch.rand(n,3),colour=torch.rand(n,3),opacity=torch.rand(n))
    tokens=gaussian_tokens(asset)
    perm=torch.randperm(n)
    assert tokens.shape==(64,20)
    assert torch.allclose(tokens[:,-1].sum(),torch.tensor(1.))
    assert torch.allclose(tokens,gaussian_tokens({k:v[perm] for k,v in asset.items()}),atol=1e-6)
    memory=GaussianMemory(tokens)
    text=torch.randn(1,512,4096)
    assert torch.equal(memory(text),text)
    memory(text).square().mean().backward()
    assert memory.net[-1].weight.grad.abs().sum()>0


def test_merge_effective_weights_and_reject_duplicate():
    torch.manual_seed(9)
    model=torch.nn.Sequential(LowRankLinear(torch.nn.Linear(12,8),rank=3))
    with torch.no_grad(): model[0].up.normal_(std=.01)
    x=torch.randn(2,4,12)
    before=model(x).detach()
    assert merge_lora_for_inference(model)==1
    assert torch.allclose(before,model(x),atol=1e-6)
    try:
        merge_lora_for_inference(model)
    except ValueError:
        pass
    else:
        raise AssertionError('Double merge accepted')


def test_shared_identity_residual_is_independent_of_text():
    torch.manual_seed(11)
    memory=GaussianMemory(torch.randn(64,20))
    with torch.no_grad(): memory.net[-1].weight.normal_(std=.002)
    positive=torch.randn(1,512,4096)
    negative=torch.randn_like(positive)
    delta_p=memory(positive)-positive
    delta_n=memory(negative)-negative
    assert torch.allclose(delta_p,delta_n,atol=5e-7)
    assert delta_p[:,:-64].count_nonzero()==0
    assert delta_p[:,-64:].abs().sum()>0
