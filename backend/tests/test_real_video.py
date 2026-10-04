import unittest
from unittest.mock import patch
import numpy as np
import torch
from real_video.representation import encode_clip,decode_state,render_fields
from real_video.model import GaussianVideoDenoiser,sample,schedule
from real_video.objectives import field_loss,appearance_loss


class RealVideoTests(unittest.TestCase):
    def test_context_weight_transfer_preserves_function(self):
        original=GaussianVideoDenoiser(classes=2,width=8).eval()
        extended=GaussianVideoDenoiser(classes=2,width=8,refinement_blocks=2).eval()
        mismatch=extended.load_state_dict(original.state_dict(),strict=False)
        self.assertEqual(mismatch.unexpected_keys,[])
        self.assertTrue(all(k.startswith(('refinements.','coordinates.')) for k in mismatch.missing_keys))
        x=torch.randn(1,9,8,8,8); t=torch.tensor([500]); label=torch.tensor([1])
        torch.testing.assert_close(original(x,t,label),extended(x,t,label),rtol=0,atol=0)

    def test_new_context_weights_train_after_zero_init(self):
        model=GaussianVideoDenoiser(classes=2,width=8,refinement_blocks=1)
        optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
        x=torch.randn(1,9,8,8,8); target=torch.randn_like(x)
        for _ in range(2):
            optimizer.zero_grad()
            (model(x,torch.tensor([500]),torch.tensor([1]))-target).square().mean().backward()
            optimizer.step()
        for name,param in model.named_parameters():
            if name.startswith(('refinements.','coordinates.')):
                self.assertIsNotNone(param.grad,name)
                self.assertTrue(torch.isfinite(param.grad).all(),name)
                self.assertGreater(param.grad.abs().sum().item(),0,name)

    def test_equal_channel_objective_matches_original(self):
        p=torch.randn(2,9,8,4,4); q=torch.randn_like(p)
        torch.testing.assert_close(field_loss(p,q), (p-q).square().mean())

    def test_appearance_objective_backpropagates_into_gaussian_fields(self):
        p=torch.randn(1,9,8,8,8,requires_grad=True)
        noisy=torch.zeros_like(p); noisy.data[:,4]=1
        loss=appearance_loss(p,noisy,torch.tensor([0.8]).reshape(1,1,1,1,1),
                             torch.zeros(1,9,1,1,1),torch.ones(1,9,1,1,1),
                             torch.rand(1,8,3,64,64))
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(p.grad[:,:6].abs().sum().item(),0)
        self.assertGreater(p.grad[:,6:].abs().sum().item(),0)

    def test_appearance_objective_skips_high_noise(self):
        p=torch.randn(1,9,8,8,8,requires_grad=True)
        loss=appearance_loss(p,p.detach(),torch.tensor([0.1]).reshape(1,1,1,1,1),
                             torch.zeros(1,9,1,1,1),torch.ones(1,9,1,1,1),
                             torch.rand(1,8,3,64,64))
        self.assertEqual(loss.item(),0)
        loss.backward()
        self.assertEqual(p.grad.abs().sum().item(),0)

    def test_degenerate_direction_gives_proper_frame(self):
        state=decode_state(torch.zeros(2,9,4,4))
        torch.testing.assert_close(state['R'].transpose(-1,-2)@state['R'],torch.eye(3).expand(2,16,3,3))
        torch.testing.assert_close(torch.linalg.det(state['R']),torch.ones(2,16))

    def test_renderer_uses_generated_axes_and_has_gradients(self):
        f=torch.zeros(1,9,1,8,8,requires_grad=True)
        with torch.no_grad():
            f[:,4]=1; f[:,2]=0.5; f[:,3]=-0.5
            f[:,6]=torch.linspace(-1,1,8)[None,None,None,:]
            f[:,7]=torch.randn(1,1,8,8)
        a=render_fields(f)
        rotated=f.detach().clone(); rotated[:,4]=0; rotated[:,5]=1
        b=render_fields(rotated)
        self.assertGreater((a-b).abs().mean().item(),0.001)
        a.mean().backward()
        self.assertTrue(torch.isfinite(f.grad).all())
        self.assertGreater(f.grad[:,:6].abs().sum().item(),0)

    def test_noise_only_generation_does_not_call_video_encoder(self):
        model=GaussianVideoDenoiser(classes=2,width=8).eval()
        with patch('real_video.representation.encode_clip',side_effect=AssertionError('Encoder used')):
            x=sample(model,torch.tensor([0]),seed=37,steps=2)
        self.assertEqual(tuple(x.shape),(1,9,8,32,32))
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(torch.isfinite(render_fields(x)).all())

    def test_flow_changes_dot_positions(self):
        rng=np.random.default_rng(10)
        frame=(rng.random((64,64,3))*255).astype(np.uint8)
        frames=np.stack([np.roll(frame,i,axis=1) for i in range(8)])
        f=encode_clip(frames)
        self.assertGreater(f[0,-1,5:-5,5:-5].mean(),0.1)
        np.testing.assert_allclose(f[4]**2+f[5]**2,1,atol=1e-5)

    def test_new_model_weights_receive_gradients(self):
        model=GaussianVideoDenoiser(classes=2,width=8)
        x=torch.randn(1,9,8,32,32)
        y=model(x,torch.tensor([500]),torch.tensor([1]))
        y.square().mean().backward()
        for module in (model.input,model.label,model.output,model.qkv):
            grads=[p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0)

    def test_generated_field_rerender_is_deterministic(self):
        fields=torch.randn(1,9,8,8,8)*0.2
        fields[:,4]=1
        a=render_fields(fields)
        b=render_fields(fields.clone())
        torch.testing.assert_close(a,b)


if __name__=='__main__': unittest.main()
