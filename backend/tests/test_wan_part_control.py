"""CPU tiny-architecture tests; no pretrained weights or video quality claim."""
import unittest
import torch
from real_video.wan_part_control import PartSpatialControl, attach_part_control


class PartControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(45)
        self.control=PartSpatialControl(16,2,3,hidden=8)
        self.features=torch.ones(1,2,2,1,2,2)
        self.coverage=torch.full((1,2,1,1,2,2),.5)
        self.intents=torch.randn(1,2,3)
        self.control.bind(self.features,self.coverage,self.intents,(1,2,2))
        self.hidden=torch.randn(1,4,16)

    def test_zero_initialized_parity_and_gradient(self):
        out=self.control(self.hidden,0)
        self.assertTrue(torch.equal(out,self.hidden))
        out.sum().backward()
        self.assertGreater(float(self.control.heads[0].weight.grad.abs().sum()),0)

    def test_empty_part_has_no_influence_even_after_head_change(self):
        with torch.no_grad(): self.control.heads[0].weight.fill_(.2)
        self.features[:,1]=0; self.coverage[:,1]=0
        before=self.control(self.hidden,0)
        self.intents[:,1]=1000
        self.assertTrue(torch.equal(before,self.control(self.hidden,0)))

    def test_zero_coverage_pixel_stays_unchanged(self):
        with torch.no_grad(): self.control.heads[0].weight.fill_(.2)
        self.features[...,0,0]=0; self.coverage[...,0,0]=0
        out=self.control(self.hidden,0)
        self.assertTrue(torch.equal(out[:,0],self.hidden[:,0]))
        self.assertFalse(torch.equal(out[:,1],self.hidden[:,1]))

    def test_partition_permutation_invariant(self):
        with torch.no_grad(): self.control.heads[0].weight.fill_(.2)
        before=self.control(self.hidden,0)
        self.control.bind(self.features.flip(1),self.coverage.flip(1),self.intents.flip(1),(1,2,2))
        torch.testing.assert_close(before,self.control(self.hidden,0))

    def test_finite_but_unrepresentable_inputs_fail_closed(self):
        self.control.bind(self.features*0,self.coverage*0,
                          torch.full((1,2,3),1e100,dtype=torch.float64),(1,2,2))
        with self.assertRaises(ValueError): self.control(self.hidden,0)
        self.control.bind(self.features,torch.full_like(self.coverage,1e-40),self.intents,(1,2,2))
        with self.assertRaises(ValueError): self.control(self.hidden,0)

    def test_protocol_bridge_rejects_restore_with_same_write_count(self):
        from real_video.gaussian_part_protocol import GaussianPartProtocol
        from real_video.gaussian_latent_memory import GaussianLatentMemory, bind_alpha_cache
        ids=torch.arange(4); gen=torch.zeros(4,dtype=torch.long)
        bank=GaussianLatentMemory(ids,2,'a'*64,generations=gen,
            initial_features=torch.ones(4,2),initial_mass=1.)
        protocol=GaussianPartProtocol(bank,ids=ids,generations=gen,part_ids=torch.tensor([0,0,1,1]))
        cache=bind_alpha_cache(dict(pixel=ids,ids=ids,weight=torch.ones(4)),
            ids=ids,generations=gen,asset_digest='a'*64,height=2,width=2)
        self.control.bind_memory_frames(protocol,[cache],2,2,self.intents,(1,2,2))
        snap=bank.snapshot(); snap['features']*=2
        bank.restore(snap)
        self.assertEqual(bank.writes,0)
        with self.assertRaises(ValueError): self.control(self.hidden,0)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError): self.control.bind(self.features,self.coverage*2,self.intents,(1,2,2))
        with self.assertRaises(ValueError): self.control.bind(self.features,self.coverage*0,self.intents,(1,2,2))
        with self.assertRaises(ValueError): self.control(self.hidden.expand(2,-1,-1),0)
        with self.assertRaises(ValueError): self.control.bind(self.features,self.coverage,self.intents,(2,2,2))

    def test_protocol_bridge_rejects_replaced_lifetime_at_execution(self):
        from real_video.gaussian_part_protocol import GaussianPartProtocol
        from real_video.gaussian_latent_memory import GaussianLatentMemory, bind_alpha_cache
        ids=torch.arange(4); generations=torch.zeros(4,dtype=torch.long)
        bank=GaussianLatentMemory(ids,2,'a'*64,generations=generations,
            initial_features=torch.ones(4,2),initial_mass=1.)
        protocol=GaussianPartProtocol(bank,ids=ids,generations=generations,part_ids=torch.tensor([0,0,1,1]))
        raw=dict(pixel=ids,ids=ids,weight=torch.ones(4))
        cache=bind_alpha_cache(raw,ids=ids,generations=generations,asset_digest='a'*64,height=2,width=2)
        self.control.bind_memory_frames(protocol,[cache],2,2,self.intents,(1,2,2))
        self.assertTrue(torch.equal(self.control(self.hidden,0),self.hidden))
        bank.advance_generations(generations+1,ids=ids)
        with self.assertRaises(ValueError): self.control(self.hidden,0)

    def test_actual_wan_block_backprop_and_trainable_weight_change(self):
        from diffusers import WanTransformer3DModel
        from real_video.wan_cat_memory import install_lora
        old_threads=torch.get_num_threads(); torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,old_threads)
        model=WanTransformer3DModel(num_attention_heads=2,attention_head_dim=8,
            in_channels=2,out_channels=2,text_dim=8,freq_dim=8,ffn_dim=32,
            num_layers=1,patch_size=(1,2,2),image_dim=None).eval().requires_grad_(False)
        latent=torch.randn(1,2,1,4,4); text=torch.randn(1,3,8); timestep=torch.tensor([500.])
        def run(): return model(hidden_states=latent,timestep=timestep,encoder_hidden_states=text,return_dict=False)[0]
        baseline=run().detach()
        install_lora(model,2)
        handles=attach_part_control(model,self.control)
        self.addCleanup(lambda:[h.remove() for h in handles])
        self.assertTrue(torch.equal(baseline,run()))
        frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        trainable={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        optimizer.zero_grad(); run().square().mean().backward(); optimizer.step()
        parameters=dict(model.named_parameters())
        self.assertTrue(all(torch.equal(p,parameters[name]) for name,p in frozen.items()))
        self.assertTrue(any(not torch.equal(p,parameters[name]) for name,p in trainable.items() if 'gaussian_part_control' in name))
        self.assertTrue(any(not torch.equal(p,parameters[name]) for name,p in trainable.items() if '.up' in name))
        self.assertFalse(torch.equal(baseline,run()))
        trained=run().detach()
        # Reuse the exact same noise, text, time and updated LoRA; ablate only
        # the learned memory branch so sensitivity is not a seed difference.
        self.control.enabled=False
        self.assertFalse(torch.equal(trained,run()))
        self.control.enabled=True
        saved={k:v.detach().clone() for k,v in self.control.state_dict().items()}
        with torch.no_grad(): self.control.heads[0].weight.zero_()
        self.control.load_state_dict(saved)
        self.assertTrue(torch.equal(trained,run()))

    def test_factory_uses_latent_output_channels_not_i2v_concat(self):
        from diffusers import WanTransformer3DModel
        from real_video.wan_part_control import install_part_adaptation
        model=WanTransformer3DModel(num_attention_heads=2,attention_head_dim=8,
            in_channels=36,out_channels=16,text_dim=8,freq_dim=8,ffn_dim=32,
            num_layers=1,patch_size=(1,2,2),image_dim=None)
        control,handles,report=install_part_adaptation(model,intent_dim=3,hidden=8,rank=2)
        self.addCleanup(lambda:[h.remove() for h in handles])
        self.assertEqual(control.channels,16)
        self.assertGreater(report['trainable_parameters'],0)
        with self.assertRaises(ValueError): install_part_adaptation(model,intent_dim=3)


if __name__ == '__main__': unittest.main()
