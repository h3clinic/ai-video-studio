import unittest
import torch
from real_video.vector_motion_acceleration import AccelerationMotionNetwork, load_acceleration_model


class AccelerationMotionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119)
        self.model=AccelerationMotionNetwork(hidden=16)
        self.observed=torch.randn(2,3,7,2)*.1
        self.adjacency=torch.rand(2,7,7)

    def test_zero_head_is_constant_velocity_not_initial_acceleration(self):
        state=self.model.initialize(self.observed,self.adjacency)
        initial=state['velocity'].clone()
        self.assertGreater(float(state['acceleration'].abs().sum()),0)
        for _ in range(20):
            state=self.model.step(state)
            torch.testing.assert_close(state['velocity'],initial,atol=0,rtol=0)
            torch.testing.assert_close(state['acceleration'],torch.zeros_like(initial),atol=0,rtol=0)

    def test_acceleration_accumulates_and_is_bounded(self):
        with torch.no_grad(): self.model.head[-1].bias.copy_(torch.tensor([2.,-2.]))
        state=self.model.initialize(self.observed,self.adjacency)
        initial=state['velocity'].clone(); p0=state['position'].clone()
        a=self.model.acceleration_scale*state['scale']*torch.tensor([2.,-2.]).tanh()
        for step in range(1,21):
            state=self.model.step(state)
            torch.testing.assert_close(state['velocity'],initial+step*a,atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(state['position'],p0+step*initial+step*(step+1)/2*a,atol=3e-6,rtol=1e-5)
            self.assertTrue(bool((state['acceleration'].abs()<=self.model.acceleration_scale*state['scale']+1e-7).all()))

    def test_reversal_not_anchored_to_initial_velocity(self):
        rest=torch.tensor([[[0.,0.],[1.,1.]]]); velocity=torch.tensor([[[.02,0.],[.02,0.]]])
        observations=torch.stack((rest-2*velocity,rest-velocity,rest),1)
        adjacency=torch.eye(2)[None]
        with torch.no_grad(): self.model.head[-1].bias.copy_(torch.tensor([-4.,0.]))
        state=self.model.initialize(observations,adjacency)
        for _ in range(10): state=self.model.step(state)
        self.assertTrue(bool((state['velocity'][...,0]<0).all()))

    def test_resume_and_input_immutability(self):
        original=self.observed.clone()
        with torch.no_grad(): torch.nn.init.normal_(self.model.head[-1].weight,std=.1)
        _,state=self.model.rollout(self.observed,self.adjacency,5)
        cloned={k:v.detach().clone() for k,v in state.items()}
        expected=self.model.step(state); actual=self.model.step(cloned)
        for key in expected: torch.testing.assert_close(expected[key],actual[key],atol=0,rtol=0)
        torch.testing.assert_close(original,self.observed,atol=0,rtol=0)

    def test_permutation_and_affine_scale_translation(self):
        with torch.no_grad(): torch.nn.init.normal_(self.model.head[-1].weight,std=.1)
        index=torch.randperm(7)
        result,_=self.model.rollout(self.observed,self.adjacency,5)
        other,_=self.model.rollout(self.observed[:,:,index],self.adjacency[:,index][:,:,index],5)
        scaled,_=self.model.rollout(self.observed*2+3,self.adjacency,5)
        torch.testing.assert_close(other,result[:,:,index],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(scaled,result*2+3,atol=3e-6,rtol=3e-6)

    def test_gradient_and_strict_loader(self):
        output,_=self.model.rollout(self.observed,self.adjacency,4)
        output.square().mean().backward()
        self.assertGreater(float(self.model.head[-1].weight.grad.abs().sum()),0)
        packet=dict(architecture=self.model.architecture,model_config=self.model.config,model=self.model.state_dict())
        loaded=load_acceleration_model(packet)
        replay,_=loaded.rollout(self.observed,self.adjacency,4)
        torch.testing.assert_close(replay,output,atol=0,rtol=0)
        packet['architecture']='residual_velocity_v1'
        with self.assertRaises(ValueError): load_acceleration_model(packet)

    def test_invalid_acceleration_scale(self):
        with self.assertRaises(ValueError): AccelerationMotionNetwork(acceleration_scale=0)

    def test_reversal_metric_fixed_centroid_does_not_invent_cv_turns(self):
        from real_video.train_acceleration_ablation import acceleration_metrics
        from real_video.vector_motion_residual import ResidualVelocityNetwork
        base=torch.tensor([[-1.,0.],[1.,0.],[0.,.5]])
        velocity=torch.tensor([[.03,.01],[-.02,0.],[.005,-.02]])
        positions=base[None,None]+torch.arange(15)[None,:,None,None]*velocity[None,None]
        quality=torch.rand(1,15,3)*.8+.2
        data=dict(position=positions,confidence=quality,visibility=torch.ones_like(quality),
                  adjacency=torch.eye(3)[None],metadata=[dict(sequence='metric_unit',split='validation')])
        metrics=acceleration_metrics(AccelerationMotionNetwork(hidden=8),ResidualVelocityNetwork(hidden=8),data,[0],'cpu')
        for name in ['learned','unchanged_residual','velocity','damped75','damped95','frozen']:
            counts=metrics['results'][name]['sequences']['metric_unit']['weighted_reversal_counts']
            self.assertEqual(counts['tp'],0)
            self.assertEqual(counts['fp'],0)
        self.assertLess(metrics['results']['learned']['macro']['acceleration_epe'],1e-5)


if __name__=='__main__': unittest.main()
