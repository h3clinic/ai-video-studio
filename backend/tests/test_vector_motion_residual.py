import unittest
import torch
from real_video.vector_motion_residual import ResidualVelocityNetwork


class ResidualVelocityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)
        self.model = ResidualVelocityNetwork(hidden=16)
        self.observed = torch.randn(2,3,7,2) * .1
        self.adjacency = torch.rand(2,7,7)

    def test_zero_head_is_exact_constant_observed_velocity(self):
        state = self.model.initialize(self.observed, self.adjacency)
        initial = state['initial_velocity'].clone()
        for _ in range(20):
            state = self.model.step(state)
            torch.testing.assert_close(state['velocity'], initial, atol=0, rtol=0)
        self.assertEqual(self.model.head[-1].out_features, 2)

    def test_residual_bounded_and_can_reverse_motion(self):
        with torch.no_grad():
            self.model.head[-1].bias.copy_(torch.tensor([-4., 4.]))
        state = self.model.initialize(self.observed, self.adjacency)
        result = self.model.step(state)
        residual = result['velocity'] - state['initial_velocity']
        self.assertTrue(bool((residual[...,0] < 0).all()))
        self.assertTrue(bool((residual[...,1] > 0).all()))
        self.assertTrue(bool((residual.abs() <= self.model.residual_scale * state['scale'] + 1e-8).all()))
        last = self.observed[:,2]
        small_velocity = torch.ones_like(last) * .0001
        sequence = torch.stack((last-2*small_velocity,last-small_velocity,last),1)
        reversed_state = self.model.step(self.model.initialize(sequence,self.adjacency))
        self.assertTrue(bool((reversed_state['velocity'][...,0] < 0).all()))

    def test_equivariance_resume_and_no_input_mutation(self):
        with torch.no_grad():
            torch.nn.init.normal_(self.model.head[-1].weight, std=.1)
        original = self.observed.clone()
        index = torch.randperm(7)
        output, state = self.model.rollout(original, self.adjacency, 5)
        other, _ = self.model.rollout(original[:,:,index], self.adjacency[:,index][:,:,index], 5)
        torch.testing.assert_close(other, output[:,:,index], atol=1e-6, rtol=1e-6)
        scaled, _ = self.model.rollout(original*2+3, self.adjacency, 5)
        torch.testing.assert_close(scaled, output*2+3, atol=2e-6, rtol=2e-6)
        copy = {key:value.detach().clone() for key,value in state.items()}
        for key, value in self.model.step(state).items():
            torch.testing.assert_close(value, self.model.step(copy)[key], atol=0, rtol=0)
        torch.testing.assert_close(original, self.observed, atol=0, rtol=0)

    def test_gradient_updates_actual_residual_weights(self):
        output, _ = self.model.rollout(self.observed, self.adjacency, 4)
        output.square().mean().backward()
        self.assertGreater(float(self.model.head[-1].weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
