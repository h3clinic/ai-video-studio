import unittest
import torch
from real_video.vector_motion_network import VectorMotionNetwork


class VectorMotionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(55)
        self.model = VectorMotionNetwork(hidden=16)
        # Nonzero heads test learned graph/GRU paths rather than a zero residual.
        torch.nn.init.normal_(self.model.head[-1].weight, std=.1)
        self.position = torch.randn(2, 3, 7, 2) * .1
        self.adj = torch.rand(2, 7, 7)

    def test_translation_scale_equivariance(self):
        p, _ = self.model.rollout(self.position, self.adj, 4)
        q, _ = self.model.rollout(self.position * 2.3 + 4, self.adj, 4)
        torch.testing.assert_close(q, p * 2.3 + 4, atol=2e-6, rtol=2e-6)

    def test_permutation_equivariance(self):
        index = torch.randperm(7)
        p, _ = self.model.rollout(self.position, self.adj, 4)
        q, _ = self.model.rollout(self.position[:, :, index], self.adj[:, index][:, :, index], 4)
        torch.testing.assert_close(q, p[:, :, index], atol=1e-6, rtol=1e-6)

    def test_fixed_state_and_exact_resume(self):
        state = self.model.initialize(self.position, self.adj)
        shape = {k: v.shape for k, v in state.items()}
        for _ in range(6):
            state = self.model.step(state)
        copy = {k: v.detach().clone() for k, v in state.items()}
        a, b = self.model.step(state), self.model.step(copy)
        self.assertEqual(shape, {k: v.shape for k, v in a.items()})
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)

    def test_observation_contract_no_mutation(self):
        p, a = self.position.clone(), self.adj.clone()
        self.model.rollout(p, a, 3)
        torch.testing.assert_close(p, self.position, atol=0, rtol=0)
        torch.testing.assert_close(a, self.adj, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.model.initialize(torch.randn(2, 15, 7, 2), self.adj)
        state = self.model.initialize(self.position, self.adj * 0)
        torch.testing.assert_close(state['adjacency'], torch.eye(7)[None].expand(2, -1, -1))

    def test_learning_gradients_reach_recurrence(self):
        output, _ = self.model.rollout(self.position, self.adj, 5)
        output.square().mean().backward()
        self.assertGreater(float(self.model.gru.weight_hh.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.message[0].weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
