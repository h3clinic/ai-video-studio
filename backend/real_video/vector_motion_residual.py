"""Observed-velocity anchored recurrence with no compulsory multiplicative decay.

Still a deterministic, projected-XY conditional predictor. It does not train
depth, anatomy or a stochastic action prior. Stored initial velocity is a fixed
reference, not an externally refreshed future observation.
"""
import torch
from torch import nn
from .vector_motion_network import VectorMotionNetwork


class ResidualVelocityNetwork(VectorMotionNetwork):
    architecture = 'residual_velocity_v1'

    def __init__(self, hidden=64, residual_scale=.02):
        super().__init__(hidden=hidden, residual_scale=residual_scale, damping_init=.75)
        self.config = dict(hidden=hidden, residual_scale=residual_scale)
        self.head[-1] = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def step(self, state):
        p, v, h, a = (state[key] for key in ['position', 'velocity', 'memory', 'adjacency'])
        batch, nodes, _ = p.shape
        neighbor = torch.cat((torch.bmm(a, h),
                              (torch.bmm(a, p) - p) / state['scale'],
                              (torch.bmm(a, v) - v) / state['scale'] * 20), -1)
        message = self.message(neighbor)
        memory = self.gru(torch.cat((self.features(state), message), -1).reshape(batch * nodes, -1),
                          h.reshape(batch * nodes, -1)).reshape(batch, nodes, -1)
        residual = self.residual_scale * state['scale'] * self.head(memory).tanh()
        velocity = state['initial_velocity'] + residual
        return dict(state, position=p + velocity, velocity=velocity, acceleration=velocity - v,
                    memory=memory, time=state['time'] + 1)
