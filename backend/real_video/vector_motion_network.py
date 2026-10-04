"""Causal learned projected-motion recurrence, not a learned 3D anatomy model.

Only three observed control frames initialize the state. The returned fixed-size
state contains every value needed by step; no images/target tracks are read.
"""
import torch
from torch import nn


class VectorMotionNetwork(nn.Module):
    def __init__(self, hidden=64, residual_scale=.01, damping_init=.975):
        super().__init__()
        self.config = dict(hidden=hidden, residual_scale=residual_scale, damping_init=damping_init)
        self.hidden = hidden
        self.residual_scale = residual_scale
        self.encoder = nn.Sequential(nn.Linear(14, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Tanh())
        self.message = nn.Sequential(nn.Linear(hidden + 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Tanh())
        self.gru = nn.GRUCell(14 + hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[2] = torch.logit(torch.tensor((damping_init - .5) * 2))

    def features(self, state):
        p, scale = state['position'], state['scale']
        global_velocity = state['velocity'].mean(1, keepdim=True).expand_as(p)
        return torch.cat(((p - state['reference']) / scale,
                          (state['reference'] - state['origin']) / scale,
                          state['velocity'] / scale * 20,
                          state['acceleration'] / scale * 20,
                          state['initial_velocity'] / scale * 20,
                          global_velocity / scale * 20,
                          state['confidence'][..., None],
                          state['time'].expand(*p.shape[:-1], 1) / 12), -1)

    def initialize(self, observed_positions, adjacency, confidence=None):
        if observed_positions.ndim != 4 or observed_positions.shape[1] != 3 or observed_positions.shape[-1] != 2:
            raise ValueError('Exactly three B x 3 x N x 2 projected observations required')
        b, _, n, _ = observed_positions.shape
        if adjacency.shape != (b, n, n):
            raise ValueError('Adjacency must have shape B x N x N')
        if not torch.isfinite(observed_positions).all() or not torch.isfinite(adjacency).all() or (adjacency < 0).any():
            raise ValueError('Finite positions and finite nonnegative adjacency required')
        # Flow-advected controls can leave the observed foreground mask, leaving
        # no legal neighbor. Preserve the node with a self-edge, not future data.
        row_sum = adjacency.sum(-1, keepdim=True)
        eye = torch.eye(n, device=adjacency.device, dtype=adjacency.dtype)[None]
        adjacency = torch.where(row_sum > 0, adjacency, eye)
        if confidence is None:
            confidence = torch.ones((b, n), device=observed_positions.device, dtype=observed_positions.dtype)
        if confidence.shape != (b, n) or not torch.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
            raise ValueError('Confidence must be finite B x N in [0,1]')
        p = observed_positions[:, 2].clone()
        velocity = observed_positions[:, 2] - observed_positions[:, 1]
        previous = observed_positions[:, 1] - observed_positions[:, 0]
        scale = (p.amax(1, keepdim=True) - p.amin(1, keepdim=True)).amax(-1, keepdim=True).clamp_min(1e-5)
        state = dict(position=p, reference=p.clone(), origin=p.mean(1, keepdim=True), scale=scale,
                     velocity=velocity, acceleration=velocity - previous, initial_velocity=velocity.clone(),
                     confidence=confidence.clone(), adjacency=adjacency / adjacency.sum(-1, keepdim=True),
                     time=torch.zeros((b, 1, 1), device=p.device, dtype=p.dtype))
        state['memory'] = self.encoder(self.features(state))
        return state

    def step(self, state):
        p, v, h, a = (state[k] for k in ['position', 'velocity', 'memory', 'adjacency'])
        batch, nodes, _ = p.shape
        neighbor = torch.cat((torch.bmm(a, h),
                              (torch.bmm(a, p) - p) / state['scale'],
                              (torch.bmm(a, v) - v) / state['scale'] * 20), -1)
        message = self.message(neighbor)
        memory = self.gru(torch.cat((self.features(state), message), -1).reshape(batch * nodes, -1),
                          h.reshape(batch * nodes, -1)).reshape(batch, nodes, -1)
        action = self.head(memory)
        damping = .5 + .5 * torch.sigmoid(action[..., 2:3])
        velocity = damping * v + self.residual_scale * state['scale'] * action[..., :2].tanh()
        return dict(state, position=p + velocity, velocity=velocity, acceleration=velocity - v,
                    memory=memory, time=state['time'] + 1)

    def rollout(self, observed_positions, adjacency, steps, confidence=None):
        if steps < 1:
            raise ValueError('Positive rollout horizon required')
        state = self.initialize(observed_positions, adjacency, confidence)
        positions = []
        for _ in range(steps):
            state = self.step(state)
            positions.append(state['position'])
        return torch.stack(positions, 1), state
