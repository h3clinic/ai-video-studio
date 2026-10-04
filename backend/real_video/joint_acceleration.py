"""Recurrent 3D joint acceleration state and explicit SO(3) integration.

Accelerations are learned afresh each step; this permits velocity reversals but
does not guarantee oscillation, anatomy, stability, conservation, or good video.
The body-frame rotation increment is a midpoint-style approximation. It is exact
for constant, collinear angular velocity/acceleration, not arbitrary changing
rotation axes. FK/skinning remains the separate articulated Gaussian decoder.

``step`` retains fixed-shape tensors and reads no history. ``rollout`` explicitly
collects states, so its returned history and training autograd grow with horizon;
fixed recurrent state alone is not a total-memory or compression claim.
"""
import math

import torch
from torch import nn


def skew(vector):
    """[... ,3] vectors to differentiable skew-symmetric matrices."""
    if vector.shape[-1] != 3:
        raise ValueError('skew requires [...,3] vectors')
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def _finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')


def _step_size(dt, reference):
    """Canonical positive step size [B,1,1] with reference device/dtype."""
    batch = reference.shape[0]
    dt = torch.as_tensor(dt, device=reference.device, dtype=reference.dtype)
    if dt.numel() == 1:
        dt = dt.reshape(1, 1, 1).expand(batch, 1, 1)
    elif dt.shape in [(batch,), (batch, 1), (batch, 1, 1)]:
        dt = dt.reshape(batch, 1, 1)
    else:
        raise ValueError('dt must be a scalar or one value per batch item')
    if not bool(torch.isfinite(dt).all()) or bool((dt <= 0).any()):
        raise ValueError('dt must be finite and strictly positive')
    return dt


def _state_contract(state):
    rotation, omega = state['rotation'], state['omega']
    if rotation.ndim != 4 or rotation.shape[-2:] != (3, 3):
        raise ValueError('rotation must have shape [B,J,3,3]')
    batch, joint_count = rotation.shape[:2]
    if batch < 1 or joint_count < 1 or not rotation.is_floating_point():
        raise ValueError('Nonempty floating pose state required')
    expected = {'omega': (batch, joint_count, 3), 'alpha': (batch, joint_count, 3),
                'root_position': (batch, 3), 'root_velocity': (batch, 3),
                'root_acceleration': (batch, 3), 'elapsed_time': (batch, 1)}
    _finite('rotation', rotation)
    for name, shape in expected.items():
        value = state[name]
        if value.shape != shape or value.device != rotation.device or value.dtype != rotation.dtype:
            raise ValueError(f'{name} must have shape {shape} with rotation device/dtype')
        _finite(name, value)
    return rotation, omega


def integrate_joint_state(state, angular_acceleration=None, root_acceleration=None, dt=None):
    """Pure explicit integration, returning a new state without in-place edits.

    If overrides are absent, use accelerations stored in ``state``. This lower
    level function supports prescribed acceleration diagnostics independent of
    learned weights. All non-dynamic state fields pass through without mutation.
    Angular velocity/acceleration are body-frame coordinate components. Root
    position, velocity, and acceleration use world coordinates (e.g. metres).
    """
    rotation, omega = _state_contract(state)
    alpha = state['alpha'] if angular_acceleration is None else angular_acceleration
    acceleration = state['root_acceleration'] if root_acceleration is None else root_acceleration
    for name, value, reference in [('angular_acceleration', alpha, omega),
                                    ('root_acceleration', acceleration, state['root_velocity'])]:
        if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f'{name} shape/device/dtype mismatch')
        _finite(name, value)
    step = _step_size(state['dt'] if dt is None else dt, rotation)
    linear_step = step[:, 0]
    increment = step * omega + .5 * step.square() * alpha
    next_rotation = rotation @ torch.matrix_exp(skew(increment))
    return dict(state,
        rotation=next_rotation,
        omega=omega + step * alpha,
        alpha=alpha,
        root_position=state['root_position'] + linear_step * state['root_velocity'] + .5 * linear_step.square() * acceleration,
        root_velocity=state['root_velocity'] + linear_step * acceleration,
        root_acceleration=acceleration,
        elapsed_time=state['elapsed_time'] + linear_step,
        dt=step)


class JointAccelerationNetwork(nn.Module):
    """Skeleton-message GRU predicting fresh angular/root acceleration.

    ``initialize(rotation, omega, root_position, root_velocity, joints, parents,
    dt, alpha=None, root_acceleration=None, hidden=None, elapsed_time=None)``
    returns a tensor-only resumable state. ``joints`` is [J,3] or [B,J,3] and the
    parent list must describe a single root at index zero, parents before child.
    ``step(state)`` predicts accelerations then integrates. ``rollout(state,n)``
    returns {'states': [state1,...,stateN], 'final_state': stateN}; initial state
    is not included. Zero-initialized output heads give constant velocities.
    """
    architecture = 'joint_acceleration_v1'

    def __init__(self, joint_count=17, hidden=48, angular_accel_scale=30., linear_accel_scale=3.):
        super().__init__()
        if not isinstance(joint_count, int) or isinstance(joint_count, bool) or joint_count < 1:
            raise ValueError('joint_count must be a positive integer')
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        if not math.isfinite(angular_accel_scale) or not math.isfinite(linear_accel_scale) or min(angular_accel_scale, linear_accel_scale) <= 0:
            raise ValueError('Acceleration scales must be finite and positive')
        self.config = dict(joint_count=joint_count, hidden=hidden,
                           angular_accel_scale=angular_accel_scale, linear_accel_scale=linear_accel_scale)
        self.joint_count, self.hidden = joint_count, hidden
        self.angular_accel_scale = angular_accel_scale
        self.linear_accel_scale = linear_accel_scale
        self.encoder = nn.Sequential(nn.Linear(22, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Tanh())
        self.message = nn.Sequential(nn.Linear(hidden + 6, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Tanh())
        self.gru = nn.GRUCell(22 + hidden, hidden)
        self.angular_head = nn.Linear(hidden, 3)
        self.linear_head = nn.Linear(hidden, 3)
        for head in [self.angular_head, self.linear_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def features(self, state):
        rotation = state['rotation']
        batch, joint_count = rotation.shape[:2]
        return torch.cat((rotation.reshape(batch, joint_count, 9),
                          state['omega'] / 5., state['alpha'] / self.angular_accel_scale,
                          state['rest_offsets'] / state['length_scale'],
                          state['root_velocity'][:, None].expand(-1, joint_count, -1) / 3.,
                          state['elapsed_time'][:, None].expand(-1, joint_count, -1) / 2.), -1)

    def initialize(self, rotation, omega, root_position, root_velocity, joints, parents, dt,
                   alpha=None, root_acceleration=None, hidden=None, elapsed_time=None):
        if rotation.ndim != 4 or rotation.shape[1:] != (self.joint_count, 3, 3) or not rotation.is_floating_point():
            raise ValueError('rotation must be floating [B,J,3,3] with configured joint count')
        batch = rotation.shape[0]
        if batch < 1:
            raise ValueError('Nonempty initial state required')
        param = self.angular_head.weight
        if rotation.device != param.device or rotation.dtype != param.dtype:
            raise ValueError('Network and initial state must share device/dtype')
        parents = torch.as_tensor(parents, device=rotation.device)
        if parents.shape != (self.joint_count,) or parents.dtype == torch.bool or parents.is_floating_point() or parents.is_complex():
            raise ValueError('parents must be integer [J]')
        parent_list = parents.tolist()
        if parent_list[0] != -1 or any(p < 0 or p >= j for j, p in enumerate(parent_list[1:], 1)):
            raise ValueError('Parents must describe one root at 0 and precede children')
        parents = parents.long()
        joints = torch.as_tensor(joints, device=rotation.device, dtype=rotation.dtype)
        if joints.shape == (self.joint_count, 3):
            joints = joints[None].expand(batch, -1, -1)
        if joints.shape != (batch, self.joint_count, 3):
            raise ValueError('joints must be [J,3] or [B,J,3]')
        _finite('joints', joints)
        offsets = torch.stack([joints[:, j] - joints[:, p] if p >= 0 else torch.zeros_like(joints[:, j])
                               for j, p in enumerate(parent_list)], 1)
        length_scale = offsets.norm(dim=-1).amax(dim=1).clamp_min(1e-6)[:, None, None]
        adjacency = torch.eye(self.joint_count, device=rotation.device, dtype=rotation.dtype)
        for j, p in enumerate(parent_list):
            if p >= 0:
                adjacency[j, p] = adjacency[p, j] = 1
        adjacency = adjacency / adjacency.sum(-1, keepdim=True)
        state = dict(rotation=rotation.clone(), omega=omega.clone(),
                     alpha=torch.zeros_like(omega) if alpha is None else alpha.clone(),
                     root_position=root_position.clone(), root_velocity=root_velocity.clone(),
                     root_acceleration=torch.zeros_like(root_velocity) if root_acceleration is None else root_acceleration.clone(),
                     elapsed_time=rotation.new_zeros(batch, 1) if elapsed_time is None else elapsed_time.clone(),
                     joints=joints.clone(), parents=parents.clone(), rest_offsets=offsets.clone(),
                     length_scale=length_scale.clone(), adjacency=adjacency[None].expand(batch, -1, -1).clone(),
                     dt=_step_size(dt, rotation).clone())
        _state_contract(state)
        identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype).expand_as(rotation)
        if not torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-4, rtol=1e-4) or bool((torch.linalg.det(rotation) <= 0).any()):
            raise ValueError('Initial rotation must contain proper orthonormal frames')
        if hidden is not None:
            if hidden.shape != (batch, self.joint_count, self.hidden) or hidden.device != rotation.device or hidden.dtype != rotation.dtype:
                raise ValueError('hidden must be [B,J,H] with state device/dtype')
            _finite('hidden', hidden)
        state['hidden'] = self.encoder(self.features(state)) if hidden is None else hidden.clone()
        return state

    def step(self, state):
        rotation, omega = _state_contract(state)
        batch, joint_count = rotation.shape[:2]
        if joint_count != self.joint_count or state['hidden'].shape != (batch, joint_count, self.hidden):
            raise ValueError('State dimensions do not match network')
        adjacency, hidden = state['adjacency'], state['hidden']
        neighbors = torch.cat((torch.bmm(adjacency, hidden),
                               (torch.bmm(adjacency, omega) - omega) / 5.,
                               (torch.bmm(adjacency, state['rest_offsets']) - state['rest_offsets']) / state['length_scale']), -1)
        message = self.message(neighbors)
        memory = self.gru(torch.cat((self.features(state), message), -1).reshape(batch*joint_count, -1),
                          hidden.reshape(batch*joint_count, -1)).reshape(batch, joint_count, self.hidden)
        alpha = self.angular_accel_scale * self.angular_head(memory).tanh()
        root_acceleration = self.linear_accel_scale * self.linear_head(memory[:, 0]).tanh()
        next_state = integrate_joint_state(state, alpha, root_acceleration)
        return dict(next_state, hidden=memory)

    def rollout(self, state, steps):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        states = []
        for _ in range(steps):
            state = self.step(state)
            states.append(state)
        return dict(states=states, final_state=state)
