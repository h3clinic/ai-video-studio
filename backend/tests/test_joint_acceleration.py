import io
import math
import unittest

import torch

from real_video.articulated_gaussian import BatchedArticulatedGaussian
from real_video.joint_acceleration import JointAccelerationNetwork, integrate_joint_state, skew


def example(model, batch=1, dt=.02):
    parameter = next(model.parameters())
    cast = lambda x: torch.as_tensor(x, device=parameter.device, dtype=parameter.dtype)
    joint_count = model.joint_count
    rotation = torch.eye(3, device=parameter.device, dtype=parameter.dtype).expand(batch, joint_count, 3, 3).clone()
    omega = cast([0., 0., .5]).expand(batch, joint_count, 3).clone()
    joints = cast([[float(j), 0., 0.] for j in range(joint_count)])
    position = cast([[.1, .2, .3]]).expand(batch, 3).clone()
    velocity = cast([[.2, -.1, .3]]).expand(batch, 3).clone()
    return model.initialize(rotation, omega, position, velocity, joints, [-1] + list(range(joint_count-1)), dt)


class JointAccelerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(607)
        torch.set_num_threads(4)

    def test_constant_acceleration_analytic_and_no_input_mutation(self):
        model = JointAccelerationNetwork(joint_count=3, hidden=12).double()
        state = example(model, batch=2, dt=.05)
        initial = {k: v.clone() for k, v in state.items()}
        alpha = state['alpha'].clone()
        alpha[..., 2] = .4
        acceleration = torch.tensor([[.3, .2, -.4], [-.1, .5, .2]], dtype=torch.float64)
        current = state
        for _ in range(20):
            current = integrate_joint_state(current, alpha, acceleration)
        expected_position = state['root_position'] + state['root_velocity'] + .5*acceleration
        expected_rotation = state['rotation'] @ torch.matrix_exp(skew(state['omega'] + .5*alpha))
        torch.testing.assert_close(current['root_position'], expected_position, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(current['root_velocity'], state['root_velocity']+acceleration, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(current['rotation'], expected_rotation, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(current['omega'], state['omega']+alpha, atol=1e-12, rtol=1e-12)
        for key in state:
            torch.testing.assert_close(state[key], initial[key], atol=0, rtol=0)

    def test_zero_heads_are_constant_velocity_not_damped(self):
        model = JointAccelerationNetwork(joint_count=3, hidden=12)
        state = example(model, dt=.1)
        result = model.rollout(state, 12)['final_state']
        torch.testing.assert_close(result['root_position'], state['root_position']+1.2*state['root_velocity'])
        torch.testing.assert_close(result['root_velocity'], state['root_velocity'], atol=0, rtol=0)
        torch.testing.assert_close(result['omega'], state['omega'], atol=0, rtol=0)
        expected = state['rotation'] @ torch.matrix_exp(skew(1.2*state['omega']))
        torch.testing.assert_close(result['rotation'], expected, atol=2e-6, rtol=2e-6)

    def test_prescribed_acceleration_can_reverse_velocity_and_oscillate(self):
        model = JointAccelerationNetwork(joint_count=1, hidden=8).double()
        state = example(model, dt=.01)
        state['omega'] = torch.tensor([[[0., 0., 1.]]], dtype=torch.float64)
        angles, velocities = [], []
        # Prescribed forcing is a numerical diagnostic, not a learned oscillator.
        for _ in range(630):
            alpha = torch.zeros_like(state['alpha'])
            alpha[..., 2] = -torch.sin(state['elapsed_time'])
            state = integrate_joint_state(state, alpha)
            angles.append(float(torch.atan2(state['rotation'][0, 0, 1, 0], state['rotation'][0, 0, 0, 0])))
            velocities.append(float(state['omega'][0, 0, 2]))
        self.assertGreater(max(angles), .95)
        self.assertLess(min(angles), -.95)
        self.assertGreater(max(velocities), .95)
        self.assertLess(min(velocities), -.95)

    def test_time_step_refinement_reduces_forcing_error(self):
        model = JointAccelerationNetwork(joint_count=1, hidden=8).double()
        def error(dt):
            state = example(model, dt=dt)
            state['omega'] = torch.tensor([[[0., 0., 1.]]], dtype=torch.float64)
            for _ in range(round(2/dt)):
                alpha = torch.zeros_like(state['alpha'])
                alpha[..., 2] = -torch.sin(state['elapsed_time'])
                state = integrate_joint_state(state, alpha)
            actual = torch.atan2(state['rotation'][0, 0, 1, 0], state['rotation'][0, 0, 0, 0])
            return abs(float(actual)-math.sin(2))
        coarse, fine = error(.1), error(.05)
        self.assertLess(fine, .6*coarse)

    def test_so3_under_noncommuting_changing_acceleration(self):
        model = JointAccelerationNetwork(joint_count=3, hidden=12).double()
        state = example(model, dt=.01)
        for _ in range(60):
            state = integrate_joint_state(state, torch.randn_like(state['alpha'])*3)
        rotation = state['rotation']
        identity = torch.eye(3, dtype=rotation.dtype).expand_as(rotation)
        torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(1, 3, dtype=rotation.dtype), atol=1e-11, rtol=1e-11)

    def test_network_refreshes_acceleration_and_recurrence_gradients(self):
        model = JointAccelerationNetwork(joint_count=3, hidden=12)
        torch.nn.init.normal_(model.angular_head.weight, std=.08)
        torch.nn.init.normal_(model.linear_head.weight, std=.08)
        states = model.rollout(example(model, dt=.05), 6)['states']
        self.assertFalse(torch.allclose(states[0]['alpha'], states[1]['alpha']))
        self.assertFalse(torch.allclose(states[0]['root_acceleration'], states[1]['root_acceleration']))
        loss = states[-1]['rotation'][..., 0, 1].sum() + states[-1]['root_position'].square().sum()
        loss.backward()
        for parameter in [model.angular_head.weight, model.linear_head.weight,
                          model.gru.weight_hh, model.message[0].weight, model.encoder[0].weight]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_integrator_gradients_to_acceleration_and_dt(self):
        model = JointAccelerationNetwork(joint_count=2, hidden=8).double()
        state = example(model)
        alpha = torch.tensor([[[.1, .2, .3], [.2, -.1, .4]]], dtype=torch.float64, requires_grad=True)
        acceleration = torch.tensor([[.2, .1, -.3]], dtype=torch.float64, requires_grad=True)
        dt = torch.tensor(.05, dtype=torch.float64, requires_grad=True)
        result = integrate_joint_state(state, alpha, acceleration, dt)
        (result['rotation'][..., 0, 1].sum()+result['root_position'].sum()).backward()
        for tensor in [alpha, acceleration, dt]:
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.)

    def test_exact_serialized_resume_and_constant_state_bytes(self):
        model = JointAccelerationNetwork(joint_count=3, hidden=12)
        torch.nn.init.normal_(model.angular_head.weight, std=.08)
        torch.nn.init.normal_(model.linear_head.weight, std=.08)
        state = example(model, batch=2)
        shapes = {k: tuple(v.shape) for k, v in state.items()}
        size = lambda s: sum(v.numel()*v.element_size() for v in s.values())
        initial_size = size(state)
        with torch.no_grad():
            state = model.rollout(state, 8)['final_state']
            payload = io.BytesIO()
            torch.save(state, payload)
            payload.seek(0)
            resumed = torch.load(payload, weights_only=True)
            a, b = model.step(state), model.step(resumed)
        self.assertEqual(shapes, {k: tuple(v.shape) for k, v in a.items()})
        self.assertEqual(initial_size, size(a))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)

    def test_rotating_gaussian_has_centripetal_acceleration_without_angular_acceleration(self):
        # Constant angular velocity still produces time-varying point velocity.
        dtype = torch.float64
        vertices = torch.tensor([[1.9, -.1, 0.], [2.1, -.1, 0.], [2., .2, 0.]], dtype=dtype)
        joints = torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=dtype)
        asset = dict(mesh_vertices=vertices, mesh_faces=torch.tensor([[0,1,2]]), face_id=torch.tensor([0]),
                     barycentric=torch.full((1,3), 1/3, dtype=dtype), frame=torch.eye(3, dtype=dtype)[None],
                     scale=torch.full((1,3), .03, dtype=dtype))
        rig = dict(joints=joints, parents=[-1,0], vertex_weights=torch.tensor([[0.,1.]]*3, dtype=dtype))
        decoder = BatchedArticulatedGaussian(asset, rig, train_skinning=False)
        model = JointAccelerationNetwork(joint_count=2, hidden=8).double()
        dt = .001
        omega = torch.tensor([[[0.,0.,0.],[0.,0.,2.]]], dtype=dtype)
        rotation = torch.matrix_exp(skew(-dt*omega))
        state = model.initialize(rotation, omega, torch.zeros(1,3,dtype=dtype), torch.zeros(1,3,dtype=dtype), joints, [-1,0],dt)
        positions = []
        for _ in range(3):
            positions.append(decoder(state['rotation'], state['root_position'])['position'][0,0])
            state = integrate_joint_state(state)
        acceleration = (positions[2]-2*positions[1]+positions[0])/dt**2
        torch.testing.assert_close(acceleration, torch.tensor([-4.,0.,0.],dtype=dtype), atol=2e-6, rtol=2e-6)

    def test_invalid_dt_state_and_parent_tree_rejected(self):
        model = JointAccelerationNetwork(joint_count=3, hidden=12)
        state = example(model)
        for dt in [0., -.1, float('nan'), torch.tensor([.1,.2])]:
            with self.assertRaises(ValueError):
                integrate_joint_state(state, dt=dt)
        with self.assertRaises(ValueError):
            integrate_joint_state(state, torch.zeros(1,2,3))
        args = (state['rotation'], state['omega'], state['root_position'], state['root_velocity'], state['joints'])
        with self.assertRaises(ValueError):
            model.initialize(*args,[-1,2,0],.1)
        with self.assertRaises(ValueError):
            model.initialize(*args,[-1,-1,0],.1)
        with self.assertRaises(ValueError):
            model.rollout(state,-1)
        self.assertEqual(model.rollout(state,0)['states'],[])


if __name__ == '__main__':
    unittest.main()
