"""Architecture identity and exact loaded update equations, not quality tests."""
from io import BytesIO
import unittest

import torch

from real_video.motion_model_io import load_motion_model
from real_video.vector_motion_network import VectorMotionNetwork
from real_video.vector_motion_residual import ResidualVelocityNetwork


def checkpoint(model, architecture=None, config_key='config'):
    packet = dict(model={key: value.detach().clone() for key, value in model.state_dict().items()})
    packet[config_key] = model.config.copy()
    if architecture is not None:
        packet['architecture'] = architecture
    return packet


def tensor_roundtrip(packet):
    stream = BytesIO()
    torch.save(packet, stream)
    stream.seek(0)
    return torch.load(stream, map_location='cpu', weights_only=True)


class MotionModelIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def observations(self, stationary=False):
        ref = torch.tensor([[.1, .2], [.8, .3], [.4, .9], [.7, .7]])
        velocity = torch.tensor([[.003, -.002], [-.001, .004], [.002, .005], [.006, -.003]])
        if stationary:
            velocity.zero_()
        observed = torch.stack([ref + t * velocity for t in range(3)])[None]
        adjacency = torch.eye(4)[None]
        confidence = torch.tensor([[1., .7, .9, .4]])
        return observed, adjacency, confidence

    def assert_roundtrip(self, model, architecture, config_key):
        torch.manual_seed(904)
        with torch.no_grad():
            model.head[-1].weight.normal_(std=.02)
            model.head[-1].bias.normal_(std=.01)
        packet = tensor_roundtrip(checkpoint(model, architecture, config_key))
        loaded = load_motion_model(packet)
        self.assertIs(type(loaded), type(model))
        self.assertFalse(loaded.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
        observed, adjacency, confidence = self.observations()
        with torch.no_grad():
            expected, expected_state = model.rollout(observed, adjacency, 12, confidence)
            actual, actual_state = loaded.rollout(observed, adjacency, 12, confidence)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for key in actual_state:
            torch.testing.assert_close(actual_state[key], expected_state[key], atol=0, rtol=0)

    def test_legacy_untagged_old_checkpoint_strict_roundtrip(self):
        self.assert_roundtrip(VectorMotionNetwork(hidden=8), None, 'config')

    def test_explicit_old_checkpoint_model_config_roundtrip(self):
        self.assert_roundtrip(VectorMotionNetwork(hidden=8), 'damped_velocity_v1', 'model_config')

    def test_residual_checkpoint_dispatches_actual_new_class(self):
        model = ResidualVelocityNetwork(hidden=8, residual_scale=.02)
        loaded = load_motion_model(checkpoint(model, 'residual_velocity_v1'))
        self.assertIs(type(loaded), ResidualVelocityNetwork)
        self.assertEqual(loaded.head[-1].out_features, 2)
        self.assertEqual(loaded.config, {'hidden': 8, 'residual_scale': .02})
        self.assert_roundtrip(model, 'residual_velocity_v1', 'model_config')

    def test_unknown_architecture_is_rejected(self):
        for tag in ['residual_velocity_v999', '', None, 123]:
            packet = checkpoint(VectorMotionNetwork(hidden=8))
            packet['architecture'] = tag
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                load_motion_model(packet)

    def test_conflicting_configuration_aliases_are_rejected(self):
        packet = checkpoint(ResidualVelocityNetwork(hidden=8), 'residual_velocity_v1')
        packet['model_config'] = dict(packet['config'], residual_scale=.5)
        with self.assertRaises(ValueError):
            load_motion_model(packet)
        packet['model_config'] = packet['config'].copy()
        loaded = load_motion_model(packet)
        self.assertIs(type(loaded), ResidualVelocityNetwork)

    def test_missing_and_extra_parameter_keys_are_not_silently_ignored(self):
        packet = checkpoint(VectorMotionNetwork(hidden=8))
        packet['model'].pop(next(iter(packet['model'])))
        with self.assertRaises(RuntimeError):
            load_motion_model(packet)
        packet = checkpoint(VectorMotionNetwork(hidden=8))
        packet['model']['invented_parameter'] = torch.ones(1)
        with self.assertRaises(RuntimeError):
            load_motion_model(packet)

    def test_same_hidden_size_does_not_allow_old_weights_as_new_model(self):
        old = checkpoint(VectorMotionNetwork(hidden=8), 'residual_velocity_v1')
        old['config'] = {'hidden': 8, 'residual_scale': .02}
        with self.assertRaises(RuntimeError):
            load_motion_model(old)
        new = checkpoint(ResidualVelocityNetwork(hidden=8), 'damped_velocity_v1')
        with self.assertRaises(RuntimeError):
            load_motion_model(new)

    def test_zero_residual_is_exact_observed_constant_velocity_not_decay_or_gain(self):
        loaded = load_motion_model(checkpoint(ResidualVelocityNetwork(hidden=8), 'residual_velocity_v1'))
        observed, adjacency, confidence = self.observations()
        state = loaded.initialize(observed, adjacency, confidence)
        initial_velocity = observed[:, 2] - observed[:, 1]
        expected_position = observed[:, 2].clone()
        with torch.no_grad():
            for _ in range(12):
                expected_position = expected_position + initial_velocity
                state = loaded.step(state)
                torch.testing.assert_close(state['velocity'], initial_velocity, atol=0, rtol=0)
                torch.testing.assert_close(state['position'], expected_position, atol=0, rtol=0)

    def test_residual_is_anchored_to_initial_not_preceding_velocity(self):
        model = ResidualVelocityNetwork(hidden=8, residual_scale=.02)
        with torch.no_grad():
            model.head[-1].bias.copy_(torch.tensor([.2, -.4]))
        loaded = load_motion_model(checkpoint(model, 'residual_velocity_v1'))
        state = loaded.initialize(*self.observations())
        # Change preceding velocity only: constant head bias makes its intended
        # anchor independently observable without changing a stored checkpoint.
        state['velocity'] = torch.full_like(state['velocity'], .75)
        expected = state['initial_velocity'] + .02 * state['scale'] * torch.tensor([.2, -.4]).tanh()
        with torch.no_grad():
            next_state = loaded.step(state)
        torch.testing.assert_close(next_state['velocity'], expected, atol=0, rtol=0)

    def test_stationary_observations_and_zero_head_do_not_invent_motion(self):
        loaded = load_motion_model(checkpoint(ResidualVelocityNetwork(hidden=8), 'residual_velocity_v1'))
        observed, adjacency, confidence = self.observations(stationary=True)
        with torch.no_grad():
            output, _ = loaded.rollout(observed, adjacency, 12, confidence)
        torch.testing.assert_close(output, observed[:, 2:3].expand_as(output), atol=0, rtol=0)

    def test_loaded_parameters_do_not_alias_checkpoint_storage(self):
        packet = checkpoint(ResidualVelocityNetwork(hidden=8), 'residual_velocity_v1')
        loaded = load_motion_model(packet)
        expected = {key: value.clone() for key, value in loaded.state_dict().items()}
        for value in packet['model'].values():
            value.fill_(999)
        for key, value in loaded.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
