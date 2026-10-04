"""Causal projected-XY acceleration ablation, not 3D generative video.

Uses the residual-velocity network's exact features/GRU/head; only the update
changes. Units are sampled-frame differences, not calibrated physical SI units.
"""
import torch
from .vector_motion_residual import ResidualVelocityNetwork


class AccelerationMotionNetwork(ResidualVelocityNetwork):
    architecture = 'integrated_acceleration_v1'

    def __init__(self, hidden=64, acceleration_scale=.005):
        if acceleration_scale <= 0:
            raise ValueError('Positive acceleration scale required')
        super().__init__(hidden=hidden, residual_scale=acceleration_scale)
        self.acceleration_scale = acceleration_scale
        self.config = dict(hidden=hidden, acceleration_scale=acceleration_scale)

    def step(self, state):
        p, v, h, adjacency = (state[key] for key in ['position','velocity','memory','adjacency'])
        batch, nodes, _ = p.shape
        neighbor = torch.cat((torch.bmm(adjacency,h),
            (torch.bmm(adjacency,p)-p)/state['scale'],
            (torch.bmm(adjacency,v)-v)/state['scale']*20),-1)
        message = self.message(neighbor)
        memory = self.gru(torch.cat((self.features(state),message),-1).reshape(batch*nodes,-1),
            h.reshape(batch*nodes,-1)).reshape(batch,nodes,-1)
        acceleration = self.acceleration_scale*state['scale']*self.head(memory).tanh()
        velocity = v+acceleration
        return dict(state,position=p+velocity,velocity=velocity,acceleration=velocity-v,
                    memory=memory,time=state['time']+1)


def load_acceleration_model(checkpoint):
    if checkpoint.get('architecture') != AccelerationMotionNetwork.architecture:
        raise ValueError('Checkpoint is not integrated acceleration')
    model = AccelerationMotionNetwork(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.eval()
