"""Explicit architecture dispatch; different motion equations cannot silently mix."""
from .vector_motion_network import VectorMotionNetwork


def load_motion_model(checkpoint):
    architecture=checkpoint.get('architecture','damped_velocity_v1')
    if 'model_config' in checkpoint and 'config' in checkpoint and checkpoint['model_config']!=checkpoint['config']:
        raise ValueError('Conflicting model configuration aliases')
    config=checkpoint['model_config'] if 'model_config' in checkpoint else checkpoint['config']
    if architecture=='damped_velocity_v1':
        model=VectorMotionNetwork(**config)
    elif architecture=='residual_velocity_v1':
        from .vector_motion_residual import ResidualVelocityNetwork
        model=ResidualVelocityNetwork(**config)
    else:
        raise ValueError('Unknown motion architecture: '+str(architecture))
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.eval()
