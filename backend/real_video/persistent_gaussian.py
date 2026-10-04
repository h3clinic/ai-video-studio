"""Causal recurrent planar Gaussian state. Not verified object/occlusion memory."""
import torch
from torch import nn


def unit(direction):
    norm = direction.norm(dim=1, keepdim=True)
    fallback = torch.zeros_like(direction); fallback[:, 0] = 1
    return torch.where(norm > 1e-6, direction/norm.clamp_min(1e-6), fallback)


def normalize_fields(fields):
    return torch.cat((fields[:, :2], fields[:, 2:4].clamp(-8, -1),
                      unit(fields[:, 4:6]), fields[:, 6:9].clamp(-1, 1)), 1)


def tracked_fields_to_state(fields):
    """Old causal flow-analysis fields B,9,Gh,Gw -> height-normalized state."""
    from .representation import decode_state
    b, _, gh, gw = fields.shape
    if gh != gw:
        raise ValueError('Legacy analysis conversion expects a square grid')
    state = decode_state(fields)
    values = torch.cat(((state['p'][..., :2]+.5)/64,
                        (state['scale'][..., :2]/64).log(), state['R'][..., :2, 0],
                        state['color']*2-1), -1)
    return values.transpose(1, 2).reshape(b, 9, gh, gw)


def wan_fields_to_state(fields, height, width):
    """Wan Gaussian attributes -> same explicit state; no RGB conversion."""
    from .wan_gaussian import gaussian_state
    b, _, gh, gw = fields.shape
    state = gaussian_state(fields, height, width)
    values = torch.cat(((state['centre']+.5)/height, (state['scale']/height).log(),
                        state['axes'][..., :, 0], state['colour']*2-1), -1)
    return values.transpose(1, 2).reshape(b, 9, gh, gw)


def state_bytes(state):
    return sum(t.numel()*t.element_size() for t in state.values())


class PersistentGaussianDynamics(nn.Module):
    """Indexed Gaussian geometry + velocity + 16 learned memory values per slot.

    Spatial convolution uses grid-index neighbours, not recovered 3D neighbours.
    No historical frames, future clip latent, or diffusion network in step().
    Coordinates and widths use image-height units; dt=1 is the training ~8fps step.
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.hidden = hidden
        self.initializer = nn.Sequential(nn.Conv2d(27, 32, 3, padding=1), nn.SiLU(),
                                         nn.Conv2d(32, hidden, 3, padding=1), nn.Tanh())
        self.gates = nn.Conv2d(11+hidden, 2*hidden, 3, padding=1)
        self.candidate = nn.Conv2d(11+hidden, hidden, 3, padding=1)
        self.action = nn.Sequential(nn.Conv2d(hidden, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 8, 1))
        nn.init.zeros_(self.action[-1].weight); nn.init.zeros_(self.action[-1].bias)

    @staticmethod
    def features(field):
        return torch.cat((field[:, :2], field[:, 2:4]+4, field[:, 4:]), 1)

    def initialize(self, first, second, dt=1.0):
        if first.shape != second.shape or first.ndim != 4 or first.shape[1] != 9 or dt <= 0:
            raise ValueError('Expected two equal B,9,H,W Gaussian states and positive dt')
        first, second = normalize_fields(first), normalize_fields(second)
        memory = self.initializer(torch.cat((self.features(first), self.features(second), second-first), 1))
        return dict(fields=second.clone(), velocity=((second[:, :2]-first[:, :2])/dt).clamp(-.08, .08),
                    memory=memory)

    def step(self, state, dt=1.0, reset_memory=False):
        if dt <= 0 or dt > 1:
            raise ValueError('Use 0 < dt <= 1; integrate longer intervals in multiple steps')
        field, velocity, memory = state['fields'], state['velocity'], state['memory']
        if reset_memory:
            memory = torch.zeros_like(memory)
        features = torch.cat((self.features(field), velocity*10), 1)
        reset, update = self.gates(torch.cat((features, memory), 1)).sigmoid().chunk(2, 1)
        candidate = self.candidate(torch.cat((features, reset*memory), 1)).tanh()
        next_memory = (1-update)*memory + update*candidate
        action = self.action(next_memory).tanh()
        next_velocity = (velocity + dt*.01*action[:, :2]).clamp(-.08, .08)
        position = field[:, :2]+dt*next_velocity
        scale = (field[:, 2:4]+dt*.04*action[:, 2:4]).clamp(-8, -1)
        angle = dt*.25*action[:, 4:5]
        u = unit(field[:, 4:6]); c, s = angle.cos(), angle.sin()
        direction = torch.cat((c*u[:, :1]-s*u[:, 1:2], s*u[:, :1]+c*u[:, 1:2]), 1)
        colour = (field[:, 6:9]+dt*.08*action[:, 5:8]).clamp(-1, 1)
        return dict(fields=torch.cat((position, scale, unit(direction), colour), 1),
                    velocity=next_velocity, memory=next_memory)


def render_state(fields, height=64, width=64, radius=5):
    """Direct explicit Gaussian rendering; positions are not grid-anchored."""
    b, _, gh, gw = fields.shape
    x = fields.flatten(2).transpose(1, 2)
    centre = x[..., :2]*height-.5
    scale = x[..., 2:4].clamp(-8, -1).exp()*height
    u = unit(fields[:, 4:6]).flatten(2).transpose(1, 2)
    axes = torch.stack((u, torch.stack((-u[..., 1], u[..., 0]), -1)), -1)
    yy, xx = torch.meshgrid(torch.arange(-radius, radius+1, device=fields.device),
                            torch.arange(-radius, radius+1, device=fields.device), indexing='ij')
    offsets = torch.stack((xx.flatten(), yy.flatten()), -1)
    pixels = centre.floor().long()[:, :, None]+offsets[None, None]
    delta = pixels.to(fields.dtype)-centre[:, :, None]
    local = torch.einsum('bnki,bnij->bnkj', delta, axes)
    weights = (-.5*(local/scale[:, :, None]).square().sum(-1)).exp()
    valid = (pixels[..., 0]>=0)&(pixels[..., 0]<width)&(pixels[..., 1]>=0)&(pixels[..., 1]<height)
    weights = weights*valid
    indices = (pixels[..., 1].clamp(0, height-1)*width+pixels[..., 0].clamp(0, width-1)).reshape(b, -1)
    denominator = fields.new_zeros(b, height*width).scatter_add(1, indices, weights.reshape(b, -1))
    colours = ((x[..., 6:9].clamp(-1, 1)+1)/2)[:, :, None]*weights[..., None]
    numerator = fields.new_zeros(b, height*width, 3).scatter_add(1, indices[..., None].expand(-1, -1, 3), colours.reshape(b, -1, 3))
    return (numerator/denominator.clamp_min(1e-6)[..., None]).reshape(b, height, width, 3).permute(0, 3, 1, 2)


class GaussianMemorySession:
    """Inference-only fixed-shape current state; output history belongs to caller."""
    def __init__(self, model, state):
        self.model = model.eval()
        self.state = {k:v.detach().clone() for k,v in state.items()}
        self.steps = 0

    @torch.no_grad()
    def advance(self, dt=1.0, reset_memory=False):
        self.state = self.model.step(self.state, dt, reset_memory)
        self.steps += 1
        return self.state['fields'].detach().clone()

    def snapshot(self):
        return dict(state={k:v.detach().cpu().clone() for k,v in self.state.items()}, steps=self.steps)

    def restore(self, packet):
        device = next(self.model.parameters()).device
        expected = {k:tuple(v.shape) for k,v in self.state.items()}
        if {k:tuple(v.shape) for k,v in packet['state'].items()} != expected:
            raise ValueError('Snapshot shape mismatch')
        if not all(torch.isfinite(v).all() for v in packet['state'].values()):
            raise ValueError('Nonfinite snapshot')
        self.state = {k:v.detach().to(device).clone() for k,v in packet['state'].items()}
        self.steps = int(packet['steps'])
