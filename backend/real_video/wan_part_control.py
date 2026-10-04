"""Trainable per-part Gaussian recall residuals inside an actual Wan DiT.

Consumes globally composited part feature/coverage maps, not cropped videos.
Caller supplies part ownership, intent embeddings and current geometry; this
module learns none of those automatically. Zero-initialized heads preserve the
base initially. Shared Wan attention still couples parts after injection.
No backbone weights are downloaded here, no future motion is predicted, and
no native Gaussian state is emitted. Compatible tensor shapes are not proof
of pretrained-model quality. See the source-linked part_weight_adapters sweep.
"""
import torch
from torch import nn
import torch.nn.functional as F


def _float(value, ndim, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != ndim
            or not value.is_floating_point() or min(value.shape) < 1
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f'{name} must be a nonempty finite floating {ndim}D tensor')


class PartSpatialControl(nn.Module):
    def __init__(self, dim, channels, intent_dim, *, blocks=(0,), hidden=64):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (dim, channels, intent_dim, hidden)):
            raise ValueError('Positive integer dimensions required')
        if not blocks or len(set(blocks)) != len(blocks) or any(type(n) is not int or n < 0 for n in blocks):
            raise ValueError('Unique nonnegative block indices required')
        self.dim, self.channels, self.intent_dim = dim, channels, intent_dim
        self.blocks = tuple(blocks)
        self.encoder = nn.Sequential(nn.Linear(channels + intent_dim, hidden), nn.SiLU())
        self.heads = nn.ModuleList(nn.Linear(hidden, dim, bias=False) for _ in blocks)
        for head in self.heads:
            nn.init.zeros_(head.weight)
        self.bound = None
        self.bound_validator = None
        self.enabled = True

    def bind(self, features, coverage, intents, grid):
        """Bind B,P,C,T,H,W features and B,P,1,T,H,W coverage at latent rate.

        B,P,K intents are external action embeddings, not generated task text.
        Do not rebind/mutate until backward completes with gradient checkpointing.
        Temporal controls must not contain target/future observations at inference.
        """
        _float(features, 6, 'features')
        _float(coverage, 6, 'coverage')
        _float(intents, 3, 'intents')
        b, p, c, t, h, w = features.shape
        if (c != self.channels or coverage.shape != (b,p,1,t,h,w)
                or intents.shape != (b,p,self.intent_dim)
                or features.device != coverage.device or features.device != intents.device):
            raise ValueError('Part condition shape/device mismatch')
        if len(grid) != 3 or any(type(n) is not int or n < 1 for n in grid) or grid[0] != t:
            raise ValueError('Token grid must preserve condition time')
        if bool((coverage < 0).any()) or bool((coverage.sum(1) > 1.00001).any()):
            raise ValueError('Part coverage must partition one globally composited cache')
        if bool((features.masked_select((coverage == 0).expand_as(features)) != 0).any()):
            raise ValueError('Uncovered part features must be zero')
        self.bound = (features, coverage, intents, tuple(grid))
        self.bound_validator = None

    def bind_memory_frames(self, protocol, caches, height, width, intents, grid):
        """Read one versioned bank through latent-rate view caches, then bind.

        Single-scene bridge. All maps retain one material-version binding and
        are rechecked before each block execution, including checkpoint replay.
        Caches/geometry are supplied controls, NOT predicted future trajectories.
        Caller must rebuild after writes as well as lifetime changes if fresh
        content is needed. This bridge does not accept raw unvalidated maps.
        """
        from .gaussian_part_protocol import GaussianPartProtocol
        if not isinstance(protocol,GaussianPartProtocol) or not caches or len(caches)!=grid[0]:
            raise ValueError('Versioned protocol and one cache per latent time required')
        maps=[protocol.read_parts(cache,height,width) for cache in caches]
        features=torch.stack([m['features'] for m in maps],dim=2)[None]
        coverage=torch.stack([m['coverage'] for m in maps],dim=1)[None,:,None]
        self.bind(features,coverage,intents,grid)
        revision=protocol.memory.revision
        def validate():
            protocol._check_current()
            if protocol.memory.revision != revision:
                raise ValueError('Memory changed after part binding; rebind before execution')
        self.bound_validator=validate

    def forward(self, hidden, head_index):
        if not self.enabled or self.bound is None:
            return hidden
        if self.bound_validator is not None:
            self.bound_validator()
        features, coverage, intents, (t, h, w) = self.bound
        if (type(head_index) is not int or not 0 <= head_index < len(self.heads)
                or hidden.ndim != 3 or hidden.shape[1:] != (t*h*w, self.dim)
                or hidden.device != features.device or hidden.shape[0] != features.shape[0]):
            raise ValueError('Part control/token mismatch; CFG batches must be explicitly bound')
        b,p,c,_,fh,fw = features.shape
        dtype = self.heads[head_index].weight.dtype
        def checked(value, name):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Nonfinite {name}; adapter precision/range exceeded')
            return value
        def pool(x):
            maps=x.permute(0,1,3,2,4,5).reshape(b*p*t,x.shape[2],fh,fw)
            maps=checked(maps.to(dtype),'converted part maps')
            return checked(F.adaptive_avg_pool2d(maps,(h,w)), 'pooled part maps').flatten(2).transpose(1,2).reshape(b,p,t*h*w,-1)
        cov = pool(coverage)
        mean = checked(pool(features)/torch.where(cov > 0,cov,torch.ones_like(cov)), 'normalized features')
        intent = checked(intents.to(dtype),'converted intents')[:,:,None].expand(-1,-1,t*h*w,-1)
        inputs=torch.where(cov > 0,torch.cat((mean,intent),-1),0.)
        encoded = checked(self.encoder(inputs), 'encoded part features')
        residual = checked(self.heads[head_index](encoded), 'part residual')
        delta = checked((residual*cov).sum(1), 'summed residual')
        return checked(hidden + delta.to(hidden.dtype), 'conditioned hidden state')


def attach_part_control(model, control):
    """Register trainable parameters and removable hooks on Diffusers Wan blocks.

    Handles must remain attached during checkpoint recomputation. For Wan2.2's
    A14B high/low experts attach SEPARATE adapters to both transformers; do not
    share trainable state implicitly or apply the TI2V-5B first-frame protocol.
    """
    if not isinstance(control, PartSpatialControl):
        raise TypeError('Expected PartSpatialControl')
    if hasattr(model,'gaussian_part_control') or max(control.blocks) >= len(model.blocks):
        raise ValueError('Duplicate adapter or invalid block index')
    if model.config.num_attention_heads * model.config.attention_head_dim != control.dim:
        raise ValueError('Adapter hidden width differs from backbone')
    model.add_module('gaussian_part_control',control)
    handles=[]
    for j,index in enumerate(control.blocks):
        handles.append(model.blocks[index].register_forward_hook(
            lambda module, inputs, output, head=j: control(output,head)))
    return handles


def install_part_adaptation(model, *, intent_dim, blocks=(0,), hidden=64, rank=4):
    """Freeze one loaded Wan expert and install real attention LoRA + part heads.

    Model loading is intentionally external: callers must pin a licensed model,
    check resources and bind separate high/low experts. Latent channels come
    from out_channels, never concatenated I2V in_channels (36 != 16 for A14B).
    Fresh adapter weights require training. No upstream files are overwritten.
    """
    from .wan_cat_memory import install_lora, LowRankLinear
    if type(rank) is not int or rank < 1:
        raise ValueError('Positive integer LoRA rank required')
    if hasattr(model, 'gaussian_part_control') or any(isinstance(m, LowRankLinear) for m in model.modules()):
        raise ValueError('Adaptation already installed')
    dim=model.config.num_attention_heads*model.config.attention_head_dim
    control=PartSpatialControl(dim,model.config.out_channels,intent_dim,blocks=blocks,hidden=hidden)
    if max(control.blocks)>=len(model.blocks):
        raise ValueError('Invalid block index')
    reference=next(model.parameters())
    control.to(device=reference.device, dtype=torch.float32)
    model.requires_grad_(False)
    names=install_lora(model,rank)
    handles=attach_part_control(model,control)
    return control, handles, dict(latent_channels=model.config.out_channels,
        hidden_width=dim, lora_projections=names,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        pretrained_quality_verified=False, gaussian_native_generation=False)
