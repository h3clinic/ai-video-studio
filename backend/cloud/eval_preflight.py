"""CPU-only fail-fast validation for the frozen A14B pilot evaluation.

This validates the exact saved pilot, not arbitrary new checkpoints. No base
model loading, CUDA, downloads, or training occurs here.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

REPO = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
REVISION = '596658fd9ca6b7b71d5057529bbf319ecbc61d74'
PILOT_HASHES = {
    'anchor.png': '477b1e33db5e89c1376656b2ab6a0b900b4d3e3eac6ff0166c30d6a296104ec2',
    'low_expert_adapter.pt': '4630bfcbfd2b648361a0e9cafafc78f7788cacc15456e2a5a07efa979f02451d',
    'gaussian_anchor.pt': '9fa40a1918d4cd6a6680b04dd6ceca7ae2c10dc8f46e6da9c420d2f2a7c66695',
}
# Pilot report stage-end RSS samples, NOT measured peaks. 192 GiB is a
# conservative admission floor above the largest observed sample, not an OOM
# guarantee. Container page cache and other processes can reduce headroom.
PILOT_MODEL_LOAD_RSS_BYTES = 173038149632
PILOT_MAX_SAMPLED_RSS_BYTES = 185348444160
MIN_HOST_AVAILABLE_BYTES = 192 * 2**30


def expected_adapter_shapes():
    shapes = {}
    for block in range(40):
        for attention in ('attn1', 'attn2'):
            for projection in ('to_q', 'to_k', 'to_v', 'to_out.0'):
                prefix = f'blocks.{block}.{attention}.{projection}'
                shapes[prefix+'.down'] = (4, 5120)
                shapes[prefix+'.up'] = (5120, 4)
    shapes['gaussian_part_control.encoder.0.weight'] = (64, 17)
    shapes['gaussian_part_control.encoder.0.bias'] = (64,)
    for head in range(4):
        shapes[f'gaussian_part_control.heads.{head}.weight'] = (5120, 64)
    return shapes


def validate_adapter(adapter):
    import torch
    if (not isinstance(adapter, dict) or adapter.get('repo') != REPO
            or adapter.get('revision') != REVISION
            or adapter.get('expert') != 'low_noise_transformer_2'):
        raise ValueError('Adapter backbone/expert metadata mismatch')
    weights = adapter.get('weights')
    shapes = expected_adapter_shapes()
    if not isinstance(weights, dict) or set(weights) != set(shapes):
        raise ValueError('Adapter parameter layout mismatch')
    for name, shape in shapes.items():
        value = weights[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Adapter parameter shape/dtype/value mismatch: {name}')
    return dict(tensors=len(shapes), parameters=sum(v.numel() for v in weights.values()),
                dtype='float32', shape_match='exact; broadcasting forbidden')


def validate_memory(snapshot, anchor_digest):
    import torch
    from real_video.gaussian_latent_memory import GaussianLatentMemory
    if (not isinstance(snapshot, dict)
            or snapshot.get('schema') != 'gaussian_latent_feature_memory_lifetimes_v2'
            or snapshot.get('asset_digest') != anchor_digest
            or snapshot.get('channels') != 16 or snapshot.get('writes') != 1):
        raise ValueError('Gaussian snapshot schema/anchor/configuration mismatch')
    for name, shape, dtype in (
            ('ids', (6240,), torch.int64), ('generations', (6240,), torch.int64),
            ('features', (6240, 16), torch.float32), ('observation_mass', (6240,), torch.float32)):
        value = snapshot.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f'Gaussian snapshot shape/dtype mismatch: {name}')
    if not torch.equal(snapshot['ids'], torch.arange(6240)):
        raise ValueError('Gaussian pilot grid row order mismatch')
    if bool((snapshot['generations'] != 0).any()) or snapshot.get('max_observation_mass') != 32.:
        raise ValueError('Gaussian pilot lifetime/cap mismatch')
    memory = GaussianLatentMemory(snapshot['ids'], 16, anchor_digest,
        generations=snapshot['generations'], max_observation_mass=32.)
    memory.restore(snapshot)  # Existing finite, range and lifetime checks on CPU.
    return dict(schema=memory.schema, gaussians=6240, channels=16, grid=[60, 104])


def validate_assets(directory):
    from PIL import Image
    from real_video.checkpoint_io import load_verified
    directory = Path(directory)
    hashes = {}
    for name, expected in PILOT_HASHES.items():
        path = directory/name
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                value.update(chunk)
        hashes[name] = value.hexdigest()
        if hashes[name] != expected:
            raise ValueError(f'Frozen pilot artifact hash mismatch: {name}')
        if path.suffix == '.pt':
            sidecar = json.loads(path.with_suffix('.pt.sha256.json').read_text(encoding='utf-8'))
            if sidecar.get('sha256') != expected:
                raise ValueError(f'Frozen pilot checkpoint sidecar mismatch: {name}')
    with Image.open(directory/'anchor.png') as image:
        if image.size != (832, 480) or image.mode != 'RGB':
            raise ValueError('Anchor must be 832x480 RGB')
        image.load()  # Check payload decoding now, not after base model loading.
    adapter = validate_adapter(load_verified(directory/'low_expert_adapter.pt'))
    memory = validate_memory(load_verified(directory/'gaussian_anchor.pt'), hashes['anchor.png'])
    return dict(ready=True, hashes=hashes, adapter=adapter, memory=memory,
                scope='Exact immutable pilot artifacts; CPU checks, no base model loaded')


def effective_memory(host_available, *, cgroup_root=Path('/sys/fs/cgroup'),
                     membership_path=Path('/proc/self/cgroup'),
                     mountinfo_path=Path('/proc/self/mountinfo')):
    """Take the minimum host availability and cgroup v1/v2 ancestor headroom.

    Includes the mounted root (important with container cgroup namespaces).
    Unreadable/malformed present accounting files fail closed. A Linux cgroup
    mount without discoverable memory accounting also fails closed.
    """
    if type(host_available) is not int or host_available < 0:
        raise ValueError('Host available memory must be nonnegative integer bytes')
    root = Path(cgroup_root)
    remaining = []
    candidates = {root, root/'memory'}
    memberships = []
    if Path(membership_path).exists():
        for line in Path(membership_path).read_text(encoding='utf-8').splitlines():
            _, controllers, member = line.split(':', 2)
            if not controllers or 'memory' in controllers.split(','):
                memberships.append((controllers, member))
                base = root if not controllers else root/'memory'
                # Root accounting still covers a namespaced/host-relative path.
                parts = Path(member.lstrip('/')).parts
                if '..' not in parts:
                    child = base.joinpath(*parts)
                    while child != base:
                        candidates.add(child)
                        child = child.parent
    # Account for mount roots that differ from the hierarchy root and custom
    # memory-controller mountpoints. /proc paths are kernel-supplied, not shell.
    if Path(mountinfo_path).exists():
        def unescape(value):
            for encoded, decoded in ((r'\040', ' '), (r'\011', '\t'),
                                     (r'\012', '\n'), (r'\134', '\\')):
                value = value.replace(encoded, decoded)
            return value
        for line in Path(mountinfo_path).read_text(encoding='utf-8').splitlines():
            left, right = line.split(' - ', 1)
            fields, fs = left.split(), right.split()
            if fs[0] not in ('cgroup', 'cgroup2'):
                continue
            if fs[0] == 'cgroup' and 'memory' not in ','.join(fs[1:]).split(','):
                continue
            hierarchy_root = PurePosixPath(unescape(fields[3]))
            mountpoint = Path(unescape(fields[4]))
            candidates.add(mountpoint)
            for controllers, member in memberships:
                if bool(controllers) != (fs[0] == 'cgroup'):
                    continue
                membership = PurePosixPath(member)
                if '..' in membership.parts:
                    raise ValueError('Cannot safely resolve cgroup membership path')
                try:
                    relative = membership.relative_to(hierarchy_root)
                except ValueError:
                    # A container cgroup namespace reports paths relative to its
                    # visible mount root rather than the host hierarchy root.
                    relative = membership.relative_to('/')
                child = mountpoint.joinpath(*relative.parts)
                if child != mountpoint and not child.exists():
                    raise RuntimeError('Cannot resolve process cgroup under memory mount')
                while child != mountpoint:
                    candidates.add(child)
                    child = child.parent
    discovered = False
    for path in sorted(candidates):
        for limit_name, current_name in (('memory.max', 'memory.current'),
                                         ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            limit_file, current_file = path/limit_name, path/current_name
            if not limit_file.exists() and not current_file.exists():
                continue
            discovered = True
            limit_text = limit_file.read_text(encoding='utf-8').strip()
            current = int(current_file.read_text(encoding='utf-8').strip())
            if current < 0:
                raise ValueError('Negative cgroup memory use')
            if limit_text == 'max':
                continue
            limit = int(limit_text)
            if limit < 0:
                raise ValueError('Negative cgroup memory limit')
            if limit >= 2**60:  # Linux cgroup-v1 unlimited sentinel.
                continue
            remaining.append(dict(path=str(path), limit_bytes=limit, current_bytes=current,
                                  remaining_bytes=max(0, limit-current)))
    if root.exists() and not discovered:
        raise RuntimeError('Cgroup mount present but memory accounting unavailable')
    effective = min([host_available]+[r['remaining_bytes'] for r in remaining])
    return dict(host_available_bytes=host_available, cgroup_constraints=remaining,
                effective_available_bytes=effective, required_bytes=MIN_HOST_AVAILABLE_BYTES,
                pilot_max_sampled_rss_bytes=PILOT_MAX_SAMPLED_RSS_BYTES,
                scope='Admission headroom, not a measured peak or guarantee against OOM')


def require_host_memory(host_available, **kwargs):
    report = effective_memory(host_available, **kwargs)
    if report['effective_available_bytes'] < MIN_HOST_AVAILABLE_BYTES:
        raise RuntimeError('Requires at least 192 GiB effective available host/container RAM')
    return report
