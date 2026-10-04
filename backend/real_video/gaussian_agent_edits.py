"""Typed, object-independent operations on genuinely persistent XYZ Gaussians.

The LLM chooses an allowed operation; it never supplies executable code or
invents a segmentation. Verified ``PartBinding`` IDs are the authority. RGB
textures require a separate, verified material UV binding to the source asset.
These are deterministic edits/replay, NOT a learned motion or 3D generator.

Math provenance: GaussianEditor (arXiv:2311.14521v2), equations 2, 3 and 6,
motivates covariance transport and semantic ID selection. Our exact rigid
transport is mu'=R(mu-pivot)+pivot+t, Sigma'=R Sigma R^T. The kinematic motion
helper is an engineering control, not that paper's proposed learned model.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from .gaussian_part_ownership import PartEdit, apply_transactions, _attribute


IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
ZERO = (0., 0., 0.)
MAX_OPERATIONS = 128
MAX_COORDINATE = 10_000.


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _vector(value, name, size=3):
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f'{name} must have {size} components')
    return tuple(_finite(v, name) for v in value)


def _unit(value, name):
    value = _finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f'{name} must be in [0,1]')
    return value


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Explicit source asset SHA256 required')
    return value


def _transpose(matrix):
    return tuple(zip(*matrix))


def _matvec(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _matmul(a, b):
    return tuple(tuple(sum(x*y for x, y in zip(row, col)) for col in zip(*b)) for row in a)


def _rotation(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError('Rotation must be 3x3')
    r = tuple(_vector(row, 'rotation') for row in value)
    gram = _matmul(_transpose(r), r)
    determinant = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
                   - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
                   + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if any(abs(gram[i][j]-IDENTITY[i][j]) > 1e-6 for i in range(3) for j in range(3)) or abs(determinant-1) > 1e-6:
        raise ValueError('Rotation must be proper SO(3), not scale, shear or reflection')
    return r


@dataclass(frozen=True)
class Operation:
    agent_id: str
    part_id: str
    expected_revision: int

    def __post_init__(self):
        if not all(isinstance(x, str) and 0 < len(x) <= 160 for x in (self.agent_id, self.part_id)):
            raise ValueError('Explicit bounded agent and part IDs required')
        if type(self.expected_revision) is not int or self.expected_revision < 0:
            raise ValueError('Nonnegative revision required')


@dataclass(frozen=True)
class RigidTransform(Operation):
    rotation: tuple = IDENTITY
    translation: tuple = ZERO
    pivot: tuple = ZERO

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'rotation', _rotation(self.rotation))
        object.__setattr__(self, 'translation', _vector(self.translation, 'translation'))
        object.__setattr__(self, 'pivot', _vector(self.pivot, 'pivot'))
        if any(abs(x) > MAX_COORDINATE for x in self.translation+self.pivot):
            raise ValueError('Rigid controls exceed bounded scene coordinates')


@dataclass(frozen=True)
class Recolour(Operation):
    rgb: tuple = (1., 1., 1.)
    strength: float = 1.

    def __post_init__(self):
        super().__post_init__()
        rgb = _vector(self.rgb, 'RGB')
        if any(not 0 <= c <= 1 for c in rgb):
            raise ValueError('RGB must be normalized [0,1]')
        object.__setattr__(self, 'rgb', rgb)
        object.__setattr__(self, 'strength', _unit(self.strength, 'strength'))


@dataclass(frozen=True)
class TexturePaint(Operation):
    texture_id: str = ''
    strength: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if (not isinstance(self.texture_id, str) or not 0 < len(self.texture_id) <= 160
                or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.texture_id)):
            raise ValueError('A broker-resolved texture ID is required, not a file path')
        object.__setattr__(self, 'strength', _unit(self.strength, 'strength'))


@dataclass(frozen=True)
class TextureImage:
    """RGB array already loaded by trusted code; source checksum is provenance."""
    rgb: tuple
    source_sha256: str

    def __post_init__(self):
        _hash(self.source_sha256)
        if not isinstance(self.rgb, (tuple, list)) or not 1 <= len(self.rgb) <= 4096:
            raise ValueError('Texture height must be 1..4096')
        width = len(self.rgb[0])
        if not 1 <= width <= 4096 or len(self.rgb) * width > 4_194_304:
            raise ValueError('Texture exceeds bounded RGB pixel budget')
        rows = []
        for row in self.rgb:
            if len(row) != width:
                raise ValueError('Texture rows must be rectangular')
            pixels = tuple(_vector(pixel, 'texture RGB') for pixel in row)
            if any(not 0 <= c <= 1 for pixel in pixels for c in pixel):
                raise ValueError('Texture RGB must be normalized [0,1]')
            rows.append(pixels)
        object.__setattr__(self, 'rgb', tuple(rows))


@dataclass(frozen=True)
class TextureBinding:
    """Trusted, material-fixed UVs, keyed by persistent IDs, not raster rows.

    UV convention is u=left->right, v=top->bottom, normalized [0,1]. A generated
    image/contact sheet does not itself establish these correspondences.
    """
    part_id: str
    asset_sha256: str
    gaussian_uv: Mapping
    binding_verified: bool = False

    def __post_init__(self):
        _hash(self.asset_sha256)
        if type(self.binding_verified) is not bool or not isinstance(self.gaussian_uv, Mapping):
            raise ValueError('Explicit verified flag and UV mapping required')
        copied = {}
        for gid, value in self.gaussian_uv.items():
            if type(gid) is not int or gid < 0:
                raise ValueError('Texture mapping requires persistent integer IDs')
            uv = _vector(value, 'UV', 2)
            if any(not 0 <= x <= 1 for x in uv):
                raise ValueError('UV must be in [0,1]; wrapping is not inferred')
            copied[gid] = uv
        object.__setattr__(self, 'gaussian_uv', MappingProxyType(copied))


def operation_from_dict(value):
    """Strict JSON boundary. No filenames, code, IDs, masks or new geometry."""
    if not isinstance(value, Mapping):
        raise ValueError('Operation must be an object')
    common = {'kind', 'agent_id', 'part_id', 'expected_revision'}
    kinds = {'rigid_transform': (RigidTransform, {'rotation', 'translation', 'pivot'}),
             'recolour': (Recolour, {'rgb', 'strength'}),
             'texture_paint': (TexturePaint, {'texture_id', 'strength'})}
    kind = value.get('kind')
    if not isinstance(kind, str) or kind not in kinds:
        raise ValueError('Unsupported operation kind')
    cls, extra = kinds[kind]
    if not common <= set(value) or not set(value) <= common | extra:
        raise ValueError('Missing or unknown operation fields')
    if kind == 'recolour' and 'rgb' not in value:
        raise ValueError('Explicit target RGB required')
    return cls(**{key: val for key, val in value.items() if key != 'kind'})


def _sample(texture, uv):
    pixels = texture.rgb
    x, y = uv[0] * (len(pixels[0])-1), uv[1] * (len(pixels)-1)
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = min(x0+1, len(pixels[0])-1), min(y0+1, len(pixels)-1)
    wx, wy = x-x0, y-y0
    return tuple((1-wy)*((1-wx)*pixels[y0][x0][c]+wx*pixels[y0][x1][c])
                 + wy*((1-wx)*pixels[y1][x0][c]+wx*pixels[y1][x1][c]) for c in range(3))


def apply_agent_operations(state, manifest, operations, *, textures=None, texture_bindings=None, asset_sha256=None):
    """Return immutable edited state, new manifest and audit receipts, atomically.

    Multiple operations on one part compose in input order into ONE ownership
    transaction. Unrelated IDs are unchanged. Failure does not mutate input.
    Texture/ownership bindings are supplied by trusted host code, never the LLM.
    RGB arithmetic uses the stored colour space; it does not infer lighting.
    """
    operations = tuple(operations)
    if not 1 <= len(operations) <= MAX_OPERATIONS:
        raise ValueError('Provide 1..128 bounded operations')
    textures, texture_bindings = textures or {}, texture_bindings or {}
    by_part = {p.part_id: p for p in manifest.parts}
    pending, owners, receipts = {}, {}, []
    for operation in operations:
        if type(operation) not in (RigidTransform, Recolour, TexturePaint):
            raise ValueError('Only typed, allowlisted operations are executable')
        part = by_part.get(operation.part_id)
        if (part is None or part.binding_verified is not True or not part.gaussian_ids or part.protected is not False
                or not part.agent_id or part.agent_id != operation.agent_id):
            raise ValueError('Verified, editable, agent-owned part required')
        if operation.expected_revision != manifest.revision:
            raise ValueError('Stale revision')
        if any(gid not in state for gid in part.gaussian_ids):
            raise ValueError('Binding references absent Gaussian')
        part_updates = pending.setdefault(part.part_id, {})
        owners[part.part_id] = operation.agent_id
        binding, texture = None, None
        if isinstance(operation, TexturePaint):
            texture = textures.get(operation.texture_id)
            binding = texture_bindings.get(part.part_id)
            if not isinstance(texture, TextureImage) or not isinstance(binding, TextureBinding):
                raise ValueError('Host-loaded texture and grounded UV binding required')
            if (not binding.binding_verified or binding.part_id != part.part_id
                    or binding.asset_sha256 != _hash(asset_sha256)
                    or set(binding.gaussian_uv) != set(part.gaussian_ids)):
                raise ValueError('Texture binding is unverified, foreign or incomplete')
        for gid in part.gaussian_ids:
            current = {**state[gid], **part_updates.get(gid, {})}
            updates = part_updates.setdefault(gid, {})
            if isinstance(operation, RigidTransform):
                if 'frame' in current or 'normal' in current:
                    raise ValueError('Use canonical covariance-only edit state to avoid stale frame/normal aliases')
                _attribute('position', current['position'])
                _attribute('covariance', current['covariance'])
                offset = tuple(current['position'][i]-operation.pivot[i] for i in range(3))
                rotated = _matvec(operation.rotation, offset)
                updates['position'] = tuple(rotated[i]+operation.pivot[i]+operation.translation[i] for i in range(3))
                if any(not math.isfinite(x) or abs(x) > MAX_COORDINATE for x in updates['position']):
                    raise ValueError('Result exceeds bounded scene coordinates')
                transported = _matmul(_matmul(operation.rotation, current['covariance']), _transpose(operation.rotation))
                # Restore exact symmetry after floating point matrix products.
                updates['covariance'] = tuple(tuple((transported[i][j]+transported[j][i])/2 for j in range(3)) for i in range(3))
            else:
                _attribute('colour', current['colour'])
                target = operation.rgb if isinstance(operation, Recolour) else _sample(texture, binding.gaussian_uv[gid])
                updates['colour'] = tuple((1-operation.strength)*current['colour'][i]+operation.strength*target[i] for i in range(3))
        receipts.append(dict(agent_id=operation.agent_id, part_id=operation.part_id,
                             operation=type(operation).__name__, gaussian_ids=tuple(part.gaussian_ids),
                             texture_sha256=texture.source_sha256 if texture else None,
                             learned_generation=False))
    edits = [PartEdit(owners[pid], pid, manifest.revision, updates) for pid, updates in pending.items()]
    output, updated_manifest = apply_transactions(state, manifest, edits)
    return output, updated_manifest, tuple(receipts)


def rigid_motion_at(agent_id, part_id, expected_revision, *, time_seconds,
                    velocity=ZERO, acceleration=ZERO, angular_velocity=ZERO,
                    angular_acceleration=ZERO, pivot=ZERO):
    """Absolute-from-canonical constant-acceleration motion, bounded to 60 sec.

    p_offset=t*v+0.5*t^2*a; R=Exp([t*omega+0.5*t^2*alpha]_x).
    Angular vectors must be collinear: this makes the exponential exact for
    this fixed-axis acceleration, instead of silently approximating changing
    noncommuting axes. Fresh controls/recorded transforms can supply arbitrary
    time-varying motion through RigidTransform. Not autonomous motion inference.
    Evaluate each timestamp from the same canonical state, never cumulatively.
    """
    t = _finite(time_seconds, 'time_seconds')
    if not 0 <= t <= 60:
        raise ValueError('Timestamp must be in [0,60] seconds')
    v, a, w, alpha = [_vector(x, name) for x, name in
                      ((velocity, 'velocity'), (acceleration, 'acceleration'),
                       (angular_velocity, 'angular velocity'), (angular_acceleration, 'angular acceleration'))]
    cross = (w[1]*alpha[2]-w[2]*alpha[1], w[2]*alpha[0]-w[0]*alpha[2], w[0]*alpha[1]-w[1]*alpha[0])
    if sum(x*x for x in cross) > 1e-18 * max(1., sum(x*x for x in w)*sum(x*x for x in alpha)):
        raise ValueError('Changing angular axes require sampled proper rotations, not this exact fixed-axis helper')
    rv = tuple(t*w[i] + .5*t*t*alpha[i] for i in range(3))
    angle = math.hypot(*rv)
    if not math.isfinite(angle):
        raise ValueError('Motion controls overflow')
    if angle == 0:
        rotation = IDENTITY
    else:
        x, y, z = (n/angle for n in rv)
        skew = ((0., -z, y), (z, 0., -x), (-y, x, 0.))
        square = _matmul(skew, skew)
        rotation = tuple(tuple(IDENTITY[i][j]+math.sin(angle)*skew[i][j]+(1-math.cos(angle))*square[i][j]
                               for j in range(3)) for i in range(3))
    translation = tuple(t*v[i]+.5*t*t*a[i] for i in range(3))
    return RigidTransform(agent_id, part_id, expected_revision, rotation, translation, pivot)


def state_from_asset(asset):
    """Adapt explicit-ID tensor/array XYZ assets without inventing segmentation.

    This intentionally emits only authoritative covariance renderer fields.
    Caller separately provides verified part bindings. Planar field packets and
    merged assets without globally unique IDs must not be relabeled as 3D here.
    ``frame``/``scale`` may define covariance, but are not stale output aliases.
    """
    def values(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        return value.tolist() if hasattr(value, 'tolist') else value

    ids = values(asset['ids'])
    if not isinstance(ids, (tuple, list)) or not ids or any(type(gid) is not int or not 0 <= gid < 2**63 for gid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Existing unique, nonnegative persistent IDs required')
    arrays = {key: values(asset[key]) for key in ('position', 'colour', 'opacity')}
    if 'covariance' in asset:
        arrays['covariance'] = values(asset['covariance'])
    else:
        frames, scales = values(asset['frame']), values(asset['scale'])
        if len(frames) != len(ids) or len(scales) != len(ids):
            raise ValueError('Frame/scale count mismatch')
        covariance = []
        for frame, scale in zip(frames, scales):
            r, s = _rotation(frame), _vector(scale, 'scale')
            if any(x <= 0 for x in s):
                raise ValueError('Positive Gaussian scales required')
            covariance.append(_matmul(tuple(tuple(r[i][j]*s[j]**2 for j in range(3)) for i in range(3)), _transpose(r)))
        arrays['covariance'] = covariance
    if any(len(array) != len(ids) for array in arrays.values()):
        raise ValueError('One attribute row per ID required')
    result = {}
    for index, gid in enumerate(ids):
        attrs = {key: array[index] for key, array in arrays.items()}
        for field, value in attrs.items():
            _attribute(field, value)
        result[gid] = attrs
    # Reuse the transaction layer's recursive copy/freeze, without a revision.
    from .gaussian_part_ownership import OwnershipManifest
    return apply_transactions(result, OwnershipManifest(0, ()), ())[0]


def state_to_arrays(state):
    """Portable renderer/checkpoint arrays, sorted by explicit ID, no file I/O.

    Includes no stale frame aliases. Values remain float64 until a renderer
    explicitly selects a dtype; conversion to float32 is not bit-exact replay.
    """
    import numpy as np
    if not isinstance(state, Mapping) or not state or any(type(gid) is not int or not 0 <= gid < 2**63 for gid in state):
        raise ValueError('Nonempty int64-ID Gaussian state required')
    ids = sorted(state)
    result = {'ids': np.asarray(ids, dtype=np.int64)}
    for key in ('position', 'covariance', 'colour', 'opacity'):
        for gid in ids:
            _attribute(key, state[gid][key])
        result[key] = np.asarray([state[gid][key] for gid in ids], dtype=np.float64)
    return result


def canonical_state_from_npz(path, *, expected_sha256, part_id, agent_id,
                             standard_deviation=.010, opacity=.9, revision=0):
    """Load bounded existing XYZ samples, binding their WHOLE asset to one agent.

    Returns ``(state, manifest, provenance)``. The original object_edit_worker
    assigns canonical isotropic sigma=.010 and opacity=.9 before world scaling;
    those defaults are declared here, not learned or recovered from the NPZ.
    A file without IDs obtains canonical IDs from its *persisted* sample order.
    The material identity is (file SHA256, canonical ID), not per-frame pixels.
    Whole-asset membership is verified by construction; anatomy, surface parts,
    hidden shape, UVs, temporal scene tracks and contact are NOT verified.
    """
    import hashlib
    import io
    from pathlib import Path
    import zipfile
    import numpy as np
    from .gaussian_part_ownership import OwnershipManifest, PartBinding

    expected_sha256 = _hash(expected_sha256)
    sigma, alpha = _finite(standard_deviation, 'standard_deviation'), _unit(opacity, 'opacity')
    if not 1e-8 <= sigma <= 100:
        raise ValueError('Canonical standard deviation must be in [1e-8,100]')
    # Reuse the strict identity/revision checks before loading anything.
    Operation(agent_id, part_id, revision)
    path = Path(path)
    if path.suffix.lower() != '.npz' or not path.is_file() or path.stat().st_size > 128*1024**2:
        raise ValueError('Bounded NPZ source asset required')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Canonical asset checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) > 32 or sum(item.file_size for item in members) > 256*1024**2:
            raise ValueError('Expanded asset exceeds bounded archive budget')
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        if 'position' not in data or 'colour' not in data:
            raise ValueError('Existing XYZ position and colour arrays required')
        position, colour = data['position'], data['colour']
        if (position.ndim != 2 or position.shape[1] != 3 or not 1 <= len(position) <= 100_000
                or position.dtype.kind != 'f' or colour.shape != position.shape or colour.dtype.kind != 'f'
                or not np.isfinite(position).all() or not np.isfinite(colour).all()
                or np.abs(position).max() > MAX_COORDINATE or (colour < 0).any() or (colour > 1).any()):
            raise ValueError('Finite bounded XYZ and normalized RGB samples required')
        explicit = 'ids' in data
        ids = data['ids'] if explicit else np.arange(len(position), dtype=np.int64)
        if explicit and (ids.dtype.kind not in 'iu' or ids.shape != (len(position),)):
            raise ValueError('One existing integer ID per sample required')
        asset = dict(ids=ids, position=position, colour=colour,
                     covariance=np.broadcast_to(np.eye(3)*sigma*sigma, (len(position), 3, 3)),
                     opacity=np.full(len(position), alpha))
        state = state_from_asset(asset)
    binding = PartBinding(part_id, tuple(state), binding_verified=True, agent_id=agent_id)
    manifest = OwnershipManifest(revision, (binding,))
    provenance = dict(source_sha256=expected_sha256, canonical_points=len(state),
                      id_origin='existing IDs' if explicit else 'persisted source sample row order',
                      identity_namespace='(source_sha256, canonical_id)', binding_scope='whole canonical asset only',
                      standard_deviation=sigma, opacity=alpha,
                      covariance_origin='explicit isotropic canonical footprint, not recovered geometry uncertainty',
                      learned_generation=False, temporal_scene_correspondence_verified=False,
                      texture_uv_verified=False, anatomical_part_segmentation_verified=False)
    return state, manifest, provenance
