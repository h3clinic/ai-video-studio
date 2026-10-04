"""Fail-closed, per-part Gaussian edit transactions; not a segmentation model.

Bindings must come from verified geometry/correspondence, never from an LLM's
semantic guess. A transaction edits existing IDs only, atomically. Creating new
geometry or transferring ownership requires a separately reviewed new manifest.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType
from collections.abc import Mapping
import math

FIELDS = frozenset({'position', 'covariance', 'colour', 'opacity'})
RELATIONS = frozenset({'object', 'part_of', 'detached_from', 'contact', 'occludes'})


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class PartBinding:
    part_id: str
    gaussian_ids: tuple
    relation: str = 'object'
    parent_id: str | None = None
    protected: bool = False
    binding_verified: bool = False
    allowed_fields: frozenset = FIELDS
    agent_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'gaussian_ids', tuple(self.gaussian_ids))
        object.__setattr__(self, 'allowed_fields', frozenset(self.allowed_fields))
        if not self.part_id or self.relation not in RELATIONS:
            raise ValueError('invalid part or relation')
        if any(type(i) is not int or i < 0 for i in self.gaussian_ids):
            raise ValueError('Gaussian IDs must be nonnegative integers')
        if len(set(self.gaussian_ids)) != len(self.gaussian_ids):
            raise ValueError('duplicate Gaussian IDs')
        if not self.allowed_fields <= FIELDS:
            raise ValueError('unknown editable field')


@dataclass(frozen=True)
class OwnershipManifest:
    revision: int
    parts: tuple

    def __post_init__(self):
        object.__setattr__(self, 'parts', tuple(self.parts))
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('invalid revision')
        by_id = {p.part_id: p for p in self.parts}
        if len(by_id) != len(self.parts):
            raise ValueError('duplicate part')
        seen = set()
        for part in self.parts:
            if seen.intersection(part.gaussian_ids):
                raise ValueError('overlapping Gaussian ownership')
            seen.update(part.gaussian_ids)
            if part.parent_id is not None and part.parent_id not in by_id:
                raise ValueError('unknown parent')
            ancestry = {part.part_id}
            parent = part.parent_id
            while parent is not None:
                if parent in ancestry:
                    raise ValueError('cyclic part graph')
                ancestry.add(parent)
                parent = by_id[parent].parent_id


@dataclass(frozen=True)
class PartEdit:
    agent_id: str
    part_id: str
    expected_revision: int
    updates: Mapping

    def __post_init__(self):
        object.__setattr__(self, 'updates', _freeze(self.updates))


def replacement_scope(manifest, target):
    """Explicit ownership closure only; no automatic red-peel equivalence."""
    by_id = {p.part_id: p for p in manifest.parts}
    if target not in by_id:
        raise ValueError('unknown target')
    selected = {target}
    while True:
        expanded = selected | {p.part_id for p in manifest.parts
                               if p.parent_id in selected and
                               p.relation in {'part_of', 'detached_from'}}
        if expanded == selected:
            break
        selected = expanded
    return tuple(p.part_id for p in manifest.parts if p.part_id in selected)


def inherit_densification(manifest, *, agent_id, part_id, expected_revision,
                          existing_ids, child_to_parent, lineage_verified=False):
    """Preserve ownership when a trusted fitter creates new Gaussian IDs.

    Metadata only: this does not create geometry or validate reconstruction.
    The fitter supplies persistent IDs (including tombstones), not row indices.
    Parent removal/pruning is deliberately unsupported in this append-only step.
    An LLM-proposed bounding box cannot set ``lineage_verified``.
    """
    if lineage_verified is not True or expected_revision != manifest.revision:
        raise ValueError('unverified lineage or stale revision')
    known = set(existing_ids)
    if any(type(gid) is not int or gid < 0 for gid in known):
        raise ValueError('persistent integer IDs required')
    if any(gid not in known for p in manifest.parts for gid in p.gaussian_ids):
        raise ValueError('manifest references absent persistent ID')
    part = next((p for p in manifest.parts if p.part_id == part_id), None)
    if part is None or not part.binding_verified or part.protected or not part.agent_id or part.agent_id != agent_id:
        raise ValueError('agent cannot densify this part')
    if not isinstance(child_to_parent, Mapping) or not 1 <= len(child_to_parent) <= 100000:
        raise ValueError('bounded nonempty fitter lineage required')
    records = []
    for child, parent in child_to_parent.items():
        if type(child) is not int or child < 0 or child in known:
            raise ValueError('new persistent ID collides with active or retired state')
        if type(parent) is not int or parent not in part.gaussian_ids:
            raise ValueError('parent outside assigned part')
        records.append(dict(child_id=child, parent_id=parent, part_id=part_id,
                            agent_id=agent_id, source_revision=manifest.revision))
    expanded = replace(part, gaussian_ids=part.gaussian_ids + tuple(child_to_parent))
    result = replace(manifest, revision=manifest.revision+1,
                     parts=tuple(expanded if p.part_id==part_id else p for p in manifest.parts))
    return result, _freeze(records)


def _number(x):
    if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
        raise ValueError('attribute must be finite numeric')
    return float(x)


def _attribute(field, value):
    if field not in FIELDS:
        raise ValueError('unknown Gaussian field')
    if field == 'opacity':
        if not 0 <= _number(value) <= 1:
            raise ValueError('opacity out of range')
        return
    if field in {'position', 'colour'}:
        if len(value) != 3:
            raise ValueError('expected three components')
        values = [_number(v) for v in value]
        if field == 'colour' and any(v < 0 or v > 1 for v in values):
            raise ValueError('colour out of range')
        return
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise ValueError('expected 3x3 covariance')
    a = [[_number(v) for v in row] for row in value]
    if any(not math.isclose(a[i][j], a[j][i], rel_tol=1e-10, abs_tol=1e-12)
           for i in range(3) for j in range(3)):
        raise ValueError('covariance must be symmetric')
    # Cholesky requires strictly positive pivots; zero-scale splats are rejected.
    l = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            pivot = a[i][j] - sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                if pivot <= 0 or not math.isfinite(pivot):
                    raise ValueError('covariance must be positive definite')
                l[i][j] = math.sqrt(pivot)
            else:
                l[i][j] = pivot / l[j][j]


def apply_transactions(state, manifest, edits):
    """Return immutable state and bumped manifest, or reject the whole batch.

    State is ``{integer_id: {position, covariance, colour, opacity, ...}}``.
    Every bound ID must exist; unowned state is retained but cannot be edited.
    Each mutable part requires an explicit assigned agent, verified binding and
    nonempty IDs. Batch revision matching permits independent concurrent edits.
    """
    by_id = {p.part_id: p for p in manifest.parts}
    for part in manifest.parts:
        if any(gid not in state for gid in part.gaussian_ids):
            raise ValueError('binding references absent Gaussian')
    edits = tuple(edits)
    touched = set()
    for edit in edits:
        if edit.expected_revision != manifest.revision:
            raise ValueError('stale revision')
        part = by_id.get(edit.part_id)
        if part is None or not part.binding_verified or not part.gaussian_ids:
            raise ValueError('unbound part')
        if part.protected or not part.agent_id or part.agent_id != edit.agent_id:
            raise ValueError('agent lacks edit permission')
        if not edit.updates:
            raise ValueError('empty transaction')
        for gid, updates in edit.updates.items():
            if type(gid) is not int or gid not in part.gaussian_ids:
                raise ValueError('Gaussian outside assigned part')
            if gid in touched:
                raise ValueError('overlapping batch edits')
            touched.add(gid)
            if not updates or not set(updates) <= part.allowed_fields:
                raise ValueError('protected or unknown field')
            for field, value in updates.items():
                _attribute(field, value)
    output = {gid: dict(attrs) for gid, attrs in state.items()}
    for edit in edits:
        for gid, updates in edit.updates.items():
            output[gid].update(updates)
    return _freeze(output), replace(manifest, revision=manifest.revision + bool(edits))
