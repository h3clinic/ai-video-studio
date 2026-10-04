"""Deterministic, lifetime-bound coordination of persistent Gaussian parts.

This is an independently implemented proposal/validation protocol, not copied
AgentVideo code, an LLM agent, or a trained video generator. Part labels are
supplied canonical metadata, not inferred anatomy. The memory remains frozen
conditioning: trainable proposal controls keep autograd, but memory/cache
geometry does not become differentiable through this module.

For global alpha coefficients A[p,i] = T[p,i] alpha[p,i], the part recall is
F[k,p] = sum_{i: part[i]=k} A[p,i] f[i]. Summing parts recovers the original
recall. Crucially, we never restart transmittance within a part or normalize its
coverage. This partition is our elementary linear decomposition of the recall
operator in Robust Dreamer (2605.30855v1, Eq. 7), not a reproduction or novelty
claim. Independently generated part images are never cropped or composited.
"""
import torch

from .gaussian_latent_memory import GaussianLatentMemory, generation_vector


class GaussianPartProtocol:
    """Bind one canonical part assignment to one bank identity/lifetime version.

    Rebuild explicitly after any material lifetime changes. A semantic part may
    overlap another in screen space without having conflicting point ownership.
    This class validates proposals; it does not execute writes or grant quality
    acceptance. Callers must revalidate immediately before committing a task.
    """

    schema = 'gaussian_part_proposal_v1'

    def __init__(self, memory, *, ids, generations, part_ids):
        if not isinstance(memory, GaussianLatentMemory):
            raise TypeError('GaussianLatentMemory required')
        memory._identity(ids)
        if memory._generations is None:
            raise ValueError('Lifetime-aware memory bank required')
        self.memory = memory
        self._ids = ids.detach().to(memory.device).long().clone()
        self._generations = generation_vector(generations, memory.count, memory.device)
        if (not isinstance(part_ids, torch.Tensor) or part_ids.shape != (memory.count,)
                or part_ids.dtype not in (torch.int32, torch.int64)
                or bool((part_ids < 0).any())):
            raise ValueError('Nonnegative integer original-row part IDs required')
        self._parts = part_ids.detach().to(memory.device).long().clone()
        self._labels = self._parts.unique(sorted=True)
        self._asset_digest = memory.asset_digest
        self._check_current()

    def _check_current(self):
        # Reuse the bank's identity validation without copying all stored features.
        self.memory._identity(self._ids)
        if (self.memory.asset_digest != self._asset_digest
                or self.memory._generations is None
                or not torch.equal(self.memory._generations, self._generations)):
            raise ValueError('Part protocol has stale asset or material lifetimes')

    def read_parts(self, cache, height, width):
        """Return P,C,H,W features and P,H,W coverage in sorted part-ID order.

        All parts retain the *global* renderer's visibility coefficients. Empty
        or completely occluded parts remain present as zero maps. Inputs are
        validated even if the global cache contains no visible fragments.
        """
        self._check_current()
        # Validate the complete cache first; part filtering cannot hide a bad row.
        self.memory._cache(cache, height, width, self._ids)
        features, coverage = [], []
        rows = cache['ids'].to(self.memory.device).long()
        weight = cache['weight'].to(self.memory.device)
        for label in self._labels:
            # No new alpha/transmittance computation, no normalization, and keep
            # the original complete row/lifetime binding on each filtered cache.
            part_cache = dict(cache, weight=weight * (self._parts[rows] == label))
            result = self.memory.read(part_cache, height, width, ids=self._ids)
            features.append(result['features'])
            coverage.append(result['coverage'])
        return dict(part_ids=self._labels.clone(), features=torch.stack(features),
                    coverage=torch.stack(coverage), asset_digest=self._asset_digest,
                    scope='Frozen shared-global-alpha Gaussian part recall')

    def _selection(self, part_id, gaussian_ids):
        if type(part_id) is not int or not bool((self._labels == part_id).any()):
            raise ValueError('Unknown integer part ID')
        if (not isinstance(gaussian_ids, torch.Tensor) or gaussian_ids.ndim != 1
                or gaussian_ids.dtype not in (torch.int32, torch.int64)
                or gaussian_ids.numel() < 1
                or gaussian_ids.unique().numel() != gaussian_ids.numel()):
            raise ValueError('Nonempty unique Gaussian ID selection required')
        selected = gaussian_ids.to(self.memory.device).long()
        # Sort/search rather than an M x N equality allocation.
        ordered, row_order = self._ids.sort()
        found = torch.searchsorted(ordered, selected)
        if (bool((found >= len(ordered)).any())
                or not torch.equal(ordered[found.clamp_max(len(ordered)-1)], selected)):
            raise ValueError('Proposal contains unknown Gaussian IDs')
        rows = row_order[found]
        if not bool((self._parts[rows] == part_id).all()):
            raise ValueError('Proposal selects another part ownership')
        return rows

    def proposal(self, part_id, gaussian_ids, controls, *, task_id):
        """Create a typed proposal; controls are M,D and remain differentiable.

        Identity tensors are copied, but controls intentionally retain their
        graph. These controls are inputs to a separately trained adapter, not
        direct mutation of geometry or model weights by a language agent.
        """
        self._check_current()
        proposal = dict(schema=self.schema, task_id=task_id, part_id=part_id,
                        asset_digest=self._asset_digest,
                        gaussian_ids=gaussian_ids.detach().clone() if isinstance(gaussian_ids, torch.Tensor) else gaussian_ids,
                        generations=self._generations.clone(), controls=controls)
        self.validate_proposals([proposal])
        return proposal

    def validate_proposals(self, proposals, *, approved_overlap_ids=None):
        """Validate all proposals atomically without writing to memory.

        Multiple tasks selecting the same Gaussian are rejected unless those
        exact IDs appear in an explicit overlap allowlist. This is a conflict
        acknowledgement only: a separate caller must define how updates merge.
        Returned controls are the original tensors; no detach/silent cast.
        """
        self._check_current()
        if not isinstance(proposals, (list, tuple)) or not proposals:
            raise ValueError('Nonempty proposal sequence required')
        if approved_overlap_ids is None:
            allowed = self._ids.new_empty(0)
        else:
            if (not isinstance(approved_overlap_ids, torch.Tensor)
                    or approved_overlap_ids.ndim != 1
                    or approved_overlap_ids.dtype not in (torch.int32, torch.int64)
                    or approved_overlap_ids.unique().numel() != approved_overlap_ids.numel()):
                raise ValueError('Unique integer overlap acknowledgement IDs required')
            allowed = approved_overlap_ids.to(self.memory.device).long()
            if not bool(torch.isin(allowed, self._ids).all()):
                raise ValueError('Unknown overlap acknowledgement ID')
        selected_rows, results, task_ids = [], [], set()
        required = {'schema', 'task_id', 'part_id', 'asset_digest', 'gaussian_ids', 'generations', 'controls'}
        for item in proposals:
            if not isinstance(item, dict) or set(item) != required or item['schema'] != self.schema:
                raise ValueError('Invalid proposal schema')
            task_id = item['task_id']
            if not isinstance(task_id, str) or not task_id.strip() or task_id in task_ids:
                raise ValueError('Unique nonempty task ID required')
            task_ids.add(task_id)
            if item['asset_digest'] != self._asset_digest:
                raise ValueError('Proposal asset mismatch')
            generation = generation_vector(item['generations'], self.memory.count, self.memory.device)
            if not torch.equal(generation, self._generations):
                raise ValueError('Proposal material lifetime mismatch')
            rows = self._selection(item['part_id'], item['gaussian_ids'])
            control = item['controls']
            if (not isinstance(control, torch.Tensor) or control.ndim != 2
                    or control.shape[0] != len(rows) or control.shape[1] < 1
                    or not control.is_floating_point() or not bool(torch.isfinite(control).all())):
                raise ValueError('Finite floating M,D controls required')
            selected_rows.append(rows)
            results.append(dict(item, original_rows=rows.clone()))
        rows, count = torch.cat(selected_rows).unique(return_counts=True)
        overlaps = self._ids[rows[count > 1]]
        if not bool(torch.isin(overlaps, allowed).all()):
            raise ValueError('Overlapping point selections require explicit acknowledgement')
        return results
