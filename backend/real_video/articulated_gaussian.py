"""Differentiable batched 3D articulation of persistent surface Gaussians.

Joint rotations drive fixed-length FK; learned convex skinning moves shared mesh
vertices, then immutable face/barycentric attachments transport Gaussian centers
and full covariances. No image-plane controls or NumPy deformation step occurs.
This is a decoder, not a motion generator, and LBS does not prevent collapse,
self-intersection, implausible anatomy, or wrong motion. The fixed support mask
prevents new joint bindings, not mistakes already present in the initial rig.
"""
import torch
from torch import nn


def _finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')


def _indices(name, value, device):
    value = torch.as_tensor(value, device=device)
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f'{name} must contain integer indices')
    return value.long()


def _basis(triangles):
    """Edge-edge-unit-normal basis; preserves unit normal under rigid motion."""
    first = triangles[..., 1, :] - triangles[..., 0, :]
    second = triangles[..., 2, :] - triangles[..., 0, :]
    cross = torch.linalg.cross(first, second, dim=-1)
    area = torch.linalg.vector_norm(cross, dim=-1)
    normal = cross / area.clamp_min(1e-12)[..., None]
    return torch.stack((first, second, normal), -1), area


class BatchedArticulatedGaussian(nn.Module):
    """Surface-aware FK/LBS decoder with trainable, bounded-support weights.

    ``local_rotation`` has shape [B,J,3,3], ``translation`` [B,3]. Optional
    ``joints`` overrides canonical joint positions with [J,3] or [B,J,3]; it
    remains differentiable and is used in both FK and rest-pose bind correction.
    Changing those positions calibrates the rig, not the canonical mesh shape.

    ``weights()`` is [V,J]. ``vertices()`` returns a dictionary with ``vertices``,
    ``joints`` (posed), and ``global_rotation``. ``forward()`` additionally returns
    ``position`` [B,N,3] and ``covariance`` [B,N,3,3]; ``gaussian_index`` can select
    persistent Gaussian IDs by row index without changing their attachment.
    """

    def __init__(self, asset, rig, *, train_skinning=True, max_influences=4,
                 min_support=1e-4, validate_inputs=True):
        super().__init__()
        vertices = torch.as_tensor(asset['mesh_vertices']).clone()
        if not vertices.is_floating_point() or vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError('mesh_vertices must be floating [V,3], V >= 3')
        device, dtype = vertices.device, vertices.dtype
        cast = lambda x: torch.as_tensor(x, device=device, dtype=dtype).clone()
        faces = _indices('mesh_faces', asset['mesh_faces'], device).clone()
        face_id = _indices('face_id', asset['face_id'], device).clone()
        barycentric = cast(asset['barycentric'])
        frames, scales = cast(asset['frame']), cast(asset['scale'])
        joints = cast(rig['joints'])
        weights = cast(rig['vertex_weights'])
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError('mesh_faces must have shape [F,3], F >= 1')
        if bool(((faces < 0) | (faces >= len(vertices))).any()):
            raise ValueError('mesh_faces index out of range')
        if face_id.ndim != 1 or not len(face_id) or bool(((face_id < 0) | (face_id >= len(faces))).any()):
            raise ValueError('face_id must be nonempty [N] in mesh face range')
        n = len(face_id)
        if barycentric.shape != (n, 3) or frames.shape != (n, 3, 3) or scales.shape != (n, 3):
            raise ValueError('Gaussian attachment/frame/scale shapes disagree')
        if joints.ndim != 2 or joints.shape[1] != 3 or not len(joints):
            raise ValueError('joints must have shape [J,3], J >= 1')
        parents = _indices('parents', rig['parents'], device)
        if parents.shape != (len(joints),):
            raise ValueError('Parent count must equal joint count')
        self.parents = tuple(parents.tolist())
        if any(parent < -1 or parent >= j for j, parent in enumerate(self.parents)):
            raise ValueError('Parents must precede children or equal -1')
        if weights.shape != (len(vertices), len(joints)):
            raise ValueError('vertex_weights must have shape [V,J]')
        for name, value in [('mesh_vertices', vertices), ('barycentric', barycentric),
                            ('frame', frames), ('scale', scales), ('joints', joints),
                            ('vertex_weights', weights)]:
            _finite(name, value)
        if bool((barycentric < 0).any()) or not torch.allclose(barycentric.sum(-1), torch.ones(n, device=device, dtype=dtype), atol=1e-5):
            raise ValueError('Barycentrics must form a nonnegative partition of unity')
        if bool((weights < 0).any()) or not torch.allclose(weights.sum(-1), torch.ones(len(vertices), device=device, dtype=dtype), atol=1e-5):
            raise ValueError('Skinning weights must form a nonnegative partition of unity')
        if bool((scales <= 0).any()):
            raise ValueError('Gaussian scales must be positive')
        eye = torch.eye(3, device=device, dtype=dtype)
        if not torch.allclose(frames @ frames.transpose(-1, -2), eye.expand_as(frames), atol=1e-4, rtol=1e-4) or bool((torch.linalg.det(frames) <= 0).any()):
            raise ValueError('Canonical frames must be proper orthonormal rotations')
        if not isinstance(max_influences, int) or isinstance(max_influences, bool) or max_influences < 1:
            raise ValueError('max_influences must be a positive integer')
        if not 0 <= min_support <= 1:
            raise ValueError('min_support must lie in [0,1]')
        support = torch.zeros_like(weights, dtype=torch.bool)
        top = weights.topk(min(max_influences, len(joints)), dim=-1).indices
        support.scatter_(1, top, True)
        support &= (weights >= min_support) & (weights > 0)
        # Even an unusually high threshold must retain the strongest initial owner.
        support.scatter_(1, weights.argmax(-1, keepdim=True), True)
        retained = weights * support
        retained = retained / retained.sum(-1, keepdim=True)
        logits = retained.clamp_min(torch.finfo(dtype).tiny).log()
        self.skinning_logits = nn.Parameter(logits, requires_grad=train_skinning)
        rest_basis, rest_area = _basis(vertices[faces])
        active = torch.zeros(len(faces), dtype=torch.bool, device=device)
        active[face_id] = True
        degenerate = rest_area < 1e-12
        if bool((degenerate & active).any()):
            raise ValueError('Degenerate Gaussian-owned rest face')
        rest_basis = torch.where(degenerate[:, None, None], eye, rest_basis)
        covariance = (frames * scales[:, None, :].square()) @ frames.transpose(-1, -2)
        for name, value in [('rest_vertices', vertices), ('faces', faces),
                            ('face_id', face_id), ('barycentric', barycentric),
                            ('rest_covariance', covariance), ('rest_joints', joints),
                            ('support_mask', support), ('initial_weights', retained),
                            ('active_faces', active), ('rest_area', rest_area),
                            ('inverse_basis', torch.linalg.inv(rest_basis))]:
            self.register_buffer(name, value)
        self.unused_degenerate_faces = int(degenerate.sum())
        self.validate_inputs = validate_inputs

    def weights(self):
        """Convex weights; unsupported entries remain identically zero."""
        return self.skinning_logits.masked_fill(~self.support_mask, -torch.inf).softmax(-1)

    def _pose(self, local_rotation, translation, joints):
        j = len(self.parents)
        if local_rotation.ndim != 4 or local_rotation.shape[1:] != (j, 3, 3):
            raise ValueError('local_rotation must have shape [B,J,3,3]')
        if local_rotation.device != self.rest_vertices.device or local_rotation.dtype != self.rest_vertices.dtype:
            raise ValueError('Pose and decoder must have the same device and dtype')
        batch = len(local_rotation)
        if batch < 1:
            raise ValueError('Pose batch must be nonempty')
        if translation is None:
            translation = local_rotation.new_zeros(batch, 3)
        if translation.shape != (batch, 3) or translation.device != local_rotation.device or translation.dtype != local_rotation.dtype:
            raise ValueError('translation must be [B,3] with pose device/dtype')
        joints = self.rest_joints if joints is None else joints
        if joints.shape == (j, 3):
            joints = joints.unsqueeze(0).expand(batch, -1, -1)
        elif joints.shape != (batch, j, 3):
            raise ValueError('joints override must have shape [J,3] or [B,J,3]')
        if joints.device != local_rotation.device or joints.dtype != local_rotation.dtype:
            raise ValueError('Joint override and pose device/dtype must match')
        if self.validate_inputs:
            _finite('local_rotation', local_rotation)
            _finite('translation', translation)
            _finite('joints', joints)
            identity = torch.eye(3, dtype=local_rotation.dtype, device=local_rotation.device)
            if not torch.allclose(local_rotation @ local_rotation.transpose(-1, -2), identity.expand_as(local_rotation), atol=1e-3, rtol=1e-3) or bool((torch.linalg.det(local_rotation) <= 0).any()):
                raise ValueError('local_rotation must contain proper orthonormal rotations')
        return local_rotation, translation, joints

    def vertices(self, local_rotation, translation=None, *, joints=None):
        local_rotation, translation, rest = self._pose(local_rotation, translation, joints)
        rotations, points = [], []
        for j, parent in enumerate(self.parents):
            if parent < 0:
                rotations.append(local_rotation[:, j])
                points.append(rest[:, j] + translation)
            else:
                rotations.append(rotations[parent] @ local_rotation[:, j])
                bone = rest[:, j] - rest[:, parent]
                points.append(points[parent] + (rotations[parent] @ bone[..., None]).squeeze(-1))
        rotation = torch.stack(rotations, 1)
        posed_joints = torch.stack(points, 1)
        offset = posed_joints - (rotation @ rest[..., None]).squeeze(-1)
        w = self.weights()
        matrices = torch.einsum('vj,bjac->bvac', w, rotation)
        shifts = torch.einsum('vj,bjc->bvc', w, offset)
        moved = (matrices @ self.rest_vertices[None, ..., None]).squeeze(-1) + shifts
        return dict(vertices=moved, joints=posed_joints, global_rotation=rotation)

    def forward(self, local_rotation, translation=None, *, joints=None,
                gaussian_index=None, return_strain=False):
        result = self.vertices(local_rotation, translation, joints=joints)
        index = slice(None)
        if gaussian_index is not None:
            index = _indices('gaussian_index', gaussian_index, self.face_id.device)
            if index.ndim != 1 or bool(((index < 0) | (index >= len(self.face_id))).any()):
                raise ValueError('gaussian_index must be [K] within Gaussian row range')
        face_id = self.face_id[index]
        # Evaluate just selected Gaussian-owned faces, not an N x J skinning tensor.
        triangles = result['vertices'][:, self.faces[face_id]]
        basis, area = _basis(triangles)
        if self.validate_inputs and bool((area < 1e-12).any()):
            raise ValueError('Gaussian-owned deformed face collapsed')
        gradient = basis @ self.inverse_basis[face_id]
        position = (triangles * self.barycentric[index][None, ..., None]).sum(-2)
        covariance = gradient @ self.rest_covariance[index] @ gradient.transpose(-1, -2)
        result.update(position=position, covariance=covariance)
        if return_strain:
            active_triangles = result['vertices'][:, self.faces[self.active_faces]]
            active_basis, active_area = _basis(active_triangles)
            active_gradient = active_basis @ self.inverse_basis[self.active_faces]
            result.update(area_ratio=active_area / self.rest_area[self.active_faces],
                          deformation_gradient=active_gradient,
                          singular_values=torch.linalg.svdvals(active_gradient),
                          deformation_determinant=torch.linalg.det(active_gradient))
        return result
