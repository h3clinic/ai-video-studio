"""Bind-corrected articulation of the existing TripoSR Gaussian dog.

The motion comes from the released, learned 27-joint quadruped controller.
This module is a MANUAL, mesh-specific rig/retarget adapter, not learned rigging,
new diffusion weights, a motion capture replay, or a procedural gait. In
particular the inferred dog's partly joined forelegs remain a geometry defect.

Rest-relative world bone rotations are converted to parent-relative FK frames.
In default ``relative`` mode the exact guidance returns the original mesh, even
though that mesh and the controller use different asymmetric bind poses.
The inverse-bind principle is standard; see Kirin 2609.01823v1, Sec. 3.3,
Eqs. 6--8. We do NOT implement its SMAL fitting or pretrained motion diffusion.
"""
import numpy as np
import torch
from scipy.sparse import coo_matrix
from .controller_rig import from_to, DualSurfaceSkinner
from .gaussian3d import forward_kinematics


JOINT_NAMES = (
    'Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'HeadSite',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandSite',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandSite',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFootSite',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFootSite',
    'Tail', 'Tail1', 'Tail1Site')
PARENTS = (-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 9, 3, 11, 12, 13, 14,
           0, 16, 17, 18, 0, 20, 21, 22, 0, 24, 25)
# Bone directions used for world-space retargeting; leaf frames inherit parent.
CHILD = (3, 2, 3, 4, 5, -1, 7, 8, 9, 10, -1, 12, 13, 14, 15, -1,
         17, 18, 19, -1, 21, 22, 23, -1, 25, 26, -1)
LEG_GROUPS = ((6, 7, 8, 9, 10), (11, 12, 13, 14, 15),
              (16, 17, 18, 19), (20, 21, 22, 23))
ALIGNED_LIMB_BONES = frozenset((7, 8, 9, 12, 13, 14, 16, 17, 18, 20, 21, 22))


def _landmarks(vertices):
    # Inspected XYZ mesh projections and horizontal cross-sections. The dog is
    # Y-up, faces approximately -X/+Z, and is NOT in a symmetric standing pose.
    # These anchors are only a provisional anatomical estimate for this asset.
    template = vertices.new_tensor([
        [.24,.51,-.055], [.13,.52,-.025], [-.03,.54,.025],
        [-.24,.575,.105], [-.35,.655,.195], [-.395,.58,.31],
        [-.23,.48,-.055], [-.32,.37,-.005], [-.365,.205,.025],
        [-.31,.105,.055], [-.255,.06,.08],
        [-.17,.48,.145], [-.13,.36,.145], [-.13,.21,.14],
        [-.18,.065,.11], [-.20,.015,.10],
        [.25,.45,-.20], [.31,.30,-.215], [.285,.20,-.25], [.23,.095,-.23],
        [.285,.455,.055], [.30,.29,.04], [.345,.205,.025], [.313,.095,.04],
        [.36,.50,-.035], [.44,.49,0.], [.515,.46,.025]])
    reference_low = vertices.new_tensor([-.4946, -.00052179, -.2799791])
    reference_high = vertices.new_tensor([.5449, .7638, .3687])
    low, high = vertices.amin(0), vertices.amax(0)
    return low+(template-reference_low)/(reference_high-reference_low)*(high-low)


def build_dog_rig(vertices, faces, smoothing_steps=8):
    """Return a 27-joint bind rig and graph-smoothed shared-vertex weights.

    Inputs can be CPU/GPU tensors; graph smoothing is CPU scipy and outputs
    return to the input device. Only scale/translation variants of THIS mesh
    are supported. The heuristic must not silently serve as a general dog rig.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not torch.isfinite(vertices).all():
        raise ValueError('Expected finite V x 3 dog vertices')
    if len(vertices) < 4 or bool(((vertices.amax(0)-vertices.amin(0)) <= 1e-6).any()):
        raise ValueError('Dog mesh must have nonzero XYZ extent')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected integer triangle faces')
    if len(faces) and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
        raise ValueError('Triangle vertex index out of range')
    if not isinstance(smoothing_steps, int) or not 0 <= smoothing_steps <= 100:
        raise ValueError('Invalid smoothing iteration count')
    joints = _landmarks(vertices)
    height = vertices[:, 1].max()-vertices[:, 1].min()
    starts = joints.clone()
    ends = joints.clone()
    for index, child in enumerate(CHILD):
        if child >= 0:
            ends[index] = joints[child]
    # Pelvis and spine must not all compete for the same long torso segment.
    ends[0] = joints[1]
    delta = ends-starts
    t = ((vertices[:, None]-starts)*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-12)
    closest = starts+t.clamp(0, 1)[..., None]*delta
    distance = (vertices[:, None]-closest).square().sum(-1)
    logits = -distance/(.068*height).square()
    for group in LEG_GROUPS:
        shoulder_height = joints[group[0], 1]
        gate = torch.sigmoid((shoulder_height-vertices[:, 1]-.025*height)/(.045*height))
        logits[:, list(group)] += torch.log(gate.clamp_min(1e-8))[:, None]
    # Leaf-joint skinning would introduce a separate arbitrary pivot at the toe.
    # The incoming hand/ankle segment owns its full paw; leaf IDs remain in FK.
    for index, child in enumerate(CHILD):
        if child < 0:
            logits[:, index] = -1e6
    weights = torch.softmax(logits, dim=-1)
    if smoothing_steps and len(faces):
        f = faces.detach().cpu().numpy()
        edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
        edges = np.concatenate((edges, edges[:, ::-1]))
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(len(vertices), len(vertices))).tocsr()
        graph.data[:] = 1.
        degree = np.asarray(graph.sum(axis=1)).reshape(-1, 1)
        value = weights.detach().cpu().numpy()
        for _ in range(smoothing_steps):
            average = graph@value/np.maximum(degree, 1)
            average = np.where(degree > 0, average, value)
            value = .7*value+.3*average
        weights = torch.as_tensor(value, device=vertices.device, dtype=vertices.dtype)
        weights /= weights.sum(-1, keepdim=True)
    forward = joints[3]-joints[0]
    forward[1] = 0
    forward /= forward.norm()
    right = torch.linalg.cross(vertices.new_tensor([0., 1., 0.]), forward)
    source_to_asset = torch.stack((right, vertices.new_tensor([0., 1., 0.]), forward), dim=1)
    paw_anchors = []
    for leaf in (10, 15, 19, 23):
        # Estimate a material sole point from vertices near this target paw.
        # It is not an all-splat minimum and never modifies individual points.
        distance = (vertices-joints[leaf]).square().sum(-1)
        near = vertices[distance.topk(min(80, len(vertices)), largest=False).indices]
        low = near[:, 1].quantile(.15)
        bottom = near[near[:, 1] <= low]
        paw_anchors.append(bottom.mean(0))
    return dict(joints=joints, parents=PARENTS, joint_names=JOINT_NAMES,
                vertex_weights=weights, source_to_asset=source_to_asset,
                paw_sole_anchors=torch.stack(paw_anchors),
                floor=float(vertices[:, 1].min()),
                scope='Manual mesh-specific bind rig; learned source motion only',
                limitations=['Inferred asymmetric running bind pose, not neutral anatomy',
                             'Some foreleg geometry is joined; skinning cannot invent separation',
                             'No solved foot contact, collisions, or anatomical guarantee'])


class DogRetargeter:
    """Stateless bind-corrected adapter for controller LOCAL 27-joint positions.

    step(local_pose, root) consumes controller-relative joints and the matching
    world-minus-local root vector. Source speed/root use controller units;
    `scale` is asset torso length / source torso length. A later scene transform
    adds its own placement scale. No angle damping, sinusoidal gait, previous
    motion packet or future video is used.

    ``aligned`` instead maps target bind directions to current source bone
    directions. It deliberately reposes the input mesh at t=0; unlike relative
    mode it need not retain the input's lifted/bent leg bias in an idle pose.
    ``aligned_legs`` applies that change only below the shoulders and at the
    hind hips; torso, head, tail and shoulder retain reference-relative frames.
    Optional sole-landmark grounding is an explicit root-height heuristic.
    """
    def __init__(self, rig, guidance, scale=None, mode='relative', ground_paws=False):
        self.rig = rig
        if mode not in ('relative', 'aligned', 'aligned_legs'):
            raise ValueError('Unknown dog retarget mode')
        self.mode = mode
        self.ground_paws = bool(ground_paws)
        device, dtype = rig['joints'].device, rig['joints'].dtype
        self.basis = rig['source_to_asset'].to(device=device, dtype=dtype)
        self.reference = torch.as_tensor(guidance, device=device, dtype=dtype).clone()
        self._validate(self.reference)
        if not torch.allclose(self.basis.T@self.basis, torch.eye(3, device=device, dtype=dtype), atol=1e-5):
            raise ValueError('Source-to-asset basis must be orthonormal')
        if float(torch.linalg.det(self.basis)) <= 0:
            raise ValueError('Source-to-asset basis must be proper')
        source_length = (self.reference[3]-self.reference[0]).norm()
        if float(source_length) < 1e-6:
            raise ValueError('Degenerate source torso')
        target_length = (rig['joints'][3]-rig['joints'][0]).norm()
        self.scale = float(target_length/source_length) if scale is None else float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid retarget scale')
        self.reference = self.reference@self.basis.T

    @staticmethod
    def _validate(source):
        if source.shape != (27, 3) or not torch.isfinite(source).all():
            raise ValueError('Expected finite 27 x 3 source pose')

    def step(self, source, root):
        source = torch.as_tensor(source, device=self.reference.device, dtype=self.reference.dtype)
        root = torch.as_tensor(root, device=self.reference.device, dtype=self.reference.dtype)
        self._validate(source)
        if root.shape != (3,) or not torch.isfinite(root).all():
            raise ValueError('Expected finite root 3-vector')
        source = source@self.basis.T
        identity = torch.eye(3, device=source.device, dtype=source.dtype)
        global_rotation = []
        for index, child in enumerate(CHILD):
            if child < 0:
                rotation = global_rotation[PARENTS[index]]
            else:
                aligned = self.mode == 'aligned' or (self.mode == 'aligned_legs' and index in ALIGNED_LIMB_BONES)
                rest = (self.rig['joints'][child]-self.rig['joints'][index]
                        if aligned else
                        self.reference[child]-self.reference[index])
                current = source[child]-source[index]
                if min(float(rest.norm()), float(current.norm())) < 1e-7:
                    rotation = identity if index == 0 else global_rotation[PARENTS[index]]
                else:
                    rotation = from_to(rest, current)
            global_rotation.append(rotation)
        global_rotation = torch.stack(global_rotation)
        local = torch.stack([global_rotation[index] if parent < 0 else
                             global_rotation[parent].T@global_rotation[index]
                             for index, parent in enumerate(PARENTS)])
        shift = (root@self.basis.T+source[0]-self.reference[0])*self.scale
        transforms = forward_kinematics(self.rig['joints'], PARENTS, local)
        soles = []
        for anchor, owner in zip(self.rig['paw_sole_anchors'], (9, 14, 18, 22)):
            soles.append(transforms[owner, :3, :3]@(anchor-self.rig['joints'][owner])+
                         transforms[owner, :3, 3]+shift)
        soles = torch.stack(soles)
        grounding = 0.
        if self.ground_paws:
            # One root correction, based only on rig sole landmarks. This
            # doesn't enforce four contacts, stop skating, fix leg lengths,
            # repair fused geometry or constitute a dynamics solver.
            grounding = float(self.rig['floor']-soles[:, 1].min())
            shift = shift.clone()
            shift[1] += grounding
            soles[:, 1] += grounding
        posed_joints = transforms[:, :3, 3]+shift
        length_error = torch.stack([
            ((posed_joints[index]-posed_joints[parent]).norm()-
             (self.rig['joints'][index]-self.rig['joints'][parent]).norm()).abs()
            for index, parent in enumerate(PARENTS) if parent >= 0]).max()
        diagnostics = dict(bone_length_max_error=float(length_error),
                           source_to_asset_scale=self.scale,
                           paw_joint_y=posed_joints[[10, 15, 19, 23], 1].detach().cpu().tolist(),
                           paw_sole_y=soles[:, 1].detach().cpu().tolist(),
                           root_grounding_delta_y=grounding,
                           retarget_mode=self.mode,
                           paw_grounding_heuristic=self.ground_paws,
                           contact_solver=False,
                           scope='New controller predictions retargeted to mesh-specific bind pose')
        return local, shift, posed_joints, diagnostics


__all__ = ['build_dog_rig', 'DogRetargeter', 'DualSurfaceSkinner', 'JOINT_NAMES', 'PARENTS']
