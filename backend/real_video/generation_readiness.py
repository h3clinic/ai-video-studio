"""Fail-closed claim boundary for the Gaussian video research program.

This is an evidence validator, not a learned quality judge or model. Hashes bind
attestations to files; they do not prove that an evaluator's assertions are true.
No external code, model, shell command, or paid service is executed here.
"""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path


NATIVE_STAGE = 'native_gaussian_state_generation'
REQUIRED_CHECKS = {
    'trained_gaussian_dynamics': ('generator_checkpoint', 'generator_code'),
    'causal_rollout': ('generator_checkpoint', 'generator_code', 'rollout', 'video'),
    'persistent_identity': ('canonical_state', 'rollout', 'video'),
    'memory_read_write': ('generator_checkpoint', 'generator_code', 'rollout', 'video'),
    'held_out_generalization': ('generator_checkpoint', 'generator_code', 'evaluation'),
    'visual_quality': ('video', 'visual_review'),
    'resource_accounting': ('generator_checkpoint', 'rollout', 'video', 'metrics'),
}
NATIVE_FLAGS = {
    'model_emits_gaussian_updates': True,
    'trained_state_writer': True,
    'persistent_state_read': True,
    'generated_state_write': True,
    'future_conditioned': False,
    'test_used_for_training': False,
    'test_used_for_selection': False,
}


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _resolve(root, descriptor):
    if not isinstance(descriptor, Mapping):
        raise ValueError('Missing artifact descriptor')
    path = (root/descriptor['path']).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Artifact must be a file inside this project')
    expected = descriptor.get('sha256')
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)
            or _hash(path) != expected):
        raise ValueError('Artifact hash is absent, invalid, or stale')
    return path


def classify_mechanism(mechanism):
    """Classify declared wiring, independently of quality or desired labels."""
    if not isinstance(mechanism, Mapping):
        return 'unverified'
    source, domain = mechanism.get('motion_source'), mechanism.get('output_domain')
    if source == 'recorded':
        return 'recorded_motion_transfer'
    if source == 'none':
        return 'static_reconstruction'
    if source == 'scripted':
        return 'scripted_gaussian_animation' if domain == 'gaussian_state' else 'scripted_rgb_animation'
    if source == 'pretrained_skeletal':
        return 'pretrained_skeletal_gaussian_animation'
    if source == 'wan_rgb':
        return ('gaussian_memory_conditioned_rgb_generation'
                if mechanism.get('persistent_state_read') is True
                and mechanism.get('generated_state_write') is True
                and mechanism.get('trained_state_writer') is True
                else 'wan_rgb_gaussian_conditioning')
    if source == 'learned_gaussian' and domain == 'gaussian_state':
        return ('future_conditioned_gaussian_fitting' if mechanism.get('future_conditioned') is True
                else 'claimed_native_gaussian_state_generation')
    return 'unverified'


def evaluate_readiness(manifest, root=Path('.')):
    """Validate readiness for native generation; absent proof never passes.

    ``artifacts`` maps role to {path, sha256}. ``checks`` maps REQUIRED_CHECKS
    names to identically shaped JSON-file descriptors. A check document needs
    schema_version=1, check_id, status='pass', evaluator={kind,name}, observation,
    and subjects={artifact_role: exact_sha256} for every required role. Native
    visual approval also revalidates the existing articulation_quality review.
    Other stage contracts remain the coordinator's responsibility.
    """
    root = Path(root).resolve()
    blockers, valid_artifacts, decisions = [], {}, {}

    def block(code, detail):
        blockers.append({'code': code, 'detail': detail})

    if not isinstance(manifest, Mapping):
        manifest = {}
    mechanism = manifest.get('mechanism', {})
    if not isinstance(mechanism, Mapping):
        mechanism = {}
    classification = classify_mechanism(mechanism)
    if manifest.get('schema_version') != 1:
        block('manifest_schema', 'An explicit schema_version=1 manifest is required.')
    if manifest.get('requested_stage') != NATIVE_STAGE:
        block('unsupported_stage', 'This validator only promotes native Gaussian-state generation.')
    if classification != 'claimed_native_gaussian_state_generation':
        block('mechanism_mismatch', f'Observed/declarative mechanism is {classification}, not native Gaussian-state generation.')
    for flag, expected in NATIVE_FLAGS.items():
        if mechanism.get(flag) is not expected:
            block(f'mechanism.{flag}', f'{flag} must explicitly equal {expected}.')
    artifacts = manifest.get('artifacts', {})
    if not isinstance(artifacts, Mapping):
        artifacts = {}
    for role, descriptor in artifacts.items():
        try:
            if not isinstance(role, str) or not role:
                raise ValueError('Artifact role must be a nonempty string')
            valid_artifacts[role] = _resolve(root, descriptor)
        except (KeyError, OSError, TypeError, ValueError) as error:
            block(f'artifact.{role}', str(error))
    checks = manifest.get('checks', {})
    if not isinstance(checks, Mapping):
        checks = {}
    for check_id, roles in REQUIRED_CHECKS.items():
        reasons = []
        try:
            evidence = json.loads(_resolve(root, checks.get(check_id)).read_text(encoding='utf-8'))
            if not isinstance(evidence, Mapping):
                raise ValueError('Check document must be an object')
            if evidence.get('schema_version') != 1 or evidence.get('check_id') != check_id:
                reasons.append('Wrong schema or check identity')
            if evidence.get('status') != 'pass':
                reasons.append('An explicit pass is absent')
            evaluator = evidence.get('evaluator', {})
            if (not isinstance(evaluator, Mapping) or evaluator.get('kind') not in ('human', 'agent', 'instrumented_test')
                    or not isinstance(evaluator.get('name'), str) or not evaluator['name'].strip()):
                reasons.append('Evaluator identity is absent')
            if not isinstance(evidence.get('observation'), str) or not evidence['observation'].strip():
                reasons.append('Observation is absent')
            subjects = evidence.get('subjects', {})
            if not isinstance(subjects, Mapping):
                subjects = {}
            for role in roles:
                if role not in valid_artifacts or subjects.get(role) != artifacts[role].get('sha256'):
                    reasons.append(f'Missing or stale subject binding: {role}')
            # Bind every additional canonical object too, not just the first.
            if check_id == 'persistent_identity':
                for role in artifacts:
                    if isinstance(role, str) and role.startswith('canonical_state_') and (role not in valid_artifacts or subjects.get(role) != artifacts[role].get('sha256')):
                        reasons.append(f'Missing or stale subject binding: {role}')
            if check_id == 'visual_quality' and not reasons:
                from .articulation_quality import decide_review
                review = json.loads(valid_artifacts['visual_review'].read_text(encoding='utf-8'))
                decision = decide_review(review, valid_artifacts['video'])
                if not decision['accepted'] or review.get('claim_type') != 'conditional_motion_generation':
                    reasons.append('The exact generated video does not have accepted articulation review')
        except (KeyError, OSError, TypeError, ValueError) as error:
            reasons.append(str(error))
        decisions[check_id] = {'passed': not reasons, 'reasons': reasons}
        if reasons:
            block(f'check.{check_id}', '; '.join(reasons))
    ready = not blockers
    return {'schema_version': 1, 'classification': NATIVE_STAGE if ready else classification,
            'requested_stage': manifest.get('requested_stage'), 'ready': ready,
            'status': 'ready' if ready else 'blocked', 'blockers': blockers, 'checks': decisions,
            'verified_artifact_count': len(valid_artifacts),
            'limitations': ['This validates evidence bindings and declared wiring, not scientific truth.',
                            'Readiness is not novelty, unrestricted autonomy, or a compute-savings claim.']}


def current_manifest(root=Path('.')):
    """Describe the existing v3 diagnostic, without manufacturing evidence.

    This does not guess the mechanism of newer experiments. Source hashes are
    cross-checked against v3's own run report; changed code becomes unverified.
    New generators must supply their own manifest and evaluation records.
    """
    root = Path(root).resolve()
    base = Path('artifacts/real_video')
    run = base/'gaussian_action_video/v3'
    paths = {
        'generator_checkpoint': base/'neural_motion_controller/v1/controller.ts',
        'generator_code': Path('real_video/generate_gaussian_actions.py'),
        'canonical_state': base/'hunyuan_gaussian/v5_full_paint/gaussian_fitted.pt',
        'canonical_state_dog': base/'gaussian_dog_insertion/v1/dog/cat_asset.pt',
        'rollout': run/'retargeted.pt', 'video': run/'gaussian_actions_7s.mp4',
        'metrics': run/'metrics.json',
        'wan_adapter_checkpoint': base/'wan_temporal_memory/v3/temporal_adapter.pt',
        'wan_adapter_code': Path('real_video/wan_temporal_control.py'),
        'latent_memory_code': Path('real_video/gaussian_latent_memory.py'),
    }
    artifacts = {role: {'path': path.as_posix(), 'sha256': _hash(root/path)}
                 for role, path in paths.items() if (root/path).is_file()}
    notes, code_matches = [], False
    try:
        recorded = json.loads((root/paths['metrics']).read_text(encoding='utf-8'))
        source_hashes = recorded['source_code_sha256']
        expected = ('generate_gaussian_actions.py', 'controller_rig.py',
                    'dog_articulation.py', 'quadruped_controller.py')
        code_matches = all(source_hashes.get(name) == _hash(root/'real_video'/name) for name in expected)
    except (OSError, KeyError, ValueError, TypeError):
        notes.append('The recorded v3 run or its source provenance is unavailable.')
    if not code_matches:
        notes.append('Current action code does not match all recorded v3 source hashes; mechanism is unverified.')
    return {'schema_version': 1, 'requested_stage': NATIVE_STAGE,
            'mechanism': {'motion_source': 'pretrained_skeletal' if code_matches else 'unverified',
                          'output_domain': 'gaussian_state',
                          'model_emits_gaussian_updates': False, 'trained_state_writer': False,
                          'persistent_state_read': False, 'generated_state_write': False,
                          'future_conditioned': False, 'test_used_for_training': False,
                          'test_used_for_selection': False},
            'artifacts': artifacts, 'checks': {}, 'notes': notes + [
                'Persistent appearance is rendered through a heuristic rig; the neural model emits joint poses.',
                'Wan temporal conditioning and latent-feature memory exist separately; no trained closed-loop Gaussian writer is integrated.',
                'The existing v3 visual-quality rejection is not promoted by this audit.']}
