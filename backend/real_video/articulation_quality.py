"""Fail-closed, evidence-linked visual review; never an automatic quality score.

These functions read and hash media, but never write artifacts or supply a
reviewer's observations. Frame indices are extraction provenance supplied by
the caller: a hash verifies an unchanged file, not that the image was extracted
correctly or that the reviewer actually looked at it. This gate makes those
attestations explicit and cannot replace inspecting the clip.
"""
from collections.abc import Mapping
import hashlib
from pathlib import Path


DIMENSIONS = (
    'paw_placement', 'limb_bending', 'body_shape', 'tearing',
    'temporal_coherence',
)
CLAIM_TYPES = ('reconstruction', 'conditional_motion_generation')
PROVENANCE_FLAGS = (
    'future_conditioned', 'test_used_for_training', 'test_used_for_selection',
)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _media(path):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f'Not a media file: {resolved}')
    return {'path': str(resolved), 'sha256': _hash(resolved)}


def create_review(video_path, frames_paths):
    """Return an unaccepted template; never infer or invent visual judgments.

    ``frames_paths`` accepts paths or mappings with ``path`` and optional
    ``frame_index``. Plain paths intentionally leave frame indices unset.
    Indices must describe this exact video, not original-source clip indices.
    All cited images should show the specific candidate being reviewed.
    """
    video = _media(video_path)
    frames = []
    for index, item in enumerate(frames_paths):
        descriptor = item if isinstance(item, Mapping) else {'path': item}
        frame = _media(descriptor['path'])
        frame.update(
            id=f'frame_{index:03d}', frame_index=descriptor.get('frame_index'),
            source_video_sha256=video['sha256'], observation='',
        )
        frames.append(frame)
    return {
        'schema_version': 1,
        'video': video,
        'frames': frames,
        'claim_type': None,
        'reviewer': {'kind': None, 'name': '', 'observed_video': False},
        'provenance': {key: None for key in PROVENANCE_FLAGS},
        'dimensions': {
            key: {'status': None, 'observation': '', 'evidence_frame_ids': []}
            for key in DIMENSIONS
        },
        'numeric_metrics': {},
        'limitations': [],
    }


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def decide_review(review, video_path):
    """Validate current media and a completed review, returning a decision.

    Accepted means only that this explicitly scoped visual review passed. It
    does not establish 3D ground truth, generalization, novelty, model training,
    or efficiency. Numeric improvements cannot substitute for visual approval.
    A reconstruction can pass despite future conditioning; a generation claim
    cannot. Missing provenance and observations fail closed as pending.
    """
    pending, failures = [], []
    if not isinstance(review, Mapping):
        return {'accepted': False, 'status': 'pending', 'visual_pass': False,
                'claim_supported': False, 'claim_type': None,
                'reasons': ['An explicit visual review is missing.']}

    def section(name):
        value = review.get(name)
        if not isinstance(value, Mapping):
            pending.append(f'{name}: missing or invalid section.')
            return {}
        return value

    if review.get('schema_version') != 1:
        failures.append('Unsupported visual-review schema.')
    video = section('video')
    actual_hash = None
    try:
        actual_path = Path(video_path).resolve(strict=True)
        actual_hash = _hash(actual_path)
        if Path(video.get('path', '')).resolve() != actual_path:
            failures.append('Reviewed video path differs from the candidate.')
        if video.get('sha256') != actual_hash:
            failures.append('Video hash mismatch: this review is stale or for another video.')
    except (OSError, TypeError, ValueError):
        failures.append('Candidate video or reviewed video path is unreadable.')

    reviewer = section('reviewer')
    if reviewer.get('kind') not in ('human', 'agent') or not _nonempty(reviewer.get('name')):
        pending.append('Reviewer identity must be explicitly supplied.')
    if reviewer.get('observed_video') is not True:
        pending.append('Reviewer must attest to inspecting the motion sequence.')

    frames = review.get('frames')
    if not isinstance(frames, list):
        frames = []
        pending.append('Frame evidence is missing or invalid.')
    valid_ids, frame_indices, frame_paths, seen_ids = set(), set(), set(), set()
    for frame in frames:
        if not isinstance(frame, Mapping):
            failures.append('Malformed frame evidence.')
            continue
        frame_id = frame.get('id')
        if not _nonempty(frame_id):
            failures.append('Frame evidence has no ID.')
            continue
        if frame_id in seen_ids:
            failures.append(f'{frame_id}: duplicate frame ID.')
            continue
        seen_ids.add(frame_id)
        frame_index = frame.get('frame_index')
        if type(frame_index) is not int or frame_index < 0:
            pending.append(f'{frame_id}: an explicit nonnegative frame index is required.')
            continue
        if not _nonempty(frame.get('observation')):
            pending.append(f'{frame_id}: a direct visual observation is required.')
            continue
        if frame.get('source_video_sha256') != actual_hash or actual_hash is None:
            failures.append(f'{frame_id}: source-video hash mismatch.')
            continue
        try:
            frame_path = Path(frame['path']).resolve(strict=True)
            if frame_path == actual_path:
                failures.append(f'{frame_id}: the video itself is not sampled-frame evidence.')
                continue
            if _hash(frame_path) != frame.get('sha256'):
                failures.append(f'{frame_id}: sampled-frame hash mismatch.')
                continue
        except (KeyError, OSError, TypeError, ValueError):
            failures.append(f'{frame_id}: sampled frame is unreadable.')
            continue
        if frame_index in frame_indices or frame_path in frame_paths:
            failures.append(f'{frame_id}: evidence must use distinct frame indices and files.')
            continue
        valid_ids.add(frame_id)
        frame_indices.add(frame_index)
        frame_paths.add(frame_path)
    if len(valid_ids) < 3:
        pending.append('At least three temporally distinct, observed, hash-verified frames are required.')

    dimensions = section('dimensions')
    all_pass = True
    cited_ids = set()
    for dimension in DIMENSIONS:
        entry = dimensions.get(dimension)
        if not isinstance(entry, Mapping):
            pending.append(f'{dimension}: review is missing.')
            all_pass = False
            continue
        status = entry.get('status')
        if status in ('fail', 'uncertain'):
            failures.append(f'{dimension}: reviewer marked {status}.')
            all_pass = False
        elif status != 'pass':
            pending.append(f'{dimension}: explicit pass/fail/uncertain judgment required.')
            all_pass = False
        if not _nonempty(entry.get('observation')):
            pending.append(f'{dimension}: direct observation text is required.')
        evidence = entry.get('evidence_frame_ids')
        if (not isinstance(evidence, list) or not evidence
                or any(not isinstance(item, str) or item not in valid_ids for item in evidence)):
            pending.append(f'{dimension}: cite verified sampled-frame evidence.')
        else:
            cited_ids.update(evidence)
    if len(cited_ids) < 3:
        pending.append('The dimension reviews must collectively cite at least three distinct frames.')

    # Keep visual outcome distinct from permission to make the requested claim.
    visual_pass = all_pass and not pending and not failures
    claim_type = review.get('claim_type')
    if claim_type not in CLAIM_TYPES:
        pending.append('Declare reconstruction or conditional_motion_generation.')
    provenance = section('provenance')
    provenance_complete = all(type(provenance.get(key)) is bool for key in PROVENANCE_FLAGS)
    if not provenance_complete:
        pending.append('All conditioning and test-use provenance flags must be explicit booleans.')
    claim_supported = claim_type in CLAIM_TYPES and provenance_complete
    if claim_type == 'conditional_motion_generation':
        for flag in PROVENANCE_FLAGS:
            if provenance.get(flag) is True:
                failures.append(f'Generation claim unsupported: {flag}=true.')
                claim_supported = False
    accepted = visual_pass and claim_supported and not pending and not failures
    return {
        'accepted': accepted,
        'status': 'accepted' if accepted else ('rejected' if failures else 'pending'),
        'visual_pass': visual_pass,
        'claim_supported': claim_supported,
        'claim_type': claim_type,
        'video_sha256': actual_hash,
        'verified_frame_count': len(valid_ids),
        'reasons': failures + pending,
    }
