"""Verify and advise away only the pinned model's own clean file page cache.

Linux man-pages: https://man7.org/linux/man-pages/man2/posix_fadvise.2.html
DONTNEED is advisory, not guaranteed; dirty pages may survive without fsync.
No global drop_caches, unlink, model loads, GPU calls, or resource-gate bypass.
Successful calls do NOT establish reclaimed RAM. The caller must take a fresh
host/cgroup measurement and apply its unchanged admission requirement.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time

REPO = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
REVISION = '596658fd9ca6b7b71d5057529bbf319ecbc61d74'
REPO_CACHE_NAME = 'models--Wan-AI--Wan2.2-I2V-A14B-Diffusers'
COMPONENTS = {'transformer', 'transformer_2', 'text_encoder', 'vae', 'tokenizer', 'scheduler'}


def _digest_valid(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def verified_file_plan(snapshot, manifest_path, manifest_sha256):
    """Validate one hashed manifest read and every file scope before advising."""
    snapshot = Path(snapshot).resolve(strict=True)
    if not snapshot.is_dir() or snapshot.name != REVISION:
        raise ValueError('Expected the exact pinned model snapshot directory')
    with Path(manifest_path).open('rb') as stream:
        payload = stream.read(4*2**20+1)
    if (len(payload) > 4*2**20 or not _digest_valid(manifest_sha256)
            or hashlib.sha256(payload).hexdigest() != manifest_sha256):
        raise ValueError('Verified download manifest pin mismatch')
    manifest = json.loads(payload)
    if (manifest.get('repo_id') != REPO or manifest.get('revision') != REVISION
            or manifest.get('status') != 'verified'
            or Path(manifest.get('local_path', '')).resolve(strict=True) != snapshot):
        raise ValueError('Download manifest is not verified for this exact snapshot')
    files = manifest.get('files')
    if not isinstance(files, list) or not 1 <= len(files) <= 1000:
        raise ValueError('Invalid verified file list')
    plan, names, total = [], set(), 0
    # HF snapshots normally link into the same repository cache's blobs, not
    # files physically under snapshots/<revision>. No other target is allowed.
    blobs = None
    if snapshot.parent.name == 'snapshots' and snapshot.parent.parent.name == REPO_CACHE_NAME:
        blobs = (snapshot.parent.parent/'blobs').resolve()
    for item in files:
        name = item.get('path')
        if (not isinstance(name, str) or not name or any(c in name for c in ('\\', ':', '\x00'))
                or any(p in ('', '.', '..') for p in name.split('/'))
                or PurePosixPath(name).is_absolute() or name in names):
            raise ValueError('Invalid or duplicated manifest file path')
        if name != 'model_index.json' and name.split('/')[0] not in COMPONENTS:
            raise ValueError('Manifest file outside allowed model components')
        size, sha = item.get('bytes'), item.get('sha256')
        if type(size) is not int or size < 0 or not _digest_valid(sha):
            raise ValueError('Invalid file size or SHA-256')
        path = snapshot.joinpath(*PurePosixPath(name).parts)
        resolved = path.resolve(strict=True)
        inside = resolved.is_relative_to(snapshot)
        blob = (blobs is not None and resolved.parent == blobs
                and re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', resolved.name) is not None)
        if not inside and not blob:
            raise ValueError('Model file resolves outside pinned snapshot/repository blobs')
        if blob and item.get('lfs_verified') is True and resolved.name != sha:
            raise ValueError('LFS blob identity disagrees with verified SHA-256')
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ValueError('Model file type/size differs from verified manifest')
        plan.append(dict(name=name, path=path, resolved=resolved, identity=_identity(info),
                         bytes=size, sha256=sha))
        names.add(name); total += size
    if ('model_index.json' not in names or type(manifest.get('expected_bytes')) is not int
            or total != manifest['expected_bytes'] or not 0 < total <= 130_000_000_000):
        raise ValueError('Manifest expected total/component list mismatch')
    return snapshot, plan, total


def release_model_cache(snapshot, manifest_path, manifest_sha256, *, max_seconds=300):
    if sys.platform != 'linux' or not hasattr(os, 'posix_fadvise') or not hasattr(os, 'POSIX_FADV_DONTNEED'):
        raise RuntimeError('Scoped cache release requires Linux posix_fadvise')
    if type(max_seconds) not in (int, float) or not 0 < max_seconds <= 300:
        raise ValueError('Cache-verification deadline must be within 300 seconds')
    started = time.monotonic()
    snapshot, plan, total = verified_file_plan(snapshot, manifest_path, manifest_sha256)
    report = dict(schema_version=1, repo_id=REPO, revision=REVISION,
        manifest_sha256=manifest_sha256, verified_bytes=0, advised_bytes=0, files=[],
        mode='SHA-256 reverify, fsync same file, POSIX_FADV_DONTNEED(offset=0,length=0)',
        cache_reclaimed_bytes=None, memory_admission_passed=False,
        scope='Advisory file-cache release only; fresh effective-memory gate still required',
        global_cache_drop=False, deleted_files=False, source='https://man7.org/linux/man-pages/man2/posix_fadvise.2.html')
    def check_deadline():
        if time.monotonic()-started >= max_seconds:
            raise TimeoutError('Scoped model cache verification deadline expired')
    for item in plan:
        check_deadline()
        if item['path'].resolve(strict=True) != item['resolved']:
            raise ValueError('Model path target changed after scope validation')
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(item['resolved'], flags)
        try:
            if _identity(os.fstat(fd)) != item['identity']:
                raise ValueError('Model file changed before verification')
            sha = hashlib.sha256()
            count = 0
            while data := os.read(fd, 8*2**20):
                check_deadline(); sha.update(data); count += len(data)
            if (count != item['bytes'] or sha.hexdigest() != item['sha256']
                    or _identity(os.fstat(fd)) != item['identity']):
                raise ValueError('Model file SHA-256/identity verification failed')
            # Flush only this FD, not the machine. Advice with length zero covers
            # all complete pages through EOF; partial tail pages may be retained.
            os.fsync(fd)
            check_deadline()
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            report['verified_bytes'] += count
            report['advised_bytes'] += count
            report['files'].append(dict(path=item['name'], bytes=count, sha256=item['sha256'],
                                        advice_returned=True))
        finally:
            os.close(fd)
    report.update(seconds=time.monotonic()-started, expected_bytes=total,
                  verification_completed=True, fresh_memory_measurement_required=True)
    return report
