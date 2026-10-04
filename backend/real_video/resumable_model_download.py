"""Preserve resumable model bytes; expose a destination only after SHA256.

The caller must establish an official, pinned source and trusted expected
size/hash (for example official Hugging Face revision + LFS metadata). HTTPS
alone is NOT proof of publisher identity. This helper executes no downloaded
content and does not disable TLS. Redirects are followed only over HTTPS.

Each destination owns stable .partial and .partial.json files. The latter
binds the source URL, expected size and SHA256, not a claim that the partial
prefix is verified. Interrupted, rejected, and hash-failing bytes are retained.
Only a complete hash-verified file is promoted. A server that ignores Range
on resume is rejected rather than appending a new full response to old bytes.
Transfers use explicit bounded byte-range windows, not an open-ended GET.
There are no hidden failure retries; caller owns wall limits.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urljoin, urlsplit

from filelock import FileLock
import httpx


READ_CHUNK_BYTES = 1024*1024


class DownloadVerificationError(ValueError):
    """The response/file is incompatible with the trusted download contract."""


class ResumeRejectedError(DownloadVerificationError):
    """The partial was preserved because safe range continuation was impossible."""


def _https(url):
    if not isinstance(url, str):
        raise ValueError('A pinned official HTTPS source URL is required')
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.fragment):
        raise ValueError('Only HTTPS URLs without embedded credentials/fragments are allowed')
    return url


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def _reject_special(path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f'Download target must be a regular file, not a link/directory: {path.name}')


def _write_manifest(path, value):
    handle, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        # The per-destination process lock serializes cooperating downloaders.
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _promote(partial, destination, metadata, expected_size, expected_sha):
    if partial.stat().st_size != expected_size or _digest(partial) != expected_sha:
        raise DownloadVerificationError('Completed partial failed size/SHA256; bytes retained for inspection')
    # Linking is an atomic no-overwrite publish on the same filesystem. Unlike
    # replace(), it cannot destroy a destination created outside our lock.
    os.link(partial, destination)
    partial.unlink()
    metadata.unlink()
    return destination


def _response_bounds(response, offset, requested_end, size):
    if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
        raise ResumeRejectedError('Encoded response rejected; partial bytes retained')
    status = response.status_code
    if status == 200:
        raise ResumeRejectedError('Server ignored bounded Range; partial retained without appending or restarting')
    elif status == 206:
        matched = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if matched is None:
            raise ResumeRejectedError('Missing or invalid Content-Range; partial retained')
        start, end, total = map(int, matched.groups())
        if start != offset or end != requested_end or total != size or not start <= end < size:
            raise ResumeRejectedError('Content-Range does not exactly match requested start/end/trusted size; partial retained')
    elif status == 416:
        raise ResumeRejectedError('Server rejected saved byte range; partial retained')
    else:
        response.raise_for_status()
        raise ResumeRejectedError(f'Unexpected response status {status}; partial retained')
    expected_body = end-offset+1
    length = response.headers.get('Content-Length')
    if length is not None and (not length.isdecimal() or int(length) != expected_body):
        raise ResumeRejectedError('Content-Length does not match the requested response span; partial retained')
    return expected_body


def download_verified(url, destination, size, sha256, *, range_bytes=16*2**20):
    """Download/resume bounded ranges, returning a verified destination ``Path``.

    Unverified bytes always remain under ``<destination>.partial``. On network
    failure they are flushed/fsynced and the exception propagates. Repeating
    this call resumes exactly the actual saved length. Completed unverified
    bytes can be verified/promoted offline by another call. Invalid complete
    prefixes require explicit human/caller repair; they are never erased.

    Each request explicitly asks for at most ``range_bytes`` bytes (16 MiB by
    default); returned Content-Range must match its exact start/end/total.
    Even the first request rejects a server that ignores this bounded Range.
    HTTPS redirects are followed at most ten times per range, with TLS
    verification and bounded connect/read timeouts. Expected SHA256 comes from a trusted
    manifest, not from the same untrusted response body. No model is loaded.
    """
    _https(url)
    if type(size) is not int or size < 1:
        raise ValueError('Positive expected byte count required')
    if not isinstance(sha256, str) or re.fullmatch(r'[0-9a-f]{64}', sha256) is None:
        raise ValueError('Expected lowercase SHA256 digest required')
    if type(range_bytes) is not int or not 1 <= range_bytes <= 64*2**20:
        raise ValueError('Range windows must be positive integers no larger than 64 MiB')
    destination = Path(destination).absolute()
    partial = destination.with_name(destination.name+'.partial')
    metadata = destination.with_name(destination.name+'.partial.json')
    lock = destination.with_name(destination.name+'.download.lock')
    destination.parent.mkdir(parents=True, exist_ok=True)
    for path in (destination, partial, metadata, lock):
        _reject_special(path)
    binding = dict(schema='verified_resumable_download_v1', source_url=url,
                   expected_size=size, expected_sha256=sha256,
                   partial_verified=False)
    with FileLock(str(lock), timeout=0):
        for path in (destination, partial, metadata):
            _reject_special(path)
        if destination.exists():
            if destination.stat().st_size == size and _digest(destination) == sha256:
                return destination
            raise DownloadVerificationError('Existing destination differs; it will not be overwritten')
        if metadata.exists():
            try:
                saved = json.loads(metadata.read_text(encoding='utf-8'))
            except (ValueError, OSError) as error:
                raise ResumeRejectedError('Unreadable partial binding; bytes retained') from error
            if saved != binding:
                raise ResumeRejectedError('Source URL/size/hash binding changed; existing partial retained')
        elif partial.exists():
            raise ResumeRejectedError('Partial has no trusted source binding; not adopted or deleted')
        else:
            _write_manifest(metadata, binding)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ResumeRejectedError('Partial is larger than trusted size; retained for inspection')
        if offset == size:
            return _promote(partial, destination, metadata, size, sha256)
        with httpx.Client(verify=True, follow_redirects=False,
                          timeout=httpx.Timeout(60., connect=15.)) as client:
            while offset < size:
                requested_end = min(size-1, offset+range_bytes-1)
                headers = {'Accept-Encoding': 'identity', 'Range':f'bytes={offset}-{requested_end}'}
                # Resolve each bounded window through the stable source URL;
                # never persist/rely on an expired signed CDN redirect URL.
                current = url
                for redirects in range(11):
                    with client.stream('GET', current, headers=headers) as response:
                        if response.status_code in (301, 302, 303, 307, 308):
                            location = response.headers.get('Location')
                            if location is None or redirects == 10:
                                raise ResumeRejectedError('Missing redirect or redirect limit; partial retained')
                            current = _https(urljoin(current, location))
                            continue
                        expected_body = _response_bounds(response, offset, requested_end, size)
                        incoming = 0
                        # Persist each 1 MiB batch, not every network packet.
                        # Interruption can discard <1 MiB buffered by httpx;
                        # all fsynced bytes remain resumable on the next call.
                        with partial.open('ab', buffering=0) as stream:
                            try:
                                for chunk in response.iter_raw(chunk_size=READ_CHUNK_BYTES):
                                    if incoming+len(chunk) > expected_body:
                                        raise DownloadVerificationError('Response exceeds declared/trusted span; accepted prefix retained')
                                    view = memoryview(chunk)
                                    while view:
                                        written = stream.write(view)
                                        if written is None or written <= 0:
                                            raise OSError('No progress writing partial file')
                                        view = view[written:]
                                    incoming += len(chunk)
                                    os.fsync(stream.fileno())
                            finally:
                                os.fsync(stream.fileno())
                        if incoming != expected_body:
                            raise DownloadVerificationError('Response ended early; received partial retained')
                        offset += incoming
                        break
            return _promote(partial, destination, metadata, size, sha256)
        raise ResumeRejectedError('No complete response; partial retained')
