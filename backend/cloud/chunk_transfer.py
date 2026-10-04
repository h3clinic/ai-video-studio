"""Offline, hash-pinned resumable transfer packaging. No network or extraction.

Upload manifest.json, this script, and only missing/corrupt numbered parts.
Pin the manifest SHA256 separately; never trust a manifest from the upload alone.
Successful parts survive a dropped connection. Assembly publishes without replacing
an existing file. This does not start compute or provide a billing deadline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

MAX_BYTES = 256 * 2**20
MAX_CHUNK = 8 * 2**20


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def no_links(path):
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError('Symlinks are not allowed')
        if component.exists() and getattr(component.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
            raise ValueError('Reparse points are not allowed')
    return path


def digest_valid(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_manifest(m):
    if not isinstance(m, dict) or m.get('schema_version') != 1:
        raise ValueError('Unsupported manifest')
    total, size = m.get('bytes'), m.get('chunk_bytes')
    if type(total) is not int or not 1 <= total <= MAX_BYTES:
        raise ValueError('Invalid total bytes')
    if type(size) is not int or not 1 <= size <= MAX_CHUNK:
        raise ValueError('Invalid chunk bytes')
    if not digest_valid(m.get('sha256')):
        raise ValueError('Invalid whole-file digest')
    parts = m.get('parts')
    count = (total + size - 1) // size
    if not isinstance(parts, list) or len(parts) != count or count > 10000:
        raise ValueError('Invalid part count')
    for i, part in enumerate(parts):
        if not isinstance(part, dict) or part.get('name') != f'part-{i:05d}.bin':
            raise ValueError('Unexpected part path/order')
        if type(part.get('bytes')) is not int or part['bytes'] != min(size, total - i * size):
            raise ValueError('Invalid part size')
        if not digest_valid(part.get('sha256')):
            raise ValueError('Invalid part digest')
    return m


def split(source, directory, chunk_bytes=4 * 2**20):
    source, directory = no_links(source), no_links(directory)
    if not source.is_file() or not 1 <= source.stat().st_size <= MAX_BYTES:
        raise ValueError('Expected a regular nonempty file up to 256MiB')
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= MAX_CHUNK:
        raise ValueError('Chunk size must be 1..8MiB')
    if (source.stat().st_size + chunk_bytes - 1) // chunk_bytes > 10000:
        raise ValueError('Too many parts')
    directory.mkdir(parents=True, exist_ok=False)
    m = dict(schema_version=1, bytes=0, chunk_bytes=chunk_bytes, parts=[])
    whole = hashlib.sha256()
    with source.open('rb') as stream:
        while data := stream.read(chunk_bytes):
            name = f'part-{len(m["parts"]):05d}.bin'
            with (directory / name).open('xb') as out:
                out.write(data)
            whole.update(data)
            m['bytes'] += len(data)
            m['parts'].append(dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    m['sha256'] = whole.hexdigest()
    validate_manifest(m)
    if source.stat().st_size != m['bytes'] or sha256(source) != m['sha256']:
        raise ValueError('Source changed during split; preserve parts for diagnosis')
    manifest = directory / 'manifest.json'
    with manifest.open('x', encoding='utf-8') as stream:
        json.dump(m, stream, indent=2)
    return dict(manifest=str(manifest), manifest_sha256=sha256(manifest), bytes=m['bytes'], parts=len(m['parts']))


def load_manifest(directory, expected_manifest_sha256):
    directory = no_links(directory)
    manifest = no_links(directory / 'manifest.json')
    with manifest.open('rb') as stream:
        data = stream.read(4 * 2**20 + 1)
    if len(data) > 4 * 2**20:
        raise ValueError('Oversized manifest')
    if not digest_valid(expected_manifest_sha256) or hashlib.sha256(data).hexdigest() != expected_manifest_sha256:
        raise ValueError('Manifest digest mismatch')
    return validate_manifest(json.loads(data))


def inspect(directory, expected_manifest_sha256):
    directory = no_links(directory)
    m = load_manifest(directory, expected_manifest_sha256)
    missing, corrupt, verified = [], [], []
    remaining = 0
    for part in m['parts']:
        path = no_links(directory / part['name'])
        if not path.exists():
            missing.append(part['name'])
        elif not path.is_file() or path.stat().st_size != part['bytes'] or sha256(path) != part['sha256']:
            corrupt.append(part['name'])
        else:
            verified.append(part['name'])
            continue
        remaining += part['bytes']
    return dict(ready=not missing and not corrupt, missing=missing, corrupt=corrupt,
                verified=verified, remaining_upload_bytes=remaining)


def assemble(directory, output, expected_manifest_sha256):
    directory, output = no_links(directory), no_links(output)
    m = load_manifest(directory, expected_manifest_sha256)
    if output.exists():
        if output.is_file() and output.stat().st_size == m['bytes'] and sha256(output) == m['sha256']:
            return dict(path=str(output), sha256=m['sha256'], bytes=m['bytes'], reused=True)
        raise FileExistsError('Will not replace a different existing output')
    state = inspect(directory, expected_manifest_sha256)
    if not state['ready']:
        raise ValueError('Incomplete transfer: ' + json.dumps(state))
    # A unique partial is retained on failure, not mistaken for a completed file.
    with tempfile.NamedTemporaryFile(mode='wb', dir=output.parent, prefix=output.name+'.', suffix='.partial', delete=False) as out:
        temporary = Path(out.name)
        whole = hashlib.sha256()
        for part in m['parts']:
            with no_links(directory / part['name']).open('rb') as stream:
                for data in iter(lambda: stream.read(2**20), b''):
                    whole.update(data)
                    out.write(data)
        out.flush()
        os.fsync(out.fileno())
    if temporary.stat().st_size != m['bytes'] or whole.hexdigest() != m['sha256']:
        raise ValueError('Assembly hash mismatch; partial preserved')
    # Hard link gives atomic no-clobber publication on NTFS/Linux, unlike replace.
    os.link(temporary, output)
    temporary.unlink()  # Only our successful temporary; output keeps the bytes.
    return dict(path=str(output), sha256=m['sha256'], bytes=m['bytes'], reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('split')
    create.add_argument('source', type=Path); create.add_argument('directory', type=Path)
    create.add_argument('--chunk-bytes', type=int, default=4 * 2**20)
    for command in ('inspect', 'assemble'):
        child = commands.add_parser(command)
        child.add_argument('directory', type=Path)
        child.add_argument('--manifest-sha256', required=True)
        if command == 'assemble': child.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'split': result = split(args.source, args.directory, args.chunk_bytes)
    elif args.command == 'inspect': result = inspect(args.directory, args.manifest_sha256)
    else: result = assemble(args.directory, args.output, args.manifest_sha256)
    print(json.dumps(result, indent=2))
    if args.command == 'inspect' and not result['ready']: raise SystemExit(2)


if __name__ == '__main__': main()
