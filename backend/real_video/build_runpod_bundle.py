"""Build an immutable, explicit-allowlist experiment archive; never upload it."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CASE = 'artifacts/real_video/runway_agents/donkey_orange_v1'
ALLOWLIST = (
    'real_video/__init__.py',
    'real_video/evaluate_detail_profile.py',
    'real_video/gaussian_edit_workers.py',
    'real_video/representation.py',
    'real_video/replay_fit.py',
    'real_video/checkpoint_io.py',
    'real_video/articulation_quality.py',
    'real_video/inspect_runway_pilot.py',
    'cloud/runpod_job.py',
    'cloud/runpod_requirements.txt',
    f'{CASE}/source/video.mp4',
    f'{CASE}/gaussian_cpu/gaussians.pt',
    f'{CASE}/gaussian_cpu/gaussians.pt.sha256.json',
    f'{CASE}/gaussian_cpu/gaussian_replay.mp4',
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def selected_files(root):
    root = Path(root).resolve(strict=True)
    selected = []
    for relative in ALLOWLIST:
        path = root / relative
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError('Allowlisted file is missing, redirected, or outside workspace')
        selected.append((relative, path))
    return selected


def build(root=ROOT, out=None):
    root = Path(root).resolve(strict=True)
    out = Path(out) if out is not None else root / 'artifacts/cloud/runpod_bundle_v1.zip'
    if out.suffix != '.zip':
        raise ValueError('A .zip output is required')
    files = selected_files(root)
    checkpoint = root / f'{CASE}/gaussian_cpu/gaussians.pt'
    sidecar = json.loads(checkpoint.with_suffix('.pt.sha256.json').read_text(encoding='utf-8'))
    if sidecar.get('sha256') != sha256(checkpoint):
        raise ValueError('Baseline checkpoint sidecar mismatch')
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, experiment='bounded_planar_detail_reconstruction',
                    no_secrets_or_api_clients=True, training=False, files=[],
                    limits=['No cloud upload or paid launch is performed by this builder.',
                            'Job timeout does not terminate a Pod or stop cloud billing.',
                            'Reconstruction only; no new neural weights or autonomous actions.'])
    # Exclusive creation preserves every earlier bundle. An interrupted archive
    # lacks the final manifest and must not be treated as a verified bundle.
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in files:
            h = hashlib.sha256()
            size = 0
            with path.open('rb') as source, archive.open(relative, 'w') as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    target.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            manifest['files'].append(dict(path=relative, bytes=size, sha256=h.hexdigest()))
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2, allow_nan=False))
    # Verify the bytes actually archived, not a second snapshot of mutable files.
    with zipfile.ZipFile(out) as archive:
        if set(archive.namelist()) != set(ALLOWLIST) | {'bundle_manifest.json'}:
            raise ValueError('Unexpected archive member')
        for entry in manifest['files']:
            if hashlib.sha256(archive.read(entry['path'])).hexdigest() != entry['sha256']:
                raise ValueError('Archive content hash mismatch')
    return dict(path=str(out.resolve()), bytes=out.stat().st_size, sha256=sha256(out),
                manifest=manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(out=args.out), indent=2))
