"""Conversion-specific provenance, independent of the shared rollout freeze hash."""
import hashlib
from pathlib import Path

CONVERSION_GLOBS = ('umi/**/*.py', 'track_a/**/*.py', 'tools/**/*.py',
                    'contract/**/*.py', 'sim/**/*.py', 'tracking/**/*.py',
                    'paths.py', 'configs/**/*.yaml', 'configs/**/*.json',
                    'configs/**/*.urdf', 'sim/**/*.xml')


def conversion_manifest(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files = {p for pattern in CONVERSION_GLOBS for p in root.glob(pattern) if p.is_file()}
    return {p.relative_to(root).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in sorted(files, key=lambda p: p.relative_to(root).as_posix())}


def conversion_digest(root=None):
    import json
    payload = json.dumps(conversion_manifest(root), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
