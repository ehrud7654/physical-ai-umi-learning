"""Read-only inventory for an unfamiliar UMI delivery directory or archive.

This deliberately does not claim that an ORB-SLAM pose, ArUco transform,
timestamp, image, or gap follows the Track A contract. Inspect the first real
episode before writing a format-specific decoder.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
from typing import BinaryIO, Callable
import zipfile


MAX_MEMBERS = 20_000
MAX_TOTAL_BYTES = 4 * 1024**3
PREVIEW_BYTES = 64 * 1024
CHUNK_BYTES = 1024 * 1024


class IntakeError(ValueError):
    """The input cannot be inventoried without ambiguity or unsafe expansion."""


def _member_name(raw: str) -> str:
    if not raw or raw.startswith(('/', '\\')) or '\\' in raw or ':' in raw:
        raise IntakeError(f"unsafe archive member name: {raw!r}")
    parts = PurePosixPath(raw).parts
    if '..' in parts:
        raise IntakeError(f"unsafe archive member name: {raw!r}")
    name = PurePosixPath(raw).as_posix()
    if name in ('', '.'):
        raise IntakeError(f"empty archive member name: {raw!r}")
    return name


def _structure(name: str, preview: bytes, size: int) -> dict[str, object] | None:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in ('.json', '.csv'):
        return None
    if size > PREVIEW_BYTES:
        return {'inspection': 'skipped_large_text_file'}
    try:
        decoded = preview.decode('utf-8-sig')
        if suffix == '.json':
            value = json.loads(decoded)
            if isinstance(value, dict):
                return {'json_top_keys': sorted(map(str, value.keys()))}
            return {'json_top_type': type(value).__name__}
        rows = csv.reader(io.StringIO(decoded))
        return {'csv_columns': next(rows, [])}
    except (UnicodeError, ValueError, csv.Error) as error:
        return {'inspection_error': type(error).__name__}


def _digest_member(open_member: Callable[[], BinaryIO], size: int,
                   name: str) -> dict[str, object]:
    digest = hashlib.sha256()
    preview = bytearray()
    counted = 0
    with open_member() as source:
        while block := source.read(CHUNK_BYTES):
            counted += len(block)
            if counted > size:
                raise IntakeError(f"member grew beyond declared size: {name}")
            digest.update(block)
            if len(preview) < PREVIEW_BYTES:
                preview.extend(block[:PREVIEW_BYTES - len(preview)])
    if counted != size:
        raise IntakeError(f"member size changed while reading: {name}")
    result: dict[str, object] = {
        'path': name, 'size_bytes': size, 'sha256': digest.hexdigest(),
    }
    structure = _structure(name, bytes(preview), size)
    if structure is not None:
        result['structure'] = structure
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _checked_entries(entries):
    entries = list(entries)
    if len(entries) > MAX_MEMBERS:
        raise IntakeError(f"more than {MAX_MEMBERS} files")
    names = [name for name, _, _ in entries]
    if len(set(names)) != len(names):
        raise IntakeError("duplicate normalized member path")
    total = sum(size for _, size, _ in entries)
    if any(size < 0 for _, size, _ in entries) or total > MAX_TOTAL_BYTES:
        raise IntakeError(f"uncompressed size exceeds {MAX_TOTAL_BYTES} bytes")
    return sorted(entries, key=lambda item: item[0]), total


def inspect_delivery(source: Path) -> dict[str, object]:
    """Fingerprint an episode envelope; every semantic quality gate stays open."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        if source.is_symlink():
            raise IntakeError("symlinked delivery root is unsupported")
        entries = []
        for path in source.rglob('*'):
            if path.is_symlink():
                raise IntakeError(f"symlinked path is unsupported: {path}")
            if path.is_file():
                name = _member_name(path.relative_to(source).as_posix())
                entries.append((name, path.stat().st_size, lambda p=path: p.open('rb')))
        entries, total = _checked_entries(entries)
        files = [_digest_member(opener, size, name) for name, size, opener in entries]
        tree_digest = hashlib.sha256()
        for item in files:
            tree_digest.update(json.dumps(
                [item['path'], item['size_bytes'], item['sha256']],
                ensure_ascii=False, separators=(',', ':'),
            ).encode('utf-8'))
            tree_digest.update(b'\n')
        artifact_type, artifact_hash = 'directory', tree_digest.hexdigest()
    elif source.is_file() and source.suffix.lower() == '.zip':
        artifact_type, artifact_hash = 'zip', _file_sha256(source)
        with zipfile.ZipFile(source) as archive:
            entries = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise IntakeError(f"symlinked ZIP member: {info.filename}")
                name = _member_name(info.filename)
                entries.append((name, info.file_size,
                                lambda i=info: archive.open(i, 'r')))
            entries, total = _checked_entries(entries)
            files = [_digest_member(opener, size, name)
                     for name, size, opener in entries]
    elif source.is_file() and (source.name.lower().endswith('.tar')
                               or source.name.lower().endswith(('.tar.gz', '.tgz'))):
        artifact_type, artifact_hash = 'tar', _file_sha256(source)
        with tarfile.open(source, 'r:*') as archive:
            entries = []
            for info in archive.getmembers():
                if info.isdir():
                    continue
                if not info.isfile():
                    raise IntakeError(f"non-regular TAR member: {info.name}")
                name = _member_name(info.name)
                entries.append((name, info.size,
                                lambda i=info: archive.extractfile(i)))
            entries, total = _checked_entries(entries)
            files = [_digest_member(opener, size, name)
                     for name, size, opener in entries]
    else:
        raise IntakeError("supply an episode directory, ZIP, TAR or TAR.GZ")

    extensions = Counter(PurePosixPath(item['path']).suffix.lower() or '<none>'
                         for item in files)
    metadata = [
        {'path': item['path'], **item['structure']}
        for item in files if 'structure' in item
    ]
    return {
        'status': 'INVENTORY_ONLY_SCHEMA_UNVERIFIED',
        'schema_verified': False, 'quality_verified': False,
        'artifact': str(source.resolve()), 'artifact_type': artifact_type,
        'artifact_sha256': artifact_hash,
        'file_count': len(files), 'total_uncompressed_bytes': total,
        'extension_counts': dict(sorted(extensions.items())),
        'metadata_structure': metadata, 'files': files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('delivery', type=Path)
    parser.add_argument('--out', type=Path, help='optional JSON report path')
    args = parser.parse_args()
    result = inspect_delivery(args.delivery)
    summary = {key: result[key] for key in (
        'status', 'artifact_type', 'artifact_sha256', 'file_count',
        'total_uncompressed_bytes', 'extension_counts', 'metadata_structure',
    )}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
        print(f"inventory saved: {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
