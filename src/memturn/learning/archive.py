"""Offline checkpoint archives. Nothing here is a source of learning state."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import uuid


def _plain(path, directory=False):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(
            info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
        raise ValueError('checkpoint archive refuses links and reparse points')
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('checkpoint archive requires ordinary files and directories')


def _digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def inventory(directory):
    """Include empty directories and hash file bytes without following links."""
    directory = Path(directory)
    _plain(directory, directory=True)
    files, directories = {}, []
    def fail(error):
        raise error
    for parent, children, names in os.walk(directory, followlinks=False, onerror=fail):
        for name in sorted(children):
            path = Path(parent) / name
            _plain(path, directory=True)
            directories.append(path.relative_to(directory).as_posix())
        for name in sorted(names):
            path = Path(parent) / name
            _plain(path)
            files[path.relative_to(directory).as_posix()] = {
                'bytes': path.stat().st_size, 'sha256': _digest(path)}
    return {'directories': sorted(directories), 'files': files}


def verify_archive(directory):
    directory = Path(directory)
    _plain(directory, directory=True)
    _plain(directory / 'archive.json')
    metadata = json.loads((directory / 'archive.json').read_text(encoding='utf-8'))
    if metadata.get('format') != 1 or inventory(directory / 'payload') != metadata['inventory']:
        raise ValueError('checkpoint archive checksum or inventory mismatch')
    return metadata


def archive_orphan(directory, artifact_root):
    """Copy, verify, then remove one orphan; caller must exclude live versions.

    Run directories are owned by one runtime process. The archive is outside
    the adapter root, is never auto-loaded, and is retained for manual recovery.
    Incomplete archive copies and interrupted cleanups preserve evidence.
    """
    artifact_root, directory = Path(artifact_root).resolve(), Path(directory)
    _plain(directory, directory=True)
    if directory.resolve().parent != artifact_root:
        raise ValueError('refusing cleanup outside artifact root')
    snapshot = inventory(directory)
    archive_root = artifact_root.parent / 'research_archive' / 'orphan_adapters'
    for parent in (archive_root.parent, archive_root):
        if parent.exists() or parent.is_symlink():
            _plain(parent, directory=True)
        else:
            parent.mkdir()
    token = directory.name + '_' + uuid.uuid4().hex
    staging, archived = archive_root / ('.pending_' + token), archive_root / token
    staging.mkdir()
    shutil.copytree(directory, staging / 'payload')
    metadata = {'format': 1, 'reason': 'uncommitted_checkpoint_recovery',
                'original_path': str(directory.resolve()), 'version': directory.name,
                'inventory': snapshot}
    for relative in snapshot['files']:
        # Windows FlushFileBuffers requires a writable handle.
        with (staging / 'payload' / relative).open('r+b') as stream:
            os.fsync(stream.fileno())
    with (staging / 'archive.json').open('w', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    verify_archive(staging)
    staging.rename(archived)
    # Verify the final copy and the unchanged source before the first unlink.
    verify_archive(archived)
    if inventory(directory) != snapshot:
        raise ValueError('checkpoint changed during archival; source preserved')
    for relative in snapshot['files']:
        path = directory / relative
        _plain(path)
        if not path.resolve().is_relative_to(artifact_root):
            raise ValueError('refusing cleanup outside artifact root')
        path.unlink()
    for relative in sorted(snapshot['directories'], key=lambda p: p.count('/'), reverse=True):
        (directory / relative).rmdir()
    directory.rmdir()
    return archived


def restore_archive(archive, destination):
    """Explicit recovery to a NEW directory; does not activate a model pointer."""
    archive, destination = Path(archive), Path(destination)
    metadata = verify_archive(archive)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('restore destination must not exist')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ('.restore_' + uuid.uuid4().hex)
    shutil.copytree(archive / 'payload', staging)
    if inventory(staging) != metadata['inventory']:
        raise ValueError('restored checkpoint checksum or inventory mismatch')
    # rename refuses an existing directory on Windows; check again on POSIX.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('restore destination must not exist')
    staging.rename(destination)
    return destination
