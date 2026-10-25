"""One writer process and an immutable, bounded-input task stream."""
import hashlib
import json
import os
from pathlib import Path

from ..contracts import Task, dumps


class RunLease:
    """OS-released advisory lock; never break a lock using a stale PID file."""
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        if self.file.seek(0, 2) == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            self.file = None
            raise RuntimeError('run directory already belongs to another process') from None
        return self

    def __exit__(self, *args):
        if self.file:
            try:
                self.file.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file, fcntl.LOCK_UN)
            finally:
                self.file.close()
                self.file = None


def inspect_manifest(path, max_bytes=16777216, expected_family=None):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError('task manifest exceeds the configured input byte limit')
    fingerprint, ids, count = hashlib.sha256(), set(), 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            task = Task(**json.loads(line))
            if (not isinstance(task.id,str) or not task.id or len(task.id.encode('utf-8')) > 256 or task.id in ids
                    or isinstance(task.max_steps,bool) or not isinstance(task.max_steps,int) or task.max_steps <= 0):
                raise ValueError('manifest requires unique nonempty task IDs and positive step limits')
            if expected_family and task.family != expected_family:
                raise ValueError('task family differs from the selected native environment')
            ids.add(task.id)
            fingerprint.update((dumps(task.__dict__) + '\n').encode('utf-8'))
            count += 1
    if not count:
        raise ValueError('empty task manifest')
    return {'sha256': fingerprint.hexdigest(), 'tasks': count, 'bytes': path.stat().st_size}


def read_manifest(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield Task(**json.loads(line))
