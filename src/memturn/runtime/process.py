"""Bounded JSON subprocess calls with wall-clock deadlines and tree cleanup."""
from collections import deque
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time


class ProcessRPC:
    def __init__(self, command, timeout=60, max_bytes=8388608, env=None, rss_limit_bytes=None):
        if timeout <= 0 or max_bytes <= 0:
            raise ValueError('process time and byte limits must be positive')
        self.timeout, self.max_bytes, self.rss_limit_bytes = timeout, max_bytes, rss_limit_bytes
        self.lock, self.closed = threading.Lock(), False
        self.replies, self.errors = queue.Queue(maxsize=2), deque(maxlen=16)
        self.sequence, self.peak_rss_bytes = 0, 0
        inherited_group = os.environ.get('MEMTURN_WORKER_GROUP') == '1'
        self.owns_group = os.name != 'nt' and not inherited_group
        env = dict(os.environ) if env is None else dict(env)
        env['MEMTURN_WORKER_GROUP'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, bufsize=-1, start_new_session=self.owns_group,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        def read():
            try:
                while True:
                    line = self.process.stdout.readline(self.max_bytes + 1)
                    if not line:
                        raise RuntimeError('worker closed its output')
                    if len(line) > self.max_bytes or not line.endswith(b'\n'):
                        raise ValueError('worker response exceeds byte limit or is truncated')
                    self.replies.put_nowait(json.loads(line))
            except Exception as error:
                try: self.replies.put_nowait({'error':type(error).__name__})
                except queue.Full: pass
        def errors():
            try:
                while True:
                    line = self.process.stderr.readline(1024)
                    if not line: break
                    self.errors.append(line.decode('utf-8', 'replace').rstrip())
            except (ValueError, OSError):
                pass
        self.reader = threading.Thread(target=read, daemon=True)
        self.error_reader = threading.Thread(target=errors, daemon=True)
        self.reader.start()
        self.error_reader.start()
        if rss_limit_bytes is not None:
            try:
                import psutil
                self.monitor = psutil.Process(self.process.pid)
            except Exception:
                self.close()
                raise RuntimeError('RSS enforcement requires psutil and process monitoring') from None
        else:
            self.monitor = None

    def request(self, operation, timeout=None, **values):
        budget = self.timeout if timeout is None else min(timeout, self.timeout)
        deadline = time.monotonic() + budget
        if budget <= 0 or not self.lock.acquire(timeout=max(0, budget)):
            raise TimeoutError('worker request deadline before admission')
        try:
            if self.closed:
                raise RuntimeError('worker is closed')
            self.sequence += 1
            body = (json.dumps(dict(op=operation, id=self.sequence, **values), ensure_ascii=False) + '\n').encode('utf-8')
            if len(body) > self.max_bytes:
                raise ValueError('worker input exceeds byte limit')
            write_errors = []
            def write():
                try:
                    view = memoryview(body)
                    while view:
                        written = self.process.stdin.write(view)
                        if not written: raise BrokenPipeError('worker input closed')
                        view = view[written:]
                    self.process.stdin.flush()
                except (OSError, ValueError) as error:
                    write_errors.append(error)
            writer = threading.Thread(target=write, daemon=True)
            writer.start()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('worker exceeded wall-clock deadline')
                if write_errors:
                    raise RuntimeError('worker input failed') from write_errors[0]
                if self.monitor is not None:
                    processes = [self.monitor, *self.monitor.children(recursive=True)]
                    rss = 0
                    for process in processes:
                        try: rss += process.memory_info().rss
                        except Exception:
                            if process.is_running(): raise
                    self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
                    if rss > self.rss_limit_bytes:
                        raise MemoryError('worker process tree exceeded RSS budget')
                try:
                    reply = self.replies.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    continue
                if reply.get('error'):
                    raise RuntimeError('worker failed: ' + str(reply['error']))
                if reply.get('id') != self.sequence:
                    raise ValueError('worker reply does not match request')
                return reply['result']
        except BaseException:
            self.close()
            raise
        finally:
            self.lock.release()

    def close(self):
        if self.closed:
            return
        self.closed = True
        process = self.process
        if not self.owns_group and process.poll() is None:
            # Nested workers share the outer group so a top-level timeout also
            # kills grandchildren. On normal nested cleanup, kill its own tree.
            try:
                import psutil
            except ImportError:
                pass  # The outer owning group is still killed on exit.
            else:
                try:
                    for child in reversed(psutil.Process(process.pid).children(recursive=True)):
                        try: child.kill()
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess:
                    pass  # Worker exited between poll() and the tree snapshot.
        if self.owns_group:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        elif os.name == 'nt' and process.poll() is None:
            try:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            try: stream.close()
            except OSError: pass


def run_worker(operation, payload, timeout, max_bytes=8388608, rss_limit_bytes=None, return_metrics=False):
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH'] = source + os.pathsep + env.get('PYTHONPATH', '')
    rpc = ProcessRPC([sys.executable, '-u', '-m', 'memturn.worker'], timeout, max_bytes,
                     env=env, rss_limit_bytes=rss_limit_bytes)
    try:
        started = time.monotonic()
        result = rpc.request(operation, payload=payload)
        return ({'result':result, 'peak_process_tree_rss_bytes':rpc.peak_rss_bytes,
                 'wall_seconds':time.monotonic()-started} if return_metrics else result)
    finally:
        rpc.close()
