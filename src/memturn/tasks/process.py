"""Optional stdio boundary between the model runtime and native simulator dependencies."""
from collections import deque
from dataclasses import asdict
import json
import os
import queue
import subprocess
import threading

from ..contracts import Transition


class ProcessEnvironment:
    def __init__(self, settings):
        self.settings, self.process = dict(settings), None
        self.timeout = float(settings.get('timeout_seconds', 120))
        if self.timeout <= 0:
            raise ValueError('timeout_seconds must be positive')
        self.replies, self.errors = queue.Queue(maxsize=4), deque(maxlen=20)
        self.sequence = 0

    def _start(self):
        command = self.settings['worker_command']
        if not isinstance(command, list) or not command or not all(isinstance(v,str) for v in command):
            raise ValueError('worker_command must be an argument array')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        process = self.process
        self.replies = queue.Queue(maxsize=4)
        replies = self.replies
        def read():
            try:
                for line in process.stdout:
                    replies.put(json.loads(line))
            except (ValueError, OSError) as error:
                replies.put({'error':type(error).__name__})
            finally:
                replies.put({'error':'worker_closed'})
        def errors():
            try:
                for line in process.stderr:
                    self.errors.append(line.rstrip()[:1000])
            except (ValueError,OSError):
                pass
        threading.Thread(target=read,daemon=True).start()
        threading.Thread(target=errors,daemon=True).start()
        native = {k:v for k,v in self.settings.items() if k not in ('worker_command','timeout_seconds')}
        self._request('configure', settings=native)

    def _request(self, operation, **values):
        self.sequence += 1
        self.process.stdin.write(json.dumps(dict(op=operation,id=self.sequence,**values),ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        try:
            result = self.replies.get(timeout=self.timeout)
        except queue.Empty:
            self.close(force=True)
            raise TimeoutError('native simulator worker timed out') from None
        if result.get('error'):
            raise RuntimeError('native worker: '+result['error'])
        if result.get('id') != self.sequence:
            raise RuntimeError('native worker response does not match request')
        return result['result']

    def reset(self, task):
        if self.process is None: self._start()
        return self._request('reset',task=asdict(task))

    def step(self, action):
        return Transition(**self._request('step',action=action))

    def close(self, force=False):
        process = self.process
        if process is None: return
        try:
            if force and process.poll() is None:
                process.kill()
            if not force and process.poll() is None:
                self._request('close')
            if process.stdin: process.stdin.close()
            process.wait(timeout=10)
        except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
            if process.poll() is None: process.kill()
            process.wait(timeout=10)
        finally:
            self.process = None
            for stream in (process.stdin,process.stdout,process.stderr):
                if stream:
                    stream.close()
