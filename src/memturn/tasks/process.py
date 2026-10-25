"""Native simulator in a bounded, terminable process tree."""
from dataclasses import asdict
import time

from ..contracts import Transition
from ..models.http import remaining_timeout
from ..runtime.process import ProcessRPC


class ProcessEnvironment:
    def __init__(self, settings):
        self.settings, self.rpc, self.process = dict(settings), None, None
        self.timeout = float(settings.get('timeout_seconds', 120))
        self.deadline = None
        if self.timeout <= 0:
            raise ValueError('timeout_seconds must be positive')

    def set_deadline(self, deadline):
        self.deadline = deadline

    def _start(self):
        command = self.settings['worker_command']
        if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
            raise ValueError('worker_command must be an argument array')
        self.rpc = ProcessRPC(command, self.timeout, self.settings.get('max_response_bytes', 8388608),
                              rss_limit_bytes=self.settings.get('rss_limit_bytes'))
        self.process = self.rpc.process
        native = {k:v for k,v in self.settings.items() if k not in
                  ('worker_command','timeout_seconds','max_response_bytes','rss_limit_bytes')}
        self._request('configure', settings=native)

    def _request(self, operation, **values):
        timeout = remaining_timeout(self.timeout)
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
        return self.rpc.request(operation, timeout=timeout, **values)

    def reset(self, task):
        if self.process is None:
            self._start()
        return self._request('reset', task=asdict(task))

    def step(self, action):
        return Transition(**self._request('step', action=action))

    def close(self, force=False):
        if self.rpc:
            self.rpc.close()
            self.rpc, self.process = None, None
