"""Domain checks are bounded computations, never an unbounded foreground hook."""
from collections import OrderedDict
from dataclasses import asdict
import hashlib

from ..contracts import dumps
from ..models.protocol import remaining_timeout
from ..runtime.process import run_worker


class ProcessVerifier:
    def __init__(self, factory, kwargs=None, timeout=120, max_bytes=33554432, max_cached=128):
        self.factory, self.kwargs, self.timeout = factory, kwargs or {}, timeout
        self.max_bytes, self.max_cached, self.cache = max_bytes, max_cached, OrderedDict()

    def _verify(self, operation, **values):
        if remaining_timeout(self.timeout) <= 0:
            raise TimeoutError('domain validation phase deadline expired')
        payload = {'factory':self.factory, 'kwargs':self.kwargs, 'operation':operation, **values}
        key = hashlib.sha256(dumps(payload).encode()).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        result = bool(run_worker('verify', payload, remaining_timeout(self.timeout), self.max_bytes))
        self.cache[key] = result
        while len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
        return result

    def verify_support(self, memory, episode, support):
        return self._verify('support', memory=memory.to_dict(), episode=episode.to_dict(), support=asdict(support))

    def verify_example(self, example, episode):
        return self._verify('example', example=asdict(example), episode=episode.to_dict())
