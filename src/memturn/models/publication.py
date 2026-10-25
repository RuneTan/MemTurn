"""Optional vLLM adapter publication; immutable names preserve in-flight tasks."""
import json
from pathlib import Path


class VLLMAdapters:
    def __init__(self, base_url, local_root, server_root, api_key_env="MEMTURN_TASK_API_KEY", timeout=60, probe=None):
        self.base_url = base_url.rstrip("/")
        self.local_root, self.server_root = Path(local_root).resolve(), server_root.rstrip("/")
        self.api_key_env, self.timeout = api_key_env, timeout
        self.probe = probe

    def _post(self, route, payload):
        from ..runtime.process import run_worker
        result = run_worker('request', {'url':self.base_url+route, 'body':payload, 'method':'POST',
            'api_key_env':self.api_key_env, 'timeout':self.timeout, 'max_bytes':1048576}, self.timeout)
        if result['status'] != 200:
            raise RuntimeError(f"adapter publication failed with HTTP {result['status']}")
        return result['body']

    def _models(self):
        from .preflight import request_json
        return {row['id']:row for row in request_json(self.base_url, '/models', self.api_key_env, self.timeout)['data']}

    def is_registered(self, version):
        return version in self._models()

    def load(self, artifact):
        path = Path(artifact.path).resolve()
        if path.parent != self.local_root:
            raise ValueError("adapter path is outside the shared root")
        target = self.server_root + '/' + path.name
        existing = self._models().get(artifact.version)
        if existing:
            if existing.get('root') != target:
                raise ValueError('server already has this immutable adapter name at a different path')
        else:
            self._post('/load_lora_adapter', {'lora_name':artifact.version, 'lora_path':target})
        loaded = self._models().get(artifact.version)
        if not loaded or loaded.get('root') != target:
            raise ValueError('server did not register the requested adapter path')
        if self.probe:
            self.probe(artifact.version)

    def unload(self, version):
        if version in self._models():
            self._post('/unload_lora_adapter', {'lora_name':version})
        if version in self._models():
            raise ValueError('server still advertises the retired adapter')
