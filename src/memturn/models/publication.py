"""Optional vLLM adapter publication; immutable names preserve in-flight tasks."""
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class VLLMAdapters:
    def __init__(self, base_url, local_root, server_root, api_key_env="MEMTURN_TASK_API_KEY", timeout=60):
        self.base_url = base_url.rstrip("/")
        self.local_root, self.server_root = Path(local_root).resolve(), server_root.rstrip("/")
        self.api_key_env, self.timeout = api_key_env, timeout

    def _post(self, route, payload):
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(self.base_url + route, json.dumps(payload).encode(), headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except HTTPError as error:
            raise RuntimeError(f"adapter publication failed with HTTP {error.code}") from None

    def load(self, artifact):
        path = Path(artifact.path).resolve()
        if path.parent != self.local_root:
            raise ValueError("adapter path is outside the shared root")
        self._post("/load_lora_adapter", {"lora_name": artifact.version,
                   "lora_path": self.server_root + "/" + path.name})

    def unload(self, version):
        self._post("/unload_lora_adapter", {"lora_name": version})
