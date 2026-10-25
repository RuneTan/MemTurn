"""Check deployment claims separately from observable HTTP behavior."""
import hashlib
import json
from pathlib import Path

from ..runtime.process import run_worker


def request_json(base_url, route, api_key_env, timeout=30, body=None):
    result = run_worker('request', {'url':base_url.rstrip('/') + route, 'api_key_env':api_key_env,
        'timeout':timeout, 'max_bytes':1048576, 'method':'POST' if body is not None else 'GET', 'body':body}, timeout)
    if result['status'] != 200:
        raise RuntimeError(f"deployment request returned HTTP {result['status']}")
    return json.loads(result['body'])


def check_deployment(settings, base_url):
    path = Path(settings['deployment_manifest'])
    raw = path.read_bytes()
    if len(raw) > 1048576:
        raise ValueError('deployment manifest exceeds byte limit')
    if hashlib.sha256(raw).hexdigest() != settings['deployment_manifest_sha256']:
        raise ValueError('deployment manifest digest differs from the frozen profile')
    record = json.loads(raw)
    for key in ('model', 'revision', 'tokenizer_model', 'tokenizer_revision', 'context_window'):
        if record.get(key) != settings[key]:
            raise ValueError('deployment manifest differs at ' + key)
    live = request_json(base_url, '/models', settings['api_key_env'])
    if settings['model'] not in {row['id'] for row in live.get('data', [])}:
        raise ValueError('declared model is not advertised by the inference server')
    return {'advertised_model':settings['model'], 'revision':settings['revision'],
            'manifest_sha256':settings['deployment_manifest_sha256'],
            'identity_evidence':'operator_attestation_plus_live_model_name_not_weight_attestation'}


def protocol_probe(client, model_version='base', observation='Protocol check: choose a valid environment command.'):
    import time
    from .http import ChatPolicy
    started = time.monotonic()
    action = ChatPolicy(client).act(observation, (), model_version, ())
    if not isinstance(action, str) or not action.strip():
        raise ValueError('candidate failed the environment-action protocol probe')
    return {'protocol_valid':True, 'seconds':time.monotonic()-started,
            'scope':'format_and_availability_only_not_memory_mastery'}
