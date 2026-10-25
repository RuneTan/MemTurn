"""Private JSON process entry point; stdout contains protocol replies only."""
from dataclasses import asdict
import contextlib
import json
import sys


def execute(operation, payload):
    if operation == 'request':
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        import os
        headers = {'Content-Type':'application/json', 'User-Agent':'MemTurn/0.1'}
        key = os.environ.get(payload.get('api_key_env', ''))
        if key: headers['Authorization'] = 'Bearer ' + key
        body = json.dumps(payload['body']).encode() if payload.get('body') is not None else None
        request = Request(payload['url'], body, headers, method=payload.get('method', 'GET'))
        try:
            with urlopen(request, timeout=payload['timeout']) as response:
                data = response.read(payload['max_bytes'] + 1)
                if len(data) > payload['max_bytes']: raise ValueError('HTTP response exceeds byte budget')
                return {'status':response.status, 'body':data.decode('utf-8')}
        except HTTPError as error:
            status = error.code
            error.close()
            return {'status':status, 'body':''}
    if operation == 'verify':
        import importlib
        from .contracts import Episode, Example, Memory, Support
        module, name = payload['factory'].split(':', 1)
        verifier = getattr(importlib.import_module(module), name)(**payload['kwargs'])
        episode = Episode.from_dict(payload['episode'])
        if payload['operation'] == 'support':
            return bool(verifier.verify_support(Memory.from_dict(payload['memory']), episode, Support(**payload['support'])))
        return bool(verifier.verify_example(Example.from_dict(payload['example']), episode))
    if operation == 'http':
        from .models.http import ChatConfig, HttpChat
        values = dict(payload['config'], isolate_process=False)
        client = HttpChat(ChatConfig(**values))
        text = client.complete(payload['messages'], payload.get('model_override'))
        return {'text':text, 'usage':client.usage}
    if operation == 'train':
        from .contracts import Artifact, Example, Memory, Step
        from .learning.huggingface import HuggingFaceLearner, SFTConfig
        learner = HuggingFaceLearner(payload['root'], SFTConfig(**payload['config']), None)
        parent = payload.get('parent')
        if parent:
            artifact = Artifact(**parent)
            learner._validate_artifact(artifact)
            learner.artifacts[artifact.version] = artifact
        histories = {(row['task_id'], row['step']):tuple(Step(**s) for s in row['history'])
                     for row in payload['histories']}
        result = learner.train(tuple(Example.from_dict(e) for e in payload['examples']),
                               payload['parent_version'],
                               tuple(Memory.from_dict(m) for m in payload['memories']), histories)
        return asdict(result)
    raise ValueError('unknown worker operation')


def main():
    # Parent caps the request too. A finite protocol bound protects standalone use.
    for line in iter(lambda: sys.stdin.buffer.readline(67108865), b''):
        request = {}
        try:
            if len(line) > 67108864 or not line.endswith(b'\n'):
                raise ValueError('worker request exceeds protocol limit')
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = execute(request['op'], request['payload'])
            response = {'id':request['id'], 'result':result}
        except Exception as error:
            # Do not leak URLs, credentials, source text or response bodies.
            response = {'id':request.get('id'), 'error':type(error).__name__}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
