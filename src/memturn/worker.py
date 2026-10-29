"""Private JSON process entry point; stdout contains protocol replies only."""
from dataclasses import asdict
import contextlib
import json
import sys


def execute(operation, payload):
    if operation == 'verify':
        import importlib
        from .contracts import Episode, Example, Memory, Support
        module, name = payload['factory'].split(':', 1)
        verifier = getattr(importlib.import_module(module), name)(**payload['kwargs'])
        episode = Episode.from_dict(payload['episode'])
        if payload['operation'] == 'support':
            return bool(verifier.verify_support(Memory.from_dict(payload['memory']), episode, Support(**payload['support'])))
        return bool(verifier.verify_example(Example.from_dict(payload['example']), episode))
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
