"""Run original simulators in their own environment; no model, rewards or feedback rewriting."""
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from memturn.contracts import Task
    from memturn.experiments import environment_for
    environment = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                with redirect_stdout(sys.stderr):
                    if request['op'] == 'configure':
                        settings = dict(request['settings'])
                        if settings.get('java_home'):
                            os.environ['JAVA_HOME'] = settings.pop('java_home')
                            os.environ['PATH'] = os.environ['JAVA_HOME']+'/bin:'+os.environ.get('PATH','')
                        if settings.get('data_path'):
                            os.environ['ALFWORLD_DATA'] = settings.pop('data_path')
                        environment = environment_for(settings)
                        result = {'configured':True}
                    elif request['op'] == 'reset':
                        result = environment.reset(Task(**request['task']))
                    elif request['op'] == 'step':
                        result = asdict(environment.step(request['action']))
                    elif request['op'] == 'close':
                        if environment: environment.close()
                        result = {'closed':True}
                    else:
                        raise ValueError('unknown native-worker operation')
                print(json.dumps({'id':request['id'],'result':result},ensure_ascii=False),flush=True)
                if request['op'] == 'close': break
            except Exception as error:
                print(type(error).__name__+': '+str(error),file=sys.stderr,flush=True)
                print(json.dumps({'id':request.get('id'),'error':type(error).__name__}),flush=True)
    finally:
        if environment: environment.close()


if __name__ == '__main__':
    main()
