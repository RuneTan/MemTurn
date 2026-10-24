"""Exercise original environment reset/step through the process boundary; no LLM or training."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main(args):
    from memturn.contracts import Episode, Step, Task
    from memturn.experiments import environment_for
    spec=json.loads(Path(args.config).read_text(encoding='utf-8'))
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    records=[]
    for item in spec['tasks']:
        task=Task(**item); env=environment_for(spec['environment']); started=time.monotonic()
        try:
            observation=env.reset(task); steps=[]; reward=0.0; success=False
            for action in spec['actions']:
                result=env.step(action)
                steps.append(Step(observation,action,result.observation,result.accepted,result.evidence))
                observation=result.observation; reward+=result.reward; success=result.success
                if result.done: break
            episode=Episode(task.id,task.family,f'{task.family}:{task.version}',tuple(steps),reward,success,
                            elapsed_seconds=time.monotonic()-started,status='native_interface_check',task_spec=item)
            records.append(episode.to_dict())
            print(json.dumps({'task':task.id,'native_environment':task.family,'steps':len(steps)},ensure_ascii=False),flush=True)
        finally:
            env.close()
    (output/'episodes.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'report.json').write_text(json.dumps({'scope':'original native interface only; not task-learning results',
        'native_version':spec['native_version'],'episodes':len(records),'model_calls':0,'training_updates':0},indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    main(parser.parse_args())
