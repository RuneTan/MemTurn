"""Bounded local-only repair diagnostics; never resume or mutate the old run."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def previous_module(package, name, relative, revision, output):
    text = subprocess.check_output(['git', 'show', f'{revision}:{relative}'], cwd=ROOT).decode('utf-8')
    (output / Path(relative).name).write_text(text, encoding='utf-8')
    module = types.ModuleType(package + '.' + name)
    module.__package__ = package
    sys.modules[module.__name__] = module
    exec(compile(text, str(output / Path(relative).name), 'exec'), module.__dict__)
    return module


def main(args):
    import torch
    from memturn.config import Config
    from memturn.contracts import Artifact, Episode, Example, Memory, Task, dumps
    from memturn.learning.huggingface import HuggingFaceLearner, SFTConfig, supervised_tokens
    from memturn.memory.retrieval import Retriever
    from memturn.memory.store import Store
    from memturn.models.http import ChatPolicy
    from memturn.models.local import LocalHFChat, LocalModelConfig
    from memturn.runtime.service import Runtime
    from memturn.tasks.toy import PREPARE, ToyEnvironment, ToyVerifier
    from memturn.tokens import HuggingFaceCounter

    root, old = Path(args.output).resolve(), Path(args.run).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'validation.jsonl').exists():
        raise ValueError('use a new output directory; validation results are append-only')
    legacy = root / 'legacy'; legacy.mkdir(exist_ok=True)
    old_http = previous_module('memturn.models', '_repair_legacy_http', 'src/memturn/models/http.py', args.baseline_revision, legacy)
    old_service = previous_module('memturn.runtime', '_repair_legacy_service', 'src/memturn/runtime/service.py', args.baseline_revision, legacy)
    old_retrieval = previous_module('memturn.memory', '_repair_legacy_retrieval', 'src/memturn/memory/retrieval.py', args.baseline_revision, legacy)
    spec = json.loads((old / 'experiment.json').read_text(encoding='utf-8'))
    settings, training = spec['profile']['model'], spec['profile']['training']
    audit = [json.loads(line) for line in (old / 'audit.jsonl').read_text(encoding='utf-8').splitlines()]
    last = [r for r in audit if r['event'] == 'adapter_published'][-1]
    original_adapter = Artifact(**{k:last[k] for k in ('version','path','backend','parent_version','metrics')})
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT/'src').rglob('*.py')}
    (root/'source_manifest.json').write_text(dumps({'baseline_revision':args.baseline_revision, 'source_hashes':source_hashes,
        'scope':'fixed-bank local diagnostics, not an independent long-run baseline', 'source_run':str(old)}), encoding='utf-8')
    backend = LocalHFChat(LocalModelConfig(str(ROOT / settings['path']), settings['device'], settings['dtype'], settings['max_input_tokens']))
    counter = HuggingFaceCounter(settings['id'], settings['revision'], backend.tokenizer)
    backend.load(original_adapter)
    bank = tuple(Memory.from_dict(m) for m in json.loads((old/'bank_0182.json').read_text(encoding='utf-8')))
    rows = []
    with (root/'validation.jsonl').open('w', encoding='utf-8') as log:
        def record(row):
            rows.append(row); log.write(dumps(row)+'\n'); log.flush()
            print(dumps({k:v for k,v in row.items() if k in ('stage','regime','condition','sample','success','correct','reward')}), flush=True)

        if not args.training_only:
            regimes = (('legacy',False,False,'base',True), ('scope_only',False,True,'base',True),
                       ('transcript_only',True,False,'base',True), ('both',True,True,'base',True),
                       ('both_adapter',True,True,original_adapter.version,True),
                       ('base_no_memory',True,True,'base',False),
                       ('adapter_no_memory',True,True,original_adapter.version,False))
            for regime, transcript, scoped, model_version, memory_enabled in regimes:
                store = Store(root/f'{regime}.sqlite', Config(**spec['profile']['runtime']), counter)
                for memory in bank: store.append(memory)
                before = store.stats()
                runtime = (Runtime if transcript else old_service.Runtime)(store, None, None, None)
                inner = Retriever(counter, store.config) if scoped else old_retrieval.Retriever(counter, store.config)
                class Search:
                    scope = None
                    def search(self, memories, query, **kwargs):
                        if scoped:
                            kwargs['applicability'] = self.scope
                        else:
                            kwargs.pop('applicability',None)
                        return inner.search(memories,query,**kwargs)
                search = Search(); runtime.retriever = search
                policy = (ChatPolicy if transcript else old_http.ChatPolicy)(backend.role('repair_evaluation',settings['task_max_new_tokens']))
                class Forced:
                    first = None
                    def act(self, observation, memories, version='base', history=()):
                        action = policy.act(observation,memories,model_version,history)
                        if self.first is None: self.first = action
                        return action
                try:
                    for ci, condition in enumerate(PREPARE):
                        family, version = condition.split(':'); search.scope = condition
                        for sample in range(2):
                            forced = Forced()
                            task = Task(f'{regime}_{ci}_{sample}', family, version, 1000000+ci*10+sample,
                                        metadata={'evaluation_only':True,'memory_enabled':memory_enabled})
                            ep = runtime.run_task(task,ToyEnvironment(),forced)
                            ep = replace(ep, model_version=model_version)
                            record({'stage':'retrieval','regime':regime,'condition':condition,'sample':sample,
                                    'model_version':model_version,
                                    'success':ep.success,'reward':ep.reward,'status':ep.status,
                                    'first_decision':forced.first,'trace':ep.to_dict()})
                    if store.stats() != before: raise AssertionError('evaluation changed the learning state')
                finally:
                    runtime.close(); store.close()

        source = [Episode.from_dict({k:r[k] for k in ('task_id','family','condition','steps','reward','success','model_version','elapsed_seconds','status')})
                  for r in audit if r['event']=='task_completed']
        first = source[:4]
        if args.training_data == 'balanced_rules':
            first = []
            for condition in PREPARE:
                first.extend([ep for ep in source if ep.condition == condition and ep.success][:2])
        clean = []
        for ep in first:
            index = next(i for i,s in enumerate(ep.steps) if s.accepted)
            example = Example(ep.task_id,index,ep.steps[0].observation,ep.steps[index].action,input_step=0)
            if not ToyVerifier().verify_example(example,ep): raise AssertionError('clean projection failed replay')
            clean.append(example)
        values = dict(training,model_id=str(ROOT/settings['path']),model_revision=settings['revision'],device=settings['device'],dtype=settings['dtype'])
        values['target_modules'] = tuple(values['target_modules'])
        if args.training_steps is not None: values['steps'] = args.training_steps
        if args.training_epochs is not None:
            values.pop('steps',None)
            values['epochs'] = args.training_epochs
        if args.training_token_budget is not None: values['max_training_tokens'] = args.training_token_budget
        (root/'training_config.json').write_text(dumps(dict(values,data=args.training_data)),encoding='utf-8')
        learner = HuggingFaceLearner(root/'adapters',SFTConfig(**values),backend)
        for example in clean:
            ids, labels = supervised_tokens(backend.tokenizer,example.prompt,example.target,values['max_length'])
            prefix = next(i for i,t in enumerate(labels) if t != -100)
            if 'Required next action:' in backend.tokenizer.decode(ids[:prefix]):
                raise AssertionError('corrective feedback leaked into clean training input')
        (root/'training_examples.json').write_text(dumps([asdict(e) for e in clean]),encoding='utf-8')
        torch.manual_seed(42)
        artifact = learner.train(tuple(clean),'base',histories={(e.task_id,e.step):() for e in clean})
        learner.prepare(artifact)
        record({'stage':'training','regime':'clean_full_coverage','metrics':artifact.metrics,'artifact':asdict(artifact)})
        trained_label = (f'clean_{values["steps"]}_step' if values.get('steps') is not None
                         else f'clean_{values.get("epochs",1)}_epoch')
        for label, version in (('base','base'),(trained_label,artifact.version)):
            policy = ChatPolicy(backend.role('training_evaluation',settings['task_max_new_tokens']))
            for ci, condition in enumerate(PREPARE):
                family, version_label = condition.split(':')
                for sample in range(2):
                    env=ToyEnvironment()
                    observation=env.reset(Task(f'clean_{ci}_{sample}',family,version_label,1000000+ci*10+sample))
                    action=policy.act(observation,(),version)
                    record({'stage':'first_action','regime':label,'condition':condition,'sample':sample,
                            'correct':action==PREPARE[condition],'action':action})
                    env.close()

        groups=defaultdict(list)
        for row in rows:
            if row['stage'] in ('retrieval','first_action'):
                groups[(row['stage'],row['regime'])].append(row)
        summary={'scope':'local-only targeted repair validation; four rules, fixed bank; no Groq calls',
                 'regimes':[{'stage':stage,'regime':regime,'tasks':len(items),
                    'successes':sum(r.get('success',r.get('correct',False)) for r in items),
                    'mean_reward':sum(r.get('reward',0) for r in items)/len(items)} for (stage,regime),items in groups.items()],
                 'training_metrics':artifact.metrics,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
        (root/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='runs/chain500_group20_20261023')
    parser.add_argument('--output',required=True)
    parser.add_argument('--baseline-revision',default='c0a1d01')
    parser.add_argument('--training-only',action='store_true')
    parser.add_argument('--training-data',choices=('first_group','balanced_rules'),default='first_group')
    training_length=parser.add_mutually_exclusive_group()
    training_length.add_argument('--training-steps',type=int)
    training_length.add_argument('--training-epochs',type=int)
    parser.add_argument('--training-token-budget',type=int)
    main(parser.parse_args())
