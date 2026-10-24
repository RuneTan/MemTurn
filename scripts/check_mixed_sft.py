"""Real LM gradient/publication interface check, NOT native memory-learning validation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main(args):
    from memturn.contracts import Episode, Example, Memory, Support
    from memturn.learning.huggingface import HuggingFaceLearner, SFTConfig
    from memturn.learning.materials import bind_example, rule_question, training_messages
    from memturn.models.local import LocalHFChat, LocalModelConfig
    episodes=tuple(Episode.from_dict(e) for e in json.loads(Path(args.episodes).read_text(encoding='utf-8')))
    if not episodes or any(e.family not in ('scienceworld','alfworld') or not e.steps for e in episodes):
        raise ValueError('provide source episodes recorded from original native environments')
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
    # Explicit diagnostic fixture: not emitted/approved by a Consolidator or domain verifier.
    memory=Memory('interface_probe','Check the public observation before choosing an environment command.',
                  'At the beginning of an environment task',tuple(Support(e.task_id,0) for e in episodes),
                  consolidated=True)
    examples=(rule_question(memory),*(bind_example(Example(e.task_id,0,e.steps[0].observation,e.steps[0].action),memory)
                                      for e in episodes))
    memory=replace(memory,examples=examples)
    (output/'diagnostic_material.json').write_text(json.dumps(memory.to_dict(),ensure_ascii=False,indent=2),encoding='utf-8')
    model=str(ROOT/'models/qwen3-1.7b')
    serving=LocalHFChat(LocalModelConfig(model))
    cfg=SFTConfig(model,'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',('q_proj','v_proj'),
                  loader_class='AutoModelForCausalLM',max_training_tokens=16384,max_seconds=300)
    learner=HuggingFaceLearner(output/'adapters',cfg,serving)
    artifacts=[]; parent='base'
    for _ in range(args.jobs):
        artifact=learner.train(examples,parent)
        learner.prepare(artifact)
        artifacts.append(artifact)
        learner.retain({artifact.version})
        parent=artifact.version
    qa=examples[0]
    response=serving.complete(training_messages(qa.prompt),role='interface_qa',max_new_tokens=96,
                              model_override=artifacts[-1].version)
    report={'scope':'mixed-target LM optimizer and adapter-load check only; diagnostic QA is not domain-verified; no memory retired',
            'native_source_tasks':[e.task_id for e in episodes],
            'jobs':[a.__dict__ for a in artifacts], 'qa_response':response,
            'memory_retirements':0,'serving_usage':serving.usage}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'jobs':len(artifacts),'optimizer_updates':[a.metrics['steps'] for a in artifacts],
                      'examples_by_type':artifacts[-1].metrics['examples_by_type'],
                      'sample_coverage':[a.metrics['sample_coverage'] for a in artifacts],
                      'scope':report['scope']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if args.jobs<1: parser.error('--jobs must be positive')
    main(args)
