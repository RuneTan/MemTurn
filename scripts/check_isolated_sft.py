"""Two real SFT jobs on explicit fixtures; no benchmark or learning-effect claim."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memturn.contracts import Example
from memturn.learning.huggingface import SFTConfig
from memturn.learning.isolated import IsolatedHuggingFaceLearner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--loader', required=True)
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-seconds', type=float, default=300)
    parser.add_argument('--rss-limit-bytes', type=int)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.offline: os.environ['HF_HUB_OFFLINE'] = '1'
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = SFTConfig(args.model,args.revision,tuple(args.targets),loader_class=args.loader,
                       device=args.device,max_seconds=args.max_seconds)
    learner = IsolatedHuggingFaceLearner(root/'adapters',config,None,rss_limit_bytes=args.rss_limit_bytes)
    examples = (Example('',-1,'What does this engineering fixture require?', 'Check the public observation.',
                        example_type='rule_qa',output_kind='text'),
                Example('fixture-a',0,'Engineering fixture: the environment permits looking.','look'),
                Example('fixture-b',0,'Engineering fixture: the environment permits waiting.','wait'))
    artifacts, parent = [], 'base'
    for _ in range(2):
        artifact = learner.train(examples,parent)
        learner._validate_artifact(artifact)
        artifacts.append(asdict(artifact))
        parent = artifact.version
    report = {'scope':'real_isolated_gradient_and_continuation_fixture_only', 'jobs':artifacts,
              'config':asdict(config), 'native_task_learning':'not_tested',
              'remote_serving_publication':'not_tested', 'training_worker_persistence':'process_exits_after_each_job'}
    (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'jobs':len(artifacts),'coverage':[a['metrics']['sample_coverage'] for a in artifacts],
                      'updates':[a['metrics']['steps'] for a in artifacts],'report':str(root/'report.json')},ensure_ascii=False))


if __name__ == '__main__': main()
