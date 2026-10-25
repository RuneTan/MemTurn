"""Prepare reviewable configurations. No model calls or experiment execution."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memturn.matrix import build_matrix, budget_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    profiles = build_matrix(json.loads(Path(args.profile).read_text(encoding='utf-8')))
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for name, profile in profiles.items():
        (root/(name+'.json')).write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'matrix.json').write_text(json.dumps({'arms':list(profiles),
        'shared_budget_signature':budget_signature(next(iter(profiles.values()))),
        'status':'configurations_only_not_experiment_results',
        'comparison_note':'same configured caps do not guarantee equal realized training tokens or updates'},indent=2),encoding='utf-8')
    print(root)


if __name__ == '__main__': main()
