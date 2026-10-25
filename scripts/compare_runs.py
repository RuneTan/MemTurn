"""Read researcher reports offline; never feeds archives back into learning."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memturn.matrix import budget_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = []
    for folder in args.runs:
        root = Path(folder)
        profile = json.loads((root/'profile.json').read_text(encoding='utf-8'))
        report = json.loads((root/'report.json').read_text(encoding='utf-8'))
        stats = report['stats']
        rows.append({'arm':profile.get('experiment_arm','baseline'), 'budget_signature':budget_signature(profile),
            'manifest_sha256':report['manifest']['sha256'], 'completed':stats['completed'], 'utility':stats['utility'],
            'active_tokens':stats['active_tokens'], 'growth':stats['growth'], 'compute':stats['compute_ledger'],
            'service':report['service'], 'errors':report['errors']})
    same = len({(row['budget_signature'],row['manifest_sha256']) for row in rows}) == 1
    result = {'same_configured_budgets_and_stream':same, 'runs':rows,
              'note':'compare confirmed and conservatively charged compute separately; no significance claim'}
    with Path(args.output).open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'same_configured_budgets_and_stream':same,'runs':len(rows)}))


if __name__ == '__main__': main()
