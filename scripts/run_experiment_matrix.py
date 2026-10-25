"""Run explicitly selected, fully configured arms sequentially."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memturn.experiments import inspect_profile, run_profile
from memturn.matrix import ARMS, budget_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arms', nargs='+', choices=tuple(ARMS), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    paths = {arm:Path(args.matrix)/(arm+'.json') for arm in args.arms}
    profiles = {arm:json.loads(path.read_text(encoding='utf-8')) for arm,path in paths.items()}
    for arm,profile in profiles.items():
        missing = inspect_profile(profile)
        if missing: raise ValueError(arm + ' has unresolved settings: ' + ', '.join(missing))
    if len({budget_signature(profile) for profile in profiles.values()}) != 1:
        raise ValueError('selected arms do not share the configured budgets and workload')
    root = Path(args.output)
    for arm,path in paths.items():
        output = root/arm
        resume = args.resume and (output/'state.sqlite').exists()
        report = run_profile(path,output,resume=resume)
        print(json.dumps({'arm':arm,'completed':report['stats']['completed'],'utility':report['stats']['utility']}),flush=True)


if __name__ == '__main__': main()
