"""Read-only hardware/dependency inventory; never prints credential values."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memturn.runtime.resources import hardware_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', default='.')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = json.dumps(hardware_report(args.path), ensure_ascii=False, indent=2)
    if args.output:
        with Path(args.output).open('x', encoding='utf-8') as stream: stream.write(report + '\n')
    else:
        print(report)


if __name__ == '__main__': main()
