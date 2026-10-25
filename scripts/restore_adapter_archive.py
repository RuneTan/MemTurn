"""Restore archived bytes to a new directory without publishing a model."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memturn.learning.archive import restore_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('destination', help='must not exist; current checkpoints are never overwritten')
    args = parser.parse_args()
    print(restore_archive(args.archive, args.destination))


if __name__ == '__main__':
    main()
