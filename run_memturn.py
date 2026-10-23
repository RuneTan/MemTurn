"""Source-checkout entry point; installation is optional for offline checks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from memturn.cli import main

if __name__ == "__main__":
    main()
