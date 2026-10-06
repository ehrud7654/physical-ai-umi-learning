"""CLI entry point; see src/umi_policy/cli.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from umi_policy.cli import main

if __name__ == "__main__":
    main()
