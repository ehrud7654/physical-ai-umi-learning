"""CLI entry point for `eval.trace_execution`.
`eval.trace_execution` 의 CLI 진입점."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.trace_execution import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
