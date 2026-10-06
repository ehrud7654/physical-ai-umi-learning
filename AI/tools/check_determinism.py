"""CLI entry point for `eval.determinism`.
`eval.determinism` 의 CLI 진입점."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.determinism import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
