"""CLI entry point for `eval.repeat`.
`eval.repeat` 의 CLI 진입점. AI/ 디렉터리에서 실행한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_limits  # noqa: E402  — numpy/torch 앞에 와야 한다

runtime_limits.claim("repeat_runs")

from eval.repeat import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
