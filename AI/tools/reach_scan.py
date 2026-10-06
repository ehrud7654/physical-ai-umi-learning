"""CLI entry point for `sim.mujoco.workspace`.
`sim.mujoco.workspace` 의 CLI 진입점.

Run from the AI/ directory. The sys.path line is what lets
`python tools/reach_scan.py` work as well as `python -m tools.reach_scan` — it is
here, in the entry-point layer, and nowhere in the library code.
AI/ 디렉터리에서 실행한다. sys.path 한 줄은 `python -m tools.reach_scan` 뿐 아니라
`python tools/reach_scan.py` 도 되게 하려고 있다. 진입점 층에만 두고 라이브러리
코드에는 두지 않는다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mujoco.workspace import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
