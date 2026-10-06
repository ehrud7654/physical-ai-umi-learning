"""CLI entry point for `sim.mujoco.view`.
`sim.mujoco.view` 의 CLI 진입점.

Run from the AI/ directory. Needs a display — do NOT set MUJOCO_GL=egl.
AI/ 디렉터리에서 실행한다. 디스플레이가 필요하므로 MUJOCO_GL=egl 을 두지 않는다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mujoco.view import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
