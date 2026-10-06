"""CLI entry point for `sim.mujoco.angle_contract`.
`sim.mujoco.angle_contract` 의 CLI 진입점.

Run from the AI/ directory. Exit code 1 means a degree-based control API can
produce a command that our contract validator would reject.
AI/ 디렉터리에서 실행한다. 종료코드 1 은 degree 기반 제어 API 가 우리 검증기가
거부할 명령을 만들 수 있다는 뜻이다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.mujoco.angle_contract import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
