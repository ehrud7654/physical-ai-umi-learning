"""CLI entry point for `eval.rollout`.
`eval.rollout` 의 CLI 진입점.

Run from the AI/ directory. The sys.path line is what lets
`python tools/eval_rollout.py` work as well as `python -m tools.eval_rollout` — it is
here, in the entry-point layer, and nowhere in the library code.
AI/ 디렉터리에서 실행한다. sys.path 한 줄은 `python -m tools.eval_rollout` 뿐 아니라
`python tools/eval_rollout.py` 도 되게 하려고 있다. 진입점 층에만 두고 라이브러리
코드에는 두지 않는다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from tools.run_umi_regression import is_shared_gpu_server

    if is_shared_gpu_server():
        raise SystemExit(
            "Shared GPU server is training-only; run rollout locally")

    from eval.rollout import main

    raise SystemExit(main())
