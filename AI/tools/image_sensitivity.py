"""CLI entry point for `eval.image_sensitivity`.
`eval.image_sensitivity` 의 CLI 진입점. AI/ 디렉터리에서 실행한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.image_sensitivity import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
