"""CLI entry point for `tracking.findings`.
`tracking.findings` 의 CLI 진입점. AI/ 디렉터리에서 실행한다.

    python tools/findings.py            # FINDINGS.md 재생성
    python tools/findings.py --brief    # 실행 전에 볼 짧은 성적표만 출력
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking.findings import brief, write  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brief", action="store_true", help="짧은 성적표만 찍고 끝낸다")
    args = p.parse_args()
    if args.brief:
        print(brief())
        return 0
    print(f"갱신: {write()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
