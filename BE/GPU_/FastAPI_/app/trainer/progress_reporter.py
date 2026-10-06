import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProgressReporter:
    def __init__(self, path: Path):
        self.path = path
        self.sequence = 0

    def report(self, status: str, stage: str, percent: int, **extra: Any) -> None:
        self.sequence += 1
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        value = {
            "status": status,
            "stage": stage,
            "percent": percent,
            "sequence": self.sequence,
            "lastProgressAt": now,
            "lastProcessHeartbeatAt": now,
            **extra,
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self.path)

