import json
import os
from pathlib import Path
from typing import Any


class SharedStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def run_directory(self, training_job_id: str, attempt: int) -> Path:
        path = self.root / "runs" / training_job_id / str(attempt)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json_atomic(self, path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)

