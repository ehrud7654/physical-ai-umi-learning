import json
import os
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.domain.enums.training_run_status import TrainingRunStatus
from app.domain.models.training_run import TrainingRun


class FileTrainingRunRepository:
    """DB가 아닌 실행별 JSON journal. 임시 파일 후 rename으로 원자 저장한다."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, run_id: str) -> Path:
        safe_name = run_id.replace(":", "__").replace("/", "_")
        return self.root / f"{safe_name}.json"

    def get(self, run_id: str) -> TrainingRun | None:
        path = self._path(run_id)
        with self._lock:
            if not path.exists():
                return None
            return self._decode(json.loads(path.read_text(encoding="utf-8")))

    def create(self, run: TrainingRun) -> bool:
        with self._lock:
            if self._path(run.run_id).exists():
                return False
            self._write(run)
            return True

    def save(self, run: TrainingRun) -> None:
        with self._lock:
            self._write(run)

    def list_active(self) -> list[TrainingRun]:
        return [run for run in self._all() if not run.status.terminal]

    def list_pending_completions(self) -> list[TrainingRun]:
        return [run for run in self._all() if run.status == TrainingRunStatus.COMPLETED and not run.completion_delivered]

    def _all(self) -> list[TrainingRun]:
        with self._lock:
            records: list[TrainingRun] = []
            for path in self.root.glob("*.json"):
                try:
                    records.append(self._decode(json.loads(path.read_text(encoding="utf-8"))))
                except (OSError, ValueError, TypeError):
                    continue
            return records

    def _write(self, run: TrainingRun) -> None:
        path = self._path(run.run_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(run), ensure_ascii=False, default=self._json_default, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _json_default(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, TrainingRunStatus):
            return value.value
        raise TypeError(f"Unsupported journal value: {type(value)}")

    @staticmethod
    def _decode(value: dict[str, Any]) -> TrainingRun:
        date_fields = (
            "execution_lease_expires_at", "created_at", "updated_at", "last_progress_at",
            "last_process_heartbeat_at", "model_completed_at", "next_completion_attempt_at",
        )
        for field in date_fields:
            if value.get(field):
                value[field] = datetime.fromisoformat(value[field])
        value["status"] = TrainingRunStatus(value["status"])
        return TrainingRun(**value)

