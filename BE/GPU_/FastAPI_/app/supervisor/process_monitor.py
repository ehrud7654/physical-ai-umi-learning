import json
from datetime import datetime

from app.core.time import utc_now
from app.core.run_lifecycle import synchronized_run_changes
from app.domain.enums.training_run_status import TrainingRunStatus
from app.domain.models.training_run import TrainingRun
from app.domain.repositories import TrainingRunRepository
from app.infrastructure.gpu.gpu_resource_manager import GpuResourceManager
from app.infrastructure.process.process_inspector import ProcessInspector
from app.infrastructure.storage.shared_storage import SharedStorage


class ProcessMonitor:
    def __init__(self, repository: TrainingRunRepository, storage: SharedStorage,
                 inspector: ProcessInspector, gpu: GpuResourceManager):
        self.repository = repository
        self.storage = storage
        self.inspector = inspector
        self.gpu = gpu

    @synchronized_run_changes
    def refresh(self, run: TrainingRun) -> None:
        # The supervisor's snapshot may predate a concurrent cancellation.
        run = self.repository.get(run.run_id) or run
        if run.status.terminal or run.status == TrainingRunStatus.CANCEL_REQUESTED:
            return
        path = self.storage.run_directory(run.training_job_id, run.attempt) / "progress.json"
        if path.exists():
            try:
                progress = json.loads(path.read_text(encoding="utf-8"))
                incoming = int(progress.get("sequence", 0))
                if incoming > run.sequence:
                    run.sequence = incoming
                    run.status = TrainingRunStatus(progress["status"])
                    run.stage = progress["stage"]
                    run.percent = int(progress["percent"])
                    run.epoch = progress.get("epoch")
                    run.total_epochs = progress.get("totalEpochs")
                    run.last_progress_at = self._date(progress.get("lastProgressAt"))
                    run.last_process_heartbeat_at = self._date(progress.get("lastProcessHeartbeatAt"))
                    run.model_completed_at = self._date(progress.get("modelCompletedAt"))
                    run.result = progress.get("result")
                    run.error = progress.get("error")
                    run.updated_at = utc_now()
                    self.repository.save(run)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        if run.status.terminal:
            self.gpu.release(run.gpu_index, run.run_id)
            return
        if run.pid is not None and not self.inspector.running(run.pid):
            run.status = TrainingRunStatus.LOST
            run.stage = "FAILED"
            run.sequence += 1
            run.error = {"code": "EXECUTION_STATE_LOST", "message": "학습 프로세스 상태를 복구할 수 없습니다."}
            run.updated_at = utc_now()
            self.repository.save(run)
            self.gpu.release(run.gpu_index, run.run_id)

    @staticmethod
    def _date(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None

