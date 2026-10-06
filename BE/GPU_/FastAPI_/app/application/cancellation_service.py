from app.core.time import utc_now
from app.core.run_lifecycle import synchronized_run_changes
from app.domain.enums.training_run_status import TrainingRunStatus
from app.domain.errors import RunNotFound
from app.domain.models.training_run import TrainingRun
from app.domain.repositories import TrainingRunRepository
from app.infrastructure.gpu.gpu_resource_manager import GpuResourceManager
from app.infrastructure.process.process_controller import ProcessController


class CancellationService:
    def __init__(self, repository: TrainingRunRepository, controller: ProcessController, gpu: GpuResourceManager):
        self.repository = repository
        self.controller = controller
        self.gpu = gpu

    @synchronized_run_changes
    def cancel(self, run_id: str, reason: str) -> tuple[TrainingRun, bool]:
        run = self.repository.get(run_id)
        if run is None:
            raise RunNotFound()
        if run.status.terminal:
            return run, False
        run.status = TrainingRunStatus.CANCEL_REQUESTED
        run.sequence += 1
        run.updated_at = utc_now()
        self.repository.save(run)
        self.controller.terminate(run.pid)
        run.status = TrainingRunStatus.CANCELED
        run.stage = "CANCELED"
        run.sequence += 1
        run.updated_at = utc_now()
        run.error = {"code": "TRAINING_CANCELED", "message": reason}
        self.repository.save(run)
        self.gpu.release(run.gpu_index, run.run_id)
        return run, True

