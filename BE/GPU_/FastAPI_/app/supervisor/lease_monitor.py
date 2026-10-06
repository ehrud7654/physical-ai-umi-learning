from app.application.cancellation_service import CancellationService
from app.core.time import utc_now
from app.domain.models.training_run import TrainingRun


class LeaseMonitor:
    def __init__(self, cancellation: CancellationService):
        self.cancellation = cancellation

    def enforce(self, run: TrainingRun) -> None:
        if not run.status.terminal and run.execution_lease_expires_at <= utc_now():
            self.cancellation.cancel(run.run_id, "실행 임대가 만료되었습니다.")

