from datetime import datetime, timezone

from app.core.time import utc_now
from app.core.run_lifecycle import synchronized_run_changes
from app.domain.errors import DomainError, RunNotFound
from app.domain.models.training_run import TrainingRun
from app.domain.repositories import TrainingRunRepository


class LeaseService:
    def __init__(self, repository: TrainingRunRepository):
        self.repository = repository

    @synchronized_run_changes
    def renew(self, run_id: str, expires_at: datetime) -> TrainingRun:
        run = self.repository.get(run_id)
        if run is None:
            raise RunNotFound()
        if run.status.terminal:
            raise DomainError(409, "RUN_STATE_CONFLICT", "종료된 실행의 임대를 갱신할 수 없습니다.")
        normalized = expires_at.astimezone(timezone.utc)
        if normalized <= utc_now():
            raise DomainError(409, "EXECUTION_LEASE_EXPIRED", "실행 임대가 이미 만료되었습니다.")
        run.execution_lease_expires_at = normalized
        run.updated_at = utc_now()
        self.repository.save(run)
        return run

