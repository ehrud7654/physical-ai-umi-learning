import logging
from datetime import timedelta

import httpx

from app.core.config import Settings
from app.core.time import utc_now
from app.domain.models.training_run import TrainingRun
from app.domain.repositories import TrainingRunRepository
from app.infrastructure.callback.gpu_spring_client import GpuSpringClient
from app.schemas.checkpoint import CheckpointResult
from app.schemas.completion import CompletionRequest

log = logging.getLogger(__name__)


class CompletionService:
    def __init__(self, repository: TrainingRunRepository, client: GpuSpringClient, settings: Settings):
        self.repository = repository
        self.client = client
        self.settings = settings

    async def deliver(self, run: TrainingRun) -> None:
        now = utc_now()
        if run.next_completion_attempt_at and run.next_completion_attempt_at > now:
            return
        if not run.result or not run.model_completed_at:
            return
        if not run.completion_event_id:
            run.completion_event_id = f"completion_{run.training_job_id}_{run.attempt}"
            self.repository.save(run)
        request = CompletionRequest(
            event_id=run.completion_event_id,
            training_job_id=run.training_job_id,
            attempt=run.attempt,
            model_completed_at=run.model_completed_at,
            result=CheckpointResult.model_validate(run.result),
        )
        try:
            run.completion_delivered = await self.client.send_completion(run.run_id, request)
        except (httpx.TransportError, httpx.TimeoutException):
            run.completion_delivered = False
        except httpx.HTTPStatusError as error:
            log.error("Completion rejected for %s: %s", run.run_id, error.response.status_code)
            run.completion_delivered = False
            if error.response.status_code == 409:
                run.error = {"code": "COMPLETION_CONFLICT", "message": error.response.text}
                run.completion_delivered = True
                run.next_completion_attempt_at = None
                self.repository.save(run)
                return
        run.next_completion_attempt_at = None if run.completion_delivered else now + timedelta(seconds=self.settings.completion_retry_seconds)
        run.updated_at = utc_now()
        self.repository.save(run)
