from datetime import datetime

from app.schemas.checkpoint import CheckpointResult
from app.schemas.common import ApiModel


class CompletionRequest(ApiModel):
    event_id: str
    training_job_id: str
    attempt: int
    model_completed_at: datetime
    result: CheckpointResult

