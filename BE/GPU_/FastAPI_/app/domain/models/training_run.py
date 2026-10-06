from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.domain.enums.training_run_status import TrainingRunStatus


@dataclass
class TrainingRun:
    run_id: str
    training_job_id: str
    attempt: int
    worker_id: str
    storage_node_id: str
    gpu_index: int
    request_hash: str
    request: dict[str, Any]
    status: TrainingRunStatus
    sequence: int
    stage: str
    percent: int
    execution_lease_expires_at: datetime
    created_at: datetime
    updated_at: datetime
    pid: int | None = None
    epoch: int | None = None
    total_epochs: int | None = None
    last_progress_at: datetime | None = None
    last_process_heartbeat_at: datetime | None = None
    model_completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    completion_event_id: str | None = None
    completion_delivered: bool = False
    next_completion_attempt_at: datetime | None = None

