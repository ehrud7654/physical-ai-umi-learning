from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from app.domain.enums.training_run_status import TrainingRunStatus
from app.schemas.checkpoint import CheckpointResult
from app.schemas.common import ApiModel


class BaseModelInput(ApiModel):
    skill_version_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    storage_key: str = Field(min_length=1)
    sha256: str
    architecture: str = Field(min_length=1)
    framework: str = Field(min_length=1)
    framework_version: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be a lowercase SHA-256")
        return value


class EpisodeInput(ApiModel):
    input_order: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    storage_key: str = Field(min_length=1)
    sha256: str
    contract_version: str = Field(min_length=1)

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be a lowercase SHA-256")
        return value


class TrainingInput(ApiModel):
    storage_key: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str
    report_storage_key: str = Field(min_length=1)
    report_sha256: str
    episode_count: int = Field(gt=0)
    training_input_status: str
    format: str
    world_frame: str = Field(min_length=1)

    @field_validator("sha256", "report_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("hash must be a lowercase SHA-256")
        return value

    @field_validator("training_input_status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value != "ready":
            raise ValueError("trainingInputStatus must be ready")
        return value

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        if value != "ZARR_ZIP":
            raise ValueError("format must be ZARR_ZIP")
        return value


class CreateTrainingRunRequest(ApiModel):
    training_job_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    worker_id: str = Field(min_length=1)
    storage_node_id: str = Field(min_length=1)
    gpu_index: int = Field(ge=0)
    training_mode: str = "FROM_SCRATCH"
    base_model: BaseModelInput | None = None
    input_snapshot_sha256: str
    episodes: list[EpisodeInput] = Field(min_length=1)
    training_input: TrainingInput
    execution_lease_expires_at: datetime

    @field_validator("input_snapshot_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("inputSnapshotSha256 must be a lowercase SHA-256")
        return value


class CreateTrainingRunResponse(ApiModel):
    run_id: str
    training_job_id: str
    attempt: int
    status: TrainingRunStatus


class RunError(ApiModel):
    code: str
    message: str


class TrainingRunResponse(ApiModel):
    run_id: str
    training_job_id: str
    attempt: int
    status: TrainingRunStatus
    sequence: int
    stage: str
    percent: int
    epoch: int | None = None
    total_epochs: int | None = None
    last_progress_at: datetime | None = None
    last_process_heartbeat_at: datetime | None = None
    model_completed_at: datetime | None = None
    result: CheckpointResult | None = None
    error: RunError | None = None


class CancelTrainingRunRequest(ApiModel):
    reason: str = Field(min_length=1, max_length=500)


class CancelTrainingRunResponse(ApiModel):
    run_id: str
    status: TrainingRunStatus


class RenewLeaseRequest(ApiModel):
    execution_lease_expires_at: datetime


class RenewLeaseResponse(ApiModel):
    run_id: str
    execution_lease_expires_at: datetime

