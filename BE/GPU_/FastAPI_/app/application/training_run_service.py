from datetime import timezone
import json
import logging

from app.core.canonical_json import sha256_hex
from app.core.config import Settings
from app.core.time import utc_now
from app.core.run_lifecycle import synchronized_run_changes
from app.domain.enums.training_run_status import TrainingRunStatus
from app.domain.errors import DomainError
from app.domain.models.training_run import TrainingRun
from app.domain.repositories import TrainingRunRepository
from app.infrastructure.gpu.gpu_resource_manager import GpuResourceManager
from app.infrastructure.process.process_launcher import ProcessLauncher
from app.infrastructure.storage.checksum import sha256_file
from app.infrastructure.storage.path_validator import PathValidator
from app.infrastructure.storage.shared_storage import SharedStorage
from app.schemas.training_run import CreateTrainingRunRequest


class TrainingRunService:
    def __init__(self, repository: TrainingRunRepository, gpu: GpuResourceManager, launcher: ProcessLauncher,
                 storage: SharedStorage, paths: PathValidator, settings: Settings):
        self.repository = repository
        self.gpu = gpu
        self.launcher = launcher
        self.storage = storage
        self.paths = paths
        self.settings = settings

    @synchronized_run_changes
    def create(self, request: CreateTrainingRunRequest) -> tuple[TrainingRun, bool]:
        self._validate(request)
        run_id = f"{request.training_job_id}:{request.attempt}"
        immutable = request.model_dump(mode="json", by_alias=True, exclude={"execution_lease_expires_at"})
        request_hash = sha256_hex(immutable)
        existing = self.repository.get(run_id)
        if existing:
            if existing.request_hash != request_hash or existing.worker_id != request.worker_id:
                raise DomainError(409, "RUN_REQUEST_CONFLICT", "같은 runId에 다른 입력이 요청되었습니다.")
            return existing, False

        self.gpu.reserve(request.gpu_index, run_id)
        now = utc_now()
        run = TrainingRun(
            run_id=run_id, training_job_id=request.training_job_id, attempt=request.attempt,
            worker_id=request.worker_id, storage_node_id=request.storage_node_id,
            gpu_index=request.gpu_index, request_hash=request_hash,
            request=request.model_dump(mode="json", by_alias=True), status=TrainingRunStatus.STARTING,
            sequence=0, stage="STARTING", percent=0,
            execution_lease_expires_at=request.execution_lease_expires_at.astimezone(timezone.utc),
            created_at=now, updated_at=now,
        )
        try:
            if not self.repository.create(run):
                return self.repository.get(run_id) or run, False
            directory = self.storage.run_directory(request.training_job_id, request.attempt)
            self.storage.write_json_atomic(directory / "request.json", run.request)
            run.pid = self.launcher.launch(
                run_id, directory, request.gpu_index, self.settings.training_command_duration_seconds
            )
            run.updated_at = utc_now()
            self.repository.save(run)
            return run, True
        except Exception:
            self.gpu.release(request.gpu_index, run_id)
            raise

    def _validate(self, request: CreateTrainingRunRequest) -> None:
        if request.worker_id != self.settings.worker_id or request.storage_node_id != self.settings.storage_node_id:
            logging.getLogger(__name__).warning(
                "Worker scope mismatch: requested workerId=%s storageNodeId=%s; configured workerId=%s storageNodeId=%s",
                request.worker_id, request.storage_node_id, self.settings.worker_id, self.settings.storage_node_id,
            )
            raise DomainError(403, "WORKER_SCOPE_MISMATCH", "인증된 worker 또는 StorageNode 범위와 다릅니다.")
        if request.execution_lease_expires_at.astimezone(timezone.utc) <= utc_now():
            raise DomainError(409, "EXECUTION_LEASE_EXPIRED", "실행 임대가 이미 만료되었습니다.")
        if request.training_mode not in {"FROM_SCRATCH", "FINE_TUNE"}:
            raise DomainError(422, "INVALID_TRAINING_MODE", "지원하지 않는 학습 모드입니다.")
        if request.training_mode == "FINE_TUNE" and request.base_model is None:
            raise DomainError(422, "BASE_MODEL_REQUIRED", "추가 학습에는 기준 모델이 필요합니다.")
        if request.training_mode == "FROM_SCRATCH" and request.base_model is not None:
            raise DomainError(422, "BASE_MODEL_NOT_ALLOWED", "처음부터 학습에는 기준 모델을 사용할 수 없습니다.")
        if request.base_model is not None:
            path = self.paths.resolve_existing_file(request.base_model.storage_key)
            if sha256_file(path) != request.base_model.sha256:
                raise DomainError(422, "BASE_MODEL_HASH_MISMATCH", "기준 모델 hash가 일치하지 않습니다.")
        orders = [episode.input_order for episode in request.episodes]
        if len(orders) != len(set(orders)):
            raise DomainError(422, "INPUT_ORDER_DUPLICATED", "Episode inputOrder가 중복되었습니다.")
        for episode in request.episodes:
            path = self.paths.resolve_existing_file(episode.storage_key)
            if sha256_file(path) != episode.sha256:
                raise DomainError(422, "EPISODE_HASH_MISMATCH", f"Episode hash가 일치하지 않습니다: {episode.episode_id}")
        training_input = self.paths.resolve_existing_file(request.training_input.storage_key)
        if training_input.stat().st_size != request.training_input.size_bytes:
            raise DomainError(422, "TRAINING_INPUT_SIZE_MISMATCH", "전처리 학습 입력 크기가 일치하지 않습니다.")
        if sha256_file(training_input) != request.training_input.sha256:
            raise DomainError(422, "TRAINING_INPUT_HASH_MISMATCH", "전처리 학습 입력 hash가 일치하지 않습니다.")
        report_path = self.paths.resolve_existing_file(request.training_input.report_storage_key)
        if sha256_file(report_path) != request.training_input.report_sha256:
            raise DomainError(422, "TRAINING_REPORT_HASH_MISMATCH", "전처리 보고서 hash가 일치하지 않습니다.")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DomainError(422, "TRAINING_REPORT_INVALID", "전처리 보고서를 읽을 수 없습니다.") from error
        if report.get("status") != "pass" or report.get("training_input_status") != "ready":
            raise DomainError(422, "TRAINING_INPUT_NOT_READY", "검증을 통과한 Zarr만 학습할 수 있습니다.")
        if report.get("episodes") != request.training_input.episode_count:
            raise DomainError(422, "TRAINING_INPUT_EPISODE_MISMATCH", "Zarr Episode 수가 일치하지 않습니다.")
        if report.get("world_frame") != request.training_input.world_frame:
            raise DomainError(422, "TRAINING_INPUT_WORLD_FRAME_MISMATCH", "Zarr 좌표계 정보가 일치하지 않습니다.")

