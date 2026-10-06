import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.application.cancellation_service import CancellationService
from app.application.completion_service import CompletionService
from app.application.lease_service import LeaseService
from app.application.training_query_service import TrainingQueryService
from app.application.training_run_service import TrainingRunService
from app.core.config import get_settings
from app.infrastructure.callback.gpu_spring_client import GpuSpringClient
from app.infrastructure.gpu.gpu_resource_manager import GpuResourceManager
from app.infrastructure.persistence.file_run_repository import FileTrainingRunRepository
from app.infrastructure.process.process_controller import ProcessController
from app.infrastructure.process.process_inspector import ProcessInspector
from app.infrastructure.process.process_launcher import ProcessLauncher
from app.infrastructure.storage.path_validator import PathValidator
from app.infrastructure.storage.shared_storage import SharedStorage
from app.supervisor.completion_retry_worker import CompletionRetryWorker
from app.supervisor.lease_monitor import LeaseMonitor
from app.supervisor.process_monitor import ProcessMonitor
from app.supervisor.training_supervisor import TrainingSupervisor


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.shared_storage_root.mkdir(parents=True, exist_ok=True)
    repository = FileTrainingRunRepository(settings.run_journal_root)
    gpu = GpuResourceManager(settings.gpu_count)
    storage = SharedStorage(settings.shared_storage_root)
    controller = ProcessController()
    cancellation = CancellationService(repository, controller, gpu)
    completion = CompletionService(
        repository,
        GpuSpringClient(settings.gpu_spring_base_url, settings.gpu_spring_callback_token),
        settings,
    )
    supervisor = TrainingSupervisor(
        repository,
        ProcessMonitor(repository, storage, ProcessInspector(), gpu),
        LeaseMonitor(cancellation),
        CompletionRetryWorker(repository, completion),
        gpu,
        settings,
    )
    app.state.settings = settings
    app.state.training_run_service = TrainingRunService(
        repository, gpu, ProcessLauncher(), storage, PathValidator(settings.shared_storage_root), settings
    )
    app.state.training_query_service = TrainingQueryService(repository)
    app.state.cancellation_service = cancellation
    app.state.lease_service = LeaseService(repository)
    app.state.supervisor = supervisor
    task = asyncio.create_task(supervisor.run())
    try:
        yield
    finally:
        supervisor.stop()
        await task

