import asyncio
import logging

from app.core.config import Settings
from app.domain.repositories import TrainingRunRepository
from app.infrastructure.gpu.gpu_resource_manager import GpuResourceManager
from app.supervisor.completion_retry_worker import CompletionRetryWorker
from app.supervisor.lease_monitor import LeaseMonitor
from app.supervisor.process_monitor import ProcessMonitor

log = logging.getLogger(__name__)


class TrainingSupervisor:
    def __init__(self, repository: TrainingRunRepository, process_monitor: ProcessMonitor,
                 lease_monitor: LeaseMonitor, completion_worker: CompletionRetryWorker,
                 gpu: GpuResourceManager, settings: Settings):
        self.repository = repository
        self.process_monitor = process_monitor
        self.lease_monitor = lease_monitor
        self.completion_worker = completion_worker
        self.gpu = gpu
        self.settings = settings
        self._stopping = asyncio.Event()

    def recover(self) -> None:
        for run in self.repository.list_active():
            self.gpu.recover(run.gpu_index, run.run_id)
            self.process_monitor.refresh(run)

    async def run(self) -> None:
        self.recover()
        while not self._stopping.is_set():
            try:
                for run in self.repository.list_active():
                    await asyncio.to_thread(self.process_monitor.refresh, run)
                    current = self.repository.get(run.run_id)
                    if current:
                        await asyncio.to_thread(self.lease_monitor.enforce, current)
                await self.completion_worker.run_once()
            except Exception:
                log.exception("Training supervisor cycle failed")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.settings.supervisor_interval_seconds)
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stopping.set()

