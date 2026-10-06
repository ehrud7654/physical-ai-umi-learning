import threading

from app.domain.errors import DomainError


class GpuResourceManager:
    def __init__(self, gpu_count: int):
        self.gpu_count = gpu_count
        self._owners: dict[int, str] = {}
        self._lock = threading.Lock()

    def reserve(self, gpu_index: int, run_id: str) -> None:
        if gpu_index >= self.gpu_count:
            raise DomainError(422, "GPU_INDEX_INVALID", "존재하지 않는 GPU 번호입니다.")
        with self._lock:
            owner = self._owners.get(gpu_index)
            if owner is not None and owner != run_id:
                raise DomainError(409, "GPU_BUSY", "선택한 GPU가 다른 학습에 사용 중입니다.")
            self._owners[gpu_index] = run_id

    def release(self, gpu_index: int, run_id: str) -> None:
        with self._lock:
            if self._owners.get(gpu_index) == run_id:
                self._owners.pop(gpu_index, None)

    def recover(self, gpu_index: int, run_id: str) -> None:
        with self._lock:
            self._owners.setdefault(gpu_index, run_id)

