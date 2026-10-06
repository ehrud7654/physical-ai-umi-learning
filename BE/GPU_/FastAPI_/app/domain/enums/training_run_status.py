from enum import StrEnum


class TrainingRunStatus(StrEnum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELED = "CANCELED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    LOST = "LOST"

    @property
    def terminal(self) -> bool:
        return self in {self.CANCELED, self.COMPLETED, self.FAILED, self.LOST}

