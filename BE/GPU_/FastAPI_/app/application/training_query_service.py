from app.domain.errors import RunNotFound
from app.core.run_lifecycle import synchronized_run_changes
from app.domain.models.training_run import TrainingRun
from app.domain.repositories import TrainingRunRepository


class TrainingQueryService:
    def __init__(self, repository: TrainingRunRepository):
        self.repository = repository

    @synchronized_run_changes
    def get(self, run_id: str) -> TrainingRun:
        run = self.repository.get(run_id)
        if run is None:
            raise RunNotFound()
        return run

