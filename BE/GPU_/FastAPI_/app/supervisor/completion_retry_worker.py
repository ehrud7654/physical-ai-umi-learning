from app.application.completion_service import CompletionService
from app.domain.repositories import TrainingRunRepository


class CompletionRetryWorker:
    def __init__(self, repository: TrainingRunRepository, completion: CompletionService):
        self.repository = repository
        self.completion = completion

    async def run_once(self) -> None:
        for run in self.repository.list_pending_completions():
            await self.completion.deliver(run)

