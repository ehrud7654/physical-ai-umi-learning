from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import (
    cancellation_service,
    lease_service,
    require_service_token,
    training_query_service,
    training_run_service,
)
from app.application.cancellation_service import CancellationService
from app.application.lease_service import LeaseService
from app.application.training_query_service import TrainingQueryService
from app.application.training_run_service import TrainingRunService
from app.domain.models.training_run import TrainingRun
from app.schemas.checkpoint import CheckpointResult
from app.schemas.common import ApiResponse
from app.schemas.training_run import (
    CancelTrainingRunRequest,
    CancelTrainingRunResponse,
    CreateTrainingRunRequest,
    CreateTrainingRunResponse,
    RenewLeaseRequest,
    RenewLeaseResponse,
    RunError,
    TrainingRunResponse,
)

router = APIRouter(prefix="/internal/training-runs", dependencies=[Depends(require_service_token)])


@router.post("", response_model=ApiResponse[CreateTrainingRunResponse], status_code=status.HTTP_202_ACCEPTED)
def create_training_run(request: CreateTrainingRunRequest, response: Response,
                        service: TrainingRunService = Depends(training_run_service)) -> ApiResponse[CreateTrainingRunResponse]:
    run, created = service.create(request)
    response.status_code = status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK
    return ApiResponse(data=CreateTrainingRunResponse(
        run_id=run.run_id, training_job_id=run.training_job_id, attempt=run.attempt, status=run.status
    ))


@router.get("/{run_id}", response_model=ApiResponse[TrainingRunResponse])
def get_training_run(run_id: str, service: TrainingQueryService = Depends(training_query_service)) -> ApiResponse[TrainingRunResponse]:
    return ApiResponse(data=_response(service.get(run_id)))


@router.post("/{run_id}/cancel", response_model=ApiResponse[CancelTrainingRunResponse], status_code=status.HTTP_202_ACCEPTED)
def cancel_training_run(run_id: str, request: CancelTrainingRunRequest, response: Response,
                        service: CancellationService = Depends(cancellation_service)) -> ApiResponse[CancelTrainingRunResponse]:
    run, accepted = service.cancel(run_id, request.reason)
    response.status_code = status.HTTP_202_ACCEPTED if accepted else status.HTTP_200_OK
    return ApiResponse(data=CancelTrainingRunResponse(run_id=run.run_id, status=run.status))


@router.patch("/{run_id}/lease", response_model=ApiResponse[RenewLeaseResponse])
def renew_training_lease(run_id: str, request: RenewLeaseRequest,
                         service: LeaseService = Depends(lease_service)) -> ApiResponse[RenewLeaseResponse]:
    run = service.renew(run_id, request.execution_lease_expires_at)
    return ApiResponse(data=RenewLeaseResponse(
        run_id=run.run_id, execution_lease_expires_at=run.execution_lease_expires_at
    ))


def _response(run: TrainingRun) -> TrainingRunResponse:
    return TrainingRunResponse(
        run_id=run.run_id, training_job_id=run.training_job_id, attempt=run.attempt,
        status=run.status, sequence=run.sequence, stage=run.stage, percent=run.percent,
        epoch=run.epoch, total_epochs=run.total_epochs, last_progress_at=run.last_progress_at,
        last_process_heartbeat_at=run.last_process_heartbeat_at,
        model_completed_at=run.model_completed_at,
        result=CheckpointResult.model_validate(run.result) if run.result else None,
        error=RunError.model_validate(run.error) if run.error else None,
    )

