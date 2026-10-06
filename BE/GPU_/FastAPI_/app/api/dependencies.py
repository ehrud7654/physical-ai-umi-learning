from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.application.cancellation_service import CancellationService
from app.application.lease_service import LeaseService
from app.application.training_query_service import TrainingQueryService
from app.application.training_run_service import TrainingRunService
from app.core.config import get_settings
from app.core.security import verify_bearer

bearer = HTTPBearer(auto_error=False)


def require_service_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    verify_bearer(credentials, get_settings().service_token)


def training_run_service(request: Request) -> TrainingRunService:
    return request.app.state.training_run_service


def training_query_service(request: Request) -> TrainingQueryService:
    return request.app.state.training_query_service


def cancellation_service(request: Request) -> CancellationService:
    return request.app.state.cancellation_service


def lease_service(request: Request) -> LeaseService:
    return request.app.state.lease_service

