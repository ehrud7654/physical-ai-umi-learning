from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain.errors import DomainError


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error(_: Request, error: DomainError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content={"error": {"code": error.code, "message": error.message}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": {"code": "VALIDATION_ERROR", "message": str(error)}})

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, error: HTTPException) -> JSONResponse:
        code = "UNAUTHORIZED" if error.status_code == 401 else "FORBIDDEN" if error.status_code == 403 else "HTTP_ERROR"
        return JSONResponse(status_code=error.status_code, content={"error": {"code": code, "message": str(error.detail)}})
