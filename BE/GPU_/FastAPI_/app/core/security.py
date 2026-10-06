import secrets

from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials


def verify_bearer(credentials: HTTPAuthorizationCredentials | None, expected: str) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token is required")
    if not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid service token")

