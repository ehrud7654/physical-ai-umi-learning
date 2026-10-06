from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=lambda value: _camel(value), populate_by_name=True)


def _camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(word.capitalize() for word in tail)


class ApiResponse(ApiModel, Generic[T]):
    data: T


class ApiError(ApiModel):
    code: str
    message: str


class ErrorEnvelope(ApiModel):
    error: ApiError

