import httpx
from urllib.parse import quote

from app.schemas.completion import CompletionRequest


class GpuSpringClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

    async def send_completion(self, run_id: str, request: CompletionRequest) -> bool:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=10.0) as client:
            response = await client.post(
                f"/internal/training-runs/{quote(run_id, safe='')}/completion",
                headers={"Authorization": f"Bearer {self.token}"},
                json=request.model_dump(mode="json", by_alias=True),
            )
        if response.status_code in (200, 202):
            return True
        if response.status_code == 409:
            response.raise_for_status()
        if response.status_code >= 500:
            return False
        response.raise_for_status()
        return False
