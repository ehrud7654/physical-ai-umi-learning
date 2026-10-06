from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(prefix="/health")


@router.get("/live")
def live() -> dict[str, str]:
    return {"status": "UP"}


@router.get("/ready")
def ready(request: Request) -> dict[str, str]:
    root: Path = request.app.state.settings.shared_storage_root
    probe = root / ".fastapi-ready"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)
    return {"status": "UP"}

