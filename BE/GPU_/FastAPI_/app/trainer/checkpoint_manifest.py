import json
import tarfile
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "architecture", "framework", "frameworkVersion", "contractVersion",
    "robotProfileId", "actionSpace", "actionSpec", "runtimeSpec", "nParams",
    "controlRateHz", "cameraNames", "inputSchema", "outputSchema",
}


def build_metadata(
    request: dict[str, Any], package: Path
) -> dict[str, Any]:
    """Read the trainer-produced metadata from the already-created model TGZ."""
    del request
    with tarfile.open(package, mode="r:") as archive:
        metadata_file = archive.extractfile("metadata.json")
        if metadata_file is None:
            raise ValueError("모델 패키지에 metadata.json이 없습니다.")
        payload = json.loads(metadata_file.read().decode("utf-8"))
    metadata = payload.get("metadata", payload)
    if not isinstance(metadata, dict):
        raise ValueError("학습 산출 metadata.json 형식이 올바르지 않습니다.")
    _validate(metadata)
    return metadata


def _validate(metadata: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - metadata.keys())
    if missing:
        raise ValueError(f"학습 산출 metadata 필드가 누락되었습니다: {', '.join(missing)}")
    n_params = metadata.get("nParams")
    if isinstance(n_params, bool) or not isinstance(n_params, int) or n_params <= 1:
        raise ValueError("nParams는 CKPT에서 계산한 1보다 큰 정수여야 합니다.")

