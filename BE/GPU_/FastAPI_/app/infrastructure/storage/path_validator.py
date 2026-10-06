from pathlib import Path

from app.domain.errors import DomainError


class PathValidator:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve_existing_file(self, storage_key: str) -> Path:
        candidate = (self.root / storage_key).resolve()
        if candidate == self.root or self.root not in candidate.parents:
            raise DomainError(422, "STORAGE_KEY_INVALID", "공유 저장소 밖의 경로는 사용할 수 없습니다.")
        if not candidate.is_file() or candidate.is_symlink():
            raise DomainError(422, "EPISODE_FILE_NOT_FOUND", f"Episode 파일이 없습니다: {storage_key}")
        return candidate

    def resolve_output(self, storage_key: str) -> Path:
        candidate = (self.root / storage_key).resolve()
        if candidate == self.root or self.root not in candidate.parents:
            raise DomainError(422, "STORAGE_KEY_INVALID", "공유 저장소 밖의 경로는 사용할 수 없습니다.")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

