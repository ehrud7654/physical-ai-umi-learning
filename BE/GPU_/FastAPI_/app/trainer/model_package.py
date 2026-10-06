import tarfile
from pathlib import Path


REQUIRED_FILES = {
    "encoder.pt",
    "denoiser.pt",
    "metadata.json",
    "dataset.report.json",
    "config.yaml",
    "reference.npz",
    "pc_check.json",
}


def find_and_validate_package(run_directory: Path) -> Path:
    packages = [
        path
        for path in run_directory.rglob("*.tgz")
        if path.is_file() and not path.is_symlink()
    ]
    if not packages:
        raise FileNotFoundError(f"학습 결과 TGZ를 찾을 수 없습니다: {run_directory}")
    package = max(packages, key=lambda path: path.stat().st_mtime_ns)
    validate_package(package)
    return package


def validate_package(package: Path) -> None:
    if package.suffix.lower() != ".tgz":
        raise ValueError("모델 패키지 확장자는 .tgz여야 합니다.")
    try:
        # r: only accepts an uncompressed TAR stream. r:* would also accept gzip.
        with tarfile.open(package, mode="r:") as archive:
            members = archive.getmembers()
    except tarfile.TarError as error:
        raise ValueError("모델 패키지는 압축하지 않은 TAR 형식의 .tgz여야 합니다.") from error

    if any(not member.isfile() for member in members):
        raise ValueError("모델 패키지에는 루트의 일반 파일만 포함할 수 있습니다.")
    names = [member.name for member in members]
    if any(Path(name).name != name or name in ("", ".", "..") for name in names):
        raise ValueError("모델 패키지는 하위 경로를 포함할 수 없습니다.")
    if len(names) != len(set(names)):
        raise ValueError("모델 패키지에 중복 파일명이 있습니다.")

    manifests = [name for name in names if name.endswith(".manifest.json")]
    expected_count = len(REQUIRED_FILES) + 1
    if len(names) != expected_count or set(names) - set(manifests) != REQUIRED_FILES or len(manifests) != 1:
        raise ValueError(
            "모델 패키지는 필수 파일 7개와 *.manifest.json 1개만 포함해야 합니다."
        )
