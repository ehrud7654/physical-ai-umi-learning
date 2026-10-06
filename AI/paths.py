"""Filesystem anchors for the AI part — resolved once, imported everywhere.
AI 파트의 파일시스템 기준점. 한 번 정하고 모든 곳에서 임포트한다.

Every module used to compute its own `parents[n]` walk, which silently broke the
moment a file moved. One module owns it now.
예전에는 모듈마다 `parents[n]` 을 각자 계산했는데, 파일이 옮겨지는 순간 조용히
깨지는 구조였다. 이제 한 모듈이 책임진다.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../AI/paths.py -> parent == .../AI
AI_ROOT: Path = Path(__file__).resolve().parent

CONFIG_DIR: Path = AI_ROOT / "configs"
DOCS_DIR: Path = AI_ROOT / "docs"
OUT_DIR: Path = AI_ROOT / "out"
# datasets/ 는 수집 데이터셋, data/ 는 수집·검증 코드다. 섞지 않는다.
DATASET_DIR: Path = AI_ROOT / "datasets"

# Scene files are simulator-specific: MJCF here, USD under sim/isaac/ later.
# 씬 파일은 시뮬레이터 고유하다. 여기는 MJCF, 나중에 sim/isaac/ 아래 USD.
MUJOCO_SCENE_DIR: Path = AI_ROOT / "sim" / "mujoco" / "scenes"

# Official SO-101 MJCF/URDF. Vendored, never edited — replaced wholesale on update.
# 공식 SO-101 MJCF/URDF. 벤더링만 하고 수정하지 않는다. 업데이트 시 통째로 교체.
THIRD_PARTY: Path = AI_ROOT / "third_party"
SO101_MJCF_DIR: Path = THIRD_PARTY / "so101_mujoco" / "SO101"

DEFAULT_CONFIG: Path = CONFIG_DIR / "so101.yaml"
DEFAULT_SCENE: Path = MUJOCO_SCENE_DIR / "pick_place.xml"
DEFAULT_EXP_LOG: Path = AI_ROOT / "EXP_LOG.jsonl"

# What counts as "the code that produced a number" — see tracking/exp_log.py.
# "수치를 만든 코드"에 해당하는 것 — tracking/exp_log.py 참조.
#
# track_a/ is deliberately NOT here. Track A code cannot change what a MuJoCo
# rollout returns; it changes what a dataset contains, and dataset provenance is
# recorded per episode by contract/episode.py. Including it would make this digest
# churn on every Track A commit and stop meaning "same code, same number".
# track_a/ 는 의도적으로 빼뒀다. 트랙 A 코드는 MuJoCo 롤아웃 결과를 바꿀 수 없다.
# 바꾸는 것은 데이터셋 내용이고, 그 출처는 contract/episode.py 가 에피소드마다
# 기록한다. 넣으면 트랙 A 커밋마다 이 해시가 흔들려서 "같은 코드면 같은 수치"라는
# 의미를 잃는다.
CODE_GLOBS: tuple[str, ...] = (
    "*.py",
    "contract/**/*.py",
    "sim/**/*.py",
    "policy/**/*.py",
    "eval/**/*.py",
    "data/**/*.py",
    "vlm/**/*.py",
    "tracking/**/*.py",
    "configs/*.yaml",
    "sim/**/scenes/*",
)


def _is_ascii(text: str) -> bool:
    """Whether every character is ASCII.
    모든 문자가 ASCII 인가."""
    return all(ord(c) < 128 for c in text)


def _windows_short_path(path: Path) -> Path | None:
    """8.3 short form of `path`, which is always ASCII — or None if unavailable.
    `path` 의 8.3 단축 형식. 항상 ASCII 다. 얻을 수 없으면 None.

    Short-name generation can be turned off per volume (`fsutil 8dot3name`), in
    which case Windows returns the long name unchanged and this returns None.
    단축 이름 생성은 볼륨별로 꺼둘 수 있고(`fsutil 8dot3name`), 그 경우 Windows 가
    긴 이름을 그대로 돌려주므로 여기서는 None 을 반환한다.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        fn = ctypes.windll.kernel32.GetShortPathNameW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        fn.restype = wintypes.DWORD
        need = fn(str(path), None, 0)
        if not need:
            return None
        buf = ctypes.create_unicode_buffer(need)
        if not fn(str(path), buf, need):
            return None
        short = Path(buf.value)
        return short if _is_ascii(str(short)) else None
    except Exception:  # noqa: BLE001 — 어떤 이유로든 실패하면 다음 수단으로 넘어간다
        return None


def _ascii_cache_root() -> Path:
    """A writable directory whose full path is ASCII.
    전체 경로가 ASCII 인 쓰기 가능한 디렉터리."""
    import hashlib
    import tempfile

    key = hashlib.sha256(str(AI_ROOT).encode("utf-8")).hexdigest()[:10]
    candidates = [Path(tempfile.gettempdir()), Path(os.environ.get("SystemDrive", "/") + "/")]
    for base in candidates:
        root = base / f"so101_mjcf_{key}"
        if _is_ascii(str(root)):
            return root
    raise RuntimeError(
        "ASCII 경로의 임시 디렉터리를 찾지 못했다. "
        "저장소를 ASCII 경로로 옮겨야 한다 (예: C:/Users/USER/so101)."
    )


_MIRROR_SUBDIRS = ("sim/mujoco/scenes", "third_party")
_mirrored: set[Path] = set()


def _mirror_ascii(scene: Path) -> Path:
    """Copy the model tree to an ASCII path and return the scene's path there.
    모델 트리를 ASCII 경로로 복사하고 그곳의 씬 경로를 반환한다.

    Only the files MuJoCo actually opens are copied: the scene and the vendored
    MJCF with its meshes. The relative layout is preserved because the scene's
    `include` and `meshdir` are relative paths.
    MuJoCo 가 실제로 여는 파일만 복사한다. 씬과 벤더링된 MJCF·메시.
    씬의 `include` 와 `meshdir` 이 상대경로이므로 배치를 그대로 보존한다.
    """
    import shutil

    root = _ascii_cache_root()
    rel = scene.resolve().relative_to(AI_ROOT)
    n_copied = 0
    for sub in _MIRROR_SUBDIRS:
        src_dir = AI_ROOT / sub
        if not src_dir.is_dir():
            continue
        for src in src_dir.rglob("*"):
            if not src.is_file():
                continue
            dst = root / src.relative_to(AI_ROOT)
            if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_copied += 1
    if root not in _mirrored:
        _mirrored.add(root)
        print(
            f"ℹ️ 경로에 ASCII 가 아닌 문자가 있어 MuJoCo 가 파일을 열지 못한다.\n"
            f"   모델을 ASCII 경로로 복사해서 로드한다: {root}\n"
            f"   (새로 복사한 파일 {n_copied}개. 원본은 건드리지 않는다)"
        )
    return root / rel


def resolve_for_mujoco(scene: Path, *, force_copy: bool = False) -> Path:
    """Return a path MuJoCo's XML loader can actually open.
    MuJoCo 의 XML 로더가 실제로 열 수 있는 경로를 반환한다.

    MuJoCo parses XML in C++ and opens files through a narrow-character API on
    Windows, so a path containing non-ASCII characters fails with
    `ParseXML: Error opening file` even though the file is readable from Python.
    This project's repo lives under a Korean-named directory, so this is not a
    hypothetical.
    MuJoCo 는 XML 을 C++ 에서 파싱하고 Windows 에서 좁은 문자 API 로 파일을 연다.
    그래서 경로에 비ASCII 문자가 있으면 파이썬으로는 읽히는 파일인데도
    `ParseXML: Error opening file` 로 실패한다. 이 프로젝트 저장소가 한글 폴더
    아래 있으므로 가정이 아니라 실제 상황이다.

    세 단계로 시도한다:
      1. 경로가 이미 ASCII 면 그대로 쓴다 (대부분의 경우)
      2. Windows 8.3 단축 경로 — 항상 ASCII 이고 복사가 없다
      3. ASCII 임시 경로로 모델 트리를 복사 (단축 이름이 꺼져 있을 때)

    `force_copy` 는 3번 경로를 직접 시험하기 위한 것이다.
    """
    p = scene.resolve()
    if not force_copy:
        if _is_ascii(str(p)):
            return p
        short = _windows_short_path(p)
        if short is not None:
            return short
    return _mirror_ascii(p)


def require(path: Path, what: str) -> Path:
    """Return `path`, or fail now with a message that names what is missing.
    `path` 를 반환하거나, 무엇이 없는지 이름을 붙여 지금 실패한다."""
    if not path.exists():
        raise FileNotFoundError(f"{what} 가 없다: {path}")
    return path
