"""Locate and verify the pinned Stanford UMI checkout. The source is referenced, never copied."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UMI_ROOT = PACKAGE_ROOT.parent / "third_party" / "umi"
UMI_COMMIT = "d095ba9590df789df5189eea5ee7e431689038a6"
PATCHES_DIR = PACKAGE_ROOT / "patches"
DATASET_FILE = "diffusion_policy/dataset/umi_dataset.py"
# Each patch leaves a marker string in the patched file; its presence proves application.
PATCH_MARKERS = {
    "0001-umi-dataset-normalizer-workers.patch": "avoid 32 spawned dataset copies",
    "0002-umi-dataset-val-start-pose-noise.patch": "val_start_pose_noise_scale",
}


def resolve_umi_root(argument: str | None) -> Path:
    return Path(argument or os.environ.get("UMI_ROOT") or DEFAULT_UMI_ROOT).resolve()


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def verify(root: Path) -> dict:
    dataset_file = root / DATASET_FILE
    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"Stanford UMI checkout not found at {root}; clone commit {UMI_COMMIT} or pass --umi-root")
    head = _git_head(root)
    if head is not None and head != UMI_COMMIT:
        raise RuntimeError(f"UMI checkout is at {head}, expected {UMI_COMMIT}")
    text = dataset_file.read_text(encoding="utf-8")
    missing = [name for name, marker in PATCH_MARKERS.items() if marker not in text]
    if missing:
        commands = "\n".join(f"  git -C {root} apply {PATCHES_DIR / name}" for name in missing)
        raise RuntimeError("UMI checkout is missing local patches. Apply them with:\n" + commands)
    return {
        "umi_root": str(root),
        "umi_commit": head or f"{UMI_COMMIT} (not a git checkout; commit unverified)",
        "patches": sorted(PATCH_MARKERS),
    }


def enable(root: Path) -> dict:
    info = verify(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    return info
