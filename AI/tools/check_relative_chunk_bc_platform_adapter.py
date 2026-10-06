"""Hardware-free contract check for the RelativeChunkBC registry adapter."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

from algorithms.relative_chunk_bc_adapter import train  # noqa: E402
from tools.convert_umi_zarr_to_v10 import episode_to_v10  # noqa: E402
from umi.relative_dataset import SCHEMA, write_episode  # noqa: E402


class Reporter:
    def __init__(self) -> None:
        self.rows = []

    def report(self, status, stage, percent, **extra) -> None:
        self.rows.append({"status": status, "stage": stage,
                          "percent": percent, **extra})


class CancelEvent:
    def __init__(self, value: bool = False) -> None:
        self.value = value

    def is_set(self) -> bool:
        return self.value


def make_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    dataset.mkdir()
    names = []
    total_rows = 0
    for episode in range(3):
        length = 14
        rgb = np.zeros((length, 224, 224, 3), dtype=np.uint8)
        rgb[..., episode] = np.arange(length, dtype=np.uint8)[:, None, None]
        position = np.zeros((length, 3), dtype=np.float64)
        position[:, 0] = np.arange(length) * (0.002 + episode * 0.0002)
        rotation = np.zeros((length, 3), dtype=np.float64)
        rotation[:, 2] = np.arange(length) * math.radians(0.5 + episode * 0.1)
        gap = np.linspace(0.079, 0.041 + episode * 0.001, length)[:, None]
        arrays = episode_to_v10(rgb, position, rotation, gap, rate_hz=10.0)
        name = f"episode_{episode:03d}"
        meta = {
            "schema": SCHEMA,
            "episode_id": name,
            "rate_hz": 10.0,
            "observation_horizon": 2,
            "action_horizon": 8,
            "training_rows": len(arrays["action"]),
        }
        write_episode(dataset / f"{name}.npz", arrays, meta)
        names.append(name)
        total_rows += len(arrays["action"])
    (dataset / "dataset.json").write_text(json.dumps({
        "schema": SCHEMA,
        "status": "SYNTHETIC_SELFTEST",
        "episodes": names,
        "n_episodes": len(names),
        "n_rows": total_rows,
        "rate_hz": 10.0,
        "observation_horizon": 2,
        "action_horizon": 8,
        "robot_base_alignment_applied": False,
        "robot_ik_applied": False,
        "physical_deployment_ready": False,
    }, indent=2), encoding="utf-8")
    return dataset


def main() -> int:
    checks = []

    def check(name: str, condition: bool) -> None:
        checks.append(bool(condition))
        print(("PASS" if condition else "FAIL") + f"  {name}")

    check("entrypoint signature", list(inspect.signature(train).parameters) == [
        "dataset", "profile", "run_dir", "reporter", "cancel_event"
    ])
    with tempfile.TemporaryDirectory(prefix="relative_chunk_registry_") as temporary:
        root = Path(temporary)
        cancelled_reporter = Reporter()
        cancelled = train(
            root / "not_read", {"config": "not_read.yaml"}, root / "cancelled",
            cancelled_reporter, CancelEvent(True),
        )
        check("pre-start cancellation returns manifest",
              cancelled["status"] == "CANCELLED"
              and (root / "cancelled/manifest.json").is_file())
        check("cancellation is reported", cancelled_reporter.rows[-1]["status"] == "CANCELLED")

        dataset = make_dataset(root)
        config = root / "tiny.yaml"
        config.write_text("""\
model:
  encoder_channels: [4, 8]
  feature_dim: 8
  spatial_grid_size: 2
  hidden_dims: [16]
  dropout: 0.0
train:
  epochs: 1
  batch_size: 4
  lr: 0.0003
  weight_decay: 0.0001
  val_fraction: 0.34
  seed: 7
  grad_clip: 1.0
  num_workers: 0
data:
  image_mean: 0.5
  image_std: 0.5
""", encoding="utf-8")
        reporter = Reporter()
        result = train(
            dataset,
            {"config": str(config), "device": "cpu"},
            root / "run", reporter, CancelEvent(False),
        )
        check("CPU smoke completes", result["status"] == "COMPLETED")
        check("last checkpoint exists", Path(result["artifacts"]["last"]["path"]).is_file())
        check("best checkpoint exists", Path(result["artifacts"]["best"]["path"]).is_file())
        check("manifest is persisted", json.loads(
            (root / "run/manifest.json").read_text(encoding="utf-8"))["status"] == "COMPLETED")
        check("reporter reaches 100%", reporter.rows[-1]["percent"] == 100)

    print(f"{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
