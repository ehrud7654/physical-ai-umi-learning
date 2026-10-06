"""Select best.ckpt from full validation, write the run manifest, inspect checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil

from .dataset_check import sha256
from .evaluation import evaluate

SELECTION_METRIC = "position_component_rmse_mm"


def list_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(p for p in (Path(run_dir) / "checkpoints").glob("*.ckpt") if p.name != "best.ckpt")


def select_best(run_dir: Path, dataset: Path, umi_info: dict, check: dict,
                batch: int = 4, seed: int = 42) -> dict:
    import torch
    run_dir = Path(run_dir)
    candidates = list_checkpoints(run_dir)
    if not candidates:
        raise FileNotFoundError(f"no checkpoints in {run_dir / 'checkpoints'}")
    evaluation_dir = run_dir / "evaluation"
    reports = {path.name: evaluate(path, dataset, evaluation_dir, batch, seed) for path in candidates}
    best_name = min(reports, key=lambda name: reports[name]["policy"][SELECTION_METRIC])
    best = reports[best_name]
    best_path = run_dir / "checkpoints/best.ckpt"
    shutil.copy2(run_dir / "checkpoints" / best_name, best_path)
    selection = {
        "criterion": f"minimum policy.{SELECTION_METRIC} on the full validation split",
        "best": best_name,
        "beats_hold_baseline": best["policy"][SELECTION_METRIC] < best["hold_current_pose_and_width"][SELECTION_METRIC],
        "candidates": {name: {"epoch": r["checkpoint_epoch"], "policy": r["policy"],
                              "hold_current_pose_and_width": r["hold_current_pose_and_width"]}
                       for name, r in reports.items()},
    }
    (evaluation_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    manifest = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "platform": platform.platform()},
        "upstream": umi_info,
        "dataset": {key: check[key] for key in ("dataset", "dataset_sha256", "frames", "episodes",
                                               "native_sample_rate_hz", "training_input_status",
                                               "camera_tcp_status", "builder_report")},
        "config_sha256": sha256(run_dir / "config.yaml"),
        "best_checkpoint": {
            "path": str(best_path), "source": best_name, "sha256": sha256(best_path),
            "epoch": best["checkpoint_epoch"], "policy": best["policy"],
            "hold_current_pose_and_width": best["hold_current_pose_and_width"],
            "beats_hold_baseline": selection["beats_hold_baseline"],
            "inference_image_transform": best["inference_image_transform"],
        },
        "physical_deployment_ready": bool(check["physical_deployment_ready"]
                                          and check["training_input_status"] == "ready"),
        "deployment_note": ("Camera->TCP validated by the dataset builder; hardware limits still apply."
                            if check["training_input_status"] == "ready" else
                            "Rebuild the dataset with a physically measured T_camera_tcp and retrain "
                            "before using this policy on hardware."),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def inspect(checkpoint: Path) -> dict:
    import dill
    import torch
    from omegaconf import OmegaConf
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    key = "ema_model" if cfg.training.use_ema else "model"
    state = payload["state_dicts"][key]
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "sha256": sha256(checkpoint),
        "epoch": int(dill.loads(payload["pickles"]["epoch"])),
        "global_step": int(dill.loads(payload["pickles"]["global_step"])),
        "weights": key,
        "parameters": int(sum(v.numel() for v in state.values() if hasattr(v, "numel"))),
        "encoder": cfg.policy.obs_encoder.model_name,
        "down_dims": list(cfg.policy.down_dims),
        "num_inference_steps": int(cfg.policy.num_inference_steps),
        "obs_down_sample_steps": int(cfg.task.obs_down_sample_steps),
        "observations": {name: {"shape": attr["shape"], "horizon": attr["horizon"], "type": attr.get("type")}
                         for name, attr in shape_meta["obs"].items()},
        "action": shape_meta["action"],
        "pose_repr": OmegaConf.to_container(cfg.task.pose_repr, resolve=True),
        "training_dataset_path": cfg.task.dataset_path,
        # LinearNormalizer stores normalizer.params_dict.<field>.<scale|offset|input_stats...>
        "normalizer_keys": sorted({k.split(".")[2] for k in state if k.startswith("normalizer.params_dict.")}),
    }
