"""Synthetic ReplayBuffer -> short official training -> full validation -> best.ckpt -> reload."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from umi_policy import checkpoint, config, dataset_check, evaluation, training, upstream  # noqa: E402


def write_synthetic(path: Path, episodes: int = 3, frames: int = 90) -> Path:
    rng = np.random.default_rng(0)
    total = episodes * frames
    t = np.tile(np.linspace(0, 1, frames, dtype=np.float32), episodes)
    position = np.stack([0.30 + 0.10 * np.sin(2 * t), 0.10 * np.cos(2 * t), 0.20 + 0.05 * t], axis=1)
    rotvec = np.stack([0.2 * t, -0.1 * t, 0.05 + 0.3 * t], axis=1)
    width = (0.02 + 0.05 * (1 - t))[:, None]
    pose = np.concatenate([position, rotvec], axis=1)
    start = np.repeat(pose.reshape(episodes, frames, 6)[:, :1], frames, axis=1).reshape(total, 6)
    end = np.repeat(pose.reshape(episodes, frames, 6)[:, -1:], frames, axis=1).reshape(total, 6)
    with zarr.ZipStore(str(path), mode="w") as store:
        root = zarr.group(store=store)
        data = root.create_group("data")
        root.create_group("meta").array("episode_ends", np.arange(1, episodes + 1) * frames)
        for key, value in {"robot0_eef_pos": position, "robot0_eef_rot_axis_angle": rotvec,
                           "robot0_gripper_width": width, "robot0_demo_start_pose": start,
                           "robot0_demo_end_pose": end}.items():
            data.array(key, value.astype(np.float32))
        images = data.create_dataset("camera0_rgb", shape=(total, 224, 224, 3), chunks=(1, 224, 224, 3),
                                     dtype=np.uint8)
        for i in range(total):
            images[i] = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    return path


def main() -> None:
    umi_root = upstream.resolve_umi_root(None)
    info = upstream.enable(umi_root)
    evaluation.self_check()
    profile = config.load_profile(ROOT / "configs/policy_resnet18_8gb.yaml")
    assert config.down_sample_steps(30.0, 10) == 3 and config.down_sample_steps(59.94, 10) == 6
    try:
        config.down_sample_steps(45.0, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("45 Hz should not down-sample to 10 Hz")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        base = Path(directory)
        dataset = write_synthetic(base / "synthetic.zarr.zip")
        try:
            dataset_check.check_dataset(dataset)
        except ValueError as error:
            assert "no_builder_report" in str(error)
        else:
            raise AssertionError("dataset without builder report must not be trainable by default")
        check = dataset_check.check_dataset(dataset, allow_development=True, native_hz=30.0)
        assert check["frames"] == 270 and check["episodes"] == 3 and check["unexpected_arrays"] == []
        cfg = config.compose_config(umi_root, profile, dataset, check["native_sample_rate_hz"],
                                    epochs=1, batch=2, seed=1, max_train_steps=2)
        assert cfg.task.obs_down_sample_steps == 3
        assert cfg.task.dataset.val_start_pose_noise_scale == 0.0 and cfg.task.dataset.action_padding
        run_dir = base / "runs" / "selftest"
        latest = training.train(cfg, run_dir)
        assert latest.is_file() and (run_dir / "config.yaml").is_file()
        try:
            training.train(cfg, run_dir)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing run directory must not be overwritten")
        manifest = checkpoint.select_best(run_dir, dataset, info, check, batch=2, seed=1)
        best = run_dir / "checkpoints/best.ckpt"
        assert best.is_file() and (run_dir / "evaluation/selection.json").is_file()
        assert manifest["best_checkpoint"]["sha256"] == dataset_check.sha256(best)
        assert np.isfinite(manifest["best_checkpoint"]["policy"]["position_component_rmse_mm"])
        assert manifest["physical_deployment_ready"] is False
        summary = checkpoint.inspect(best)
        assert summary["obs_down_sample_steps"] == 3 and summary["action"]["shape"] == [10]
        assert "camera0_rgb" in summary["observations"] and "action" in summary["normalizer_keys"]
        saved = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert saved["best_checkpoint"]["source"] == manifest["best_checkpoint"]["source"]
    print("TRAINER_SELF_TEST_OK")


if __name__ == "__main__":
    main()
