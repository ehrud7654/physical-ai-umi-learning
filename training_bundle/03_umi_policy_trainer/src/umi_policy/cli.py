import argparse
import json
from pathlib import Path

from .checkpoint import inspect, select_best
from .config import compose_config, load_profile
from .dataset_check import check_dataset
from .evaluation import self_check
from .training import train
from .upstream import enable, resolve_umi_root

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PACKAGE_ROOT.parents[1] / "runs"
DEFAULT_PROFILE = PACKAGE_ROOT / "configs/policy_resnet18_8gb.yaml"


def _print(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def run_check(args) -> None:
    _print(check_dataset(args.dataset, args.allow_development, args.native_hz))


def run_train(args) -> None:
    umi_root = resolve_umi_root(args.umi_root)
    info = enable(umi_root)
    check = check_dataset(args.dataset, args.allow_development, args.native_hz)
    profile = load_profile(args.profile)
    cfg = compose_config(umi_root, profile, args.dataset, check["native_sample_rate_hz"],
                         args.epochs, args.batch, args.seed, args.max_train_steps)
    run_dir = Path(args.runs_dir) / args.run_id
    latest = train(cfg, run_dir)
    (run_dir / "dataset_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(f"training finished: {latest}")
    if args.no_select:
        return
    self_check()
    _print(select_best(run_dir, args.dataset, info, check, args.eval_batch, args.seed or 42))


def run_select(args) -> None:
    from omegaconf import OmegaConf
    umi_root = resolve_umi_root(args.umi_root)
    info = enable(umi_root)
    self_check()
    run_dir = Path(args.run_dir)
    dataset = Path(args.dataset or OmegaConf.load(run_dir / "config.yaml").task.dataset_path)
    check = check_dataset(dataset, allow_development=True, native_hz=args.native_hz)
    _print(select_best(run_dir, dataset, info, check, args.batch, args.seed))


def run_inspect(args) -> None:
    enable(resolve_umi_root(args.umi_root))
    _print(inspect(args.checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Stanford UMI diffusion policy from a 02_umi_dataset_builder ReplayBuffer.")
    parser.add_argument("--umi-root",
                        help="pinned Stanford UMI checkout (default: UMI_ROOT env or ../third_party/umi)")
    sub = parser.add_subparsers(required=True)

    check = sub.add_parser("check", help="Validate a dataset.zarr.zip and its builder report.")
    check.add_argument("dataset", type=Path)
    check.add_argument("--allow-development", action="store_true")
    check.add_argument("--native-hz", type=float)
    check.set_defaults(handler=run_check)

    fit = sub.add_parser("train", help="Train, then evaluate every checkpoint and pick best.ckpt.")
    fit.add_argument("dataset", type=Path)
    fit.add_argument("--run-id", required=True)
    fit.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    fit.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    fit.add_argument("--epochs", type=int)
    fit.add_argument("--batch", type=int)
    fit.add_argument("--seed", type=int)
    fit.add_argument("--max-train-steps", type=int, help="steps per epoch cap; smoke tests only")
    fit.add_argument("--eval-batch", type=int, default=4)
    fit.add_argument("--allow-development", action="store_true",
                     help="accept datasets whose builder status is not trainable; recorded in manifest")
    fit.add_argument("--native-hz", type=float,
                     help="override native sample rate when no builder report exists")
    fit.add_argument("--no-select", action="store_true", help="skip post-training evaluation and best.ckpt")
    fit.set_defaults(handler=run_train)

    select = sub.add_parser("select", help="Re-evaluate saved checkpoints of a run and pick best.ckpt.")
    select.add_argument("run_dir", type=Path)
    select.add_argument("--dataset", type=Path, help="defaults to the run's recorded dataset path")
    select.add_argument("--batch", type=int, default=4)
    select.add_argument("--seed", type=int, default=42)
    select.add_argument("--native-hz", type=float)
    select.set_defaults(handler=run_select)

    show = sub.add_parser("inspect", help="Reload a checkpoint and print its contract.")
    show.add_argument("checkpoint", type=Path)
    show.set_defaults(handler=run_inspect)

    args = parser.parse_args()
    args.handler(args)
