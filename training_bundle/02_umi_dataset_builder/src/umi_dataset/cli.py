import argparse
import json
import shutil
from pathlib import Path

from .gripper import extract_gripper
from .recording import load_recording
from .replay_buffer import build_replay_buffer, inspect_replay_buffer
from .slam_mask import build_mask, overlay
from .slam_tag import estimate_slam_tag
from .telemetry import prepare_telemetry


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(args) -> None:
    recording = load_recording(args.session, require_success=False)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = [args.output / name for name in
               ("telemetry.json", "gripper_width.csv", "gripper_report.json", "prepare_report.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to overwrite prepared files in {args.output}")
    camera_imu = _load(args.camera_imu)
    gripper = _load(args.gripper)
    telemetry_report = prepare_telemetry(args.session, camera_imu, outputs[0])
    gripper_report = extract_gripper(args.session, camera_imu, gripper, outputs[1], outputs[2])
    shutil.copy2(PACKAGE_ROOT / "slam" / "s22_mono_inertial.yaml",
                 args.output / "s22_mono_inertial.yaml")
    report = {
        "status": "pass",
        "session": recording["session"].name,
        "outcome": recording["manifest"].get("outcome"),
        "frame_count": recording["frame_count"],
        "telemetry": telemetry_report,
        "gripper": gripper_report,
        "next": "run ORB-SLAM3 and place camera_trajectory.csv in this directory",
    }
    outputs[3].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def slam_tag(args) -> None:
    output = args.output or args.prepared / "tx_slam_tag.json"
    if output.exists():
        raise FileExistsError(output)
    report = estimate_slam_tag(args.session, args.prepared / "slam/camera_trajectory.csv",
                               _load(args.camera_imu), _load(args.tag), output)
    print(json.dumps({key: report[key] for key in
                      ("status", "frames_with_tag", "samples_used", "position_spread_m")}, indent=2))


def slam_mask(args) -> None:
    report = build_mask(args.config, args.output)
    if args.preview_session:
        preview = args.output.with_suffix(".preview.jpg")
        overlay(args.preview_session / "video.mp4", args.output, preview, args.preview_frame)
        report["preview"] = str(preview)
    print(json.dumps(report, indent=2))


def build(args) -> None:
    report = build_replay_buffer(args.plan, args.output, args.dataset_config,
                                 args.camera_tcp, args.allow_unrated)
    print(json.dumps({key: report[key] for key in
                      ("status", "dataset", "episodes", "frames", "training_input_status",
                       "physical_deployment_ready")}, indent=2))


def inspect(args) -> None:
    print(json.dumps(inspect_replay_buffer(args.dataset), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stanford UMI input from S22 collector sessions.")
    subparsers = parser.add_subparsers(required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Create telemetry and gripper labels.")
    prepare_parser.add_argument("session", type=Path)
    prepare_parser.add_argument("output", type=Path)
    prepare_parser.add_argument("--camera-imu", type=Path, default=PACKAGE_ROOT / "configs/s22_camera_imu.json")
    prepare_parser.add_argument("--gripper", type=Path, default=PACKAGE_ROOT / "configs/s22_gripper.json")
    prepare_parser.set_defaults(handler=prepare)

    tag_parser = subparsers.add_parser(
        "slam-tag", help="Estimate the table-marker world frame from a mapping session's SLAM trajectory.")
    tag_parser.add_argument("session", type=Path, help="mapping session (app recording with the table marker)")
    tag_parser.add_argument("prepared", type=Path, help="its prepared dir containing slam/camera_trajectory.csv")
    tag_parser.add_argument("--output", type=Path, help="default <prepared>/tx_slam_tag.json")
    tag_parser.add_argument("--camera-imu", type=Path, default=PACKAGE_ROOT / "configs/s22_camera_imu.json")
    tag_parser.add_argument("--tag", type=Path, default=PACKAGE_ROOT / "configs/s22_slam_tag.json")
    tag_parser.set_defaults(handler=slam_tag)

    mask_parser = subparsers.add_parser(
        "slam-mask", help="Render the gripper mask PNG for ORB-SLAM3 from configs/s22_slam_mask.json (UMI-style).")
    mask_parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs/s22_slam_mask.json")
    mask_parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "configs/s22_slam_mask.png")
    mask_parser.add_argument("--preview-session", type=Path, help="recording whose frame is overlaid for a visual check")
    mask_parser.add_argument("--preview-frame", type=int, default=0)
    mask_parser.set_defaults(handler=slam_mask)

    build_parser = subparsers.add_parser("build", help="Create dataset.zarr.zip from an episode plan.")
    build_parser.add_argument("plan", type=Path)
    build_parser.add_argument("output", type=Path)
    build_parser.add_argument("--dataset-config", type=Path, default=PACKAGE_ROOT / "configs/dataset.json")
    build_parser.add_argument("--camera-tcp", type=Path, default=PACKAGE_ROOT / "configs/s22_camera_tcp.json")
    build_parser.add_argument("--allow-unrated", action="store_true",
                              help="Development only; report will not mark input ready.")
    build_parser.set_defaults(handler=build)

    inspect_parser = subparsers.add_parser("inspect", help="Reload and list a generated ReplayBuffer.")
    inspect_parser.add_argument("dataset", type=Path)
    inspect_parser.set_defaults(handler=inspect)
    args = parser.parse_args()
    args.handler(args)
