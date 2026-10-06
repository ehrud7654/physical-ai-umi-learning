"""Raw app sessions -> prepare -> ORB-SLAM3 Atlas -> gate -> UMI Zarr.

Stanford UMI workflow: one mapping session (raw_mapping/) is tracked first with --save_map and
its table marker fixes the world frame; every demonstration then runs with --load_map so
tracking relocalises from the first frames instead of initialising. Every stage skips work
whose output already exists, so re-running after adding sessions only processes the new ones.
Nothing under <root>/raw or <root>/raw_mapping is modified. Training and robot
execution are deliberately outside this package."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
BUILDER = PACKAGE_ROOT / "dataset_builder"
CONFIG_DIR = Path(os.environ.get("UMI_CONFIG_DIR", BUILDER / "configs")).resolve()
for path in (BUILDER / "src", BUILDER / "slam"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from umi_dataset import cli as builder_cli  # noqa: E402
from umi_dataset.camera_tcp import load_camera_tcp  # noqa: E402
from umi_dataset.episode import build_episode  # noqa: E402
from umi_dataset.recording import load_recording  # noqa: E402
from umi_dataset.replay_buffer import build_replay_buffer  # noqa: E402
from umi_dataset.slam_tag import estimate_slam_tag, load_slam_tag  # noqa: E402
import numpy as np  # noqa: E402
import validate_orbslam_trajectory  # noqa: E402

MAPPING_DIR = "raw_mapping"
MAP_FILE = "map_atlas.osa"
MAPPING_MIN_TRACKED_RATIO = 0.9
MAPPING_ATTEMPTS = 3
MINIMUM_TRAINING_EPISODES = 1
DEMO_ATTEMPTS = 2
LATE_RELOCALIZATION_S = 0.5
DEMO_MIN_TRACKED_RATIO = 0.95


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sessions(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if (p / "manifest.json").is_file()) if folder.is_dir() else []


SLAM_SETTINGS = Path(os.environ.get(
    "UMI_SLAM_SETTINGS", BUILDER / "slam/s22_mono_inertial.yaml"
)).resolve()


def prepare(session: Path, output: Path) -> dict:
    report = output / "prepare_report.json"
    receipt = output / "prepare_receipt.json"
    inputs = {
        "camera_imu_sha256": _sha256(CONFIG_DIR / "s22_camera_imu.json"),
        "gripper_sha256": _sha256(CONFIG_DIR / "s22_gripper.json"),
    }
    if report.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else None
        if previous != inputs:
            raise RuntimeError(f"prepared output uses a different calibration: {output}")
    if not report.exists():
        with contextlib.redirect_stdout(io.StringIO()):  # 02 prints its report; we keep the file
            builder_cli.prepare(argparse.Namespace(
                session=session, output=output,
                camera_imu=CONFIG_DIR / "s22_camera_imu.json",
                gripper=CONFIG_DIR / "s22_gripper.json"))
        receipt.write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    # The SLAM settings file is configuration, not a derived product: always use the current one.
    shutil.copy2(SLAM_SETTINGS, output / SLAM_SETTINGS.name)
    return json.loads(report.read_text(encoding="utf-8"))


def slam_timeout_s(duration_s: float) -> float:
    """ORB-SLAM3 occasionally hangs at shutdown; UMI's batch runner uses 16x the video duration."""
    return 60.0 + 16.0 * duration_s


def slam(session: Path, prepared: Path, output: Path, load_map: Path | None = None,
         save_map: Path | None = None, timeout_s: float | None = None) -> dict:
    if load_map is not None and save_map is not None:
        raise ValueError("an ORB-SLAM run cannot load and save the shared Atlas at the same time")
    trajectory = output / "camera_trajectory.csv"
    receipt = output / "receipt.json"
    mask = CONFIG_DIR / "s22_slam_mask.png"
    tag_config = CONFIG_DIR / "s22_slam_tag.json"
    binary = Path(os.environ.get("ORB_SLAM_BIN", ""))
    atlas_sha256 = _sha256(load_map) if load_map else None
    wanted = {"load_map_sha256": atlas_sha256,
              "mask_sha256": _sha256(mask) if mask.exists() else None,
              "tag_config_sha256": _sha256(tag_config),
              "settings_sha256": _sha256(SLAM_SETTINGS),
              "binary_sha256": _sha256(binary) if binary.is_file() else None}
    if trajectory.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
        if previous != wanted:
            shutil.rmtree(output)  # the map or mask this trajectory was made with has changed; redo it
    if not trajectory.exists():
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        tag = json.loads(tag_config.read_text(encoding="utf-8"))
        env["ORB_SLAM_MASK"] = str(mask)
        env["INIT_TAG_ID"] = str(tag["tag_id"])
        env["INIT_TAG_SIZE_M"] = str(tag["tag_size_m"])
        if load_map:
            env["ORB_SLAM_LOAD_MAP"] = str(load_map)
        if save_map:
            env["ORB_SLAM_SAVE_MAP"] = str(save_map)
        # ORB-SLAM3 segfaults (exit 139) on roughly one run in three and succeeds when repeated; retry twice.
        returncode = None
        for attempt in (1, 2, 3):
            with (output / "slam.log").open("w", encoding="utf-8") as log:
                # Own process group so a timeout kills gopro_slam itself, not just the sh wrapper
                # (orphaned hung binaries were holding ~850 MB each).
                process = subprocess.Popen(
                    ["sh", str(BUILDER / "slam/run_orbslam_linux.sh"), str(session), str(prepared), str(output)],
                    stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                try:
                    returncode = process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    returncode = "timeout"
                    log.write(f"\n[pipeline] killed after {timeout_s:.0f}s (hung at shutdown?)\n")
            if load_map and _sha256(load_map) != atlas_sha256:
                raise RuntimeError("shared Atlas was modified while processing a demonstration")
            if returncode == 0 and trajectory.exists() and (save_map is None or save_map.exists()):
                break
            trajectory.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"ORB-SLAM3 failed three times (last: {returncode}); see {output / 'slam.log'}")
        receipt.write_text(json.dumps(wanted), encoding="utf-8")
    if load_map and _sha256(load_map) != atlas_sha256:
        raise RuntimeError("shared Atlas was modified while processing a demonstration")
    validation = validate_orbslam_trajectory.validate(trajectory)
    validation["atlas_integrity"] = {
        "status": "pass" if load_map else "not_applicable",
        "sha256_before": atlas_sha256,
        "sha256_after": _sha256(load_map) if load_map else None,
        "save_map_used": save_map is not None,
    }
    (output / "trajectory_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def build_map(root: Path, mapping: Path) -> dict:
    """Track the mapping session, save its atlas, and fix the world frame on the table marker."""
    prepared = root / "processed_mapping" / mapping.name
    prepare(mapping, prepared)
    timeout = slam_timeout_s(load_recording(mapping, require_success=False)["duration_s"])
    slam_dir = prepared / "slam"
    map_path = slam_dir / MAP_FILE
    if not (slam_dir / "camera_trajectory.csv").exists() or not (slam_dir / "accepted.json").exists():
        # ORB-SLAM3 is multi-threaded and not repeatable: the same video can track 93% on one run and
        # 32% on the next. Run up to MAPPING_ATTEMPTS times and keep the best map.
        best, best_dir = None, prepared / "slam_best"
        for attempt in range(1, MAPPING_ATTEMPTS + 1):
            shutil.rmtree(slam_dir, ignore_errors=True)
            validation = slam(mapping, prepared, slam_dir, save_map=map_path, timeout_s=timeout)
            fatal = [e for e in validation["errors"] if e != "tracking was lost at the end"]
            score = 0 if fatal else validation["tracked_frames"]
            print(f"mapping attempt {attempt}: tracked {validation['tracked_frames']} frames, init "
                  f"{validation['first_tracked_time_s']}, ratio {validation['tracked_ratio_after_initialization']:.2f}"
                  f"{'  ' + str(fatal) if fatal else ''}", flush=True)
            if best is None or score > best:
                best = score
                shutil.rmtree(best_dir, ignore_errors=True)
                shutil.copytree(slam_dir, best_dir)
            if not fatal and validation["tracked_ratio_after_initialization"] >= MAPPING_MIN_TRACKED_RATIO \
                    and validation["tracked_frames"] >= 0.8 * validation["frames"]:
                break
        shutil.rmtree(slam_dir, ignore_errors=True)
        best_dir.rename(slam_dir)
        (slam_dir / "accepted.json").write_text(json.dumps({"attempts": attempt, "tracked_frames": best}), encoding="utf-8")
    validation = json.loads((slam_dir / "trajectory_validation.json").read_text(encoding="utf-8"))
    # A map only needs good coverage; losing tracking in the final frames does not spoil it.
    fatal = [e for e in validation["errors"] if e != "tracking was lost at the end"]
    if fatal:
        raise RuntimeError(f"mapping trajectory failed validation: {fatal}")
    if validation["tracked_ratio_after_initialization"] < MAPPING_MIN_TRACKED_RATIO:
        raise RuntimeError(f"mapping tracked only {validation['tracked_ratio_after_initialization']:.2f} "
                           f"of frames after initialization; re-record it slowly with more texture in view")
    tag_json = prepared / "tx_slam_tag.json"
    if not tag_json.exists():
        estimate_slam_tag(mapping, prepared / "slam/camera_trajectory.csv",
                          json.loads((CONFIG_DIR / "s22_camera_imu.json").read_text(encoding="utf-8")),
                          json.loads((CONFIG_DIR / "s22_slam_tag.json").read_text(encoding="utf-8")),
                          tag_json)
    tag = json.loads(tag_json.read_text(encoding="utf-8"))
    return {"session": mapping.name, "map": str(map_path), "map_sha256": _sha256(map_path),
            "slam_tag": str(tag_json), "tracked_frames": validation["tracked_frames"],
            "first_tracked_time_s": validation["first_tracked_time_s"],
            "tag_frames": tag["frames_with_tag"], "tag_samples_used": tag["samples_used"],
            "tag_position_spread_m": tag["position_spread_m"]}


def gate(outcome: str | None, validation: dict, minimum_frames: int, maximum_lost_after_init: int | None,
         allow_unrated: bool = False, allow_tracking_loss: bool = False) -> list[str]:
    """Reasons a session must stay out of the training set; empty list means included."""
    reasons = []
    if outcome != "success" and not allow_unrated:
        reasons.append(f"outcome={outcome}")
    if validation["status"] != "pass":
        reasons.extend(validation["errors"] or ["trajectory validation failed"])
    losses = validation["loss_episodes_after_initialization"]
    if losses and not allow_tracking_loss:
        reasons.append(f"tracking lost {losses} time(s) after initialization")
    if validation.get("first_tracked_frame") is not None and maximum_lost_after_init is not None:
        lost = validation["frames"] - validation["first_tracked_frame"] - validation["tracked_frames"]
        if lost > maximum_lost_after_init:
            reasons.append(f"{lost} frames lost after initialization > {maximum_lost_after_init}")
    if validation["tracked_frames"] < minimum_frames:
        reasons.append(f"tracked {validation['tracked_frames']} < {minimum_frames} frames")
    return reasons


def session_tag(session: Path, processed: Path) -> Path | None:
    """Per-recording table-marker frame from the warm-up seconds; None when the marker was not seen."""
    tag_json = processed / "tx_slam_tag.json"
    if tag_json.exists():
        return tag_json
    try:
        estimate_slam_tag(session, processed / "slam/camera_trajectory.csv",
                          json.loads((CONFIG_DIR / "s22_camera_imu.json").read_text(encoding="utf-8")),
                          json.loads((CONFIG_DIR / "s22_slam_tag.json").read_text(encoding="utf-8")),
                          tag_json)
    except ValueError:
        return None
    return tag_json


def build(root: Path, sessions: list[str], dataset_dir: Path, name: str, allow_unrated: bool,
          slam_tag: Path | None, per_session_tags: dict[str, Path] | None = None,
          trim_override_s: float | None = None) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output = dataset_dir / f"{name}.zarr.zip"
    report_path = output.with_suffix(".report.json")
    dataset_config = CONFIG_DIR / "dataset.json"
    if trim_override_s is not None:
        # With a shared map, demonstrations track from the first frame: there is no warm-up to cut.
        config = json.loads(dataset_config.read_text(encoding="utf-8"))
        config["episode_start_trim_s"] = trim_override_s
        dataset_config = dataset_dir / f"{name}.dataset_config.json"
        wanted_config = json.dumps(config, indent=2)
        if dataset_config.exists() and dataset_config.read_text(encoding="utf-8") != wanted_config:
            raise FileExistsError(f"existing dataset config differs: {dataset_config}")
        if not dataset_config.exists():
            dataset_config.write_text(wanted_config, encoding="utf-8")
    if output.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        expected = {
            "dataset": _sha256(dataset_config),
            "camera_tcp": _sha256(CONFIG_DIR / "s22_camera_tcp.json"),
        }
        actual = existing.get("config_sha256", {})
        if any(actual.get(key) != value for key, value in expected.items()):
            raise FileExistsError(f"existing dataset uses different configuration: {output}")
        return {**existing, "reused_existing": True}
    plan = dataset_dir / f"{name}.episodes.json"
    relative = lambda path: os.path.relpath(path, dataset_dir)  # noqa: E731
    per_session_tags = per_session_tags or {}
    plan.write_text(json.dumps({
        "schema_version": 1,
        **({"slam_tag": relative(slam_tag)} if slam_tag else {}),
        "episodes": [{
            "session": relative(root / "raw" / s),
            "trajectory": relative(root / "processed" / s / "slam/camera_trajectory.csv"),
            "gripper": relative(root / "processed" / s / "gripper_width.csv"),
            **({"slam_tag": relative(per_session_tags[s])} if s in per_session_tags else {}),
        } for s in sessions]}, indent=2), encoding="utf-8")
    return build_replay_buffer(plan, output, dataset_config,
                               CONFIG_DIR / "s22_camera_tcp.json", allow_unrated)


def run(root: Path, dataset_name: str = "dataset", allow_unrated: bool = False,
        allow_tracking_loss: bool = False, require_mapping: bool = False,
        allow_missing_tag: bool = False, mapping_session_id: str | None = None,
        demonstration_session_ids: list[str] | None = None,
        external_atlas: Path | None = None, external_alignment: Path | None = None) -> dict:
    root = Path(root).resolve()
    available_sessions = {path.name: path for path in _sessions(root / "raw")}
    requested_sessions = demonstration_session_ids or sorted(available_sessions)
    unknown_sessions = sorted(set(requested_sessions) - set(available_sessions))
    if unknown_sessions:
        raise FileNotFoundError(f"unknown demonstration sessions: {', '.join(unknown_sessions)}")
    sessions = [available_sessions[name] for name in requested_sessions]
    if not sessions:
        raise FileNotFoundError(f"no app sessions under {root / 'raw'}")
    if not os.environ.get("ORB_SLAM_BIN"):
        raise EnvironmentError("ORB_SLAM_BIN is not set; run `source env.sh` first")
    dataset_config = json.loads((CONFIG_DIR / "dataset.json").read_text(encoding="utf-8"))
    report = {"root": str(root), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "status": "running", "dataset_name": dataset_name, "sessions": [],
              "roles": {
                  "mapping_session_id": mapping_session_id,
                  "demonstration_session_ids": requested_sessions,
                  "role_inference_from_outcome_duration_filename_or_order": False,
              }}
    report_path = root / "pipeline_report.json"
    write = lambda: report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")  # noqa: E731

    mappings = _sessions(root / MAPPING_DIR)
    mapping_info = None
    if external_atlas is not None or external_alignment is not None:
        if external_atlas is None or external_alignment is None:
            raise ValueError("external Atlas and alignment must be provided together")
        alignment = json.loads(Path(external_alignment).read_text(encoding="utf-8"))
        if alignment.get("status") not in (None, "pass", "PASS"):
            raise ValueError("external Atlas alignment status is not pass")
        load_slam_tag(external_alignment)  # rigid-transform validation
        mapping_info = {
            "source": "external_atlas",
            "session": None,
            "map": str(Path(external_atlas).resolve()),
            "map_sha256": _sha256(Path(external_atlas)),
            "slam_tag": str(Path(external_alignment).resolve()),
            "slam_tag_sha256": _sha256(Path(external_alignment)),
            "outcome_used_as_training_label": False,
        }
        report["roles"]["role_source"] = "external_atlas"
        report["mapping"] = mapping_info
        print(f"external Atlas {mapping_info['map_sha256'][:12]} with verified alignment", flush=True)
    elif mappings:
        by_name = {path.name: path for path in mappings}
        if mapping_session_id is not None:
            if mapping_session_id not in by_name:
                raise FileNotFoundError(f"mapping session not found under {root / MAPPING_DIR}: {mapping_session_id}")
            selected_mapping = by_name[mapping_session_id]
            report["roles"]["role_source"] = "mapping_session_id"
        elif len(mappings) == 1:
            selected_mapping = mappings[0]
            report["roles"]["role_source"] = "raw_mapping_layout"
        else:
            raise ValueError("multiple Mapping sessions exist; mappingSessionId is required")
        mapping_info = build_map(root, selected_mapping)
        mapping_manifest = json.loads((selected_mapping / "manifest.json").read_text(encoding="utf-8"))
        mapping_info.update(
            source="mapping_session",
            recorded_outcome=mapping_manifest.get("outcome"),
            outcome_used_as_training_label=False,
        )
        report["mapping"] = mapping_info
        print(f"mapping {mapping_info['session']}  tracked {mapping_info['tracked_frames']} frames, "
              f"tag samples {mapping_info['tag_samples_used']}", flush=True)
    elif require_mapping:
        raise FileNotFoundError(f"no mapping session under {root / MAPPING_DIR}")
    else:
        report["mapping"] = None
        report["roles"]["role_source"] = "independent_per_episode_diagnostic"
        print(f"no {MAPPING_DIR}/ session: each recording initialises SLAM on its own and keeps its own "
              "world origin", flush=True)
    load_map = Path(mapping_info["map"]) if mapping_info else None
    trim_s = float(dataset_config.get("episode_start_trim_s", 0.0))
    write()

    included = []
    per_session_tags: dict[str, Path] = {}
    for session in sessions:
        processed = root / "processed" / session.name
        row = {"session": session.name, "warnings": []}
        try:
            recording = load_recording(session, require_success=False)
            row.update(outcome=recording["manifest"].get("outcome"), frames=recording["frame_count"],
                       duration_s=round(recording["duration_s"], 2))
            prepared = prepare(session, processed)
            row["gripper_markers"] = {
                "semantics": "jaw markers used to derive gap; not table ArUco ID 13",
                "detection_rate": round(prepared["gripper"]["detection_rate"], 4),
                "max_missing_run": prepared["gripper"]["maximum_missing_run_frames"],
            }
            row["table_marker_id13"] = {
                "required_for_demonstration": not bool(load_map),
                "semantics": (
                    "diagnostic_only; shared aligned Atlas supplies the table frame"
                    if load_map else "required to align this independently initialised episode"
                ),
            }
            attempts_file = processed / "slam_attempts.json"
            for attempt in range(1, DEMO_ATTEMPTS + 1):
                validation = slam(session, processed, processed / "slam", load_map=load_map,
                                  timeout_s=slam_timeout_s(recording["duration_s"]))
                receipt = (processed / "slam/receipt.json").read_text(encoding="utf-8")
                # Attempts are remembered per receipt (map/mask/settings/binary) so a session that already
                # failed DEMO_ATTEMPTS times is not re-run on every pipeline invocation.
                history = json.loads(attempts_file.read_text(encoding="utf-8")) if attempts_file.exists() else {}
                used = history.get("attempts", 0) + 1 if history.get("receipt") == receipt else 1
                attempts_file.write_text(json.dumps({"receipt": receipt, "attempts": used}), encoding="utf-8")
                reasons = gate(row["outcome"], validation, dataset_config["minimum_episode_frames"],
                               dataset_config.get("maximum_lost_frames_after_initialization"),
                               allow_unrated, allow_tracking_loss)
                if load_map and validation["tracked_ratio_after_initialization"] < DEMO_MIN_TRACKED_RATIO:
                    reasons.append(
                        f"tracked coverage {validation['tracked_ratio_after_initialization']:.4f} "
                        f"< {DEMO_MIN_TRACKED_RATIO:.4f}"
                    )
                if load_map and validation["pose_coverage"] < DEMO_MIN_TRACKED_RATIO:
                    reasons.append(
                        f"pose coverage {validation['pose_coverage']:.4f} "
                        f"< {DEMO_MIN_TRACKED_RATIO:.4f}"
                    )
                if load_map and validation.get("first_tracked_time_s") is not None \
                        and validation["first_tracked_time_s"] > LATE_RELOCALIZATION_S:
                    reasons.append(
                        f"first pose {validation['first_tracked_time_s']:.3f}s > "
                        f"{LATE_RELOCALIZATION_S:.3f}s"
                    )
                tracking_only = reasons and all("lost" in r or "tracked" in r for r in reasons)
                if not tracking_only or used >= DEMO_ATTEMPTS:
                    attempt = used
                    break
                # Same non-repeatability as mapping: a lost run often tracks fully when repeated.
                shutil.rmtree(processed / "slam", ignore_errors=True)
            row["slam"] = {key: validation[key] for key in (
                "status", "first_tracked_time_s", "tracked_frames",
                "pose_coverage", "final_frame_tracked",
                "loss_episodes_after_initialization", "path_length_m")}
            row["slam"]["first_tracked_pose"] = validation.get("first_tracked_pose")
            row["slam"]["pose_continuity"] = validation.get("pose_continuity")
            row["slam"]["atlas_integrity"] = validation.get("atlas_integrity")
            row["slam"]["attempts"] = attempt
            first = validation["first_tracked_time_s"]
            if load_map and first is not None and first > LATE_RELOCALIZATION_S:
                row["warnings"].append(f"relocalised only at {first:.2f}s despite the map; "
                                       "start recordings looking at the mapped workspace")
            if not load_map and first is not None and first > trim_s:
                row["warnings"].append(f"SLAM initialised at {first:.2f}s, after the {trim_s}s warm-up trim; "
                                       "the start of the task is cut")
            if not load_map and not reasons:
                # Without a shared map each recording gets its world frame from the marker it saw
                # during warm-up, so every included session must have seen it.
                tag = session_tag(session, processed)
                if tag is not None:
                    per_session_tags[session.name] = tag
                    row["table_tag"] = json.loads(tag.read_text(encoding="utf-8"))["samples_used"]
                elif not allow_missing_tag:
                    reasons.append("table marker (ArUco 13) not seen while tracked; keep it in view during warm-up")
                else:
                    row["warnings"].append("no table marker; world frame is this recording's SLAM origin")
        except Exception as error:  # one broken session must not stop the batch; the reason is recorded
            reasons = [f"{type(error).__name__}: {error}"]
        row["included"] = not reasons
        row["excluded_because"] = reasons
        report["sessions"].append(row)
        if not reasons:
            included.append(session.name)
        print(f"{session.name}  {'INCLUDED' if not reasons else 'excluded: ' + '; '.join(reasons)}"
              f"{'  WARN: ' + '; '.join(row['warnings']) if row['warnings'] else ''}", flush=True)
    report["included"] = len(included)
    report["submitted"] = len(sessions)
    report["excluded"] = len(sessions) - len(included)
    write()
    if len(included) < MINIMUM_TRAINING_EPISODES:
        report["stopped"] = (f"only {len(included)} session(s) passed the gate; "
                             f"need at least {MINIMUM_TRAINING_EPISODES}")
        report["status"] = "stopped"
        write()
        return report
    # Dry-run 02's episode assembly per session so one bad label file excludes that session with a
    # reason instead of aborting the whole build.
    camera_tcp, _ = load_camera_tcp(CONFIG_DIR / "s22_camera_tcp.json")
    episode_config = dict(dataset_config, episode_start_trim_s=0.0 if mapping_info else trim_s)
    shared_tag = Path(mapping_info["slam_tag"]) if mapping_info else None
    for row in report["sessions"]:
        if not row["included"]:
            continue
        name = row["session"]
        tag_path = shared_tag or per_session_tags.get(name)
        try:
            recording = load_recording(root / "raw" / name, require_success=not allow_unrated)
            episode_data, selected, episode_report = build_episode(
                recording, root / "processed" / name / "slam/camera_trajectory.csv",
                root / "processed" / name / "gripper_width.csv", camera_tcp, episode_config,
                np.linalg.inv(load_slam_tag(tag_path)) if tag_path else None)
            row["episode_frames"] = int(len(selected))
            row["episode_gate_metrics"] = episode_report
            row["start_tcp_6dof"] = {
                "position_m": episode_data["robot0_eef_pos"][0].astype(float).tolist(),
                "rotation_axis_angle_rad": episode_data["robot0_eef_rot_axis_angle"][0].astype(float).tolist(),
                "frame": "table_tag" if tag_path else "slam_origin_of_this_recording",
            }
        except ValueError as error:
            row["included"] = False
            row["excluded_because"].append(f"episode assembly: {error}")
            included.remove(name)
            print(f"{name}  excluded at assembly: {error}", flush=True)
    report["included"] = len(included)
    report["excluded"] = len(sessions) - len(included)
    if len(included) < MINIMUM_TRAINING_EPISODES:
        report["stopped"] = (f"only {len(included)} session(s) survived episode assembly; "
                             f"need at least {MINIMUM_TRAINING_EPISODES}")
        report["status"] = "stopped"
        write()
        return report
    if not load_map and per_session_tags and len(per_session_tags) != len(included):
        # 02 refuses mixed frames; drop the untagged ones here so the report says why.
        untagged = [s for s in included if s not in per_session_tags]
        for row in report["sessions"]:
            if row["session"] in untagged:
                row["included"] = False
                row["excluded_because"].append("no table marker while other sessions have one (mixed world frames)")
        included = [s for s in included if s in per_session_tags]
        report["included"] = len(included)
        report["excluded"] = len(sessions) - len(included)
    dataset_report = build(root, included, root / "dataset", dataset_name, allow_unrated,
                           Path(mapping_info["slam_tag"]) if mapping_info else None, per_session_tags,
                           trim_override_s=0.0 if mapping_info else None)
    report["dataset"] = {key: dataset_report[key] for key in (
        "dataset", "episodes", "frames", "training_input_status", "physical_deployment_ready", "world_frame")}
    report["dataset"]["reused_existing"] = dataset_report.get("reused_existing", False)
    if report["dataset"]["reused_existing"] and dataset_report["episodes"] != len(included):
        # ORB-SLAM3 is multi-threaded and not bit-reproducible, so the gate can admit a different
        # set on a re-run; the existing dataset still wins unless a new --dataset-name is given.
        report["warning"] = (f"reused dataset has {dataset_report['episodes']} episodes but the gate now "
                             f"admits {len(included)}; pass --dataset-name <new> to rebuild")
        print("WARNING:", report["warning"], flush=True)
    report["status"] = "ready"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write()
    return report
