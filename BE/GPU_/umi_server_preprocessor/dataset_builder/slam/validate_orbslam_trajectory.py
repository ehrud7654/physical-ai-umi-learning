import argparse
import csv
import json
import math
from pathlib import Path


def validate(csv_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("trajectory CSV is empty")

    parsed = []
    for row in rows:
        parsed.append(
            {
                "frame": int(row["frame_idx"]),
                "time": float(row["timestamp"]),
                "tracked": row["state"] == "2" and row["is_lost"].lower() == "false",
                "keyframe": row["is_keyframe"].lower() == "true",
                "position": tuple(float(row[name]) for name in ("x", "y", "z")),
                "quaternion": tuple(float(row[name]) for name in ("q_x", "q_y", "q_z", "q_w")),
            }
        )

    tracked = [row for row in parsed if row["tracked"]]
    first = tracked[0] if tracked else None
    last = tracked[-1] if tracked else None
    after_init = [row for row in parsed if first and row["frame"] >= first["frame"]]
    loss_episodes = 0
    in_loss = False
    for row in after_init:
        if not row["tracked"] and not in_loss:
            loss_episodes += 1
            in_loss = True
        elif row["tracked"]:
            in_loss = False

    path_length = 0.0
    for previous, current in zip(parsed, parsed[1:]):
        if previous["tracked"] and current["tracked"] and current["frame"] == previous["frame"] + 1:
            path_length += math.dist(previous["position"], current["position"])

    finite = all(
        math.isfinite(value)
        for row in tracked
        for value in (*row["position"], *row["quaternion"])
    )
    contiguous = all(current["frame"] == previous["frame"] + 1 for previous, current in zip(parsed, parsed[1:]))
    monotonic_time = all(current["time"] >= previous["time"] for previous, current in zip(parsed, parsed[1:]))
    axes = list(zip(*(row["position"] for row in tracked))) if tracked else []
    extents = [max(axis) - min(axis) for axis in axes] if axes else []
    errors = []
    if not tracked:
        errors.append("no tracked frames")
    if not finite:
        errors.append("non-finite tracked pose")
    if not contiguous:
        errors.append("frame indices are not contiguous")
    if not monotonic_time:
        errors.append("timestamps are not monotonic")
    if last and last["frame"] != parsed[-1]["frame"]:
        errors.append("tracking was lost at the end")

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "frames": len(parsed),
        "duration_s": parsed[-1]["time"] - parsed[0]["time"],
        "tracked_frames": len(tracked),
        "first_tracked_frame": first["frame"] if first else None,
        "first_tracked_time_s": first["time"] if first else None,
        "tracked_ratio_after_initialization": len(tracked) / len(after_init) if after_init else 0.0,
        "loss_episodes_after_initialization": loss_episodes,
        "keyframes": sum(row["keyframe"] for row in parsed),
        "path_length_m": path_length,
        "position_extent_m": extents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an ORB-SLAM3 camera trajectory CSV.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.csv)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(report["status"] != "pass")


if __name__ == "__main__":
    main()
