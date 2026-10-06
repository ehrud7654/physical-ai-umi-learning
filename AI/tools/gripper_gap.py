"""Generate per-frame UMI gripper gaps from fluorescent magenta markers."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from umi.gripper_gap import measure_gap


def _single(path: Path) -> dict:
    with Image.open(path) as image:
        return measure_gap(image.convert("RGB")).json_dict()


def _bundle(bundle: Path, output: Path) -> dict:
    """Process every arpose recording in a delivery ZIP without modifying raw data."""
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    counts = {key: 0 for key in "DMTX"}
    episodes = []
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        pose_files = sorted(name for name in names if name.endswith("/poses.csv"))
        for pose_name in pose_files:
            prefix = str(PurePosixPath(pose_name).parent)
            rows = list(csv.DictReader(io.StringIO(archive.read(pose_name).decode("utf-8-sig"))))
            target = output / PurePosixPath(prefix).name
            target.mkdir()
            report_rows = []
            with (target / "gripper.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame_index", "gap_m", "status"])
                for row in rows:
                    image_name = row.get("image", "")
                    image_path = PurePosixPath(image_name)
                    # arpose.episode/1 stores only ``000000.jpg`` in poses.csv
                    # while the actual archive member is under frames/.  Also
                    # accept an already-qualified frames/... path.
                    if image_path.parent == PurePosixPath('.'):
                        image_path = PurePosixPath("frames") / image_path
                    member = str(PurePosixPath(prefix) / image_path)
                    if not image_name or member not in names:
                        result = {"gap_m": None, "status": "X", "reason": "image_missing"}
                    else:
                        with Image.open(io.BytesIO(archive.read(member))) as image:
                            result = measure_gap(image.convert("RGB")).json_dict()
                    gap = "" if result["gap_m"] is None else f'{result["gap_m"]:.9f}'
                    writer.writerow([row["index"], gap, result["status"]])
                    counts[result["status"]] += 1
                    report_rows.append({"frame_index": int(row["index"]), **result})
            (target / "gripper_qc.json").write_text(
                json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            episodes.append({"episode": target.name, "frames": len(rows)})
    summary = {"episodes": episodes, "status_counts": counts}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--bundle", type=Path)
    parser.add_argument("--out", type=Path, help="required with --bundle; raw ZIP is never modified")
    args = parser.parse_args()
    if args.image:
        print(json.dumps(_single(args.image), ensure_ascii=False, indent=2))
        return
    if args.out is None:
        parser.error("--out is required with --bundle")
    print(json.dumps(_bundle(args.bundle, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
