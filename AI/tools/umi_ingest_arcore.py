"""Local-only ARCore ingestion. Defaults to validation without writing a dataset."""
import argparse
import io
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_umi_regression import is_shared_gpu_server


def pilot_summary(raw):
    import numpy as np
    statuses = [str(value) for value in raw.gripper_status]
    counts = {key: statuses.count(key) for key in 'DMTX'}
    longest_missing, current = 0, 0
    for status in statuses:
        current = current + 1 if status in ('T', 'X') else 0
        longest_missing = max(longest_missing, current)
    finite = raw.gripper_gap_m[np.isfinite(raw.gripper_gap_m)]
    duration = float(raw.pose_timestamp[-1] - raw.pose_timestamp[0]) if raw.meta.n_steps > 1 else 0.0
    producer_recorded = raw.meta.notes.get('gripper_producer') is not None
    issues = []
    if duration < 20.0:
        issues.append('usable_duration_under_20s')
    if not finite.size:
        issues.append('no_valid_gripper_gap')
    if not producer_recorded:
        issues.append('gripper_producer_missing')
    return {
        'status_counts': counts,
        'longest_tx_run': longest_missing,
        'gap_min_m': float(finite.min()) if finite.size else None,
        'gap_max_m': float(finite.max()) if finite.size else None,
        'usable_duration_s': duration,
        'pilot_duration_20s_pass': duration >= 20.0,
        'effective_start_s': raw.meta.notes.get('effective_start_s'),
        'gripper_producer_recorded': producer_recorded,
        'pilot_issues': issues,
        'pilot_pass': not issues,
    }


def main():
    if is_shared_gpu_server():
        raise SystemExit('Shared GPU server is training-only')
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True,
                        help='JSON calibration_id, t_cam_to_pinch, t_arcore_world_to_base')
    parser.add_argument('--quality',type=Path,required=True,
                        help='JSON tracking_valid_values, warmup_s, usable_segments, frames_dropped; optional stabilized_at_s/demo_start_s/pre_stabilized')
    parser.add_argument('--skill-id')
    parser.add_argument('--out',type=Path,help='New output directory; omit for validation only')
    args=parser.parse_args()
    import numpy as np
    from PIL import Image
    from track_a.convert.arcore import to_raw
    from umi.raw import write_raw
    from umi.provenance import conversion_digest
    calibration=json.loads(args.calibration.read_text(encoding='utf-8'))
    quality=json.loads(args.quality.read_text(encoding='utf-8'))
    if args.out is not None and args.out.exists():
        raise SystemExit('Output directory already exists; use a new name')
    def decode(data):
        with Image.open(io.BytesIO(data)) as image:
            return np.array(image.convert('RGB'))
    raw=to_raw(bundle=args.bundle,image_decoder=decode,skill_id=args.skill_id,
               **calibration,**quality)
    raw.meta.notes['tool_sha']=conversion_digest()
    print(json.dumps({'steps':raw.meta.n_steps,'cameras':raw.meta.cameras,
        'source_pose_rows':raw.meta.notes['source_pose_rows'],
        'pilot':pilot_summary(raw),
        'training_ready':False,'reason':'raw ingestion complete; IK conversion and contract validation still required'},
        ensure_ascii=False))
    if args.out is not None:
        print(write_raw(raw,args.out))


if __name__=='__main__': main()
