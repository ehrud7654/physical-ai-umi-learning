"""ARCore bundle to RawEpisode. Synthetic-tested; real capture verification pending.

Transforms map coordinates: T_base_world @ T_world_camera @ T_camera_pinch.
The caller must provide measured calibration and reviewed quality policy.
summary.txt grammar and tracking enums remain externally supplied contracts.
"""
from pathlib import Path
import numpy as np
from umi.raw import RawEpisode
from track_a.convert.arcore_bundle import decode_bundle

__all__ = ["to_raw"]


def to_raw(*, bundle: Path, t_cam_to_pinch: np.ndarray,
           t_cam_to_pinch_axes: str,
           t_arcore_world_to_base: np.ndarray, calibration_id: str,
           tracking_valid_values, warmup_s, usable_segments,
           frames_dropped, image_decoder, recording_id=None, skill_id=None,
           stabilized_at_s=None, demo_start_s=None,
           pre_stabilized=False) -> RawEpisode:
    """Read ZIP/root directory without extracting or re-encoding source images.

    usable_segments: half-open original pose ROW offsets.
    warmup_s: normally >=3 seconds from first pose timestamp.  Zero is accepted
      only when the capture protocol explicitly records pre_stabilized=True.
    stabilized_at_s/demo_start_s: optional seconds from the first pose timestamp.
    The latest of these offsets and warmup_s is the first eligible sample.
    image_decoder: bytes -> original-resolution RGB uint8 array.
    tracking_valid_values: explicit whitelist from the recorder contract.
    Only cam_wrist is returned; cam_front is not fabricated.
    """
    return decode_bundle(bundle=bundle, t_camera_pinch=t_cam_to_pinch,
        t_camera_pinch_axes=t_cam_to_pinch_axes,
        t_base_world=t_arcore_world_to_base, calibration_id=calibration_id,
        recording_id=recording_id, skill_id=skill_id, tracking_valid_values=tracking_valid_values,
        warmup_s=warmup_s, usable_segments=usable_segments, frames_dropped=frames_dropped,
        image_decoder=image_decoder, stabilized_at_s=stabilized_at_s,
        demo_start_s=demo_start_s, pre_stabilized=pre_stabilized)
