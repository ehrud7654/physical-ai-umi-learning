"""Focused tests for the read-only real/sim input-support audits."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.audit_v10_handoff_observation_support import _features, _summarize
from tools.audit_v10_real_visual_coverage import _bbox, _stats
from tools.compare_v10_real_label_sim_response import _association
from tools.probe_v10_real_image_position import _position_bins


def main() -> int:
    image = np.zeros((3, 224, 224), dtype=np.uint8)
    image[:, 30:190, 80:120] = np.array([255, 200, 0], dtype=np.uint8)[:, None, None]
    box = _bbox(image)
    assert box is not None
    assert np.allclose(box, [100 / 224, 110 / 224, 40 / 224, 160 / 224])

    summary = _stats([[0.1, 0.2], [0.3, 0.4]])
    assert summary["samples"] == 2
    assert np.allclose(summary["p50"], [0.2, 0.3])

    pair = np.zeros((2, 3, 224, 224), dtype=np.uint8)
    pair[1] = 255
    proprio = np.zeros((2, 10), dtype=np.float32)
    proprio[0, :3] = [0.01, 0.0, 0.0]
    proprio[0, 3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    proprio[1, -1] = 0.065
    feature = _features(pair, proprio, gap_min_m=0.06, gap_max_m=0.07)
    assert feature is not None
    assert np.isclose(feature["h2_translation_m"], 0.01)
    assert np.isclose(feature["h2_rotation_deg"], 0.0)
    assert np.isclose(feature["h2_rgb_mae_uint8"], 255.0)
    assert _features(pair, proprio, gap_min_m=0.03, gap_max_m=0.04) is None
    assert _summarize([feature])["samples"] == 1

    relation = _association(np.array([
        [0.4, 0.02], [0.5, 0.0], [0.6, -0.02]]))
    assert np.isclose(relation["pearson_r"], -1.0)
    assert np.isclose(relation["slope_m_per_normalized_image_x"], -0.2)
    bins = _position_bins([
        {"bbox_center_x": 0.40, "recorded_target_x_m": 0.02,
         "pred_original_proprio_x_m": 0.001},
        {"bbox_center_x": 0.50, "recorded_target_x_m": 0.001,
         "pred_original_proprio_x_m": 0.0},
    ])
    assert bins["left_x_lt_0.45"]["recorded_large_x_gt_10mm"] == 1
    assert np.isclose(bins["left_x_lt_0.45"]["original_proprio_x_mae_m"], .019)
    assert bins["right_x_gt_0.55"]["original_proprio_x_mae_m"] is None
    print("v10 localisation factor checks: 5/5 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
