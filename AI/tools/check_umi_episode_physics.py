"""Checks the per-demonstration object/physics metadata contract."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from umi.episode_physics import SCHEMA, load_episode_physics, validate_document


def main() -> int:
    config = Path(__file__).resolve().parents[1] / "configs/real/umi_episode_physics_20260911.json"
    episode = load_episode_physics(config, "rec_20260911_150920")
    assert episode.object_type == "yellow_cup_proxy"
    assert episode.half_size_m.tolist() == [0.02, 0.02, 0.05]
    assert episode.grasp_height_m == 0.08
    assert episode.evidence == "provisional"
    np.testing.assert_allclose(episode.object_center_from_pinch_m, [0, 0, .03])
    assert np.isclose(np.linalg.det(np.asarray(episode.grasp_rotation_base)), 1.0)

    invalid = {"schema": SCHEMA, "episodes": {"bad": {
        "object_type": "cup", "shape": "box", "size_m": [0.04, -0.04, 0.1],
        "grasp_height_m": 0.2, "evidence": "guessed", "source": "",
        "object_center_from_pinch_m": [0, 0, float("nan")],
        "registration_evidence": "guessed", "grasp_rotation_base": [[1, 0, 0]] * 3}}}
    problems = validate_document(invalid)
    assert any("positive" in problem for problem in problems)
    assert any("cannot exceed" in problem for problem in problems)
    assert any("evidence" in problem for problem in problems)
    assert any("source" in problem for problem in problems)
    assert any("object_center_from_pinch" in problem for problem in problems)
    assert any("registration_evidence" in problem for problem in problems)
    assert any("grasp_rotation_base" in problem for problem in problems)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "meta.json"
        path.write_text(json.dumps({"schema": SCHEMA, "episodes": {
            "known": {"object_type": "cup", "shape": "box",
                      "size_m": [0.04, 0.04, 0.10], "grasp_height_m": 0.08,
                      "evidence": "verified", "source": "caliper",
                      "object_center_from_pinch_m": [0, 0, 0],
                      "registration_evidence": "verified",
                      "grasp_rotation_base": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}}),
                        encoding="utf-8")
        try:
            load_episode_physics(path, "unknown")
        except ValueError as exc:
            assert "no physics metadata" in str(exc)
        else:
            raise AssertionError("unlabelled episodes must be rejected")
    print("PASS: per-episode object type, size, grasp height and evidence contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
