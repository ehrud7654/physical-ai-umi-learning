"""Policy-free counterfactual TCP targets for displaced simulation objects.

Reference anchors are reconstructed from the recorded first future step.
Every future world target receives the object displacement once, then is
re-expressed relative to the actual robot TCP. This prevents a receding-horizon
replay from adding the same displacement at every control cycle.
"""
from __future__ import annotations

import numpy as np

from umi.relative_dataset import relative_vector, vector_to_transform


def reference_anchors(actions: np.ndarray, *, start_row: int,
                      start_pose: np.ndarray) -> np.ndarray:
    """Reconstruct recorded TCP anchors from a predeclared fixed start row."""
    chunks = np.asarray(actions, dtype=np.float64)
    origin = np.asarray(start_pose, dtype=np.float64)
    if (chunks.ndim != 3 or chunks.shape[1:] != (8, 10)
            or origin.shape != (4, 4) or not np.isfinite(chunks).all()
            or not np.isfinite(origin).all() or not 0 <= start_row < len(chunks)):
        raise ValueError("expected finite action[N,8,10] and fixed start pose")
    anchors = [origin.copy()]
    for row in range(start_row + 1, len(chunks)):
        anchors.append(anchors[-1] @ vector_to_transform(chunks[row - 1, 0]))
    return np.stack(anchors)


def shifted_chunk_from_reference(
        action: np.ndarray, *, reference_pose: np.ndarray,
        actual_pose: np.ndarray, delta_world_xyz: np.ndarray) -> np.ndarray:
    """Shift each absolute world target once; preserve recorded rotation/gap."""
    chunk = np.asarray(action, dtype=np.float64)
    reference = np.asarray(reference_pose, dtype=np.float64)
    actual = np.asarray(actual_pose, dtype=np.float64)
    delta = np.asarray(delta_world_xyz, dtype=np.float64)
    if (chunk.shape != (8, 10) or reference.shape != (4, 4)
            or actual.shape != (4, 4) or delta.shape != (3,)
            or not np.isfinite(chunk).all() or not np.isfinite(reference).all()
            or not np.isfinite(actual).all() or not np.isfinite(delta).all()):
        raise ValueError("expected finite chunk[8,10], two poses[4,4], delta[3]")
    translated = []
    for future in chunk:
        world_target = reference @ vector_to_transform(future)
        world_target[:3, 3] += delta
        translated.append(relative_vector(actual, world_target, float(future[9])))
    return np.stack(translated).astype(np.float32)


def dense_executed_future_rows(
        images: np.ndarray, poses: np.ndarray, gaps: np.ndarray,
        timestamps: np.ndarray, source_rows: np.ndarray,
        *, rate_hz: float = 10.0) -> dict[str, np.ndarray]:
    """Make H=2/K=8 targets from measured simulator samples, not commands.

    The caller must sample the simulator on the stated clock. There is no
    interpolation, endpoint clamp, repeated image, or invented controller
    command. These remain non-training-ready counterfactual candidates.
    """
    rgb = np.asarray(images)
    tcp = np.asarray(poses, dtype=np.float64)
    gap = np.asarray(gaps, dtype=np.float64)
    time = np.asarray(timestamps, dtype=np.float64)
    row = np.asarray(source_rows, dtype=np.int64)
    n = len(time)
    if (not np.isfinite(rate_hz) or rate_hz <= 0 or n < 10
            or rgb.shape != (n, 3, 224, 224) or rgb.dtype != np.uint8
            or tcp.shape != (n, 4, 4) or gap.shape != (n,)
            or row.shape != (n,) or not np.isfinite(tcp).all()
            or not np.isfinite(gap).all() or not np.isfinite(time).all()
            or np.any(np.diff(row) < 0)
            or not np.allclose(np.diff(time), 1.0 / rate_hz,
                               atol=1e-6, rtol=0)):
        raise ValueError("need at least 10 finite, uniformly timed 10Hz samples")
    samples = {key: [] for key in (
        "image", "proprio", "action", "observation_timestamp",
        "action_timestamp", "source_row")}
    for current in range(1, n - 8):
        history = (current - 1, current)
        future = range(current + 1, current + 9)
        samples["image"].append(rgb[list(history)])
        samples["proprio"].append(np.stack([
            relative_vector(tcp[current], tcp[index], float(gap[index]))
            for index in history]))
        samples["action"].append(np.stack([
            relative_vector(tcp[current], tcp[index], float(gap[index]))
            for index in future]))
        samples["observation_timestamp"].append(time[list(history)])
        samples["action_timestamp"].append(time[list(future)])
        samples["source_row"].append(int(row[current]))
    return {
        "image": np.stack(samples["image"]).astype(np.uint8),
        "proprio": np.stack(samples["proprio"]).astype(np.float32),
        "action": np.stack(samples["action"]).astype(np.float32),
        "observation_timestamp": np.stack(
            samples["observation_timestamp"]).astype(np.float64),
        "action_timestamp": np.stack(samples["action_timestamp"]).astype(np.float64),
        "source_row": np.asarray(samples["source_row"], dtype=np.int64),
    }
