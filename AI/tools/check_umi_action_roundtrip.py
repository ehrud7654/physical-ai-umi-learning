"""MuJoCo kinematic roundtrip (no rendering, dynamics or motor commands)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
from sim.mujoco.kinematics import pinch_offset_for_gap
from tools.umi_mujoco import (MujocoIK, gap_from_angle, load_config, build_model,
                              DEFAULT_CONFIG, DEFAULT_SCENE)
from umi.ik import quat_to_matrix
from umi.action_ik import actions_to_joints
from umi.arm_preflight import prepare_arm_commands
from umi.convert import invert_gap_curve
from tools.umi_mujoco import MujocoArmAdapter


def main():
    cfg = load_config(DEFAULT_CONFIG)
    model = build_model(cfg, DEFAULT_SCENE)
    ik = MujocoIK(model, cfg)
    data = mujoco.MjData(model)
    curve = cfg['grasp']['gap_curve']
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
    pf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'pad_fixed')
    pm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'pad_moving')
    midpoint_x_errors = []
    for angle, _ in curve:
        data.qpos[:6] = 0.0
        data.qpos[5] = angle
        mujoco.mj_forward(model, data)
        rot = data.xmat[gb].reshape(3, 3)
        origin = data.xpos[gb]
        fixed = rot.T @ (data.geom_xpos[pf] - origin)
        moving = rot.T @ (data.geom_xpos[pm] - origin)
        fixed_inner_x = fixed[0] + model.geom_size[pf][0]
        moving_inner_x = moving[0] - model.geom_size[pm][0]
        measured_midpoint_x = 0.5 * (fixed_inner_x + moving_inner_x)
        configured_gap = gap_from_angle(angle, curve)
        configured_midpoint_x = pinch_offset_for_gap(
            cfg['grasp'], configured_gap)[0]
        midpoint_x_errors.append(abs(measured_midpoint_x - configured_midpoint_x))
    if max(midpoint_x_errors) > 5e-4:
        raise AssertionError('gap-dependent midpoint disagrees with pad geometry')
    print(f'pad midpoint x max error_mm={max(midpoint_x_errors)*1000:.4f}')
    def fk(q):
        pos, quat = ik.forward_pose(q)
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = quat_to_matrix(quat), pos
        return t
    rng = np.random.default_rng(20260909)
    pos_errors, rot_errors, gap_errors, jumps = [], [], [], []
    for trial in range(5):
        q0 = (ik.ranges[:, 0] + ik.ranges[:, 1]) / 2
        q0[:5] += rng.uniform(-0.15, 0.15, 5)
        current = fk(q0)
        direction = rng.uniform(-0.04, 0.04, 6)
        truth = q0 + np.linspace(0, 1, 24)[:, None] * direction
        absolute = np.stack([fk(q) for q in truth])
        relative = np.linalg.inv(current) @ absolute
        a = np.zeros((len(truth), 10))
        a[:, :3] = relative[:, :3, 3]
        a[:, 3:9] = relative[:, :2, :3].reshape(-1, 6)
        a[:, 9] = [gap_from_angle(q[5], curve) for q in truth]
        recovered = actions_to_joints(a, current, q0, ik, ik.ranges, curve, max_step_rad=0.1)
        class Policy:
            def predict_action(self, observation):
                return {'action_pred': a}
        commands = prepare_arm_commands(Policy(), {}, t_current=current,
            arm_current_rad=q0[:5], gripper_current_m=gap_from_angle(q0[5], curve),
            solver=MujocoArmAdapter(ik, curve), ranges_rad=ik.ranges[:5],
            gap_range_m=[min(row[1] for row in curve)/100, max(row[1] for row in curve)/100],
            max_step_rad=0.1, max_gap_step_m=0.01, max_position_error_m=0.005,
            max_axis_error_deg=5, max_roll_error_deg=5)
        restored = np.array([np.r_[c.arm_positions_rad, invert_gap_curve(c.gripper_width_m, curve)]
                             for c in commands])
        np.testing.assert_allclose(restored, recovered, atol=1e-10)
        for target, q, gap in zip(absolute, recovered, a[:, 9]):
            achieved = fk(q)
            pos_errors.append(np.linalg.norm(target[:3, 3] - achieved[:3, 3]))
            cosine = (np.trace(target[:3, :3].T @ achieved[:3, :3]) - 1) / 2
            rot_errors.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            gap_errors.append(abs(gap_from_angle(q[5], curve) - gap))
        jumps.append(np.abs(np.diff(np.vstack([q0, recovered]), axis=0)).max())
    print(f'MuJoCo={mujoco.__version__}, seed=20260909, targets={len(pos_errors)}')
    print(f'max position_mm={max(pos_errors)*1000:.8f}, rotation_deg={max(rot_errors):.8f}')
    print(f'max gap_m={max(gap_errors):.3e}, joint_step_rad={max(jumps):.8f}')
    if max(pos_errors) > 5e-3 or max(rot_errors) > 0.1 or max(gap_errors) > 1e-8:
        raise AssertionError('roundtrip error budget exceeded')
    # An unreachable Cartesian target must not yield an executable chunk.
    a = a[:1].copy()
    a[0, :3] = [10, 10, 10]
    try:
        actions_to_joints(a, current, q0, ik, ik.ranges, curve, max_step_rad=0.1)
    except ValueError:
        print('unreachable target: rejected; PASS')
    else:
        raise AssertionError('unreachable target accepted')


if __name__ == '__main__':
    main()
