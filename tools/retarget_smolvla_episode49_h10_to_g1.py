#!/usr/bin/env python3
"""Offline-only SmolVLA ALOHA FK to the established G1/Dex3 retargeting pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# The established G1 environment intentionally has no pandas. Imported legacy
# modules use pandas only in dataset-loading paths that this NPZ-only tool never
# calls, so expose a non-functional placeholder rather than changing environments.
try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pandas"] = types.ModuleType("pandas")

import generate_aloha_work_posture_dex3 as work  # noqa: E402
import generate_horizontal_approach_hold as approach_mod  # noqa: E402
import search_g1_task_ready_anchor as align_mod  # noqa: E402
import validate_g1_targets_and_sparse_ik as ik  # noqa: E402
import validate_smolvla_in_stationary_aloha_mujoco as aloha_validate  # noqa: E402
from map_aloha_gripper_to_dex3 import full_qpos, model_layout, poses, hand_trajectory  # noqa: E402

DEFAULT_ALOHA = ROOT / "evaluation/mujoco_stationary_aloha_validation/episode_000049/smolvla_h10_trajectory.npz"
DEFAULT_FK = ROOT / "evaluation/mujoco_stationary_aloha_validation/episode_000049/end_effector_trajectories.npz"
DEFAULT_XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
DEFAULT_OUT = ROOT / "converted_runs/smolvla_20k_episode49_h10_g1"
G1_MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
REFERENCE = ROOT / "converted_runs/magsafe_20260723_162750/wrist_relative_rotation/g1_wrist_relative_full_trajectory.npz"
CHECKPOINT = ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407/checkpoints/020000/pretrained_model"
FPS = 30.0
ANCHOR_LEFT = np.array([.20, .28, .94])
ANCHOR_RIGHT = np.array([.20, -.28, .94])
ALIGN_RPY = np.array([0., -7., 0.])
ORIENTATION_WEIGHT = .0075
NOMINAL_WEIGHT = .005
APPROACH_FRAMES = 90
HOLD_FRAMES = 15
C_LEFT = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
C_RIGHT = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aloha-trajectory-npz", type=Path, default=DEFAULT_ALOHA)
    p.add_argument("--aloha-fk-npz", type=Path, default=DEFAULT_FK)
    p.add_argument("--mujoco-xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--episode", type=int, default=49)
    p.add_argument("--position-scale", type=float, default=.42)
    p.add_argument("--orientation-mode", choices=("existing_filtered_wrist_relative_C",),
                   default="existing_filtered_wrist_relative_C")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def matrix_quat(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def relative_rotation(rot: np.ndarray) -> np.ndarray:
    return np.einsum("ji,tjk->tik", rot[0], rot)


def existing_filtered_relative(rot: np.ndarray) -> np.ndarray:
    rv = Rotation.from_matrix(relative_rotation(rot)).as_rotvec()
    smooth = savgol_filter(rv, 21, 3, axis=0, mode="interp")
    smooth -= smooth[0]
    return Rotation.from_rotvec(smooth).as_matrix()


def map_relative(rot: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.einsum("ij,tjk,kl->til", c.T, rot, c)


def rotation_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Rotation.from_matrix(np.einsum("tji,tjk->tik", a, b)).as_rotvec(), axis=1)


def metrics(x: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(x)), "max": float(np.max(x)), "p95": float(np.percentile(x, 95))}


def load_inputs(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.episode != 49:
        raise RuntimeError("This safety-scoped script only permits episode 49")
    for path in (args.aloha_trajectory_npz, args.aloha_fk_npz, args.mujoco_xml, G1_MODEL, REFERENCE, CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(path)
    with np.load(args.aloha_trajectory_npz, allow_pickle=False) as z:
        if "raw" not in z.files:
            raise RuntimeError(f"Expected key 'raw'; actual={z.files}")
        raw = z["raw"].astype(np.float64)
        timestamp = z["timestamp"].astype(np.float64)
    with np.load(args.aloha_fk_npz, allow_pickle=False) as z:
        required = [f"smolvla_h10_{s}" for s in (
            "left_position_m", "right_position_m", "left_quaternion_wxyz", "right_quaternion_wxyz")]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise RuntimeError(f"Missing FK keys: {missing}; actual={z.files}")
        stored = {k: z[k].astype(np.float64) for k in required}
    if raw.ndim != 2 or raw.shape[1] != 14 or not np.isfinite(raw).all():
        raise RuntimeError(f"Invalid raw source: {raw.shape}")
    return {"raw": raw, "timestamp": timestamp, **stored}


def source_fk(args: argparse.Namespace, inp: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    model, _ = aloha_validate.load_validated_model(args.mujoco_xml)
    qpos, _ = aloha_validate.mapped_qpos(inp["raw"])
    generated = aloha_validate.fk(model, qpos)
    keys = {
        "left_position_m": "smolvla_h10_left_position_m",
        "right_position_m": "smolvla_h10_right_position_m",
        "left_quaternion_wxyz": "smolvla_h10_left_quaternion_wxyz",
        "right_quaternion_wxyz": "smolvla_h10_right_quaternion_wxyz",
    }
    max_diff = max(float(np.max(np.abs(generated[k] - inp[v]))) for k, v in keys.items())
    return generated, max_diff


def task_targets(info: dict[str, Any], fk: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    align = align_mod.make_align_rotation(ALIGN_RPY)
    dl = (fk["left_position_m"] - fk["left_position_m"][0]) @ align.T
    dr = (fk["right_position_m"] - fk["right_position_m"][0]) @ align.T
    lp, rp = ANCHOR_LEFT + scale * dl, ANCHOR_RIGHT + scale * dr
    model = info["model"]; data = mujoco.MjData(model)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    ik.assign_arm_qpos(data, info["stand_qpos"], info["arm_qpos_ids"], info["stand_arm_q"])
    mujoco.mj_forward(model, data)
    nominal_l, nominal_r = work.horizontal_inward_rotations(data.xmat[torso].reshape(3, 3))
    lrot, rrot = matrix_quat(fk["left_quaternion_wxyz"]), matrix_quat(fk["right_quaternion_wxyz"])
    mapped_l = map_relative(existing_filtered_relative(lrot), C_LEFT)
    mapped_r = map_relative(existing_filtered_relative(rrot), C_RIGHT)
    target_l = np.einsum("ij,tjk->tik", nominal_l, mapped_l)
    target_r = np.einsum("ij,tjk->tik", nominal_r, mapped_r)
    return {"lp": lp, "rp": rp, "target_l": target_l, "target_r": target_r,
            "aloha_lrot": lrot, "aloha_rrot": rrot, "mapped_l": mapped_l, "mapped_r": mapped_r}


def solve_task(info: dict[str, Any], targets: dict[str, np.ndarray], task_ready: np.ndarray) -> dict[str, np.ndarray]:
    n = len(targets["lp"])
    nominal_l = ik.mat_to_quat_wxyz(targets["target_l"][0])
    nominal_r = ik.mat_to_quat_wxyz(targets["target_r"][0])
    baseline = work.trajectory(info, targets["lp"], targets["rp"], nominal_l, nominal_r,
                               task_ready, task_ready, NOMINAL_WEIGHT, None, 120)
    model = info["model"]; data = mujoco.MjData(model)
    q = np.zeros((n, 14)); sl = np.zeros((n, 3)); sr = np.zeros((n, 3))
    lrot = np.zeros((n, 3, 3)); rrot = np.zeros((n, 3, 3))
    le = np.zeros(n); re = np.zeros(n); lo = np.zeros(n); ro = np.zeros(n)
    q[0] = baseline["q"][0]
    state = work.state(info, data, q[0])
    sl[0], sr[0] = state["left_pos"], state["right_pos"]
    lrot[0] = Rotation.from_quat(state["left_quat"][[1, 2, 3, 0]]).as_matrix()
    rrot[0] = Rotation.from_quat(state["right_quat"][[1, 2, 3, 0]]).as_matrix()
    le[0], re[0] = np.linalg.norm(sl[0]-targets["lp"][0]), np.linalg.norm(sr[0]-targets["rp"][0])
    lo[0] = rotation_error(lrot[:1], targets["target_l"][:1])[0]
    ro[0] = rotation_error(rrot[:1], targets["target_r"][:1])[0]
    prev = q[0].copy()
    for t in range(1, n):
        result = work.solve_frame(
            info, data, targets["lp"][t], targets["rp"][t],
            ik.mat_to_quat_wxyz(targets["target_l"][t]), ik.mat_to_quat_wxyz(targets["target_r"][t]),
            baseline["q"][t], prev, baseline["q"][0], ORIENTATION_WEIGHT, 120)
        q[t] = result["q"]; sl[t], sr[t] = result["s"]["left_pos"], result["s"]["right_pos"]
        lrot[t] = Rotation.from_quat(result["s"]["left_quat"][[1, 2, 3, 0]]).as_matrix()
        rrot[t] = Rotation.from_quat(result["s"]["right_quat"][[1, 2, 3, 0]]).as_matrix()
        le[t], re[t], lo[t], ro[t] = result["le"], result["re"], result["lo"], result["ro"]
        prev = q[t].copy()
        if t % 150 == 0:
            print(f"IK {t}/{n-1}", flush=True)
    actual_rel_l, actual_rel_r = relative_rotation(lrot), relative_rotation(rrot)
    return {"q": q, "sl": sl, "sr": sr, "lrot": lrot, "rrot": rrot, "le": le, "re": re,
            "lo": lo, "ro": ro, "relerr_l": rotation_error(actual_rel_l, targets["mapped_l"]),
            "relerr_r": rotation_error(actual_rel_r, targets["mapped_r"]), "baseline": baseline["q"]}


def dex3_task(info: dict[str, Any], raw: np.ndarray) -> dict[str, Any]:
    model, hands, arm_addr, actuator_names, actuator_joints = model_layout(G1_MODEL, info["joint_names"])
    hand_poses = poses(model, hands)
    # Exact established mapping: ALOHA larger width=open; closure=(.044-width)/.044.
    closure_l = np.clip((.044 - raw[:, 6]) / .044, 0, 1)
    closure_r = np.clip((.044 - raw[:, 13]) / .044, 0, 1)
    left, left_ratio = hand_trajectory(closure_l, *hand_poses["left"])
    right, right_ratio = hand_trajectory(closure_r, *hand_poses["right"])
    return {"model": model, "hands": hands, "arm_addr": arm_addr, "actuator_names": actuator_names,
            "actuator_joints": actuator_joints, "left": left, "right": right,
            "left_ratio": left_ratio, "right_ratio": right_ratio, "poses": hand_poses}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inp = load_inputs(args)
    fk, fk_diff = source_fk(args, inp)
    if fk_diff > 2e-6:
        raise RuntimeError(f"Regenerated MuJoCo FK differs from validated NPZ: {fk_diff}")
    info = ik.validate_model(G1_MODEL)
    with np.load(REFERENCE, allow_pickle=False) as z:
        task_ready = z["task_arm"][0].astype(np.float64)
    targets = task_targets(info, fk, args.position_scale)
    if args.max_frames:
        n = min(args.max_frames, len(inp["raw"]))
        inp = {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(fk["left_position_m"]) else v)
               for k, v in inp.items()}
        fk = {k: v[:n] for k, v in fk.items()}
        targets = {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim > 0
                        and len(v) == len(targets["lp"]) else v) for k, v in targets.items()}
    solved = solve_task(info, targets, task_ready)
    gd = dex3_task(info, inp["raw"])

    approach_arm = approach_mod.minimum_jerk(info["stand_arm_q"], solved["q"][0], APPROACH_FRAMES)
    approach_left = approach_mod.minimum_jerk(gd["poses"]["left"][0], gd["left"][0], APPROACH_FRAMES)
    approach_right = approach_mod.minimum_jerk(gd["poses"]["right"][0], gd["right"][0], APPROACH_FRAMES)
    hold_arm = np.repeat(solved["q"][0][None], HOLD_FRAMES, axis=0)
    hold_left = np.repeat(gd["left"][0][None], HOLD_FRAMES, axis=0)
    hold_right = np.repeat(gd["right"][0][None], HOLD_FRAMES, axis=0)
    full_arm = np.vstack((approach_arm, hold_arm, solved["q"]))
    full_left = np.vstack((approach_left, hold_left, gd["left"]))
    full_right = np.vstack((approach_right, hold_right, gd["right"]))
    task_start = APPROACH_FRAMES + HOLD_FRAMES
    base = gd["model"].key_qpos[0].copy()
    full_q = full_qpos(gd["model"], base, full_arm, full_left, full_right, gd["arm_addr"], gd["hands"])

    limits = info["joint_limits"]
    violations = (full_arm < limits[:, 0]-1e-9) | (full_arm > limits[:, 1]+1e-9)
    collision, cross, pairs = work.contacts(info, full_arm)
    pos_success = (solved["le"] <= .005) & (solved["re"] <= .005)
    step = np.abs(np.diff(full_arm, axis=0)); velocity = step * FPS
    acceleration = np.abs(np.diff(full_arm, n=2, axis=0)) * FPS**2
    norms = np.linalg.norm(np.diff(solved["q"], axis=0), axis=1)
    branch = np.zeros(len(solved["q"]), bool)
    for t in range(1, len(solved["q"])):
        local = np.median(norms[max(0, t-10):min(len(norms), t+9)])
        branch[t] = norms[t-1] > max(.15, 8*max(local, 1e-5))
    rel_target = targets["rp"] - targets["lp"]; rel_solved = solved["sr"] - solved["sl"]
    rel_rmse = float(np.sqrt(np.mean(np.sum((rel_solved-rel_target)**2, axis=1))))
    source_rel = fk["right_position_m"] - fk["left_position_m"]
    mapped_source_rel = (source_rel-source_rel[0]) @ align_mod.make_align_rotation(ALIGN_RPY).T * args.position_scale
    solved_rel_delta = rel_solved-rel_solved[0]
    rel_corr = float(np.corrcoef(mapped_source_rel.reshape(-1), solved_rel_delta.reshape(-1))[0, 1])
    angle = lambda x: np.linalg.norm(Rotation.from_matrix(x).as_rotvec(), axis=1)
    corr_l = float(np.corrcoef(angle(targets["mapped_l"]), angle(relative_rotation(solved["lrot"])))[0, 1])
    corr_r = float(np.corrcoef(angle(targets["mapped_r"]), angle(relative_rotation(solved["rrot"])))[0, 1])
    finite = all(np.isfinite(x).all() for x in (full_arm, full_left, full_right, full_q))
    boundary_arm = float(np.max(np.abs(hold_arm[-1]-solved["q"][0])))
    boundary_hand = float(max(np.max(np.abs(hold_left[-1]-gd["left"][0])),
                              np.max(np.abs(hold_right[-1]-gd["right"][0]))))
    safety_pass = bool(pos_success.all() and not violations.any() and finite and not branch.any()
                       and boundary_arm <= 1e-12 and boundary_hand <= 1e-12)
    report = {
        "evaluation_label": "training-set teacher-forced sanity evaluation",
        "source_episode": 49, "source_checkpoint": str(CHECKPOINT),
        "source_file": str(args.aloha_trajectory_npz), "source_trajectory_key": "raw (chunk_stitched_h10 export)",
        "source_frames": len(inp["raw"]), "mujoco_fk_regeneration_max_abs_difference": fk_diff,
        "existing_pipeline": ["validate_smolvla_in_stationary_aloha_mujoco.py",
                              "generate_aloha_work_posture_dex3.py", "generate_wrist_relative_full.py",
                              "generate_horizontal_approach_hold.py", "map_aloha_gripper_to_dex3.py"],
        "position_scale": args.position_scale, "align_rotation_rpy_deg": ALIGN_RPY.tolist(),
        "anchors_m": {"left": ANCHOR_LEFT.tolist(), "right": ANCHOR_RIGHT.tolist()},
        "orientation_mode": args.orientation_mode, "orientation_weight": ORIENTATION_WEIGHT,
        "nominal_regularization_weight": NOMINAL_WEIGHT,
        "C_left": C_LEFT.tolist(), "C_right": C_RIGHT.tolist(),
        "approach_frames": APPROACH_FRAMES, "hold_frames": HOLD_FRAMES,
        "task_start_frame": task_start, "full_frames": len(full_arm),
        "ik_success_rate": float(pos_success.mean()),
        "left_position_error_mm": {k: v*1000 for k, v in metrics(solved["le"]).items()},
        "right_position_error_mm": {k: v*1000 for k, v in metrics(solved["re"]).items()},
        "left_relative_orientation_error_deg": {k: float(np.degrees(v)) for k, v in metrics(solved["relerr_l"]).items()},
        "right_relative_orientation_error_deg": {k: float(np.degrees(v)) for k, v in metrics(solved["relerr_r"]).items()},
        "joint_limit_violation_count": int(violations.sum()), "branch_discontinuity_count": int(branch.sum()),
        "nan_inf_count": 0 if finite else 1, "max_frame_joint_step": float(step.max(initial=0)),
        "max_joint_velocity_rad_s": float(velocity.max(initial=0)),
        "max_joint_acceleration_rad_s2": float(acceleration.max(initial=0)),
        "self_collision_check": "AVAILABLE_MUJOCO", "self_collision_frames": int(collision.sum()),
        "cross_arm_collision_frames": int(cross.sum()), "collision_pairs": pairs.tolist(),
        "relative_position_rmse_mm": rel_rmse*1000, "relative_motion_correlation": rel_corr,
        "relative_orientation_angle_correlation": {"left": corr_l, "right": corr_r},
        "gripper_mapping": "existing Dex3 closure curves; closure=clip((0.044-source)/0.044,0,1)",
        "gripper_transition_frame": {
            "left": np.flatnonzero(np.diff(gd["left_ratio"]) != 0).astype(int).tolist(),
            "right": np.flatnonzero(np.diff(gd["right_ratio"]) != 0).astype(int).tolist()},
        "approach_task_boundary_jump": {"arm_rad": boundary_arm, "hand_rad": boundary_hand},
        "safety_pass_for_isaac": safety_pass,
    }
    payload = dict(
        source_episode=np.asarray(49), source_checkpoint=np.asarray(str(CHECKPOINT)),
        source_trajectory_key=np.asarray("chunk_stitched_h10"), source_aloha_action=inp["raw"].astype(np.float32),
        aloha_left_position=fk["left_position_m"], aloha_right_position=fk["right_position_m"],
        aloha_left_orientation=matrix_quat(fk["left_quaternion_wxyz"]),
        aloha_right_orientation=matrix_quat(fk["right_quaternion_wxyz"]),
        g1_target_left_position=targets["lp"], g1_target_right_position=targets["rp"],
        g1_target_left_orientation=targets["target_l"], g1_target_right_orientation=targets["target_r"],
        g1_arm_joint_trajectory=solved["q"], g1_hand_commands=np.stack((gd["left_ratio"], gd["right_ratio"]), axis=1),
        full_g1_joint_trajectory=full_q, ik_success=pos_success,
        position_error_left=solved["le"], position_error_right=solved["re"],
        orientation_error_left=solved["relerr_l"], orientation_error_right=solved["relerr_r"],
        fps=np.asarray(FPS), task_start_frame=np.asarray(task_start),
        arm_joint_names=info["joint_names"], dex3_left_joint_names=np.asarray(gd["hands"]["left"]["names"]),
        dex3_right_joint_names=np.asarray(gd["hands"]["right"]["names"]),
        approach_arm=approach_arm, hold_arm=hold_arm, task_arm=solved["q"], full_arm=full_arm,
        approach_left_dex3=approach_left, hold_left_dex3=hold_left, task_left_dex3=gd["left"], full_left_dex3=full_left,
        approach_right_dex3=approach_right, hold_right_dex3=hold_right, task_right_dex3=gd["right"], full_right_dex3=full_right,
        g1_target_achieved_left_position=solved["sl"], g1_target_achieved_right_position=solved["sr"],
        g1_target_achieved_left_orientation=solved["lrot"], g1_target_achieved_right_orientation=solved["rrot"],
        ik_branch_discontinuity=branch, joint_limit_violation=violations,
        self_collision_flag=collision, cross_arm_collision_flag=cross,
    )
    if args.execute and not args.dry_run:
        tmp = args.output_dir / "g1_smolvla_episode49_h10_trajectory.npz.incomplete"
        with tmp.open("wb") as f:
            np.savez_compressed(f, **payload)
        os.replace(tmp, args.output_dir / "g1_smolvla_episode49_h10_trajectory.npz")
    else:
        report["dry_run"] = True
    atomic_json(args.output_dir / "retargeting_report.json", report)
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    args = parse_args()
    report = run(args)
    if args.execute and not report["safety_pass_for_isaac"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
