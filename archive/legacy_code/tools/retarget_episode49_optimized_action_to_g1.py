#!/usr/bin/env python3
"""Retarget episode 49 optimized_action to G1 with trajectory-level temporal IK."""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pandas"] = types.ModuleType("pandas")

import generate_aloha_work_posture_dex3 as work  # noqa: E402
import generate_horizontal_approach_hold as approach_mod  # noqa: E402
import search_g1_task_ready_anchor as align_mod  # noqa: E402
import validate_g1_targets_and_sparse_ik as ik  # noqa: E402
import validate_smolvla_in_stationary_aloha_mujoco as aloha  # noqa: E402
from map_aloha_gripper_to_dex3 import full_qpos, hand_trajectory, model_layout, poses  # noqa: E402

SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
OUT = ROOT / "converted_runs/smolvla_20k_episode49_consensus_g1/g1_episode49_consensus_trajectory.npz"
REPORT = OUT.with_name("g1_episode49_consensus_report.json")
ALOHA_XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
G1_XML = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
REFERENCE = ROOT / "converted_runs/magsafe_20260723_162750/wrist_relative_rotation/g1_wrist_relative_full_trajectory.npz"
FPS = 30.0
SCALE = .42
ALIGN_RPY = np.array([0., -7., 0.])
ANCHOR_L = np.array([.20, .28, .94])
ANCHOR_R = np.array([.20, -.28, .94])
C_L = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
C_R = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
APPROACH, HOLD = 90, 15
LIMITS = {"step": .1035, "velocity": 3.1062, "acceleration": 79.2}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--temporal-iterations", type=int, default=8)
    return p.parse_args()


def relrot(r: np.ndarray) -> np.ndarray:
    return np.einsum("ji,tjk->tik", r[0], r)


def maprot(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.einsum("ij,tjk,kl->til", c.T, relrot(r), c)


def rot_from_wxyz(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def rotation_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Rotation.from_matrix(np.einsum("tji,tjk->tik", actual, target)).as_rotvec(), axis=1)


def stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x)
    return {k: float(v) for k, v in (
        ("mean", np.mean(x)), ("max", np.max(x, initial=0)),
        ("p95", np.percentile(x, 95)), ("p99", np.percentile(x, 99)))}


def load_source(path: Path, max_frames: int | None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        print("NPZ keys:", z.files)
        if "optimized_action" not in z.files:
            raise RuntimeError(f"Missing optimized_action; actual={z.files}")
        raw = z["optimized_action"].astype(np.float64)
        timestamp = z["timestamp"].astype(np.float64)
    if raw.shape != (990, 14) or not np.isfinite(raw).all():
        raise RuntimeError(f"optimized_action must be finite [990,14], got {raw.shape}")
    if max_frames:
        raw, timestamp = raw[:max_frames], timestamp[:max_frames]
    print(f"sole source trajectory: optimized_action {raw.shape}")
    return raw, timestamp


def targets(info: dict, fk: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    align = align_mod.make_align_rotation(ALIGN_RPY)
    lp = ANCHOR_L + SCALE * ((fk["left_position_m"] - fk["left_position_m"][0]) @ align.T)
    rp = ANCHOR_R + SCALE * ((fk["right_position_m"] - fk["right_position_m"][0]) @ align.T)
    data = mujoco.MjData(info["model"])
    torso = mujoco.mj_name2id(info["model"], mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    ik.assign_arm_qpos(data, info["stand_qpos"], info["arm_qpos_ids"], info["stand_arm_q"])
    mujoco.mj_forward(info["model"], data)
    nl, nr = work.horizontal_inward_rotations(data.xmat[torso].reshape(3, 3))
    ml = maprot(rot_from_wxyz(fk["left_quaternion_wxyz"]), C_L)
    mr = maprot(rot_from_wxyz(fk["right_quaternion_wxyz"]), C_R)
    return {"lp": lp, "rp": rp, "lr": np.einsum("ij,tjk->tik", nl, ml),
            "rr": np.einsum("ij,tjk->tik", nr, mr), "mapped_l": ml, "mapped_r": mr}


def frame_state(info: dict, data: mujoco.MjData, q: np.ndarray) -> dict:
    ik.assign_arm_qpos(data, info["stand_qpos"], info["arm_qpos_ids"], q)
    mujoco.mj_forward(info["model"], data)
    return ik.current_bimanual_state(info, data)


def position_seed(info: dict, tar: dict, nominal: np.ndarray) -> np.ndarray:
    """Position-only continuation seed; temporal solve below, never a final candidate."""
    n = len(tar["lp"])
    q = np.empty((n, 14))
    data = mujoco.MjData(info["model"])
    prev = nominal.copy()
    lq = ik.mat_to_quat_wxyz(tar["lr"][0])
    rq = ik.mat_to_quat_wxyz(tar["rr"][0])
    for t in range(n):
        r = work.solve_frame(info, data, tar["lp"][t], tar["rp"][t], lq, rq,
                             prev, prev, nominal, 0.0, 100)
        q[t], prev = r["q"], r["q"]
        if t % 150 == 0:
            print(f"position seed {t}/{n-1}", flush=True)
    return q


def add_block(rows, cols, vals, rhs, row0: int, col0: int, a: np.ndarray, b: np.ndarray) -> int:
    rr, cc = np.nonzero(a)
    rows.extend((row0 + rr).tolist()); cols.extend((col0 + cc).tolist()); vals.extend(a[rr, cc].tolist())
    rhs.extend(np.asarray(b).tolist())
    return row0 + len(b)


def temporal_solve(info: dict, tar: dict, initial: np.ndarray, nominal: np.ndarray,
                   ori_weight: float, iterations: int) -> np.ndarray:
    """Whole-trajectory sparse Gauss-Newton with Cartesian and temporal residuals."""
    q = initial.copy()
    n, d = q.shape
    lim = info["joint_limits"]
    data = mujoco.MjData(info["model"])
    # Residual weights: position, bimanual relative position, velocity,
    # acceleration and nominal pose. Orientation is introduced only in stage 2.
    wp, wr, wv, wa, wn = 3.0, .80, .018, .030, .001
    for iteration in range(iterations):
        rows: list[int] = []; cols: list[int] = []; vals: list[float] = []; rhs: list[float] = []
        row = 0
        for t in range(n):
            s = frame_state(info, data, q[t])
            jl = np.hstack((s["left_jac"][:3], np.zeros((3, 7))))
            jr = np.hstack((np.zeros((3, 7)), s["right_jac"][:3]))
            row = add_block(rows, cols, vals, rhs, row, t*d, wp*jl, wp*(tar["lp"][t]-s["left_pos"]))
            row = add_block(rows, cols, vals, rhs, row, t*d, wp*jr, wp*(tar["rp"][t]-s["right_pos"]))
            jrel = jr - jl
            erel = (tar["rp"][t]-tar["lp"][t]) - (s["right_pos"]-s["left_pos"])
            row = add_block(rows, cols, vals, rhs, row, t*d, wr*jrel, wr*erel)
            if ori_weight:
                _, jo, _, _, _, _ = ik.build_error_and_jacobian(
                    s, tar["lp"][t], ik.mat_to_quat_wxyz(tar["lr"][t]),
                    tar["rp"][t], ik.mat_to_quat_wxyz(tar["rr"][t]), 0.0, ori_weight)
                eo, _, _, _, _, _ = ik.build_error_and_jacobian(
                    s, tar["lp"][t], ik.mat_to_quat_wxyz(tar["lr"][t]),
                    tar["rp"][t], ik.mat_to_quat_wxyz(tar["rr"][t]), 0.0, ori_weight)
                row = add_block(rows, cols, vals, rhs, row, t*d, jo[[3,4,5,9,10,11]], eo[[3,4,5,9,10,11]])
            row = add_block(rows, cols, vals, rhs, row, t*d, wn*np.eye(d), wn*(nominal-q[t]))
        eye = np.eye(d)
        for t in range(1, n):
            row = add_block(rows, cols, vals, rhs, row, (t-1)*d, np.hstack((-wv*eye, wv*eye)), -wv*(q[t]-q[t-1]))
        for t in range(1, n-1):
            row = add_block(rows, cols, vals, rhs, row, (t-1)*d,
                            np.hstack((wa*eye, -2*wa*eye, wa*eye)), -wa*(q[t+1]-2*q[t]+q[t-1]))
        A = coo_matrix((vals, (rows, cols)), shape=(row, n*d)).tocsr()
        dq = lsqr(A, np.asarray(rhs), damp=2e-3, atol=2e-5, btol=2e-5, iter_lim=180)[0].reshape(n, d)
        # Trust region and bound projection are part of optimization, not output post-processing.
        alpha = .45 if iteration < 2 else .7
        q = np.minimum(np.maximum(q + alpha*np.clip(dq, -.035, .035), lim[:, 0]), lim[:, 1])
        print(f"temporal IK orientation={ori_weight:.4g} iteration {iteration+1}/{iterations} "
              f"max_update={np.max(np.abs(alpha*dq)):.6f}", flush=True)
    return q


def evaluate(info: dict, tar: dict, q: np.ndarray) -> dict[str, np.ndarray]:
    n = len(q); data = mujoco.MjData(info["model"])
    lp = np.empty((n, 3)); rp = np.empty((n, 3)); lr = np.empty((n, 3, 3)); rr = np.empty((n, 3, 3))
    for t in range(n):
        s = frame_state(info, data, q[t])
        lp[t], rp[t] = s["left_pos"], s["right_pos"]
        lr[t] = Rotation.from_quat(s["left_quat"][[1,2,3,0]]).as_matrix()
        rr[t] = Rotation.from_quat(s["right_quat"][[1,2,3,0]]).as_matrix()
    return {"lp": lp, "rp": rp, "lr": lr, "rr": rr,
            "le": np.linalg.norm(lp-tar["lp"], axis=1), "re": np.linalg.norm(rp-tar["rp"], axis=1),
            "lo": rotation_error(lr, tar["lr"]), "ro": rotation_error(rr, tar["rr"]),
            "rel": np.linalg.norm((rp-lp)-(tar["rp"]-tar["lp"]), axis=1)}


def hands(info: dict, raw: np.ndarray) -> dict:
    model, hs, addr, an, aj = model_layout(G1_XML, info["joint_names"])
    ps = poses(model, hs)
    lc = np.clip((.044-raw[:, 6])/.044, 0, 1); rc = np.clip((.044-raw[:, 13])/.044, 0, 1)
    left, lratio = hand_trajectory(lc, *ps["left"]); right, rratio = hand_trajectory(rc, *ps["right"])
    return {"model": model, "hands": hs, "addr": addr, "poses": ps, "left": left, "right": right,
            "lratio": lratio, "rratio": rratio, "actuator_names": an, "actuator_joints": aj}


def main() -> int:
    a = args()
    for p in (a.source, ALOHA_XML, G1_XML, REFERENCE):
        if not p.exists(): raise FileNotFoundError(p)
    if a.output.exists(): raise FileExistsError(a.output)
    raw, timestamp = load_source(a.source, a.max_frames)
    amodel, _ = aloha.load_validated_model(ALOHA_XML)
    aqpos, clipped = aloha.mapped_qpos(raw)
    fk = aloha.fk(amodel, aqpos)
    info = ik.validate_model(G1_XML)
    with np.load(REFERENCE, allow_pickle=False) as z:
        nominal = z["task_arm"][0].astype(np.float64)
    tar = targets(info, fk)
    seed = position_seed(info, tar, nominal)
    position_q = temporal_solve(info, tar, seed, nominal, 0.0, a.temporal_iterations)
    position_ev = evaluate(info, tar, position_q)
    candidates = [(0.0, position_q.copy(), position_ev,
                   np.max(np.abs(np.diff(position_q, axis=0)), initial=0),
                   np.max(np.abs(np.diff(position_q, n=2, axis=0))*FPS**2, initial=0))]
    for ow in (.0015, .003, .005, .0075):
        current = temporal_solve(info, tar, position_q, nominal, ow, max(3, a.temporal_iterations//2))
        ev = evaluate(info, tar, current)
        step = np.max(np.abs(np.diff(current, axis=0)), initial=0)
        acc = np.max(np.abs(np.diff(current, n=2, axis=0))*FPS**2, initial=0)
        candidates.append((ow, current.copy(), ev, step, acc))
    # Partial orientation is preferred once continuity degrades or limits are exceeded.
    viable = [x for x in candidates if x[3] <= LIMITS["step"] and x[4] <= LIMITS["acceleration"]
              and max(x[2]["le"].max(), x[2]["re"].max()) <= .005]
    chosen = min(viable, key=lambda x: np.percentile(np.r_[x[2]["lo"], x[2]["ro"]], 95)) if viable else candidates[0]
    ow, task_q, ev, _, _ = chosen
    gd = hands(info, raw)
    app_arm = approach_mod.minimum_jerk(info["stand_arm_q"], task_q[0], APPROACH)
    app_l = approach_mod.minimum_jerk(gd["poses"]["left"][0], gd["left"][0], APPROACH)
    app_r = approach_mod.minimum_jerk(gd["poses"]["right"][0], gd["right"][0], APPROACH)
    hold_arm = np.repeat(task_q[:1], HOLD, axis=0)
    hold_l = np.repeat(gd["left"][:1], HOLD, axis=0); hold_r = np.repeat(gd["right"][:1], HOLD, axis=0)
    full_arm = np.vstack((app_arm, hold_arm, task_q))
    full_l = np.vstack((app_l, hold_l, gd["left"])); full_r = np.vstack((app_r, hold_r, gd["right"]))
    full_q = full_qpos(gd["model"], gd["model"].key_qpos[0].copy(), full_arm, full_l, full_r, gd["addr"], gd["hands"])
    lim = info["joint_limits"]; violations = (full_arm < lim[:,0]-1e-9) | (full_arm > lim[:,1]+1e-9)
    collision, cross, pairs = work.contacts(info, full_arm)
    step = np.abs(np.diff(full_arm, axis=0)); vel = step*FPS; acc = np.abs(np.diff(full_arm,n=2,axis=0))*FPS**2
    norms = np.linalg.norm(np.diff(task_q,axis=0),axis=1); branch = np.zeros(len(task_q),bool)
    for t in range(1,len(task_q)):
        local=np.median(norms[max(0,t-10):min(len(norms),t+9)])
        branch[t]=norms[t-1] > max(.15,8*max(local,1e-5))
    failed = (ev["le"] > .005) | (ev["re"] > .005)
    finite = all(np.isfinite(x).all() for x in (task_q,full_q,ev["lp"],ev["rp"]))
    boundary = float(np.max(np.abs(hold_arm[-1]-task_q[0])))
    safety = bool(not failed.any() and not branch.any() and not violations.any() and
                  not collision.any() and finite and step.max(initial=0)<=LIMITS["step"] and
                  vel.max(initial=0)<=LIMITS["velocity"] and acc.max(initial=0)<=LIMITS["acceleration"] and boundary<=1e-12)
    report = {
        "source_file": str(a.source), "source_trajectory_key": "optimized_action", "source_shape": list(raw.shape),
        "forbidden_trajectory_keys_used": [], "aloha_mapping_clipped_frames": clipped,
        "solver": "whole-trajectory sparse Gauss-Newton", "position_only_initialization": True,
        "orientation_candidate_weights": [0.0,.0015,.003,.005,.0075], "selected_orientation_weight": ow,
        "position_scale": SCALE, "align_rotation_rpy_deg": ALIGN_RPY.tolist(),
        "anchors_m": {"left":ANCHOR_L.tolist(),"right":ANCHOR_R.tolist()},
        "approach_frames": APPROACH, "hold_frames": HOLD, "task_start_frame": APPROACH+HOLD,
        "ik_success": bool(not failed.any()), "failed_frames": np.flatnonzero(failed).tolist(),
        "branch_discontinuity_count": int(branch.sum()), "joint_limit_violation_count": int(violations.sum()),
        "self_collision_frames": int(collision.sum()), "cross_arm_collision_frames": int(cross.sum()),
        "collision_pairs": pairs.tolist(), "nan_inf_count": 0 if finite else 1,
        "joint_step_rad": stats(step), "joint_velocity_rad_s": stats(vel), "joint_acceleration_rad_s2": stats(acc),
        "left_position_error_mm": {k:v*1000 for k,v in stats(ev["le"]).items()},
        "right_position_error_mm": {k:v*1000 for k,v in stats(ev["re"]).items()},
        "left_orientation_error_deg": {k:float(np.degrees(v)) for k,v in stats(ev["lo"]).items()},
        "right_orientation_error_deg": {k:float(np.degrees(v)) for k,v in stats(ev["ro"]).items()},
        "bimanual_relative_position_error_mm": {k:v*1000 for k,v in stats(ev["rel"]).items()},
        "approach_task_boundary_jump_rad": boundary, "safety_thresholds": LIMITS,
        "safety_pass": safety, "verdict": "G1_CONSENSUS_READY_FOR_ISAACLAB" if safety else "G1_CONSENSUS_SAFETY_BLOCKED",
    }
    payload = dict(source_trajectory_key=np.asarray("optimized_action"), source_optimized_action=raw.astype(np.float32),
        source_timestamp=timestamp, aloha_qpos=aqpos, aloha_left_position=fk["left_position_m"],
        aloha_right_position=fk["right_position_m"], g1_target_left_position=tar["lp"],
        g1_target_right_position=tar["rp"], g1_target_left_orientation=tar["lr"],
        g1_target_right_orientation=tar["rr"], position_only_initial_arm=position_q,
        task_arm=task_q, full_arm=full_arm, full_left_dex3=full_l, full_right_dex3=full_r,
        full_g1_joint_trajectory=full_q, g1_target_achieved_left_position=ev["lp"],
        g1_target_achieved_right_position=ev["rp"], g1_target_achieved_left_orientation=ev["lr"],
        g1_target_achieved_right_orientation=ev["rr"], position_error_left=ev["le"],
        position_error_right=ev["re"], orientation_error_left=ev["lo"], orientation_error_right=ev["ro"],
        bimanual_relative_position_error=ev["rel"], ik_success=~failed, ik_branch_discontinuity=branch,
        joint_limit_violation=violations, self_collision_flag=collision, cross_arm_collision_flag=cross,
        arm_joint_names=info["joint_names"], fps=np.asarray(FPS), task_start_frame=np.asarray(APPROACH+HOLD))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.execute:
        tmp=a.output.with_suffix(a.output.suffix+".incomplete")
        with tmp.open("wb") as f: np.savez_compressed(f,**payload)
        os.replace(tmp,a.output)
    else:
        report["dry_run"]=True
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    tmpj=REPORT.with_suffix(".json.incomplete"); tmpj.write_text(json.dumps(report,indent=2)); os.replace(tmpj,REPORT)
    print(json.dumps(report,indent=2))
    return 0 if safety else 2


if __name__ == "__main__":
    raise SystemExit(main())
