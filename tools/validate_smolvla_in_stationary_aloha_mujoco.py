#!/usr/bin/env python3
"""Read-only Stationary ALOHA MuJoCo validation for dataset and SmolVLA trajectories."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import mujoco
import numpy as np
import pandas as pd

from replay_stationary_gopark import (
    EXPECTED_ACTUATORS,
    EXPECTED_BODIES,
    EXPECTED_QPOS_JOINTS,
    TCP_OFFSET_LOCAL,
    load_validated_model,
    map_row_to_qpos,
    validate_model_or_raise,
)

LOG = logging.getLogger("mujoco_aloha_validation")
DEFAULT_ROOT = Path("/home/jbnu/aloha_g1_dataset/lerobot_magsafe_50_cam_high_v3")
DEFAULT_PRED = Path("/home/jbnu/aloha_g1_dataset/evaluation/smolvla_20k_chunk_stitched_preflight")
DEFAULT_XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
DEFAULT_OUT = Path("/home/jbnu/aloha_g1_dataset/evaluation/mujoco_stationary_aloha_validation")
ISAAC_LIMITS = np.array([
    [-3.05433, 3.05433], [0, 3.14159], [0, 2.35619], [-1.57080007, 1.57080007],
    [-1.5708, 1.5708], [-3.14159, 3.14159], [0, .044],
    [-3.05433, 3.05433], [0, 3.14159], [0, 2.35619], [-1.57080007, 1.57080007],
    [-1.5708, 1.5708], [-3.14159, 3.14159], [0, .044],
], dtype=np.float64)
DATASET_NAMES = [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--repo-id", default="local/magsafe_aloha_50_cam_high_v3")
    p.add_argument("--prediction-dir", type=Path, default=DEFAULT_PRED)
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 24, 49])
    p.add_argument("--model-xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    tmp = path.with_suffix(path.suffix + ".incomplete")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def stack_column(series: pd.Series, name: str) -> np.ndarray:
    x = np.stack([np.asarray(v, dtype=np.float64) for v in series], axis=0)
    if x.ndim != 2 or x.shape[1] != 14 or not np.isfinite(x).all():
        raise RuntimeError(f"{name} must be finite [T,14], got {x.shape}")
    return x


def load_episode(root: Path, episode: int) -> dict[str, np.ndarray]:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards under {root / 'data'}")
    columns = ["observation.state", "action", "timestamp", "frame_index", "episode_index"]
    parts = []
    for path in files:
        shard = pd.read_parquet(path, columns=columns)
        selected = shard[shard["episode_index"] == episode]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise RuntimeError(f"Episode {episode} not found in {len(files)} parquet shards")
    df = pd.concat(parts, ignore_index=True).sort_values("frame_index")
    return {
        "state": stack_column(df["observation.state"], "observation.state"),
        "action": stack_column(df["action"], "action"),
        "timestamp": np.asarray(df["timestamp"], dtype=np.float64),
        "frame_index": np.asarray(df["frame_index"], dtype=np.int64),
    }


def dataset_limits(model: mujoco.MjModel) -> np.ndarray:
    limits = np.zeros((14, 2), dtype=np.float64)
    joint_names = EXPECTED_QPOS_JOINTS[:6] + [EXPECTED_QPOS_JOINTS[6]] + EXPECTED_QPOS_JOINTS[8:14] + [EXPECTED_QPOS_JOINTS[14]]
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        limits[i] = model.jnt_range[jid]
    return limits


def mapped_qpos(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Use the existing replay mapping verbatim; raw remains untouched."""
    out = np.empty((len(raw), 16), dtype=np.float64)
    clip_frames = 0
    for i, row in enumerate(raw):
        q, lc, rc = map_row_to_qpos(row)
        out[i] = q
        clip_frames += int(lc or rc)
    return out, clip_frames


def raw_limit_counts(raw: np.ndarray, limits: np.ndarray) -> tuple[int, list[int]]:
    mask = (raw < limits[:, 0]) | (raw > limits[:, 1])
    return int(mask.sum()), mask.sum(axis=0).astype(int).tolist()


def mapped_limit_counts(qpos: np.ndarray, model: mujoco.MjModel) -> tuple[int, list[int]]:
    counts = []
    for i, name in enumerate(EXPECTED_QPOS_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        counts.append(int(((qpos[:, i] < lo) | (qpos[:, i] > hi)).sum()))
    return sum(counts), counts


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(a * b, axis=1))
    return np.degrees(2 * np.arccos(np.clip(dots, -1, 1)))


def fk(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in EXPECTED_BODIES]
    lp, rp = np.empty((len(qpos), 3)), np.empty((len(qpos), 3))
    lq, rq = np.empty((len(qpos), 4)), np.empty((len(qpos), 4))
    for i, q in enumerate(qpos):
        data.qpos[:] = 0
        data.qpos[:16] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        for bid, pos, quat in ((ids[0], lp, lq), (ids[1], rp, rq)):
            rot = np.array(data.xmat[bid]).reshape(3, 3)
            pos[i] = data.xpos[bid] + rot @ TCP_OFFSET_LOCAL
            mujoco.mju_mat2Quat(quat[i], rot.reshape(9))
    return {"left_position_m": lp, "right_position_m": rp, "left_quaternion_wxyz": lq,
            "right_quaternion_wxyz": rq, "relative_position_m": rp - lp}


def trajectory_metrics(raw: np.ndarray, qpos: np.ndarray, limits: np.ndarray, model: mujoco.MjModel,
                       initial: np.ndarray) -> dict[str, Any]:
    d = np.diff(raw, axis=0)
    absd = np.abs(d)
    vel = absd * 30
    acc = np.abs(np.diff(raw, n=2, axis=0)) * 900
    rv, per = raw_limit_counts(raw, limits)
    mv, mper = mapped_limit_counts(qpos, model)
    return {
        "frames": len(raw), "finite": bool(np.isfinite(raw).all()), "nan_inf_count": int((~np.isfinite(raw)).sum()),
        "raw_joint_limit_violations": rv, "raw_per_joint_violations": per,
        "mapped_joint_limit_violations": mv, "mapped_per_qpos_violations": mper,
        "max_joint_jump": float(absd.max(initial=0)), "p99_joint_jump": float(np.percentile(absd, 99)) if len(d) else 0,
        "max_velocity_30hz": float(vel.max(initial=0)), "p99_velocity_30hz": float(np.percentile(vel, 99)) if len(d) else 0,
        "max_acceleration_30hz": float(acc.max(initial=0)),
        "first_target_vs_state0_max_abs": float(np.abs(raw[0] - initial).max()),
        "left_gripper_min": float(raw[:, 6].min()), "left_gripper_max": float(raw[:, 6].max()),
        "right_gripper_min": float(raw[:, 13].min()), "right_gripper_max": float(raw[:, 13].max()),
        "gripper_transition_count": int(np.count_nonzero(np.diff(raw[:, 6])) + np.count_nonzero(np.diff(raw[:, 13]))),
    }


def lag_analysis(episodes: list[dict[str, np.ndarray]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lag in range(11):
        aa, ss = [], []
        for ep in episodes:
            aa.append(ep["action"][:len(ep["action"]) - lag or None])
            ss.append(ep["state"][lag:])
        a, s = np.concatenate(aa), np.concatenate(ss)
        diff = a - s
        for j in range(15):
            x = a.reshape(-1) if j == 14 else a[:, j]
            y = s.reshape(-1) if j == 14 else s[:, j]
            dd = x - y
            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else float("nan")
            rows.append({"lag": lag, "joint_index": "overall" if j == 14 else j,
                         "joint_name": "overall" if j == 14 else DATASET_NAMES[j],
                         "rmse": float(np.sqrt(np.mean(dd * dd))), "mae": float(np.mean(np.abs(dd))),
                         "correlation": corr, "median_offset_action_minus_state": float(np.median(dd)),
                         "sign_consistency": float(np.mean(np.sign(x) == np.sign(y))),
                         "action_min": float(x.min()), "action_max": float(x.max()),
                         "state_min": float(y.min()), "state_max": float(y.max())})
    overall = [r for r in rows if r["joint_index"] == "overall"]
    best = min(overall, key=lambda r: r["rmse"])
    for r in rows:
        same_joint = [x for x in rows if x["joint_index"] == r["joint_index"]]
        r["optimal_lag"] = min(same_joint, key=lambda x: x["rmse"])["lag"]
    semantics = {
        "optimal_lag_frames": int(best["lag"]), "optimal_overall_rmse": best["rmse"],
        "optimal_overall_mae": best["mae"], "optimal_overall_correlation": best["correlation"],
        "interpretation_rule": "Direct follower joint target only when values closely match a future observed follower state without fitted scale/offset.",
        "direct_joint_target_supported": bool(best["correlation"] > .99 and best["rmse"] < .02),
    }
    return rows, semantics


def dynamic_replay(model: mujoco.MjModel, initial_qpos: np.ndarray,
                   targets: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    data.qpos[:16] = initial_qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    substeps = max(1, int(round((1 / 30) / dt)))
    actual = np.empty((len(targets), 14), dtype=np.float64)
    ctrl_qpos = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]
    for i, target in enumerate(targets):
        data.ctrl[:] = target
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        actual[i] = data.qpos[ctrl_qpos]
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f"Dynamic divergence at frame {i}")
    err = actual - targets
    return ({"performed": True, "physics_steps": len(targets) * substeps, "substeps_per_target": substeps,
             "tracking_rmse": float(np.sqrt(np.mean(err * err))), "tracking_max_error": float(np.abs(err).max())},
            actual)


def record_dynamic_video(path: Path, series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Record target/actual tracking diagnostics from real mj_step results."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = {"expert_state": "black", "expert_action": "tab:blue", "smolvla_h10": "tab:orange"}
    lines: dict[str, tuple[Any, Any]] = {}
    for name, (target, actual) in series.items():
        lines[name] = (axes[0].plot([], [], color=colors[name], label=f"{name} target j3")[0],
                       axes[0].plot([], [], color=colors[name], ls="--", label=f"{name} actual j3")[0])
    axes[0].legend(fontsize=7, ncol=2); axes[0].set_ylabel("joint 3")
    axes[1].set_ylabel("max |target-actual|"); axes[1].set_xlabel("frame")
    max_len = max(len(v[0]) for v in series.values())
    axes[0].set_xlim(0, max_len); axes[0].set_ylim(-1.7, 1.7)
    axes[1].set_xlim(0, max_len)
    max_err = max(float(np.abs(a - t).max()) for t, a in series.values())
    axes[1].set_ylim(0, max(max_err * 1.05, .01))
    err_lines = {name: axes[1].plot([], [], color=colors[name], label=name)[0] for name in series}
    axes[1].legend(fontsize=7)
    writer = FFMpegWriter(fps=10, bitrate=1600)
    stride = 3
    with writer.saving(fig, str(path), dpi=110):
        for i in range(0, max_len, stride):
            for name, (target, actual) in series.items():
                n = min(i + 1, len(target)); x = np.arange(n)
                lines[name][0].set_data(x, target[:n, 3]); lines[name][1].set_data(x, actual[:n, 3])
                err_lines[name].set_data(x, np.abs(actual[:n] - target[:n]).max(axis=1))
            axes[0].set_title(f"MuJoCo dynamic position-target replay — frame {i}")
            writer.grab_frame()
    plt.close(fig)


def plot_episode(out: Path, state: np.ndarray, action: np.ndarray, pred: np.ndarray,
                 fks: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(7, 2, figsize=(15, 17), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(state[:, j], label="state", lw=.8)
        ax.plot(action[:, j], label="action", lw=.7)
        ax.plot(pred[:, j], label="h10", lw=.7)
        ax.set_title(DATASET_NAMES[j])
    axes[0, 0].legend()
    fig.tight_layout(); fig.savefig(out / "joint_trajectories.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 4))
    for x, name in ((state, "state"), (action, "action"), (pred, "h10")):
        ax.plot(x[:, 3], label=name)
    ax.axhline(-1.5708, color="k", ls="--"); ax.axhline(1.5708, color="k", ls="--")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "joint3_detail.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for x, name in ((state, "state"), (action, "action"), (pred, "h10")):
        axes[0].plot(x[:, 6], label=name); axes[1].plot(x[:, 13], label=name)
    axes[0].legend(); fig.tight_layout(); fig.savefig(out / "grippers.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for name, fkdata in fks.items():
        axes[0].plot(fkdata["left_position_m"] * 1000, label=[f"{name}-x", f"{name}-y", f"{name}-z"])
        axes[1].plot(fkdata["right_position_m"] * 1000, label=[f"{name}-x", f"{name}-y", f"{name}-z"])
    axes[0].set_title("left TCP mm"); axes[1].set_title("right TCP mm")
    fig.tight_layout(); fig.savefig(out / "hand_xyz.png", dpi=130); plt.close(fig)


def record_kinematic_video(path: Path, fks: dict[str, dict[str, np.ndarray]]) -> None:
    """Record a headless FK diagnostic video; this does not step physics."""
    colors = {"expert_state": "black", "expert_action": "tab:blue", "smolvla_h10": "tab:orange"}
    all_pos = np.concatenate([np.concatenate([v["left_position_m"], v["right_position_m"]]) for v in fks.values()])
    lo, hi = all_pos.min(axis=0), all_pos.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) * .6, .05)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
           zlim=(center[2]-radius, center[2]+radius), xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    artists = {}
    for name, d in fks.items():
        artists[name] = (
            ax.plot([], [], [], color=colors[name], label=f"{name} left")[0],
            ax.plot([], [], [], color=colors[name], ls="--", label=f"{name} right")[0],
            ax.scatter([], [], [], color=colors[name], s=18),
        )
    ax.legend(fontsize=7)
    writer = FFMpegWriter(fps=10, bitrate=1800)
    stride = 3
    with writer.saving(fig, str(path), dpi=120):
        for i in range(0, len(next(iter(fks.values()))["left_position_m"]), stride):
            for name, d in fks.items():
                left, right, points = artists[name]
                lp, rp = d["left_position_m"], d["right_position_m"]
                left.set_data(lp[:i+1, 0], lp[:i+1, 1]); left.set_3d_properties(lp[:i+1, 2])
                right.set_data(rp[:i+1, 0], rp[:i+1, 1]); right.set_3d_properties(rp[:i+1, 2])
                points._offsets3d = ([lp[i, 0], rp[i, 0]], [lp[i, 1], rp[i, 1]], [lp[i, 2], rp[i, 2]])
            ax.set_title(f"Kinematic FK replay — frame {i}")
            writer.grab_frame()
    plt.close(fig)


def model_mapping(model: mujoco.MjModel, xml: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actuators, rows = [], []
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        jid = int(model.actuator_trnid[aid, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuators.append({"ctrl_index": aid, "actuator_name": name, "joint_name": jname,
                          "ctrl_range": model.actuator_ctrlrange[aid].tolist()})
    map_joints = EXPECTED_QPOS_JOINTS[:6] + [EXPECTED_QPOS_JOINTS[6]] + EXPECTED_QPOS_JOINTS[8:14] + [EXPECTED_QPOS_JOINTS[14]]
    for i, (dn, jn) in enumerate(zip(DATASET_NAMES, map_joints)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        rows.append({"dataset_index": i, "dataset_name": dn, "mujoco_primary_joint": jn,
                     "qpos_index": int(model.jnt_qposadr[jid]), "actuator": EXPECTED_ACTUATORS[i],
                     "ctrl_index": i, "lower": float(model.jnt_range[jid, 0]), "upper": float(model.jnt_range[jid, 1]),
                     "mapping_note": "duplicated to both carriage qpos" if i in (6, 13) else "direct"})
    return {
        "xml": str(xml), "nq": model.nq, "nv": model.nv, "nu": model.nu, "timestep": float(model.opt.timestep),
        "qpos_joints_first16": [{"qpos_index": i, "name": n} for i, n in enumerate(EXPECTED_QPOS_JOINTS)],
        "actuators": actuators, "fk_bodies": EXPECTED_BODIES, "tcp_offset_local_m": TCP_OFFSET_LOCAL.tolist(),
        "existing_replay_method": "qpos assignment then mj_forward", "action_source_supported": True,
        "position_actuator_mapping_validated": [x["actuator_name"] for x in actuators] == EXPECTED_ACTUATORS,
        "raw_preserved": True, "gripper_mapping": "existing replay clips simulation qpos to [0,0.044], duplicates to mirrored carriages",
    }, rows


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for p in (args.dataset_root, args.prediction_dir, args.model_xml):
        if not p.exists():
            raise FileNotFoundError(p)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.execute:
        raise RuntimeError(f"Output directory is non-empty; refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, xml = load_validated_model(args.model_xml)
    validate_model_or_raise(model)
    limits = dataset_limits(model)
    mapping, mapping_rows = model_mapping(model, xml)
    atomic_json(args.output_dir / "mujoco_model_mapping.json", mapping)
    atomic_csv(args.output_dir / "dataset_to_mujoco_joint_mapping.csv", mapping_rows)

    all_eps = [load_episode(args.dataset_root, i) for i in range(50)]
    lag_rows, semantics = lag_analysis(all_eps)
    av = sum(raw_limit_counts(ep["action"], limits)[0] for ep in all_eps)
    sv = sum(raw_limit_counts(ep["state"], limits)[0] for ep in all_eps)
    semantics.update({"episodes": 50, "frames": int(sum(len(x["state"]) for x in all_eps)),
                      "action_mujoco_limit_violations": av, "state_mujoco_limit_violations": sv,
                      "checkpoint_context": "training-set sanity evaluation"})
    atomic_csv(args.output_dir / "action_state_lag_analysis.csv", lag_rows)
    atomic_json(args.output_dir / "action_semantics_report.json", semantics)
    fig, ax = plt.subplots(figsize=(8, 4))
    ov = [r for r in lag_rows if r["joint_index"] == "overall"]
    ax.plot([r["lag"] for r in ov], [r["rmse"] for r in ov], marker="o")
    ax.set(xlabel="lag frames", ylabel="overall RMSE"); fig.tight_layout()
    fig.savefig(args.output_dir / "action_state_lag_rmse.png", dpi=140); plt.close(fig)

    episodes = sorted(set(args.episodes))
    safety_rows, fk_rows, joint3_rows = [], [], []
    final_eps: dict[str, Any] = {}
    for episode in episodes:
        source = load_episode(args.dataset_root, episode)
        pred_path = args.prediction_dir / f"episode_{episode:06d}_chunk_stitched.npz"
        with np.load(pred_path, allow_pickle=False) as z:
            pred = np.asarray(z["chunk_stitched_h10"], dtype=np.float64)
        if pred.shape != source["state"].shape or not np.isfinite(pred).all():
            raise RuntimeError(f"Prediction mismatch episode {episode}: {pred.shape} vs {source['state'].shape}")
        n = min(len(pred), args.max_frames) if args.max_frames else len(pred)
        trajectories = {"expert_state": source["state"][:n], "expert_action": source["action"][:n], "smolvla_h10": pred[:n]}
        epout = args.output_dir / f"episode_{episode:06d}"; epout.mkdir(exist_ok=True)
        qposes, fks, reports = {}, {}, {}
        dynamic_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, raw in trajectories.items():
            qpos, clips = mapped_qpos(raw)
            qposes[name] = qpos
            met = trajectory_metrics(raw, qpos, limits, model, source["state"][0])
            met["existing_mapping_gripper_clip_frames"] = clips
            direct_allowed = name == "expert_state" or semantics["direct_joint_target_supported"]
            met["kinematic_replay_status"] = "SUCCESS" if direct_allowed and met["finite"] else "BLOCKED_ACTION_SEMANTICS"
            if direct_allowed:
                fks[name] = fk(model, qpos)
            dynamic_allowed = (mapping["position_actuator_mapping_validated"] and direct_allowed
                               and met["mapped_joint_limit_violations"] == 0 and met["nan_inf_count"] == 0)
            met["dynamic_replay_status"] = "AVAILABLE_NOT_EXECUTED" if dynamic_allowed else "DYNAMIC_REPLAY_NOT_AVAILABLE"
            if args.execute and not args.dry_run and dynamic_allowed:
                ctrl = raw.copy()
                ctrl[:, 6] = qpos[:, 6]; ctrl[:, 13] = qpos[:, 14]
                met["dynamic"], actual = dynamic_replay(model, qposes["expert_state"][0], ctrl)
                dynamic_series[name] = (ctrl, actual)
                met["dynamic_replay_status"] = "SUCCESS"
            reports[name] = met
            safety_rows.append({"episode": episode, "trajectory": name, **{k: v for k, v in met.items() if not isinstance(v, (list, dict))}})
            np.savez_compressed(epout / f"{name}_trajectory.npz", raw=raw.astype(np.float32),
                                mapped_qpos=qpos.astype(np.float32), frame_index=source["frame_index"][:n],
                                timestamp=source["timestamp"][:n])
        if set(fks) == set(trajectories):
            state_fk = fks["expert_state"]
            for name in ("expert_action", "smolvla_h10"):
                x = fks[name]
                lrmse = float(np.sqrt(np.mean(np.sum((x["left_position_m"] - state_fk["left_position_m"]) ** 2, axis=1))) * 1000)
                rrmse = float(np.sqrt(np.mean(np.sum((x["right_position_m"] - state_fk["right_position_m"]) ** 2, axis=1))) * 1000)
                lori = float(np.sqrt(np.mean(quat_angle_deg(x["left_quaternion_wxyz"], state_fk["left_quaternion_wxyz"]) ** 2)))
                rori = float(np.sqrt(np.mean(quat_angle_deg(x["right_quaternion_wxyz"], state_fk["right_quaternion_wxyz"]) ** 2)))
                fk_rows.append({"episode": episode, "trajectory_vs_expert_state": name,
                                "left_position_rmse_mm": lrmse, "right_position_rmse_mm": rrmse,
                                "left_orientation_rmse_deg": lori, "right_orientation_rmse_deg": rori})
            np.savez_compressed(epout / "end_effector_trajectories.npz",
                                **{f"{name}_{key}": val.astype(np.float32) for name, d in fks.items() for key, val in d.items()})
        for frame in range(n):
            vals = [trajectories[k][frame, 3] for k in trajectories]
            if any(v < limits[3, 0] or v > limits[3, 1] or v < ISAAC_LIMITS[3, 0] or v > ISAAC_LIMITS[3, 1] for v in vals):
                joint3_rows.append({"episode": episode, "frame": int(source["frame_index"][frame]),
                                    "expert_state": vals[0], "expert_action": vals[1], "smolvla_h10": vals[2],
                                    "mujoco_lower": limits[3, 0], "mujoco_upper": limits[3, 1],
                                    "isaac_lower": ISAAC_LIMITS[3, 0], "isaac_upper": ISAAC_LIMITS[3, 1]})
        atomic_json(epout / "safety_report.json", reports)
        if set(fks) == set(trajectories):
            plot_episode(epout, *trajectories.values(), fks)
            if args.record_video:
                record_kinematic_video(epout / "kinematic_replay.mp4", fks)
        if args.record_video and dynamic_series:
            record_dynamic_video(epout / "dynamic_replay.mp4", dynamic_series)
        final_eps[str(episode)] = reports

    atomic_csv(args.output_dir / "trajectory_safety_summary.csv", safety_rows)
    if fk_rows:
        atomic_csv(args.output_dir / "fk_comparison_summary.csv", fk_rows)
    else:
        atomic_csv(args.output_dir / "fk_comparison_summary.csv", [{"status": "NOT_AVAILABLE"}])
    if joint3_rows:
        atomic_csv(args.output_dir / "joint3_detailed_analysis.csv", joint3_rows)
    else:
        atomic_csv(args.output_dir / "joint3_detailed_analysis.csv", [{"status": "NO_RELEVANT_FRAMES"}])
    report = {"model_xml": str(xml), "repo_id": args.repo_id, "episodes": episodes,
              "action_semantics": semantics, "episode_results": final_eps,
              "isaac_lab_executed": False, "g1_conversion_executed": False, "hardware_commands_sent": False,
              "dynamic_execution_requested": bool(args.execute and not args.dry_run)}
    atomic_json(args.output_dir / "final_validation_report.json", report)
    LOG.info("Completed: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
