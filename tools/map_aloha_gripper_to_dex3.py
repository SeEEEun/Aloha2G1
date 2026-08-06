#!/usr/bin/env python3
"""Map GoPark episode-0 ALOHA grippers to safe, generic Dex3 hand motion.

This is an offline data conversion and validation tool.  It never communicates
with robot hardware and does not claim task-specific grasp success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from scipy.signal import butter, medfilt, savgol_filter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "GoPark"
DEFAULT_ARM = ROOT / "g1_high_quality_position_trajectory.npz"
DEFAULT_MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
DEFAULT_OUTPUT = ROOT / "g1_arm_dex3_trajectory.npz"
DEFAULT_REPORT = ROOT / "g1_arm_dex3_mapping_report.txt"
DEFAULT_PLOT = ROOT / "aloha_dex3_gripper_mapping.png"
ALOHA_MODEL = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
REQUIRED_ARM_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--arm-input", type=Path, default=DEFAULT_ARM)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    return p.parse_args()


def names(model, kind, count):
    return [mujoco.mj_id2name(model, kind, i) or f"unnamed_{i}" for i in range(count)]


def load_source(dataset: Path):
    info = json.loads((dataset / "meta/info.json").read_text())
    feature = info["features"]["observation.state"]
    feature_names = list(feature["names"])
    if feature["shape"] != [14] or len(feature_names) != 14:
        raise ValueError(f"unexpected observation.state feature: {feature}")
    # Resolve grippers from metadata plus the replay model actuator/joint mapping, not position guesses.
    model = mujoco.MjModel.from_xml_path(str(ALOHA_MODEL))
    actuator_joint = {}
    for aid, aname in enumerate(names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)):
        jid = int(model.actuator_trnid[aid, 0])
        actuator_joint[aname] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    expected = {"left": "follower_left_gripper", "right": "follower_right_gripper"}
    resolved = {}
    for side, actuator in expected.items():
        if actuator not in actuator_joint:
            raise ValueError(f"ALOHA actuator missing: {actuator}")
        candidates = [i for i, n in enumerate(feature_names) if n == f"{side}_joint_6"]
        if len(candidates) != 1:
            raise ValueError(f"cannot uniquely resolve {side} gripper from metadata: {feature_names}")
        resolved[side] = candidates[0]
    df = pd.read_parquet(dataset / "data/chunk-000/episode_000000.parquet",
                         columns=["observation.state", "timestamp", "frame_index"])
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    timestamps = df["timestamp"].to_numpy(dtype=np.float64)
    if state.shape[1] != len(feature_names) or not np.array_equal(df.frame_index, np.arange(len(df))):
        raise ValueError("source state width or frame_index is inconsistent")
    fps = float(info["fps"])
    timestamp_fps = 1.0 / np.mean(np.diff(timestamps))
    if abs(timestamp_fps - fps) > 1e-3:
        raise ValueError(f"metadata/timestamp fps mismatch: {fps} vs {timestamp_fps}")
    # The replay model uses two mirrored carriage joints. Larger qpos separates their bodies.
    # The model limit therefore establishes increasing value == opening, independently per hand.
    limits = {}
    for side in ("left", "right"):
        jn = f"follower_{side}_left_carriage_joint"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        limits[side] = model.jnt_range[jid].copy()
        if not (limits[side][1] > limits[side][0]):
            raise ValueError(f"invalid ALOHA gripper limit {side}: {limits[side]}")
    return state, timestamps, fps, feature_names, resolved, limits, actuator_joint


def filter_candidates(x: np.ndarray, fps: float):
    candidates = {
        "moving_median_window_5": medfilt(x, kernel_size=5),
        "savgol_window_5_poly_2": savgol_filter(x, 5, 2, mode="interp"),
        "zero_phase_butterworth_order_2_cutoff_4Hz": sosfiltfilt(butter(2, 4.0, fs=fps, output="sos"), x),
    }
    return candidates


def threshold_events(x: np.ndarray):
    """Return opening/closing starts and maximum closure using robust derivative thresholds."""
    dx = np.diff(x, prepend=x[0])
    threshold = max(float(np.percentile(np.abs(dx), 75)) * 1.5, 0.002)
    opening = np.flatnonzero(dx < -threshold)  # closure ratio decreases
    closing = np.flatnonzero(dx > threshold)
    return (int(opening[0]) if opening.size else -1,
            int(closing[0]) if closing.size else -1,
            int(np.argmax(x)))


def event_delta(raw: np.ndarray, filtered: np.ndarray):
    a, b = threshold_events(raw), threshold_events(filtered)
    return np.array([y-x if x >= 0 and y >= 0 else -999 for x, y in zip(a, b)], dtype=np.int64)


def model_layout(path: Path, arm_names):
    model = mujoco.MjModel.from_xml_path(str(path))
    actuator_names = names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    actuator_joints = []
    for i in range(model.nu):
        actuator_joints.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                 int(model.actuator_trnid[i, 0])))
    if list(arm_names) != REQUIRED_ARM_NAMES:
        raise ValueError(f"arm NPZ ordering differs from required model order: {arm_names}")
    for n in arm_names:
        if n not in actuator_joints:
            raise ValueError(f"arm joint absent from actuator ordering: {n}")
    hands = {}
    for side in ("left", "right"):
        hand_names = [j for j in actuator_joints if j.startswith(f"{side}_hand_")]
        if len(hand_names) != 7:
            raise ValueError(f"expected 7 {side} Dex3 actuators, got {hand_names}")
        jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hand_names])
        hands[side] = dict(names=hand_names, jids=jids, qadr=model.jnt_qposadr[jids].copy(),
                           ranges=model.jnt_range[jids].copy(),
                           actuator_ids=np.array([actuator_joints.index(n) for n in hand_names]))
    arm_jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm_names])
    return model, hands, model.jnt_qposadr[arm_jids].copy(), actuator_names, actuator_joints


def poses(model, hand):
    """Open is the XML stand keyframe; grasp stays well inside every hard limit."""
    stand = model.key_qpos[0] if model.nkey else np.zeros(model.nq)
    result = {}
    for side in ("left", "right"):
        h = hand[side]
        open_pose = stand[h["qadr"]].copy()
        # The distributed XML stand key uses +/-1.05 for thumb_1 while its hard
        # limit is +/-1.0472. Keep the intended posture but bring it safely in-range.
        open_margin = 0.005 * (h["ranges"][:, 1] - h["ranges"][:, 0])
        open_pose = np.clip(open_pose, h["ranges"][:, 0] + open_margin,
                            h["ranges"][:, 1] - open_margin)
        sign = -1.0 if side == "left" else 1.0
        # actuator order: thumb0, thumb1, thumb2, then two 2-joint fingers (order differs R/L).
        grasp = np.array([-0.18, sign * -0.48, sign * -1.35,
                          sign * 1.02, sign * 1.22, sign * 1.02, sign * 1.22])
        # Mirrored thumb flexion signs: left thumb1/thumb2 positive; right negative.
        if side == "left": grasp[:3] = [-0.18, 0.48, 1.35]
        else: grasp[:3] = [-0.18, -0.48, -1.35]
        margin = 0.08 * (h["ranges"][:, 1] - h["ranges"][:, 0])
        grasp = np.clip(grasp, h["ranges"][:, 0] + margin, h["ranges"][:, 1] - margin)
        result[side] = (open_pose, grasp)
    return result


def closure_curves(g):
    # Thumb begins gently; index/middle are near-linear with a small deterministic stagger.
    def delayed(x, delay, power):
        return np.clip((x-delay)/(1-delay), 0, 1) ** power
    return np.column_stack([g**1.18, g**1.08, g**1.02,
                            delayed(g, .00, .98), delayed(g, .00, 1.00),
                            delayed(g, .07, .92), delayed(g, .07, .95)])


def hand_trajectory(g, open_pose, grasp_pose):
    c = closure_curves(g)
    return open_pose + c * (grasp_pose-open_pose), c.mean(axis=1)


def full_qpos(model, base, arm_q, left_q, right_q, arm_qadr, hands):
    out = np.repeat(base[None, :], len(arm_q), axis=0)
    out[:, arm_qadr] = arm_q
    out[:, hands["left"]["qadr"]] = left_q
    out[:, hands["right"]["qadr"]] = right_q
    return out


def contact_scan(model, qpos):
    data = mujoco.MjData(model)
    flags = np.zeros(len(qpos), dtype=bool)
    frame_pairs, pair_depths = [], []
    for t, q in enumerate(qpos):
        data.qpos[:] = q; data.qvel[:] = 0; mujoco.mj_forward(model, data)
        pairs, depths = {}, {}
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body_{b1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body_{b2}"
            pair = " <-> ".join(sorted((n1, n2)))
            pairs[pair] = min(pairs.get(pair, 0.0), float(c.dist))
        flags[t] = bool(pairs)
        frame_pairs.append("; ".join(sorted(pairs)))
        pair_depths.append(pairs)
    static = pair_depths[0]
    dynamic = sorted({p for d in pair_depths[1:] for p in d if p not in static})
    increased = sorted({p for d in pair_depths[1:] for p, dist in d.items()
                        if p in static and dist < static[p] - 1e-5})
    return flags, np.asarray(frame_pairs, dtype="U1024"), static, dynamic, increased


def main():
    a = args()
    for p in (a.output, a.report, a.plot):
        if p.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {p}")
    state, timestamps, fps, feature_names, gi, aloha_limits, aloha_actuators = load_source(a.dataset)
    with np.load(a.arm_input, allow_pickle=False) as z:
        arm = {k: z[k] for k in z.files}
    task_arm = arm["task_joint_trajectory_q"].astype(np.float64)
    exec_arm = arm["execution_joint_trajectory_q"].astype(np.float64)
    if len(state) != len(task_arm):
        raise ValueError(f"source/task frame mismatch ({len(state)} vs {len(task_arm)}); explicit resampling required")
    pre, post = int(arm["pre_roll_frame_count"]), int(arm["post_roll_frame_count"])
    if len(exec_arm) != pre + len(task_arm) + post:
        raise ValueError("execution trajectory is not pre+task+post")
    raw = {s: state[:, gi[s]] for s in ("left", "right")}
    filtered, candidate_stats = {}, {}
    normalized = {}
    delays = {}
    for side in ("left", "right"):
        lo, hi = aloha_limits[side]
        raw_closure = np.clip((hi-raw[side])/(hi-lo), 0, 1)
        cands = filter_candidates(raw[side], fps)
        # Window-5 Savitzky-Golay removes isolated noise with zero phase and best edge timing here.
        filtered[side] = cands["savgol_window_5_poly_2"]
        normalized[side] = np.clip((hi-filtered[side])/(hi-lo), 0, 1)
        delays[side] = event_delta(raw_closure, normalized[side])
        candidate_stats[side] = {k: [float(np.sqrt(np.mean((v-raw[side])**2))),
                                      int(np.max(np.abs(event_delta(raw_closure, np.clip((hi-v)/(hi-lo),0,1)))))]
                                 for k, v in cands.items()}
    model, hands, arm_qadr, actuator_names, actuator_joints = model_layout(a.model, arm["arm_joint_names"])
    pose = poses(model, hands)
    ltask, lratio = hand_trajectory(normalized["left"], *pose["left"])
    rtask, rratio = hand_trajectory(normalized["right"], *pose["right"])
    def execution(q):
        # Cubic smoothstep from XML stand/open pose to first task pose.
        u = np.linspace(0, 1, pre, endpoint=False); u = (3*u*u-2*u*u*u)[:, None]
        lead = pose["left" if q is ltask else "right"][0] + u*(q[0]-pose["left" if q is ltask else "right"][0])
        return np.vstack((lead, q, np.repeat(q[-1][None, :], post, axis=0)))
    lexec, rexec = execution(ltask), execution(rtask)
    base = model.key_qpos[0].copy() if model.nkey else np.zeros(model.nq)
    full_task = full_qpos(model, base, task_arm, ltask, rtask, arm_qadr, hands)
    full_exec = full_qpos(model, base, exec_arm, lexec, rexec, arm_qadr, hands)
    _, _, xml_default_contacts, _, _ = contact_scan(model, base[None, :])
    collision_flag, collision_pairs, static, dynamic, increased = contact_scan(model, full_task)
    all_hand = np.column_stack((lexec, rexec))
    velocity = np.diff(all_hand, axis=0) * fps
    ranges = np.vstack((hands["left"]["ranges"], hands["right"]["ranges"]))
    margins = np.minimum(all_hand-ranges[:,0], ranges[:,1]-all_hand)
    violations = int(np.count_nonzero(margins < -1e-10))
    lcorr = float(np.corrcoef(normalized["left"], lratio)[0,1])
    rcorr = float(np.corrcoef(normalized["right"], rratio)[0,1])
    independent = not np.allclose(normalized["left"], normalized["right"], atol=1e-6)
    validation = np.array([
        len(state), len(task_arm), len(exec_arm), lcorr, rcorr,
        np.max(np.abs(np.diff(all_hand, axis=0))), np.max(np.abs(velocity)), violations,
        len(static), len(dynamic), len(increased), int(independent),
        np.count_nonzero(~np.isfinite(full_exec)), len(np.unique(ltask, axis=0)), len(np.unique(rtask, axis=0))
    ], dtype=np.float64)
    validation_names = np.array(["source_frames","task_frames","execution_frames","left_correlation",
        "right_correlation","max_finger_step_rad","max_finger_velocity_rad_s","limit_violations",
        "static_contact_pairs","new_dynamic_contact_pairs","increased_penetration_pairs","hands_independent",
        "nonfinite_values","left_unique_poses","right_unique_poses"])
    mapping_method = ("deterministic per-finger interpolation from ALOHA 1-DoF closure; "
                      "thumb gentle power curve, index/middle near-linear, second finger pair delayed 0.07")
    normalization_parameters = json.dumps({s: {"method":"ALOHA model joint limits", "open":float(aloha_limits[s][1]),
        "closed":float(aloha_limits[s][0]), "clip":[0,1], "percentiles":np.percentile(raw[s],[1,99]).tolist(),
        "observed_minmax":[float(raw[s].min()),float(raw[s].max())]} for s in ("left","right")})
    np.savez_compressed(a.output,
        fps=np.float64(fps), source_frame_count=np.int64(len(state)),
        aloha_left_gripper_raw=raw["left"], aloha_right_gripper_raw=raw["right"],
        aloha_left_gripper_filtered=filtered["left"], aloha_right_gripper_filtered=filtered["right"],
        aloha_left_gripper_normalized=normalized["left"], aloha_right_gripper_normalized=normalized["right"],
        dex3_left_joint_names=np.array(hands["left"]["names"]), dex3_right_joint_names=np.array(hands["right"]["names"]),
        dex3_left_qpos_address=hands["left"]["qadr"], dex3_right_qpos_address=hands["right"]["qadr"],
        dex3_left_joint_range=hands["left"]["ranges"], dex3_right_joint_range=hands["right"]["ranges"],
        dex3_left_open_pose=pose["left"][0], dex3_right_open_pose=pose["right"][0],
        dex3_left_grasp_pose=pose["left"][1], dex3_right_grasp_pose=pose["right"][1],
        g1_arm_task_q=task_arm, g1_left_dex3_task_q=ltask, g1_right_dex3_task_q=rtask,
        g1_arm_execution_q=exec_arm, g1_left_dex3_execution_q=lexec, g1_right_dex3_execution_q=rexec,
        g1_full_task_qpos=full_task, g1_full_execution_qpos=full_exec,
        dex3_left_closure_ratio=lratio, dex3_right_closure_ratio=rratio,
        left_mapping_correlation=np.float64(lcorr), right_mapping_correlation=np.float64(rcorr),
        finger_joint_velocity=velocity, finger_joint_limit_margin=margins,
        collision_flag=collision_flag, collision_body_pairs=collision_pairs,
        xml_default_collision_body_pairs=np.array(sorted(xml_default_contacts)),
        static_collision_body_pairs=np.array(sorted(static)), dynamic_collision_body_pairs=np.array(dynamic),
        increased_penetration_body_pairs=np.array(increased),
        arm_joint_names=arm["arm_joint_names"], arm_qpos_address=arm_qadr,
        actuator_names=np.array(actuator_names), actuator_joint_order=np.array(actuator_joints),
        aloha_feature_names=np.array(feature_names), aloha_gripper_indices=np.array([gi["left"],gi["right"]]),
        filter_event_delay_frames=np.vstack((delays["left"],delays["right"])),
        mapping_method=np.array(mapping_method), normalization_parameters=np.array(normalization_parameters),
        validation_summary=validation, validation_summary_names=validation_names)
    t = np.arange(len(state))/fps
    fig, ax = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    for i, side in enumerate(("left", "right")):
        basei = i*2
        ax[basei].plot(t, raw[side], alpha=.55, label="raw opening (m)")
        ax[basei].plot(t, filtered[side], lw=1.3, label="filtered opening (m)")
        ax[basei].set_ylabel(f"ALOHA {side}"); ax[basei].legend(); ax[basei].grid(alpha=.25)
        ax[basei+1].plot(t, normalized[side], label="ALOHA closure", alpha=.6)
        ax[basei+1].plot(t, lratio if side=="left" else rratio, label="Dex3 closure")
        ax[basei+1].set_ylabel("closure [0,1]"); ax[basei+1].legend(); ax[basei+1].grid(alpha=.25)
    ax[4].plot(t, ltask[:,0], label=hands["left"]["names"][0])
    ax[4].plot(t, ltask[:,3], label=hands["left"]["names"][3])
    ax[4].plot(t, rtask[:,0], '--', label=hands["right"]["names"][0])
    ax[4].plot(t, rtask[:,3], '--', label=hands["right"]["names"][3])
    ax[4].set_ylabel("joint q [rad]"); ax[4].set_xlabel("time [s]"); ax[4].legend(ncol=2); ax[4].grid(alpha=.25)
    fig.suptitle("ALOHA episode 0 gripper to G1 Dex3 mapping (30 Hz)"); fig.tight_layout(); fig.savefig(a.plot, dpi=160); plt.close(fig)
    report = f"""ALOHA episode 0 -> G1 Dex3 mapping report
================================================
Purpose: generic human-like open/close mapping only; NOT a task-specific grasp and no grasp-success claim.

ALOHA data
- observation.state dimension: {state.shape[1]}
- feature names: {feature_names}
- left gripper: index {gi['left']}, name {feature_names[gi['left']]}
- right gripper: index {gi['right']}, name {feature_names[gi['right']]}
- left raw min/max/mean: {raw['left'].min():.9f} / {raw['left'].max():.9f} / {raw['left'].mean():.9f}
- right raw min/max/mean: {raw['right'].min():.9f} / {raw['right'].max():.9f} / {raw['right'].mean():.9f}
- direction (both hands): increasing value = OPEN; established from ALOHA model's mirrored carriage joint actuator and [0, 0.044] limits.
- frames / sampling rate: {len(state)} / {fps:.6f} Hz; timestamps confirm {1/np.mean(np.diff(timestamps)):.6f} Hz.
- alignment: source and G1 task are both {len(state)} frames; no task resampling. Execution = {pre} pre-roll + {len(state)} task + {post} hold.

Normalization and filtering
- chosen normalization: actual ALOHA model joint limits, closure=(0.044-value)/0.044, clipped [0,1].
- alternatives checked: observed min/max and 1st-99th percentiles (details in NPZ normalization_parameters).
- chosen filter: zero-phase Savitzky-Golay window=5, polynomial=2, separately per hand.
- candidate [RMSE, max event shift frames]: {json.dumps(candidate_stats)}
- event delay [opening start, closing start, maximum closure] frames: left {delays['left'].tolist()}, right {delays['right'].tolist()} (-999 means not detected).

Dex3 model structure
- actuator ordering: {actuator_names}
- arm ordering verified unchanged: {list(arm['arm_joint_names'])}; qpos {arm_qadr.tolist()}
- left Dex3 names/qpos/range: {list(zip(hands['left']['names'], hands['left']['qadr'].tolist(), hands['left']['ranges'].tolist()))}
- right Dex3 names/qpos/range: {list(zip(hands['right']['names'], hands['right']['qadr'].tolist(), hands['right']['ranges'].tolist()))}
- open pose (XML stand keyframe): left {pose['left'][0].tolist()}, right {pose['right'][0].tolist()}
- generic grasp pose (interior of limits): left {pose['left'][1].tolist()}, right {pose['right'][1].tolist()}
- mapping: {mapping_method}

Quantitative validation
- left/right Pearson correlation: {lcorr:.9f} / {rcorr:.9f}
- maximum finger step / velocity at 30 Hz: {validation[5]:.9f} rad / {validation[6]:.9f} rad/s
- joint-limit violations: {violations}; NaN/inf: {int(validation[12])}
- left/right unique task poses: {int(validation[13])} / {int(validation[14])}; independent: {independent}
- static frame-0 contact pairs ({len(static)}): {sorted(static)}
- XML unmodified stand-pose contacts ({len(xml_default_contacts)}): {sorted(xml_default_contacts)}
  These reproduce the previously reported thumb_1_link <-> wrist_yaw_link contacts; the safe clipped open pose does not silently suppress their record.
- new dynamic collision pairs ({len(dynamic)}): {dynamic}
- static pairs with increased penetration ({len(increased)}): {increased}
- collision contacts are reported, not suppressed. collision_flag includes all model contacts.

Visual review still required
- verify left/right opening direction and independent timing against the ALOHA replay/video;
- verify arm/hand synchronization, no abnormal thumb/finger interpenetration, no limit sticking, and no visible jitter.
- This deterministic conversion preserves 1-DoF closure timing/degree/speed approximately; it is not original per-finger motion.
"""
    a.report.write_text(report)
    # Reload and independently verify the stored artifact.
    with np.load(a.output, allow_pickle=False) as z:
        required = ["fps","source_frame_count","aloha_left_gripper_raw","aloha_right_gripper_raw",
            "aloha_left_gripper_filtered","aloha_right_gripper_filtered","aloha_left_gripper_normalized",
            "aloha_right_gripper_normalized","dex3_left_joint_names","dex3_right_joint_names",
            "g1_arm_task_q","g1_left_dex3_task_q","g1_right_dex3_task_q","g1_arm_execution_q",
            "g1_left_dex3_execution_q","g1_right_dex3_execution_q","g1_full_task_qpos",
            "g1_full_execution_qpos","finger_joint_velocity","finger_joint_limit_margin","collision_flag"]
        missing = [k for k in required if k not in z]
        assert not missing, missing
        assert z["g1_arm_task_q"].shape == (713,14) and z["g1_full_execution_qpos"].shape == (803,model.nq)
        assert z["g1_left_dex3_task_q"].shape == z["g1_right_dex3_task_q"].shape == (713,7)
        assert all(np.isfinite(z[k]).all() for k in z.files if z[k].dtype.kind in "fciub")
        assert np.min(z["finger_joint_limit_margin"]) >= -1e-10
        assert len(np.unique(z["g1_left_dex3_task_q"],axis=0)) > 1 and len(np.unique(z["g1_right_dex3_task_q"],axis=0)) > 1
        assert np.allclose(np.diff(np.column_stack((z["g1_left_dex3_execution_q"],z["g1_right_dex3_execution_q"])),axis=0)*fps,
                           z["finger_joint_velocity"])
    print(f"ALOHA gripper indices: left={gi['left']}, right={gi['right']}")
    print(f"raw ranges: left=[{raw['left'].min():.9f}, {raw['left'].max():.9f}], right=[{raw['right'].min():.9f}, {raw['right'].max():.9f}]")
    print("direction: increasing = OPEN (both); normalization: ALOHA model limits [0, 0.044]")
    print(f"Dex3 left joints: {hands['left']['names']}\nDex3 right joints: {hands['right']['names']}")
    print(f"open poses: L={pose['left'][0].tolist()} R={pose['right'][0].tolist()}")
    print(f"grasp poses: L={pose['left'][1].tolist()} R={pose['right'][1].tolist()}")
    print(f"mapping correlation: left={lcorr:.9f}, right={rcorr:.9f}; event delays L={delays['left'].tolist()} R={delays['right'].tolist()}")
    print(f"Dex3 task/execution: L={ltask.shape}/{lexec.shape}, R={rtask.shape}/{rexec.shape}; unique={int(validation[13])}/{int(validation[14])}")
    print(f"max finger step={validation[5]:.9f} rad; velocity={validation[6]:.9f} rad/s; limit violations={violations}")
    print(f"static contacts={len(static)}; new dynamic collisions={len(dynamic)}; increased penetration={len(increased)}")
    print(f"created: {a.output.resolve()}\ncreated: {a.report.resolve()}\ncreated: {a.plot.resolve()}")
    print(f"play: /home/jbnu/miniconda3/envs/aloha_mujoco/bin/python {ROOT/'play_g1_arm_dex3_trajectory_mujoco.py'} --input {a.output.resolve()} --mode arm-dex3 --speed 0.5")


if __name__ == "__main__":
    main()
