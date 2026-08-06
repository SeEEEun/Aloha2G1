#!/usr/bin/env python3
"""Episode 49 optimized_action -> G1 using a validated natural start and relative bimanual motion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

import retarget_episode49_optimized_action_to_g1 as latest

ROOT = Path("/home/jbnu/aloha_g1_dataset")
NATURAL_START = ROOT / (
    "converted_runs/magsafe_20260723_162750/dynamic_bimanual_spacing/"
    "g1_dynamic_bimanual_full_trajectory.npz"
)
NATURAL_VALIDATION_REPORT = NATURAL_START.with_name("full_trajectory_report.txt")
OUT = ROOT / (
    "converted_runs/smolvla_20k_episode49_consensus_relative_g1/"
    "g1_episode49_consensus_relative_trajectory.npz"
)
REPORT = OUT.with_name("g1_episode49_consensus_relative_report.json")
VERDICT_READY = "G1_RELATIVE_BIMANUAL_READY_FOR_VISUAL_REVIEW"
VERDICT_BLOCKED = "G1_RELATIVE_BIMANUAL_SAFETY_BLOCKED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=latest.SOURCE)
    p.add_argument("--natural-start", type=Path, default=NATURAL_START)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--temporal-iterations", type=int, default=8)
    return p.parse_args()


def actual_full_qpos_contacts(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check qpos + mj_forward contacts, excluding only same-hand internal contacts."""
    data = mujoco.MjData(model)
    collision = np.zeros(len(qpos), dtype=bool)
    cross = np.zeros(len(qpos), dtype=bool)
    pairs: set[str] = set()
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            names = [
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
                ) or "world"
                for geom in (contact.geom1, contact.geom2)
            ]
            joined = "|".join(sorted(names))
            same_hand = (
                (names[0].startswith("left_hand") and names[1].startswith("left_"))
                or (names[1].startswith("left_hand") and names[0].startswith("left_"))
                or (names[0].startswith("right_hand") and names[1].startswith("right_"))
                or (names[1].startswith("right_hand") and names[0].startswith("right_"))
            )
            relevant = any(word in "".join(names) for word in (
                "hand", "wrist", "elbow", "shoulder", "torso"
            ))
            is_cross = (
                (names[0].startswith("left_") and names[1].startswith("right_"))
                or (names[1].startswith("left_") and names[0].startswith("right_"))
            )
            if relevant and not same_hand:
                collision[t] = True
                pairs.add(joined)
            if is_cross:
                cross[t] = True
    return collision, cross, np.asarray(sorted(pairs))


def load_natural_start(path: Path, info: dict) -> dict[str, np.ndarray]:
    """Read the already validated calibration qpos; never derive it from episode 49."""
    with np.load(path, allow_pickle=False) as z:
        required = ("task_arm", "g1_full_full_qpos", "task_start_frame")
        missing = [key for key in required if key not in z.files]
        if missing:
            raise RuntimeError(f"NATURAL_G1_START_POSE_NOT_FOUND: missing {missing} in {path}")
        task_start = int(z["task_start_frame"])
        arm_q = z["task_arm"][0].astype(np.float64)
        full_qpos = z["g1_full_full_qpos"][task_start].astype(np.float64)
    if arm_q.shape != (14,) or full_qpos.shape != (50,):
        raise RuntimeError(
            f"NATURAL_G1_START_POSE_NOT_FOUND: invalid q shapes {arm_q.shape}/{full_qpos.shape}"
        )
    if not np.isfinite(arm_q).all() or not np.isfinite(full_qpos).all():
        raise RuntimeError("NATURAL_G1_START_POSE_NOT_FOUND: non-finite qpos")
    data = mujoco.MjData(info["model"])
    state = latest.frame_state(info, data, arm_q)
    limits = info["joint_limits"]
    margin = np.minimum(arm_q-limits[:, 0], limits[:, 1]-arm_q)
    collision, cross, pairs = actual_full_qpos_contacts(info["model"], full_qpos[None])
    if np.any(margin < -1e-9) or collision.any() or cross.any():
        raise RuntimeError(
            "NATURAL_G1_START_POSE_NOT_FOUND: stored pose fails independent "
            f"limit/collision validation; min_margin={margin.min()}, pairs={pairs.tolist()}"
        )
    left, right = state["left_pos"].copy(), state["right_pos"].copy()
    return {
        "arm_q": arm_q, "full_qpos": full_qpos, "left": left, "right": right,
        "midpoint": .5*(left+right), "relative": right-left, "margin": margin,
        "collision": collision, "cross": cross, "pairs": pairs,
    }


def relative_targets(fk: dict[str, np.ndarray], start: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The requested midpoint/relative-vector mapping, with no spacing clamp."""
    aloha_left = fk["left_position_m"]
    aloha_right = fk["right_position_m"]
    aloha_midpoint = .5*(aloha_left+aloha_right)
    aloha_relative = aloha_right-aloha_left
    rotation = latest.align_mod.make_align_rotation(latest.ALIGN_RPY)
    delta_midpoint = (aloha_midpoint-aloha_midpoint[0]) @ rotation.T
    delta_relative = (aloha_relative-aloha_relative[0]) @ rotation.T
    target_midpoint = start["midpoint"] + latest.SCALE*delta_midpoint
    target_relative = start["relative"] + latest.SCALE*delta_relative
    left = target_midpoint-.5*target_relative
    right = target_midpoint+.5*target_relative

    # Orientation arrays are needed only by the legacy frame seed API. The
    # temporal solve below uses orientation_weight=0 exactly as the selected
    # position-priority candidate in the latest converter.
    lrot = np.repeat(np.eye(3)[None], len(left), axis=0)
    rrot = np.repeat(np.eye(3)[None], len(right), axis=0)
    return {
        "lp": left, "rp": right, "lr": lrot, "rr": rrot,
        "aloha_midpoint": aloha_midpoint, "aloha_relative": aloha_relative,
        "target_midpoint": target_midpoint, "target_relative": target_relative,
    }


def scalar_stats(x: np.ndarray) -> dict[str, float]:
    return latest.stats(np.asarray(x))


def main() -> int:
    a = parse_args()
    for path in (a.source, a.natural_start, latest.ALOHA_XML, latest.G1_XML):
        if not path.exists():
            if path == a.natural_start:
                raise RuntimeError(f"NATURAL_G1_START_POSE_NOT_FOUND: {path}")
            raise FileNotFoundError(path)
    if a.output.exists():
        raise FileExistsError(a.output)

    raw, timestamp = latest.load_source(a.source, a.max_frames)
    aloha_model, _ = latest.aloha.load_validated_model(latest.ALOHA_XML)
    aloha_qpos, clipped = latest.aloha.mapped_qpos(raw)
    fk = latest.aloha.fk(aloha_model, aloha_qpos)
    info = latest.ik.validate_model(latest.G1_XML)
    start = load_natural_start(a.natural_start, info)
    targets = relative_targets(fk, start)

    # Reuse the latest converter's position seed and whole-trajectory temporal
    # solve verbatim. The validated natural posture replaces its former nominal.
    seed = latest.position_seed(info, targets, start["arm_q"])
    task_arm = latest.temporal_solve(
        info, targets, seed, start["arm_q"], 0.0, a.temporal_iterations
    )
    achieved = latest.evaluate(info, targets, task_arm)
    hand = latest.hands(info, raw)

    approach_arm = latest.approach_mod.minimum_jerk(
        info["stand_arm_q"], task_arm[0], latest.APPROACH
    )
    approach_left = latest.approach_mod.minimum_jerk(
        hand["poses"]["left"][0], hand["left"][0], latest.APPROACH
    )
    approach_right = latest.approach_mod.minimum_jerk(
        hand["poses"]["right"][0], hand["right"][0], latest.APPROACH
    )
    hold_arm = np.repeat(task_arm[:1], latest.HOLD, axis=0)
    hold_left = np.repeat(hand["left"][:1], latest.HOLD, axis=0)
    hold_right = np.repeat(hand["right"][:1], latest.HOLD, axis=0)
    full_arm = np.vstack((approach_arm, hold_arm, task_arm))
    full_left = np.vstack((approach_left, hold_left, hand["left"]))
    full_right = np.vstack((approach_right, hold_right, hand["right"]))
    full_qpos = latest.full_qpos(
        hand["model"], hand["model"].key_qpos[0].copy(), full_arm,
        full_left, full_right, hand["addr"], hand["hands"]
    )

    task_start = latest.APPROACH+latest.HOLD
    limits = info["joint_limits"]
    violations = (full_arm < limits[:, 0]-1e-9) | (full_arm > limits[:, 1]+1e-9)
    collision, cross, collision_pairs = actual_full_qpos_contacts(hand["model"], full_qpos)
    failed = (achieved["le"] > .005) | (achieved["re"] > .005)
    finite = all(np.isfinite(x).all() for x in (
        raw, task_arm, full_qpos, achieved["lp"], achieved["rp"]
    ))
    step = np.abs(np.diff(full_arm, axis=0))
    velocity = step*latest.FPS
    acceleration = np.abs(np.diff(full_arm, n=2, axis=0))*latest.FPS**2
    norms = np.linalg.norm(np.diff(task_arm, axis=0), axis=1)
    branch = np.zeros(len(task_arm), dtype=bool)
    for t in range(1, len(task_arm)):
        local = np.median(norms[max(0,t-10):min(len(norms),t+9)])
        branch[t] = norms[t-1] > max(.15, 8*max(local, 1e-5))

    achieved_midpoint = .5*(achieved["lp"]+achieved["rp"])
    achieved_relative = achieved["rp"]-achieved["lp"]
    aloha_distance = np.linalg.norm(targets["aloha_relative"], axis=1)
    g1_distance = np.linalg.norm(achieved_relative, axis=1)
    distance_correlation = float(np.corrcoef(
        aloha_distance-aloha_distance[0], g1_distance-g1_distance[0]
    )[0, 1])
    midpoint_error = np.linalg.norm(achieved_midpoint-targets["target_midpoint"], axis=1)
    relative_error = np.linalg.norm(achieved_relative-targets["target_relative"], axis=1)
    midpoint_rmse = float(np.sqrt(np.mean(midpoint_error**2)))
    relative_rmse = float(np.sqrt(np.mean(relative_error**2)))
    boundary = float(np.max(np.abs(hold_arm[-1]-task_arm[0])))

    safety = bool(
        not failed.any() and not branch.any() and not violations.any()
        and not collision.any() and not cross.any() and finite
        and step.max(initial=0) <= latest.LIMITS["step"]
        and velocity.max(initial=0) <= latest.LIMITS["velocity"]
        and acceleration.max(initial=0) <= latest.LIMITS["acceleration"]
        and boundary == 0.0
    )
    report = {
        "verdict": VERDICT_READY if safety else VERDICT_BLOCKED,
        "safety_pass": safety,
        "latest_temporal_converter_reused": str(Path(latest.__file__).resolve()),
        "natural_start_pose_source": str(a.natural_start.resolve()),
        "natural_start_validation_report": str(NATURAL_VALIDATION_REPORT.resolve()),
        "source_file": str(a.source.resolve()), "source_trajectory_key": "optimized_action",
        "source_shape": list(raw.shape), "forbidden_trajectory_keys_used": [],
        "position_scale": latest.SCALE,
        "axis_alignment_rpy_deg": latest.ALIGN_RPY.tolist(),
        "target_mapping": "first-frame-relative midpoint and relative-vector",
        "spacing_clamp": None, "manual_frame_edits": False,
        "natural_start": {
            "left_position_m": start["left"].tolist(),
            "right_position_m": start["right"].tolist(),
            "midpoint_m": start["midpoint"].tolist(),
            "relative_vector_m": start["relative"].tolist(),
            "hand_distance_m": float(np.linalg.norm(start["relative"])),
            "joint_limit_min_margin_rad": float(start["margin"].min()),
            "collision": bool(start["collision"].any()),
            "cross_arm_collision": bool(start["cross"].any()),
        },
        "wide_anchor_start_distance_m": float(np.linalg.norm(latest.ANCHOR_R-latest.ANCHOR_L)),
        "g1_hand_distance_m": scalar_stats(g1_distance),
        "aloha_g1_distance_change_correlation": distance_correlation,
        "midpoint_trajectory_rmse_mm": midpoint_rmse*1000,
        "relative_vector_trajectory_rmse_mm": relative_rmse*1000,
        "left_position_error_mm": {k:v*1000 for k,v in scalar_stats(achieved["le"]).items()},
        "right_position_error_mm": {k:v*1000 for k,v in scalar_stats(achieved["re"]).items()},
        "ik_success_rate": float((~failed).mean()),
        "ik_failed_frames": np.flatnonzero(failed).tolist(),
        "branch_discontinuity_count": int(branch.sum()),
        "joint_limit_violation_count": int(violations.sum()),
        "self_collision_frames": int(collision.sum()),
        "cross_arm_collision_frames": int(cross.sum()),
        "collision_pairs": collision_pairs.tolist(),
        "nan_inf_count": 0 if finite else 1,
        "joint_step_rad": scalar_stats(step),
        "joint_velocity_rad_s": scalar_stats(velocity),
        "joint_acceleration_rad_s2": scalar_stats(acceleration),
        "approach_task_boundary_jump_rad": boundary,
        "approach_frames": latest.APPROACH, "hold_frames": latest.HOLD,
        "task_start_frame": task_start,
        "aloha_mapping_clipped_frames": clipped,
        "safety_thresholds": latest.LIMITS,
        "review_gate": "No Isaac Lab readiness verdict before user visual review.",
    }
    g1_hand_commands = np.stack((hand["lratio"], hand["rratio"]), axis=1)
    payload = dict(
        optimized_action=raw.astype(np.float32),
        source_timestamp=timestamp,
        aloha_left_position=fk["left_position_m"],
        aloha_right_position=fk["right_position_m"],
        aloha_midpoint=targets["aloha_midpoint"],
        aloha_relative_vector=targets["aloha_relative"],
        g1_start_qpos=start["full_qpos"],
        g1_start_arm_q=start["arm_q"],
        g1_start_left_position=start["left"],
        g1_start_right_position=start["right"],
        g1_start_midpoint=start["midpoint"],
        g1_start_relative_vector=start["relative"],
        g1_target_left_position=targets["lp"],
        g1_target_right_position=targets["rp"],
        g1_target_midpoint=targets["target_midpoint"],
        g1_target_relative_vector=targets["target_relative"],
        g1_achieved_left_position=achieved["lp"],
        g1_achieved_right_position=achieved["rp"],
        g1_achieved_midpoint=achieved_midpoint,
        g1_achieved_relative_vector=achieved_relative,
        g1_arm_joint_trajectory=task_arm,
        full_g1_joint_trajectory=full_qpos,
        full_arm=full_arm,
        full_left_dex3=full_left,
        full_right_dex3=full_right,
        g1_hand_commands=g1_hand_commands,
        ik_success=~failed,
        ik_branch_discontinuity=branch,
        joint_limit_violation=violations,
        self_collision_flag=collision,
        cross_arm_collision_flag=cross,
        task_start_frame=np.asarray(task_start),
        fps=np.asarray(latest.FPS),
        arm_joint_names=info["joint_names"],
    )

    a.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = a.output.with_name("g1_episode49_consensus_relative_report.json")
    if a.execute:
        tmp = a.output.with_suffix(a.output.suffix+".incomplete")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **payload)
        os.replace(tmp, a.output)
    else:
        report["dry_run"] = True
    tmp_report = report_path.with_suffix(".json.incomplete")
    tmp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp_report, report_path)
    print(json.dumps(report, indent=2))
    return 0 if safety else 2


if __name__ == "__main__":
    raise SystemExit(main())
