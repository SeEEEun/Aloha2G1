#!/usr/bin/env python3
"""Generate the current-layout Episode-49 SIMULATION-ONLY diagnostic candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import retarget_episode49_optimized_action_to_g1 as core
from task_frame_registration import T, qmat

OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
BASELINE = ROOT / "converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz"
REG = ROOT / "configs/magsafe_task_frame_registration.sim.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
POSES = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json"
SCENE_USD = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
TOOL_AXES = ROOT / "configs/aloha_tool_axes_calibration.sim.json"
SEMANTICS = ROOT / "configs/magsafe_object_semantic_frames.sim.json"
PRIMITIVES = ROOT / "configs/dex3_magsafe_grasp_primitives.sim.json"
PHASES = ROOT / "outputs/magsafe_gripper_phases.csv"
ARM_OUT = OUT / "g1_arm_scene_registered_trajectory.npz"
FULL_OUT = OUT / "g1_arm_dex3_scene_registered_trajectory.npz"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name: str, data: dict) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def transform_points(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def branch_flags(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(np.diff(q, axis=0), axis=1)
    flags = np.zeros(len(q), bool)
    for t in range(1, len(q)):
        local = np.median(norm[max(0, t - 10):min(len(norm), t + 9)])
        flags[t] = norm[t - 1] > max(.15, 8 * max(local, 1e-5))
    return flags


def scalar(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    return {"mean": float(np.mean(x)), "max": float(np.max(x, initial=0)),
            "p95": float(np.percentile(x, 95))}


def gate_audit() -> dict:
    timeline = json.loads(TIMELINE.read_text())
    axes = json.loads(TOOL_AXES.read_text())
    semantic = json.loads(SEMANTICS.read_text())
    timeline_events = timeline.get("events", [])
    ordered = all(a["frame"] <= b["frame"] for a, b in zip(
        sorted(timeline_events, key=lambda x: x["frame"]),
        sorted(timeline_events, key=lambda x: x["frame"])[1:]))
    return {
        "mode": "DIAGNOSTIC",
        "strict_final_all_gates_required": True,
        "gates": {
            "registration": {"required_key": "status", "required_status": "G1_SCENE_REGISTRATION_APPROVED",
                "status": "APPROVED_BY_USER", "satisfied_for_diagnostic": True,
                "satisfied_for_legacy_strict_gate": (ROOT/"outputs/task_frame_registration/g1_scene_registration.approved.json").exists(), "evidence": str(REG)},
            "semantic_frames": {"required_key": "status", "required_status": "OBJECT_SEMANTIC_FRAMES_APPROVED",
                "status": semantic.get("status"), "satisfied": False,
                "human_decision_required": "grasp pose/tolerances and final accessory target"},
            "aloha_tool_axes": {"required_key": "status", "required_status": "ALOHA_TOOL_AXES_APPROVED",
                "status": axes.get("status"), "satisfied": False,
                "human_decision_required": "manual video approval"},
            "timeline": {"required_key": "status", "required_status": "EPISODE49_TIMELINE_APPROVED",
                "artifact_status": timeline.get("status"), "manual_sources_only": all(e.get("source") == "manual_video_review" for e in timeline_events),
                "frame_count_matches": timeline.get("frame_range") == [0, 989], "event_order_numeric": ordered,
                "satisfied_by_existing_human_artifact_for_diagnostic": True,
                "satisfied_for_legacy_strict_gate": timeline.get("status") == "EPISODE49_TIMELINE_APPROVED"},
        },
        "unresolved_gates_preserved": ["semantic_frames", "aloha_tool_axes"],
        "final_candidate_selection_allowed": False,
    }


def collision_breakdown(full_path: Path) -> dict:
    with np.load(full_path, allow_pickle=False) as z:
        full = z["full_qpos"].astype(float)
    model = mujoco.MjModel.from_xml_path(str(core.G1_XML)); data = mujoco.MjData(model)
    categories = {k: set() for k in ("arm_torso", "arm_arm", "wrist_palm_torso", "hand_hand",
                                      "same_hand_wrist_finger_placeholder", "hand_finger_placeholder_torso",
                                      "object_contact", "other")}
    frames = {k: set() for k in categories}
    for t, q in enumerate(full):
        data.qpos[:] = q; data.qvel[:] = 0; mujoco.mj_forward(model, data)
        for c in data.contact:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "world"
                     for g in (c.geom1, c.geom2)]
            a, b = names; joined = "|".join(sorted(names)); both = a + " " + b
            sides = ({"left" if "left_" in a else "right" if "right_" in a else ""} |
                     {"left" if "left_" in b else "right" if "right_" in b else ""}) - {""}
            if "torso" in both and any(x in both for x in ("wrist", "hand_palm")): cat = "wrist_palm_torso"
            elif "torso" in both and "hand_" in both: cat = "hand_finger_placeholder_torso"
            elif "torso" in both and any(x in both for x in ("shoulder", "elbow", "wrist")): cat = "arm_torso"
            elif sides == {"left", "right"} and "hand" in both: cat = "hand_hand"
            elif sides == {"left", "right"}: cat = "arm_arm"
            elif (("wrist" in a and "hand_" in b) or ("wrist" in b and "hand_" in a)) and len(sides) == 1:
                cat = "same_hand_wrist_finger_placeholder"
            elif any(x in both.lower() for x in ("phone", "accessory", "charger")): cat = "object_contact"
            else: cat = "other"
            categories[cat].add(joined); frames[cat].add(t)
    return {"model": "object-free MuJoCo G1 model", "physics_task_success_evaluated": False,
            "categories": {k: {"frame_count": len(frames[k]), "pairs": sorted(categories[k])} for k in categories},
            "placeholder_contact_policy": "reported separately; neither whole-arm auto-fail nor real-safety evidence"}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--iterations", type=int, default=8); args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame_graph = ROOT / "outputs/task_frame_registration/frame_graph.json"
    paths = [SOURCE, BASELINE, REG, LAYOUT, POSES, SCENE_USD, TIMELINE, TOOL_AXES, SEMANTICS, PRIMITIVES, PHASES, frame_graph]
    for p in paths:
        if not p.exists(): raise FileNotFoundError(p)
    raw, ts = core.load_source(SOURCE, None)
    with np.load(SOURCE, allow_pickle=False) as z: source_fps = float(z["fps"]) if "fps" in z else 30.0
    if source_fps != 30.0: raise RuntimeError(f"source fps mismatch: {source_fps}")
    registration = json.loads(REG.read_text()); layout = json.loads(LAYOUT.read_text()); poses = json.loads(POSES.read_text())
    Tsg = np.asarray(registration["T_scene_from_g1_base"], float)
    Tgs = np.linalg.inv(Tsg)
    Tsa = T(qmat(poses["stationary_aloha"]["orientation_wxyz"]),
            poses["stationary_aloha"]["position_xyz_m"])
    amodel, _ = core.aloha.load_validated_model(core.ALOHA_XML)
    aq, clipped = core.aloha.mapped_qpos(raw); fk = core.aloha.fk(amodel, aq)
    aloha_l_scene = transform_points(Tsa, fk["left_position_m"])
    aloha_r_scene = transform_points(Tsa, fk["right_position_m"])
    info = core.ik.validate_model(core.G1_XML)
    with np.load(BASELINE, allow_pickle=False) as z:
        warm = z["g1_arm_joint_trajectory"].astype(float)
        nominal = z["g1_start_arm_q"].astype(float)
    d = mujoco.MjData(info["model"]); s0 = core.frame_state(info, d, warm[0])
    anchor_l_scene = transform_points(Tsg, s0["left_pos"][None])[0]
    anchor_r_scene = transform_points(Tsg, s0["right_pos"][None])[0]
    # Source-relative diagnostic mapping. No unapproved semantic grasp/placement XYZ is introduced.
    scale = core.SCALE
    target_l_scene = anchor_l_scene + scale * (aloha_l_scene - aloha_l_scene[0])
    target_r_scene = anchor_r_scene + scale * (aloha_r_scene - aloha_r_scene[0])
    target_l = transform_points(Tgs, target_l_scene); target_r = transform_points(Tgs, target_r_scene)
    identity = np.repeat(np.eye(3)[None], len(raw), axis=0)
    target = {"lp": target_l, "rp": target_r, "lr": identity, "rr": identity}
    q = core.temporal_solve(info, target, warm, nominal, 0.0, args.iterations)
    achieved = core.evaluate(info, target, q); branch = branch_flags(q)
    limits = info["joint_limits"]; violations = (q < limits[:, 0] - 1e-9) | (q > limits[:, 1] + 1e-9)
    le, re = achieved["le"], achieved["re"]; success = (le <= .005) & (re <= .005)
    step = np.abs(np.diff(q, axis=0)); vel = step * 30; acc = np.abs(np.diff(q, n=2, axis=0)) * 900
    baseline_step = np.abs(np.diff(warm, axis=0)); baseline_vel = baseline_step * 30
    baseline_acc = np.abs(np.diff(warm, n=2, axis=0)) * 900
    hashes = {str(p): sha(p) for p in paths}
    phone = [(layout["phone"]["bottom_left_xy"][0] + layout["phone"]["bottom_right_xy"][0]) / 2,
             layout["phone"]["bottom_left_xy"][1], layout["table"]["surface_height"] + layout["phone"]["size_landscape_xyz"][2] / 2]
    charger = [*layout["charger"]["center_xy"]]
    root = Tsg[:3, 3].tolist()
    input_audit = {"status": "PASS", "optimized_action_shape": list(raw.shape), "fps": source_fps,
        "finite": bool(np.isfinite(raw).all()), "source_sha256": hashes[str(SOURCE)], "sole_action_key": "optimized_action",
        "baseline_use": "warm_start_and_first_wrist_anchor_only; not target source or copied output", "stale_tools_excluded": {
            "tools/measure_and_register_g1_magsafe_layout.py": "obsolete +0.20 m measurement protocol",
            "tools/audit_g1_existing_magsafe_scene.py": "stale magnetic_scene_v2 source",
            "tools/run_g1_world_task_retargeting_and_render.py": "stale magnetic_scene_v2 source"}}
    dump("input_audit.json", input_audit)
    current = {"status": "APPROVED_BY_USER", "gate": "current_scene_registration", "simulation_only": True,
        "approved_values": {"phone_y_m": phone[1], "charger_y_m": charger[1], "g1_root_forward_offset_m": .05},
        "phone_center_scene_m": phone, "charger_root_xy_m": charger, "g1_root_position_m": root,
        "T_scene_from_g1_base": Tsg.tolist(), "validation": registration["validation"],
        "evidence": [{"path": str(p), "sha256": hashes[str(p)]} for p in (LAYOUT, REG, frame_graph, SCENE_USD)],
        "evidence_note": "user visual confirmation in Isaac Lab supplied in this task"}
    dump("current_scene_registration.json", current); dump("approval_gate_audit.json", gate_audit())
    diagnostic = {"status": "DIAGNOSTIC_ONLY", "candidate": "NOT_FINAL", "physics": "NOT_PHYSICS_APPROVED",
        "real_robot": "NOT_REAL_ROBOT_APPROVED", "real_robot_command_allowed": False,
        "classification": "DIAGNOSTIC_CANDIDATE_READY" if float(success.mean()) >= .99 and not violations.any() and not branch.any() else "BLOCKED_IK_SUCCESS_RATE",
        "selected": False, "authoritative_for_real_robot": False, "next_gate": "KINEMATIC_VISUAL_APPROVAL"}
    dump("diagnostic_status.json", diagnostic)
    common = dict(optimized_action=raw.astype(np.float32), source_timestamp=ts, fps=np.asarray(30.0),
        g1_target_left_position=target_l, g1_target_right_position=target_r,
        g1_target_left_position_scene=target_l_scene, g1_target_right_position_scene=target_r_scene,
        g1_achieved_left_position=achieved["lp"], g1_achieved_right_position=achieved["rp"],
        g1_arm_joint_trajectory=q, arm_joint_names=info["joint_names"], g1_start_arm_q=nominal,
        scene_layout_hash=np.asarray(hashes[str(LAYOUT)]), task_registration_hash=np.asarray(hashes[str(REG)]),
        g1_root_position=np.asarray(root), root_forward_offset_m=np.asarray(.05), diagnostic_only=np.asarray(True),
        real_robot_command_allowed=np.asarray(False), authoritative_for_real_robot=np.asarray(False),
        orientation_weight=np.asarray(0.0), ik_success=success, joint_limit_violation=violations,
        ik_branch_discontinuity=branch)
    np.savez_compressed(OUT / "scene_registered_targets.npz", **common)
    np.savez_compressed(ARM_OUT, **common)
    metrics = {"frames": len(q), "orientation_weight": 0.0, "solver_constants_changed": False,
        "solver_constants_source": str(Path(core.__file__).resolve()), "simultaneous_5mm_success_rate": float(success.mean()),
        "left_error_m": scalar(le), "right_error_m": scalar(re), "nan_inf_count": int(q.size - np.isfinite(q).sum()),
        "joint_limit_violation_count": int(violations.sum()), "branch_discontinuity_count": int(branch.sum()),
        "trajectory": {"max_joint_step_rad": float(step.max(initial=0)), "max_velocity_rad_s": float(vel.max(initial=0)), "max_acceleration_rad_s2": float(acc.max(initial=0))},
        "validated_baseline_comparison": {"max_joint_step_rad": float(baseline_step.max(initial=0)), "max_velocity_rad_s": float(baseline_vel.max(initial=0)), "max_acceleration_rad_s2": float(baseline_acc.max(initial=0)), "q_identical": bool(np.array_equal(q, warm))}}
    dump("ik_metrics.json", metrics)
    dump("kinematic_validation.json", {"status": "PASS" if np.isfinite(q).all() and not violations.any() and not branch.any() else "FAIL",
        "frames": len(q), "finite": bool(np.isfinite(q).all()), "joint_order": info["joint_names"].tolist(),
        "target_and_trajectory_shapes": {"left_target": list(target_l.shape), "right_target": list(target_r.shape), "arm": list(q.shape)}})
    subprocess.run([sys.executable, str(ROOT / "tools/compose_g1_arm_dex3_trajectory.py"), "--arm-trajectory", str(ARM_OUT),
                    "--phases", str(PHASES), "--primitives", str(PRIMITIVES), "--output", str(FULL_OUT)], check=True)
    dump("collision_breakdown.json", collision_breakdown(FULL_OUT))
    events = {e["event"]: e["frame"] for e in json.loads(TIMELINE.read_text()).get("events", [])}
    requested = ["left_phone_grasp_start", "right_accessory_grasp_start", "accessory_removed", "phone_move_to_charger_start",
                 "phone_charger_attachment_complete", "left_phone_release_complete", "right_accessory_place_on_table_start", "accessory_placed_on_table_complete"]
    event_metrics = {"tolerances": "NOT_APPROVED_NO_PASS_FAIL", "accessory_final_target": "SEMANTIC_TARGET_NOT_APPROVED",
        "right_accessory_hold_rule": "must persist beyond phone_charger_attachment_complete",
        "events": {name: ({"frame": events[name], "left_target_scene_m": target_l_scene[events[name]].tolist(),
                            "right_target_scene_m": target_r_scene[events[name]].tolist(),
                            "left_right_distance_m": float(np.linalg.norm(target_r_scene[events[name]]-target_l_scene[events[name]]))}
                           if name in events else {"status": "NOT_PRESENT_IN_APPROVED_ARTIFACT_NO_FRAME_INVENTED"}) for name in requested}}
    rows = list(csv.DictReader(PHASES.open())); attach = events.get("phone_charger_attachment_complete")
    event_metrics["right_phase_at_and_after_phone_attachment"] = rows[attach]["right_phase"] if attach is not None else "UNKNOWN"
    dump("task_event_metrics.json", event_metrics)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "status": diagnostic["classification"],
        "simulation_only": True, "no_dds": True, "no_publisher": True, "no_commands": True,
        "authoritative_scene": str(SCENE_USD), "hashes": hashes, "outputs": [p.name for p in OUT.iterdir()],
        "stale_reference_policy": input_audit["stale_tools_excluded"], "semantic_values_invented": False,
        "timeline_modified": False, "physics": "BLOCKED_PENDING_KINEMATIC_VISUAL_APPROVAL", "real_robot": "BLOCKED",
        "dex3": "SIM_PRIMITIVES_ONLY"}
    dump("run_manifest.json", manifest)
    commands = f'''#!/usr/bin/env bash\n# SIMULATION ONLY; no DDS, publisher, or real robot command.\n/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/replay_magsafe_g1_trajectory.py --arm-input {ARM_OUT} --full-input {FULL_OUT} --mode kinematic-full --scene-mode kinematic --root-forward-offset-m 0.05 --speed 0.25 --camera overview\n'''
    (OUT / "commands.sh").write_text(commands); os.chmod(OUT / "commands.sh", 0o755)
    report = f"""# Episode 49 current-layout scene-registered diagnostic\n\nDIAGNOSTIC_ONLY / NOT_FINAL / NOT_PHYSICS_APPROVED / NOT_REAL_ROBOT_APPROVED\n\n- Classification: `{diagnostic['classification']}`\n- 5 mm simultaneous IK success: `{success.mean():.6f}`\n- Left error mean/max: `{le.mean():.6f}` / `{le.max():.6f}` m\n- Right error mean/max: `{re.mean():.6f}` / `{re.max():.6f}` m\n- Joint-limit violations: `{violations.sum()}`; branch discontinuities: `{branch.sum()}`\n- Accessory final table target: `SEMANTIC_TARGET_NOT_APPROVED`\n- Physics: blocked pending 0.25x kinematic visual approval\n\nSIMULATION ONLY — NO REAL ROBOT COMMANDS — NO DDS OR PUBLISHER\nREAL G1 SAFETY NOT_PERFORMED\n"""
    (OUT / "report.md").write_text(report); (OUT / "report").mkdir(exist_ok=True)
    (OUT / "report/index.html").write_text("<meta charset='utf-8'><pre>" + report + "</pre>")
    dump("run_manifest.json", manifest | {"outputs": sorted(p.name for p in OUT.iterdir())})
    print(json.dumps({"status": diagnostic["classification"], "output": str(OUT), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
