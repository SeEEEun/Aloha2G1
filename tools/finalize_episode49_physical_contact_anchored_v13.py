#!/usr/bin/env python3
"""Finalize the fail-closed v13 physical-contact audit and visual evidence.

No G1 target or IK is solved here.  Existing v12 renderfix frames are used only
as immutable warm-start mesh-motion provenance and are visibly labelled as a
failed v13 contact diagnostic.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from pxr import Usd
from scipy.optimize import differential_evolution

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT / "tools")]

import build_episode49_physical_contact_anchored_v13 as build
import retarget_episode49_optimized_action_to_g1 as core

OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_physical_contact_anchored_v13"
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
RENDERFIX = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12_renderfix"
KEYS = [169, 216, 319, 334, 523]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height,duration",
        "-of", "json", str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    return {
        "decoded_frame_count": int(stream["nb_read_frames"]),
        "frame_rate": stream["r_frame_rate"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration_s": float(stream.get("duration", 0.0)),
        "sha256": sha(path),
    }


def remux_failure_video(source: Path, output: Path, label: str) -> dict:
    metadata = {
        "status": "FAILED_DIAGNOSTIC_CANDIDATE_NOT_APPROVED",
        "content": "v12 warm-start G1 articulation mesh; no v13 physical-contact IK exists",
        "source_video": str(source.resolve()),
        "source_video_sha256": sha(source),
        "v12_trajectory": str((V12 / f"position_only_{label}_arm_trajectory.npz").resolve()),
        "v12_trajectory_sha256": sha(V12 / f"position_only_{label}_arm_trajectory.npz"),
        "active_scene_sha256": sha(build.ACTIVE_SCENE),
        "frame_count": 990,
        "fps": 7.5,
        "no_v13_ik": True,
    }
    temporary = output.with_suffix(".incomplete.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-map", "0",
        "-c", "copy", "-metadata", f"title=V13 CONTACT GATE FAILED - V12 {label.upper()} WARM START",
        "-metadata", "comment=" + json.dumps(metadata, separators=(",", ":")),
        "-movflags", "+faststart", str(temporary),
    ], check=True)
    os.replace(temporary, output)
    result = probe(output)
    result.update(metadata)
    if result["decoded_frame_count"] != 990:
        raise RuntimeError(f"video frame count mismatch: {output}: {result}")
    return result


def charger_reach_bound() -> dict:
    """Compute an optimistic triangle-inequality lower bound for left-A gap."""
    info = core.ik.validate_model(core.G1_XML)
    stage = Usd.Stage.Open(str(build.ACTIVE_SCENE))
    _, runtime = build.hand_geometry_audit(stage, info)
    spec = runtime["left_A"]
    model = info["model"]
    data = mujoco.MjData(model)
    distal = build.body_id(model, spec["link"])
    shoulder = build.body_id(model, "left_shoulder_pitch_link")
    limits = np.vstack((info["joint_limits"][:7], spec["limits"]))

    def state(x):
        data.qpos[:] = info["stand_qpos"]
        arm = info["stand_arm_q"].copy()
        arm[:7] = x[:7]
        data.qpos[info["arm_qpos_ids"]] = arm
        data.qpos[spec["qpos"]] = x[7:]
        mujoco.mj_forward(model, data)
        rotation = data.xmat[distal].reshape(3, 3)
        contact = data.xpos[distal] + rotation @ spec["proxy"]
        return contact, data.xpos[shoulder].copy()

    result = differential_evolution(
        lambda x: -float(np.linalg.norm(state(x)[0] - state(x)[1])),
        [tuple(row) for row in limits], seed=61, maxiter=500, popsize=18,
        tol=1e-10, polish=True, workers=1, updating="immediate",
    )
    contact, shoulder_model = state(result.x)
    maximum = float(np.linalg.norm(contact - shoulder_model))
    env = json.loads((OUT / "environment_audit.json").read_text())
    pad = np.asarray(env["charger_pad_pose"], float)
    pad_vertical = build.normalize(pad[:3, 1])
    pad_normal = build.normalize(pad[:3, 2])
    phone_rotation = np.column_stack((pad_vertical, -pad_normal, np.cross(pad_vertical, -pad_normal)))
    if np.linalg.det(phone_rotation) < 0:
        phone_rotation[:, 2] *= -1
    phone_pose = build.make_transform(phone_rotation, pad[:3, 3])
    phone_dims = np.asarray(json.loads(build.LAYOUT.read_text())["phone"]["size_landscape_xyz"], float)
    shoulder_scene = build.model_to_scene(shoulder_model)
    local = phone_pose[:3, :3].T @ (shoulder_scene - phone_pose[:3, 3])
    nearest_local = np.clip(local, -0.5 * phone_dims, 0.5 * phone_dims)
    nearest_scene = phone_pose[:3, 3] + phone_pose[:3, :3] @ nearest_local
    shoulder_to_surface = float(np.linalg.norm(nearest_scene - shoulder_scene))
    lower_bound = shoulder_to_surface - maximum
    return {
        "status": "ACTIVE_KINEMATIC_REACH_BOUND_COMPUTED",
        "active_left_shoulder_scene_m": shoulder_scene,
        "desired_phone_nearest_surface_scene_m": nearest_scene,
        "shoulder_to_nearest_phone_surface_m": shoulder_to_surface,
        "maximum_shoulder_to_left_A_collision_cap_m": maximum,
        "optimistic_triangle_lower_bound_gap_m": lower_bound,
        "optimistic_triangle_lower_bound_gap_mm": lower_bound * 1000.0,
        "required_gate_mm": 5.0,
        "gate_possible_even_under_optimistic_direction_free_bound": lower_bound <= .005,
        "max_reach_optimizer_success": bool(result.success),
        "max_reach_optimizer_message": str(result.message),
        "maximizing_arm_and_thumb_q_rad": result.x,
        "interpretation": "This bound ignores direction coupling and is optimistic; the full directional active-FK minimum gap is larger.",
    }


def ring_volume_gap(point: np.ndarray, pose: np.ndarray, inner: float, outer: float, depth: float) -> float:
    local = pose[:3, :3].T @ (point - pose[:3, 3])
    radial = float(np.hypot(local[0], local[2]))
    radial_gap = max(inner - radial, radial - outer, 0.0)
    depth_gap = max(abs(float(local[1])) - 0.5 * depth, 0.0)
    return float(np.hypot(radial_gap, depth_gap))


def write_gap_csv() -> list[dict]:
    with np.load(V12 / "position_only_exact_arm_trajectory.npz", allow_pickle=False) as payload:
        left = payload["achieved_left_position_scene"].astype(float)
        right = payload["achieved_right_position_scene"].astype(float)
        left_r = payload["achieved_left_rotation_scene"].astype(float)
        right_r = payload["achieved_right_rotation_scene"].astype(float)
    env = json.loads((OUT / "environment_audit.json").read_text())
    layout = json.loads(build.LAYOUT.read_text())
    phone = np.asarray(env["phone_pose"], float)
    pad = np.asarray(env["charger_pad_pose"], float)
    pad_vertical, pad_normal = build.normalize(pad[:3, 1]), build.normalize(pad[:3, 2])
    desired_r = np.column_stack((pad_vertical, -pad_normal, np.cross(pad_vertical, -pad_normal)))
    if np.linalg.det(desired_r) < 0:
        desired_r[:, 2] *= -1
    charger_phone = build.make_transform(desired_r, pad[:3, 3])
    phone_dims = np.asarray(layout["phone"]["size_landscape_xyz"], float)
    accessory_npz = np.load(V12 / "target_accessory_pose_trajectory.npz", allow_pickle=False)
    accessory = accessory_npz["pose"].astype(float)
    inner = 0.5 * float(layout["accessory"]["main_inner_diameter"])
    outer = 0.5 * float(layout["accessory"]["main_outer_diameter"])
    depth = float(layout["accessory"]["main_depth"])
    reach = json.loads((OUT / "charger_active_fk_contact_reach_audit.json").read_text())
    old = json.loads((OUT / "v12_physical_anchor_failure_audit.json").read_text())

    rows = []
    for frame, event, palm, rotation, object_pose in (
        (169, "left_phone", left[169], left_r[169], phone),
        (319, "right_accessory", right[319], right_r[319], accessory[319]),
        (523, "left_charger", left[523], left_r[523], charger_phone),
    ):
        wrist = palm - rotation @ build.PALM_OFFSET["left" if event.startswith("left") else "right"]
        if event == "right_accessory":
            palm_gap = ring_volume_gap(palm, object_pose, inner, outer, depth)
            wrist_gap = ring_volume_gap(wrist, object_pose, inner, outer, depth)
            best_finger = old["exact_right_C_ring_gap_mm"]
        else:
            palm_gap = abs(build.nearest_box_surface(palm, object_pose, phone_dims)[0])
            wrist_gap = abs(build.nearest_box_surface(wrist, object_pose, phone_dims)[0])
            best_finger = (
                reach["left_A"]["minimum_absolute_surface_gap_m"] * 1000.0
                if frame == 523 else np.nan
            )
        rows.append({
            "action_index": frame, "event": event,
            "v12_achieved_palm_x_m": palm[0], "v12_achieved_palm_y_m": palm[1], "v12_achieved_palm_z_m": palm[2],
            "v12_palm_to_physical_surface_gap_mm": palm_gap * 1000.0,
            "v12_wrist_to_physical_surface_gap_mm": wrist_gap * 1000.0,
            "best_feasible_contact_proxy_gap_mm": best_finger,
            "v12_target_achieved_error_is_contact_proof": False,
        })
    with (OUT / "v12_object_gap_keyframes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def plot_vectors() -> None:
    failure = json.loads((OUT / "v12_physical_anchor_failure_audit.json").read_text())
    selected = json.loads((OUT / "selected_physical_carrier_anchors.json").read_text())
    charger = json.loads((OUT / "charger_active_fk_contact_reach_audit.json").read_text())
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    panels = [
        ("action 169: phone", np.asarray(failure["direct_geometry_rows"][0]["old_palm_position_m"]),
         np.asarray(selected["left_phone_contact_start"]["palm_position_m"]),
         np.asarray(selected["left_phone_contact_start"]["A_target_surface_m"]),
         np.asarray(selected["left_phone_contact_start"]["A_contact_proxy_m"])),
        ("action 319: accessory", np.asarray(failure["reported_old_anchors"]["right"]["accessory_grasp"]["position_m"]),
         np.asarray(selected["right_accessory"]["palm_position_action_319"]),
         np.asarray(selected["right_accessory"]["accessory_pose_action_319"])[:3, 3],
         np.asarray(selected["right_accessory"]["accessory_pose_action_319"])[:3, 3]),
        ("action 523: charger phone", np.asarray(failure["direct_geometry_rows"][1]["old_palm_position_m"]),
         np.asarray(selected["left_stable_carrier_from_action_216"]["palm_position_m"]),
         np.asarray(charger["left_A"]["nearest_phone_surface_position_m"]),
         np.asarray(charger["left_A"]["contact_proxy_scene_position_m"])),
    ]
    for index, (title, old, carrier, surface, contact) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        for point, color, label in ((old, "tab:red", "v12 palm"), (carrier, "tab:blue", "v13 best carrier"),
                                    (surface, "tab:green", "physical surface"), (contact, "tab:orange", "contact proxy")):
            ax.scatter(*point, s=45, c=color, label=label)
        ax.plot(*np.vstack((old, carrier)).T, c="tab:purple", lw=2, label="anchor correction")
        ax.plot(*np.vstack((contact, surface)).T, c="black", lw=2, label="physical gap")
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.legend(fontsize=7)
    fig.suptitle("V12 target-achieved agreement is not physical contact | V13 best-failure geometry")
    fig.savefig(OUT / "v12_anchor_object_vectors.png", dpi=180)
    plt.close(fig)


def make_sheet(view: str, output: Path, closeup=False) -> None:
    source = np.load(RENDERFIX / "rendered_keyframes_nullspace_review.npz", allow_pickle=False)
    cells = []
    reach = json.loads((OUT / "charger_active_fk_contact_reach_audit.json").read_text())
    contact = json.loads((OUT / "physical_contact_reachability_metrics.json").read_text())
    descriptions = {
        169: f"phone A/B={contact['left_action_169_A_gap_mm']:.2f}/{contact['left_action_169_B_gap_mm']:.2f} mm PASS",
        216: "carrier acquisition endpoint; no v13 IK",
        319: f"right-C local ring={contact['right_action_319_C_ring_gap_mm']:.3f} mm (local only)",
        334: "source removal endpoint; best-failure target only",
        523: f"charger active-FK A/B={reach['left_A']['minimum_absolute_surface_gap_m']*1000:.2f}/{reach['left_B']['minimum_absolute_surface_gap_m']*1000:.3f} mm FAIL",
    }
    for frame in KEYS:
        image = cv2.cvtColor(source[f"rgb_{view}_{frame}"], cv2.COLOR_RGB2BGR)
        if closeup:
            image = image[35:520, 265:940]
        image = cv2.resize(image, (480, 310 if closeup else 270), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((image.shape[0] + 72, image.shape[1], 3), np.uint8)
        canvas[:image.shape[0]] = image
        cv2.putText(canvas, f"action {frame} | {view}", (9, image.shape[0] + 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, descriptions[frame], (9, image.shape[0] + 47), cv2.FONT_HERSHEY_SIMPLEX, .40, (40, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "V12 MESH + V13 GEOMETRY DIAGNOSTIC | NOT APPROVED", (9, image.shape[0] + 66), cv2.FONT_HERSHEY_SIMPLEX, .35, (80, 80, 255), 1, cv2.LINE_AA)
        cells.append(canvas)
    sheet = np.hstack(cells)
    cv2.imwrite(str(output), sheet)


def write_reports(video_audit: dict, reach_bound: dict, rows: list[dict]) -> None:
    contact = json.loads((OUT / "physical_contact_reachability_metrics.json").read_text())
    fidelity = json.loads((OUT / "aloha_fidelity_metrics.json").read_text())
    environment = json.loads((OUT / "environment_audit.json").read_text())
    static_reach = json.loads((OUT / "g1_arm_static_reach_audit.json").read_text())
    input_audit = json.loads((OUT / "input_hash_audit.json").read_text())
    backups = sorted((OUT / "backups").glob("*"))
    report = f"""# Episode 49 physical-contact-anchored v13

## Final status

- **BLOCKED_CHARGER_PHYSICAL_CARRIER**
- **BLOCKED_ALOHA_FIDELITY** (best-failure path correlation {fidelity['minimum_major_phase_fidelity']['path_shape']:.6f} < 0.90)
- **POSITION_ONLY_IK_NOT_RUN_PRE_IK_GATE_FAILED**
- **DEX3_TRAJECTORY_NOT_APPLIED / KINEMATIC DIAGNOSTIC ONLY / NOT APPROVED**

The v12 numeric IK is retained only as a warm-start and actual-mesh motion diagnostic. It is not accepted as physical task success.

## Immutable inputs and scene

- Optimized action, timestamps, +7 alignment, timeline, scale 0.42, phase library, G1 root and object layout were not changed.
- Active G1 root: `{environment['g1_root_expected']}`; forward offset: `{environment['g1_root_forward_offset_m']}` m.
- Scene byte-identical before/after: `{environment['scene_byte_identical']}`.
- Timestamped backup: `{backups[-1] if backups else 'missing'}`.

## Physical carrier findings

- Action 169 phone contact-start: A={contact['left_action_169_A_gap_mm']:.3f} mm, B={contact['left_action_169_B_gap_mm']:.3f} mm — PASS.
- Action 319 right-C local ring carrier: {contact['right_action_319_C_ring_gap_mm']:.6f} mm — local geometry PASS, not a full-task approval.
- Action 523 arbitrary-SE(3) best carrier: A={contact['left_action_523_A_gap_mm']:.3f} mm, B={contact['left_action_523_B_gap_mm']:.3f} mm; this pose is not simultaneously realizable by the active arm/contact chain.
- Action 523 full active-FK global minimum found: A={reach_bound['directional_active_fk_min_gap_mm']:.3f} mm, B={reach_bound['directional_active_fk_B_min_gap_mm']:.6f} mm.
- Optimistic direction-free lower bound for A: {reach_bound['optimistic_triangle_lower_bound_gap_mm']:.3f} mm > 5 mm. Maximum shoulder→A contact-cap reach is {reach_bound['maximum_shoulder_to_left_A_collision_cap_m']:.6f} m, while shoulder→nearest desired-phone surface is {reach_bound['shoulder_to_nearest_phone_surface_m']:.6f} m.
- The phone center and normal can be defined exactly on the fixed pad (0 mm, 0 deg), but left-A cannot remain locally reachable there with the fixed root/layout.

## V12 physical failure audit

- V12 palm→phone surface at action 169: {rows[0]['v12_palm_to_physical_surface_gap_mm']:.3f} mm.
- V12 right palm→ring volume at action 319: {rows[1]['v12_palm_to_physical_surface_gap_mm']:.3f} mm; v12 future right-C proxy gap was {rows[1]['best_feasible_contact_proxy_gap_mm']:.3f} mm.
- V12 palm→desired charger-phone surface at action 523: {rows[2]['v12_palm_to_physical_surface_gap_mm']:.3f} mm.

## ALOHA best-failure fidelity

- Selected diagnostic residual: `{fidelity['selected_candidate']}`.
- Minimum major-phase path/speed/rotation correlations: {fidelity['minimum_major_phase_fidelity']['path_shape']:.6f} / {fidelity['minimum_major_phase_fidelity']['speed']:.6f} / {fidelity['minimum_major_phase_fidelity']['rotation_progress']:.6f}.
- No position-only IK was allowed because physical contact and ALOHA path gates did not both pass.

## Visual evidence

- Marker-free Exact warm-start mesh diagnostic: `{OUT/'g1_physical_position_exact_robot_only_FAILED_DIAGNOSTIC.mp4'}`
- Marker-free Nullspace warm-start mesh diagnostic: `{OUT/'g1_physical_position_nullspace_robot_only_FAILED_DIAGNOSTIC.mp4'}`
- Overview/side/contact sheets: `{OUT/'physical_anchor_contact_sheet_overview.png'}`, `{OUT/'physical_anchor_contact_sheet_side.png'}`, `{OUT/'physical_anchor_contact_sheet_closeup.png'}`
- Geometry vectors: `{OUT/'v12_anchor_object_vectors.png'}`

The videos contain the authoritative Isaac Lab scene and moving v12 articulation mesh only to prove the renderer path; they are explicitly not a v13 physical-contact trajectory.

## Single next decision

The fixed current root/layout makes the required action-523 left A+B contact gate infeasible. A new trajectory cannot legitimately pass without changing a currently immutable assumption (root/layout, contact semantics, or the 5 mm A+B requirement). No such change was made automatically.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    report_dir = OUT / "report"; report_dir.mkdir(exist_ok=True)
    body = html.escape(report)
    (report_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>v13 physical contact audit</title>"
        "<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;background:#111;color:#eee}pre{white-space:pre-wrap}</style>"
        f"<pre>{body}</pre>", encoding="utf-8",
    )
    dump(OUT / "ik_metrics.json", {
        "status": "POSITION_ONLY_IK_NOT_RUN_PRE_IK_GATE_FAILED",
        "blocking_state": "BLOCKED_CHARGER_PHYSICAL_CARRIER",
        "new_exact_trajectory_generated": False,
        "new_nullspace_trajectory_generated": False,
        "v12_q_used_only_as_warm_start_and_visual_failure_provenance": True,
    })
    dump(OUT / "collision_breakdown.json", {
        "status": "NOT_RUN_BECAUSE_POSITION_ONLY_IK_WAS_NOT_AUTHORIZED_BY_PRE_IK_GATE",
        "collision_pass_claimed": False,
    })
    dump(OUT / "isaaclab_visual_validation.json", {
        "status": "FAILED_DIAGNOSTIC_VISUALS_READY_NOT_APPROVED",
        "active_scene": str(build.ACTIVE_SCENE.resolve()),
        "active_scene_sha256": sha(build.ACTIVE_SCENE),
        "scene_hash_unchanged": environment["scene_byte_identical"],
        "renderfix_articulation_path": str((ROOT/'isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py').resolve()),
        "videos": video_audit,
        "actual_g1_mesh_moves": True,
        "v13_physical_ik_rendered": False,
        "reason": "pre-IK physical carrier gate failed",
    })


def commands() -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail
cd /home/jbnu/aloha_g1_dataset
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6

# V12 warm-start mesh only; v13 contact gate failed. NOT APPROVED.
# Exact overview GUI
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py --trajectory exact --mode review --cameras overview --gui --viz kit

# Nullspace overview GUI
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py --trajectory nullspace --mode review --cameras overview --gui --viz kit

# Nullspace side GUI
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py --trajectory nullspace --mode review --cameras side --gui --viz kit
"""
    path = OUT / "commands.sh"; path.write_text(text, encoding="utf-8"); path.chmod(0o755)


def manifest() -> None:
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and "backups" not in path.parts and path.name != "run_manifest.json":
            files[str(path.relative_to(OUT))] = {"size": path.stat().st_size, "sha256": sha(path)}
    dump(OUT / "run_manifest.json", {
        "method": build.METHOD,
        "final_status": ["BLOCKED_CHARGER_PHYSICAL_CARRIER", "BLOCKED_ALOHA_FIDELITY"],
        "pre_ik_gate_pass": False,
        "new_position_only_ik_generated": False,
        "gated_outputs_intentionally_absent": [
            "physical_position_exact_arm_trajectory.npz",
            "physical_position_nullspace_arm_trajectory.npz",
            "successful v13 Isaac Lab review videos",
        ],
        "no_orientation_optimization": True,
        "no_dex3_trajectory": True,
        "no_physics": True,
        "no_hardware_dds_publisher": True,
        "files": files,
    })


def test_results(video_audit: dict) -> None:
    input_audit = json.loads((OUT / "input_hash_audit.json").read_text())
    environment = json.loads((OUT / "environment_audit.json").read_text())
    source = np.load(build.SOURCE, allow_pickle=False)
    diagnostic = np.load(OUT / "physical_corrected_targets.npz", allow_pickle=False)
    registered = np.load(OUT / "globally_registered_aloha_targets.npz", allow_pickle=False)
    checks = {
        "optimized_action_shape_is_990x14": source["optimized_action"].shape == (990, 14),
        "optimized_action_unchanged": np.array_equal(source["optimized_action"], diagnostic["optimized_action"]),
        "timestamps_unchanged": np.array_equal(source["timestamp"], diagnostic["timestamps"]),
        "action_to_observation_lag_is_7": input_audit["action_to_observation_lag_frames"] == 7,
        "workspace_scale_is_0_42": float(registered["global_workspace_scale"]) == .42,
        "source_phase_library_hash_unchanged": sha(V12 / "aloha_phase_motion_library.npz") == "1d17663a0d02cb1fe608715c4b619c294b9528644766e47ccb464269231c97f3",
        "g1_root_offset_unchanged": environment["g1_root_forward_offset_m"] == .15,
        "active_scene_hashes_unchanged": environment["scene_byte_identical"],
        "no_new_exact_ik_trajectory": not (OUT / "physical_position_exact_arm_trajectory.npz").exists(),
        "no_new_nullspace_ik_trajectory": not (OUT / "physical_position_nullspace_arm_trajectory.npz").exists(),
        "exact_failure_video_has_990_frames": video_audit["exact"]["decoded_frame_count"] == 990,
        "nullspace_failure_video_has_990_frames": video_audit["nullspace"]["decoded_frame_count"] == 990,
        "dex3_geometry_not_scaled": True,
        "dex3_trajectory_not_generated": True,
        "orientation_optimization_not_run": True,
        "physics_not_run": True,
        "dds_publisher_hardware_not_used": True,
    }
    dump(OUT / "test_results.json", {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "semantic_result": "FAIL_CLOSED_PRE_IK_GATE_AS_REQUIRED",
        "physical_contact_success_claimed": False,
    })


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reach_bound = charger_reach_bound()
    directional = json.loads((OUT / "charger_active_fk_contact_reach_audit.json").read_text())
    reach_bound["directional_active_fk_min_gap_mm"] = directional["left_A"]["minimum_absolute_surface_gap_m"] * 1000.0
    reach_bound["directional_active_fk_B_min_gap_mm"] = directional["left_B"]["minimum_absolute_surface_gap_m"] * 1000.0
    reach_bound["final_gate_pass"] = False
    reach_bound["blocking_state"] = "BLOCKED_CHARGER_PHYSICAL_CARRIER"
    dump(OUT / "charger_physical_reachability_bound.json", reach_bound)
    contact_metrics = json.loads((OUT / "physical_contact_reachability_metrics.json").read_text())
    contact_metrics.update({
        "left_action_523_active_arm_and_A_global_minimum_gap_mm": reach_bound["directional_active_fk_min_gap_mm"],
        "left_action_523_active_arm_and_B_global_minimum_gap_mm": reach_bound["directional_active_fk_B_min_gap_mm"],
        "left_action_523_optimistic_direction_free_A_lower_bound_gap_mm": reach_bound["optimistic_triangle_lower_bound_gap_mm"],
        "arbitrary_SE3_carrier_gap_is_not_arm_reachability_proof": True,
        "physical_contact_success_claimed": False,
    })
    dump(OUT / "physical_contact_reachability_metrics.json", contact_metrics)
    rows = write_gap_csv()
    plot_vectors()
    make_sheet("overview", OUT / "physical_anchor_contact_sheet_overview.png")
    make_sheet("side", OUT / "physical_anchor_contact_sheet_side.png")
    make_sheet("overview", OUT / "physical_anchor_contact_sheet_closeup.png", closeup=True)
    video_audit = {
        "exact": remux_failure_video(
            RENDERFIX / "g1_exact_robot_only_motion_proof.mp4",
            OUT / "g1_physical_position_exact_robot_only_FAILED_DIAGNOSTIC.mp4", "exact",
        ),
        "nullspace": remux_failure_video(
            RENDERFIX / "g1_nullspace_robot_only_motion_proof.mp4",
            OUT / "g1_physical_position_nullspace_robot_only_FAILED_DIAGNOSTIC.mp4", "nullspace",
        ),
    }
    commands()
    write_reports(video_audit, reach_bound, rows)
    test_results(video_audit)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    manifest()
    print(json.dumps({
        "status": ["BLOCKED_CHARGER_PHYSICAL_CARRIER", "BLOCKED_ALOHA_FIDELITY"],
        "optimistic_left_A_lower_bound_gap_mm": reach_bound["optimistic_triangle_lower_bound_gap_mm"],
        "directional_left_A_min_gap_mm": reach_bound["directional_active_fk_min_gap_mm"],
        "new_ik_generated": False,
        "videos": {key: value["decoded_frame_count"] for key, value in video_audit.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
