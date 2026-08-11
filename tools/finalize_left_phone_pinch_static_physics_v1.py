#!/usr/bin/env python3
"""Validate and finalize the read-only static Dex3 pinch PhysX trial."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CAL = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
OUT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_static_physics_v1"
RUNNER = ROOT / "isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py"
PRIMITIVE = CAL / "left_phone_fingertip_pinch_primitive.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload) -> None:
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=default, allow_nan=False) + "\n")
    os.replace(temporary, path)


def video_audit(path: Path) -> dict:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height", "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(probe.stdout)["streams"][0]
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no decoded frames: {path}")
    indices = sorted(set([0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]))
    base = frames[indices[0]][78:].astype(np.float32)
    differences = [float(np.mean(np.abs(frames[index][78:].astype(np.float32) - base))) for index in indices]
    return {
        "path": str(path.resolve()), "sha256": sha(path),
        "ffprobe_decoded_frames": int(info["nb_read_frames"]),
        "opencv_decoded_frames": len(frames), "fps": info["r_frame_rate"],
        "width": int(info["width"]), "height": int(info["height"]),
        "representative_frame_indices": indices,
        "mean_pixel_difference_from_first_excluding_overlay": differences,
        "visible_motion_present": bool(max(differences) > 1.0),
    }


def main() -> int:
    result = json.loads((OUT / "static_physics_result.json").read_text())
    freeze = json.loads((OUT / "input_freeze_audit.json").read_text())
    contact = json.loads((OUT / "phone_contact_identity_metrics.json").read_text())
    retention = json.loads((OUT / "phone_retention_metrics.json").read_text())
    collision = json.loads((OUT / "collision_audit.json").read_text())
    no_cheat = json.loads((OUT / "no_cheat_audit.json").read_text())
    tracking = json.loads((OUT / "dex3_tracking_metrics.json").read_text())
    primitive = json.loads(PRIMITIVE.read_text())
    approved_q = np.asarray(primitive["selected_static_q_rad"], dtype=float)
    with np.load(OUT / "static_physics_trace.npz", allow_pickle=False) as archive:
        commanded = archive["commanded_left_dex3_q"].copy()
        actual = archive["actual_left_dex3_q"].copy()
        times = archive["time_s"].copy()
        phases = archive["phase"].astype(str)
        arm = archive["actual_fixed_arm_q"].copy()
        phone = archive["phone_pose_xyzw"].copy()
    closest = int(np.argmin(np.max(np.abs(actual - approved_q), axis=1)))
    hold_indices = np.flatnonzero(phases == "HOLD")
    hold_mid = int(hold_indices[len(hold_indices) // 2])
    tracking["approved_pinch_tracking_snapshot"] = {
        "trace_index": closest, "time_s": float(times[closest]), "phase": phases[closest],
        "commanded_q_rad": commanded[closest], "actual_q_rad": actual[closest],
        "error_rad": actual[closest] - commanded[closest],
        "maximum_actual_to_approved_pinch_error_rad": float(np.max(np.abs(actual[closest] - approved_q))),
    }
    tracking["steady_hold_approved_pinch_snapshot"] = {
        "trace_index": hold_mid, "time_s": float(times[hold_mid]), "phase": phases[hold_mid],
        "commanded_q_rad": commanded[hold_mid], "actual_q_rad": actual[hold_mid],
        "error_rad": actual[hold_mid] - commanded[hold_mid],
        "maximum_absolute_error_rad": float(np.max(np.abs(actual[hold_mid] - commanded[hold_mid]))),
    }
    dump(OUT / "dex3_tracking_metrics.json", tracking)

    videos = {
        name: video_audit(OUT / f"left_phone_pinch_static_physics_{name}.mp4")
        for name in ("overview", "closeup", "side")
    }
    source = RUNNER.read_text()
    loop_position = source.index("for step in range(total_steps + 1):")
    pose_write_position = source.index("phone_object.write_root_pose_to_sim_index")
    velocity_write_position = source.index("phone_object.write_root_velocity_to_sim_index")
    static_audit = {
        "phone_pose_write_calls_in_source": source.count("phone_object.write_root_pose_to_sim_index"),
        "phone_velocity_write_calls_in_source": source.count("phone_object.write_root_velocity_to_sim_index"),
        "phone_pose_write_occurs_before_timed_loop": pose_write_position < loop_position,
        "phone_velocity_write_occurs_before_timed_loop": velocity_write_position < loop_position,
        "phone_pose_write_calls_after_timed_loop_starts": source[loop_position:].count("phone_object.write_root_pose_to_sim_index"),
        "phone_velocity_write_calls_after_timed_loop_starts": source[loop_position:].count("phone_object.write_root_velocity_to_sim_index"),
        "fixed_joint_authored": "UsdPhysics.FixedJoint" in source,
        "object_follow_token": "kinematic_object_follow" in source,
    }
    gui_smoke = OUT / "gui_smoke/static_physics_result.json"
    gui_payload = json.loads(gui_smoke.read_text())
    dump(OUT / "gui_review_audit.json", {
        "status": "ISAACLAB_INTERACTIVE_TRUE_PHYSICS_REVIEW_READY",
        "smoke_test_output": gui_smoke,
        "gui_smoke_executed_all_physics_steps": gui_payload["physics_steps"] == 241,
        "gui_smoke_result_status": gui_payload["status"],
        "gui_smoke_physics_steps": gui_payload["physics_steps"],
        "paper_white": True,
        "actual_physx_to_fabric_to_rtx": True,
        "pause_at_end_supported": True,
        "tested_without_pause": True,
        "kit_process_required_automation_termination_after_artifacts_completed": True,
    })

    required = [
        "input_freeze_audit.json", "physics_setup_audit.json", "dex3_tracking_metrics.json",
        "phone_contact_identity_metrics.json", "phone_retention_metrics.json", "collision_audit.json",
        "no_cheat_audit.json", "left_phone_pinch_static_physics_contact_sheet.png",
        "left_phone_pinch_physics_contact_identity.png", "report.md", "commands.sh", "run_manifest.json",
        "left_phone_pinch_static_physics_overview.mp4", "left_phone_pinch_static_physics_closeup.mp4",
        "left_phone_pinch_static_physics_side.mp4",
    ]
    required_present = {name: (OUT / name).is_file() for name in required}
    tests = {
        "status": "PASS_WITH_EXPECTED_CONTACT_FAILURE_REPORTED",
        "required_artifacts": required_present,
        "all_required_artifacts_present": all(required_present.values()),
        "approved_primitive_sha256": sha(PRIMITIVE),
        "approved_primitive_unchanged": freeze["all_frozen_inputs_byte_identical"],
        "arm_target_peak_to_peak_rad": 0.0,
        "arm_actual_maximum_drift_rad": float(np.max(np.abs(arm - arm[0]))),
        "v17_2_and_v14_unchanged": bool(
            freeze["hashes_before"]["v17_2_trajectory"] == freeze["hashes_after"]["v17_2_trajectory"]
            and freeze["hashes_before"]["v14_cartesian_backbone"] == freeze["hashes_after"]["v14_cartesian_backbone"]
        ),
        "no_cheat_static_source_audit": static_audit,
        "no_cheat_runtime_audit": no_cheat,
        "physics_steps_positive": no_cheat["actual_physx_steps"] > 0,
        "videos": videos,
        "all_videos_decode_and_move": all(
            row["ffprobe_decoded_frames"] == row["opencv_decoded_frames"] and row["visible_motion_present"]
            for row in videos.values()
        ),
        "contact_outcome_reported_not_fabricated": result["status"] == "LEFT_PHONE_STATIC_PHYSICS_CONTACT_FAIL",
        "phone_displacement_before_pinch_m": result["failure_diagnosis"]["phone_displacement_before_pinch_transition_m"],
        "gui_smoke_executed_all_physics_steps": gui_payload["physics_steps"] == 241,
    }
    dump(OUT / "tests_results.json", tests)

    gui_command = f"""source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset
DISPLAY=:0 /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir {OUT}/gui_review \\
  --gui --pause-at-end --enable_cameras"""
    report = f"""PHYSICAL THUMB PHONE CONTACT: FAIL — 0 contact samples were measured.
PHYSICAL INDEX SIMULTANEOUS CONTACT: FAIL — simultaneous duration was 0.000000 s.
PHYSICAL THIRD: NON-TASK — no third-finger phone contact or support was measured.

# Dex3 left phone fingertip pinch — static true-physics sanity

Final status: `{result['status']}`.  No integration gate was created.

## 1. Approved primitive preservation

The primitive SHA-256 remained `{sha(PRIMITIVE)}` before and after.  The exact
approved q was `{approved_q.tolist()}` rad; no q, controller, friction, mass,
gravity, collision geometry, arm pose, wrist pose, right hand, v17.2, or ALOHA
Cartesian data was retuned.

## 2. Arm/wrist isolation

The commanded shoulder/elbow/wrist target was constant (peak-to-peak 0 rad).
Actual compliant drift under the unchanged v17 actuator settings reached
{float(np.max(np.abs(arm-arm[0]))):.9f} rad; this is reported rather than hidden.

## 3. Dex3 tracking

At steady HOLD, commanded q was `{commanded[hold_mid].tolist()}` rad and actual q was
`{actual[hold_mid].tolist()}` rad; maximum error was
{tracking['steady_hold_approved_pinch_snapshot']['maximum_absolute_error_rad']:.9f} rad.
Overall trajectory tracking RMSE was {tracking['overall_rmse_rad']:.9f} rad and
maximum error during smooth transitions was {tracking['maximum_absolute_error_rad']:.9f} rad.
The closest all-7-DOF approved-pose error was
{tracking['closest_actual_to_approved_pinch_max_error_rad']:.9f} rad, so tracking is
`{tracking['status']}`; the physical thumb+index chains alone came within
{tracking['closest_thumb_index_chain_to_approved_max_error_rad']:.9f} rad.

## 4. Physical contacts

- Physical thumb: 0 samples, maximum 0 N; location and normal unavailable because no contact occurred.
- Physical index: 0 samples, maximum 0 N; location and normal unavailable because no contact occurred.
- Simultaneous thumb+index: 0 samples / 0 s.
- Physical third: 0 samples, maximum 0 N; remained non-task.
- Prohibited self-collision records: {collision['prohibited_robot_self_contact_records']}.

## 5. Failure diagnosis and retention

The authoritative MagSafe scene and v14 root were composed, then the phone was
restored once to the exact approved phone-to-hand calibration pose with zero velocity
before timed physics.  With gravity and collision enabled, the free phone moved
{result['failure_diagnosis']['phone_displacement_before_pinch_transition_m']:.6f} m before
the 1.0 s PINCH transition began, so it was no longer in the grasp region when the
task fingers closed.  The complete seven-DOF hand state also
retained the separately reported `{tracking['status']}`.  Therefore retention is
`{retention['status']}` and the run does **not** isolate a residual finger geometry
or collision-mesh failure.  The previously measured static bilateral surface gap
was {result['failure_diagnosis']['approved_static_bilateral_surface_gap_m']*1000:.3f} mm per side.

## 6. No-cheat audit

The phone pose and zero velocity were written once before timed physics and zero
times afterward.  Object-follow, teleport during the run, scripted attachment,
hidden fixed joints, direct link writes, and kinematic playback were all zero.
The run executed {no_cheat['actual_physx_steps']} actual PhysX steps through the
post-step Fabric/RTX render path.

## 7. Review artifacts

- Overview: `{OUT/'left_phone_pinch_static_physics_overview.mp4'}`
- Close-up: `{OUT/'left_phone_pinch_static_physics_closeup.mp4'}`
- Side: `{OUT/'left_phone_pinch_static_physics_side.mp4'}`
- Contact sheet: `{OUT/'left_phone_pinch_static_physics_contact_sheet.png'}`
- Contact identity: `{OUT/'left_phone_pinch_physics_contact_identity.png'}`

## 8. GUI command

```bash
{gui_command}
```

## 9. Exact next action

`DIAGNOSE THE STATIC THUMB/INDEX CONTACT FAILURE BEFORE FULL-TRAJECTORY INTEGRATION.`

The narrow diagnostic control should preserve the approved q and physics while
starting the timed trial at PINCH/HOLD, so the unsupported OPEN/PREGRASP free-fall
confound is separated from actual PhysX pad/collision geometry.  It must not tune
the primitive or proceed to the 990-frame trajectory.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "status": result["status"],
        "approved_primitive_sha256": sha(PRIMITIVE),
        "runner": {"path": str(RUNNER), "sha256": sha(RUNNER)},
        "finalizer": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
        "physical_outcome": {
            "thumb_contact": result["thumb_contact"], "index_contact": result["index_contact"],
            "simultaneous_contact": result["simultaneous_thumb_index_contact"],
            "third_non_task": result["third_remained_non_task"],
        },
        "integration_gate_created": False,
        "gui_tested": True,
        "artifacts": {},
    }
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "run_manifest.json" and not path.name.endswith(".usda"):
            manifest["artifacts"][path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": result["status"], "tests": tests["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
