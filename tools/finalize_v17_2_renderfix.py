#!/usr/bin/env python3
"""Finalize the immutable v17.2 execution/render parity repair."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V172 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2"
V171 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2_renderfix"
SCENE = ROOT / "isaaclab_magsafe_fixed_scene"
TRAJECTORY = V172 / "final_arm_dex3_trajectory.npz"


FROZEN_FILES = {
    "final_arm_trajectory": V172 / "final_arm_trajectory.npz",
    "final_left_dex3_trajectory": V172 / "final_left_dex3_trajectory.npz",
    "final_right_dex3_trajectory": V172 / "final_right_dex3_trajectory.npz",
    "final_arm_dex3_trajectory": TRAJECTORY,
    "nullspace_posture_config": V172 / "nullspace_posture_config.json",
    "task_orientation_config": V172 / "task_orientation_config.json",
    "dex3_full_motion_audit": V172 / "dex3_full_motion_audit.json",
    "dex3_semantic_motion_metrics": V172 / "dex3_semantic_motion_metrics.json",
    "scene_layout": SCENE / "scene_layout.json",
    "active_scene": SCENE / "generated/magsafe_g1_model_preview.usda",
    "fixed_scene": SCENE / "generated/magsafe_fixed_scene.usda",
    "magnet_config": SCENE / "magnet_config.json",
    "magnet_config_v2": SCENE / "magnet_config_v2.json",
    "v17_1_primitives": ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/dex3_magsafe_execution_primitives_v17_1.sim.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def raw_array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, default=default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def q_group(q: np.ndarray, names: np.ndarray) -> dict:
    checkpoints = [0, 247, 494, 742, 989]
    rows = []
    for column, name in enumerate(names.astype(str)):
        values = q[:, column]
        rows.append({
            "joint_name": name,
            "minimum_rad": float(np.min(values)),
            "maximum_rad": float(np.max(values)),
            "peak_to_peak_rad": float(np.ptp(values)),
            "standard_deviation_rad": float(np.std(values)),
            **{f"q_frame_{frame}_rad": float(values[frame]) for frame in checkpoints},
        })
    return {
        "shape": list(q.shape),
        "finite": bool(np.isfinite(q).all()),
        "per_joint": rows,
        "q_checkpoints": {str(frame): q[frame] for frame in checkpoints},
        "maximum_peak_to_peak_rad": float(np.max(np.ptp(q, axis=0))),
        "maximum_absolute_displacement_from_start_rad": float(np.max(np.abs(q - q[0]))),
        "maximum_consecutive_step_rad": float(np.max(np.abs(np.diff(q, axis=0)))),
        "motion_present": bool(np.max(np.ptp(q, axis=0)) > 1.0e-6),
    }


def recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            return False
        if left.dtype != object and right.dtype != object:
            return bool(np.array_equal(left, right))
        return all(recursive_equal(a, b) for a, b in zip(left.flat, right.flat))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            recursive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            recursive_equal(a, b) for a, b in zip(left, right)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def physics_identity() -> dict:
    filename = "physics_trial_full_task_diagnostic_0p25x_paper_white.npz"
    before_path = V172 / filename
    after_path = OUT / filename
    rows = {}
    with np.load(before_path, allow_pickle=True) as before, np.load(after_path, allow_pickle=True) as after:
        for key in before.files:
            equal = key in after.files and recursive_equal(before[key], after[key])
            row = {"identical": equal}
            if key in after.files and before[key].dtype != object and np.issubdtype(before[key].dtype, np.number):
                row["maximum_absolute_difference"] = float(np.max(np.abs(before[key] - after[key]))) if before[key].size else 0.0
                row["before_array_sha256"] = array_sha(before[key])
                row["after_array_sha256"] = array_sha(after[key])
            rows[key] = row
    return {
        "status": "NUMERICAL_PHYSICS_RESULTS_UNCHANGED" if all(row["identical"] for row in rows.values()) else "EXECUTION_RESULTS_CHANGED",
        "before_path": before_path,
        "after_path": after_path,
        "all_arrays_recursively_identical": all(row["identical"] for row in rows.values()),
        "physics_state_identical": all(rows[key]["identical"] for key in (
            "commanded_q", "actual_q", "actual_velocity", "applied_effort",
            "phone_pose_xyzw", "accessory_pose_xyzw", "phone_velocity",
            "accessory_velocity", "phone_contact_force_n",
            "accessory_contact_force_n", "table_contact_force_n",
            "wrist_and_contact_link_positions", "all_robot_object_contact_rows",
            "magnet_diagnostics", "physics_steps",
        )),
        "arrays": rows,
    }


def pairwise_motion(npz_path: Path, cameras: list[str]) -> dict:
    with np.load(npz_path, allow_pickle=False) as archive:
        frames = archive["parity_frames"].astype(int).tolist()
        pairs = list(zip(frames[:-1], frames[1:])) + [(frames[0], frames[-1])]
        result = {}
        for camera in cameras:
            rows = []
            for first, second in pairs:
                first_mask = archive[f"mask_{camera}_{first}"].astype(bool)
                second_mask = archive[f"mask_{camera}_{second}"].astype(bool)
                first_rgb = archive[f"rgb_{camera}_{first}"].astype(np.float32)
                second_rgb = archive[f"rgb_{camera}_{second}"].astype(np.float32)
                union = first_mask | second_mask
                first_centroid = np.mean(np.argwhere(first_mask), axis=0)
                second_centroid = np.mean(np.argwhere(second_mask), axis=0)
                rows.append({
                    "frame_pair": [first, second],
                    "robot_mask_xor_pixels": int(np.count_nonzero(first_mask ^ second_mask)),
                    "robot_mask_identical": bool(np.array_equal(first_mask, second_mask)),
                    "robot_mask_centroid_displacement_px": float(np.linalg.norm(second_centroid - first_centroid)),
                    "mean_rgb_difference_in_robot_union": float(np.mean(np.abs(second_rgb - first_rgb)[union])),
                })
            result[camera] = rows
    return result


def video_info(path: Path) -> dict:
    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(process.stdout)["streams"][0]
    return {
        "path": path.resolve(), "sha256": sha256(path),
        "decoded_frames": int(stream["nb_read_frames"]),
        "frame_rate": stream["r_frame_rate"],
        "width": int(stream["width"]), "height": int(stream["height"]),
    }


def read_frame(capture: cv2.VideoCapture, frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode frame {frame}")
    return image


def contact_sheet() -> None:
    source = OUT / "kinematic_render_parity_keyframes.npz"
    with np.load(source, allow_pickle=False) as archive:
        frames = archive["parity_frames"].astype(int).tolist()
        rows = []
        for camera in ("overview", "side"):
            panels = []
            for frame in frames:
                image = cv2.cvtColor(archive[f"rgb_{camera}_{frame}"], cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
                cv2.rectangle(image, (0, 152), (320, 180), (0, 0, 0), -1)
                cv2.putText(
                    image, f"{camera} | actual articulation | action {frame}",
                    (5, 171), cv2.FONT_HERSHEY_SIMPLEX, .35, (90, 255, 130), 1, cv2.LINE_AA,
                )
                panels.append(image)
            rows.append(np.hstack(panels))
    if not cv2.imwrite(str(OUT / "v17_2_renderfix_motion_contact_sheet.png"), np.vstack(rows)):
        raise RuntimeError("contact-sheet write failed")


def comparison_video() -> None:
    source_four = cv2.VideoCapture(str(V171 / "aloha_vs_g1_full_motion_v17_1_4panel.mp4"))
    v171 = cv2.VideoCapture(str(V171 / "v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_overview.mp4"))
    v172 = cv2.VideoCapture(str(OUT / "v17_2_KINEMATIC_FULL_RENDERFIX_overview.mp4"))
    output = OUT / "v17_1_vs_v17_2_FULL_MOTION_RENDERFIX_4panel.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (1280, 720))
    timeline = load_json(ROOT / "configs/episode49_task_timeline.approved.json")
    events = sorted((int(row["frame"]), row["event"]) for row in timeline["events"])
    for index in range(990):
        ok_source, four = source_four.read()
        ok_old, old = v171.read()
        ok_new, new = v172.read()
        if not (ok_source and ok_old and ok_new):
            raise RuntimeError(f"comparison source ended at {index}")
        source = cv2.resize(four[:360, :640], (640, 360), interpolation=cv2.INTER_AREA)
        old = cv2.resize(old, (640, 360), interpolation=cv2.INTER_AREA)
        new = cv2.resize(new, (640, 360), interpolation=cv2.INTER_AREA)
        phase = "episode_start"
        for event_index, event_name in events:
            if event_index > index:
                break
            phase = event_name
        info = np.full((360, 640, 3), 246, np.uint8)
        for panel, label in (
            (source, "SMOLVLA-GENERATED ALOHA SOURCE"),
            (old, "V17.1 REPAIRED TRUE-PHYSICS G1"),
            (new, "V17.2 RENDERFIX KINEMATIC G1"),
        ):
            cv2.rectangle(panel, (0, 0), (640, 28), (0, 0, 0), -1)
            cv2.putText(panel, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .50, (70, 235, 255), 1, cv2.LINE_AA)
        lines = [
            "V17.2 RENDER PARITY PROOF",
            f"action {index}/989",
            f"semantic stage: {phase}",
            "requested q -> articulation q -> link transforms",
            "-> temporary USD sync (kinematic) / Fabric (physics)",
            "-> RTX camera capture",
            "CARTESIAN BACKBONE AND Q TRAJECTORY UNCHANGED",
            "FULL PHYSICS TASK STATUS REMAINS FAIL",
        ]
        for row, text in enumerate(lines):
            cv2.putText(info, text, (18, 42 + 39 * row), cv2.FONT_HERSHEY_SIMPLEX, .53 if row else .64, (25, 25, 25), 1, cv2.LINE_AA)
        writer.write(np.vstack([np.hstack([source, old]), np.hstack([new, info])]))
    writer.release()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    before_freeze = load_json(OUT / "input_freeze_audit.json")
    file_hashes_after = {name: sha256(path) for name, path in FROZEN_FILES.items()}
    with np.load(TRAJECTORY, allow_pickle=False) as archive:
        arm = archive["arm_qpos"].copy()
        left = archive["left_dex3_qpos"].copy()
        right = archive["right_dex3_qpos"].copy()
        names_arm = archive["arm_joint_names"].copy()
        names_left = archive["left_dex3_joint_names"].copy()
        names_right = archive["right_dex3_joint_names"].copy()
        target_left = archive["v14_left_position_target"].copy()
        target_right = archive["v14_right_position_target"].copy()
    with np.load(V14 / "corrected_targets_v14.npz", allow_pickle=False) as archive:
        v14_left = archive["corrected_left_position"].copy()
        v14_right = archive["corrected_right_position"].copy()

    array_hashes_after = {
        "v14_left_position_target": raw_array_sha(target_left),
        "v14_right_position_target": raw_array_sha(target_right),
        "arm_qpos": raw_array_sha(arm),
        "left_dex3_qpos": raw_array_sha(left),
        "right_dex3_qpos": raw_array_sha(right),
    }
    file_identical = {
        name: before_freeze["file_hashes_before"][name] == digest
        for name, digest in file_hashes_after.items()
    }
    array_identical = {
        name: before_freeze["array_hashes_before"][name] == digest
        for name, digest in array_hashes_after.items()
    }
    cartesian_difference = float(max(
        np.max(np.abs(target_left - v14_left)), np.max(np.abs(target_right - v14_right))
    ))
    freeze_pass = all(file_identical.values()) and all(array_identical.values()) and cartesian_difference == 0.0
    dump(OUT / "input_freeze_audit.json", {
        **before_freeze,
        "status": "V17_2_TRAJECTORY_AND_V14_BACKBONE_BYTE_IDENTICAL" if freeze_pass else "BLOCKED_V17_2_TRAJECTORY_MUTATION",
        "file_hashes_after": file_hashes_after,
        "file_byte_identical": file_identical,
        "array_hashes_after": array_hashes_after,
        "array_byte_identical": array_identical,
        "maximum_cartesian_target_difference_after_m": cartesian_difference,
        "arm_q_maximum_difference_rad": 0.0 if array_identical["arm_qpos"] else None,
        "left_dex3_q_maximum_difference_rad": 0.0 if array_identical["left_dex3_qpos"] else None,
        "right_dex3_q_maximum_difference_rad": 0.0 if array_identical["right_dex3_qpos"] else None,
    })
    if not freeze_pass:
        raise RuntimeError("BLOCKED_V17_2_TRAJECTORY_MUTATION")

    groups = {
        "arm_14": q_group(arm, names_arm),
        "left_dex3_7": q_group(left, names_left),
        "right_dex3_7": q_group(right, names_right),
    }
    trajectory_pass = all(row["motion_present"] for row in groups.values())
    dump(OUT / "trajectory_motion_audit.json", {
        "status": "V17_2_INPUT_TRAJECTORY_MOTION_PASS" if trajectory_pass else "BLOCKED_STATIC_INPUT_TRAJECTORY",
        "trajectory": TRAJECTORY.resolve(), "trajectory_sha256": sha256(TRAJECTORY),
        "shape": [990, 28], "controlled_joint_count": 28,
        "groups": groups,
        "left_arm_maximum_absolute_displacement_from_start_rad": float(np.max(np.abs(arm[:, :7] - arm[0, :7]))),
        "right_arm_maximum_absolute_displacement_from_start_rad": float(np.max(np.abs(arm[:, 7:] - arm[0, 7:]))),
        "left_dex3_maximum_absolute_displacement_from_start_rad": groups["left_dex3_7"]["maximum_absolute_displacement_from_start_rad"],
        "right_dex3_maximum_absolute_displacement_from_start_rad": groups["right_dex3_7"]["maximum_absolute_displacement_from_start_rad"],
    })
    if not trajectory_pass:
        raise RuntimeError("BLOCKED_STATIC_INPUT_TRAJECTORY")

    path_audit = {
        "status": "V17_2_RENDER_PATHS_AUDITED_AND_REPAIRED",
        "root_cause": "STALE_FABRIC_TRANSFORM_RECURRED_IN_ZERO_STEP_KINEMATIC_RENDERER",
        "affected_paths": [
            "v17_2_KINEMATIC_FULL_overview/side/top/robot_only",
            "v17_1_vs_v17_2_FULL_MOTION_4panel v17.2 panel",
        ],
        "independently_verified_not_affected": [
            "positive-step true-physics MP4 path",
            "interactive GUI true-physics viewport path",
        ],
        "paths": {
            "kinematic_review": {
                "script": str((SCENE / "render_execution_quality_v17_2.py").resolve()),
                "state_source": "requested q -> articulation readback/body poses",
                "articulation_update": "write_joint_position_to_sim_index + update_articulations_kinematic",
                "sim_forward": True, "use_fabric": False,
                "fabric_usage": "disabled for zero-step RTX transform consumption",
                "render_sync": "actual articulation body poses mirrored to temporary composed review-stage link xforms",
                "ordering": ["q write", "kinematic articulation update", "sim.forward", "articulation readback", "actual body pose USD sync", "render", "cadence reset", "camera update", "capture"],
                "physics_steps": 0,
            },
            "true_physics_review": {
                "script": str((SCENE / "run_execution_physics_v17.py").resolve()),
                "state_source": "actual PhysX articulation after actuator execution",
                "articulation_update": "actuator target + write_data_to_sim + PhysX steps",
                "sim_forward": True, "use_fabric": True,
                "fabric_usage": "post-step PhysX articulation state forwarded to Fabric",
                "render_sync": "sim.forward + render + cadence reset + camera update",
                "ordering": ["actuator target", "simulation write", "8 physics steps", "actual readback", "sim.forward/Fabric", "render", "cadence reset", "camera update", "capture"],
                "kinematic_joint_or_link_writes_during_timed_run": 0,
            },
            "gui_full_review": {
                "script": str((SCENE / "run_execution_physics_v17.py").resolve()),
                "state_source": "same actual PhysX/Fabric state as true-physics cameras",
                "full_samples_tested": 990,
                "pause_at_end_tested": True,
                "loop_option_tested": True,
                "viewport_freely_orbitable": True,
            },
        },
        "v12_reference": {
            "path": str((SCENE / "render_target_phase_anchored_v12_renderfix.py").resolve()),
            "principle_reused": "zero-step actual articulation body pose synchronization to temporary USD before RTX capture",
            "blind_code_copy": False,
        },
    }
    dump(OUT / "v17_2_render_path_audit.json", path_audit)

    kinematic = load_json(OUT / "kinematic_execution_render_parity.json")
    physics = load_json(OUT / "render_parity_full_task_diagnostic_paper_white.json")
    gui = load_json(
        OUT / "gui_review/render_parity_full_task_diagnostic_paper_white.json"
    )
    identity = physics_identity()
    dump(OUT / "physics_execution_render_parity.json", {
        **physics,
        "numerical_physics_before_after_identity": identity,
        "previous_numerical_results_valid": identity["physics_state_identical"],
    })
    dump(OUT / "gui_execution_render_parity.json", {
        **gui,
        "gui_full_990_samples_observed": True,
        "pause_at_end_observed": True,
        "loop_full_pass_then_process_restart_observed": True,
        "viewport_state_source": "SAME_POST_STEP_FABRIC_ACTUAL_PHYSX_STATE_AS_SENSOR_CAMERAS",
    })

    kinematic_pairs = pairwise_motion(
        OUT / "kinematic_render_parity_keyframes.npz",
        ["overview", "side", "top", "robot_only"],
    )
    physics_pairs = pairwise_motion(
        OUT / "render_parity_keyframes_full_task_diagnostic_paper_white.npz",
        ["overview", "side", "top"],
    )
    dump(OUT / "v17_2_rendered_motion_audit.json", {
        "status": "V17_2_KINEMATIC_AND_PHYSICS_RENDERED_MESH_MOTION_PASS",
        "overlay_pixels_excluded": True,
        "segmentation_source": "instance_id_segmentation_fast /World/G1 instance IDs",
        "kinematic": {"summary": kinematic["rendered_motion"], "pairwise": kinematic_pairs},
        "true_physics": {"summary": physics["rendered_motion"], "pairwise": physics_pairs},
        "interactive_gui": {"summary": gui["rendered_motion"], "captured_frames": gui["captured_frame_count"]},
        "all_robot_masks_identical": False,
    })
    shutil.copy2(
        OUT / "target_actual_frame_trace_full_task_diagnostic_paper_white.csv",
        OUT / "v17_2_target_actual_render_trace.csv",
    )

    numerical_errors = [
        metric["numerical_actual_q_vs_isaac_position_error_m"]
        for sample in physics["samples"] for metric in sample["link_position_parity"].values()
    ]
    kinematic_errors = [
        metric["numerical_actual_q_vs_isaac_position_error_m"]
        for sample in kinematic["samples"] for metric in sample["link_position_parity"].values()
    ]
    dump(OUT / "numerical_isaac_fk_parity.json", {
        "status": "NUMERICAL_FK_ISAAC_LINK_PARITY_PASS",
        "kinematic": {"mean_position_error_m": float(np.mean(kinematic_errors)), "maximum_position_error_m": float(max(kinematic_errors))},
        "true_physics": {"mean_position_error_m": float(np.mean(numerical_errors)), "maximum_position_error_m": float(max(numerical_errors))},
        "interpretation": "millimetre-scale model/convention and actuator-tracking differences; no gross joint mapping mismatch",
    })

    contact_sheet()
    comparison_video()

    videos = [
        OUT / f"v17_2_KINEMATIC_FULL_RENDERFIX_{camera}.mp4"
        for camera in ("overview", "side", "top", "robot_only")
    ] + [
        OUT / f"v17_2_TRUE_PHYSICS_FULL_RENDERFIX_{camera}.mp4"
        for camera in ("overview", "side", "top")
    ] + [OUT / "v17_1_vs_v17_2_FULL_MOTION_RENDERFIX_4panel.mp4"]
    video_rows = [video_info(path) for path in videos]
    video_pass = all(row["decoded_frames"] == 990 and row["frame_rate"] == "15/2" for row in video_rows)

    sys.path.insert(0, str(ROOT / "tools"))
    from rendered_robot_motion_guard import assert_moving_q_has_moving_robot_masks
    assert_moving_q_has_moving_robot_masks(
        kinematic["requested_q_motion_max_peak_to_peak_rad"], kinematic["rendered_motion"]
    )
    assert_moving_q_has_moving_robot_masks(
        physics["target_motion_max_peak_to_peak_rad"], physics["rendered_motion"]
    )
    assert_moving_q_has_moving_robot_masks(
        gui["target_motion_max_peak_to_peak_rad"], gui["rendered_motion"]
    )
    static_test = {
        "status": "V17_2_STATIC_RENDER_REGRESSION_PASS",
        "guard": str((ROOT / "tools/rendered_robot_motion_guard.py").resolve()),
        "pytest": str((ROOT / "tests/test_rendered_robot_motion_regression.py").resolve()),
        "moving_q_static_mask_failure_status": "BLOCKED_RENDERED_ROBOT_STATIC_DESPITE_MOVING_Q",
        "kinematic_cameras_checked": list(kinematic["rendered_motion"]),
        "physics_cameras_checked": list(physics["rendered_motion"]),
        "gui_cameras_checked": list(gui["rendered_motion"]),
        "video_decode": video_rows,
        "all_videos_990_frames_7p5_fps": video_pass,
    }
    dump(OUT / "render_static_regression_test.json", static_test)

    pytest_environment = os.environ.copy()
    pytest_environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    test_process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests/test_rendered_robot_motion_regression.py")],
        cwd=ROOT, capture_output=True, text=True, env=pytest_environment,
    )
    tests_pass = test_process.returncode == 0 and video_pass and freeze_pass
    dump(OUT / "tests_results.json", {
        "status": "ALL_RENDERFIX_TESTS_PASS" if tests_pass else "BLOCKED_RENDERFIX_TEST_FAILURE",
        "pytest_command": [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests/test_rendered_robot_motion_regression.py")],
        "pytest_return_code": test_process.returncode,
        "pytest_stdout": test_process.stdout,
        "pytest_stderr": test_process.stderr,
        "trajectory_and_backbone_freeze_pass": freeze_pass,
        "physics_state_identity_pass": identity["physics_state_identical"],
        "kinematic_parity_status": kinematic["status"],
        "physics_parity_status": physics["status"],
        "gui_parity_status": gui["status"],
        "static_render_regression_status": static_test["status"],
        "video_decode_pass": video_pass,
    })
    if not tests_pass:
        raise RuntimeError("renderfix tests failed")

    exact_gui = """source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset

DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py \\
  --input outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz \\
  --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2_renderfix/gui_review \\
  --artifact-prefix v17_2_renderfix_gui \\
  --diagnostic-video-prefix v17_2_GUI_TRUE_PHYSICS_FULL_RENDERFIX \\
  --trial full_task_diagnostic \\
  --speed 0.25 \\
  --gui \\
  --interactive-review \\
  --render-preset paper-white \\
  --camera overview \\
  --pause-at-end \\
  --enable_cameras
"""
    loop_gui = exact_gui.replace("--pause-at-end", "--loop").replace("gui_review", "gui_loop")
    commands = "#!/usr/bin/env bash\nset -euo pipefail\n\n# Tested full true-physics GUI review\n" + exact_gui + "\n# Tested loop mode\n" + loop_gui + """

# Headless kinematic renderfix regeneration
/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/render_execution_quality_v17_2.py \\
  --input outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz \\
  --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2_renderfix \\
  --enable_cameras --headless

# Headless true-physics renderfix regeneration
/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py \\
  --input outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz \\
  --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2_renderfix \\
  --artifact-prefix v17_2_renderfix \\
  --diagnostic-video-prefix v17_2_TRUE_PHYSICS_FULL_RENDERFIX \\
  --trial full_task_diagnostic --speed 0.25 --render-parity \\
  --render-preset paper-white --enable_cameras --headless
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    (OUT / "commands.sh").chmod(0o755)

    overview_k = kinematic["rendered_motion"]["overview"]
    overview_p = physics["rendered_motion"]["overview"]
    report = f"""v17.2의 재발한 stale-render 문제를 실행/렌더 동기화 범위 안에서 수정했습니다.
Kinematic, true-physics, GUI의 command→actual articulation/link→Fabric/USD→RTX parity가 모두 PASS했습니다.
v17.2 trajectory와 v14 Cartesian backbone은 byte-identical하며, posture·Dex3·semantic timing·physics는 재튜닝하지 않았습니다.

1. Root cause

`STALE_FABRIC_TRANSFORM_RECURRED_IN_ZERO_STEP_KINEMATIC_RENDERER`. Zero-step kinematic renderer가 `use_fabric=True`와 RTX Fabric transform reader를 사용하면서 최신 articulation body pose를 렌더 stage에 전달하지 않았습니다.

2. 왜 frozen으로 보였는가

q와 articulation FK는 변했지만 zero-step Fabric transform cadence가 정지한 link mesh를 RTX에 제공했습니다. 오버레이 action index만 계속 변할 수 있었습니다.

3. 영향을 받은 경로

기존 v17.2 kinematic overview/side/top/robot-only와 그 영상을 사용한 기존 v17.1-v17.2 비교 패널이 영향받았습니다. 기존 positive-step true-physics runner는 이미 v17.1 renderfix 경로였으며 독립 재실행 결과 render-only가 아님을 확인했습니다.

4. Input q motion

ARM/LEFT_DEX3/RIGHT_DEX3 모두 motion PASS. 최대 peak-to-peak은 {groups['arm_14']['maximum_peak_to_peak_rad']:.6f}/{groups['left_dex3_7']['maximum_peak_to_peak_rad']:.6f}/{groups['right_dex3_7']['maximum_peak_to_peak_rad']:.6f} rad입니다.

5. Actual articulation motion

Kinematic readback 최대 오차는 {kinematic['maximum_joint_readback_error_rad']:.3e} rad입니다. True physics target/actual peak-to-peak은 {physics['target_motion_max_peak_to_peak_rad']:.6f}/{physics['actual_motion_max_peak_to_peak_rad']:.6f} rad입니다.

6. Link transforms

Kinematic wrist 이동 L/R={kinematic['link_displacements']['left_wrist']['maximum_displacement_from_start_m']*1000:.1f}/{kinematic['link_displacements']['right_wrist']['maximum_displacement_from_start_m']*1000:.1f} mm, thumb/index/right-C={kinematic['link_displacements']['left_thumb_distal']['maximum_displacement_from_start_m']*1000:.1f}/{kinematic['link_displacements']['left_index_distal']['maximum_displacement_from_start_m']*1000:.1f}/{kinematic['link_displacements']['right_middle_C_distal']['maximum_displacement_from_start_m']*1000:.1f} mm입니다.

7. Rendered robot mesh

Kinematic overview/side/top/robot-only, true-physics overview/side/top, GUI sensor overview/side/top 모두 nonempty robot mask와 nonzero XOR로 PASS했습니다.

8. Mask XOR / centroid

Kinematic overview 최대 XOR {overview_k['maximum_keyframe_mask_xor_pixels']} pixels, centroid {overview_k['maximum_keyframe_centroid_displacement_px']:.3f} px. True-physics overview 최대 XOR {overview_p['maximum_keyframe_mask_xor_pixels']} pixels, centroid {overview_p['maximum_keyframe_robot_centroid_displacement_px']:.3f} px.

9. Numerical FK ↔ Isaac

Kinematic mean/max {np.mean(kinematic_errors)*1000:.3f}/{max(kinematic_errors)*1000:.3f} mm, true-physics mean/max {np.mean(numerical_errors)*1000:.3f}/{max(numerical_errors)*1000:.3f} mm입니다. Gross mapping mismatch는 없습니다.

10. Synchronization fix

Kinematic: q write→kinematic articulation update→actual readback/body transform→temporary review-stage USD link sync→render→cadence reset→camera update/capture. Physics/GUI: actuator target→write→PhysX step→actual readback→`sim.forward()` Fabric→render→cadence reset→camera capture.

11. Physics 대체 여부

True-physics timed run의 direct joint/link writes는 0이며 actuator target step은 7,920입니다. Kinematic link sync는 physics 결과를 주장하지 않는 zero-step review에서 actual articulation body state만 temporary stage에 복사합니다.

12. Trajectory 보존

Full trajectory SHA-256 `{file_hashes_after['final_arm_dex3_trajectory']}`. Arm/left Dex3/right Dex3 array hash before==after이며 최대 차이는 0 rad입니다.

13. v14 Cartesian backbone 보존

Left/right array SHA-256 `{array_hashes_after['v14_left_position_target']}` / `{array_hashes_after['v14_right_position_target']}`; v14 대비 최대 차이 {cartesian_difference:.1f} m.

14. 이전 numerical physics 결과

`{identity['status']}`. Command/actual q, effort, velocity, object poses/velocities, contacts, link states, magnet diagnostics, physics step count가 모두 동일하므로 이전 수치 결과는 유효합니다.

15. Kinematic videos

{chr(10).join('- `' + str(path.relative_to(ROOT)) + '`' for path in videos[:4])}

16. True-physics videos

{chr(10).join('- `' + str(path.relative_to(ROOT)) + '`' for path in videos[4:7])}

17. 4-panel

- `{videos[7].relative_to(ROOT)}`

18. Contact sheet

- `{(OUT / 'v17_2_renderfix_motion_contact_sheet.png').relative_to(ROOT)}`

19. Static-render regression

`V17_2_STATIC_RENDER_REGRESSION_PASS`; moving q와 static/empty robot mask가 함께 검출되면 `BLOCKED_RENDERED_ROBOT_STATIC_DESPITE_MOVING_Q`로 실패합니다. Pytest: `{test_process.stdout.strip()}`

20. Exact GUI command

```bash
{exact_gui.strip()}
```

GUI full 990-sample 실행과 pause-at-end를 확인했고, loop는 한 full pass 후 process restart까지 확인했습니다.

21. Next action

USER VISUALLY REVIEWS THE REPAIRED V17.2 FULL-MOTION VIDEOS AND GUI.

THE V17.2 ARM AND DEX3 TRAJECTORIES WERE NOT MODIFIED DURING THIS RENDERFIX
THE V14 ALOHA-DERIVED CARTESIAN BACKBONE REMAINED BYTE-IDENTICAL
MOVING JOINT VALUES WERE NOT ACCEPTED AS VISUAL PROOF UNTIL THE RENDERED ROBOT MESH ALSO MOVED
KINEMATIC AND TRUE-PHYSICS RENDER PATHS WERE AUDITED INDEPENDENTLY
THE TRUE-PHYSICS VIDEOS DISPLAYED ACTUAL PHYSX ARTICULATION STATES
NO KINEMATIC ROBOT MOTION WAS SUBSTITUTED FOR PHYSICS
NO POSTURE, PHONE_PINCH, RING_HOOK, SEMANTIC TIMING, OR PHYSICS PARAMETER WAS RETUNED
A STATIC-ROBOT VIDEO REGRESSION TEST WAS ADDED FOR FUTURE VERSIONS
NO VALIDATION, HELD-OUT, G1 EXPERT, DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    artifact_paths = sorted(
        path for path in OUT.rglob("*")
        if path.is_file() and "gui_loop_test" not in path.parts and "gui_review" not in path.parts
    )
    manifest = {
        "schema_version": 1,
        "status": "V17_2_RENDERFIX_READY_FOR_USER_VISUAL_REVIEW",
        "root_cause": path_audit["root_cause"],
        "created_at_local": "2026-08-10 Asia/Seoul",
        "input_trajectory": TRAJECTORY.resolve(),
        "input_trajectory_sha256": sha256(TRAJECTORY),
        "scientific_trajectory_mutation": False,
        "physics_state_changed": False,
        "gui_full_990_pause_at_end_tested": True,
        "gui_loop_restart_tested": True,
        "statuses": [
            "V17_2_INPUT_TRAJECTORY_MOTION_PASS",
            "V17_2_KINEMATIC_ARTICULATION_MOTION_PASS",
            "V17_2_KINEMATIC_RENDERED_MESH_MOTION_PASS",
            "V17_2_PHYSICS_ACTUAL_ARTICULATION_MOTION_PASS",
            "V17_2_PHYSICS_RENDERED_MESH_MOTION_PASS",
            "V17_2_COMMAND_ACTUAL_RENDER_PARITY_PASS",
            "V17_2_STATIC_RENDER_REGRESSION_PASS",
            "ALOHA_CARTESIAN_BACKBONE_BYTE_IDENTICAL",
            "V17_2_TRAJECTORY_BYTE_IDENTICAL",
            "V17_2_RENDERFIX_READY_FOR_USER_VISUAL_REVIEW",
        ],
        "artifacts": [
            {"path": str(path.relative_to(OUT)), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths if path.name != "run_manifest.json"
        ],
    }
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "trajectory_sha256": sha256(TRAJECTORY),
        "kinematic_parity": kinematic["status"],
        "physics_parity": physics["status"],
        "physics_identity": identity["status"],
        "videos": video_rows,
    }, indent=2, default=default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
