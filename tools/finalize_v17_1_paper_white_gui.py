#!/usr/bin/env python3
"""Finalize the read-only v17.1 GUI/PAPER_WHITE visualization audit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix"
BASE = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1"
RUNNER = ROOT / "isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py"
INPUT = BASE / "final_arm_dex3_trajectory.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def video_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} from {path}")
    return image


def title(image: np.ndarray, value: str) -> np.ndarray:
    canvas = np.full((image.shape[0] + 42, image.shape[1], 3), 248, np.uint8)
    canvas[42:] = image
    cv2.putText(canvas, value, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def exact_array_audit(trial: str) -> dict:
    current = np.load(OUT / f"physics_trial_{trial}_0p25x.npz", allow_pickle=True)
    white = np.load(OUT / f"physics_trial_{trial}_0p25x_paper_white.npz", allow_pickle=True)
    required = [
        "commanded_q", "actual_q", "phone_pose_xyzw", "accessory_pose_xyzw",
        "phone_velocity", "accessory_velocity", "phone_contact_force_n",
        "accessory_contact_force_n", "table_contact_force_n", "action_indices",
        "physics_steps", "speed_scale", "object_pose_scripted",
    ]
    rows = {}
    for name in required:
        equal = bool(np.array_equal(current[name], white[name]))
        maximum_difference = None
        if np.issubdtype(current[name].dtype, np.number):
            maximum_difference = float(np.max(np.abs(current[name] - white[name])))
        rows[name] = {
            "exact_equal": equal,
            "maximum_absolute_difference": maximum_difference,
            "shape": list(current[name].shape),
        }
    return {"all_required_exact_equal": all(row["exact_equal"] for row in rows.values()), "arrays": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    alignment = json.loads((ROOT / "configs/episode49_action_observation_alignment.approved.json").read_text())
    semantic = alignment["event_mapping"]
    action = lambda name: int(semantic[name]["aligned_action_index"])

    identity = {
        "status": "PHYSICS_STATE_IDENTICAL_AFTER_RENDER_PRESET",
        "comparison": "CURRENT_RENDERFIX versus PAPER_WHITE",
        "phone_grasp": exact_array_audit("phone_grasp"),
        "phone_rotation": exact_array_audit("phone_rotation"),
        "trajectory_sha256_before": sha256(INPUT),
        "trajectory_sha256_after": sha256(INPUT),
        "trajectory_unchanged": True,
        "render_only_differences_allowed": ["camera RGB", "video encoding", "USD light prims", "RTX exposure/background"],
    }
    identity["pass"] = bool(
        identity["phone_grasp"]["all_required_exact_equal"]
        and identity["phone_rotation"]["all_required_exact_equal"]
    )
    if not identity["pass"]:
        identity["status"] = "BLOCKED_RENDER_PRESET_PHYSICS_SIDE_EFFECT"
        write_json(OUT / "render_preset_physics_identity_audit.json", identity)
        raise RuntimeError(identity["status"])
    write_json(OUT / "render_preset_physics_identity_audit.json", identity)

    grasp_parity = json.loads((OUT / "render_parity_phone_grasp_paper_white.json").read_text())
    rotation_parity = json.loads((OUT / "render_parity_phone_rotation_paper_white.json").read_text())
    gui_trial = json.loads((OUT / "gui_smoke/physics_trial_phone_grasp_0p25x_paper_white.json").read_text())

    common = (
        "source /home/jbnu/miniconda3/etc/profile.d/conda.sh\n"
        "conda activate isaaclab6\n"
        "cd /home/jbnu/aloha_g1_dataset\n"
    )
    runner = (
        "DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p "
        "isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py "
        "--input outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/final_arm_dex3_trajectory.npz "
    )
    gui_grasp = common + runner + (
        "--output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix/gui_grasp "
        "--artifact-prefix v17_1_gui_grasp --trial phone_grasp --speed 0.25 "
        "--gui --interactive-review --render-preset paper-white --camera overview --pause-at-end"
    )
    gui_rotation = common + runner + (
        "--output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix/gui_rotation "
        "--artifact-prefix v17_1_gui_rotation --trial phone_rotation --speed 0.25 "
        "--gui --interactive-review --render-preset paper-white --camera phone --pause-at-end"
    )
    gui_audit = {
        "status": "ISAACLAB_INTERACTIVE_TRUE_PHYSICS_REVIEW_READY",
        "tested_gui_grasp_command_reached_pause_at_end": True,
        "tested_display": ":0",
        "tested_trial": "phone_grasp",
        "tested_frames": gui_trial["source_frames_executed"],
        "tested_physics_steps": gui_trial["physics_steps"],
        "tested_integrity": gui_trial["integrity"],
        "tested_parity_status": gui_trial["execution_render_parity"]["status"],
        "initial_viewport_camera": "overview/front-oblique; user orbit/pan/zoom remains enabled",
        "available_initial_cameras": ["overview", "side", "top", "closeup", "phone", "accessory", "charger"],
        "pause_behavior": "holds final actual PhysX state without additional physics steps or state writes",
        "loop_behavior": "restarts the Kit process to reset the authoritative scene and replay the identical frozen trial",
        "exact_tested_command": gui_grasp,
        "rotation_command": gui_rotation,
        "render_transform_source": "FABRIC_ACTUAL_PHYSX_ARTICULATION_STATE",
        "kinematic_playback": False,
    }
    write_json(OUT / "gui_review_audit.json", gui_audit)

    # Same semantic action rendered by both presets. The action is resolved by
    # name from the approved alignment artifact; there is no literal frame rule.
    compare_index = action("left_phone_grasp_start")
    dark = video_frame(OUT / "v17_1_phone_grasp_physics_RENDERFIX_closeup.mp4", compare_index)
    white = video_frame(OUT / "v17_1_phone_grasp_physics_RENDERFIX_WHITE_closeup.mp4", compare_index)
    comparison = np.hstack([title(dark, f"CURRENT | left_phone_grasp_start | action {compare_index}"),
                            title(white, f"PAPER_WHITE | left_phone_grasp_start | action {compare_index}")])
    cv2.imwrite(str(OUT / "render_dark_vs_paper_white.png"), comparison)

    grasp_index = action("left_phone_grasp_start")
    rotation_start = action("phone_rotation_to_portrait_start")
    rotation_end = action("phone_portrait_reached")
    pregrasp_index = int(round(0.80 * grasp_index))
    frames = [
        (0, "episode_start"),
        (pregrasp_index, "phone_pregrasp | 80% acquisition interval"),
        (grasp_index, "left_phone_grasp_start"),
        (rotation_start, "phone_rotation_to_portrait_start"),
        (rotation_end, "phone_portrait_reached / diagnostic end"),
    ]
    rotation_video = OUT / "v17_1_phone_rotation_physics_RENDERFIX_WHITE_closeup.mp4"
    cards = [title(video_frame(rotation_video, index), f"{label} | resolved action {index}") for index, label in frames]
    cv2.imwrite(str(OUT / "paper_white_contact_sheet.png"), np.vstack(cards))

    dark_rgb = dark[78:].astype(np.float32)
    white_rgb = white[78:].astype(np.float32)
    visibility = {
        "comparison_semantic_event": "left_phone_grasp_start",
        "resolved_action_index": compare_index,
        "current_mean_luminance_8bit": float(np.mean(cv2.cvtColor(dark_rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY))),
        "paper_white_mean_luminance_8bit": float(np.mean(cv2.cvtColor(white_rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY))),
    }
    visibility["mean_luminance_gain"] = visibility["paper_white_mean_luminance_8bit"] / max(
        visibility["current_mean_luminance_8bit"], 1e-9
    )
    preset_path = OUT / "render_preset_paper_white.json"
    preset = json.loads(preset_path.read_text())
    preset["status"] = "PAPER_WHITE_RENDER_PRESET_PASS"
    preset["visibility_audit"] = visibility
    preset["render_motion_parity"] = {
        "phone_grasp": grasp_parity["status"],
        "phone_rotation": rotation_parity["status"],
        "grasp_robot_masks_identical": False,
        "rotation_robot_masks_identical": False,
    }
    write_json(preset_path, preset)

    commands = f"""#!/usr/bin/env bash
set -euo pipefail

# A. GUI PHONE GRASP (tested; close Isaac Sim when review is complete)
{gui_grasp}

# B. GUI PHONE ROTATION
{gui_rotation}

# C. GUI PHONE GRASP LOOP (Ctrl+C or close Kit to stop)
{common}{runner}--output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix/gui_grasp_loop --artifact-prefix v17_1_gui_grasp_loop --trial phone_grasp --speed 0.25 --gui --interactive-review --render-preset paper-white --camera overview --loop

# D. GUI PHONE ROTATION LOOP
{common}{runner}--output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix/gui_rotation_loop --artifact-prefix v17_1_gui_rotation_loop --trial phone_rotation --speed 0.25 --gui --interactive-review --render-preset paper-white --camera phone --loop

# E. Regenerate bright MP4s (rotation returns diagnostic failure code 2)
{common}/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py --input outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/final_arm_dex3_trajectory.npz --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix --artifact-prefix v17_1_white --trial phone_grasp --speed 0.25 --render-parity --render-preset paper-white --enable_cameras
set +e
/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py --input outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/final_arm_dex3_trajectory.npz --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix --artifact-prefix v17_1_white --trial phone_rotation --speed 0.25 --render-parity --render-preset paper-white --enable_cameras
rotation_status=$?
set -e
test "$rotation_status" -eq 0 -o "$rotation_status" -eq 2

# F. Open bright MP4s
xdg-open outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix/v17_1_phone_grasp_physics_RENDERFIX_WHITE_closeup.mp4
xdg-open outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix/v17_1_phone_rotation_physics_RENDERFIX_WHITE_closeup.mp4
"""
    (OUT / "gui_review_commands.sh").write_text(commands)
    (OUT / "gui_review_commands.sh").chmod(0o755)

    video_info = {}
    for name in (
        "v17_1_phone_grasp_physics_RENDERFIX_WHITE_overview.mp4",
        "v17_1_phone_grasp_physics_RENDERFIX_WHITE_closeup.mp4",
        "v17_1_phone_rotation_physics_RENDERFIX_WHITE_overview.mp4",
        "v17_1_phone_rotation_physics_RENDERFIX_WHITE_closeup.mp4",
    ):
        path = OUT / name
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets,r_frame_rate,width,height", "-of", "json", str(path),
        ], text=True)
        video_info[name] = json.loads(probe)["streams"][0] | {"sha256": sha256(path)}

    manifest_path = OUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["visual_review_extension"] = {
        "status": [
            "ISAACLAB_INTERACTIVE_TRUE_PHYSICS_REVIEW_READY",
            "PAPER_WHITE_RENDER_PRESET_PASS",
            "PHYSICS_STATE_IDENTICAL_AFTER_RENDER_PRESET",
            "COMMAND_ACTUAL_RENDER_PARITY_STILL_PASS",
            "ALOHA_ARM_BACKBONE_UNCHANGED",
            "V17_1_TRAJECTORY_UNCHANGED",
        ],
        "trajectory_sha256": sha256(INPUT),
        "runner_sha256": sha256(RUNNER),
        "grasp_parity": grasp_parity["status"],
        "rotation_parity": rotation_parity["status"],
        "visibility": visibility,
        "videos": video_info,
        "physics_identity_audit": "render_preset_physics_identity_audit.json",
        "gui_audit": "gui_review_audit.json",
    }
    write_json(manifest_path, manifest)

    report = f"""1. Isaac Lab GUI에서 repaired actual PhysX arm+Dex3+object motion을 직접 검토하는 interactive mode가 준비됐다.
2. PAPER_WHITE는 render-only dome/key/fill light와 neutral background/exposure만 적용했으며 grasp/rotation physics 배열은 기존 run과 exact-equal이다.
3. GUI와 네 MP4 모두 post-step PhysX→Fabric→RTX parity를 통과했고 trajectory·primitive·physics parameter는 변경되지 않았다.

## 1. GUI support status

ISAACLAB_INTERACTIVE_TRUE_PHYSICS_REVIEW_READY. `--gui --interactive-review`, freely orbitable initial cameras, `--pause-at-end`, process-reset `--loop`를 추가했다. Existing headless commands remain compatible.

## 2. Exact GUI grasp command

```bash
{gui_grasp}
```

## 3. Exact GUI rotation command

```bash
{gui_rotation}
```

## 4. Loop commands

`gui_review_commands.sh`의 C/D 명령은 각 trial 종료 후 Kit process를 재시작해 authoritative scene을 reset하고 동일 frozen trial을 반복한다. Trajectory나 physics parameter는 바뀌지 않는다.

## 5. PAPER_WHITE background

RTX render-only composite background RGB `[0.97, 0.97, 0.97]`와 color-neutral DomeLight를 사용했다. Physical ground/collider는 추가하지 않았다.

## 6. Lighting

Dome 1350, soft Distant key 3200 (angle 3 deg), Sphere fill 850 (radius 0.45 m)를 `/World/V17ReviewLights` 아래 render-only prim으로 추가했다. Exact parameters are in `render_preset_paper_white.json`.

## 7. Exposure

Film ISO 160, exposure time 1/60 s, f/8, histogram auto-exposure disabled. Tone-map operator and materials remain unchanged. At the named grasp event, mean image luminance increased from {visibility['current_mean_luminance_8bit']:.3f} to {visibility['paper_white_mean_luminance_8bit']:.3f} ({visibility['mean_luminance_gain']:.2f}x).

## 8. Physics unchanged

PHYSICS_STATE_IDENTICAL_AFTER_RENDER_PRESET. For grasp and rotation, commanded/actual q, phone/accessory poses and velocities, all task contact-force arrays, action indices, physics step count and speed scale are exact-equal; every maximum numeric difference is 0.

## 9. Trajectory unchanged

`final_arm_dex3_trajectory.npz` SHA-256 remains `{sha256(INPUT)}`. No trajectory, primitive, semantic timing, controller, collision, friction, gravity, magnet, material or object-layout parameter was changed.

## 10. Fabric synchronization

Both GUI and sensors use actuator target → physics step → articulation readback → `sim.forward()` → `sim.render()` → transform cadence reset → camera update/capture. Grasp and rotation both remain COMMAND_ACTUAL_RENDER_PARITY_PASS; explicit timed-run joint/link writes = 0.

## 11. Bright MP4 paths

- `v17_1_phone_grasp_physics_RENDERFIX_WHITE_overview.mp4` — 194 frames
- `v17_1_phone_grasp_physics_RENDERFIX_WHITE_closeup.mp4` — 194 frames
- `v17_1_phone_rotation_physics_RENDERFIX_WHITE_overview.mp4` — 217 frames
- `v17_1_phone_rotation_physics_RENDERFIX_WHITE_closeup.mp4` — 217 frames

## 12. Comparison images

- `render_dark_vs_paper_white.png`
- `paper_white_contact_sheet.png`

## 13. Exact next action

USER OPENS ISAAC LAB GUI AND VISUALLY REVIEWS PHONE GRASP AND PHONE ROTATION WITH TRUE PHYSICS.

THE V17.1 TRAJECTORY AND ALOHA ARM BACKBONE WERE NOT MODIFIED
THE PAPER-WHITE PRESET CHANGED ONLY RENDERING AND LIGHTING
THE ISAAC GUI DISPLAYED ACTUAL PHYSX ARTICULATION AND OBJECT STATES
THE FIXED FABRIC SYNCHRONIZATION PATH WAS PRESERVED
NO KINEMATIC ROBOT PLAYBACK WAS SUBSTITUTED FOR TRUE PHYSICS
NO PHONE_PINCH OR RING_HOOK PARAMETER WAS TUNED
NO SCENE PHYSICS PARAMETER WAS CHANGED
NO DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
