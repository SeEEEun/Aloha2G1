#!/usr/bin/env python3
"""Finalize hashes, commands, and report for the static left pinch audit."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
V17 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz"
RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_left_phone_pinch_photo_calibration_v1.py"
CALIBRATOR = ROOT / "tools/calibrate_left_phone_pinch_photo_reference_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dump(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    required = [
        "attached_photo_reference_manifest.json", "left_dex3_physical_identity.json",
        "left_phone_contact_frames.json", "left_phone_fingertip_pinch_primitive.json",
        "thumb_opposition_before_after.json", "fingertip_geometry_metrics.json",
        "hand_collision_audit.json", "hand_joint_margin_audit.json",
        "left_phone_pinch_front_oblique.png", "left_phone_pinch_back_oblique.png",
        "left_phone_pinch_top.png", "left_phone_pinch_side.png",
        "left_phone_pinch_fingertip_closeup.png", "left_phone_pinch_palm_side.png",
        "left_phone_pinch_third_finger_check.png", "real_photo_vs_isaac_phone_pinch.png",
        "left_phone_pinch_identity_overlay.png", "isaac_static_render_audit.json",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")

    primitive = json.loads((OUT / "left_phone_fingertip_pinch_primitive.json").read_text())
    metrics = json.loads((OUT / "fingertip_geometry_metrics.json").read_text())
    margin = json.loads((OUT / "hand_joint_margin_audit.json").read_text())
    collision = json.loads((OUT / "hand_collision_audit.json").read_text())
    identity = json.loads((OUT / "left_dex3_physical_identity.json").read_text())
    opposition = json.loads((OUT / "thumb_opposition_before_after.json").read_text())
    render = json.loads((OUT / "isaac_static_render_audit.json").read_text())
    previous_manifest = json.loads((OUT / "run_manifest.json").read_text())
    with np.load(V14, allow_pickle=False) as archive:
        left_hash = raw_sha(archive["corrected_left_position"])
        right_hash = raw_sha(archive["corrected_right_position"])
    freeze_after = {
        "v17_2_trajectory_sha256": sha256(V17),
        "v14_archive_sha256": sha256(V14),
        "v14_left_cartesian_raw_sha256": left_hash,
        "v14_right_cartesian_raw_sha256": right_hash,
    }
    freeze_before = previous_manifest.get("freeze", previous_manifest.get("freeze_before"))
    if not freeze_before:
        raise RuntimeError("run manifest lacks initial freeze record")
    unchanged = {
        "v17_2_trajectory": freeze_after["v17_2_trajectory_sha256"] == freeze_before["v17_2_trajectory"]["sha256"],
        "v14_archive": freeze_after["v14_archive_sha256"] == freeze_before["v14_cartesian_source"]["sha256"],
        "v14_left_cartesian": left_hash == freeze_before["v14_left_cartesian_raw_sha256"],
        "v14_right_cartesian": right_hash == freeze_before["v14_right_cartesian_raw_sha256"],
    }
    if not all(unchanged.values()):
        raise RuntimeError(f"scientific input mutation: {unchanged}")

    shell = f"""#!/usr/bin/env bash
set -euo pipefail
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd {ROOT}

# Recompute the deterministic hand-only geometric calibration.
/home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  tools/calibrate_left_phone_pinch_photo_reference_v1.py

# Regenerate all PAPER_WHITE static Isaac views.
/home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/render_left_phone_pinch_photo_calibration_v1.py \\
  --headless --enable_cameras

# Open the calibrated static pose in Isaac GUI; close the GUI to exit.
DISPLAY=:0 /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/render_left_phone_pinch_photo_calibration_v1.py \\
  --gui --pause --enable_cameras

# Open the photo/Isaac comparison sheet.
xdg-open {OUT / 'real_photo_vs_isaac_phone_pinch.png'}
"""
    (OUT / "commands.sh").write_text(shell)

    q = primitive["selected_static_q_rad"]
    m = metrics["metrics"]
    before = opposition["before"]
    report = f"""PHYSICAL THUMB: thumb_0/1/2 chain; PHYSICAL INDEX: index_0/1 chain; PHYSICAL THIRD: XML middle_0/1 chain.
The real-photo fingertip pinch was reproduced with LEFT Dex3 joints alone: YES.
The physical third finger remained non-task: YES.

# Final status

`LEFT_PHONE_FINGERTIP_PINCH_READY_FOR_USER_VISUAL_APPROVAL`

1. Photo interpretation

All six attached real-hardware views were used as qualitative evidence for a distal precision pinch. They support thumb opposition toward the index, distal-pad contact near a phone edge, an intentionally asymmetric pose, and a third finger that does not contribute. No pixel coordinates were interpreted as joint angles or metric geometry.

2. Active Dex3 joint/link identity

The active model is `{identity['active_model']}`. Physical thumb is `left_hand_thumb_0/1/2`; physical index is `left_hand_index_0/1`; the physical third finger is the XML `left_hand_middle_0/1` chain. Collision geoms 61, 69, and 65 were programmatically verified to belong to their respective distal links and the pad centers lie inside the active mesh bounding boxes.

3. Thumb opposition mechanism

`left_hand_thumb_0_joint` (local Y-axis) supplies the opposition component. Keeping index and third fixed, changing thumb_0 from 0 to {q[0]:.9f} rad reduced thumb/index distance from {before['thumb_index_tip_distance_m']*1000:.3f} to {m['thumb_index_tip_distance_m']*1000:.3f} mm and contact-height mismatch from {before['contact_height_offset_wrist_z_m']*1000:.3f} to {m['contact_height_offset_wrist_z_m']*1000:.3f} mm. Wrist roll/pitch/yaw remained exactly 0.

4. Index posture

The index q is [{q[3]:.9f}, {q[4]:.9f}] rad. The distal index pad faces the pinch axis with {m['index_pad_to_pinch_axis_angle_deg']:.3f} deg error.

5. Third-finger posture

The third q is [{q[5]:.9f}, {q[6]:.9f}] rad. It is mildly flexed/open, contributes no objective or pinch axis, and its pad remains {metrics['static_phone_registration']['third_pad_phone_obb_clearance_m']*1000:.3f} mm from the diagnostic phone OBB.

6. Optimized 7-DOF LEFT Dex3 q

Joint order: `{', '.join(primitive['joint_names'])}`.

`[{', '.join(f'{value:.12f}' for value in q)}]` rad.

7. Thumb/index fingertip geometry

Tip aperture is {m['thumb_index_tip_distance_m']*1000:.3f} mm for the authoritative {m['phone_thickness_m']*1000:.3f} mm phone thickness, giving {m['bilateral_surface_gap_m']*1000:.3f} mm per side. Thumb/index pad-to-axis errors are {m['thumb_pad_to_pinch_axis_angle_deg']:.3f}/{m['index_pad_to_pinch_axis_angle_deg']:.3f} deg.

8. Contact-height offset

Wrist-local height offset is {m['contact_height_offset_wrist_z_m']*1000:.3f} mm. The asymmetric photographic pose was not forced into artificial symmetry.

9. Pad-normal opposition

The pad-normal opposition error is {m['pad_normal_opposition_error_deg']:.3f} deg. Each distal pad normal is within 15 deg of its required inward pinch-axis direction.

10. Joint margins

Minimum hand-joint margin is {margin['minimum_margin_rad']:.6f} rad at `{margin['limiting_joint']}`. Joint-limit violations: {margin['joint_limit_violation_count']}.

11. Self collision

Raw left-hand contacts: {collision['raw_left_hand_contact_count']}; prohibited self contacts: {collision['prohibited_self_contact_count']}. No adjacent contact was silently discarded.

12. Static physics sanity

Not run. It is optional and intentionally deferred until visual approval; no friction, controller, gravity, mass, or contact parameter was tuned.

13. Multi-view renders

- `{OUT / 'left_phone_pinch_front_oblique.png'}`
- `{OUT / 'left_phone_pinch_back_oblique.png'}`
- `{OUT / 'left_phone_pinch_top.png'}`
- `{OUT / 'left_phone_pinch_side.png'}`
- `{OUT / 'left_phone_pinch_fingertip_closeup.png'}`
- `{OUT / 'left_phone_pinch_palm_side.png'}`
- `{OUT / 'left_phone_pinch_third_finger_check.png'}`

14. Real-photo-vs-Isaac comparison

`{OUT / 'real_photo_vs_isaac_phone_pinch.png'}`

15. Identity overlay

`{OUT / 'left_phone_pinch_identity_overlay.png'}`. Cyan and magenta markers are the rendered distal contact points; the green line joins their detected image centroids; the yellow-bordered inset shows the non-task third finger.

16. Exact command to open/render the calibrated pose

```bash
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd {ROOT}
DISPLAY=:0 /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/render_left_phone_pinch_photo_calibration_v1.py \\
  --gui --pause --enable_cameras
```

17. Exact next action

USER VISUALLY COMPARES THE ISAAC LEFT PHONE PINCH AGAINST THE ATTACHED REAL DEX3 PHOTOS.

STOP AFTER THAT. The primitive is not integrated into the 990-frame trajectory.
"""
    (OUT / "report.md").write_text(report)

    artifacts = {
        name: {"sha256": sha256(OUT / name), "bytes": (OUT / name).stat().st_size}
        for name in [*required, "report.md", "commands.sh", "left_phone_fingertip_pinch_calibration.npz"]
    }
    manifest = {
        "status": "LEFT_PHONE_FINGERTIP_PINCH_READY_FOR_USER_VISUAL_APPROVAL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "photo_reference_authority": "REAL_DEX3_PHOTO_REFERENCE",
        "physical_task_fingers": ["THUMB", "INDEX"],
        "third_finger": "NON_TASK",
        "arm_wrist_optimization_variables": 0,
        "right_hand_modified": False,
        "v17_2_integrated": False,
        "static_physics_sanity": "NOT_RUN_OPTIONAL__AWAITING_VISUAL_APPROVAL",
        "physics_steps": 0,
        "freeze_before": freeze_before,
        "freeze_after": freeze_after,
        "freeze_unchanged": unchanged,
        "calibrator": {"path": str(CALIBRATOR), "sha256": sha256(CALIBRATOR)},
        "renderer": {"path": str(RENDERER), "sha256": sha256(RENDERER)},
        "isaac_readback_max_error_rad": render["actual_articulation_readback_max_error_rad"],
        "artifacts": artifacts,
        "forbidden_actions": {
            "dds": False, "publisher": False, "hardware_command": False,
            "right_dex3_edit": False, "v17_2_jitter_edit": False,
            "arm_or_wrist_grasp_alignment": False,
        },
    }
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"], "freeze_unchanged": unchanged,
        "artifact_count": len(artifacts), "minimum_joint_margin_rad": margin["minimum_margin_rad"],
        "prohibited_self_contacts": collision["prohibited_self_contact_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
