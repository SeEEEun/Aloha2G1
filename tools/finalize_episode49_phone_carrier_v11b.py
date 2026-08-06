#!/usr/bin/env python3
"""Validate and finalize the Episode-49 phone-carrier audit v11b."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_phone_carrier_audit_v11b"
V11 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
ACTION = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"

REQUIRED = [
    "input_audit.json",
    "alignment_audit.json",
    "phone_carrier_candidates.json",
    "phone_carrier_transform_comparison.json",
    "stable_carrier_interval.json",
    "reconstructed_phone_trajectories.npz",
    "reconstructed_accessory_trajectories.npz",
    "accessory_surface_gap_metrics.json",
    "phone_pad_metrics.json",
    "single_rigid_carrier_viability.json",
    "phone_carrier_candidate_4panel.mp4",
    "phone_carrier_contact_sheet_overview.png",
    "phone_carrier_contact_sheet_closeup.png",
    "accessory_contact_sheet_closeup.png",
    "carrier_transform_comparison.png",
    "visual_audit.json",
]
FORBIDDEN_OUTPUTS = [
    "g1_targets.npz", "g1_arm_trajectory.npz", "position_only_g1_targets.npz",
    "position_only_g1_arm_trajectory.npz", "phasewarp.npz", "orientation_targets.npz",
    "dex3_trajectory.npz", "physics_results.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def decoded_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(path)
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def matrix_markdown(matrix: list[list[float]]) -> str:
    return "\n".join("  [" + ", ".join(f"{value:+.9f}" for value in row) + "]" for row in matrix)


def main() -> int:
    missing = [name for name in REQUIRED if not (OUT / name).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    forbidden = {name: (OUT / name).exists() for name in FORBIDDEN_OUTPUTS}
    if any(forbidden.values()):
        raise RuntimeError(f"Forbidden downstream outputs exist: {forbidden}")

    input_audit = load_json(OUT / "input_audit.json")
    alignment = load_json(OUT / "alignment_audit.json")
    candidates = load_json(OUT / "phone_carrier_candidates.json")
    comparison = load_json(OUT / "phone_carrier_transform_comparison.json")
    stable = load_json(OUT / "stable_carrier_interval.json")
    accessory = load_json(OUT / "accessory_surface_gap_metrics.json")
    pad = load_json(OUT / "phone_pad_metrics.json")
    viability = load_json(OUT / "single_rigid_carrier_viability.json")
    visual = load_json(OUT / "visual_audit.json")

    if input_audit["status"] != "PASS_FROZEN_V11_INPUTS_REUSED":
        raise RuntimeError("Frozen input audit failed")
    if alignment["status"] != "PASS_USER_APPROVED_PLUS_7_FRAME_ALIGNMENT_REUSED_UNCHANGED":
        raise RuntimeError("Approved alignment audit failed")
    if comparison["status"] != "CONTACT_START_176_IS_NOT_A_VALID_RIGID_CARRIER_LOCK":
        raise RuntimeError("Frame-176 carrier was not properly reclassified")
    if pad["status"] != "PASS_PHONE_ON_CHARGER_POSE_DEFINITION":
        raise RuntimeError("Authoritative phone-on-pad pose definition failed")
    if viability["status"] != "SOURCE_PHONE_RELATION_NOT_SINGLE_RIGID_TRANSFORM":
        raise RuntimeError("Unexpected primary carrier decision")
    if viability["secondary_status"] != "BLOCKED_ACCESSORY_ATTACHMENT_OR_RIGHT_GRASP_SEMANTICS":
        raise RuntimeError("Unexpected accessory decision")
    if viability["optimized_action_task_validity_finalized_as_failure"] is not False:
        raise RuntimeError("optimized_action task validity was prematurely failed")
    if viability["single_rigid_phone_carrier_valid"] is not False:
        raise RuntimeError("Single rigid carrier was incorrectly approved")
    if stable["carrier_model_valid_from_observed_frame"] is not None:
        raise RuntimeError("An unsupported stable carrier onset was assigned")

    action_hash = sha256(ACTION)
    timeline_hash = sha256(TIMELINE)
    if input_audit["optimized_action"]["sha256"] != action_hash:
        raise RuntimeError("optimized_action hash changed")
    if not (
        input_audit["approved_timeline"]["sha256_before"]
        == input_audit["approved_timeline"]["sha256_after"]
        == timeline_hash
    ):
        raise RuntimeError("Approved timeline changed")

    with np.load(ACTION, allow_pickle=False) as source, np.load(
        OUT / "reconstructed_phone_trajectories.npz", allow_pickle=False
    ) as phone, np.load(OUT / "reconstructed_accessory_trajectories.npz", allow_pickle=False) as accessory_npz:
        if source["optimized_action"].shape != (990, 14) or not np.isfinite(source["optimized_action"]).all():
            raise RuntimeError("Source action changed shape or finiteness")
        expected_lookup = np.maximum(np.arange(990, dtype=np.int64) - 7, 0)
        if not np.array_equal(phone["aligned_action_index"], expected_lookup):
            raise RuntimeError("Phone artifact alignment changed")
        if not np.array_equal(accessory_npz["aligned_action_index"], expected_lookup):
            raise RuntimeError("Accessory artifact alignment changed")
        if bool(phone["optimized_action_modified"]):
            raise RuntimeError("Phone audit claims optimized_action modification")
        if bool(phone["approved_timeline_modified"]):
            raise RuntimeError("Phone audit claims timeline modification")
        if bool(phone["physics"]) or bool(accessory_npz["physics"]):
            raise RuntimeError("Physics was enabled")
        if bool(accessory_npz["right_hand_carrier_generated"]):
            raise RuntimeError("A circular right-hand carrier was generated")
        if not np.array_equal(phone["phone_visual_valid_mask"][1, 176:223], np.zeros(47, dtype=bool)):
            raise RuntimeError("Unresolved acquisition interval was fabricated")

    video = OUT / "phone_carrier_candidate_4panel.mp4"
    frame_count = decoded_frames(video)
    if frame_count != 990 or visual["video"]["decoded_frames"] != 990:
        raise RuntimeError(f"Video frame count mismatch: {frame_count}")

    rows = candidates["candidates"]
    pairwise = {(row["first"], row["second"]): row for row in comparison["pairwise"]}
    a_b = pairwise[("CONTACT_START_176", "CHARGER_ANCHORED_530")]
    b_c = pairwise[("CHARGER_ANCHORED_530", "OBSERVED_GEOMETRY_530")]
    pad_rows = pad["candidate_frame_530_metrics"]
    portrait_rows = pad["candidate_frame_223_portrait_metrics"]
    accessory_rows = accessory["candidate_metrics"]

    stable_rows = stable["rows"]
    min_center = min(stable_rows, key=lambda row: row["charger_carrier_phone_center_to_initial_m"])
    min_rotation = min(stable_rows, key=lambda row: row["charger_carrier_phone_rotation_to_initial_deg"])

    report = f"""# Episode 49 phone carrier audit v11b

1. 승인된 +7-frame alignment와 optimized_action은 그대로 유지했고 frame 176을 전체 carry의 rigid lock으로 사용하지 않았다.
2. Frame 530 charger geometry로 역산한 carrier는 pad pose를 정확히 재현하지만 frame-176 carrier와 {a_b['translation_difference_m']*1000:.1f} mm / {a_b['rotation_difference_deg']:.1f} deg 다르다.
3. 한 개의 rigid phone carrier는 검증되지 않았고 accessory 의미도 아직 막혀 있으므로 G1, phasewarp, orientation target은 생성하지 않았다.

## Final status

- `{viability['status']}`
- `{viability['secondary_status']}`
- `BLOCKED_PHONE_CARRIER_MODEL` (interim reclassification)
- `OPTIMIZED_ACTION_TASK_VALIDITY_NOT_FINALIZED_AS_FAILURE`
- `NO_G1_NO_PHASEWARP_NO_ORIENTATION_NO_DEX3_NO_PHYSICS`

## 1. Frame 176: contact-start 또는 rigid carrier lock

- 판정: `CONTACT_START_176_IS_NOT_A_VALID_RIGID_CARRIER_LOCK`.
- CONTACT_START_176와 CHARGER_ANCHORED_530의 차이는 `{a_b['translation_difference_m']*1000:.3f} mm / {a_b['rotation_difference_deg']:.3f} deg`이며 권장 gate `10 mm / 10 deg`를 크게 넘는다.
- charger carrier를 frame 176에 적용하면 known initial phone 대비 center error `{stable_rows[0]['charger_carrier_phone_center_to_initial_m']*1000:.3f} mm`, rotation error `{stable_rows[0]['charger_carrier_phone_rotation_to_initial_deg']:.3f} deg`, table penetration `{-stable_rows[0]['charger_carrier_phone_minimum_table_clearance_m']*1000:.3f} mm`이다.

## 2. CONTACT_START_176 transform

- observed frame `176`, aligned optimized action index `169`.
- translation norm `{rows['CONTACT_START_176']['translation_norm_m']*1000:.3f} mm`, rotation angle `{rows['CONTACT_START_176']['rotation_angle_deg']:.3f} deg`.

```text
{matrix_markdown(rows['CONTACT_START_176']['T_phone_from_left_TCP'])}
```

## 3. CHARGER_ANCHORED_530 transform

- observed frame `530`, aligned optimized action index `523`.
- desired phone center = source pad face center; phone +Y back normal = negative pad outward normal; phone +X long axis = upward pad tangent axis.
- translation norm `{rows['CHARGER_ANCHORED_530']['translation_norm_m']*1000:.3f} mm`, rotation angle `{rows['CHARGER_ANCHORED_530']['rotation_angle_deg']:.3f} deg`.
- multiplication-order unit-test errors are at floating-point zero.

```text
{matrix_markdown(rows['CHARGER_ANCHORED_530']['T_phone_from_left_TCP'])}
```

## 4. OBSERVED_GEOMETRY_530 transform

- observation.state[530]은 carrier geometry calibration에만 사용했으며 motion source는 계속 optimized_action이다.
- translation norm `{rows['OBSERVED_GEOMETRY_530']['translation_norm_m']*1000:.3f} mm`, rotation angle `{rows['OBSERVED_GEOMETRY_530']['rotation_angle_deg']:.3f} deg`.

```text
{matrix_markdown(rows['OBSERVED_GEOMETRY_530']['T_phone_from_left_TCP'])}
```

## 5. Transform differences

- CONTACT_START_176 vs CHARGER_ANCHORED_530: `{a_b['translation_difference_m']*1000:.3f} mm / {a_b['rotation_difference_deg']:.3f} deg` — 다른 rigid relation.
- CHARGER_ANCHORED_530 vs OBSERVED_GEOMETRY_530: `{b_c['translation_difference_m']*1000:.3f} mm / {b_c['rotation_difference_deg']:.3f} deg` — `10 mm / 10 deg` 이내.
- 이는 charger-end geometry가 optimized/action과 observation state 사이에서 일관되며, frame-176 contact-start lock 가정이 불일치의 원인임을 보여준다.

## 6. Stable carrier-valid interval

- frames 176–223에서 known initial phone pose, table clearance, gripper contact를 동시에 만족하는 onset은 없었다.
- `carrier_model_valid_from_observed_frame = null`이며 새로운 event를 만들지 않았다.
- center-only 최솟값은 frame `{min_center['observed_frame']}`의 `{min_center['charger_carrier_phone_center_to_initial_m']*1000:.3f} mm`; rotation 최솟값은 frame `{min_rotation['observed_frame']}`의 `{min_rotation['charger_carrier_phone_rotation_to_initial_deg']:.3f} deg`이다.
- frames 176–222 object state는 unresolved로 저장/표시되며 보간, waypoint, snap을 사용하지 않았다.

## 7. Frame 223 portrait consistency

- CONTACT_START_176 long-axis-to-vertical error: `{portrait_rows['CONTACT_START_176']['long_axis_to_nearest_world_vertical_deg']:.3f} deg`.
- CHARGER_ANCHORED_530: `{portrait_rows['CHARGER_ANCHORED_530']['long_axis_to_nearest_world_vertical_deg']:.3f} deg`.
- OBSERVED_GEOMETRY_530: `{portrait_rows['OBSERVED_GEOMETRY_530']['long_axis_to_nearest_world_vertical_deg']:.3f} deg`.
- charger candidates are contact-start candidate보다 portrait에 가깝지만 단일 carrier 승인 근거로 충분하지 않다.

## 8. Frame 326/341 accessory surface gaps

| Candidate | f326/action | f326 gap | f341/action | f341 attached-hypothesis gap | min f300–350 |
|---|---:|---:|---:|---:|---:|
| CONTACT_START_176 | 319 | {accessory_rows['CONTACT_START_176']['events']['326']['right_gripper_to_ring_surface_gap_m']*1000:.1f} mm | 334 | {accessory_rows['CONTACT_START_176']['events']['341']['right_gripper_to_ring_surface_gap_m']*1000:.1f} mm | {accessory_rows['CONTACT_START_176']['minimum_gap_frames_300_350']['gap_m']*1000:.1f} mm @ f{accessory_rows['CONTACT_START_176']['minimum_gap_frames_300_350']['observed_frame']} |
| CHARGER_ANCHORED_530 | 319 | {accessory_rows['CHARGER_ANCHORED_530']['events']['326']['right_gripper_to_ring_surface_gap_m']*1000:.1f} mm | 334 | {accessory_rows['CHARGER_ANCHORED_530']['events']['341']['right_gripper_to_ring_surface_gap_m']*1000:.1f} mm | {accessory_rows['CHARGER_ANCHORED_530']['minimum_gap_frames_300_350']['gap_m']*1000:.1f} mm @ f{accessory_rows['CHARGER_ANCHORED_530']['minimum_gap_frames_300_350']['observed_frame']} |
| OBSERVED_GEOMETRY_530 | 319 | {accessory_rows['OBSERVED_GEOMETRY_530']['events']['326']['right_gripper_to_ring_surface_gap_m']*1000:.1f} mm | 334 | {accessory_rows['OBSERVED_GEOMETRY_530']['events']['341']['right_gripper_to_ring_surface_gap_m']*1000:.1f} mm | {accessory_rows['OBSERVED_GEOMETRY_530']['minimum_gap_frames_300_350']['gap_m']*1000:.1f} mm @ f{accessory_rows['OBSERVED_GEOMETRY_530']['minimum_gap_frames_300_350']['observed_frame']} |

Frame 341은 verified phone-back attachment를 유지한 독립 경계 가설이다. Frame 326에서 right carrier를 맞춘 뒤 341 gap을 다시 재는 순환 검증은 생성하지 않았다.

## 9. Frame 530 phone-pad error

| Candidate | Center error | Back-normal error | Full rotation error |
|---|---:|---:|---:|
| CONTACT_START_176 | {pad_rows['CONTACT_START_176']['center_to_pad_face_m']*1000:.3f} mm | {pad_rows['CONTACT_START_176']['back_normal_to_desired_deg']:.3f} deg | {pad_rows['CONTACT_START_176']['full_rotation_to_desired_deg']:.3f} deg |
| CHARGER_ANCHORED_530 | {pad_rows['CHARGER_ANCHORED_530']['center_to_pad_face_m']*1000:.3f} mm | {pad_rows['CHARGER_ANCHORED_530']['back_normal_to_desired_deg']:.3f} deg | {pad_rows['CHARGER_ANCHORED_530']['full_rotation_to_desired_deg']:.3f} deg |
| OBSERVED_GEOMETRY_530 | {pad_rows['OBSERVED_GEOMETRY_530']['center_to_pad_face_m']*1000:.3f} mm | {pad_rows['OBSERVED_GEOMETRY_530']['back_normal_to_desired_deg']:.3f} deg | {pad_rows['OBSERVED_GEOMETRY_530']['full_rotation_to_desired_deg']:.3f} deg |

## 10. One fixed carrier viability

- 한 fixed transform이 initial contact, portrait hold, accessory phase, charger placement을 모두 설명하지 못한다.
- `SOURCE_PHONE_RELATION_NOT_SINGLE_RIGID_TRANSFORM`.
- Frame 530 gate는 통과하지만 accessory acquisition/removed gate는 실패하므로 `BLOCKED_ACCESSORY_ATTACHMENT_OR_RIGHT_GRASP_SEMANTICS`이다.
- 단 하나의 diagnostic piecewise candidate만 저장했다: known initial pose → frames 176–222 unresolved → frame 223부터 CHARGER_ANCHORED_530 rigid hypothesis. `NOT YET APPROVED`이며 G1에 사용하지 않는다.

## 11. Visual outputs

- 4-panel 990-frame video: `{visual['video']['path']}`
- Overview contact sheet: `{visual['contact_sheets']['phone_carrier_contact_sheet_overview.png']['path']}`
- Phone close-up sheet: `{visual['contact_sheets']['phone_carrier_contact_sheet_closeup.png']['path']}`
- Accessory close-up sheet: `{visual['contact_sheets']['accessory_contact_sheet_closeup.png']['path']}`
- Transform comparison: `{visual['contact_sheets']['carrier_transform_comparison.png']['path']}`
- Video SHA-256: `{visual['video']['sha256']}`; decoded frames `{frame_count}`, fps `7.5`.

## 12. The single next user decision

`CHARGER_ANCHORED_530`를 frame 223 이후의 **diagnostic carrier hypothesis로만** 유지하고 frames 176–222를 unresolved로 둔 채, accessory local axes·ring center/gap orientation·right contact proxy·frame-326/341 semantics audit으로 진행할지 승인해야 한다. 이는 piecewise carrier를 G1에 승인하는 결정이 아니다.

THE APPROVED ALOHA ACTION, TIMELINE, AND SEVEN-FRAME ALIGNMENT WERE NOT CHANGED
FRAME 176 CONTACT WAS NOT ASSUMED TO BE A RIGID GRASP WITHOUT VERIFICATION
THE PHONE CARRIER TRANSFORM WAS TESTED USING THE KNOWN FRAME-530 CHARGER ATTACHMENT
NO G1 TRAJECTORY, PHASEWARP, OR ORIENTATION TARGET WAS GENERATED
SIMULATION AND SOURCE OBJECT-RELATION DIAGNOSTICS ONLY
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    commands = f"""#!/usr/bin/env bash
set -euo pipefail

# Numeric carrier candidates and source object-state audit only.
MUJOCO_GL=egl /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  /home/jbnu/aloha_g1_dataset/tools/audit_episode49_phone_carrier_v11b.py

# Actual Stationary ALOHA replay with three object-carrier hypotheses.
MUJOCO_GL=egl /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  /home/jbnu/aloha_g1_dataset/tools/render_episode49_phone_carrier_v11b.py

# Fail-closed validation, report, and manifest.
/home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  /home/jbnu/aloha_g1_dataset/tools/finalize_episode49_phone_carrier_v11b.py

# GUI video review.
ffplay -autoexit -loop 0 \\
  {visual['video']['path']}

# GUI contact-sheet review.
xdg-open {visual['contact_sheets']['carrier_transform_comparison.png']['path']}
"""
    command_path = OUT / "commands.sh"
    command_path.write_text(commands, encoding="utf-8")
    command_path.chmod(0o755)

    gate_audit = {
        "status": "NO_DOWNSTREAM_ROBOT_OR_MOTION_GENERATION",
        "primary_carrier_status": viability["status"],
        "secondary_accessory_status": viability["secondary_status"],
        "optimized_action_task_validity_finalized_as_failure": False,
        "forbidden_outputs_present": forbidden,
        "g1_target_generated": False,
        "g1_ik_executed": False,
        "phasewarp_executed": False,
        "orientation_target_generated": False,
        "dex3_executed": False,
        "physics_executed": False,
        "dds_or_publisher_or_hardware": False,
    }
    dump(OUT / "downstream_gate_audit.json", gate_audit)

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(OUT.iterdir()):
        if not path.is_file() or path.name == "run_manifest.json" or path.name.startswith("."):
            continue
        row: dict[str, Any] = {"path": str(path.resolve()), "sha256": sha256(path)}
        if path.suffix == ".mp4":
            row["decoded_frames"] = decoded_frames(path)
        files[path.name] = row

    tool_names = [
        "audit_episode49_phone_carrier_v11b.py",
        "render_episode49_phone_carrier_v11b.py",
        "finalize_episode49_phone_carrier_v11b.py",
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": viability["status"],
        "secondary_status": viability["secondary_status"],
        "interim_reclassification": "BLOCKED_PHONE_CARRIER_MODEL",
        "output_directory": str(OUT.resolve()),
        "immutable_inputs": {
            "optimized_action": {"path": str(ACTION.resolve()), "sha256": action_hash, "modified": False},
            "approved_timeline": {"path": str(TIMELINE.resolve()), "sha256": timeline_hash, "modified": False},
            "action_to_observation_lag_frames": 7,
            "fps": 30.0,
            "event_frames_modified": False,
        },
        "decisions": {
            "frame_176_is_rigid_lock": False,
            "single_rigid_carrier_valid": False,
            "stable_carrier_valid_from_observed_frame": None,
            "phone_on_charger_pose_definition": "PASS",
            "piecewise_diagnostic_approved": False,
            "optimized_action_task_validity_failed": False,
        },
        "files": files,
        "tools": {
            name: {
                "path": str((ROOT / "tools" / name).resolve()),
                "sha256": sha256(ROOT / "tools" / name),
            }
            for name in tool_names
        },
        "safety": {
            "simulation_only": True,
            "source_object_relation_diagnostics_only": True,
            "g1_target": False,
            "g1_ik": False,
            "phasewarp": False,
            "orientation_target": False,
            "dex3": False,
            "physics": False,
            "dds": False,
            "publisher": False,
            "hardware": False,
        },
    }
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "secondary_status": manifest["secondary_status"],
        "video_frames": frame_count,
        "timeline_sha256": timeline_hash,
        "optimized_action_sha256": action_hash,
        "forbidden_outputs_present": any(forbidden.values()),
        "report": str((OUT / "report.md").resolve()),
        "manifest": str((OUT / "run_manifest.json").resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
