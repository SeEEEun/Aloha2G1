#!/usr/bin/env python3
"""Finalize the v11 report after the user-approved +7-frame latency audit.

This script is intentionally fail-closed: the approved latency lookup is
validated without modifying the source action or timeline, and the absence of
all downstream G1 artifacts is asserted when the source hand/object semantic
gate is blocked.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
ACTION = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"

APPROVAL = OUT / "action_to_observation_latency.approved.json"
ALIGNMENT = OUT / "source_action_latency_alignment.npz"
RELATIONS = OUT / "source_hand_object_relations_recomputed.json"
PARITY = OUT / "source_parity_metrics.json"
VISUAL = OUT / "latency_aligned_visual_diagnostic.json"

DOWNSTREAM_FILES = [
    "position_only_g1_targets.npz",
    "position_only_g1_arm_trajectory.npz",
    "position_only_ik_metrics.json",
    "position_only_anchor_metrics.json",
    "aloha_source_vs_g1_position_only_4panel.mp4",
    "g1_position_only_overview.mp4",
    "g1_position_only_side.mp4",
    "g1_position_only_top.mp4",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def relation_gap(relations: dict[str, Any], key: str) -> float:
    return float(relations["semantic_grasp_proximity_gate"]["distances"][key])


def main() -> int:
    for path in (ACTION, TIMELINE, APPROVAL, ALIGNMENT, RELATIONS, PARITY, VISUAL):
        if not path.is_file():
            raise FileNotFoundError(path)

    approval = load_json(APPROVAL)
    relations = load_json(RELATIONS)
    parity = load_json(PARITY)
    visual = load_json(VISUAL)

    if approval["status"] != "USER_APPROVED_ACTION_TO_OBSERVATION_LATENCY":
        raise RuntimeError("Latency approval artifact is not authoritative")
    if relations["status"] != "BLOCKED_SOURCE_HAND_OBJECT_RELATION":
        raise RuntimeError("Finalizer is scoped to the current semantic relation blocker")
    if relations["round_trip_tests"]["pass"] is not True:
        raise RuntimeError("Transform round-trip tests must pass before semantic diagnosis")
    if relations["semantic_grasp_proximity_gate"]["pass"] is not False:
        raise RuntimeError("Expected the source hand/object semantic gate to remain blocked")
    if parity["source_relation_gate_passed"] is not False:
        raise RuntimeError("Parity artifact disagrees with the source relation gate")
    if parity["status"] != "BLOCKED_OPTIMIZED_ACTION_TASK_VALIDITY":
        raise RuntimeError("The v11 primary failure classification is not task validity")
    if any(bool(parity[key]) for key in ("phasewarp_executed", "g1_ik_executed", "orientation_sweep_executed", "physics_executed")):
        raise RuntimeError("A forbidden downstream stage was executed")

    with np.load(ACTION, allow_pickle=False) as source, np.load(ALIGNMENT, allow_pickle=False) as aligned:
        optimized_action = source["optimized_action"]
        stored_action = aligned["optimized_action"]
        lookup = aligned["action_sample_index_for_observed_frame"]
        precommand = aligned["diagnostic_precommand_hold_mask"]
        terminal_indices = aligned["post_command_terminal_sample_indices"]
        terminal_values = aligned["post_command_terminal_action"]

    if not np.array_equal(optimized_action, stored_action):
        raise RuntimeError("The aligned artifact changed optimized_action")
    if not np.array_equal(lookup, np.maximum(np.arange(990, dtype=np.int64) - 7, 0)):
        raise RuntimeError("The observed-frame lookup is not max(f-7, 0)")
    if not np.array_equal(precommand, np.arange(990) < 7):
        raise RuntimeError("Frames 0-6 are not the exact diagnostic pre-command hold")
    if not np.array_equal(terminal_indices, np.arange(983, 990, dtype=np.int64)):
        raise RuntimeError("Post-command terminal sample indices were not retained")
    if not np.array_equal(terminal_values, optimized_action[983:990]):
        raise RuntimeError("Post-command terminal sample values changed")

    timeline_hash = sha256(TIMELINE)
    provenance = approval["provenance"]
    if not (
        provenance["approved_timeline_sha256_before"]
        == provenance["approved_timeline_sha256_after"]
        == timeline_hash
    ):
        raise RuntimeError("Approved timeline hash changed")

    downstream_presence = {name: (OUT / name).exists() for name in DOWNSTREAM_FILES}
    if any(downstream_presence.values()):
        raise RuntimeError(f"Forbidden downstream output exists: {downstream_presence}")

    gate_audit = {
        "status": "POSITION_ONLY_G1_NOT_RUN_DUE_TO_BLOCKED_OPTIMIZED_ACTION_TASK_VALIDITY",
        "source_relation_status": relations["status"],
        "source_relation_authoritative": relations["authoritative"],
        "action_to_observation_latency_approved": True,
        "action_to_observation_lag_frames": 7,
        "event_frames_and_video_timestamps_unchanged": True,
        "optimized_action_byte_values_unchanged": True,
        "timeline_sha256": timeline_hash,
        "downstream_files_present": downstream_presence,
        "phasewarp_executed": False,
        "g1_ik_executed": False,
        "orientation_objective_executed": False,
        "physics_executed": False,
        "reason": relations["semantic_grasp_proximity_gate"]["exact_blocker"],
    }
    dump_json(OUT / "downstream_gate_audit.json", gate_audit)

    phone_gap = relation_gap(relations, "left_phone_grasp_m")
    accessory_gap = relation_gap(relations, "right_accessory_grasp_m")
    removed_gap = relation_gap(relations, "right_accessory_removed_m")
    charger_gap = relation_gap(relations, "left_phone_at_charger_m")
    event_errors = parity["latency_aligned_event_tcp_errors_vs_observation_state"]
    event_map = approval["event_to_action_sample"]

    report = f"""# Episode 49 source FK parity v11 — approved latency continuation

1. 승인된 `observed frame f -> optimized_action[f-7]` 규칙을 별도 lookup으로 적용했으며 action, event frame, timestamp, timeline JSON은 변경하지 않았다.
2. 지연 정렬된 optimized_action FK는 observation.state와 태스크 방향 및 event TCP 위치가 대응하고, 모든 transform round-trip이 수치 허용오차를 통과했다.
3. 그러나 moving accessory와 오른손 gripper의 표면 간격이 frame 326/341에서 각각 {accessory_gap*1000:.1f}/{removed_gap*1000:.1f} mm로 20 mm gate를 넘으므로 G1 target/IK는 실행하지 않았다.

## Final status

- `PASS_USER_APPROVED_ACTION_TO_OBSERVATION_LATENCY`
- `SOURCE_FK_NUMERIC_PARITY_PASS`
- `BLOCKED_OPTIMIZED_ACTION_TASK_VALIDITY`
- `BLOCKED_SOURCE_HAND_OBJECT_RELATION`
- `POSITION_ONLY_G1_NOT_RUN`
- `NO_PHASEWARP_RERUN`
- `NO_ORIENTATION_SWEEP_RERUN`

## 1. optimized_action representation

- 절대 joint-position target, 이미 denormalized됨; delta/velocity가 아니다.
- arm channel 단위는 radian, gripper channel 단위는 carriage displacement metre이다.
- 원본 NPZ SHA-256: `{provenance['optimized_action_npz_sha256']}`
- alignment NPZ의 `optimized_action`은 원본 배열과 `np.array_equal == true`이다.

## 2. Exact ALOHA joint mapping

- dataset channel name -> MJCF joint name -> runtime qpos address로 매핑했다.
- follower arm만 사용했고 leader arm, left/right swap, actuator/qpos 혼동, degree 변환, gripper 중복 소비가 없다.
- 상세 매핑: `aloha_joint_mapping_audit.json`

## 3. observation.state / raw action / optimized_action 비교

- 세 입력은 서로 분리된 finite `[990,14]` 배열이며 각각 독립 FK NPZ를 유지한다.
- optimized_action과 raw action의 best lag는 0 frame, 두 action과 observation.state의 command-to-observed lag는 +7 frames이다.
- 지연 정렬 major-phase 방향 cosine의 최솟값(raw action 기준): `{parity['latency_aligned_minimum_direction_cosine_vs_raw_action']:.6f}`.
- observation.state 대비 event TCP 오차: f176 left `{event_errors['frame_176_left_tcp_vs_observation_state_m']*1000:.2f} mm`, f326 right `{event_errors['frame_326_right_tcp_vs_observation_state_m']*1000:.2f} mm`, f341 right `{event_errors['frame_341_right_tcp_vs_observation_state_m']*1000:.2f} mm`, f530 left `{event_errors['frame_530_left_tcp_vs_observation_state_m']*1000:.2f} mm`.

## 4. 기존 v10 source render가 정지해 보인 원인

- `V10_RENDER_OUTPUT_STALE_RELATIVE_TO_NUMERIC_Q`.
- v10은 direct tensor joint write 뒤 Isaac/RTX articulation visual transform 동기화가 되지 않은 stale-render 경로였다. v11은 joint-name 기반 qpos와 MuJoCo forward/render로 교차검증했다.
- 이는 optimized_action이 정적이라는 증거가 아니다.

## 5. Frame alignment 결과

- `action_to_observation_lag_frames = 7`
- `action_sample_for_observed_frame = observed_frame - 7`
- `fps = 30`, `latency_seconds = 0.23333333333333334`
- event/sample: `{', '.join(f'{row["observed_frame"]}->{row["optimized_action_sample"]}' for row in event_map.values())}`.
- frames 0-6은 action[0] diagnostic pre-command hold이다. negative indexing, wrap, extrapolation은 없다.
- samples 983-989는 원본 action record의 post-command terminal samples로 값과 순서를 보존했다.
- timeline SHA-256은 전후 모두 `{timeline_hash}`이며 raw video와 approved timeline은 이동/재작성되지 않았다.

## 6. Root/TCP transform chain 결과

- root transform과 TCP offset `[0.1487, 0, -0.00105] m`은 각각 정확히 한 번 적용했다.
- 최대 relation round-trip position error: `{relations['round_trip_tests']['maximum_position_error_m']:.3e} m`.
- 최대 relation round-trip rotation error: `{relations['round_trip_tests']['maximum_rotation_error_rad']:.3e} rad`.
- tolerance `1e-8 m / 1e-8 rad`를 통과했다.

## 7. Recomputed hand-object transforms

- f176/action[169] phone<-left TCP translation norm: `{relations['relations']['left_phone_grasp']['translation_norm_m']:.6f} m`; phone surface-to-left-gripper gap `{phone_gap*1000:.1f} mm`.
- f326/action[319] accessory<-right TCP translation norm: `{relations['relations']['right_accessory_grasp']['translation_norm_m']:.6f} m`; accessory surface-to-right-gripper gap `{accessory_gap*1000:.1f} mm`.
- f341/action[334] removal displacement norm: `{relations['relations']['right_accessory_removed']['removal_translation_norm_m']:.6f} m`; rotation `{relations['relations']['right_accessory_removed']['removal_rotation_deg']:.3f} deg`; surface gap `{removed_gap*1000:.1f} mm`.
- f530/action[523] phone<-left TCP translation norm: `{relations['relations']['phone_charger_attachment']['translation_norm_m']:.6f} m`; phone carrier gap `{charger_gap*1000:.1f} mm`; reconstructed phone-center/pad-face-center distance `{relations['relations']['phone_charger_attachment']['phone_center_to_charger_pad_face_center_m']:.6f} m`.
- physical hand-object offsets에는 workspace scale 0.42를 적용하지 않았다.

## 8. Source task-validity 판정

- Action/FK motion parity와 transform algebra는 통과했다.
- Source hand-object semantic gate는 실패했다: `{relations['semantic_grasp_proximity_gate']['exact_blocker']}`
- 현재 relation artifact는 `authoritative: false`이며 G1 anchor로 사용할 수 없다.

## 9. Position-only G1 결과

- 실행하지 않았다. Source relation gate 통과 전에는 position-only target, G1 temporal IK, phasewarp, orientation objective를 생성하지 않는 fail-closed 규칙을 적용했다.
- 부재 검증: `downstream_gate_audit.json`의 모든 downstream path가 `false`이다.

## 10. 사용자가 볼 영상

- 승인 지연 정렬 4-panel: `{visual['videos']['aloha_source_latency_aligned_relation_4panel.mp4']['path']}` (990 frames, 7.5 fps)
- 승인 지연 정렬 optimized_action replay: `{visual['videos']['source_optimized_action_latency_aligned_replay.mp4']['path']}` (990 frames, 7.5 fps)
- blocker key-frame sheet: `{visual['contact_sheet']['path']}`
- 4-panel은 raw cam_high | observation.state | raw action[f-7] | optimized_action[f-7]이며, frame 326/341에 accessory error arrow와 수치 gap을 표시한다.

## 11. 다음 승인 항목

현재 immutable 조건을 유지하면 G1 retargeting은 진행할 수 없다. 다음 단계에는 frame-176 rigid phone carrier relation, source accessory attachment transform, 또는 frame-326 grasp semantics 중 어느 가정을 재검토할지에 대한 사용자 지시가 필요하다. 자동 event 이동이나 임의 object offset은 적용하지 않았다.

THE RAW ALOHA VIDEO WAS NOT USED AS A SUBSTITUTE FOR OPTIMIZED_ACTION FK
THE OPTIMIZED_ACTION SOURCE REPLAY WAS VERIFIED BEFORE G1 RETARGETING
ALOHA ROOT, JOINT MAPPING, AND TCP OFFSETS WERE APPLIED EXACTLY ONCE
PHYSICAL HAND-OBJECT OFFSETS WERE NOT SCALED BY WORKSPACE SCALE
NO PHASEWARP OR ORIENTATION OBJECTIVE WAS ALLOWED BEFORE SOURCE FK PARITY
SIMULATION ONLY — NO REAL ROBOT COMMANDS — NO DDS OR PUBLISHER
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    commands = f"""#!/usr/bin/env bash
set -euo pipefail

# Rebuild the base v11 numeric audits and 990-frame Stationary ALOHA replays.
MUJOCO_GL=egl /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  /home/jbnu/aloha_g1_dataset/tools/build_source_fk_parity_v11.py --render

# Apply the approved latency lookup and recompute relations. Exit 2 is the
# expected fail-closed code while the source semantic relation gate is blocked.
set +e
/home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  /home/jbnu/aloha_g1_dataset/tools/continue_source_fk_parity_v11_after_latency_approval.py
relation_status=$?
set -e
if [[ $relation_status -ne 0 && $relation_status -ne 2 ]]; then
  exit $relation_status
fi

# Render the approved-latency source diagnostic; no G1, physics, or hardware.
MUJOCO_GL=egl /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  /home/jbnu/aloha_g1_dataset/tools/render_latency_aligned_source_relation_v11.py

# Revalidate the fail-closed downstream gate and regenerate this report/manifest.
/home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  /home/jbnu/aloha_g1_dataset/tools/finalize_source_fk_parity_v11_after_latency.py

# GUI review: raw | state | raw action[f-7] | optimized_action[f-7].
ffplay -autoexit -loop 0 \\
  {visual['videos']['aloha_source_latency_aligned_relation_4panel.mp4']['path']}

# GUI review: latency-aligned optimized_action source replay only.
ffplay -autoexit -loop 0 \\
  {visual['videos']['source_optimized_action_latency_aligned_replay.mp4']['path']}
"""
    command_path = OUT / "commands.sh"
    command_path.write_text(commands, encoding="utf-8")
    command_path.chmod(0o755)

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "run_manifest.json" and not path.name.startswith("."):
            row: dict[str, Any] = {"path": str(path.resolve()), "sha256": sha256(path)}
            if path.suffix == ".mp4":
                video = visual.get("videos", {}).get(path.name)
                if video is not None:
                    row["decoded_frames"] = int(video["decoded_frames"])
            files[path.name] = row

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_OPTIMIZED_ACTION_TASK_VALIDITY",
        "detail_status": "BLOCKED_SOURCE_HAND_OBJECT_RELATION",
        "passed_statuses": [
            "PASS_USER_APPROVED_ACTION_TO_OBSERVATION_LATENCY",
            "SOURCE_FK_NUMERIC_PARITY_PASS",
            "SOURCE_TRANSFORM_CHAIN_PASS",
        ],
        "downstream_status": "POSITION_ONLY_G1_NOT_RUN",
        "output_directory": str(OUT.resolve()),
        "approval": {
            "action_to_observation_lag_frames": 7,
            "action_sample_for_observed_frame": "observed_frame - 7",
            "fps": 30,
            "latency_seconds": 7 / 30,
            "event_frames_unchanged": True,
            "video_timestamps_unchanged": True,
            "optimized_action_unchanged": True,
            "timeline_sha256": timeline_hash,
            "terminal_samples_retained": list(range(983, 990)),
        },
        "semantic_gate": relations["semantic_grasp_proximity_gate"],
        "files": files,
        "tools": {
            name: {"path": str((ROOT / "tools" / name).resolve()), "sha256": sha256(ROOT / "tools" / name)}
            for name in (
                "build_source_fk_parity_v11.py",
                "continue_source_fk_parity_v11_after_latency_approval.py",
                "render_latency_aligned_source_relation_v11.py",
                "finalize_source_fk_parity_v11_after_latency.py",
            )
        },
        "safety": {
            "simulation_only": True,
            "physics": False,
            "real_robot_commands": False,
            "dds": False,
            "publisher": False,
            "phasewarp": False,
            "g1_target_generation": False,
            "g1_ik": False,
            "orientation_objective": False,
            "dex3_contact_ik": False,
        },
    }
    dump_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "timeline_sha256": timeline_hash,
        "right_accessory_grasp_gap_mm": accessory_gap * 1000,
        "right_accessory_removed_gap_mm": removed_gap * 1000,
        "downstream_outputs_present": any(downstream_presence.values()),
        "manifest": str((OUT / "run_manifest.json").resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
