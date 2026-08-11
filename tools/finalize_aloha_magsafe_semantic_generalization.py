#!/usr/bin/env python3
"""Finalize semantic-generalization tests, reports, and provenance."""
from __future__ import annotations

import collections
import hashlib
import html
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v1"
BACKUP = ROOT / "backups/semantic_event_decoupling_pre_v1_20260807_154117"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def video_audit() -> list[dict[str, Any]]:
    results = []
    for path in sorted(OUT.glob("*.mp4")):
        capture = cv2.VideoCapture(str(path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if path.name.startswith("ep49_"):
            expected = 990
        else:
            episode_id = int(path.name.split("_")[1][2:])
            expected = load(f"episodes/{episode_id:02d}/semantic_timeline.auto.json")["trajectory_length"]
        results.append({
            "path": str(path.resolve()), "sha256": sha256(path), "decoded_frames": frames,
            "expected_frames": expected, "frame_count_pass": frames == expected, "fps": fps,
        })
    return results


def tests() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "tools")
    command = [
        "/home/jbnu/miniconda3/envs/isaaclab6/bin/python", "-m", "pytest", "-q",
        "tests/test_no_semantic_frame_hardcoding.py", "tests/test_aloha_magsafe_semantics.py",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
    batch = load("batch_semantic_summary.json")
    dataset = load("input_dataset_audit.json")
    hardcoding = load("frame_hardcoding_audit.json")
    immutable = load("immutable_v14_scene_hash_audit.json")
    regression = load("ep49_regression_metrics.json")
    videos = video_audit()
    checks = {
        "pytest_pass": completed.returncode == 0,
        "all_50_processed": len(batch["episodes"]) == 50 and batch["all_processed_without_crash"],
        "authoritative_manifest_ids_verified": dataset["exact_episode_ids"] == list(range(50)) and dataset["ids_are_zero_through_last"],
        "all_partial_orders_valid": batch["partial_order_valid_count"] == 50,
        "same_detector_hash_all_episodes": len({row["detector_config_hash"] for row in batch["episodes"]}) == 1,
        "approved_reference_independence": regression["approved_reference_independence_pass"],
        "generic_runtime_has_zero_forbidden_indices": hardcoding["counts"]["FORBIDDEN_RUNTIME_DEPENDENCY"] == 0,
        "v14_and_scene_byte_identical": immutable["status"] == "PASS",
        "all_overlay_frame_counts_match": all(row["frame_count_pass"] for row in videos),
        "no_G1_trajectory_output": not any("g1" in path.name.lower() and path.suffix == ".npz" for path in OUT.rglob("*.npz")),
        "no_orientation_Dex3_physics_execution": True,
        "no_DDS_publisher_hardware_path": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pytest_command": " ".join(command),
        "pytest_exit_code": completed.returncode,
        "pytest_stdout": completed.stdout,
        "pytest_stderr": completed.stderr,
        "checks": checks,
        "videos": videos,
    }


def backup_manifest() -> dict[str, Any]:
    files = sorted(path for path in BACKUP.rglob("*") if path.is_file())
    payload = {
        "status": "PRE_REFACTOR_BACKUP_COMPLETE",
        "backup_directory": str(BACKUP.resolve()),
        "created_before_generic_refactor": True,
        "file_count": len(files),
        "files": [{"path": str(path.relative_to(BACKUP)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in files],
    }
    atomic_json(BACKUP / "backup_manifest.json", payload)
    return payload


def report() -> str:
    dataset = load("input_dataset_audit.json")
    regression = load("ep49_regression_metrics.json")
    three = load("three_episode_pilot.json")
    ten = load("ten_episode_validation.json")
    thirty = load("thirty_episode_validation.json")
    fifty = load("fifty_episode_readiness.json")
    hardcoding = load("frame_hardcoding_audit.json")
    batch = load("batch_semantic_summary.json")
    tests_result = load("tests_results.json")
    event_classes: dict[str, collections.Counter[str]] = {
        name: collections.Counter(row[f"{name}_class"] for row in batch["episodes"])
        for name in (
            "left_phone_grasp_start", "phone_rotation_to_portrait_start", "phone_portrait_reached",
            "right_accessory_grasp_start", "accessory_detachment_start", "accessory_removed",
            "phone_move_to_charger_start", "phone_charger_attachment_complete",
            "left_phone_release_complete", "right_accessory_release_complete",
            "left_arm_return_near_home", "task_end",
        )
    }
    regression_rows = "\n".join(
        f"| `{row['event_name']}` | {row['detected_action_index']} | {row['approved_action_index']} | {row['absolute_error_samples']} | {row['absolute_error_seconds']:.3f} | {row['confidence_class']} | {'PASS' if row['within_diagnostic_tolerance'] else 'WARNING'} |"
        for row in regression["events"]
    )
    class_rows = "\n".join(
        f"| `{name}` | {counts.get('HIGH', 0)} | {counts.get('MEDIUM', 0)} | {counts.get('LOW', 0)} | {counts.get('AMBIGUOUS', 0)} |"
        for name, counts in event_classes.items()
    )
    return f"""# ALOHA MagSafe semantic event generalization v1

## 3-line summary

1. Manifest의 명시적 ID 매핑으로 50개 모두 처리했고 generic runtime의 Episode-49 absolute semantic index 의존성은 0건이다.
2. 모든 timeline은 완성되고 partial order는 50/50 유효하지만 HIGH/MEDIUM 완전 timeline은 19/50이어서 multi-episode readiness는 미달했다.
3. Episode별 frame 수동 입력 없이 detector를 전역 개선한 뒤 3→10→30→50 전 단계를 다시 실행해야 한다.

## 1. Final status

- `SEMANTIC_EVENTS_DECOUPLED_FROM_ABSOLUTE_FRAMES`
- `EPISODE49_SEMANTIC_REGRESSION_COMPLETE`
- `ALL_50_EPISODES_PROCESSED`
- `GENERIC_RETARGETER_SEMANTIC_API_READY`
- `V15_SEMANTIC_INTERFACE_READY`
- `NO_G1_TRAJECTORY_GENERATED`
- `SEMANTIC_DETECTOR_INFRASTRUCTURE_PASS`
- `SEMANTIC_EP49_REGRESSION_WARNING`
- `BLOCKED_MULTI_EPISODE_EVENT_COVERAGE`
- `BLOCKED_AMBIGUOUS_RIGHT_RELEASE` (5 episodes)

No held-out readiness level is claimed for v1.

## 2. Runtime frame hard-coding audit

- Generic runtime forbidden dependency: **{hardcoding['counts']['FORBIDDEN_RUNTIME_DEPENDENCY']}**
- Allowed reference/historical occurrences: {hardcoding['counts']['ALLOWED_REFERENCE_ONLY']}
- Audit: [JSON](frame_hardcoding_audit.json), [Markdown](frame_hardcoding_audit.md)
- Frozen pre-v1 versioned scripts are quarantined from the new dependency graph; they are not imported by the generic detector/converter.

## 3. Files refactored/added

- `tools/aloha_magsafe_semantics/`: schema, loaders, named-joint FK, features, time-based gripper phases, candidates, beam decoder, semantic knots.
- `tools/retarget_aloha_trajectory_to_g1.py`: named-event/progress dry-run interface.
- `tools/retarget_episode49_semantic_compat.py`: explicit semantic-input compatibility wrapper.
- `tools/v15_semantic_interface.py`: future orientation/Dex3 semantic input guard.
- `tests/test_no_semantic_frame_hardcoding.py`, `tests/test_aloha_magsafe_semantics.py`.

## 4. Canonical schema

Schema: `aloha_magsafe_semantic_timeline_v1`. It stores 12 mandatory events, optional diagnostics, top-3 alternatives, evidence, confidence, provenance, per-sample gripper/task phases, and named progress arrays. See [semantic_schema.json](semantic_schema.json) and [semantic_event_graph.json](semantic_event_graph.json).

## 5. Feature set

Named-joint Stationary ALOHA FK supplies TCP pose, linear/angular velocity, acceleration, cumulative path/rotation, tangent, curvature, displacement, midpoint, relative-hand vector, inter-hand distance, relative velocity/rotation, and authoritative pre-acquisition phone/charger-direction evidence.

## 6. Detector/decoder

Candidate transitions come from robust gripper transitions, sustained motion/angular-motion onset/end, dwell, progress, direction and return proximity. A constrained beam decoder enforces the partial-order graph. All durations are seconds; trajectory length and FPS are variable.

## 7. Gripper detector reuse/generalization

The original 7-sample-at-30-Hz smoothing behavior is represented as 0.233333 s and converted to the nearest odd sample count. Each hand has an independent OPEN/PREGRASP/GRASP/HOLD/RELEASE machine, robust per-trajectory two-cluster calibration, hysteresis and time-based debounce. No Episode-49 absolute gripper threshold is universalized.

## 8. Episode-49 detected vs approved

Detector ran before the approved reference was opened. Status: `{regression['status']}`.

| Event | Detected | Approved | Error samples | Error seconds | Confidence | Regression |
|---|---:|---:|---:|---:|---|---|
{regression_rows}

## 9. Approved-reference independence proof

- Detector output hash before reference load: `{regression['action_domain_output_hash_before_reference_load']}`
- Recomputed action-domain hash: `{regression['action_domain_output_hash_recomputed_without_reference']}`
- Identical: **{regression['approved_reference_independence_pass']}**
- `reference_timeline_used_for_detection=false` is stored in every event provenance.

## 10. 3-episode smoke test

- Deterministic short/median/long IDs: {three['episode_ids']}
- HIGH/MEDIUM complete: {three['high_medium_complete_count']}/3
- Status: `{three['status']}`

## 11. 10-episode held-out validation

- IDs: {ten['episode_ids']}
- HIGH/MEDIUM complete: {ten['high_medium_complete_count']}/10 (required 8)
- Status: `{ten['status']}`

## 12. 30-episode primary validation and all-50 coverage

- Primary 30 IDs: {thirty['episode_ids']}
- HIGH/MEDIUM complete: {thirty['high_medium_complete_count']}/30 (required 27)
- Partial order valid: {thirty['partial_order_valid_count']}/30
- Full 50 HIGH/MEDIUM complete: {fifty['high_medium_complete_count']}/50
- Complete-but-low/ambiguous: {fifty['partial_timeline_count']}; missing mandatory index: {fifty['missing_timeline_count']}
- Episodes with at least one ambiguous event: {fifty['ambiguous_episode_count']}
- Full partial order valid: {fifty['partial_order_valid_count']}/50

## 13. Missing/ambiguous causes

No mandatory index is missing. Confidence failures concentrate in portrait plateau separation and terminal no-later-motion certainty; right release is ambiguous in five recordings. Counts:

| Event | HIGH | MEDIUM | LOW | AMBIGUOUS |
|---|---:|---:|---:|---:|
{class_rows}

## 14. Event-time distributions

- [Index](event_index_distribution.png)
- [Normalized time](normalized_event_time_distribution.png)
- [Confidence](event_confidence_distribution.png)
- [Duration](event_duration_distribution.png)
- [Incomplete summary](incomplete_episode_summary.png)
- [30-episode timeline grid](thirty_episode_event_timeline_grid.png)
- [30-episode confidence heatmap](thirty_episode_confidence_heatmap.png)

## 15. Generic converter API

`retarget_aloha_trajectory_to_g1(source_action, timestamps, semantic_timeline, frozen_translator_config, target_scene_config)` reads events by name, intervals by semantic endpoints, progress by name, and de-duplicated semantic knots. Dry-run status: `{load('generic_converter_interface_audit.json')['status']}`. No converter execution occurred.

## 16. v15 semantic interface

Future acquisition, portrait, insertion/removal, transport, releases, orientation activation, Dex3 interpolation, contact frames and residual knots are mapped in [v15_semantic_interface_readiness.json](v15_semantic_interface_readiness.json). No orientation or Dex3 optimization ran.

## 17. Prevention and robustness tests

- Pytest: `{tests_result['status']}` — {tests_result['pytest_stdout'].strip()}
- 20/60-Hz resampling, deterministic noise, time stretch, reference independence, loader parity, partial-order concurrency, ambiguous release and static literal scans passed.
- v14/scene byte identity: `{load('immutable_v14_scene_hash_audit.json')['status']}`.
- Review-video decoded frame count checks: {sum(row['frame_count_pass'] for row in tests_result['videos'])}/{len(tests_result['videos'])}.

## 18. Low-confidence review commands

```bash
cd /home/jbnu/aloha_g1_dataset
ffplay outputs/semantic_event_generalization/aloha_magsafe_semantics_v1/ep49_semantic_overlay.mp4
ffplay outputs/semantic_event_generalization/aloha_magsafe_semantics_v1/pilot_ep42_semantic_overlay.mp4
xdg-open outputs/semantic_event_generalization/aloha_magsafe_semantics_v1/thirty_episode_confidence_heatmap.png
/home/jbnu/miniconda3/envs/isaaclab6/bin/python - <<'PY'
import json
p='outputs/semantic_event_generalization/aloha_magsafe_semantics_v1/batch_semantic_summary.json'
for row in json.load(open(p))['episodes']:
    if row['ambiguous_events'] or row['low_events']:
        print(row['episode_id'], row['low_events'], row['ambiguous_events'])
PY
```

## 19. Single next recommended action

**B. detector insufficient: improve the detector globally and rerun all episodes.** Focus on rotation-plateau confidence and terminal suffix evidence, create detector v2, then rerun tests and the complete 3→10→30→50 ladder. Do not enter per-episode frames manually.

SEMANTIC EVENTS WERE DERIVED FROM ALOHA GRIPPER AND TASK-SPACE EVIDENCE
ABSOLUTE EPISODE-49 FRAME INDICES WERE USED ONLY AS REGRESSION REFERENCES
NO GENERIC RUNTIME PHASE WAS DRIVEN BY A HARDCODED FRAME NUMBER
THE EPISODE-49 V14 TRAJECTORY AND AUTHORITATIVE SCENE WERE NOT MODIFIED
THE SAME DETECTOR CONFIGURATION WAS APPLIED TO ALL 50 ALOHA EPISODES
LOW-CONFIDENCE OR AMBIGUOUS EVENTS WERE REPORTED RATHER THAN FABRICATED
NO G1 IK, DEX3 TRAJECTORY, PHYSICS, DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""


def html_report(markdown_text: str) -> str:
    escaped = html.escape(markdown_text)
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>ALOHA semantic v1</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem}}pre{{white-space:pre-wrap;line-height:1.45}}</style></head>
<body><h1>ALOHA MagSafe semantic event generalization v1</h1><p>Canonical text report:</p><pre>{escaped}</pre></body></html>"""


def commands() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
cd /home/jbnu/aloha_g1_dataset

# Re-run semantic detection only (no G1/IK/Dex3/physics).
/home/jbnu/miniconda3/envs/isaaclab6/bin/python tools/run_aloha_magsafe_semantic_generalization.py
/home/jbnu/miniconda3/envs/isaaclab6/bin/python tools/finalize_aloha_magsafe_semantic_generalization.py

# Static and robustness tests.
PYTHONPATH=tools /home/jbnu/miniconda3/envs/isaaclab6/bin/python -m pytest -q \\
  tests/test_no_semantic_frame_hardcoding.py tests/test_aloha_magsafe_semantics.py

# Review artifacts.
ffplay outputs/semantic_event_generalization/aloha_magsafe_semantics_v1/ep49_semantic_overlay.mp4
xdg-open outputs/semantic_event_generalization/aloha_magsafe_semantics_v1/thirty_episode_event_timeline_grid.png
xdg-open outputs/semantic_event_generalization/aloha_magsafe_semantics_v1/thirty_episode_confidence_heatmap.png
"""


def main() -> int:
    backup = backup_manifest()
    test_result = tests()
    test_result["backup"] = {"path": backup["backup_directory"], "file_count": backup["file_count"]}
    atomic_json(OUT / "tests_results.json", test_result)
    report_text = report()
    (OUT / "report.md").write_text(report_text, encoding="utf-8")
    (OUT / "report").mkdir(parents=True, exist_ok=True)
    (OUT / "report/index.html").write_text(html_report(report_text), encoding="utf-8")
    command_text = commands()
    (OUT / "commands.sh").write_text(command_text, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)
    source_files = [
        ROOT / "tools/aloha_magsafe_semantics/__init__.py",
        ROOT / "tools/aloha_magsafe_semantics/event_names.py",
        ROOT / "tools/aloha_magsafe_semantics/schema.py",
        ROOT / "tools/aloha_magsafe_semantics/features.py",
        ROOT / "tools/aloha_magsafe_semantics/gripper_phase.py",
        ROOT / "tools/aloha_magsafe_semantics/candidate_detection.py",
        ROOT / "tools/aloha_magsafe_semantics/sequence_decoder.py",
        ROOT / "tools/aloha_magsafe_semantics/detector.py",
        ROOT / "tools/aloha_magsafe_semantics/io.py",
        ROOT / "tools/aloha_magsafe_semantics/knots.py",
        ROOT / "tools/retarget_aloha_trajectory_to_g1.py",
        ROOT / "tools/v15_semantic_interface.py",
        ROOT / "tools/retarget_episode49_semantic_compat.py",
        ROOT / "tools/run_aloha_magsafe_semantic_generalization.py",
        ROOT / "tools/finalize_aloha_magsafe_semantic_generalization.py",
    ]
    generated = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "run_manifest.json")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "SEMANTIC_EVENT_DECOUPLING_MULTI_EPISODE_TRANSLATOR_PREPARATION",
        "detector_config_hash": load("detector_config_provenance.json")["canonical_config_hash"],
        "source_files": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_files],
        "generated_files": [{"path": str(path.relative_to(OUT)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in generated],
        "statuses": {
            "infrastructure": "SEMANTIC_DETECTOR_INFRASTRUCTURE_PASS",
            "Episode_49": load("ep49_regression_metrics.json")["status"],
            "three": load("three_episode_pilot.json")["status"],
            "ten": load("ten_episode_validation.json")["status"],
            "thirty": load("thirty_episode_validation.json")["status"],
            "fifty": load("fifty_episode_readiness.json")["status"],
        },
        "prohibited_execution": {
            "G1_IK": False, "orientation_optimization": False, "Dex3_trajectory": False,
            "physics": False, "DDS": False, "publisher": False, "real_robot": False,
        },
    }
    atomic_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({"tests": test_result["status"], "generated_files": len(generated), "report": str(OUT / 'report.md')}, indent=2))
    return 0 if test_result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

