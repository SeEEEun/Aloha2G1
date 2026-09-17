#!/usr/bin/env python3
"""Close the final-task hard gate with reproducible blocker evidence.

This finalizer performs no simulation and changes no scientific input.  It
audits the completed bounded search, verifies immutable checkpoint/split
identities, and writes fail-closed reports.  It intentionally does not create
a freeze manifest or ACT physical results when the common scripted handoff
gate has not passed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_task_completion_v1"
HANDOFF = OUT / "02_handoff_to_transport_grasp"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
SELECTED = OUT / "01_right_transport_grasp/SELECTED_RIGHT_TRANSPORT_GRASP.json"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
ACT_A = (
    ROOT
    / "outputs/paper_core_ab/act_a40/train/checkpoints/100000/pretrained_model/model.safetensors"
)
ACT_B = (
    ROOT
    / "outputs/paper_core_ab/act_b40/train/checkpoints/020000/pretrained_model/model.safetensors"
)
EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    SELECTED: "10d406038795f2f4dfc38ecdfd408de176d63297f05dd4814fc1a4b29392b8c9",
    HELDOUT: "a86181b049d0f521d1167c2b58bc15f3a7cb6ad87ee9a1f634ef02c04adcc710",
    ACT_A: "7e9fe737c3fd8ad3919cf3887dad58a732f6651e84e1eab7c1af8267ee16912c",
    ACT_B: "4c3c52a853cc242c6ba97fa6fa8d2dde96f6d65e737e291cfb99265f3c1b5198",
}
FORCE_N = 0.015
TABLE_N = 0.015


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_summary(event: dict[str, np.ndarray], stage: str) -> dict[str, Any] | None:
    mask = event["stage"].astype(str) == stage
    if not np.any(mask):
        return None
    forces = np.column_stack(
        [
            event["thumb_force_n"],
            event["index_force_n"],
            event["middle_force_n"],
        ]
    )[mask]
    table = event["table_contact_force_n"][mask]
    position = event["object_position_world_m"][mask]
    speed = np.linalg.norm(event["object_linear_velocity_m_s"][mask], axis=1)
    return {
        "all_three_fraction": float(np.mean(np.all(forces >= FORCE_N, axis=1))),
        "any_digit_fraction": float(np.mean(np.any(forces >= FORCE_N, axis=1))),
        "table_free_fraction": float(np.mean(table < TABLE_N)),
        "mean_force_n_thumb_index_middle": forces.mean(axis=0).tolist(),
        "object_z_min_m": float(np.min(position[:, 2])),
        "object_z_max_m": float(np.max(position[:, 2])),
        "maximum_speed_m_s": float(np.max(speed, initial=0.0)),
    }


def audit_handoff(directory: Path) -> dict[str, Any]:
    offline_candidates = sorted(directory.glob("offline*report.json"))
    offline_path = offline_candidates[0] if offline_candidates else None
    event_candidates = sorted(directory.glob("physics*/event_log.npz"))
    result: dict[str, Any] = {
        "candidate": directory.name,
        "offline_report": str(offline_path) if offline_path else None,
        "physics_run": bool(event_candidates),
    }
    if offline_path:
        offline = read_json(offline_path)
        result["offline_status"] = offline.get("status")
        result["command"] = offline.get("command")
        result["command_sha256"] = offline.get("command_sha256")
        result["offline_collision_frame_counts"] = offline.get("offline", {}).get(
            "collision_frame_counts"
        )
        result["offline_joint_limit_violation_count"] = offline.get("offline", {}).get(
            "joint_limit_violation_count"
        )
    if not event_candidates:
        result["physical_status"] = "NOT_RUN"
        return result
    event_path = event_candidates[0]
    trial_path = event_path.parent / "trial_result.json"
    trial = read_json(trial_path)
    with np.load(event_path, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    runtime_gate = trial.get("runtime_right_three_digit_gate", {})
    result.update(
        {
            "event_log": str(event_path),
            "event_log_sha256": sha256_file(event_path),
            "trial_result": str(trial_path),
            "trial_result_sha256": sha256_file(trial_path),
            "executed_control_frames": int(trial.get("executed_control_frames", 0)),
            "requested_control_frames": int(trial.get("requested_control_frames", 0)),
            "pre_release_three_digit_gate": runtime_gate.get("status", "UNKNOWN"),
            "command_completed": bool(trial.get("command_completed", False)),
            "maximum_object_linear_speed_m_s": float(
                trial.get("artifact_checks", {}).get(
                    "maximum_object_linear_speed_m_s", 0.0
                )
            ),
            "artifact_gate": trial.get("artifact_checks", {}).get("status"),
            "state_restoration_used": bool(
                trial.get("state_restoration", {}).get("used", False)
            ),
            "prohibited_attachment_used": bool(
                trial.get("prohibited_attachment_used", False)
            ),
        }
    )
    for stage in (
        "RIGHT_T4_PRETRANSFER_VERIFICATION",
        "LEFT_RIGHT_COORDINATED_CLEARANCE",
        "RIGHT_SUPPORTED_CONVERGE_TO_R14",
        "RIGHT_CONVERGE_TO_TRANSPORT_GRASP",
        "RIGHT_TRANSPORT_GRASP_VERIFICATION",
        "RIGHT_THREE_DIGIT_VERIFICATION",
        "LEFT_THUMB_RELEASE",
        "RIGHT_POST_RELEASE_RETENTION",
        "RIGHT_VERTICAL_CLEARANCE",
        "RIGHT_HIGH_STABILIZATION",
    ):
        summary = stage_summary(event, stage)
        if summary is not None:
            result.setdefault("stages", {})[stage] = summary
    stages = result.get("stages", {})
    selected_verification = stages.get("RIGHT_TRANSPORT_GRASP_VERIFICATION")
    if selected_verification is None:
        selected_verification = stages.get("RIGHT_THREE_DIGIT_VERIFICATION")
    selected_verification_pass = bool(
        selected_verification is not None
        and selected_verification["all_three_fraction"] >= 0.95
        and selected_verification["table_free_fraction"] >= 0.99
    )
    result["selected_transport_grasp_verification_pass"] = selected_verification_pass
    high = stages.get("RIGHT_HIGH_STABILIZATION")
    high_pass = bool(
        high is None
        or (
            high["all_three_fraction"] >= 0.95
            and high["table_free_fraction"] >= 0.99
        )
    )
    result["high_stabilization_pass"] = high_pass
    if result["pre_release_three_digit_gate"] != "PASS":
        result["physical_status"] = "PRE_RELEASE_GATE_FAIL"
    elif not selected_verification_pass:
        result["physical_status"] = "SELECTED_TRANSPORT_GRASP_VERIFICATION_FAIL"
    elif not high_pass:
        result["physical_status"] = "SELECTED_TRANSPORT_HIGH_STABILIZATION_FAIL"
    elif not result["command_completed"]:
        result["physical_status"] = "INCOMPLETE_AFTER_GATE"
    else:
        result["physical_status"] = "GATE_COMMAND_COMPLETED"
    return result


def main() -> int:
    actual_hashes = {str(path): sha256_file(path) for path in EXPECTED}
    mismatches = {
        str(path): {"expected": expected, "actual": actual_hashes[str(path)]}
        for path, expected in EXPECTED.items()
        if actual_hashes[str(path)] != expected
    }
    if mismatches:
        raise RuntimeError(f"immutable input changed: {mismatches}")

    selected = read_json(SELECTED)
    if selected.get("status") != "PASS" or selected["repeatability"] != {
        **selected["repeatability"],
        "successes": 3,
        "trials": 3,
    }:
        raise RuntimeError("selected RIGHT transport grasp evidence is not 3/3 PASS")

    handoff_rows = [
        audit_handoff(path)
        for path in sorted(HANDOFF.glob("H[0-9][0-9]_*"))
        if path.is_dir()
    ]
    physics_rows = [row for row in handoff_rows if row["physics_run"]]
    coordinated = [
        row
        for row in physics_rows
        if row["candidate"].startswith(("H20_", "H21_", "H22_"))
    ]
    if len(coordinated) != 3 or any(
        row.get("pre_release_three_digit_gate") == "PASS" for row in coordinated
    ):
        raise RuntimeError("bounded coordinated handoff conclusion changed")

    integrated_audit = read_json(HANDOFF / "CANDIDATE_PHYSICS_AUDIT.json")
    r26 = next(
        row
        for row in integrated_audit["candidates"]
        if row["candidate"] == "R26_HANDOFF_R14_INDEX0_P010"
    )
    if r26.get("held_motion_speed_gate_pass", True):
        raise RuntimeError("R26 no longer demonstrates the recorded dynamics failure")

    heldout = read_json(HELDOUT)
    heldout_indices = [int(entry["final_dataset_index"]) for entry in heldout["entries"]]
    split_seed = int(heldout["split_contract"]["split_seed"])

    blocker = {
        "schema_version": "final_task_completion_v1_handoff_blocker",
        "final_status": "HANDOFF_TO_TRANSPORT_GRASP_BLOCKED",
        "right_transport_grasp": {
            "status": "PASS",
            "candidate": selected["candidate_id"],
            "right_only_bin_transport": "PASS",
            "repeatability": selected["repeatability"],
            "report": str(SELECTED),
            "report_sha256": actual_hashes[str(SELECTED)],
        },
        "bounded_search": {
            "right_grasp_physics_candidate_logs": len(
                list((OUT / "01_right_transport_grasp/candidates").glob("*/physics*/event_log.npz"))
            ),
            "handoff_offline_candidates": len(handoff_rows),
            "handoff_physics_candidates": len(physics_rows),
            "handoff_candidates": handoff_rows,
            "integrated_local_grasp_audit": str(
                HANDOFF / "CANDIDATE_PHYSICS_AUDIT.json"
            ),
        },
        "best_near_pass": {
            "candidate": r26["candidate"],
            "pre_release_three_digit_gate": r26["pre_release_three_digit_gate"],
            "post_release_all_three_fraction": r26["stages"][
                "RIGHT_POST_RELEASE_RETENTION"
            ]["all_three_contact_fraction"],
            "vertical_all_three_fraction": r26["stages"][
                "RIGHT_TRANSPORT_VERTICAL_CLEARANCE"
            ]["all_three_contact_fraction"],
            "horizontal_all_three_fraction": r26["stages"][
                "RIGHT_TRANSPORT_TO_BIN"
            ]["all_three_contact_fraction"],
            "maximum_held_object_speed_m_s": r26["maximum_held_object_speed_m_s"],
            "frozen_gate_m_s": 1.0,
            "reason_rejected": "held-object dynamics gate violation during horizontal transport",
            "event_log": r26["event_log"],
            "event_log_sha256": r26["event_log_sha256"],
        },
        "terminal_gate": {
            "handoff_to_transport_grasp": "FAIL",
            "continuous_scripted_full_task": "NOT_RUN_BY_HARD_GATE",
            "scripted_repeatability": "0/0",
            "environment_frozen": False,
            "act_a_b_physics_evaluation": "NOT_RUN_BY_HARD_GATE",
        },
        "paper_inputs_verified_unchanged": {
            "act_a_checkpoint": str(ACT_A.parent),
            "act_a_model_sha256": actual_hashes[str(ACT_A)],
            "act_b_checkpoint": str(ACT_B.parent),
            "act_b_model_sha256": actual_hashes[str(ACT_B)],
            "heldout_manifest": str(HELDOUT),
            "heldout_manifest_sha256": actual_hashes[str(HELDOUT)],
            "heldout_final_dataset_indices": heldout_indices,
            "split_seed": split_seed,
        },
        "prohibited_mechanism_used": False,
        "real_robot_used": False,
    }
    atomic_json(HANDOFF / "HANDOFF_TO_TRANSPORT_GRASP_RESULT.json", blocker)

    csv_path = HANDOFF / "HANDOFF_BOUNDED_SEARCH_AUDIT.csv"
    fields = [
        "candidate",
        "offline_status",
        "physics_run",
        "pre_release_three_digit_gate",
        "physical_status",
        "executed_control_frames",
        "requested_control_frames",
        "command_sha256",
        "event_log_sha256",
    ]
    temporary = csv_path.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in handoff_rows:
            writer.writerow({field: row.get(field) for field in fields})
    os.replace(temporary, csv_path)

    coordinated_lines = []
    for row in coordinated:
        converge = row.get("stages", {}).get("RIGHT_SUPPORTED_CONVERGE_TO_R14", {})
        coordinated_lines.append(
            "| {candidate} | {gate} | {table:.3f} | {any_contact:.3f} | {speed:.3f} |".format(
                candidate=row["candidate"],
                gate=row.get("pre_release_three_digit_gate", "UNKNOWN"),
                table=converge.get("table_free_fraction", 0.0),
                any_contact=converge.get("any_digit_fraction", 0.0),
                speed=row.get("maximum_object_linear_speed_m_s", 0.0),
            )
        )
    result_md = f"""# Handoff to transport-capable grasp result

## Decision

`HANDOFF_TO_TRANSPORT_GRASP = FAIL`

The independent selected RIGHT grasp remains valid and completed the full
RIGHT-only grasp-to-bin task in 3/3 byte-identical simulations.  No tested
collision-free continuous handoff connected the verified LEFT hold to that
opposed transport topology while preserving physical support.

## Selected RIGHT transport grasp

- Candidate: `{selected['candidate_id']}`.
- RIGHT-only bin transport: PASS, 3/3.
- Selected report: `{SELECTED.relative_to(ROOT)}`.
- Report SHA256: `{actual_hashes[str(SELECTED)]}`.
- Final 7D RIGHT vector: `{selected['right_transport_hold_model_order_7d_rad']}`.

## Strongest integrated near-pass

`R26_HANDOFF_R14_INDEX0_P010` passed acquisition, complete LEFT release,
RIGHT-only retention, LEFT retreat, vertical clearance, horizontal contact,
descent, and bin release.  It is rejected because the doll accelerated to
{r26['maximum_held_object_speed_m_s']:.6f} m/s while still commanded as held,
above the unchanged 1.0 m/s dynamics gate.  The wrist itself was moving only
about 0.05 m/s at the event; this was a contact-topology launch, not a command,
IK, collision, or interpolation discontinuity.

## Final coordinated refinement

All three candidates were offline-valid (zero collision and limit failures),
preserved table-free support through the RIGHT bridge, and failed during the
same transition into the selected R14 opposition topology:

| Candidate | Pre-release gate | Conversion table-free fraction | Conversion any-contact fraction | Max object speed (m/s) |
|---|---:|---:|---:|---:|
{chr(10).join(coordinated_lines)}

The 90/120/150-frame bounds therefore did not change the failure mechanism.
No additional wrist grid, T4 path tuning, doll/material tuning, or threshold
relaxation was performed.

## Hard-gate consequence

- Continuous scripted full task: NOT RUN.
- Scripted repeatability: 0/0.
- Common environment/controller frozen: NO.
- ACT-A40 vs ACT-B40 HELDOUT8 physics: NOT RUN.
- A/B checkpoints, split, datasets, and retargeting trajectories: unchanged.

Detailed machine-readable result: `HANDOFF_TO_TRANSPORT_GRASP_RESULT.json`.
Bounded-search table: `HANDOFF_BOUNDED_SEARCH_AUDIT.csv`.

HANDOFF_TO_TRANSPORT_GRASP_BLOCKED
"""
    atomic_text(HANDOFF / "HANDOFF_TO_TRANSPORT_GRASP_RESULT.md", result_md)

    atomic_json(
        OUT / "03_continuous_scripted_task/BLOCKED_BY_HANDOFF_GATE.json",
        {
            "status": "NOT_RUN_BY_HARD_GATE",
            "blocking_gate": "HANDOFF_TO_TRANSPORT_GRASP",
            "blocker_result": str(
                HANDOFF / "HANDOFF_TO_TRANSPORT_GRASP_RESULT.json"
            ),
        },
    )
    atomic_json(
        OUT / "05_freeze/NOT_FROZEN.json",
        {
            "frozen": False,
            "reason": "continuous scripted full-task success and repeatability were not established",
            "freeze_manifest_created": False,
        },
    )
    atomic_json(
        OUT / "06_act_ab_heldout8/NOT_RUN_HARD_GATE.json",
        {
            "status": "NOT_RUN_BY_HARD_GATE",
            "reason": "no valid frozen common physical execution controller",
            "act_a_completed": 0,
            "act_b_completed": 0,
            "act_a_full_success": None,
            "act_b_full_success": None,
            "paper_inputs_verified_unchanged": blocker[
                "paper_inputs_verified_unchanged"
            ],
        },
    )
    atomic_text(
        OUT / "07_paper_results/NOT_GENERATED_HARD_GATE.md",
        "# Physical A/B figures not generated\n\n"
        "The common scripted handoff-to-transport gate failed, so ACT-A/B "
        "physics evaluation and result figures were not run. Existing paper "
        "results were not modified.\n",
    )

    status_md = """# Final task completion status

Updated: 2026-08-30 (Asia/Seoul)

## Terminal gate

`HANDOFF_TO_TRANSPORT_GRASP = FAIL`

## Completed

- Previous positive evidence recovered and hash-verified.
- Transport-capable RIGHT grasp selected.
- RIGHT-only full bin transport repeated 3/3.
- Twenty-two bounded handoff constructions audited; eleven reached physics.
- Final coordinated 90/120/150-frame refinement completed under unchanged physics.

## Not started by hard gate

- Continuous scripted full-task validation.
- Scripted repeatability.
- Final environment/controller freeze.
- ACT-A40/B40 HELDOUT8 physical evaluation.
- Physical A/B figures and videos.

The immutable ACT checkpoints and HELDOUT8 manifest were re-hashed and remain unchanged.
"""
    atomic_text(OUT / "CURRENT_STATUS.md", status_md)

    final_report = f"""# Final overnight completion report

## 1. Previous evidence recovered

The authoritative evidence inventory is at
`00_previous_evidence/PREVIOUS_EVIDENCE_SUMMARY.md`.  It verifies bilateral
P14 standalone graspability, the physics-step-4080 RIGHT-ownership keyframe,
T4/T5 vertical retention, and the prior `TRUE_GRASP_RETENTION_FAILURE`
classification.  Frozen doll config SHA256: `{actual_hashes[str(CONFIG)]}`.

## 2. Selected RIGHT transport-capable grasp

`{selected['candidate_id']}` with 7D vector
`{selected['right_transport_hold_model_order_7d_rad']}`.

## 3. Why it is transport-capable

Its thumb is on the opposing side of the doll while index and middle support
the other side.  The full-transport three-digit fraction is
{selected['contact_topology']['full_transport_three_digit_fraction']:.6f};
the maximum sensor gap is
{selected['contact_topology']['maximum_three_digit_sensor_gap_s']:.4f} s.

## 4. RIGHT-only bin transport result

PASS, 3/3 byte-identical complete grasp-to-bin runs.  The final object COM was
`{selected['right_only_metrics']['terminal_object_com_world_m']}` m and settled
inside the bin.

## 5. Handoff construction result

FAIL.  Eleven bounded handoff candidates reached physics after offline pruning.
The final 90/120/150-frame coordinated refinement kept the doll supported
through the approach bridge but all three candidates lost support during the
same R14 opposition conversion.  R26 was the closest full-tail result but
violated the unchanged held-object speed gate ({r26['maximum_held_object_speed_m_s']:.6f}
m/s versus 1.0 m/s).

## 6. Full scripted continuous result

NOT RUN BY HARD GATE.  A valid handoff into the transport-capable grasp was not
established; running the nominal full task would have violated the user’s gate.

## 7. Scripted repeatability

0/0.  No continuous success existed to repeat.

## 8. Final frozen environment/controller hashes

No final freeze was created.  The prerequisite continuous success and 3/3
repeatability were absent.  `05_freeze/NOT_FROZEN.json` records this explicitly.

## 9. GUI/video paths

No successful continuous-run video exists.  Rendering a failed candidate was
not substituted for the requested success proof.

## 10. ACT-A/B checkpoint identities

- ACT-A40: `{ACT_A.parent}`, model SHA256 `{actual_hashes[str(ACT_A)]}`.
- ACT-B40: `{ACT_B.parent}`, model SHA256 `{actual_hashes[str(ACT_B)]}`.

## 11. HELDOUT8 identities

Final dataset indices: `{heldout_indices}`; seed `{split_seed}`; manifest SHA256
`{actual_hashes[str(HELDOUT)]}`.

## 12. Per-stage ACT-A results

NOT RUN BY HARD GATE (0/8 completed).

## 13. Per-stage ACT-B results

NOT RUN BY HARD GATE (0/8 completed).

## 14. Full Task Success Rate A

Not computed; no frozen common controller.

## 15. Full Task Success Rate B

Not computed; no frozen common controller.

## 16. Controller intervention audit

Not applicable because ACT physical execution did not start.  No policy command
or method-specific trajectory was overridden.

## 17. Failures/blockers

The unresolved blocker is physical connection from the valid LEFT hold into the
independently transport-capable RIGHT opposition topology.  Local handoff grasps
remain one-sided and fail horizontal retention; conversion into R14 loses the
doll before LEFT release.  Doll physics, safety thresholds, and A/B artifacts
were not changed.

## 18. Paper-safe interpretation

This run establishes a valid common RIGHT-only grasp/transport primitive but
does not establish a valid common physical handoff controller.  It therefore
provides blocker evidence, not ACT-A/B physical task-success evidence.

## 19. Claims that remain unsupported

- Continuous scripted physical Doll-Handoff-to-Bin success.
- Repeatable common handoff execution.
- ACT-A40 versus ACT-B40 physical task-success rates.
- Autonomous or real-G1 manipulation.

## Artifact index

- Previous evidence: `00_previous_evidence/PREVIOUS_EVIDENCE_SUMMARY.md`
- RIGHT grasp: `01_right_transport_grasp/SELECTED_RIGHT_TRANSPORT_GRASP.md`
- Handoff audit: `02_handoff_to_transport_grasp/HANDOFF_TO_TRANSPORT_GRASP_RESULT.md`
- Machine result: `02_handoff_to_transport_grasp/HANDOFF_TO_TRANSPORT_GRASP_RESULT.json`
- Candidate CSV: `02_handoff_to_transport_grasp/HANDOFF_BOUNDED_SEARCH_AUDIT.csv`

HANDOFF_TO_TRANSPORT_GRASP_BLOCKED
"""
    atomic_text(OUT / "FINAL_OVERNIGHT_REPORT.md", final_report)

    evidence_files = [
        OUT / "00_previous_evidence/PREVIOUS_EVIDENCE_SUMMARY.md",
        SELECTED,
        OUT / "01_right_transport_grasp/SELECTED_RIGHT_TRANSPORT_GRASP.md",
        HANDOFF / "CANDIDATE_PHYSICS_AUDIT.json",
        HANDOFF / "CANDIDATE_PHYSICS_AUDIT.csv",
        HANDOFF / "CANDIDATE_PHYSICS_AUDIT.md",
        HANDOFF / "HANDOFF_TO_TRANSPORT_GRASP_RESULT.json",
        HANDOFF / "HANDOFF_TO_TRANSPORT_GRASP_RESULT.md",
        HANDOFF / "HANDOFF_BOUNDED_SEARCH_AUDIT.csv",
        OUT / "FINAL_OVERNIGHT_REPORT.md",
        OUT / "CURRENT_STATUS.md",
    ]
    manifest = {
        "schema_version": "final_task_completion_v1_blocker_evidence_manifest",
        "status": "HANDOFF_TO_TRANSPORT_GRASP_BLOCKED",
        "note": "Evidence manifest only; this is not an environment freeze manifest.",
        "files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in evidence_files
        ],
        "immutable_inputs": actual_hashes,
    }
    atomic_json(OUT / "BLOCKER_EVIDENCE_MANIFEST.json", manifest)
    print(json.dumps(blocker, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
