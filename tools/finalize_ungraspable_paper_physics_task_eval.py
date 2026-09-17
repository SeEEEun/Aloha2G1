#!/usr/bin/env python3
"""Fail-closed finalization when the policy-independent proxy cannot lift."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/paper_physics_task_eval"
PRE = OUTPUT / "PREEXPERIMENT_FREEZE_MANIFEST.json"
CONFIG = ROOT / "configs/paper_eval_doll_rigid_proxy_v1.json"
SUCCESS = ROOT / "configs/paper_physical_success_definition_v1.json"
GRID0 = OUTPUT / "environment_calibration/material_grid/grid_result.json"
GRID1 = OUTPUT / "environment_calibration/material_grid_review1/grid_result.json"
SUMMARY = OUTPUT / "environment_calibration/calibration_summary.json"
REVIEW = OUTPUT / "environment_calibration/BOUNDED_REVIEW_1.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_frozen_inputs(pre: dict[str, Any]) -> None:
    records = [
        *pre["paper_core_manifests_and_results"],
        *pre["frozen_tables_1_and_2"],
        *pre["scene_and_safety_dependencies"],
    ]
    for method in ("a", "b"):
        records.extend(pre["methods"][method]["files"])
    for episode in pre["episodes"]:
        records.extend(
            (
                episode["source_video"],
                episode["fair_a_trajectory"],
                episode["proposed_b_trajectory"],
            )
        )
    changed = [row["path"] for row in records if sha256_file(Path(row["path"])) != row["sha256"]]
    if changed:
        raise RuntimeError(f"frozen paper input changed: {changed}")


def calibration_rows(grid: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in grid["runs"]:
        report_path = Path(run["report"])
        report = read_json(report_path)
        if report["policy_or_checkpoint_used"] or report["policy_specific_logic"]:
            raise RuntimeError("calibration result consulted a learned policy")
        side = run["side"]
        physical = report["physical_success"]
        rows.append(
            {
                "material": run["material"],
                "side": side,
                "status": report["status"],
                "lift_pass": bool(report[f"{side.upper()}_LIFT_PASS"]),
                "maximum_object_com_world_z_m": physical["diagnostics"]["maximum_object_com_world_z_m"],
                "contact_frame_count": physical["diagnostics"][f"{side}_contact_frame_count"],
                "artifact_checks": report["artifact_checks"],
                "report": record(report_path),
                "event_log": record(Path(report["event_log"])),
            }
        )
    return rows


def main() -> None:
    pre = read_json(PRE)
    verify_frozen_inputs(pre)
    config = read_json(CONFIG)
    success = read_json(SUCCESS)
    initial_grid = read_json(GRID0)
    reviewed_grid = read_json(GRID1)
    summary = read_json(SUMMARY)
    review = read_json(REVIEW)
    if [row["name"] for row in config["material_candidates"]] != ["LOW", "MEDIUM", "HIGH"]:
        raise RuntimeError("material grid changed")
    if any(summary["candidate_passes_both_hands"].values()) or summary["recommended_candidate_under_predeclared_rule"] is not None:
        raise RuntimeError("ungraspable finalizer called despite a bilateral lift pass")
    if bool(config["freeze"]["frozen"]) or bool(config["policy_evaluation_allowed"]):
        raise RuntimeError("failed proxy was incorrectly enabled for policy evaluation")
    if review.get("additional_review_allowed") is not False:
        raise RuntimeError("bounded review contract is not closed")
    for name in ("teacher_forced_trajectories", "preflight", "physics_replay"):
        if any(path.is_file() for path in (OUTPUT / name).rglob("*")):
            raise RuntimeError(f"policy stage unexpectedly contains files: {name}")

    initial_rows = calibration_rows(initial_grid)
    reviewed_rows = calibration_rows(reviewed_grid)
    if len(initial_rows) != 6 or len(reviewed_rows) != 6 or any(row["lift_pass"] for row in reviewed_rows):
        raise RuntimeError("expected two complete, all-fail policy-independent grids")

    environment_dependencies = {
        "proxy_config": record(CONFIG),
        "physical_success_definition": record(SUCCESS),
        "scene_layout": record(Path(config["source_scene"]["layout"])),
        "scene_usd": record(Path(config["source_scene"]["scene_usd"])),
        "g1_preview_usd": record(Path(config["source_scene"]["g1_preview_usd"])),
        "initial_grid": record(GRID0),
        "bounded_review": record(REVIEW),
        "reviewed_grid": record(GRID1),
        "calibration_summary": record(SUMMARY),
    }
    environment_hash = hashlib.sha256(
        json.dumps(environment_dependencies, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    environment = {
        "schema_version": "paper_physics_environment_manifest_v1",
        "status": "NOT_FROZEN_ENVIRONMENT_NOT_GRASPABLE",
        "experiment_name": "PHYSICS_BASED_POLICY_TRAJECTORY_TASK_EVALUATION",
        "environment_hash": environment_hash,
        "graspable": False,
        "left_scripted_lift_pass": False,
        "right_scripted_lift_pass": False,
        "selected_material": None,
        "mass_kg": float(config["object"]["mass_kg"]),
        "collision_proxy": config["object"]["collision_shape"],
        "collision_dimensions_m": config["object"]["collision_dimensions_m"],
        "friction_candidates": config["material_candidates"],
        "policy_evaluation_allowed": False,
        "policy_results_consulted": False,
        "bounded_review_count": 1,
        "dependencies": environment_dependencies,
        "reason": "no LOW/MEDIUM/HIGH material passed both policy-independent scripted lifts after the single bounded review",
    }
    environment_path = OUTPUT / "frozen_environment/PHYSICS_ENVIRONMENT_MANIFEST.json"
    atomic_json(environment_path, environment)

    stage_not_started = {
        "status": "NOT_STARTED_BY_GRASPABILITY_GATE",
        "reason": "policy-independent rigid proxy calibration did not produce a freeze-eligible environment",
        "policy_or_checkpoint_executed": False,
        "real_hardware": False,
    }
    placeholders = {
        OUTPUT / "teacher_forced_trajectories/GENERATION_NOT_STARTED.json": stage_not_started,
        OUTPUT / "preflight/PREFLIGHT_NOT_STARTED.json": stage_not_started,
        OUTPUT / "physics_replay/REPLAY_NOT_STARTED.json": stage_not_started,
        OUTPUT / "videos/VIDEOS_NOT_CREATED.json": stage_not_started,
        OUTPUT / "figures/FIGURES_NOT_CREATED.json": stage_not_started,
        OUTPUT / "tables/TABLE3_NOT_CREATED_ENVIRONMENT_GATE.json": {
            **stage_not_started,
            "table_name": "TABLE 3 — PHYSICS-BASED POLICY TRAJECTORY TASK SUCCESS",
            "fabricated_zero_counts": False,
        },
    }
    for path, payload in placeholders.items():
        atomic_json(path, payload)

    decision = {
        "schema_version": "paper_physics_task_eval_result_v1",
        "status": "PHYSICS_ENVIRONMENT_NOT_GRASPABLE",
        "experiment_name": "PHYSICS_BASED_POLICY_TRAJECTORY_TASK_EVALUATION",
        "rigid_doll_environment": environment,
        "calibration": {
            "initial_grid": initial_rows,
            "single_bounded_review": review,
            "reviewed_grid": reviewed_rows,
            "candidate_passes_both_hands": summary["candidate_passes_both_hands"],
        },
        "teacher_forced_trajectories": {"ACT-A40": "0/8 NOT_STARTED", "ACT-B40": "0/8 NOT_STARTED"},
        "preflight": {"ACT-A40": "NA", "ACT-B40": "NA"},
        "physical_task_success": {"ACT-A40": "NA", "ACT-B40": "NA"},
        "PHYSICAL_PHASE_COMPLETION_SCORE": {"ACT-A40": "NA", "ACT-B40": "NA"},
        "RPL": {"ACT-A40": "NA", "ACT-B40": "NA"},
        "PHYSICAL_SWPE": {"ACT-A40": "NA", "ACT-B40": "NA"},
        "interaction_geometry": {"ACT-A40": "NA", "ACT-B40": "NA"},
        "safety": {"ACT-A40": "NOT_EXECUTED", "ACT-B40": "NOT_EXECUTED"},
        "table3": {"status": "NOT_CREATED", "record": str(OUTPUT / "tables/TABLE3_NOT_CREATED_ENVIRONMENT_GATE.json")},
        "videos": {"status": "NOT_CREATED", "record": str(OUTPUT / "videos/VIDEOS_NOT_CREATED.json")},
        "claim_control": {
            "autonomous_onboard_visual_control": False,
            "closed_loop_g1_policy_success": False,
            "real_g1_success": False,
            "teacher_forced_policy_physics_result": False,
        },
        "frozen_paper_inputs_verified_unchanged": True,
        "tables_1_and_2_unchanged": True,
        "real_g1": "NOT_STARTED_BY_DESIGN",
        "stopped_at_graspability_gate": True,
    }
    metrics_path = OUTPUT / "success_metrics/PHYSICS_ENVIRONMENT_NOT_GRASPABLE.json"
    atomic_json(metrics_path, decision)

    report = f"""# PHYSICS_BASED_POLICY_TRAJECTORY_TASK_EVALUATION

## RIGID DOLL ENVIRONMENT

- graspable: **NO**
- left scripted lift: **FAIL** for LOW, MEDIUM, and HIGH
- right scripted lift: **FAIL** for LOW, MEDIUM, and HIGH
- mass: `{config['object']['mass_kg']:.3f} kg`
- friction: no candidate selected (`LOW`, `MEDIUM`, `HIGH` all failed bilateral calibration)
- collision proxy: `{config['object']['collision_shape']}`; diameter `{config['object']['collision_dimensions_m']['diameter']:.3f} m`
- environment hash: `{environment_hash}`

The first grid exposed a policy-independent staging error: the open-hand approach displaced the free sphere before closure. The single bounded review corrected the primitive to reset the authoritative OPEN hand around the doll before closing. The reviewed grid still produced zero stable hand-contact frames and no lift for either hand at any friction preset, with explosive object displacement flagged by the artifact gate. No second review was performed.

## POLICY STAGES

- Teacher-forced ACT-A40 trajectories: **0/8, NOT STARTED**
- Teacher-forced ACT-B40 trajectories: **0/8, NOT STARTED**
- Preflight: **NA**
- Physics replay: **NA**
- Physical PCS / RPL / SWPE / interaction metrics: **NA**
- Table 3: **NOT CREATED**; zero policy success counts were not fabricated
- Videos: **NOT CREATED**

The proxy was never frozen or enabled for policy evaluation. No ACT checkpoint was executed after the graspability gate failed. Tables 1–2 and all frozen paper-core inputs remain byte-identical.

This result makes no autonomous onboard-G1, closed-loop visual, real-robot, or real-world success claim.

Real G1: **NOT_STARTED_BY_DESIGN**

PHYSICS_ENVIRONMENT_NOT_GRASPABLE
"""
    report_path = OUTPUT / "FINAL_PHYSICS_TASK_EVAL_REPORT.md"
    atomic_text(report_path, report)
    final_manifest = {
        **decision,
        "environment_manifest": record(environment_path),
        "success_metrics": record(metrics_path),
        "final_report": record(report_path),
        "preexperiment_freeze_manifest": record(PRE),
        "final_decision": "PHYSICS_ENVIRONMENT_NOT_GRASPABLE",
    }
    atomic_json(OUTPUT / "FINAL_PHYSICS_TASK_EVAL_MANIFEST.json", final_manifest)
    print(json.dumps({
        "status": final_manifest["final_decision"],
        "environment_hash": environment_hash,
        "report": str(report_path),
        "manifest": str(OUTPUT / "FINAL_PHYSICS_TASK_EVAL_MANIFEST.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
