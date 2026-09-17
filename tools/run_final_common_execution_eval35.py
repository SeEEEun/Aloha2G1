#!/usr/bin/env python3
"""Run and aggregate ACT-A40 versus ACT-B40 EVAL35 physical evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_execution_layer import load_frozen_evaluator, sha256_file
from tools.common_execution_isaac_runtime import verify_execution_freeze


ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
SYSTEM_PYTHON = Path("/usr/bin/python3")
LAUNCHER = ROOT / "tools/run_common_execution_layer_isaac.py"
SCORER = ROOT / "tools/score_common_execution_full_task.py"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
OUT = ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer"
COMMAND_MANIFEST = OUT / "EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
BASE_EVAL10 = (
    ROOT
    / "outputs/final_contact_constrained_eval/04_eval10_preparation/EVAL10_RETARGETING_MANIFEST.json"
)
EVAL35 = OUT / "EVAL35_MANIFEST.json"
ENVIRONMENT_FREEZE = (
    ROOT / "outputs/final_contact_constrained_eval/03_freeze/FREEZE_MANIFEST.json"
)
EVALUATOR_FREEZE = (
    ROOT
    / "outputs/final_representation_neutral_eval/00_frozen_evaluator/EVALUATOR_FREEZE_MANIFEST.json"
)
EXECUTION_FREEZE = OUT / "COMMON_EXECUTION_LAYER_FREEZE_MANIFEST.json"
RUNS = OUT / "runs"
STAGES = (
    "PHYSICAL_READINESS",
    "LEFT_GRASP",
    "HANDOFF",
    "RIGHT_OWNERSHIP",
    "NO_DROP_TO_BIN",
    "BIN_ENTRY",
    "BIN_SETTLE",
    "FULL_TASK_SUCCESS",
)
EXPECTED_LEGACY_NEW = {
    8: ("new_unseen_20260901_140555", "GoPark_20260901_140555"),
    9: ("new_unseen_20260901_140822", "GoPark_20260901_140822"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_hash_manifest(path: Path) -> None:
    manifest = read_json(path)
    if manifest.get("status") != "FROZEN":
        raise RuntimeError(f"manifest is not FROZEN: {path}")
    for row in manifest.get("files", []):
        dependency = Path(row["path"]).resolve()
        if not dependency.is_file():
            raise FileNotFoundError(f"missing frozen dependency: {dependency}")
        actual = sha256_file(dependency)
        if actual != row["sha256"]:
            raise RuntimeError(
                f"frozen dependency hash drift: {dependency}: {actual} != {row['sha256']}"
            )


def tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def verify_eval35_identity() -> dict[str, Any]:
    base = read_json(BASE_EVAL10)
    if base.get("evaluation_set") != "EVAL10" or len(base.get("eval_entries", [])) != 10:
        raise RuntimeError("authoritative base EVAL10 identity is unavailable")
    base_new = {int(row["eval_index"]): row for row in base.get("new_unseen_2", [])}
    if set(base_new) != {8, 9}:
        raise RuntimeError("base EVAL10 does not contain exact NEW_UNSEEN_2 indices")
    for index, (stable_id, source_name) in EXPECTED_LEGACY_NEW.items():
        row = base_new[index]
        if row.get("stable_episode_id") != stable_id or row.get("source_name") != source_name:
            raise RuntimeError(f"base EVAL10 exact episode mismatch at {index}")
    eval35 = read_json(EVAL35)
    if eval35.get("status") != "PASS_IDENTITY_FROZEN":
        raise RuntimeError("EVAL35 identity manifest did not pass")
    if eval35.get("evaluation_set") != "EVAL35" or eval35.get("source_count") != 35:
        raise RuntimeError("evaluation identity is not exact EVAL35")
    if eval35.get("base_eval10_manifest_sha256") != sha256_file(BASE_EVAL10):
        raise RuntimeError("base EVAL10 manifest hash drift")
    entries = eval35.get("eval_entries", [])
    if entries[:10] != base["eval_entries"]:
        raise RuntimeError("EVAL35 indices 0..9 do not preserve EVAL10 verbatim")
    new_rows = eval35.get("new_20260902_evaluation_only", [])
    expected_sources = sorted(
        path for path in (ROOT / "raw_recordings").glob("GoPark_20260902_*") if path.is_dir()
    )
    if len(expected_sources) != 25 or len(new_rows) != 25:
        raise RuntimeError("EVAL35 does not contain every exact GoPark_20260902_* source")
    if [int(row["eval_index"]) for row in new_rows] != list(range(10, 35)):
        raise RuntimeError("EVAL35 new evaluation-only indices are not exact 10..34")
    if [row["source_name"] for row in new_rows] != [path.name for path in expected_sources]:
        raise RuntimeError("EVAL35 source names/order drifted")
    constraints = eval35.get("evaluation_only_constraints", {})
    prohibited = (
        "evaluator_calibration_allowed",
        "evaluator_tuning_allowed",
        "training_allowed",
        "checkpoint_selection_allowed",
        "controller_or_environment_tuning_allowed",
        "episode_replacement_allowed",
        "performance_based_selection_allowed",
        "physical_rollout_before_valid_frozen_evaluator_allowed",
    )
    if any(constraints.get(key) is not False for key in prohibited):
        raise RuntimeError("EVAL35 evaluation-only exclusions are incomplete")
    evaluator_files = sorted(
        path for path in EVALUATOR_FREEZE.parent.rglob("*") if path.is_file()
    )
    for row, source in zip(new_rows, expected_sources, strict=True):
        if row.get("status") != "PASS" or row.get("evaluation_only") is not True:
            raise RuntimeError(f"invalid evaluation-only source record: {source.name}")
        for key in (
            "evaluator_calibration_allowed",
            "evaluator_tuning_allowed",
            "training_allowed",
            "checkpoint_selection_allowed",
            "controller_tuning_allowed",
            "outcome_used_for_episode_selection",
        ):
            if row.get(key) is not False:
                raise RuntimeError(f"source exclusion missing for {source.name}: {key}")
        file_checks = (
            (source / "data/chunk-000/episode_000000.parquet", row["source_parquet_sha256"]),
            (source / "meta/info.json", row["source_metadata_sha256"]),
            (source / "meta/tasks.jsonl", row["source_task_metadata_sha256"]),
        )
        for path, declared in file_checks:
            if sha256_file(path) != declared:
                raise RuntimeError(f"EVAL35 source hash drift: {path}")
        contamination_needles = [source.name, row["source_parquet_sha256"]]
        for camera_name, camera in row["cameras"].items():
            images = sorted(Path(camera["image_root"]).glob("frame_*.png"))
            if tree_sha256(images) != camera["frame_tree_sha256"]:
                raise RuntimeError(f"EVAL35 camera hash drift: {source.name}: {camera_name}")
            contamination_needles.append(camera["frame_tree_sha256"])
        for evaluator_path in evaluator_files:
            data = evaluator_path.read_bytes()
            if any(needle.encode("utf-8") in data for needle in contamination_needles):
                raise RuntimeError(
                    f"evaluation-only EVAL35 source referenced by evaluator: {evaluator_path}"
                )
    return eval35


def verify_eval35() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    eval35 = verify_eval35_identity()
    if not COMMAND_MANIFEST.is_file():
        raise FileNotFoundError(
            "EVAL35 ACT command manifest is unavailable; inference/preparation must remain "
            "separate from evaluator calibration and may run only after evaluator freeze"
        )
    commands = read_json(COMMAND_MANIFEST)
    records = {
        (str(row["method"]), int(row["eval_index"])): row
        for row in commands.get("records", [])
    }
    expected = {
        (f"ACT-{method}40", index)
        for method in ("A", "B")
        for index in range(35)
    }
    if set(records) != expected:
        raise RuntimeError("physical command manifest is not exact ACT-A40/B40 x EVAL35")
    for key, row in records.items():
        command = Path(row["physical_command"])
        if sha256_file(command) != row["physical_command_sha256"]:
            raise RuntimeError(f"ACT physical command hash drift: {key}")
        reference = Path(row["semantic_reference"])
        if sha256_file(reference) != row["semantic_reference_sha256"]:
            raise RuntimeError(f"semantic reference hash drift: {key}")
    identities = {int(row["eval_index"]): row["stable_episode_id"] for row in eval35["eval_entries"]}
    for index, stable_id in identities.items():
        for method in ("A", "B"):
            if records[(f"ACT-{method}40", index)]["stable_episode_id"] != stable_id:
                raise RuntimeError(f"EVAL35 command identity mismatch at {index}")
    return eval35, records


def prerequisites() -> tuple[Any, dict[tuple[str, int], dict[str, Any]]]:
    if not EVALUATOR_FREEZE.is_file():
        raise FileNotFoundError(f"frozen evaluator unavailable: {EVALUATOR_FREEZE}")
    evaluator = load_frozen_evaluator(EVALUATOR_FREEZE)
    if not EXECUTION_FREEZE.is_file():
        raise FileNotFoundError(f"execution freeze unavailable: {EXECUTION_FREEZE}")
    verify_execution_freeze(EXECUTION_FREEZE, evaluator.evaluator_sha256)
    verify_hash_manifest(ENVIRONMENT_FREEZE)
    _, records = verify_eval35()
    return evaluator, records


def run_one(method: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    method_lower = method.lower()
    output = RUNS / f"act_{method_lower}40/eval_{index:02d}_{row['stable_episode_id']}"
    completion = output / "RUN_MANIFEST.json"
    if completion.is_file():
        cached = read_json(completion)
        required = (
            output / "event_log.npz",
            output / "robot_bin_contacts.npz",
            output / "COMMON_EXECUTION_RUNTIME_SUMMARY.json",
            output / "COMMON_EXECUTION_TASK_RESULT.json",
        )
        if cached.get("status") not in {"PHYSICAL_PASS", "PHYSICAL_FAIL"} or any(
            not value.is_file() for value in required
        ):
            raise RuntimeError(f"incomplete cached run: {output}")
        return cached
    if output.exists():
        raise FileExistsError(f"refusing to overwrite incomplete run: {output}")
    output.mkdir(parents=True)
    invocation = [
        str(ISAAC),
        str(LAUNCHER),
        "--evaluator-freeze-manifest",
        str(EVALUATOR_FREEZE),
        "--execution-freeze-manifest",
        str(EXECUTION_FREEZE),
        "--config",
        str(CONFIG),
        "--side",
        "right",
        "--geometry",
        "FROZEN_COMPRESSED_SHORT_55",
        "--profile",
        "P14",
        "--output-dir",
        str(output),
        "--scripted-command-path",
        str(row["physical_command"]),
        "--object-spawn-side",
        "left",
        "--audit-robot-bin",
        "--full-task-audit",
        "--bin-height-m",
        "0.150",
        "--bin-rim-bevel-m",
        "0.003",
        "--headless",
    ]
    invocation_manifest = {
        "schema_version": "representation_neutral_common_execution_invocation_v1",
        "method": f"ACT-{method}40",
        "eval_index": index,
        "stable_episode_id": row["stable_episode_id"],
        "provenance": row["provenance"],
        "physical_command": row["physical_command"],
        "physical_command_sha256": row["physical_command_sha256"],
        "evaluator_freeze_manifest": str(EVALUATOR_FREEZE),
        "evaluator_freeze_manifest_sha256": sha256_file(EVALUATOR_FREEZE),
        "execution_freeze_manifest": str(EXECUTION_FREEZE),
        "execution_freeze_manifest_sha256": sha256_file(EXECUTION_FREEZE),
        "environment_freeze_manifest": str(ENVIRONMENT_FREEZE),
        "environment_freeze_manifest_sha256": sha256_file(ENVIRONMENT_FREEZE),
        "continuous_contact_constrained_run": True,
        "arm_trajectory_source": "method-specific frozen ACT deployment-safe prediction",
        "wrist_rescue_allowed": False,
        "common_override_scope": "DEX3_ONLY",
        "object_pose_writes_during_execution_allowed": False,
        "state_restoration_allowed": False,
        "invocation": invocation,
    }
    atomic_json(output / "INVOCATION_MANIFEST.json", invocation_manifest)
    started = time.monotonic()
    with (output / "engine.log").open("w", encoding="utf-8") as stream:
        engine = subprocess.run(
            invocation,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    required_engine = (
        output / "event_log.npz",
        output / "robot_bin_contacts.npz",
        output / "trial_result.json",
        output / "COMMON_EXECUTION_RUNTIME_SUMMARY.json",
    )
    if engine.returncode not in (0, 2) or any(not value.is_file() for value in required_engine):
        atomic_json(
            output / "INFRASTRUCTURE_FAILURE.json",
            {
                "engine_returncode": engine.returncode,
                "missing": [str(value) for value in required_engine if not value.is_file()],
            },
        )
        raise RuntimeError(f"Isaac infrastructure failure: {output / 'engine.log'}")
    scored = subprocess.run(
        [str(SYSTEM_PYTHON), str(SCORER), "--run-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "scorer.log").write_text(scored.stdout + scored.stderr, encoding="utf-8")
    result_path = output / "COMMON_EXECUTION_TASK_RESULT.json"
    if scored.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"physical scorer failure: {output / 'scorer.log'}")
    result = read_json(result_path)
    artifacts = [*required_engine, result_path, output / "COMMON_EXECUTION_TASK_RESULT.md"]
    completion_value = {
        **{key: invocation_manifest[key] for key in ("method", "eval_index", "stable_episode_id", "provenance")},
        "schema_version": "representation_neutral_common_execution_run_v1",
        "status": "PHYSICAL_PASS" if result["outcomes"]["FULL_TASK_SUCCESS"] else "PHYSICAL_FAIL",
        "wall_seconds": time.monotonic() - started,
        "engine_exit_code": engine.returncode,
        "scorer_exit_code": scored.returncode,
        "outcomes": result["outcomes"],
        "first_failure_stage": result["first_failure_stage"],
        "fairness_audit": result["fairness_audit"],
        "artifacts": {
            value.name: {"path": str(value), "sha256": sha256_file(value)}
            for value in artifacts
        },
    }
    atomic_json(completion, completion_value)
    return completion_value


def aggregate(results: list[dict[str, Any]], evaluator: Any) -> None:
    if len(results) != 70:
        raise RuntimeError("refusing partial EVAL35 aggregation")
    by_method: dict[str, list[dict[str, Any]]] = {
        method: sorted(
            [row for row in results if row["method"] == method],
            key=lambda row: int(row["eval_index"]),
        )
        for method in ("ACT-A40", "ACT-B40")
    }
    if any(len(rows) != 35 for rows in by_method.values()):
        raise RuntimeError("A/B aggregation is not 35 + 35")
    for method, filename in (
        ("ACT-A40", "ACT_A_EVAL35_FINAL_PHYSICAL.json"),
        ("ACT-B40", "ACT_B_EVAL35_FINAL_PHYSICAL.json"),
    ):
        rows = by_method[method]
        stage_counts = {
            stage: sum(bool(row["outcomes"][stage]) for row in rows) for stage in STAGES
        }
        atomic_json(
            OUT / filename,
            {
                "schema_version": "representation_neutral_act_eval35_final_physical_v1",
                "method": method,
                "evaluation_set": "EVAL35",
                "frozen_evaluator_sha256": evaluator.evaluator_sha256,
                "execution_layer_freeze_sha256": sha256_file(EXECUTION_FREEZE),
                "episodes": rows,
                "stage_success_count": stage_counts,
                "stage_success_percent": {
                    stage: 100.0 * count / 35.0 for stage, count in stage_counts.items()
                },
            },
        )
    with (OUT / "ACT_AB_STAGE_SUCCESS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("method", "stage", "successes", "total", "success_percent")
        )
        writer.writeheader()
        for method, rows in by_method.items():
            for stage in STAGES:
                count = sum(bool(row["outcomes"][stage]) for row in rows)
                writer.writerow(
                    {
                        "method": method,
                        "stage": stage,
                        "successes": count,
                        "total": 35,
                        "success_percent": f"{100.0 * count / 35.0:.1f}",
                    }
                )
    with (OUT / "ACT_AB_FIRST_FAILURE_STAGE.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "method",
                "eval_index",
                "stable_episode_id",
                "provenance",
                "first_failure_stage",
                *STAGES,
            ),
        )
        writer.writeheader()
        for method, rows in by_method.items():
            for row in rows:
                writer.writerow(
                    {
                        "method": method,
                        "eval_index": row["eval_index"],
                        "stable_episode_id": row["stable_episode_id"],
                        "provenance": row["provenance"],
                        "first_failure_stage": row["first_failure_stage"] or "NONE",
                        **{stage: int(bool(row["outcomes"][stage])) for stage in STAGES},
                    }
                )
    a = sum(row["outcomes"]["FULL_TASK_SUCCESS"] for row in by_method["ACT-A40"])
    b = sum(row["outcomes"]["FULL_TASK_SUCCESS"] for row in by_method["ACT-B40"])
    stage_rows = []
    for stage in STAGES:
        ac = sum(row["outcomes"][stage] for row in by_method["ACT-A40"])
        bc = sum(row["outcomes"][stage] for row in by_method["ACT-B40"])
        stage_rows.append(f"| {stage.replace('_', ' ').title()} | {ac}/35 | {bc}/35 |")
    matrix_rows = []
    for index in range(35):
        ar = by_method["ACT-A40"][index]
        br = by_method["ACT-B40"][index]
        matrix_rows.append(
            f"| {index} | {ar['stable_episode_id']} | "
            f"{'PASS' if ar['outcomes']['FULL_TASK_SUCCESS'] else ar['first_failure_stage']} | "
            f"{'PASS' if br['outcomes']['FULL_TASK_SUCCESS'] else br['first_failure_stage']} |"
        )
    text = "\n".join(
        [
            "# ACT-A40 vs ACT-B40 final physical Full Task Success",
            "",
            f"Frozen evaluator SHA256: `{evaluator.evaluator_sha256}`",
            f"Common execution freeze manifest SHA256: `{sha256_file(EXECUTION_FREEZE)}`",
            "",
            f"ACT-A Full Task Success = **{a}/35 = {100.0*a/35.0:.1f}%**",
            f"ACT-B Full Task Success = **{b}/35 = {100.0*b/35.0:.1f}%**",
            f"B - A = **{100.0*(b-a)/35.0:.1f} percentage points**",
            "",
            "## Stage success",
            "",
            "| Stage | ACT-A40 | ACT-B40 |",
            "|---|---:|---:|",
            *stage_rows,
            "",
            "## EVAL35 episode matrix",
            "",
            "| Eval | Episode | ACT-A40 | ACT-B40 |",
            "|---:|---|---|---|",
            *matrix_rows,
            "",
            "Primary criterion: GRASP -> NEVER DROP -> HANDOFF -> NEVER DROP -> DOLL IN BIN -> SETTLED.",
        ]
    )
    (OUT / "ACT_AB_FULL_TASK_SUCCESS.md").write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("A", "B"))
    parser.add_argument("--eval-index", type=int, choices=range(35))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    evaluator, records = prerequisites()
    selected_methods = (args.method,) if args.method else ("A", "B")
    selected_indices = (args.eval_index,) if args.eval_index is not None else tuple(range(35))
    if not args.aggregate_only:
        for method in selected_methods:
            for index in selected_indices:
                run_one(method, index, records[(f"ACT-{method}40", index)])
    results = []
    for method in ("A", "B"):
        for index in range(35):
            row = records[(f"ACT-{method}40", index)]
            path = (
                RUNS
                / f"act_{method.lower()}40/eval_{index:02d}_{row['stable_episode_id']}/RUN_MANIFEST.json"
            )
            if path.is_file():
                results.append(read_json(path))
    if len(results) == 70:
        aggregate(results, evaluator)
    print(json.dumps({"completed_runs": len(results), "required_runs": 70}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
