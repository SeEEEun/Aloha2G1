#!/usr/bin/env python3
"""Verify and aggregate the frozen standardized-grasp DEV35 A/B experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/standardized_grasp_ab_dev35"
FREEZE = OUT / "02_freeze/STANDARDIZED_GRASP_FINAL_FREEZE.json"
COMMANDS = OUT / "01_prepared_commands/STANDARDIZED_GRASP_AB_COMMAND_MANIFEST.json"
RESULTS = OUT / "05_results"
PAPER = OUT / "06_paper_artifacts"
STAGES = (
    "LIFT_SUCCESS",
    "LEFT_RETENTION_SUCCESS",
    "HANDOFF_SUCCESS",
    "RIGHT_OWNERSHIP_SUCCESS",
    "RIGHT_TRANSPORT_RETENTION",
    "BIN_ENTRY_SUCCESS",
    "BIN_SETTLE_SUCCESS",
    "POST_GRASP_FULL_TASK_SUCCESS",
)
STAGE_LABELS = {
    "LIFT_SUCCESS": "Lift",
    "LEFT_RETENTION_SUCCESS": "Left retention",
    "HANDOFF_SUCCESS": "Handoff",
    "RIGHT_OWNERSHIP_SUCCESS": "Right ownership",
    "RIGHT_TRANSPORT_RETENTION": "Right transport retention",
    "BIN_ENTRY_SUCCESS": "Bin entry",
    "BIN_SETTLE_SUCCESS": "Bin settle",
    "POST_GRASP_FULL_TASK_SUCCESS": "Post-grasp full task",
}
FAILURES = ("LIFT", "LEFT_RETENTION", "HANDOFF", "RIGHT_OWNERSHIP", "RIGHT_TRANSPORT", "BIN_ENTRY", "SETTLE", "SUCCESS")
RELEASES = ("CLEAN_COMMANDED_RELEASE", "PREMATURE_DROP_INTO_BIN", "PREMATURE_DROP_OUTSIDE_BIN")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def percent(count: int) -> float:
    return 100.0 * count / 35.0


def clopper_pearson(successes: int, trials: int = 35, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return [100.0 * lower, 100.0 * upper]


def load_method(method: str, expected_freeze_sha: str) -> list[dict[str, Any]]:
    directory = OUT / ("03_a_results" if method == "A" else "04_b_results") / "rollouts"
    manifests = sorted(directory.glob("eval_*/RUN_MANIFEST.json"))
    if len(manifests) != 35:
        raise RuntimeError(f"{method}: {len(manifests)}/35 complete run manifests")
    records: list[dict[str, Any]] = []
    for path in manifests:
        run = path.parent
        record = read_json(path)
        invocation = read_json(run / "INVOCATION_MANIFEST.json")
        result = read_json(run / "STANDARDIZED_GRASP_POST_GRASP_RESULT.json")
        if record.get("status") not in {"PHYSICAL_PASS", "PHYSICAL_FAIL"}:
            raise RuntimeError(f"non-final run status: {path}")
        if result.get("status") not in {"PASS", "FAIL"} or result.get("integrity", {}).get("hard_physical_validity") is not True:
            raise RuntimeError(f"invalid physical run: {path}")
        if invocation.get("freeze_sha256") != expected_freeze_sha:
            raise RuntimeError(f"mixed freeze: {path}")
        if invocation.get("method") != method or record.get("method") != method:
            raise RuntimeError(f"method mismatch: {path}")
        for name, artifact in record.get("artifacts", {}).items():
            artifact_path = run / name
            if not artifact_path.is_file() or sha256(artifact_path) != artifact["sha256"]:
                raise RuntimeError(f"trace/artifact drift: {artifact_path}")
        integrity = record["integrity"]
        if not (
            integrity["commanded_hard_limit_violations"] == 0
            and integrity["measured_arm_hard_limit_violations"] == 0
            and integrity["measured_dex3_hard_limit_violations"] == 0
            and integrity["branch_discontinuities"] == 0
            and integrity["object_pose_writes_after_initialization"] == 0
            and integrity["arm_rescue"] is False
            and integrity["wrist_rescue"] is False
            and integrity["finite"] is True
        ):
            raise RuntimeError(f"integrity failure: {path}")
        if record["outcomes"].get("STANDARDIZED_INITIAL_GRASP") is not True:
            raise RuntimeError(f"standardized control missing: {path}")
        records.append(record)
    records.sort(key=lambda row: int(row["eval_index"]))
    if [int(row["eval_index"]) for row in records] != list(range(35)):
        raise RuntimeError(f"{method}: membership/order is not exactly DEV35")
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {stage: sum(bool(row["outcomes"][stage]) for row in records) for stage in STAGES}
    failures = {stage: sum(row["first_failure_stage"] == stage for row in records) for stage in FAILURES}
    releases = {key: sum(bool(row["outcomes"][key]) for row in records) for key in RELEASES}
    return {
        "valid_runs": len(records),
        "standardized_initial_grasp_count": sum(bool(row["outcomes"]["STANDARDIZED_INITIAL_GRASP"]) for row in records),
        "stage_counts": counts,
        "stage_percent": {key: percent(value) for key, value in counts.items()},
        "stage_clopper_pearson_95_percent_ci_percent": {key: clopper_pearson(value) for key, value in counts.items()},
        "first_failure_counts": failures,
        "first_failure_percent": {key: percent(value) for key, value in failures.items()},
        "release_counts": releases,
    }


def main() -> int:
    freeze_sha = sha256(FREEZE)
    freeze = read_json(FREEZE)
    commands = read_json(COMMANDS)
    if freeze.get("status") != "FROZEN_BEFORE_EVAL35" or freeze.get("evaluation_label") != "DEV35 STANDARDIZED-GRASP PHYSICAL EVALUATION":
        raise RuntimeError("wrong or missing standardized-grasp freeze")
    a, b = load_method("A", freeze_sha), load_method("B", freeze_sha)
    if any(a[i]["stable_episode_id"] != b[i]["stable_episode_id"] for i in range(35)):
        raise RuntimeError("A/B episode mismatch")
    if len(commands.get("records", [])) != 70:
        raise RuntimeError("command manifest is not 70/70")
    sa, sb = summarize(a), summarize(b)

    paired = {
        "A_FAIL_B_FAIL": sum(not a[i]["outcomes"][STAGES[-1]] and not b[i]["outcomes"][STAGES[-1]] for i in range(35)),
        "A_SUCCESS_B_FAIL": sum(a[i]["outcomes"][STAGES[-1]] and not b[i]["outcomes"][STAGES[-1]] for i in range(35)),
        "A_FAIL_B_SUCCESS": sum(not a[i]["outcomes"][STAGES[-1]] and b[i]["outcomes"][STAGES[-1]] for i in range(35)),
        "A_SUCCESS_B_SUCCESS": sum(a[i]["outcomes"][STAGES[-1]] and b[i]["outcomes"][STAGES[-1]] for i in range(35)),
    }
    discordant = paired["A_SUCCESS_B_FAIL"] + paired["A_FAIL_B_SUCCESS"]
    mcnemar = 1.0 if discordant == 0 else float(binomtest(paired["A_FAIL_B_SUCCESS"], discordant, 0.5).pvalue)
    effects = np.asarray([
        int(b[i]["outcomes"][STAGES[-1]]) - int(a[i]["outcomes"][STAGES[-1]]) for i in range(35)
    ], dtype=np.float64)
    rng = np.random.default_rng(20260904)
    boot = effects[rng.integers(0, 35, size=(200_000, 35))].mean(axis=1) * 100.0
    effect_ci = [float(value) for value in np.percentile(boot, [2.5, 97.5])]
    at, bt = sa["stage_counts"][STAGES[-1]], sb["stage_counts"][STAGES[-1]]
    paired_result = {
        "counts": paired,
        "A_post_grasp_success_count": at,
        "B_post_grasp_success_count": bt,
        "A_post_grasp_success_percent": percent(at),
        "B_post_grasp_success_percent": percent(bt),
        "B_minus_A_percentage_points": percent(bt - at),
        "A_clopper_pearson_95_percent_ci_percent": clopper_pearson(at),
        "B_clopper_pearson_95_percent_ci_percent": clopper_pearson(bt),
        "paired_bootstrap_95_percent_ci_percentage_points": effect_ci,
        "paired_bootstrap_seed": 20260904,
        "paired_bootstrap_replicates": 200_000,
        "discordant_pairs": discordant,
        "exact_McNemar_two_sided_p": mcnemar,
        "interpretation": "Matched DEV35 descriptive result; no significance claim is made unless supported by the exact test.",
    }

    episode_rows: list[dict[str, Any]] = []
    for display, (ra, rb) in enumerate(zip(a, b, strict=True), 1):
        row: dict[str, Any] = {"episode": display, "eval_index": display - 1, "stable_episode_id": ra["stable_episode_id"]}
        for method, record in (("A", ra), ("B", rb)):
            for stage in STAGES:
                row[f"{method}_{stage}"] = int(record["outcomes"][stage])
            for key in RELEASES:
                row[f"{method}_{key}"] = int(record["outcomes"][key])
            row[f"{method}_FIRST_FAILURE_STAGE"] = record["first_failure_stage"]
            row[f"{method}_COMMON_IK_FAILURE_STAGE"] = record["common_ik_failure_stage"] or "NONE"
            row[f"{method}_CONTACT_TOPOLOGY"] = json.dumps(record["contact_topology"], sort_keys=True)
            row[f"{method}_MAX_DOLL_COM_LIFT_MM"] = record["diagnostics"]["maximum_doll_com_lift_mm"]
        episode_rows.append(row)
    write_csv(RESULTS / "PER_EPISODE_STANDARDIZED_GRASP_DEV35.csv", episode_rows)

    per_episode_md = [
        "# Per-episode standardized-grasp DEV35 physical outcomes", "",
        "The initial LEFT grasp is a controlled physical initialization and is not a competitive success metric.", "",
        "| # | Episode | A first failure | A full | B first failure | B full |", "|---:|---|---|---:|---|---:|",
    ]
    for row in episode_rows:
        per_episode_md.append(f"| {row['episode']:02d} | `{row['stable_episode_id']}` | {row['A_FIRST_FAILURE_STAGE']} | {row['A_POST_GRASP_FULL_TASK_SUCCESS']} | {row['B_FIRST_FAILURE_STAGE']} | {row['B_POST_GRASP_FULL_TASK_SUCCESS']} |")
    atomic_text(RESULTS / "PER_EPISODE_STANDARDIZED_GRASP_DEV35.md", "\n".join(per_episode_md) + "\n")

    stage_rows = []
    for stage in STAGES:
        ac, bc = sa["stage_counts"][stage], sb["stage_counts"][stage]
        stage_rows.append({
            "metric": stage, "A_count": ac, "A_percent": percent(ac), "B_count": bc,
            "B_percent": percent(bc), "B_minus_A_percentage_points": percent(bc - ac),
        })
    write_csv(RESULTS / "STANDARDIZED_GRASP_AB_STAGE_SUCCESS.csv", stage_rows)
    failure_rows = [{
        "first_failure_stage": stage, "A_count": sa["first_failure_counts"][stage],
        "A_percent": sa["first_failure_percent"][stage], "B_count": sb["first_failure_counts"][stage],
        "B_percent": sb["first_failure_percent"][stage],
    } for stage in FAILURES]
    write_csv(RESULTS / "STANDARDIZED_GRASP_AB_FIRST_FAILURE.csv", failure_rows)
    atomic_json(RESULTS / "STANDARDIZED_GRASP_AB_PAIRED_OUTCOMES.json", paired_result)

    aggregate = {
        "schema_version": "standardized_grasp_ab_dev35_final_results_v1",
        "status": "FINAL_COMPARABLE_35_PLUS_35",
        "evaluation_label": "DEV35 STANDARDIZED-GRASP PHYSICAL EVALUATION",
        "not_end_to_end_grasp_acquisition": True,
        "freeze_path": str(FREEZE.resolve()), "freeze_sha256": freeze_sha,
        "control_condition": {
            "common_initial_physical_grasp": "PASS", "A_standardized_initial_grasp": "35/35",
            "B_standardized_initial_grasp": "35/35", "same_initial_state": True,
            "same_Dex3_controller": True, "same_IK": True, "same_physics": True,
        },
        "only_intended_difference": {"A": "WRIST POST-GRASP TARGET", "B": "INTERACTION POST-GRASP TARGET"},
        "A": sa, "B": sb, "paired": paired_result,
        "integrity": {"valid_rows": 70, "matched_pairs": 35, "arm_rescue": False, "wrist_rescue": False, "object_pose_writes_after_initialization": 0},
        "runs": {"A": a, "B": b},
    }
    atomic_json(RESULTS / "FINAL_STANDARDIZED_GRASP_NUMERIC_RESULTS.json", aggregate)

    table = [
        "# Standardized-grasp A/B physical results", "",
        "| Metric | A — Wrist | B — Interaction | B-A |", "|---|---:|---:|---:|",
        "| Standardized initial grasp | 35/35 | 35/35 | controlled |",
    ]
    for stage in STAGES:
        ac, bc = sa["stage_counts"][stage], sb["stage_counts"][stage]
        table.append(f"| {STAGE_LABELS[stage]} | {ac}/35 ({percent(ac):.1f}%) | {bc}/35 ({percent(bc):.1f}%) | {percent(bc-ac):+.1f} pp |")
    atomic_text(PAPER / "TABLE_STANDARDIZED_GRASP_AB_PHYSICAL_RESULTS.md", "\n".join(table) + "\n")

    numeric = [
        "# Final standardized-grasp A/B physical numeric results", "",
        "**Scope:** DEV35 STANDARDIZED-GRASP PHYSICAL EVALUATION. This is not end-to-end grasp acquisition and DEV35 is not described as untouched or unseen.", "",
        f"Freeze SHA256: `{freeze_sha}`", "",
        "Both methods began from the same physically validated LEFT-grasp state. The post-grasp trajectory was rebased to that shared state, while each method's relative SE(3) evolution was preserved. IK, Dex3 control, handoff logic, doll, bin, physics, and scorer were common.", "",
        *table[2:], "",
        f"A post-grasp success 95% exact CI: [{clopper_pearson(at)[0]:.1f}, {clopper_pearson(at)[1]:.1f}]%.",
        f"B post-grasp success 95% exact CI: [{clopper_pearson(bt)[0]:.1f}, {clopper_pearson(bt)[1]:.1f}]%.",
        f"B−A: {percent(bt-at):+.1f} pp; paired bootstrap 95% CI [{effect_ci[0]:.1f}, {effect_ci[1]:.1f}] pp.",
        f"Exact McNemar p={mcnemar:.6g} ({discordant} discordant pairs).", "",
        "## First failure distribution", "",
        *[f"- {stage}: A {sa['first_failure_counts'][stage]}/35; B {sb['first_failure_counts'][stage]}/35" for stage in FAILURES], "",
        "## Release diagnostics", "",
        *[f"- {key}: A {sa['release_counts'][key]}/35; B {sb['release_counts'][key]}/35" for key in RELEASES], "",
        "## Limitation", "",
        "The controlled initialization removes initial grasp acquisition from the comparison. Results therefore quantify post-grasp manipulation progression in contact-constrained simulation, not autonomous end-to-end task completion or real-G1 performance.",
    ]
    atomic_text(RESULTS / "FINAL_STANDARDIZED_GRASP_NUMERIC_RESULTS.md", "\n".join(numeric) + "\n")
    print(json.dumps({"status": aggregate["status"], "freeze_sha256": freeze_sha, "A_full": at, "B_full": bt, "B_minus_A_pp": percent(bt-at)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
