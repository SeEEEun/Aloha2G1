#!/usr/bin/env python3
"""Verify and aggregate the final matched episode-registered A/B EVAL35 runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_episode_registered_eval35"
FREEZE = OUT / "01_freeze/FINAL_EVAL35_FREEZE_MANIFEST.json"
REGISTRATION = OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
RESULTS = OUT / "04_results"
PAPER = OUT / "05_paper_artifacts"
STAGES = (
    "LEFT_GRASP_SUCCESS",
    "HANDOFF_SUCCESS",
    "RIGHT_OWNERSHIP_SUCCESS",
    "NO_DROP_TO_BIN",
    "BIN_ENTRY_SUCCESS",
    "BIN_SETTLE_SUCCESS",
    "CLEAN_COMMANDED_RELEASE",
    "PREMATURE_DROP_INTO_BIN",
    "PREMATURE_DROP_OUTSIDE_BIN",
    "FULL_TASK_SUCCESS",
)
LABELS = {
    "LEFT_GRASP_SUCCESS": "Left grasp",
    "HANDOFF_SUCCESS": "Handoff",
    "RIGHT_OWNERSHIP_SUCCESS": "Right ownership",
    "NO_DROP_TO_BIN": "No-drop-to-bin",
    "BIN_ENTRY_SUCCESS": "Bin entry",
    "BIN_SETTLE_SUCCESS": "Bin settle",
    "CLEAN_COMMANDED_RELEASE": "Clean release",
    "PREMATURE_DROP_INTO_BIN": "Premature drop into bin",
    "PREMATURE_DROP_OUTSIDE_BIN": "Premature drop outside bin",
    "FULL_TASK_SUCCESS": "Full task",
}
FAILURES = (
    "LEFT_GRASP",
    "HANDOFF",
    "RIGHT_OWNERSHIP",
    "TRANSPORT",
    "BIN_ENTRY",
    "SETTLE",
    "NONE_SUCCESS",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def proportion_ci(successes: int, trials: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [100.0 * lower, 100.0 * upper]


def percent(count: int) -> float:
    return 100.0 * count / 35.0


def load_method(letter: str, freeze_sha: str) -> list[dict[str, Any]]:
    root = OUT / ("02_act_a_results" if letter == "A" else "03_act_b_results") / "rollouts"
    manifests = sorted(root.glob("eval_*/RUN_MANIFEST.json"))
    if len(manifests) != 35:
        raise RuntimeError(f"ACT-{letter} has {len(manifests)}/35 run manifests")
    rows: list[dict[str, Any]] = []
    for path in manifests:
        value = read_json(path)
        if value.get("status") not in {"PHYSICAL_PASS", "PHYSICAL_FAIL"}:
            raise RuntimeError(f"invalid final status: {path}")
        run = path.parent
        invocation = read_json(run / "INVOCATION_MANIFEST.json")
        if invocation.get("freeze_manifest_sha256") != freeze_sha:
            raise RuntimeError(f"mixed-freeze run: {path}")
        for name, artifact in value.get("artifacts", {}).items():
            artifact_path = run / name
            if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
                raise RuntimeError(f"artifact drift: {artifact_path}")
        integrity = value["integrity"]
        if not (
            integrity.get("hard_physical_validity") is True
            and integrity.get("commanded_hard_limit_violation_scalar_count") == 0
            and integrity.get("measured_hard_limit_violation_scalar_count") == 0
            and integrity.get("non_finite_state_count") == 0
            and integrity.get("branch_discontinuity_count") == 0
            and integrity.get("object_pose_writes_after_initialization") == 0
            and value["fairness_audit"].get("arm_rescue_used") is False
            and value["fairness_audit"].get("wrist_rescue_used") is False
        ):
            raise RuntimeError(f"physical integrity failure: {path}")
        rows.append(value)
    rows.sort(key=lambda row: int(row["eval_index"]))
    if [int(row["eval_index"]) for row in rows] != list(range(35)):
        raise RuntimeError(f"ACT-{letter} ordering/membership mismatch")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: sum(bool(row["outcomes"][key]) for row in rows) for key in STAGES}
    failures = {key: sum(row["first_failure_stage"] == key for row in rows) for key in FAILURES}
    return {
        "valid_runs": len(rows),
        "stage_counts": counts,
        "stage_percent": {key: percent(value) for key, value in counts.items()},
        "stage_binomial_95_percent_ci_percent": {key: proportion_ci(value, 35) for key, value in counts.items()},
        "first_failure_counts": failures,
        "first_failure_percent": {key: percent(value) for key, value in failures.items()},
        "release_counts": {
            key: sum(row["release_classification"] == key for row in rows)
            for key in ("CLEAN_COMMANDED_RELEASE", "PREMATURE_DROP_INTO_BIN", "PREMATURE_DROP_OUTSIDE_BIN")
        },
        "contact_topologies": [row["contact_topology"] for row in rows],
    }


def main() -> int:
    freeze_sha = sha256_file(FREEZE)
    freeze = read_json(FREEZE)
    registration = read_json(REGISTRATION)
    a = load_method("A", freeze_sha)
    b = load_method("B", freeze_sha)
    if any(a[i]["stable_episode_id"] != b[i]["stable_episode_id"] for i in range(35)):
        raise RuntimeError("A/B episodes are not matched in identical order")
    if any(a[i]["episode_registration_entry_sha256"] != b[i]["episode_registration_entry_sha256"] for i in range(35)):
        raise RuntimeError("A/B registration entries differ")
    sa, sb = summarize(a), summarize(b)
    paired = {
        "A_FAIL_B_FAIL": sum(not a[i]["outcomes"]["FULL_TASK_SUCCESS"] and not b[i]["outcomes"]["FULL_TASK_SUCCESS"] for i in range(35)),
        "A_SUCCESS_B_FAIL": sum(a[i]["outcomes"]["FULL_TASK_SUCCESS"] and not b[i]["outcomes"]["FULL_TASK_SUCCESS"] for i in range(35)),
        "A_FAIL_B_SUCCESS": sum(not a[i]["outcomes"]["FULL_TASK_SUCCESS"] and b[i]["outcomes"]["FULL_TASK_SUCCESS"] for i in range(35)),
        "A_SUCCESS_B_SUCCESS": sum(a[i]["outcomes"]["FULL_TASK_SUCCESS"] and b[i]["outcomes"]["FULL_TASK_SUCCESS"] for i in range(35)),
    }
    discordant = paired["A_SUCCESS_B_FAIL"] + paired["A_FAIL_B_SUCCESS"]
    mcnemar_p = 1.0 if discordant == 0 else float(
        binomtest(paired["A_FAIL_B_SUCCESS"], discordant, 0.5, alternative="two-sided").pvalue
    )
    paired_effects = np.asarray(
        [float(b[i]["outcomes"]["FULL_TASK_SUCCESS"]) - float(a[i]["outcomes"]["FULL_TASK_SUCCESS"]) for i in range(35)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260904)
    bootstrap = paired_effects[rng.integers(0, 35, size=(200000, 35))].mean(axis=1) * 100.0
    effect_ci = [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])]
    a_tsr = sa["stage_counts"]["FULL_TASK_SUCCESS"]
    b_tsr = sb["stage_counts"]["FULL_TASK_SUCCESS"]
    paired_report = {
        "schema_version": "final_episode_registered_eval35_paired_outcomes_v1",
        "status": "PASS",
        "counts": paired,
        "ACT_A_TSR_count": a_tsr,
        "ACT_B_TSR_count": b_tsr,
        "ACT_A_TSR_percent": percent(a_tsr),
        "ACT_B_TSR_percent": percent(b_tsr),
        "B_minus_A_percentage_points": percent(b_tsr - a_tsr),
        "ACT_A_Clopper_Pearson_95_percent_CI_percent": proportion_ci(a_tsr, 35),
        "ACT_B_Clopper_Pearson_95_percent_CI_percent": proportion_ci(b_tsr, 35),
        "paired_bootstrap_95_percent_CI_percentage_points": effect_ci,
        "paired_bootstrap_replicates": 200000,
        "paired_bootstrap_seed": 20260904,
        "discordant_pairs": discordant,
        "exact_McNemar_two_sided_p": mcnemar_p,
        "interpretation": "Descriptive matched physical result; statistical significance is claimed only if supported by the exact test.",
    }

    output_rows: list[dict[str, Any]] = []
    for index, (ra, rb) in enumerate(zip(a, b, strict=True), 1):
        row: dict[str, Any] = {
            "episode": index,
            "eval_index": index - 1,
            "stable_episode_id": ra["stable_episode_id"],
            "source_recording": ra["provenance"],
            "registration_entry_sha256": ra["episode_registration_entry_sha256"],
        }
        for letter, value in (("A", ra), ("B", rb)):
            for stage in STAGES:
                row[f"{letter}_{stage}"] = int(bool(value["outcomes"][stage]))
            row[f"{letter}_FIRST_FAILURE_STAGE"] = value["first_failure_stage"]
            row[f"{letter}_CONTACT_TOPOLOGY"] = json.dumps(value["contact_topology"], sort_keys=True)
        output_rows.append(row)
    fields = list(output_rows[0])
    write_csv(RESULTS / "PER_EPISODE_EVAL35_RESULTS.csv", fields, output_rows)
    md = ["# Per-episode final physical EVAL35 results", "", "| # | Episode | A G | A H | A R | A Bin | A Full | B G | B H | B R | B Bin | B Full |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in output_rows:
        md.append(
            f"| {row['episode']:02d} | `{row['stable_episode_id']}` | {row['A_LEFT_GRASP_SUCCESS']} | {row['A_HANDOFF_SUCCESS']} | {row['A_RIGHT_OWNERSHIP_SUCCESS']} | {row['A_BIN_SETTLE_SUCCESS']} | {row['A_FULL_TASK_SUCCESS']} | {row['B_LEFT_GRASP_SUCCESS']} | {row['B_HANDOFF_SUCCESS']} | {row['B_RIGHT_OWNERSHIP_SUCCESS']} | {row['B_BIN_SETTLE_SUCCESS']} | {row['B_FULL_TASK_SUCCESS']} |"
        )
    atomic_text(RESULTS / "PER_EPISODE_EVAL35_RESULTS.md", "\n".join(md) + "\n")

    stage_rows = []
    for key in STAGES:
        ac, bc = sa["stage_counts"][key], sb["stage_counts"][key]
        stage_rows.append({"metric": key, "ACT_A_count": ac, "ACT_A_percent": percent(ac), "ACT_B_count": bc, "ACT_B_percent": percent(bc), "B_minus_A_percentage_points": percent(bc - ac)})
    write_csv(RESULTS / "ACT_AB_STAGE_SUCCESS.csv", list(stage_rows[0]), stage_rows)
    failure_rows = []
    for key in FAILURES:
        failure_rows.append({"first_failure_stage": "SUCCESS" if key == "NONE_SUCCESS" else key, "ACT_A_count": sa["first_failure_counts"][key], "ACT_A_percent": sa["first_failure_percent"][key], "ACT_B_count": sb["first_failure_counts"][key], "ACT_B_percent": sb["first_failure_percent"][key]})
    write_csv(RESULTS / "ACT_AB_FIRST_FAILURE_DISTRIBUTION.csv", list(failure_rows[0]), failure_rows)
    atomic_json(RESULTS / "ACT_AB_PAIRED_OUTCOMES.json", paired_report)
    atomic_text(
        RESULTS / "ACT_AB_PAIRED_OUTCOMES.md",
        "# Matched ACT-A/B full-task outcomes\n\n"
        + "\n".join(f"- {key.replace('_', ' ')}: {value}" for key, value in paired.items())
        + f"\n- B−A: {paired_report['B_minus_A_percentage_points']:.1f} pp"
        + f"\n- Paired bootstrap 95% CI: [{effect_ci[0]:.1f}, {effect_ci[1]:.1f}] pp"
        + f"\n- Exact McNemar p: {mcnemar_p:.6g}\n",
    )

    aggregate = {
        "schema_version": "final_episode_registered_act_ab_eval35_results_v1",
        "status": "FINAL_COMPARABLE_35_PLUS_35",
        "freeze_manifest": str(FREEZE.resolve()),
        "freeze_manifest_sha256": freeze_sha,
        "scientific_bundle_sha256": freeze["scientific_bundle_sha256"],
        "registration_entries": registration["EVAL35_count"],
        "matched_A_B_registration_entries": registration["A_B_identical_object_pose_count"],
        "ACT_A": sa,
        "ACT_B": sb,
        "paired": paired_report,
        "integrity": {"valid_rows": 70, "matched_pairs": 35, "arm_rescue": False, "wrist_rescue": False, "object_pose_writes_after_initialization": 0},
        "runs": {"ACT_A": a, "ACT_B": b},
    }
    atomic_json(OUT / "02_act_a_results/ACT_A_EVAL35_RESULTS.json", {"method": "ACT-A40", **sa, "runs": a})
    atomic_json(OUT / "03_act_b_results/ACT_B_EVAL35_RESULTS.json", {"method": "ACT-B40", **sb, "runs": b})
    atomic_json(RESULTS / "FINAL_NUMERIC_RESULTS.json", aggregate)

    table = ["# Final ACT-A vs ACT-B physical EVAL35", "", "| Metric | ACT-A | ACT-B | B-A |", "|---|---:|---:|---:|"]
    for key in ("LEFT_GRASP_SUCCESS", "HANDOFF_SUCCESS", "RIGHT_OWNERSHIP_SUCCESS", "NO_DROP_TO_BIN", "BIN_ENTRY_SUCCESS", "BIN_SETTLE_SUCCESS", "CLEAN_COMMANDED_RELEASE", "FULL_TASK_SUCCESS"):
        ac, bc = sa["stage_counts"][key], sb["stage_counts"][key]
        table.append(f"| {LABELS[key]} | {ac}/35 ({percent(ac):.1f}%) | {bc}/35 ({percent(bc):.1f}%) | {percent(bc-ac):+.1f} pp |")
    atomic_text(PAPER / "TABLE_FINAL_ACT_AB_PHYSICAL_EVAL35.md", "\n".join(table) + "\n")

    numeric = [
        "# Final episode-registered ACT-A/B EVAL35 numeric results", "",
        f"Freeze SHA256: `{freeze_sha}`", f"Scientific bundle SHA256: `{freeze['scientific_bundle_sha256']}`", "",
        "All 35 matched episodes used the same episode-conditioned, source-derived object pose for A and B. The contact model, common topology-neutral Dex3 controller, physics, hard-limit handling, and scorer were identical. No arm or wrist rescue was used.", "",
        *table[2:], "",
        f"ACT-A TSR 95% CI: [{proportion_ci(a_tsr,35)[0]:.1f}, {proportion_ci(a_tsr,35)[1]:.1f}]%",
        f"ACT-B TSR 95% CI: [{proportion_ci(b_tsr,35)[0]:.1f}, {proportion_ci(b_tsr,35)[1]:.1f}]%",
        f"B−A paired effect: {percent(b_tsr-a_tsr):+.1f} pp (paired bootstrap 95% CI [{effect_ci[0]:.1f}, {effect_ci[1]:.1f}] pp).",
        f"Exact McNemar p={mcnemar_p:.6g}; no significance claim is made unless supported by this test.", "",
        "## First failure", "",
        *[f"- {('SUCCESS' if key == 'NONE_SUCCESS' else key)}: A {sa['first_failure_counts'][key]}/35; B {sb['first_failure_counts'][key]}/35" for key in FAILURES], "",
        "## Release diagnostics", "",
        *[f"- {key}: A {sa['release_counts'][key]}/35; B {sb['release_counts'][key]}/35" for key in sa["release_counts"]], "",
        "## Limitations", "",
        "This is a contact-constrained simulation result, not a real-G1 physical-success claim. The rigid body is a qualified plush approximation, and the sample contains 35 matched episodes.",
    ]
    atomic_text(RESULTS / "FINAL_NUMERIC_RESULTS.md", "\n".join(numeric) + "\n")
    atomic_text(OUT / "FINAL_ACT_AB_EVAL35_NUMERIC_RESULTS.md", "\n".join(numeric) + "\n")

    review_rows = []
    for row in output_rows:
        review_rows.append({
            "episode": row["episode"],
            "automatic_A_grasp": row["A_LEFT_GRASP_SUCCESS"],
            "automatic_A_handoff": row["A_HANDOFF_SUCCESS"],
            "automatic_A_ownership": row["A_RIGHT_OWNERSHIP_SUCCESS"],
            "automatic_A_bin": row["A_BIN_SETTLE_SUCCESS"],
            "automatic_A_full": row["A_FULL_TASK_SUCCESS"],
            "automatic_B_grasp": row["B_LEFT_GRASP_SUCCESS"],
            "automatic_B_handoff": row["B_HANDOFF_SUCCESS"],
            "automatic_B_ownership": row["B_RIGHT_OWNERSHIP_SUCCESS"],
            "automatic_B_bin": row["B_BIN_SETTLE_SUCCESS"],
            "automatic_B_full": row["B_FULL_TASK_SUCCESS"],
            "human_A_full": "", "human_B_full": "", "notes": "",
        })
    write_csv(OUT / "EVAL35_FINAL_PHYSICAL_HUMAN_REVIEW.csv", list(review_rows[0]), review_rows)
    print(json.dumps({"status": aggregate["status"], "ACT_A_full": a_tsr, "ACT_B_full": b_tsr, "B_minus_A_pp": percent(b_tsr-a_tsr), "freeze_sha256": freeze_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
