"""Aggregate A/B diagnostics, trajectory plots, and final acceptance report."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import (
    CONFIG_OUTPUT,
    METHODS,
    OUTPUT,
    ROOT,
    atomic_csv,
    atomic_json,
    load_json,
    scalar_stats,
    sha256_file,
)


def _episode_path(output: Path, method: str, stable: str, suffix: str) -> Path:
    if suffix == "trajectory":
        return output / method / "trajectories" / f"{stable}.npz"
    return output / method / "metrics" / f"{stable}{suffix}"


def _mean_pair(metrics: Mapping[str, Any], section: str, left: str, right: str) -> float:
    values = metrics.get(section, {})
    return 0.5 * (
        float(values.get(left, {}).get("mean", 0.0))
        + float(values.get(right, {}).get("mean", 0.0))
    )


def _comparison_row(
    record: Mapping[str, Any],
    event: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode_index": int(record["episode_index"]),
        "episode": record["stable_episode_id"],
        "source_directory": record["source_name"],
        "source_event_validity": bool(event["source_semantic_valid"]),
        "source_anomalies": ";".join(event.get("anomalies", [])),
    }
    for method, prefix in (("baseline", "A"), ("proposed", "B")):
        metrics = values[method]
        task = metrics.get("task_space", {})
        collisions = metrics.get("collisions", {})
        scene = metrics.get("scene_diagnostics", {})
        row.update(
            {
                f"{prefix}_conversion_attempted": bool(
                    metrics.get("conversion_attempted", False)
                ),
                f"{prefix}_conversion_PASS": metrics.get("status") == "PASS",
                f"{prefix}_status": metrics.get("status", "MISSING"),
                f"{prefix}_kinematic_PASS": bool(metrics.get("kinematic_pass", False)),
                f"{prefix}_IK_success": float(metrics.get("ik_success_rate", 0.0)),
                f"{prefix}_mean_IK_task_error_m": float(
                    metrics.get("mean_ik_task_error_m", 0.0)
                ),
                f"{prefix}_max_IK_task_error_m": float(
                    metrics.get("max_ik_task_error_m", 0.0)
                ),
                f"{prefix}_wrist_error_m": _mean_pair(
                    metrics,
                    "task_space",
                    "left_wrist_target_error_m",
                    "right_wrist_target_error_m",
                ),
                f"{prefix}_orientation_error_rad": 0.5
                * (
                    float(
                        task.get("left_wrist_orientation_error_rad", {}).get(
                            "mean", 0.0
                        )
                    )
                    + float(
                        task.get("right_wrist_orientation_error_rad", {}).get(
                            "mean", 0.0
                        )
                    )
                ),
                f"{prefix}_physical_grasp_frame_error_m": _mean_pair(
                    metrics,
                    "task_space",
                    "left_physical_grasp_frame_target_error_m",
                    "right_physical_grasp_frame_target_error_m",
                ),
                f"{prefix}_bimanual_relation_error_m": float(
                    metrics.get("bimanual", {})
                    .get("inter_hand_relation_error_m", {})
                    .get("mean", 0.0)
                ),
                f"{prefix}_collision_frames": int(
                    collisions.get("invalid_self_body_collision_frames", 0)
                ),
                f"{prefix}_collisions": int(
                    collisions.get("invalid_self_body_collision_frames", 0)
                )
                > 0,
                f"{prefix}_release_inside_bin": bool(
                    scene.get("right_final_release_xy_inside_bin_opening", False)
                ),
                f"{prefix}_max_velocity_rad_s": float(
                    metrics.get("maximum_joint_velocity_rad_s", 0.0)
                ),
                f"{prefix}_max_acceleration_rad_s2": float(
                    metrics.get("maximum_joint_acceleration_rad_s2", 0.0)
                ),
                f"{prefix}_joint_limit_violations": int(
                    metrics.get("joint_limit_violation_count", 0)
                ),
                f"{prefix}_branch_discontinuities": int(
                    metrics.get("branch_discontinuity_count", 0)
                ),
            }
        )
    row["B_RIGHT_GRASP_before_LEFT_RELEASE"] = bool(
        values["proposed"].get("semantics", {}).get(
            "right_grasp_before_left_release", False
        )
    )
    row["B_dual_hold_sec"] = float(
        values["proposed"].get("semantics", {}).get("dual_hold_sec", 0.0)
    )
    return row


def _numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return scalar_stats(np.asarray([float(row[key]) for row in rows], dtype=np.float64))


def _trajectory_plot(
    output: Path,
    record: Mapping[str, Any],
    event: Mapping[str, Any],
) -> Path:
    stable = str(record["stable_episode_id"])
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        with np.load(
            _episode_path(output, method, stable, "trajectory"), allow_pickle=False
        ) as values:
            trajectories[method] = {
                name: np.asarray(values[name]) for name in values.files
            }
    baseline = trajectories["baseline"]
    proposed = trajectories["proposed"]
    time = np.asarray(baseline["timestamp"], dtype=np.float64)
    if not len(time):
        raise RuntimeError(f"cannot plot empty trajectory: {stable}")
    source_left = np.asarray(
        baseline["target_left_interaction_frame_position_world"], dtype=np.float64
    )
    source_right = np.asarray(
        baseline["target_right_interaction_frame_position_world"], dtype=np.float64
    )
    coordinate_colors = ("tab:red", "tab:green", "tab:blue")
    coordinate_names = ("X", "Y", "Z")
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    for side, values, axis in (
        ("left", source_left, axes[0, 0]),
        ("right", source_right, axes[0, 1]),
    ):
        for coordinate, (name, color) in enumerate(
            zip(coordinate_names, coordinate_colors)
        ):
            axis.plot(time, values[:, coordinate], color=color, lw=1.0, label=name)
        axis.set_title(f"Source ALOHA {side} physical task-EE (world)")
        axis.set_ylabel("position [m]")
        axis.legend(ncol=3, fontsize=8)

    for side, axis in (("left", axes[1, 0]), ("right", axes[1, 1])):
        for method, style in (("baseline", "--"), ("proposed", "-")):
            values = np.asarray(
                trajectories[method][
                    f"achieved_{side}_physical_grasp_frame_position_world"
                ],
                dtype=np.float64,
            )
            for coordinate, (name, color) in enumerate(
                zip(coordinate_names, coordinate_colors)
            ):
                axis.plot(
                    time,
                    values[:, coordinate],
                    color=color,
                    ls=style,
                    lw=0.9,
                    alpha=0.85,
                    label=f"{method[0].upper()} {name}",
                )
        axis.set_title(f"G1 {side} physical whole-hand grasp frame")
        axis.set_ylabel("position [m]")
        axis.legend(ncol=3, fontsize=7)

    source_distance = np.linalg.norm(source_right - source_left, axis=1)
    axes[2, 0].plot(time, source_distance, color="black", lw=1.2, label="source task-EE")
    for method, color in (("baseline", "tab:orange"), ("proposed", "tab:purple")):
        axes[2, 0].plot(
            time,
            np.asarray(trajectories[method]["inter_grasp_frame_distance_m"]),
            color=color,
            lw=1.0,
            label=f"{method} physical grasp frame",
        )
    axes[2, 0].set_title("Inter-hand / inter-grasp-frame distance")
    axes[2, 0].set_ylabel("distance [m]")
    axes[2, 0].legend(fontsize=8)

    selected = (0, 3, 5, 7, 10, 12)
    arm_names = np.asarray(proposed["g1_arm_joint_names"]).astype(str)
    for joint, color in zip(selected, plt.cm.tab10(np.linspace(0, 1, len(selected)))):
        axes[2, 1].plot(
            time,
            np.asarray(baseline["g1_arm_qpos"])[:, joint],
            ls="--",
            color=color,
            lw=0.75,
            alpha=0.75,
        )
        axes[2, 1].plot(
            time,
            np.asarray(proposed["g1_arm_qpos"])[:, joint],
            color=color,
            lw=0.95,
            label=arm_names[joint].replace("_joint", ""),
        )
    axes[2, 1].set_title("Selected G1 arm joints (A dashed, B solid)")
    axes[2, 1].set_ylabel("q [rad]")
    axes[2, 1].legend(ncol=2, fontsize=6)

    phase_code = {"OPEN": 0, "PREGRASP": 1, "GRASP": 2, "HOLD": 3, "RELEASE": 4}
    for side, color, offset in (("left", "tab:green", 0.0), ("right", "tab:blue", 0.08)):
        labels = np.asarray(proposed[f"{side}_hand_phase"]).astype(str)
        axes[3, 0].plot(
            time,
            [phase_code[value] + offset for value in labels],
            color=color,
            lw=1.1,
            label=side,
        )
    axes[3, 0].set_yticks(list(phase_code.values()), list(phase_code))
    axes[3, 0].set_title("Proposed semantic Dex3 phase")
    axes[3, 0].set_ylabel("phase")
    axes[3, 0].legend(fontsize=8)

    for method, color in (("baseline", "tab:orange"), ("proposed", "tab:purple")):
        success = np.asarray(trajectories[method]["ik_success_per_frame"], dtype=float)
        axes[3, 1].plot(time, success, color=color, lw=1.0, label=f"{method} IK")
    axes[3, 1].set_ylim(-0.05, 1.05)
    axes[3, 1].set_title("Per-frame numerical IK gate")
    axes[3, 1].set_ylabel("accepted")
    axes[3, 1].legend(fontsize=8)

    event_colors = {
        "LEFT_GRASP": "green",
        "RIGHT_GRASP": "blue",
        "LEFT_RELEASE": "orange",
        "RIGHT_FINAL_RELEASE": "red",
    }
    for name, color in event_colors.items():
        frame = event["frames"].get(name)
        if frame is None:
            continue
        event_time = time[int(frame)]
        for axis in axes.flat:
            axis.axvline(event_time, color=color, ls=":", lw=0.8, alpha=0.85)
    for axis in axes[-1]:
        axis.set_xlabel("source time [s]")
    fig.suptitle(
        f"{stable} / {record['source_name']} | event lines: LG green, RG blue, LR orange, RFR red"
    )
    fig.tight_layout()
    path = output / "comparison/trajectory_plots" / f"{stable}_ab_diagnostics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _active_reference_audit(output: Path) -> dict[str, Any]:
    files = [
        path
        for path in sorted((ROOT / "tools/doll_handoff_retargeting").glob("*.py"))
        # This reporting module necessarily names the forbidden vocabulary in
        # order to audit it; it has no trajectory, model, or rendering role.
        if path.name != "report.py"
    ] + [
        ROOT / "tools/retarget_doll_handoff_batch.py",
        ROOT / "tools/render_doll_handoff_comparisons.py",
        ROOT / "tools/finalize_doll_handoff_retargeting.py",
    ] + sorted((ROOT / "configs/doll_handoff_retargeting").glob("*.json"))
    forbidden = ("phone", "accessory", "charger", "magsafe", "ring hook", "ring_hook")
    matches: list[dict[str, Any]] = []
    for path in files:
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.lower()
            for term in forbidden:
                if term in lowered:
                    matches.append(
                        {
                            "term": term,
                            "file": str(path.relative_to(ROOT)),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
    core_files = [
        ROOT / "tools/doll_handoff_retargeting/retarget.py",
        ROOT / "tools/doll_handoff_retargeting/models.py",
        ROOT / "tools/doll_handoff_retargeting/events.py",
        ROOT / "tools/doll_handoff_retargeting/pipeline.py",
    ]
    hardcoding_patterns = {
        "episode_specific_condition": re.compile(
            r"\bif\s+episode(?:_index)?\s*==\s*\d+", re.IGNORECASE
        ),
        "noninitial_frame_specific_condition": re.compile(
            r"\bif\s+frame\s*==\s*(?!0\b)\d+", re.IGNORECASE
        ),
    }
    hardcoding: list[dict[str, Any]] = []
    for path in core_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for name, pattern in hardcoding_patterns.items():
                if pattern.search(line):
                    hardcoding.append(
                        {
                            "category": name,
                            "file": str(path.relative_to(ROOT)),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
    report = {
        "schema_version": "doll_handoff_active_reference_audit_v1",
        "scanned_files": [str(path.relative_to(ROOT)) for path in files if path.is_file()],
        "forbidden_active_reference_count": len(matches),
        "forbidden_active_references": matches,
        "episode_or_noninitial_frame_hardcoding_count": len(hardcoding),
        "episode_or_noninitial_frame_hardcoding": hardcoding,
        "active_magsafe_assumption_count": len(matches),
        "scene_table_asset_provenance_note": (
            "The frozen scene config retains its authoritative table asset source provenance; "
            "that provenance is not an active task assumption or spawned task object."
        ),
        "pass": not matches and not hardcoding,
    }
    atomic_json(CONFIG_OUTPUT / "active_reference_audit.json", report)
    return report


def build_reports(output: Path = OUTPUT, generate_plots: bool = True) -> dict[str, Any]:
    source = load_json(output / "source_audit/source_manifest.json")
    events = load_json(output / "event_audit/events.json")
    records = source["records"]
    rows: list[dict[str, Any]] = []
    metric_values: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        episode_index = int(record["episode_index"])
        stable = str(record["stable_episode_id"])
        values = {
            method: load_json(_episode_path(output, method, stable, ".json"))
            for method in METHODS
        }
        metric_values[episode_index] = values
        rows.append(_comparison_row(record, events[str(episode_index)], values))
    comparison = output / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    atomic_csv(comparison / "per_episode_comparison.csv", rows)

    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key.startswith(("A_", "B_"))
        and isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
    ]
    aggregate = {
        "schema_version": "doll_handoff_ab_aggregate_v1",
        "episode_count": len(rows),
        "source_semantic_valid_count": sum(row["source_event_validity"] for row in rows),
        "source_semantic_invalid_count": sum(not row["source_event_validity"] for row in rows),
        "baseline": {
            "attempted_count": sum(row["A_conversion_attempted"] for row in rows),
            "pass_count": sum(row["A_kinematic_PASS"] for row in rows),
            "fail_count": sum(not row["A_kinematic_PASS"] for row in rows),
            "release_inside_bin_count": sum(row["A_release_inside_bin"] for row in rows),
            "collision_fail_episode_count": sum(row["A_collisions"] for row in rows),
        },
        "proposed": {
            "attempted_count": sum(row["B_conversion_attempted"] for row in rows),
            "pass_count": sum(row["B_kinematic_PASS"] for row in rows),
            "fail_count": sum(not row["B_kinematic_PASS"] for row in rows),
            "release_inside_bin_count": sum(row["B_release_inside_bin"] for row in rows),
            "collision_fail_episode_count": sum(row["B_collisions"] for row in rows),
            "handoff_order_valid_count": sum(
                row["B_RIGHT_GRASP_before_LEFT_RELEASE"] for row in rows
            ),
        },
        "statistics": {key: _numeric_summary(rows, key) for key in numeric_keys},
        "paired_mean_differences_B_minus_A": {
            name: float(
                np.mean(
                    [
                        float(row[f"B_{name}"]) - float(row[f"A_{name}"])
                        for row in rows
                    ]
                )
            )
            for name in (
                "IK_success",
                "wrist_error_m",
                "physical_grasp_frame_error_m",
                "bimanual_relation_error_m",
                "collision_frames",
                "max_velocity_rad_s",
                "max_acceleration_rad_s2",
            )
        },
        "statistical_significance_claim": "NOT_PERFORMED_BY_DESIGN",
        "dataset_packaging": "NOT_STARTED_BY_DESIGN",
        "policy_training": "NOT_STARTED_BY_DESIGN",
    }
    atomic_json(comparison / "aggregate_summary.json", aggregate)

    source_failures = [
        {
            "episode_index": row["episode_index"],
            "episode": row["episode"],
            "source_directory": row["source_directory"],
            "anomalies": row["source_anomalies"].split(";")
            if row["source_anomalies"]
            else [],
        }
        for row in rows
        if not row["source_event_validity"]
    ]
    atomic_json(
        comparison / "source_semantic_failures.json",
        {
            "count": len(source_failures),
            "failures": source_failures,
            "silent_discard_count": 0,
        },
    )
    failures: list[dict[str, Any]] = []
    for record in records:
        episode_index = int(record["episode_index"])
        stable = str(record["stable_episode_id"])
        for method in METHODS:
            metrics = metric_values[episode_index][method]
            if metrics.get("status") == "PASS":
                continue
            validation = load_json(_episode_path(output, method, stable, ".validation.json"))
            failures.append(
                {
                    "episode_index": episode_index,
                    "stable_episode_id": stable,
                    "source_name": record["source_name"],
                    "method": method,
                    "status": metrics.get("status"),
                    "first_failure_gate": validation.get("first_failure_gate"),
                    "checks": validation.get("checks", {}),
                    "source_anomaly": not events[str(episode_index)][
                        "source_semantic_valid"
                    ],
                    "global_algorithm_result_retained": True,
                    "episode_specific_fix_applied": False,
                }
            )
    atomic_json(
        comparison / "retargeting_failures.json",
        {
            "failure_record_count": len(failures),
            "failed_episode_count": len({row["episode_index"] for row in failures}),
            "by_method": {
                method: sum(row["method"] == method for row in failures)
                for method in METHODS
            },
            "failures": failures,
            "silent_discard_count": 0,
        },
    )

    baseline_metrics = [metric_values[index]["baseline"] for index in range(len(records))]
    grasp_offsets = np.asarray(
        [
            value["scene_diagnostics"][
                "source_left_grasp_minus_modeled_doll_world_m"
            ]
            for value in baseline_metrics
        ],
        dtype=np.float64,
    )
    release_offsets = np.asarray(
        [
            np.asarray(
                value["scene_diagnostics"]["source_right_final_release_world_m"],
                dtype=np.float64,
            )
            - np.asarray(
                value["scene_diagnostics"]["bin_opening_center_world_m"],
                dtype=np.float64,
            )
            for value in baseline_metrics
        ]
    )
    registration = {
        "schema_version": "doll_handoff_source_scene_semantics_v1",
        "source_left_grasp_minus_modeled_doll_world_m": {
            "mean": np.mean(grasp_offsets, axis=0),
            "std": np.std(grasp_offsets, axis=0),
            "norm_m": scalar_stats(np.linalg.norm(grasp_offsets, axis=1)),
        },
        "source_final_release_minus_modeled_bin_opening_world_m": {
            "mean": np.mean(release_offsets, axis=0),
            "std": np.std(release_offsets, axis=0),
            "norm_m": scalar_stats(np.linalg.norm(release_offsets, axis=1)),
        },
        "diagnosis": (
            "The two source landmarks do not support one common compensating translation: "
            "grasp and release offsets differ in direction and height. Identity metric "
            "registration is retained; episode-specific anchors are forbidden."
        ),
        "global_registration_changed": False,
        "shared_by_A_and_B": True,
        "episode_specific_correction_count": 0,
        "scene_coordinates_used_as_waypoints": False,
    }
    atomic_json(comparison / "source_scene_semantics.json", registration)

    plot_paths: list[str] = []
    if generate_plots:
        common = load_json(CONFIG_OUTPUT / "common_config.json")
        selected = sorted(
            set(map(int, common["render"]["representative_episode_indices"]))
            | set(map(int, common["render"]["smoke_episode_indices"]))
        )
        for episode_index in selected:
            plot_paths.append(
                str(
                    _trajectory_plot(
                        output, records[episode_index], events[str(episode_index)]
                    ).resolve()
                )
            )
    aggregate["trajectory_plots"] = plot_paths
    atomic_json(comparison / "aggregate_summary.json", aggregate)
    active_references = _active_reference_audit(output)
    runbook = f"""# Doll-Handoff retargeting audit runbook

Run from `{ROOT}` with the confirmed IsaacLab conda environment:

```bash
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd {ROOT}

python tools/retarget_doll_handoff_batch.py --method both --episodes smoke
python tools/retarget_doll_handoff_batch.py --method baseline --episodes all
python tools/retarget_doll_handoff_batch.py --method proposed --episodes all
MUJOCO_GL=egl python tools/render_doll_handoff_comparisons.py --episodes all
python tools/finalize_doll_handoff_retargeting.py
```

All entry points are offline. Dataset packaging, policy training, physics tuning,
and real-robot commands are disabled by configuration and are not implemented by
these commands.
"""
    (output / "RUNBOOK.md").write_text(runbook, encoding="utf-8")
    return {
        "rows": rows,
        "aggregate": aggregate,
        "source_failures": source_failures,
        "retargeting_failures": failures,
        "registration": registration,
        "active_references": active_references,
    }


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def final_console_report(output: Path = OUTPUT) -> tuple[bool, list[str]]:
    reports = build_reports(output, generate_plots=True)
    rows = reports["rows"]
    aggregate = reports["aggregate"]
    source = load_json(output / "source_audit/source_manifest.json")
    event_aggregate = load_json(output / "event_audit/aggregate_event_statistics.json")
    fairness = load_json(CONFIG_OUTPUT / "fairness_report.json")
    freeze = load_json(CONFIG_OUTPUT / "freeze_manifest.json")
    visual_path = output / "comparison/visual_review_manifest.json"
    visual = load_json(visual_path) if visual_path.is_file() else {}
    blockers: list[str] = []
    if source.get("enumerated_count") != 50:
        blockers.append(f"source audit enumerated {source.get('enumerated_count')} instead of 50")
    if source.get("valid_count") != 50:
        blockers.append(f"valid source count is {source.get('valid_count')} instead of 50")
    for prefix, method in (("A", "baseline"), ("B", "proposed")):
        attempted = sum(bool(row[f"{prefix}_conversion_attempted"]) for row in rows)
        if attempted != 50:
            blockers.append(f"{method} attempted {attempted}/50")
    if not fairness.get("pass"):
        blockers.append("common-solver fairness report failed")
    if freeze.get("status") != "FROZEN" or not freeze.get(
        "smoke_execution_integrity_pass"
    ):
        blockers.append("smoke config freeze is absent or failed")
    for key, path in {
        "common": CONFIG_OUTPUT / "common_config.json",
        "baseline": CONFIG_OUTPUT / "baseline_config.json",
        "proposed": CONFIG_OUTPUT / "proposed_config.json",
    }.items():
        expected = freeze.get("config_sha256", {}).get(key)
        if not expected or not path.is_file() or sha256_file(path) != expected:
            blockers.append(f"frozen {key} config checksum mismatch")
    representatives = set([0, 10, 20, 30, 40, 49])
    present = set(visual.get("representative_videos_present", []))
    if not representatives.issubset(present):
        blockers.append(
            f"representative videos missing indices {sorted(representatives - present)}"
        )
    if not visual.get("every_rendered_failure_has_video"):
        blockers.append("one or more conversion/semantic failures lacks a review video")
    if visual.get("rendered_episode_count") != 50:
        blockers.append(
            f"visual review rendered {visual.get('rendered_episode_count', 0)}/50 episodes"
        )
    for path in (
        output / "comparison/per_episode_comparison.csv",
        output / "comparison/aggregate_summary.json",
    ):
        if not path.is_file():
            blockers.append(f"missing aggregate artifact {path}")
    if not reports["active_references"].get("pass"):
        blockers.append("active MagSafe/episode-hardcoding audit failed")
    manifest_paths = sorted(
        path
        for method in METHODS
        for path in (output / method / "metrics").glob(
            "doll_handoff_20260820_ep[0-9][0-9][0-9].manifest.json"
        )
    )
    if len(manifest_paths) != 100:
        blockers.append(f"expected 100 conversion manifests, found {len(manifest_paths)}")
    for path in manifest_paths:
        manifest = load_json(path)
        if manifest.get("policy_training") not in (False, "NOT_PERFORMED"):
            blockers.append(f"unexpected policy-training state in {path}")
            break
        if manifest.get("dataset_packaging") not in (False, "NOT_PERFORMED"):
            blockers.append(f"unexpected dataset-packaging state in {path}")
            break
    expected_scene_hash = load_json(CONFIG_OUTPUT / "common_config.json")[
        "scene_config_sha256"
    ]
    scene_path = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
    if sha256_file(scene_path) != expected_scene_hash:
        blockers.append("approved scene config changed during retargeting")

    a = aggregate["baseline"]
    b = aggregate["proposed"]
    stats = aggregate["statistics"]
    config_hashes = freeze.get("config_sha256", {})
    representative_video = (
        output
        / "comparison/videos/doll_handoff_20260820_ep020_comparison_overview.mp4"
    )
    episode0 = output / "comparison/videos/doll_handoff_20260820_ep000_comparison_overview.mp4"
    episode49 = output / "comparison/videos/doll_handoff_20260820_ep049_comparison_overview.mp4"
    failure_videos = [
        entry["outputs"]["comparison_overview"]
        for entry in visual.get("entries", [])
        if entry["episode_index"] in visual.get("failure_video_episode_indices", [])
    ]
    print("DOLL-HANDOFF 50-EPISODE RETARGETING")
    print()
    print("SOURCE")
    print(f"valid source episodes: {source.get('valid_count', 0)} / 50")
    print(
        "source semantic valid: "
        f"{event_aggregate.get('source_semantic_valid_count', 0)} / 50"
    )
    print(
        "source anomalies: "
        + (
            ", ".join(
                f"{key}={value}"
                for key, value in event_aggregate.get("anomaly_counts", {}).items()
                if value
            )
            or "NONE"
        )
    )
    print()
    print("TASK SCENE")
    print(f"path: {ROOT / 'isaaclab_doll_handoff_scene'}")
    print(f"scene config: {ROOT / 'isaaclab_doll_handoff_scene/scene_layout.json'}")
    scene = load_json(ROOT / "isaaclab_doll_handoff_scene/scene_layout.json")
    print(
        "G1 pelvis-table gap: "
        f"{float(scene['g1']['pelvis_to_table_front_target_m']):.3f} m"
    )
    print(
        "MagSafe references active: "
        f"{reports['active_references']['active_magsafe_assumption_count']}"
    )
    print()
    print("BASELINE A")
    print(f"converted: {a['attempted_count']} / 50")
    print(f"kinematic PASS: {a['pass_count']} / 50")
    print(f"mean IK success: {_fmt(stats['A_IK_success']['mean'])}")
    print(f"mean wrist error: {_fmt(stats['A_wrist_error_m']['mean'])} m")
    print(
        "mean physical grasp-frame diagnostic error: "
        f"{_fmt(stats['A_physical_grasp_frame_error_m']['mean'])} m"
    )
    print(f"collision FAIL: {a['collision_fail_episode_count']} / 50")
    print(f"release-inside-bin: {a['release_inside_bin_count']} / 50")
    print(
        "semantic warnings: "
        f"{event_aggregate.get('source_semantic_invalid_count', 0)}"
    )
    print()
    print("PROPOSED B")
    print(f"converted: {b['attempted_count']} / 50")
    print(f"kinematic PASS: {b['pass_count']} / 50")
    print(f"mean IK success: {_fmt(stats['B_IK_success']['mean'])}")
    print(f"mean wrist error: {_fmt(stats['B_wrist_error_m']['mean'])} m")
    print(
        "mean physical grasp-frame error: "
        f"{_fmt(stats['B_physical_grasp_frame_error_m']['mean'])} m"
    )
    print(
        "mean bimanual relation error: "
        f"{_fmt(stats['B_bimanual_relation_error_m']['mean'])} m"
    )
    print(f"handoff order valid: {b['handoff_order_valid_count']} / 50")
    print(f"collision FAIL: {b['collision_fail_episode_count']} / 50")
    print(f"release-inside-bin: {b['release_inside_bin_count']} / 50")
    print()
    print("A vs B")
    differences = aggregate["paired_mean_differences_B_minus_A"]
    print(
        "major differences: "
        f"B-A IK success={differences['IK_success']:+.6f}; "
        f"wrist error={differences['wrist_error_m']:+.6f} m; "
        "physical grasp-frame error="
        f"{differences['physical_grasp_frame_error_m']:+.6f} m; "
        f"bimanual relation error={differences['bimanual_relation_error_m']:+.6f} m"
    )
    print()
    print("VISUAL REVIEW")
    print(f"representative videos: {len(present)} / 6")
    print(f"  {representative_video}")
    print(f"failure videos: {len(failure_videos)}")
    for path in failure_videos[:3]:
        print(f"  {path}")
    if len(failure_videos) > 3:
        print(f"  ... {len(failure_videos) - 3} more in visual_review_manifest.json")
    print()
    print("CONFIG FREEZE")
    print(f"common config SHA256: {config_hashes.get('common', 'MISSING')}")
    print(f"baseline config SHA256: {config_hashes.get('baseline', 'MISSING')}")
    print(f"proposed config SHA256: {config_hashes.get('proposed', 'MISSING')}")
    print()
    print("POLICY A")
    print("NOT_STARTED_BY_DESIGN")
    print()
    print("POLICY B")
    print("NOT_STARTED_BY_DESIGN")
    print()
    print("OPEN COMMANDS")
    print(f"xdg-open '{representative_video}'")
    print(f"xdg-open '{episode0}'")
    print(f"xdg-open '{episode49}'")
    print(f"xdg-open '{output / 'comparison/per_episode_comparison.csv'}'")
    print(f"xdg-open '{output / 'comparison/aggregate_summary.json'}'")
    if blockers:
        print()
        print("BLOCKERS")
        for blocker in blockers:
            print(f"- {blocker}")
        print("BLOCKED_DOLL_HANDOFF_RETARGETING")
        return False, blockers
    print()
    print("READY_FOR_DOLL_HANDOFF_RETARGETING_REVIEW")
    return True, []


__all__ = ["build_reports", "final_console_report"]
