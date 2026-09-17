#!/usr/bin/env python3
"""Evaluate the fixed HELDOUT8 ACT-A/B source-conditioned Isaac rollouts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pyarrow.parquet as pq

from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.paper_core_source_rollout_common import (
    ROOT,
    atomic_json,
    atomic_npz,
    frozen_interfaces,
    read_json,
    rollout_dynamics,
    sha256_file,
)


ROLLOUT_ROOT = ROOT / "outputs/paper_core_ab/source_conditioned_rollout"
HELDOUT_MANIFEST = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EVALUATION_CONTRACT = ROOT / "outputs/paper_core_ab/source_rollout_evaluation_contract.json"
TABLES = ROOT / "outputs/paper_core_ab/tables"
FIGURES = ROOT / "outputs/paper_core_ab/figures"
METHOD_DATASETS = {
    "a": ROOT / "datasets/doll_handoff_fair_a_heldout8",
    "b": ROOT / "datasets/doll_handoff_proposed_b_heldout8",
}
FPS = 30.0


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("RMS of empty array")
    return float(np.sqrt(np.mean(np.square(array))))


def stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1) * scale
    if not array.size or not np.isfinite(array).all():
        raise RuntimeError("statistics require finite non-empty values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def parquet_actions(root: Path) -> np.ndarray:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(root)
    table = pq.read_table(files, columns=["action"])
    result = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if result.shape[1:] != (28,) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid held-out action arrays: {root}")
    return result


def episode_offsets(entries: list[dict[str, Any]]) -> list[int]:
    values = [0]
    for entry in entries[:-1]:
        values.append(values[-1] + int(entry["frames"]))
    return values


def branch_count(q: np.ndarray, thresholds: Mapping[str, Any]) -> int:
    values = np.asarray(q, dtype=np.float64)
    arm_norms = np.linalg.norm(np.diff(values[:, :14], axis=0), axis=1)
    absolute = float(thresholds["branch_absolute_step_norm_rad"])
    multiplier = float(thresholds["branch_local_multiplier"])
    count = 0
    for index, value in enumerate(arm_norms):
        local = float(
            np.median(arm_norms[max(0, index - 9) : min(len(arm_norms), index + 10)])
        )
        count += int(value > max(absolute, multiplier * max(local, 1e-6)))
    return count


def ordering_metrics(
    q: np.ndarray,
    method_target: np.ndarray,
    event_map: Mapping[str, int],
    left_phase: np.ndarray,
) -> dict[str, Any]:
    count = len(q)
    right_open_frame = int(event_map["RIGHT_CLOSE_ONSET"]) - 1
    right_hold_frame = int(event_map["RIGHT_STABLE_HOLD"])
    left_hold_frame = int(event_map["LEFT_STABLE_HOLD"])
    post_release = np.flatnonzero(
        (np.arange(len(left_phase)) > int(event_map["LEFT_RELEASE"]))
        & (left_phase.astype(str) == "OPEN")
    )
    if not len(post_release):
        raise RuntimeError("source episode lacks a post-release left-open frame")
    prototypes = {
        "right_open": method_target[right_open_frame, 21:28],
        "right_hold": method_target[right_hold_frame, 21:28],
        "left_hold": method_target[left_hold_frame, 14:21],
        "left_open": method_target[int(post_release[0]), 14:21],
    }
    right_hold_distance = np.linalg.norm(q[:, 21:28] - prototypes["right_hold"], axis=1)
    right_open_distance = np.linalg.norm(q[:, 21:28] - prototypes["right_open"], axis=1)
    left_open_distance = np.linalg.norm(q[:, 14:21] - prototypes["left_open"], axis=1)
    left_hold_distance = np.linalg.norm(q[:, 14:21] - prototypes["left_hold"], axis=1)
    right_candidates = np.flatnonzero(
        (np.arange(count) >= int(event_map["RIGHT_CLOSE_ONSET"]))
        & (right_hold_distance < right_open_distance)
    )
    left_candidates = np.flatnonzero(
        (np.arange(count) > int(event_map["LEFT_STABLE_HOLD"]))
        & (left_open_distance < left_hold_distance)
    )
    right_acquire = int(right_candidates[0]) if len(right_candidates) else None
    left_release = int(left_candidates[0]) if len(left_candidates) else None
    return {
        "right_acquire_frame": right_acquire,
        "left_release_frame": left_release,
        "right_acquire_before_left_release": bool(
            right_acquire is not None
            and left_release is not None
            and right_acquire < left_release
        ),
        "source_reference_right_acquire_frame": int(event_map["RIGHT_ACQUIRE"]),
        "source_reference_left_release_frame": int(event_map["LEFT_RELEASE"]),
        "detection": "nearest method-specific open/hold prototypes with frozen causal search windows",
    }


def evaluate_one(
    method: str,
    output_episode: int,
    entry: dict[str, Any],
    method_actions: np.ndarray,
    offset: int,
    g1: G1Kinematics,
    names: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    source_episode = int(entry["final_dataset_index"])
    path = ROLLOUT_ROOT / method / f"heldout_{output_episode:02d}_source_{source_episode:02d}"
    report_path = path / "rollout_report.json"
    arrays_path = path / "rollout_arrays.npz"
    if not report_path.is_file() or not arrays_path.is_file():
        return {
            "method": method.upper(),
            "output_episode": output_episode,
            "source_episode": source_episode,
            "status": "MISSING",
            "path": str(path),
        }
    report = read_json(report_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if arrays["joint_names"].astype(str).tolist() != names:
        raise RuntimeError(f"rollout named order changed: {path}")
    executed = int(report["executed_frames"])
    requested = int(report["requested_frames"])
    if requested != int(entry["frames"]):
        raise RuntimeError(f"rollout/source length mismatch: {path}")
    if arrays["commanded_action"].shape != (executed, 28) or arrays["measured_state"].shape != (
        executed + 1,
        28,
    ):
        raise RuntimeError(f"rollout array length mismatch: {path}")
    if arrays["source_frame_index"].shape[0] < executed:
        raise RuntimeError(f"rollout source clock shorter than executed prefix: {path}")
    source_frames = arrays["source_frame_index"][:executed].astype(np.int64)
    if not np.array_equal(source_frames, np.arange(executed, dtype=np.int64)):
        raise RuntimeError(f"source video did not advance monotonically at 30 Hz: {path}")
    q = arrays["measured_state"][1 : executed + 1].astype(np.float64)
    commands = arrays["commanded_action"].astype(np.float64)
    if not np.isfinite(q).all() or not np.isfinite(commands).all():
        raise RuntimeError(f"non-finite rollout trajectory: {path}")
    if executed == 0:
        return {
            "method": method.upper(),
            "output_episode": output_episode,
            "source_episode": source_episode,
            "status": report["status"],
            "path": str(path),
            "executed_frames": 0,
            "requested_frames": requested,
            "complete": False,
            "safety_abort": report["safety_abort"],
        }

    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = np.asarray([lookup[str(name)] for name in g1.arm_joint_names], dtype=np.int64)
    hand_indices = {
        side: np.asarray([lookup[name] for name in g1.hand_joint_names[side]], dtype=np.int64)
        for side in ("left", "right")
    }
    geometry = g1.trajectory_geometry(
        q[:, arm_indices],
        q[:, hand_indices["left"]],
        q[:, hand_indices["right"]],
        float(thresholds["collision_penetration_tolerance_m"]),
    )
    with np.load(entry["a_trajectory_path"], allow_pickle=False) as archive_a, np.load(
        entry["b_trajectory_path"], allow_pickle=False
    ) as archive_b:
        source_wrist = {
            side: g1.model_to_world_position(
                archive_a[f"target_{side}_wrist_position_model"][:executed].astype(np.float64)
            )
            for side in ("left", "right")
        }
        source_hand = {
            side: archive_b[f"source_{side}_interaction_frame_position_world"][:executed].astype(
                np.float64
            )
            for side in ("left", "right")
        }
        event_map = {
            str(name): int(frame)
            for name, frame in zip(
                archive_b["event_names"], archive_b["event_frames"], strict=True
            )
        }
        left_phase = archive_b["left_hand_phase"].astype(str)
    predicted_wrist = {
        side: geometry[f"{side}_wrist_position_world"] for side in ("left", "right")
    }
    predicted_hand = {
        side: geometry[f"{side}_grasp_position_world"] for side in ("left", "right")
    }
    wrist_error = {
        side: np.linalg.norm(predicted_wrist[side] - source_wrist[side], axis=1)
        for side in ("left", "right")
    }
    hand_error = {
        side: np.linalg.norm(predicted_hand[side] - source_hand[side], axis=1)
        for side in ("left", "right")
    }
    relation_error = np.linalg.norm(
        (predicted_hand["right"] - predicted_hand["left"])
        - (source_hand["right"] - source_hand["left"]),
        axis=1,
    )
    method_target = method_actions[offset : offset + requested].astype(np.float64)
    if method_target.shape != (requested, 28):
        raise RuntimeError(f"method target extraction failed: {method} {output_episode}")
    ordering = ordering_metrics(q, method_target, event_map, left_phase)
    collision_counts = {
        key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()
    }
    hard_collision_flags = np.logical_or.reduce(
        [
            np.asarray(geometry["collision_flags"][key], dtype=bool)
            for key in ("ARM_TORSO", "CROSS_ARM", "WRIST_OR_PALM_TORSO", "OTHER")
        ]
    )
    hard_collision_frames = int(np.count_nonzero(hard_collision_flags))
    violation = (q < lower[None] - 1e-9) | (q > upper[None] + 1e-9)
    target_delta = q - method_target[:executed]
    derived_path = path / "trajectory_evaluation_arrays.npz"
    atomic_npz(
        derived_path,
        joint_names=np.asarray(names),
        source_frame_index=source_frames,
        source_timestamp_seconds=source_frames.astype(np.float64) / FPS,
        measured_q=q.astype(np.float32),
        commanded_q=commands.astype(np.float32),
        method_specific_retargeted_target_q=method_target[:executed].astype(np.float32),
        predicted_left_wrist_world=predicted_wrist["left"].astype(np.float32),
        predicted_right_wrist_world=predicted_wrist["right"].astype(np.float32),
        source_left_wrist_world=source_wrist["left"].astype(np.float32),
        source_right_wrist_world=source_wrist["right"].astype(np.float32),
        predicted_left_whole_hand_world=predicted_hand["left"].astype(np.float32),
        predicted_right_whole_hand_world=predicted_hand["right"].astype(np.float32),
        source_left_interaction_world=source_hand["left"].astype(np.float32),
        source_right_interaction_world=source_hand["right"].astype(np.float32),
        left_wrist_error_m=wrist_error["left"].astype(np.float32),
        right_wrist_error_m=wrist_error["right"].astype(np.float32),
        left_whole_hand_error_m=hand_error["left"].astype(np.float32),
        right_whole_hand_error_m=hand_error["right"].astype(np.float32),
        bimanual_relation_error_m=relation_error.astype(np.float32),
    )
    result = {
        "method": method.upper(),
        "output_episode": output_episode,
        "source_episode": source_episode,
        "stable_episode_id": entry["stable_episode_id"],
        "status": report["status"],
        "complete": report["status"] == "PASS" and executed == requested,
        "path": str(path),
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "rollout_arrays": str(arrays_path),
        "rollout_arrays_sha256": sha256_file(arrays_path),
        "trajectory_evaluation_arrays": str(derived_path),
        "trajectory_evaluation_arrays_sha256": sha256_file(derived_path),
        "executed_frames": executed,
        "requested_frames": requested,
        "duration_seconds": executed / FPS,
        "source_video_sha256": report["source_video"]["sha256"],
        "source_frame_hashes": arrays["source_rgb_sha256"][:executed].astype(str),
        "wrist_error_mm": stats(
            np.concatenate((wrist_error["left"], wrist_error["right"])), 1000.0
        ),
        "whole_hand_error_mm": stats(
            np.concatenate((hand_error["left"], hand_error["right"])), 1000.0
        ),
        "bimanual_relation_error_mm": stats(relation_error, 1000.0),
        "method_target_tracking": {
            "full_q_rmse_rad": rms(target_delta),
            "arm_rmse_rad": rms(target_delta[:, :14]),
            "dex3_rmse_rad": rms(target_delta[:, 14:]),
        },
        "handoff_ordering": ordering,
        "joint_limits": {
            "violation_elements": int(np.count_nonzero(violation)),
            "violation_frames": int(np.count_nonzero(np.any(violation, axis=1))),
        },
        "collision": {
            "category_frame_counts": collision_counts,
            "hard_collision_frame_incidence": hard_collision_frames,
            "distal_hand_hand_reported_separately": collision_counts.get(
                "DISTAL_HAND_HAND", 0
            ),
        },
        "branch_discontinuity_count": branch_count(q, thresholds),
        "table_contact": report["table_contact"],
        "command_dynamics": rollout_dynamics(commands),
        "measured_dynamics": rollout_dynamics(q),
        "safety_abort": report["safety_abort"],
        "videos": report["videos"],
    }
    atomic_json(path / "trajectory_evaluation.json", result)
    return result


def aggregate_method(rows: list[dict[str, Any]], eligible_outputs: set[int]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("complete")]
    eligible = [row for row in complete if row["output_episode"] in eligible_outputs]
    if not eligible:
        return {
            "completed": len(complete),
            "total": len(rows),
            "primary_paired_eligible": 0,
            "status": "NO_PAIRED_COMPLETE_EPISODES",
        }
    metric_arrays = {"wrist": [], "whole_hand": [], "bimanual": []}
    for row in eligible:
        with np.load(row["trajectory_evaluation_arrays"], allow_pickle=False) as archive:
            metric_arrays["wrist"].extend(
                (archive["left_wrist_error_m"], archive["right_wrist_error_m"])
            )
            metric_arrays["whole_hand"].extend(
                (archive["left_whole_hand_error_m"], archive["right_whole_hand_error_m"])
            )
            metric_arrays["bimanual"].append(archive["bimanual_relation_error_m"])
    return {
        "status": "PASS",
        "completed": len(complete),
        "total": len(rows),
        "primary_paired_eligible": len(eligible),
        "primary_paired_output_episodes": sorted(eligible_outputs),
        "wrist_error_mm": stats(np.concatenate(metric_arrays["wrist"]), 1000.0),
        "whole_hand_error_mm": stats(np.concatenate(metric_arrays["whole_hand"]), 1000.0),
        "bimanual_relation_error_mm": stats(
            np.concatenate(metric_arrays["bimanual"]), 1000.0
        ),
        "handoff_ordering": {
            "successful": int(
                sum(row["handoff_ordering"]["right_acquire_before_left_release"] for row in eligible)
            ),
            "total": len(eligible),
        },
        "joint_limit_violation_frames": int(
            sum(row["joint_limits"]["violation_frames"] for row in eligible)
        ),
        "hard_collision_frame_incidence": int(
            sum(row["collision"]["hard_collision_frame_incidence"] for row in eligible)
        ),
        "distal_hand_hand_frame_incidence": int(
            sum(row["collision"]["distal_hand_hand_reported_separately"] for row in eligible)
        ),
        "branch_discontinuity_count": int(
            sum(row["branch_discontinuity_count"] for row in eligible)
        ),
        "table_contact_occurrence_count": int(
            sum(row["table_contact"]["occurrence_count"] for row in eligible)
        ),
        "command_dynamics": {
            key: (
                float(max(row["command_dynamics"][key] for row in eligible))
                if "max" in key or key == "maximum_joint_step_rad"
                else float(np.mean([row["command_dynamics"][key] for row in eligible]))
            )
            for key in eligible[0]["command_dynamics"]
        },
        "measured_dynamics": {
            key: (
                float(max(row["measured_dynamics"][key] for row in eligible))
                if "max" in key or key == "maximum_joint_step_rad"
                else float(np.mean([row["measured_dynamics"][key] for row in eligible]))
            )
            for key in eligible[0]["measured_dynamics"]
        },
        "per_episode": eligible,
    }


def labeled_panel(bgr: np.ndarray, label: str) -> np.ndarray:
    value = bgr.copy()
    cv2.rectangle(value, (0, 0), (value.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        value,
        label,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return value


def compose_representative_assets(
    manifest: dict[str, Any], rows: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    final_episode = int(manifest["representative_episode"])
    output_episode = next(
        index
        for index, entry in enumerate(manifest["entries"])
        if int(entry["final_dataset_index"]) == final_episode
    )
    entry = manifest["entries"][output_episode]
    source_path = Path(entry["source_rgb_identity"]["canonical_video_path"])
    row_a = next(row for row in rows["a"] if row["output_episode"] == output_episode)
    row_b = next(row for row in rows["b"] if row["output_episode"] == output_episode)
    if not row_a.get("complete") or not row_b.get("complete"):
        return {
            "status": "REPRESENTATIVE_PAIR_NOT_COMPLETE",
            "source_final_episode": final_episode,
            "output_episode": output_episode,
        }
    FIGURES.mkdir(parents=True, exist_ok=True)
    video_records = {}
    still_records = {}
    phase_frames = manifest["complete_source_phase_audit"][str(final_episode)][
        "nine_probe_frames"
    ]
    selected_stills = {
        "left_grasp": int(phase_frames["left_grasp_owned"]),
        "left_transport": int(phase_frames["left_transport"]),
        "handoff": int(phase_frames["dual_contact_transfer"]),
        "right_owned": int(phase_frames["right_owned"]),
        "release": int(phase_frames["release"]),
    }
    for view in ("overview", "three_quarter"):
        robot_a = Path(row_a["videos"][view]["path"])
        robot_b = Path(row_b["videos"][view]["path"])
        if not robot_a.is_file() or not robot_b.is_file():
            raise FileNotFoundError(f"representative robot video missing for {view}")
        captures = [cv2.VideoCapture(str(path)) for path in (source_path, robot_a, robot_b)]
        if not all(capture.isOpened() for capture in captures):
            raise RuntimeError(f"could not open representative videos for {view}")
        out = FIGURES / f"figure4_source_vs_act_a_vs_act_b_{view}.mp4"
        writer = cv2.VideoWriter(
            str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 480)
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not create {out}")
        stills: dict[int, np.ndarray] = {}
        frame = 0
        while True:
            decoded = [capture.read() for capture in captures]
            if not all(ok for ok, _ in decoded):
                break
            panels = [
                labeled_panel(decoded[0][1], "SOURCE ALOHA"),
                labeled_panel(decoded[1][1], "ACT-A40 G1"),
                labeled_panel(decoded[2][1], "ACT-B40 G1"),
            ]
            combined = np.hstack(panels)
            writer.write(combined)
            if frame in selected_stills.values():
                stills[frame] = combined.copy()
            frame += 1
        writer.release()
        for capture in captures:
            capture.release()
        expected = int(entry["frames"])
        if frame != expected:
            raise RuntimeError(f"representative video length mismatch {view}: {frame}/{expected}")
        video_records[view] = {
            "path": str(out),
            "sha256": sha256_file(out),
            "frames": frame,
            "source_final_episode": final_episode,
            "selection_rule": manifest[
                "representative_episode_rule_frozen_before_policy_results"
            ],
        }
        if view == "overview":
            for phase, index in selected_stills.items():
                image = stills.get(index)
                if image is None:
                    raise RuntimeError(f"missing representative still frame {phase}/{index}")
                path = FIGURES / f"figure4_{phase}_source_vs_act_a_vs_act_b.png"
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError(f"could not write {path}")
                still_records[phase] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "frame": index,
                    "source_timestamp_seconds": index / FPS,
                }
    return {
        "status": "PASS",
        "source_final_episode": final_episode,
        "heldout_output_episode": output_episode,
        "selection_rule_frozen_before_policy_results": manifest[
            "representative_episode_rule_frozen_before_policy_results"
        ],
        "videos": video_records,
        "stills": still_records,
        "physical_contact_success_claimed": False,
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    keys = list(rows[0])
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    lines.extend("| " + " | ".join(str(row[key]) for key in keys) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    contract = read_json(EVALUATION_CONTRACT)
    if contract["status"] != "FROZEN_BEFORE_ANY_PAPER_SOURCE_CONDITIONED_ROLLOUT":
        raise RuntimeError("source rollout evaluator contract is not frozen")
    manifest = read_json(HELDOUT_MANIFEST)
    entries = manifest["entries"]
    offsets = episode_offsets(entries)
    method_actions = {method: parquet_actions(path) for method, path in METHOD_DATASETS.items()}
    names, lower, upper, _ = frozen_interfaces()
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    thresholds = read_json(
        ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
    )["unchanged_acceptance"]
    rows = {
        method: [
            evaluate_one(
                method,
                output_episode,
                entry,
                method_actions[method],
                offsets[output_episode],
                g1,
                names,
                lower,
                upper,
                thresholds,
            )
            for output_episode, entry in enumerate(entries)
        ]
        for method in ("a", "b")
    }
    complete_sets = {
        method: {row["output_episode"] for row in values if row.get("complete")}
        for method, values in rows.items()
    }
    paired_complete = complete_sets["a"] & complete_sets["b"]
    # The same source decoder/clock must be exact for every executed paired prefix.
    source_identity_rows = []
    for output_episode in range(8):
        row_a = rows["a"][output_episode]
        row_b = rows["b"][output_episode]
        if "source_frame_hashes" not in row_a or "source_frame_hashes" not in row_b:
            exact = False
            compared = 0
        else:
            compared = min(len(row_a["source_frame_hashes"]), len(row_b["source_frame_hashes"]))
            exact = bool(
                row_a["source_video_sha256"] == row_b["source_video_sha256"]
                and np.array_equal(
                    np.asarray(row_a["source_frame_hashes"][:compared]),
                    np.asarray(row_b["source_frame_hashes"][:compared]),
                )
            )
        source_identity_rows.append(
            {
                "output_episode": output_episode,
                "source_episode": int(entries[output_episode]["final_dataset_index"]),
                "compared_prefix_frames": compared,
                "exact_source_video_and_decoded_frame_identity": exact,
            }
        )
    aggregates = {
        method: aggregate_method(values, paired_complete) for method, values in rows.items()
    }
    representative_assets = compose_representative_assets(manifest, rows)

    table_rows = []
    for metric, key, statistic, unit in [
        ("completed rollouts", "completed", None, "episodes"),
        ("paired-complete eligible", "primary_paired_eligible", None, "episodes"),
        ("source wrist error mean", "wrist_error_mm", "mean", "mm"),
        ("source wrist error p95", "wrist_error_mm", "p95", "mm"),
        ("whole-hand error mean", "whole_hand_error_mm", "mean", "mm"),
        ("whole-hand error p95", "whole_hand_error_mm", "p95", "mm"),
        ("bimanual relation error mean", "bimanual_relation_error_mm", "mean", "mm"),
        ("bimanual relation error p95", "bimanual_relation_error_mm", "p95", "mm"),
        ("joint-limit violation frames", "joint_limit_violation_frames", None, "frames"),
        ("hard collision frame incidence", "hard_collision_frame_incidence", None, "frames"),
        ("branch discontinuities", "branch_discontinuity_count", None, "events"),
        ("table contact occurrences", "table_contact_occurrence_count", None, "frames"),
    ]:
        values = []
        for method in ("a", "b"):
            value = aggregates[method].get(key)
            if statistic is not None and isinstance(value, dict):
                value = value[statistic]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))
        table_rows.append(
            {"metric": metric, "ACT-A40": values[0], "ACT-B40": values[1], "unit": unit}
        )
    table_rows.extend(
        [
            {
                "metric": "right-acquire-before-left-release",
                "ACT-A40": (
                    f"{aggregates['a']['handoff_ordering']['successful']}/"
                    f"{aggregates['a']['handoff_ordering']['total']}"
                    if "handoff_ordering" in aggregates["a"]
                    else "NA"
                ),
                "ACT-B40": (
                    f"{aggregates['b']['handoff_ordering']['successful']}/"
                    f"{aggregates['b']['handoff_ordering']['total']}"
                    if "handoff_ordering" in aggregates["b"]
                    else "NA"
                ),
                "unit": "episodes",
            },
            {
                "metric": "measured max adjacent step",
                "ACT-A40": (
                    f"{aggregates['a']['measured_dynamics']['maximum_joint_step_rad']:.6f}"
                    if "measured_dynamics" in aggregates["a"]
                    else "NA"
                ),
                "ACT-B40": (
                    f"{aggregates['b']['measured_dynamics']['maximum_joint_step_rad']:.6f}"
                    if "measured_dynamics" in aggregates["b"]
                    else "NA"
                ),
                "unit": "rad",
            },
            {
                "metric": "measured jerk RMS",
                "ACT-A40": (
                    f"{aggregates['a']['measured_dynamics']['jerk_rms_rad_s3']:.3f}"
                    if "measured_dynamics" in aggregates["a"]
                    else "NA"
                ),
                "ACT-B40": (
                    f"{aggregates['b']['measured_dynamics']['jerk_rms_rad_s3']:.3f}"
                    if "measured_dynamics" in aggregates["b"]
                    else "NA"
                ),
                "unit": "rad/s^3",
            },
            *[
                {
                    "metric": f"{trajectory_label} {metric_label}",
                    "ACT-A40": (
                        f"{aggregates['a'][trajectory_key][metric_key]:.{precision}f}"
                        if trajectory_key in aggregates["a"]
                        else "NA"
                    ),
                    "ACT-B40": (
                        f"{aggregates['b'][trajectory_key][metric_key]:.{precision}f}"
                        if trajectory_key in aggregates["b"]
                        else "NA"
                    ),
                    "unit": unit,
                }
                for trajectory_label, trajectory_key, metric_label, metric_key, unit, precision in [
                    (
                        "command",
                        "command_dynamics",
                        "max adjacent step",
                        "maximum_joint_step_rad",
                        "rad",
                        6,
                    ),
                    (
                        "command",
                        "command_dynamics",
                        "qdot RMS",
                        "qdot_rms_rad_s",
                        "rad/s",
                        6,
                    ),
                    (
                        "command",
                        "command_dynamics",
                        "qddot RMS",
                        "qddot_rms_rad_s2",
                        "rad/s^2",
                        3,
                    ),
                    (
                        "command",
                        "command_dynamics",
                        "jerk RMS",
                        "jerk_rms_rad_s3",
                        "rad/s^3",
                        3,
                    ),
                    (
                        "measured",
                        "measured_dynamics",
                        "qdot RMS",
                        "qdot_rms_rad_s",
                        "rad/s",
                        6,
                    ),
                    (
                        "measured",
                        "measured_dynamics",
                        "qddot RMS",
                        "qddot_rms_rad_s2",
                        "rad/s^2",
                        3,
                    ),
                ]
            ],
        ]
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    atomic_csv(TABLES / "table3_source_conditioned_rollout.csv", table_rows)
    (TABLES / "table3_source_conditioned_rollout.md").write_text(
        markdown_table(table_rows), encoding="utf-8"
    )
    status = (
        "PASS"
        if len(paired_complete) == 8
        and all(row["exact_source_video_and_decoded_frame_identity"] for row in source_identity_rows)
        and representative_assets["status"] == "PASS"
        else "INCOMPLETE_OR_SAFETY_ABORT"
    )
    result = {
        "schema_version": "paper_core_source_conditioned_rollout_result_v1",
        "status": status,
        "experiment_name": "SOURCE-VIDEO-CONDITIONED G1 POLICY ROLLOUT",
        "evaluation_contract": str(EVALUATION_CONTRACT),
        "evaluation_contract_sha256": sha256_file(EVALUATION_CONTRACT),
        "completed_sets": {key: sorted(value) for key, value in complete_sets.items()},
        "paired_complete_output_episodes": sorted(paired_complete),
        "source_input_identity": source_identity_rows,
        "methods": aggregates,
        "all_rollouts": rows,
        "representative_assets": representative_assets,
        "table3": table_rows,
        "claims": {
            "source_video_conditioned_generated_motion": True,
            "g1_onboard_visual_policy": False,
            "physical_manipulation_success": False,
            "real_robot": False,
            "sim_to_real": False,
        },
    }
    atomic_json(ROLLOUT_ROOT / "experiment3_result.json", result)
    atomic_json(TABLES / "table3_source_conditioned_rollout.json", {"rows": table_rows})
    print(json.dumps(result, indent=2, default=json_default))


if __name__ == "__main__":
    main()
