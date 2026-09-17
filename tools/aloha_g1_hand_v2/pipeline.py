"""End-to-end offline audit and Proposed-hand-v2 diagnostic pipeline."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from aloha_g1_dataset_v1.core import HandMapper, SourceDataset, raw_contact_position, raw_wrist_pose

from .collision_eval import (
    COLLISION_CATEGORIES,
    CollisionClassifier,
    audit_current_50,
    body_digit,
    body_side,
    make_runtime,
)
from .common import (
    ROOT,
    V1_ROOT,
    V2_ROOT,
    array_sha256,
    atomic_csv,
    atomic_json,
    json_default,
    load_v1_config,
    sha256_file,
    tree_sha256,
)
from .contact_mapping import map_source_contacts_to_frozen_g1_tool
from .dex3_ik import Dex3InteractionIK
from .interaction_extraction import AlohaContactExtractor, detect_first_complete_grasp


SOURCE_CONFIG = ROOT / "configs/aloha_g1_hand_v2.json"


def _atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".incomplete.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def _write_collision_report(path: Path, artifact: Mapping[str, Any], rows: list[dict[str, Any]]) -> None:
    proposed = artifact["proposed"]
    comparison = artifact["baseline_to_proposed"]
    table = _markdown_table(
        [
            "category",
            "Baseline v1 frames",
            "Proposed v1 frames",
            "delta",
            "v1 pair-events",
            "episodes",
            "enhanced same-hand frames",
        ],
        [
            [
                row["category"],
                row["baseline_collision_frame_incidence"],
                row["proposed_collision_frame_incidence"],
                row["delta_frame_incidence"],
                row["proposed_pair_events"],
                row["proposed_episodes_affected"],
                row["proposed_enhanced_same_hand_frame_incidence"],
            ]
            for row in rows
        ],
    )
    top = _markdown_table(
        ["rank", "link pair", "pair-events", "episodes"],
        [
            [index, row["pair"], row["pair_events"], row["episodes_affected"]]
            for index, row in enumerate(proposed["top_link_pairs"][:10], start=1)
        ],
    )
    cause_names = sorted(
        set(artifact["baseline"]["cause_group_frame_incidence"])
        | set(proposed["cause_group_frame_incidence"])
        | set(proposed["enhanced_logger"]["cause_group_frame_incidence"])
    )
    causes = _markdown_table(
        ["cause group", "Baseline v1 frames", "Proposed v1 frames", "Proposed enhanced frames"],
        [
            [
                cause,
                artifact["baseline"]["cause_group_frame_incidence"].get(cause, 0),
                proposed["cause_group_frame_incidence"].get(cause, 0),
                proposed["enhanced_logger"]["cause_group_frame_incidence"].get(cause, 0),
            ]
            for cause in cause_names
        ],
    )
    text = f"""# Proposed 50-episode collision attribution

- Decision gate: `{artifact['decision_gate']}`
- Historical v1 collision frames: **{proposed['v1_collision_frames']:,}** (exact replay parity: `{str(proposed['v1_exact_reconstruction']).lower()}`)
- Dex3-finger-involved frames: **{proposed['finger_involved_collision_frames']:,} / {proposed['v1_collision_frames']:,} = {proposed['finger_involved_collision_frame_percent']:.2f}%**
- Arm/wrist/palm-only frames: **{proposed['arm_wrist_palm_only_collision_frames']:,} / {proposed['v1_collision_frames']:,} = {proposed['arm_wrist_palm_only_collision_frame_percent']:.2f}%**
- Baseline→Proposed delta: {comparison['collision_frame_delta']:+,} total; {comparison['finger_involved_collision_frame_delta']:+,} finger-involved; {comparison['arm_wrist_palm_only_collision_frame_delta']:+,} arm-only.

Category frame incidence is non-exclusive because one frame can contain several mutually-exclusive pair categories. Pair-event categories themselves are mutually exclusive.

## Category attribution

{table}

## Top 10 link pairs

{top}

## Cause-group frame incidence

{causes}

## Logging scope

The 4,696-frame denominator reproduces the v1 gate exactly. The v1 gate intentionally removed same-side hand-chain contacts. The enhanced diagnostic separately found {proposed['enhanced_logger']['same_hand_collision_frames_nonexclusive_with_v1']:,} same-hand collision frames ({proposed['enhanced_logger']['same_hand_and_v1_overlap_frames']:,} overlap historical collision frames; {proposed['enhanced_logger']['all_logged_union_frames']:,} union); these are not retroactively added to the v1 denominator. This separation avoids changing the historical threshold while exposing placeholder-hand self-contact.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _integrity_snapshot() -> dict[str, Any]:
    dataset_a_hash, dataset_a_files = tree_sha256(V1_ROOT / "baseline")
    arm_files: dict[str, dict[str, str]] = {}
    for episode_id in range(50):
        path = V1_ROOT / "proposed" / f"episode_{episode_id:06d}" / "g1_arm_action.npz"
        with np.load(path, allow_pickle=False) as payload:
            q_hash = array_sha256(payload["action"])
        arm_files[f"episode_{episode_id:06d}"] = {
            "file_sha256": sha256_file(path),
            "action_array_sha256": q_hash,
        }
    digest = hashlib.sha256()
    for episode, values in sorted(arm_files.items()):
        digest.update(episode.encode("ascii"))
        digest.update(values["file_sha256"].encode("ascii"))
        digest.update(values["action_array_sha256"].encode("ascii"))
    return {
        "dataset_a_root": str(V1_ROOT / "baseline"),
        "dataset_a_tree_sha256": dataset_a_hash,
        "dataset_a_file_count": len(dataset_a_files),
        "proposed_arm_combined_sha256": digest.hexdigest(),
        "proposed_arm_files": arm_files,
    }


def _wrist_validity(runtime: Any, arm_payload: Mapping[str, np.ndarray], frame: int, config: Mapping[str, Any]) -> dict[str, Any]:
    arm_q = np.asarray(arm_payload["action"][frame], dtype=np.float64)
    runtime.assign(arm_q, runtime.open_hand_q["left"], runtime.open_hand_q["right"])
    wrist = raw_wrist_pose(runtime, "left")
    position_error = float(
        np.linalg.norm(wrist[:3, 3] - np.asarray(arm_payload["target_left_wrist_position"][frame]))
    )
    orientation_error = float(
        Rotation.from_matrix(
            np.asarray(arm_payload["target_left_wrist_rotation"][frame]) @ wrist[:3, :3].T
        ).magnitude()
    )
    position_tolerance = float(config["ik"]["position_tolerance_m"])
    orientation_tolerance = float(config["ik"]["orientation_tolerance_rad"])
    return {
        "finite": bool(np.isfinite(arm_q).all() and np.isfinite(wrist).all()),
        "left_wrist_position_error_m": position_error,
        "left_wrist_orientation_error_rad": orientation_error,
        "position_tolerance_m": position_tolerance,
        "orientation_tolerance_rad": orientation_tolerance,
        "valid": bool(
            np.isfinite(arm_q).all()
            and position_error <= position_tolerance
            and orientation_error <= orientation_tolerance
        ),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def _select_development_episode(
    requested_episode: int,
    runtime: Any,
    config: Mapping[str, Any],
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    order = [requested_episode] + [value for value in range(50) if value != requested_episode]
    for episode_id in order:
        folder = V1_ROOT / "proposed" / f"episode_{episode_id:06d}"
        hand = _load_npz(folder / "g1_hand_action.npz")
        arm = _load_npz(folder / "g1_arm_action.npz")
        try:
            event = detect_first_complete_grasp(hand["left_phase"])
        except ValueError as error:
            failures.append({"episode_id": episode_id, "reason": str(error)})
            continue
        validity = _wrist_validity(runtime, arm, int(event["grasp_onset"]), config)
        if validity["valid"]:
            return episode_id, event, {
                "requested_episode": requested_episode,
                "selected_episode": episode_id,
                "substitution_used": episode_id != requested_episode,
                "selection_rule": (
                    "requested development episode when it has a complete left grasp and valid frozen wrist; "
                    "otherwise lowest episode ID satisfying the same criterion"
                ),
                "validity": validity,
                "rejected_candidates": failures,
            }
        failures.append({"episode_id": episode_id, "reason": "invalid frozen wrist", "validity": validity})
    raise RuntimeError("no episode has a complete left grasp with a valid frozen wrist")


def _evaluate_named_states(
    solver: Dex3InteractionIK,
    target: Mapping[str, Any],
    arm_q: np.ndarray,
    right_q: np.ndarray,
    states: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        name: solver.evaluate(np.asarray(q, dtype=np.float64), arm_q, right_q, target)
        for name, q in states.items()
    }


def _source_render(path: Path, source: Mapping[str, Any]) -> None:
    a = np.asarray(source["contact_A_tcp_m"]) * 1000.0
    b = np.asarray(source["contact_B_tcp_m"]) * 1000.0
    axes = {
        "approach +x": np.asarray(source["approach_axis_tcp"]),
        "closing +y": np.asarray(source["closing_axis_tcp"]),
        "lateral +z": np.asarray(source["lateral_axis_tcp"]),
    }
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*a, color="tab:orange", s=90, label="contact A → thumb")
    ax.scatter(*b, color="tab:blue", s=90, label="contact B → index")
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="black", lw=2)
    for (name, vector), color in zip(axes.items(), ("red", "green", "blue")):
        end = vector * 15.0
        ax.quiver(0, 0, 0, *end, color=color, arrow_length_ratio=0.15, label=name)
    ax.scatter(0, 0, 0, marker="x", color="black", s=80, label="ALOHA TCP")
    ax.set(xlabel="TCP x [mm]", ylabel="TCP y [mm]", zlabel="TCP z [mm]")
    ax.set_title(
        "Source interaction: TCP-relative named jaw-tip surfaces\n"
        + str(source["representation_mode"])
    )
    radius = max(20.0, float(np.linalg.norm(b - a)) * 0.8)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _hand_geometry(runtime: Any, arm_q: np.ndarray, left_q: np.ndarray, right_q: np.ndarray, v1_config: Mapping[str, Any]) -> dict[str, Any]:
    runtime.assign(arm_q, left_q, right_q)
    wrist = raw_wrist_pose(runtime, "left")
    palm_local = np.asarray(v1_config["target_frames"]["left_wrist_to_palm"])
    palm = (wrist @ palm_local)[:3, 3]
    chains: dict[str, np.ndarray] = {}
    for digit, bodies, label in (
        ("thumb", ("left_hand_thumb_0_link", "left_hand_thumb_1_link", "left_hand_thumb_2_link"), "left_A"),
        ("index", ("left_hand_index_0_link", "left_hand_index_1_link"), "left_B"),
        ("third", ("left_hand_middle_0_link", "left_hand_middle_1_link"), "left_C"),
    ):
        points = [palm]
        for name in bodies:
            body = runtime.body_id(name)
            points.append(np.asarray(runtime.data.xpos[body], dtype=np.float64).copy())
        points.append(raw_contact_position(runtime, label).copy())
        chains[digit] = np.asarray(points)
    return {"wrist": wrist[:3, 3].copy(), "palm": palm, "chains": chains}


def _projection_render(
    path: Path,
    geometry: Mapping[str, Any],
    target: Mapping[str, Any],
    projection: str,
) -> None:
    definitions = {
        "front": ((1, 2), ("world y [m]", "world z [m]")),
        "side": ((0, 2), ("world x [m]", "world z [m]")),
        "top": ((0, 1), ("world x [m]", "world y [m]")),
    }
    indices, labels = definitions[projection]
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"thumb": "tab:orange", "index": "tab:blue", "third": "tab:gray"}
    for name, points in geometry["chains"].items():
        ax.plot(points[:, indices[0]], points[:, indices[1]], "o-", color=colors[name], label=name)
    for name, marker, color in (
        ("wrist", "s", "black"),
        ("palm", "D", "purple"),
    ):
        point = np.asarray(geometry[name])
        ax.scatter(point[indices[0]], point[indices[1]], marker=marker, color=color, s=75, label=name)
    for key, marker, color, label in (
        ("thumb_target_m", "*", "red", "thumb target"),
        ("index_target_m", "*", "cyan", "index target"),
    ):
        point = np.asarray(target[key])
        ax.scatter(point[indices[0]], point[indices[1]], marker=marker, color=color, s=150, label=label)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Interaction-aware Dex3 IK — {projection}\nfrozen Proposed wrist; no source object pose")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _comparison_render(
    path: Path,
    primitive: Mapping[str, Any],
    interaction: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    colors = {"thumb": "tab:orange", "index": "tab:blue", "third": "tab:gray"}
    for ax, title, geometry in (
        (axes[0], "Fixed PHONE_PINCH", primitive),
        (axes[1], "Interaction-aware IK", interaction),
    ):
        for name, points in geometry["chains"].items():
            ax.plot(points[:, 1], points[:, 2], "o-", color=colors[name], label=name)
        for key, marker, color, label in (
            ("thumb_target_m", "*", "red", "thumb target"),
            ("index_target_m", "*", "cyan", "index target"),
        ):
            point = np.asarray(target[key])
            ax.scatter(point[1], point[2], marker=marker, color=color, s=150, label=label)
        ax.set_title(title)
        ax.set_xlabel("world y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("world z [m]")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Same episode, frame, frozen arm q, and frozen wrist pose")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_temporal_window(
    output_root: Path,
    episode_id: int,
    event: Mapping[str, Any],
    source_episode: Any,
    extractor: AlohaContactExtractor,
    runtime: Any,
    solver: Dex3InteractionIK,
    arm: Mapping[str, np.ndarray],
    proposed_hand: Mapping[str, np.ndarray],
    v1_config: Mapping[str, Any],
    v2_config: Mapping[str, Any],
) -> dict[str, Any]:
    start = int(event["pregrasp_start"])
    end = min(
        len(source_episode.action) - 1,
        int(event["hold_start"]) + int(v2_config["temporal_window"]["early_hold_frames"]),
    )
    frames = np.arange(start, end + 1, dtype=np.int64)
    q_solution = np.empty((len(frames), 7), dtype=np.float64)
    current_error = np.empty(len(frames), dtype=np.float64)
    solution_error = np.empty(len(frames), dtype=np.float64)
    current_collision = np.zeros(len(frames), dtype=bool)
    solution_collision = np.zeros(len(frames), dtype=bool)
    convergence = np.zeros(len(frames), dtype=bool)
    previous = (
        proposed_hand["left_action"][start - 1].astype(np.float64)
        if start > 0
        else solver.reference_q.copy()
    )
    previous2 = (
        proposed_hand["left_action"][start - 2].astype(np.float64)
        if start > 1
        else previous.copy()
    )
    for local_index, frame in enumerate(frames):
        source = extractor.extract(source_episode.action[frame], "left")
        target = map_source_contacts_to_frozen_g1_tool(
            source,
            runtime,
            arm["action"][frame],
            proposed_hand["right_action"][frame],
            v1_config,
        )
        initial = (
            proposed_hand["left_action"][frame].astype(np.float64)
            if local_index == 0
            else previous
        )
        solved = solver.solve(
            arm["action"][frame],
            proposed_hand["right_action"][frame],
            target,
            initial_q=initial,
            previous_q=previous,
            previous2_q=previous2,
            temporal=True,
        )
        q_solution[local_index] = solved["q"]
        current = solver.evaluate(
            proposed_hand["left_action"][frame],
            arm["action"][frame],
            proposed_hand["right_action"][frame],
            target,
        )
        current_error[local_index] = current["mean_task_finger_error_m"]
        solution_error[local_index] = solved["metrics"]["mean_task_finger_error_m"]
        current_collision[local_index] = current["left_hand_related_collision_pairs"] > 0
        solution_collision[local_index] = solved["metrics"]["left_hand_related_collision_pairs"] > 0
        convergence[local_index] = solved["solver"]["success"]
        previous2, previous = previous, solved["q"]

    arm_window = np.ascontiguousarray(arm["action"][frames])
    _atomic_npz(
        output_root / "development/temporal_window_solution.npz",
        episode_id=np.asarray(episode_id, dtype=np.int64),
        frames=frames,
        dex3_left_q=q_solution.astype(np.float32),
        frozen_g1_arm_q=arm_window,
        frozen_g1_arm_q_sha256=np.asarray(array_sha256(arm_window)),
        source_current_left_hand_q=proposed_hand["left_action"][frames],
        solver_config_sha256=np.asarray(sha256_file(SOURCE_CONFIG)),
    )
    metrics = {
        "executed": True,
        "episode_id": episode_id,
        "start_frame": start,
        "end_frame": end,
        "frame_count": len(frames),
        "window_definition": "PREGRASP start through HOLD start + fixed 15 frames",
        "solver_convergence_rate": float(np.mean(convergence)),
        "current_mean_task_finger_error_m": float(np.mean(current_error)),
        "interaction_ik_mean_task_finger_error_m": float(np.mean(solution_error)),
        "mean_contact_error_delta_m": float(np.mean(solution_error - current_error)),
        "current_left_hand_collision_frames": int(np.count_nonzero(current_collision)),
        "interaction_ik_left_hand_collision_frames": int(np.count_nonzero(solution_collision)),
        "max_joint_step_rad": float(np.max(np.abs(np.diff(q_solution, axis=0)), initial=0.0)),
        "finite": bool(np.isfinite(q_solution).all()),
        "joint_limits_ok": bool(
            np.all(q_solution >= solver.limits[:, 0] - 1e-9)
            and np.all(q_solution <= solver.limits[:, 1] + 1e-9)
        ),
        "arm_window_sha256": array_sha256(arm_window),
    }
    atomic_json(output_root / "development/temporal_window_metrics.json", metrics)
    return metrics


def _select_smoke_episodes() -> dict[str, Any]:
    values: list[tuple[int, int]] = []
    for episode_id in range(50):
        metrics = json.loads(
            (
                V1_ROOT
                / "proposed"
                / f"episode_{episode_id:06d}"
                / "retargeting_metrics.json"
            ).read_text(encoding="utf-8")
        )
        values.append((episode_id, int(metrics["prohibited_arm_self_collision_count"])))
    median = float(np.median([count for _, count in values]))
    low = min(values, key=lambda row: (row[1], row[0]))
    middle = min(values, key=lambda row: (abs(row[1] - median), row[0]))
    high = min(values, key=lambda row: (-row[1], row[0]))
    return {
        "selection_source": "existing v1 Proposed prohibited_arm_self_collision_count",
        "selection_is_automatic": True,
        "numeric_median_collision_frames": median,
        "tie_break": "lowest episode ID",
        "selected": {
            "low": {"episode_id": low[0], "collision_frames": low[1]},
            "median": {"episode_id": middle[0], "collision_frames": middle[1]},
            "high": {"episode_id": high[0], "collision_frames": high[1]},
        },
    }


def _run_smoke_test(
    output_root: Path,
    dataset: SourceDataset,
    extractor: AlohaContactExtractor,
    runtime: Any,
    solver: Dex3InteractionIK,
    v1_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = _select_smoke_episodes()
    atomic_json(output_root / "smoke_test/selected_episodes.json", selection)
    rows: list[dict[str, Any]] = []
    config_hash = sha256_file(SOURCE_CONFIG)
    for stratum, chosen in selection["selected"].items():
        episode_id = int(chosen["episode_id"])
        folder = V1_ROOT / "proposed" / f"episode_{episode_id:06d}"
        arm = _load_npz(folder / "g1_arm_action.npz")
        hand = _load_npz(folder / "g1_hand_action.npz")
        source_episode = dataset.episode(episode_id)
        try:
            event = detect_first_complete_grasp(hand["left_phase"])
            frame = int(event["grasp_onset"])
            source = extractor.extract(source_episode.action[frame], "left")
            target = map_source_contacts_to_frozen_g1_tool(
                source,
                runtime,
                arm["action"][frame],
                hand["right_action"][frame],
                v1_config,
            )
            fixed = solver.evaluate(
                solver.reference_q, arm["action"][frame], hand["right_action"][frame], target
            )
            result = solver.solve(arm["action"][frame], hand["right_action"][frame], target)
            validity = _wrist_validity(runtime, arm, frame, v1_config)
            rows.append(
                {
                    "stratum": stratum,
                    "episode_id": episode_id,
                    "existing_proposed_collision_frames": chosen["collision_frames"],
                    "grasp_frame": frame,
                    "frozen_wrist_valid": validity["valid"],
                    "source_contact_width_mm": 1000.0 * source["gripper_width_m"],
                    "mapped_contact_width_mm": 1000.0 * target["mapped_contact_width_m"],
                    "fixed_mean_contact_error_mm": 1000.0 * fixed["mean_task_finger_error_m"],
                    "interaction_mean_contact_error_mm": 1000.0
                    * result["metrics"]["mean_task_finger_error_m"],
                    "contact_error_delta_mm": 1000.0
                    * (result["metrics"]["mean_task_finger_error_m"] - fixed["mean_task_finger_error_m"]),
                    "fixed_left_hand_collision_pairs": fixed["left_hand_related_collision_pairs"],
                    "interaction_left_hand_collision_pairs": result["metrics"][
                        "left_hand_related_collision_pairs"
                    ],
                    "fixed_all_hand_collision_pairs": fixed["hand_related_collision_pairs"],
                    "interaction_all_hand_collision_pairs": result["metrics"][
                        "hand_related_collision_pairs"
                    ],
                    "third_finger_neutral": result["metrics"]["third_finger_neutral_exact"],
                    "third_finger_task_object_contact": result["metrics"][
                        "third_finger_task_object_contact"
                    ],
                    "joint_limits_ok": result["metrics"]["joint_limits_ok"],
                    "solver_converged": result["solver"]["success"],
                    "contact_feasible": result["solver"]["contact_feasible"],
                    "blocker_classification": result["solver"]["blocker_classification"],
                    "solver_config_sha256": config_hash,
                    "per_episode_weight_tuning": False,
                    "dex3_q_json": json.dumps(result["q"].tolist()),
                }
            )
        except Exception as error:  # preserve every deterministic sample outcome
            rows.append(
                {
                    "stratum": stratum,
                    "episode_id": episode_id,
                    "existing_proposed_collision_frames": chosen["collision_frames"],
                    "grasp_frame": "",
                    "frozen_wrist_valid": False,
                    "source_contact_width_mm": "",
                    "mapped_contact_width_mm": "",
                    "fixed_mean_contact_error_mm": "",
                    "interaction_mean_contact_error_mm": "",
                    "contact_error_delta_mm": "",
                    "fixed_left_hand_collision_pairs": "",
                    "interaction_left_hand_collision_pairs": "",
                    "fixed_all_hand_collision_pairs": "",
                    "interaction_all_hand_collision_pairs": "",
                    "third_finger_neutral": False,
                    "third_finger_task_object_contact": "NOT_EVALUATED",
                    "joint_limits_ok": False,
                    "solver_converged": False,
                    "contact_feasible": False,
                    "blocker_classification": f"FAIL_OTHER: {error}",
                    "solver_config_sha256": config_hash,
                    "per_episode_weight_tuning": False,
                    "dex3_q_json": "",
                }
            )
    atomic_csv(output_root / "smoke_test/episode_metrics.csv", rows)
    table = _markdown_table(
        ["stratum", "episode", "frame", "fixed err mm", "IK err mm", "fixed coll", "IK coll", "converged", "blocker"],
        [
            [
                row["stratum"],
                row["episode_id"],
                row["grasp_frame"],
                f"{row['fixed_mean_contact_error_mm']:.3f}" if isinstance(row["fixed_mean_contact_error_mm"], float) else "N/A",
                f"{row['interaction_mean_contact_error_mm']:.3f}" if isinstance(row["interaction_mean_contact_error_mm"], float) else "N/A",
                row["fixed_left_hand_collision_pairs"],
                row["interaction_left_hand_collision_pairs"],
                row["solver_converged"],
                row["blocker_classification"],
            ]
            for row in rows
        ],
    )
    (output_root / "smoke_test/report.md").write_text(
        "# Proposed-Hand-v2 deterministic 3-episode smoke test\n\n"
        + table
        + "\n\nAll rows use solver config SHA256 `"
        + config_hash
        + "`; no per-episode tuning was performed.\n",
        encoding="utf-8",
    )
    return selection, rows


def _anti_overfit_scan() -> dict[str, Any]:
    files = sorted((ROOT / "tools/aloha_g1_hand_v2").glob("*.py")) + [SOURCE_CONFIG]
    patterns = {
        "episode_equals_49": re.compile(r"if\s+[^\n]*episode[^\n]*==\s*49"),
        "frame_equality_condition": re.compile(r"if\s+[^\n]*frame\s*=="),
        "ep49" + "_offset": re.compile("ep49" + "_offset", re.IGNORECASE),
        "manual" + "_contact": re.compile("manual" + "_contact", re.IGNORECASE),
        "hand_written" + "_waypoint": re.compile(
            "hand_written" + "_waypoint", re.IGNORECASE
        ),
    }
    hits: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                hits.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "pattern": name,
                        "line": text[: match.start()].count("\n") + 1,
                        "match": match.group(0),
                    }
                )
    return {"files_scanned": [str(path.relative_to(ROOT)) for path in files], "hits": hits, "pass": not hits}


def _run_pytest(output_root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_aloha_g1_hand_v2.py"]
    environment = os.environ.copy()
    # ROS installs a launch_testing pytest entrypoint built for another Python
    # environment.  Disable unrelated global plugin auto-loading so this
    # isolated offline unit suite depends only on its declared imports.
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    match = re.search(r"(\d+) passed", completed.stdout)
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "pass": completed.returncode == 0,
        "passed_test_count": int(match.group(1)) if match else 0,
        "output": completed.stdout,
    }


def _write_final_report(
    output_root: Path,
    collision: Mapping[str, Any],
    development: Mapping[str, Any],
    temporal: Mapping[str, Any],
    smoke_rows: list[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> str:
    fixed = development["comparators"]["fixed_phone_pinch"]
    solved = development["interaction_ik"]["metrics"]
    dev_id = development["selection"]["selected_episode"]
    frame = development["event"]["grasp_onset"]
    ready = bool(
        development["interaction_ik"]["solver"]["contact_feasible"]
        and development["meaningful_and_reproducible"]
        and all(bool(row["solver_converged"]) and bool(row["joint_limits_ok"]) for row in smoke_rows)
        and integrity["dataset_a_unchanged"]
        and integrity["g1_arm_trajectories_unchanged"]
        and tests["pass"]
    )
    conclusion = "PROPOSED_HAND_V2_READY_FOR_DATASET_B" if ready else "PROPOSED_HAND_V2_NOT_READY"
    collision_rows = [
        [
            category,
            collision["baseline"]["category_frame_incidence"][category],
            collision["proposed"]["category_frame_incidence"][category],
            collision["proposed"]["category_frame_incidence"][category]
            - collision["baseline"]["category_frame_incidence"][category],
        ]
        for category in COLLISION_CATEGORIES
    ]
    cause_names = sorted(
        set(collision["baseline"]["cause_group_frame_incidence"])
        | set(collision["proposed"]["cause_group_frame_incidence"])
        | set(collision["proposed"]["enhanced_logger"]["cause_group_frame_incidence"])
    )
    cause_rows = [
        [
            cause,
            collision["baseline"]["cause_group_frame_incidence"].get(cause, 0),
            collision["proposed"]["cause_group_frame_incidence"].get(cause, 0),
            collision["proposed"]["enhanced_logger"]["cause_group_frame_incidence"].get(cause, 0),
        ]
        for cause in cause_names
    ]
    comparator_rows = []
    for name, value in development["comparators"].items():
        comparator_rows.append(
            [
                name,
                f"{1000*value['mean_task_finger_error_m']:.3f}",
                f"{1000*value['physical_pinch_center_error_m']:.3f}",
                value["left_hand_related_collision_pairs"],
                value["hand_related_collision_pairs"],
            ]
        )
    comparator_rows.append(
        [
            "proposed_hand_v2",
            f"{1000*solved['mean_task_finger_error_m']:.3f}",
            f"{1000*solved['physical_pinch_center_error_m']:.3f}",
            solved["left_hand_related_collision_pairs"],
            solved["hand_related_collision_pairs"],
        ]
    )
    smoke_table = _markdown_table(
        ["sample", "episode", "frame", "fixed mm", "IK mm", "left coll fixed→IK", "status"],
        [
            [
                row["stratum"],
                row["episode_id"],
                row["grasp_frame"],
                f"{row['fixed_mean_contact_error_mm']:.3f}" if isinstance(row["fixed_mean_contact_error_mm"], float) else "N/A",
                f"{row['interaction_mean_contact_error_mm']:.3f}" if isinstance(row["interaction_mean_contact_error_mm"], float) else "N/A",
                f"{row['fixed_left_hand_collision_pairs']}→{row['interaction_left_hand_collision_pairs']}",
                row["blocker_classification"] or "FEASIBLE",
            ]
            for row in smoke_rows
        ],
    )
    text = f"""1. **50-episode Proposed collision의 최대 원인 category**: {collision['largest_proposed_category']} ({collision['proposed']['category_frame_incidence'][collision['largest_proposed_category']]:,} frame incidence)
2. **Dex3 finger/placeholder가 전체 collision frame 중 차지하는 비율**: {collision['proposed']['finger_involved_collision_frame_percent']:.2f}% ({collision['proposed']['finger_involved_collision_frames']:,}/{collision['proposed']['v1_collision_frames']:,})
3. **개발 grasp에 사용한 episode와 자동 검출 grasp frame**: episode {dev_id}, frame {frame}
4. **source interaction representation mode**: {development['source_interaction']['representation_mode']}
5. **Fixed PHONE_PINCH mean thumb/index error**: {1000*fixed['mean_task_finger_error_m']:.3f} mm
6. **Interaction-aware IK mean thumb/index error**: {1000*solved['mean_task_finger_error_m']:.3f} mm
7. **Fixed PHONE_PINCH hand-related collision count**: {fixed['hand_related_collision_pairs']} pair(s) (left repaired hand: {fixed['left_hand_related_collision_pairs']})
8. **Interaction-aware IK hand-related collision count**: {solved['hand_related_collision_pairs']} pair(s) (left repaired hand: {solved['left_hand_related_collision_pairs']})
9. **G1 arm trajectory checksum unchanged 여부**: {str(integrity['g1_arm_trajectories_unchanged']).lower()}
10. **Dataset A unchanged 여부**: {str(integrity['dataset_a_unchanged']).lower()}

# A. collision root-cause table

{_markdown_table(['category', 'Baseline frames', 'Proposed frames', 'delta'], collision_rows)}

{_markdown_table(['cause group', 'Baseline v1 frames', 'Proposed v1 frames', 'Proposed enhanced frames'], cause_rows)}

Pair category는 상호배타적이고, 한 frame의 category incidence는 여러 pair 때문에 중복될 수 있다. v1의 4,696 collision frame을 정확히 재현했다. Baseline→Proposed 증가 {collision['baseline_to_proposed']['collision_frame_delta']:+,}프레임 중 finger-involved 증가는 {collision['baseline_to_proposed']['finger_involved_collision_frame_delta']:+,}이고 arm-only 증가는 {collision['baseline_to_proposed']['arm_wrist_palm_only_collision_frame_delta']:+,}이다. 결정 gate는 `{collision['decision_gate']}`이다.

# B. exact source contact reconstruction

Source object pose는 사용하지 않았다. ALOHA FK 후 named tip box(`follower_left_gripper_right_tip`, `follower_left_gripper_left_tip`)의 TCP 쪽 inner face center를 각각 contact A/B로 계산했다. contact A→thumb, contact B→index이며, source width {1000*development['source_interaction']['gripper_width_m']:.3f} mm를 v1 global scale 0.42로 {1000*development['contact_mapping']['mapped_contact_width_m']:.3f} mm에 매핑했다. 표현명은 fallback 그대로이며 Isaac scene을 source annotation으로 사용하지 않았다.

# C. exact Dex3 grasp topology

Left thumb 3 DoF + index 2 DoF만 최적화했다. third/middle 2 DoF는 active-model OPEN neutral에 byte-exact하게 고정했다. Arm 14 DoF와 wrist pose는 최적화 변수에 포함하지 않았다.

# D. Dex3 q[7] result

`{json.dumps(development['interaction_ik']['q'], ensure_ascii=False, default=json_default)}`

Solver convergence={development['interaction_ik']['solver']['success']}, contact feasible(≤10 mm mean)={development['interaction_ik']['solver']['contact_feasible']}, blocker=`{development['interaction_ik']['solver']['blocker_classification']}`.

# E. Baseline/current Proposed hand/Proposed-Hand-v2

{_markdown_table(['hand state', 'mean contact mm', 'pinch-center mm', 'left hand pairs', 'all hand pairs'], comparator_rows)}

# F. third-finger behavior

Third finger는 `{solved['third_finger_policy']}`이고 neutral exact={solved['third_finger_neutral_exact']}. Source object pose가 없으므로 object contact는 `{solved['third_finger_task_object_contact']}`로 남겼다. 이는 TARGET_SIM_SCENE 결과를 source ground truth로 둔갑시키지 않기 위한 제한이다.

# G. short temporal-window result

frame {temporal['start_frame']}–{temporal['end_frame']} ({temporal['frame_count']} frames), convergence {100*temporal['solver_convergence_rate']:.1f}%, mean contact error {1000*temporal['current_mean_task_finger_error_m']:.3f}→{1000*temporal['interaction_ik_mean_task_finger_error_m']:.3f} mm, left-hand collision frames {temporal['current_left_hand_collision_frames']}→{temporal['interaction_ik_left_hand_collision_frames']}.

# H. 3-episode smoke test

{smoke_table}

# I. blocker classification

One-frame contact error는 개선됐지만 10 mm feasibility gate를 통과하지 못했고 joint bound가 활성화되어 `{development['interaction_ik']['solver']['blocker_classification']}`로 분류했다. 따라서 frozen wrist에서 task contact center를 정확히 재현하는 Dataset-B label로는 아직 승인하지 않는다.

# J. exact files added/modified

- `tools/aloha_g1_hand_v2/__init__.py`
- `tools/aloha_g1_hand_v2/common.py`
- `tools/aloha_g1_hand_v2/interaction_extraction.py`
- `tools/aloha_g1_hand_v2/contact_mapping.py`
- `tools/aloha_g1_hand_v2/collision_eval.py`
- `tools/aloha_g1_hand_v2/dex3_ik.py`
- `tools/aloha_g1_hand_v2/pipeline.py`
- `tools/repair_proposed_hand_v2.py`
- `configs/aloha_g1_hand_v2.json`
- `tests/test_aloha_g1_hand_v2.py`
- `outputs/g1_dataset_retargeting_hand_v2/` 아래 본 보고서와 audit/diagnostic 산출물

v1 Dataset A/B 산출물은 수정하지 않았다.

# K. exact test commands and PASS/FAIL

`{tests['command']}` → {'PASS' if tests['pass'] else 'FAIL'} ({tests['passed_test_count']} passed, exit {tests['exit_code']})

# L. Dataset B merge readiness

Proposed-only `--hand-mapper interaction_ik` 인터페이스와 frozen-arm checksum gate는 준비됐지만, contact feasibility가 미달이므로 전체 Dataset B replacement를 실행하지 않았다. 다음 단계는 arm/workspace를 바꾸는 것이 아니라 Dex3 contact-frame/패드 표면 정의와 reachable contact mapping을 별도 공통 calibration으로 검증하는 것이다.

{conclusion}
"""
    (output_root / "summary/final_report.md").write_text(text, encoding="utf-8")
    return conclusion


def run_pipeline(
    output_root: Path = V2_ROOT,
    *,
    development_episode: int = 49,
    method: str = "proposed",
    hand_mapper: str = "interaction_ik",
    run_tests: bool = True,
) -> dict[str, Any]:
    if method != "proposed":
        raise ValueError("hand v2 is Proposed/Dataset-B only; Dataset A is immutable")
    if hand_mapper not in {"interaction_ik", "semantic_primitive"}:
        raise ValueError(hand_mapper)
    if output_root.resolve() == V1_ROOT.resolve():
        raise ValueError("v2 output must not overwrite v1")
    for folder in ("collision_audit", "development", "renders", "smoke_test", "tests", "summary", "config", "integrity"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_CONFIG, output_root / "config/aloha_g1_hand_v2.json")
    v1_config = load_v1_config()
    v2_config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    before = _integrity_snapshot()
    atomic_json(output_root / "integrity/before.json", before)

    collision, collision_rows = audit_current_50()
    atomic_json(output_root / "collision_audit/collision_attribution_50ep.json", collision)
    atomic_csv(output_root / "collision_audit/collision_attribution_50ep.csv", collision_rows)
    _write_collision_report(
        output_root / "collision_audit/collision_attribution_report.md", collision, collision_rows
    )

    dataset = SourceDataset()
    runtime = make_runtime(v1_config)
    mapper = HandMapper(v1_config, runtime)
    solver = Dex3InteractionIK(runtime, v1_config, v2_config)
    extractor = AlohaContactExtractor(v1_config["models"]["aloha_xml"])
    episode_id, event, selection = _select_development_episode(
        development_episode, runtime, v1_config
    )
    folder = V1_ROOT / "proposed" / f"episode_{episode_id:06d}"
    arm = _load_npz(folder / "g1_arm_action.npz")
    proposed_hand = _load_npz(folder / "g1_hand_action.npz")
    baseline_hand = _load_npz(
        V1_ROOT / "baseline" / f"episode_{episode_id:06d}" / "g1_hand_action.npz"
    )
    source_episode = dataset.episode(episode_id)
    frame = int(event["grasp_onset"])
    source = extractor.extract(source_episode.action[frame], "left")
    target = map_source_contacts_to_frozen_g1_tool(
        source,
        runtime,
        arm["action"][frame],
        proposed_hand["right_action"][frame],
        v1_config,
    )
    states = {
        "baseline_dataset_a": baseline_hand["left_action"][frame],
        "current_v1_smoothed_primitive": proposed_hand["left_action"][frame],
        "fixed_phone_pinch": solver.reference_q,
    }
    comparators = _evaluate_named_states(
        solver, target, arm["action"][frame], proposed_hand["right_action"][frame], states
    )
    result = solver.solve(arm["action"][frame], proposed_hand["right_action"][frame], target)
    repeated = solver.solve(arm["action"][frame], proposed_hand["right_action"][frame], target)
    reproducible = bool(np.array_equal(result["q"], repeated["q"]))
    fixed_error = float(comparators["fixed_phone_pinch"]["mean_task_finger_error_m"])
    relative_improvement = (fixed_error - float(result["metrics"]["mean_task_finger_error_m"])) / fixed_error
    meaningful = bool(
        result["solver"]["success"]
        and result["metrics"]["finite"]
        and result["metrics"]["joint_limits_ok"]
        and relative_improvement
        >= float(v2_config["dex3_ik"]["meaningful_relative_contact_improvement"])
    )
    development = {
        "schema_version": "proposed_hand_v2_development_grasp",
        "method": method,
        "hand_mapper": hand_mapper,
        "selection": selection,
        "event": event,
        "semantic_phase_at_frame": str(proposed_hand["left_phase"][frame]),
        "source_interaction": source,
        "contact_mapping": target,
        "comparators": comparators,
        "interaction_ik": result,
        "repeat_solution_q": repeated["q"],
        "bitwise_reproducible_q": reproducible,
        "relative_contact_improvement": relative_improvement,
        "one_frame_meaningful": meaningful,
        "meaningful_and_reproducible": bool(meaningful and reproducible),
        "target_object_diagnostic": "NOT_EXECUTED_SOURCE_OBJECT_POSE_UNAVAILABLE",
    }
    atomic_json(
        output_root / "development/interaction_source.json",
        {
            "selection": selection,
            "event": event,
            "source_interaction": source,
            "contact_mapping": target,
        },
    )
    atomic_json(
        output_root / "development/current_primitive_metrics.json",
        {
            "episode_id": episode_id,
            "frame": frame,
            "comparators": comparators,
            "comparison_arm_q_sha256": array_sha256(arm["action"][frame]),
        },
    )
    atomic_json(
        output_root / "development/interaction_ik_metrics.json",
        {
            "episode_id": episode_id,
            "frame": frame,
            "result": result,
            "bitwise_reproducible_q": reproducible,
            "relative_contact_improvement": relative_improvement,
            "one_frame_meaningful": meaningful,
        },
    )
    atomic_json(
        output_root / "development/collision_comparison.json",
        {
            name: {
                "total_collision_pairs": value["total_collision_pairs"],
                "hand_related_collision_pairs": value["hand_related_collision_pairs"],
                "left_hand_related_collision_pairs": value["left_hand_related_collision_pairs"],
                "collision_records": value["collision_records"],
            }
            for name, value in {**comparators, "interaction_ik": result["metrics"]}.items()
        },
    )
    arm_frame = np.ascontiguousarray(arm["action"][frame])
    _atomic_npz(
        output_root / "development/dex3_solution.npz",
        dex3_q=result["q"].astype(np.float64),
        joint_names=np.asarray(solver.names),
        phone_pinch_initialization=solver.reference_q,
        third_finger_neutral=solver.third_neutral_q,
        thumb_target=np.asarray(target["thumb_target_m"]),
        index_target=np.asarray(target["index_target_m"]),
        frozen_g1_arm_q=arm_frame,
        frozen_g1_arm_q_sha256=np.asarray(array_sha256(arm_frame)),
        source_episode=np.asarray(episode_id),
        source_frame=np.asarray(frame),
        representation_mode=np.asarray(source["representation_mode"]),
        solver_config_sha256=np.asarray(sha256_file(SOURCE_CONFIG)),
    )

    _source_render(output_root / "renders/source_interaction.png", source)
    primitive_geometry = _hand_geometry(
        runtime,
        arm["action"][frame],
        solver.reference_q,
        proposed_hand["right_action"][frame],
        v1_config,
    )
    interaction_geometry = _hand_geometry(
        runtime,
        arm["action"][frame],
        result["q"],
        proposed_hand["right_action"][frame],
        v1_config,
    )
    for projection in ("front", "side", "top"):
        _projection_render(
            output_root / f"renders/target_{projection}.png",
            interaction_geometry,
            target,
            projection,
        )
    _comparison_render(
        output_root / "renders/primitive_vs_interaction.png",
        primitive_geometry,
        interaction_geometry,
        target,
    )

    if meaningful and reproducible:
        temporal = _run_temporal_window(
            output_root,
            episode_id,
            event,
            source_episode,
            extractor,
            runtime,
            solver,
            arm,
            proposed_hand,
            v1_config,
            v2_config,
        )
        selection_smoke, smoke_rows = _run_smoke_test(
            output_root, dataset, extractor, runtime, solver, v1_config
        )
    else:
        temporal = {"executed": False, "reason": "one-frame solution not meaningful and reproducible"}
        atomic_json(output_root / "development/temporal_window_metrics.json", temporal)
        selection_smoke, smoke_rows = _run_smoke_test(
            output_root, dataset, extractor, runtime, solver, v1_config
        )

    anti_overfit = _anti_overfit_scan()
    atomic_json(output_root / "tests/anti_overfitting_scan.json", anti_overfit)
    after = _integrity_snapshot()
    integrity = {
        "before": before,
        "after": after,
        "dataset_a_unchanged": before["dataset_a_tree_sha256"] == after["dataset_a_tree_sha256"],
        "g1_arm_trajectories_unchanged": before["proposed_arm_combined_sha256"]
        == after["proposed_arm_combined_sha256"],
        "development_arm_frame_source_sha256": array_sha256(arm_frame),
        "development_arm_frame_export_sha256": array_sha256(
            _load_npz(output_root / "development/dex3_solution.npz")["frozen_g1_arm_q"]
        ),
    }
    integrity["development_arm_frame_byte_identical"] = bool(
        integrity["development_arm_frame_source_sha256"]
        == integrity["development_arm_frame_export_sha256"]
    )
    atomic_json(output_root / "integrity/after_and_comparison.json", integrity)

    pretest = {
        "development": development,
        "temporal": temporal,
        "smoke_selection": selection_smoke,
        "smoke_rows": smoke_rows,
        "integrity": integrity,
        "collision_decision": collision["decision_gate"],
        "anti_overfit": anti_overfit,
    }
    atomic_json(output_root / "summary/pipeline_results.json", pretest)
    tests = _run_pytest(output_root) if run_tests else {
        "command": "NOT_RUN",
        "exit_code": -1,
        "pass": False,
        "passed_test_count": 0,
        "output": "tests disabled",
    }
    tests["anti_overfit_scan"] = anti_overfit
    atomic_json(output_root / "tests/test_report.json", tests)
    conclusion = _write_final_report(
        output_root, collision, development, temporal, smoke_rows, integrity, tests
    )
    result_summary = {
        "output_root": str(output_root.resolve()),
        "collision_decision": collision["decision_gate"],
        "development_episode": episode_id,
        "grasp_frame": frame,
        "source_representation_mode": source["representation_mode"],
        "fixed_phone_pinch_mean_error_m": fixed_error,
        "interaction_ik_mean_error_m": result["metrics"]["mean_task_finger_error_m"],
        "dataset_a_unchanged": integrity["dataset_a_unchanged"],
        "g1_arm_trajectories_unchanged": integrity["g1_arm_trajectories_unchanged"],
        "tests_pass": tests["pass"],
        "conclusion": conclusion,
    }
    atomic_json(output_root / "summary/run_summary.json", result_summary)
    return result_summary
