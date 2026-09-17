#!/usr/bin/env python3
"""Symmetric TRAIN-only common XYZ+yaw search with the actual sequential IK.

The search changes one rigid task placement for all six smoke trajectories.  It
never changes an individual wrist target and the solver receives no method ID.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.doll_handoff_retargeting.retarget import SharedTemporalIK


OUT = ROOT / "outputs/single_variable_ab_common_execution"
SMOKE = OUT / "04_workspace_registration/smoke"
REGISTRATION = OUT / "04_workspace_registration/COMMON_TASK_REGISTRATION_TRAIN_SMOKE.json"
SEARCH = OUT / "workspace_solver_search"
CSV_PATH = OUT / "COMMON_WORKSPACE_SOLVER_SEARCH.csv"
EPISODES = (0, 24, 49)
METHODS = ("baseline", "proposed")
SIDES = ("left", "right")


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    fields = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_path(method: str, episode: int) -> Path:
    paths = list((SMOKE / method / "trajectories").glob(f"*ep{episode:03d}.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"archive count for {method} episode {episode}: {len(paths)}")
    return paths[0]


def candidates(registration: dict[str, Any]) -> list[dict[str, Any]]:
    current = np.asarray(
        registration["common_workspace_registration"]["translation_xyz_m"], dtype=float
    )
    source_points = np.stack(
        [np.asarray(row["source_object_pose"]["position_xyz_m"], dtype=float) for row in registration["entries"]]
        + [np.asarray(registration["entries"][0]["source_bin_pose"]["position_xyz_m"], dtype=float)]
    )
    pivot = np.mean(source_points, axis=0)

    rows: list[tuple[str, np.ndarray, float]] = [
        ("current_minimum_norm", current, 0.0),
        ("translate_x_minus_40mm", current + [-0.040, 0.0, 0.0], 0.0),
        ("translate_x_plus_40mm", current + [0.040, 0.0, 0.0], 0.0),
        ("translate_y_minus_40mm", current + [0.0, -0.040, 0.0], 0.0),
        ("translate_y_plus_40mm", current + [0.0, 0.040, 0.0], 0.0),
        ("translate_z_minus_40mm", current + [0.0, 0.0, -0.040], 0.0),
        ("translate_z_plus_40mm", current + [0.0, 0.0, 0.040], 0.0),
        ("prior_y080_z120", np.asarray([0.0, -0.080, -0.120]), 0.0),
        ("prior_yaw20_solver_probe", np.asarray([0.10519411050561855, -0.1922185722417424, -0.120]), 20.0),
    ]
    for yaw in (-20.0, 20.0):
        rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        translation = pivot + current - rotation @ pivot
        rows.append((f"pivot_preserving_yaw_{yaw:+.0f}", translation, yaw))
    # A small deterministic refinement around the best coarse feasibility
    # region.  These offsets are selected only from sequential IK acceptance,
    # never from grasp or task outcomes.
    for yaw in (15.0, 25.0, 30.0):
        rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        translation = pivot + current - rotation @ pivot
        rows.append((f"refine_pivot_yaw_{yaw:+.0f}", translation, yaw))
    rotation20 = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()
    translation20 = pivot + current - rotation20 @ pivot
    rows.extend(
        (
            ("refine_yaw20_y_plus_10mm", translation20 + [0.0, 0.010, 0.0], 20.0),
            ("refine_yaw20_y_plus_20mm", translation20 + [0.0, 0.020, 0.0], 20.0),
            ("refine_yaw20_x_minus_10mm", translation20 + [-0.010, 0.0, 0.0], 20.0),
        )
    )
    output: list[dict[str, Any]] = []
    for name, translation, yaw in rows:
        rotation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        output.append(
            {
                "name": name,
                "translation_xyz_m": np.asarray(translation, dtype=float),
                "yaw_deg": float(yaw),
                "matrix": matrix,
                "translation_norm_m": float(np.linalg.norm(translation)),
                "search_reason": "bounded common level-preserving XYZ/yaw TRAIN-smoke candidate",
            }
        )
    return output


def transformed_targets(
    values: dict[str, np.ndarray],
    g1: G1Kinematics,
    old_matrix: np.ndarray,
    new_matrix: np.ndarray,
) -> dict[str, Any]:
    target: dict[str, Any] = {
        "task_orientation_constrained": False,
        "orientation_gauge_weight_multiplier": 0.0,
        "direct_static_grasp_frame_position_objective": False,
        "explicit_bimanual_objective": False,
    }
    old_rotation = old_matrix[:3, :3]
    old_translation = old_matrix[:3, 3]
    new_rotation = new_matrix[:3, :3]
    new_translation = new_matrix[:3, 3]
    for side in SIDES:
        current_model = values[f"target_{side}_wrist_position_model"].astype(float)
        current_world = g1.model_to_world_position(current_model)
        pre_world = (current_world - old_translation) @ old_rotation
        moved_world = pre_world @ new_rotation.T + new_translation
        target[f"{side}_wrist_position"] = g1.world_to_model_position(moved_world)

        current_rotation_model = values[f"target_{side}_wrist_rotation_model"].astype(float)
        current_rotation_world = g1.model_to_world_rotation(current_rotation_model)
        pre_rotation_world = np.einsum("ij,tjk->tik", old_rotation.T, current_rotation_world)
        moved_rotation_world = np.einsum("ij,tjk->tik", new_rotation, pre_rotation_world)
        target[f"{side}_wrist_rotation"] = g1.world_to_model_rotation(moved_rotation_world)
    return target


def result_row(
    candidate: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    required_rate: float,
) -> dict[str, Any]:
    rates = [float(row["frame_acceptance_rate"]) for row in trajectory_rows]
    counts = {
        method: sum(
            float(row["frame_acceptance_rate"]) >= required_rate
            for row in trajectory_rows
            if row["method"] == method
        )
        for method in METHODS
    }
    total_accepted = sum(int(row["accepted_frames"]) for row in trajectory_rows)
    total_frames = sum(int(row["frame_count"]) for row in trajectory_rows)
    maximum_residual = max(float(row["maximum_position_residual_m"]) for row in trajectory_rows)
    return {
        "candidate": candidate["name"],
        "x_m": float(candidate["translation_xyz_m"][0]),
        "y_m": float(candidate["translation_xyz_m"][1]),
        "z_m": float(candidate["translation_xyz_m"][2]),
        "yaw_deg": float(candidate["yaw_deg"]),
        "minimum_trajectory_frame_acceptance": min(rates),
        "minimum_method_episode_acceptance_count": min(counts.values()),
        "A_episode_pass_count": counts["baseline"],
        "B_episode_pass_count": counts["proposed"],
        "total_accepted_frames": total_accepted,
        "total_frames": total_frames,
        "total_frame_acceptance_rate": total_accepted / total_frames,
        "maximum_cartesian_residual_m": maximum_residual,
        "transform_translation_norm_m": float(candidate["translation_norm_m"]),
        "transform_yaw_magnitude_deg": abs(float(candidate["yaw_deg"])),
        "all_six_pass": bool(min(rates) >= required_rate),
        "candidate_detail_json": str((SEARCH / f"{candidate['name']}.json").resolve()),
    }


def ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["minimum_trajectory_frame_acceptance"]),
        -int(row["minimum_method_episode_acceptance_count"]),
        -int(row["total_accepted_frames"]),
        float(row["maximum_cartesian_residual_m"]),
        float(row["transform_translation_norm_m"]),
        float(row["transform_yaw_magnitude_deg"]),
        str(row["candidate"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[], help="candidate name; repeatable")
    args = parser.parse_args()
    SEARCH.mkdir(parents=True, exist_ok=True)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    old_matrix = np.asarray(
        registration["entries"][0]["common_workspace_transform_matrix"], dtype=float
    )
    common = load_common_config(SMOKE / "config/common_config.json")
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal = np.asarray(common["resolved"]["canonical_g1_nominal_q"], dtype=float)
    solver = SharedTemporalIK(common, g1, nominal)
    required_rate = float(common["shared_temporal_ik"]["required_success_rate"])
    archives: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    input_hashes: dict[str, str] = {}
    for method in METHODS:
        for episode in EPISODES:
            path = archive_path(method, episode)
            with np.load(path, allow_pickle=False) as source:
                archives[(method, episode)] = {key: source[key] for key in source.files}
            input_hashes[str(path.resolve())] = sha256(path)

    all_candidates = candidates(registration)
    if args.only:
        allowed = set(args.only)
        all_candidates = [row for row in all_candidates if row["name"] in allowed]
        missing = allowed - {row["name"] for row in all_candidates}
        if missing:
            raise RuntimeError(f"unknown candidates: {sorted(missing)}")

    aggregate: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(all_candidates, start=1):
        cache_path = SEARCH / f"{candidate['name']}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("input_hashes") == input_hashes
                and cached.get("common_config_sha256") == sha256(SMOKE / "config/common_config.json")
                and np.allclose(cached.get("transform_matrix"), candidate["matrix"], atol=0.0, rtol=0.0)
            ):
                print(f"[{candidate_index}/{len(all_candidates)}] {candidate['name']}: cached", flush=True)
                aggregate.append(cached["aggregate"])
                continue
        trajectories: list[dict[str, Any]] = []
        q_payload: dict[str, np.ndarray] = {}
        for method in METHODS:
            for episode in EPISODES:
                target = transformed_targets(
                    archives[(method, episode)], g1, old_matrix, candidate["matrix"]
                )
                solved = solver.solve(target)
                meta = solved["reprojection_meta"]
                accepted = np.asarray([bool(row["accepted"]) for row in meta])
                residual = np.asarray([float(row["position_error_max_m"]) for row in meta])
                key = f"{method}_ep{episode:03d}"
                q_payload[key] = np.asarray(solved["q"], dtype=np.float32)
                trajectories.append(
                    {
                        "method": method,
                        "episode_index": episode,
                        "frame_count": len(meta),
                        "accepted_frames": int(np.count_nonzero(accepted)),
                        "frame_acceptance_rate": float(np.mean(accepted)),
                        "episode_pass": bool(float(np.mean(accepted)) >= required_rate),
                        "mean_position_residual_m": float(np.mean(residual)),
                        "p95_position_residual_m": float(np.quantile(residual, 0.95)),
                        "maximum_position_residual_m": float(np.max(residual)),
                        "elapsed_sec": float(solved["elapsed_sec"]),
                        "finite_q": bool(np.isfinite(solved["q"]).all()),
                        "joint_limit_violation_count": int(
                            np.count_nonzero(
                                (solved["q"] < g1.arm_limits[:, 0] - 1e-9)
                                | (solved["q"] > g1.arm_limits[:, 1] + 1e-9)
                            )
                        ),
                    }
                )
                print(
                    f"[{candidate_index}/{len(all_candidates)}] {candidate['name']} "
                    f"{method} ep{episode:02d}: {np.mean(accepted):.6f}",
                    flush=True,
                )
        aggregate_row = result_row(candidate, trajectories, required_rate)
        payload = {
            "schema_version": "common_workspace_actual_sequential_position_ik_candidate_v1",
            "scope": "TRAIN_ONLY_SMOKE_0_24_49",
            "position_only": True,
            "common_for_A_B": True,
            "level_preserving": True,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "candidate": candidate,
            "transform_matrix": candidate["matrix"],
            "required_frame_acceptance_rate": required_rate,
            "trajectories": trajectories,
            "aggregate": aggregate_row,
            "input_hashes": input_hashes,
            "common_config_sha256": sha256(SMOKE / "config/common_config.json"),
            "solver_implementation_sha256": sha256(ROOT / "tools/doll_handoff_retargeting/retarget.py"),
        }
        atomic_json(cache_path, payload)
        np.savez_compressed(SEARCH / f"{candidate['name']}_q.npz", **q_payload)
        aggregate.append(aggregate_row)
        atomic_csv(CSV_PATH, sorted(aggregate, key=ranking_key))

    aggregate = sorted(aggregate, key=ranking_key)
    atomic_csv(CSV_PATH, aggregate)
    summary = {
        "schema_version": "common_workspace_actual_sequential_position_ik_search_v1",
        "status": "PASS" if any(row["all_six_pass"] for row in aggregate) else "COMMON_STATIC_REGISTRATION_INSUFFICIENT",
        "scope": "TRAIN_ONLY_SMOKE_0_24_49",
        "candidate_count": len(aggregate),
        "objective": [
            "maximize minimum trajectory frame acceptance across six trajectories",
            "maximize minimum episode-pass count across A and B",
            "maximize total accepted frames",
            "minimize maximum Cartesian residual",
            "minimize common transform magnitude",
        ],
        "actual_sequential_position_only_solver": True,
        "A_B_equal_weight": True,
        "best": aggregate[0],
        "all_six_pass_candidates": [row["candidate"] for row in aggregate if row["all_six_pass"]],
        "candidate_table": str(CSV_PATH.resolve()),
    }
    atomic_json(OUT / "COMMON_WORKSPACE_SOLVER_SEARCH.json", summary)
    print(json.dumps(native(summary), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
