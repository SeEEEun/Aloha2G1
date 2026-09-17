#!/usr/bin/env python3
"""Verify representation-only parity on the fixed non-test smoke episodes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4"
OUT = ROOT / "outputs/single_variable_ab_reset"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def main() -> int:
    scene = json.loads(
        (ROOT / "isaaclab_doll_handoff_scene/scene_layout.json").read_text()
    )
    baseline = json.loads((SMOKE / "config/baseline_config.json").read_text())
    common = json.loads((SMOKE / "config/common_config.json").read_text())
    root_position = np.asarray(scene["g1"]["root_position_world_xyz_m"], dtype=np.float64)
    root_quat_wxyz = np.asarray(
        scene["g1"]["root_orientation_world_wxyz"], dtype=np.float64
    )
    root_rotation = Rotation.from_quat(root_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    model_pelvis = np.asarray(common["resolved"]["g1_model_pelvis_xyz_m"], dtype=np.float64)
    transforms = baseline["resolved"]["fixed_target_tool_to_g1_wrist"]

    a_paths = sorted((SMOKE / "baseline/trajectories").glob("*.npz"))
    b_paths = sorted((SMOKE / "proposed/trajectories").glob("*.npz"))
    b_by_name = {path.name: path for path in b_paths}
    rows: list[dict[str, Any]] = []
    for a_path in a_paths:
        b_path = b_by_name[a_path.name]
        with np.load(a_path, allow_pickle=False) as a, np.load(
            b_path, allow_pickle=False
        ) as b:
            row: dict[str, Any] = {
                "source_episode_id": scalar_text(a["source_episode_id"]),
                "a_path": str(a_path),
                "b_path": str(b_path),
                "same_timestamp": bool(np.array_equal(a["timestamp"], b["timestamp"])),
                "same_source_episode_id": scalar_text(a["source_episode_id"])
                == scalar_text(b["source_episode_id"]),
                "same_frame_count": len(a["timestamp"]) == len(b["timestamp"]),
                "same_source_frame_index": bool(
                    np.array_equal(a["source_frame_index"], b["source_frame_index"])
                ),
                "same_event_frames": bool(
                    np.array_equal(a["event_frames"], b["event_frames"])
                ),
                "same_left_phase": bool(
                    np.array_equal(a["left_hand_phase"], b["left_hand_phase"])
                ),
                "same_right_phase": bool(
                    np.array_equal(a["right_hand_phase"], b["right_hand_phase"])
                ),
                "same_left_dex3": bool(
                    np.array_equal(a["left_dex3_qpos"], b["left_dex3_qpos"])
                ),
                "same_right_dex3": bool(
                    np.array_equal(a["right_dex3_qpos"], b["right_dex3_qpos"])
                ),
                "same_arm_joint_order": bool(
                    np.array_equal(a["g1_arm_joint_names"], b["g1_arm_joint_names"])
                ),
                "same_dex3_joint_order": bool(
                    np.array_equal(a["left_dex3_joint_names"], b["left_dex3_joint_names"])
                    and np.array_equal(a["right_dex3_joint_names"], b["right_dex3_joint_names"])
                ),
                "same_common_config_sha256": scalar_text(a["common_config_sha256"])
                == scalar_text(b["common_config_sha256"]),
                "same_implementation_sha256": scalar_text(a["implementation_sha256"])
                == scalar_text(b["implementation_sha256"]),
                "arm_trajectory_differs_as_intended": not np.array_equal(
                    a["g1_arm_qpos"], b["g1_arm_qpos"]
                ),
                "archive_common_hash_matches_file": scalar_text(
                    a["common_config_sha256"]
                ) == sha256(SMOKE / "config/common_config.json")
                and scalar_text(b["common_config_sha256"])
                == sha256(SMOKE / "config/common_config.json"),
            }
            a_reference_errors: list[float] = []
            b_reference_errors: list[float] = []
            for side in ("left", "right"):
                wrist_to_tool = np.asarray(transforms[side], dtype=np.float64)
                for archive, errors in (
                    (a, a_reference_errors),
                    (b, b_reference_errors),
                ):
                    wrist_position_model = np.asarray(
                        archive[f"target_{side}_wrist_position_model"],
                        dtype=np.float64,
                    )
                    wrist_rotation_model = np.asarray(
                        archive[f"target_{side}_wrist_rotation_model"],
                        dtype=np.float64,
                    )
                    reconstructed_model = wrist_position_model + np.einsum(
                        "tij,j->ti", wrist_rotation_model, wrist_to_tool[:3, 3]
                    )
                    reconstructed_world = (
                        (reconstructed_model - model_pelvis) @ root_rotation.T
                        + root_position
                    )
                    reference = np.asarray(
                        archive[f"target_{side}_interaction_frame_position_world"],
                        dtype=np.float64,
                    )
                    errors.extend(
                        np.linalg.norm(reconstructed_world - reference, axis=1).tolist()
                    )
            row["a_fixed_tcp_tool_reconstruction_max_error_mm"] = float(
                1000.0 * max(a_reference_errors)
            )
            row["b_interaction_tool_reconstruction_max_error_mm"] = float(
                1000.0 * max(b_reference_errors)
            )
            rows.append(row)

    parity_keys = [
        "same_timestamp",
        "same_source_episode_id",
        "same_frame_count",
        "same_source_frame_index",
        "same_event_frames",
        "same_left_phase",
        "same_right_phase",
        "same_left_dex3",
        "same_right_dex3",
        "same_arm_joint_order",
        "same_dex3_joint_order",
        "same_common_config_sha256",
        "same_implementation_sha256",
        "archive_common_hash_matches_file",
    ]
    parity = all(bool(row[key]) for row in rows for key in parity_keys)
    reconstruction_max = max(
        row["a_fixed_tcp_tool_reconstruction_max_error_mm"] for row in rows
    )
    b_reconstruction_max = max(
        row["b_interaction_tool_reconstruction_max_error_mm"] for row in rows
    )
    status = {
        "schema_version": "single_variable_train_smoke_verification_v1",
        "status": (
            "BLOCKED_COMMON_SOLVER"
            if parity and reconstruction_max < 1e-3 and b_reconstruction_max < 1e-3
            else "FAIL"
        ),
        "smoke_episode_count": len(rows),
        "smoke_source": "TRAIN_ONLY_INDICES_0_24_49",
        "physical_rollouts": 0,
        "representation_modes": {"A": "WRIST", "B": "INTERACTION"},
        "common_parity_pass": parity,
        "a_reference_fixed_tool_transform_pass": reconstruction_max < 1e-3,
        "a_reference_reconstruction_max_error_mm": reconstruction_max,
        "a_reference_competent_at_target_generation_level": reconstruction_max < 1e-3,
        "b_reference_competent_at_target_generation_level": b_reconstruction_max < 1e-3,
        "b_reference_reconstruction_max_error_mm": b_reconstruction_max,
        "common_solver_smoke_pass": False,
        "common_solver_smoke_evidence": {
            "A": {"pass": 0, "total": 3, "statuses": ["FAIL_IK"] * 3},
            "B": {"pass": 0, "total": 3, "statuses": ["FAIL_IK"] * 3},
        },
        "rows": rows,
        "inputs": {
            "common_config": str(SMOKE / "config/common_config.json"),
            "common_config_sha256": sha256(SMOKE / "config/common_config.json"),
            "fairness_report": str(SMOKE / "config/fairness_report.json"),
            "fairness_report_sha256": sha256(SMOKE / "config/fairness_report.json"),
        },
    }
    atomic_json(OUT / "SINGLE_VARIABLE_TRAIN_SMOKE_VERIFICATION.json", status)

    lines = [
        "# Single-variable TRAIN smoke verification",
        "",
        f"Status: **{status['status']}**",
        "",
        "No physics and no EVAL35 outcomes were used.",
        "",
        f"- TRAIN smoke episodes: **{len(rows)}/3** (indices 0, 24, 49)",
        f"- Common timeline/Dex3/schema/hash parity: **{'PASS' if parity else 'FAIL'}**",
        f"- A fixed task-TCP→G1-wrist reconstruction: **{'PASS' if reconstruction_max < 1e-3 else 'FAIL'}**",
        f"- Maximum A reconstruction error: **{reconstruction_max:.9f} mm**",
        f"- B interaction-frame reconstruction: **{'PASS' if b_reconstruction_max < 1e-3 else 'FAIL'}**",
        f"- Maximum B reconstruction error: **{b_reconstruction_max:.9f} mm**",
        "- A common-solver kinematic gate: **0/3 PASS**",
        "- B common-solver kinematic gate: **0/3 PASS**",
        "",
        "The reset A reference no longer contains the old coordinate/workspace mapping bug: its fixed wrist-to-tool transform reconstructs the registered task TCP to numerical precision. Both branches nevertheless fail the same frozen common IK acceptance gate on this smoke set. Dataset generation, training, freeze, and physical evaluation are therefore blocked pending a common-solver audit; no method-specific solver change is permitted.",
        "",
        "| episode | timeline | Dex3 | implementation | A TCP reconstruction (mm) | B interaction reconstruction (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source_episode_id']} | {'PASS' if row['same_event_frames'] else 'FAIL'} | "
            f"{'PASS' if row['same_left_dex3'] and row['same_right_dex3'] else 'FAIL'} | "
            f"{'PASS' if row['same_implementation_sha256'] else 'FAIL'} | "
            f"{row['a_fixed_tcp_tool_reconstruction_max_error_mm']:.9f} | "
            f"{row['b_interaction_tool_reconstruction_max_error_mm']:.9f} |"
        )
    (OUT / "SINGLE_VARIABLE_TRAIN_SMOKE_VERIFICATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: status[k] for k in (
        "status",
        "common_parity_pass",
        "a_reference_fixed_tool_transform_pass",
        "a_reference_reconstruction_max_error_mm",
        "b_reference_competent_at_target_generation_level",
        "b_reference_reconstruction_max_error_mm",
        "common_solver_smoke_pass",
    )}, indent=2))
    return 0 if status["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
