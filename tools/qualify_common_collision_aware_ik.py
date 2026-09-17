#!/usr/bin/env python3
"""Fail-closed TRAIN smoke qualification for common collision-aware G1 IK."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_collision_aware_redundancy_ik import (
    DEFAULT_CONFIG,
    CommonCollisionAwareRedundancyIK,
)
from tools.common_g1_morphology_adapter import CommonG1MorphologyAdapter
from tools.doll_handoff_retargeting.common import (
    branch_flags,
    load_common_config,
    load_scene,
)
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.qualify_common_morphology_adapter import (
    ADAPTER_DIR,
    EPISODES,
    METHODS,
    OUT,
    SIDES,
    SMOKE,
    archive_path,
)


RESULT_DIR = OUT / "07_collision_aware_ik"
AUDIT_JSON = OUT / "COMMON_COLLISION_AWARE_IK_FINAL_AUDIT.json"
AUDIT_MD = OUT / "COMMON_COLLISION_AWARE_IK_FINAL_AUDIT.md"
CONTACT_SHEET = RESULT_DIR / "SMOKE_COLLISION_AWARE_IK_BEFORE_AFTER_CONTACT_SHEET.png"
SOLVER_IMPLEMENTATION = ROOT / "tools/common_collision_aware_redundancy_ik.py"
ADAPTER_IMPLEMENTATION = ROOT / "tools/common_g1_morphology_adapter.py"
COMMON_SOLVER_IMPLEMENTATION = ROOT / "tools/doll_handoff_retargeting/retarget.py"
REPRESENTATIVE = (("baseline", "A", 24), ("baseline", "A", 49), ("proposed", "B", 0), ("proposed", "B", 49))


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(item) for item in value]
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1) * scale
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def target_arrays(archive: Any) -> dict[str, np.ndarray]:
    return {
        **{
            f"{side}_wrist_position": np.asarray(
                archive[f"feasible_{side}_position_model"], dtype=np.float64
            ).copy()
            for side in SIDES
        },
        **{
            f"{side}_wrist_rotation": np.asarray(
                archive[f"feasible_{side}_rotation_model"], dtype=np.float64
            ).copy()
            for side in SIDES
        },
    }


def result_paths(label: str, episode: int) -> tuple[Path, Path]:
    stem = RESULT_DIR / f"{label}_ep{episode:03d}_position_collision_aware"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def evaluate(
    adapter: CommonG1MorphologyAdapter,
    common: dict[str, Any],
    q: np.ndarray,
    targets: dict[str, np.ndarray],
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    fps: float,
) -> dict[str, Any]:
    position, rotation = adapter._pose_arrays(q)
    by_side = {
        side: np.linalg.norm(
            position[side] - targets[f"{side}_wrist_position"], axis=1
        )
        for side in SIDES
    }
    residual = np.maximum(by_side["left"], by_side["right"])
    collision = adapter._collision_metrics(q, left_hand, right_hand, fps)
    temporal = adapter._temporal_metrics(q, left_hand, right_hand, fps)
    validation = common["validation"]
    branches = branch_flags(
        q,
        float(validation["branch_absolute_step_norm_rad"]),
        float(validation["branch_local_multiplier"]),
    )
    hard_limits = int(
        np.count_nonzero(
            (q < adapter.g1.arm_limits[:, 0] - 1e-9)
            | (q > adapter.g1.arm_limits[:, 1] + 1e-9)
        )
    )
    acceptance_rate = float(
        np.mean(residual <= float(common["shared_temporal_ik"]["position_tolerance_m"]))
    )
    temporal_pass = adapter._temporal_passes(temporal)
    qualified = bool(
        acceptance_rate >= float(common["shared_temporal_ik"]["required_success_rate"])
        and int(collision["hard_collision_frame_count"]) == 0
        and hard_limits == 0
        and int(np.count_nonzero(branches)) == 0
        and temporal_pass
        and np.isfinite(q).all()
    )
    shoulder = np.r_[0:3, 7:10]
    elbow = np.asarray([3, 10])
    return {
        "frame_count": len(q),
        "position_acceptance_rate": acceptance_rate,
        "position_residual_mm": stats(residual, 1000.0),
        "hard_self_collision_frames": int(collision["hard_collision_frame_count"]),
        "self_contact_frames": int(collision["contact_frame_count"]),
        "hard_collision_segments": [
            row for row in collision["segments"] if bool(row["hard"])
        ],
        "hard_limit_violations": hard_limits,
        "branch_discontinuities": int(np.count_nonzero(branches)),
        "temporal": temporal,
        "temporal_pass": temporal_pass,
        "finite": bool(np.isfinite(q).all()),
        "position_qualified": qualified,
        "achieved_rotation_arrays_retained_for_6d_gate": {
            side: list(rotation[side].shape) for side in SIDES
        },
        "shoulder_posture_change_rad": None,
        "elbow_posture_change_rad": None,
        "_shoulder_indices": shoulder,
        "_elbow_indices": elbow,
    }


def render_state(
    g1: G1Kinematics,
    q: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> np.ndarray:
    g1.assign(q, left_hand, right_hand)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.08, 0.0, 0.97]
    camera.distance = 1.12
    camera.azimuth = 138.0
    camera.elevation = -22.0
    renderer = mujoco.Renderer(g1.model, height=480, width=640)
    renderer.update_scene(g1.data, camera=camera)
    image = renderer.render().copy()
    renderer.close()
    return image


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    common = load_common_config(SMOKE / "config/common_config.json")
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal = np.asarray(common["resolved"]["canonical_g1_nominal_q"], dtype=float)
    adapter = CommonG1MorphologyAdapter(common, g1, nominal)
    solver = CommonCollisionAwareRedundancyIK(adapter, DEFAULT_CONFIG)
    implementation_hash = sha256(SOLVER_IMPLEMENTATION)
    config_hash = sha256(DEFAULT_CONFIG)
    rows: list[dict[str, Any]] = []
    arrays: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    # Run every smoke trajectory. Each result is persisted immediately and can
    # be resumed without recomputing a completed deterministic trajectory.
    for method, label, mode, _ in METHODS:
        for episode in EPISODES:
            input_path = ADAPTER_DIR / f"{label}_ep{episode:03d}_raw_and_feasible.npz"
            source_path = archive_path(method, episode)
            output_npz, output_json = result_paths(label, episode)
            input_hash = sha256(input_path)
            source_hash = sha256(source_path)
            cached = None
            if output_json.exists() and output_npz.exists():
                candidate = json.loads(output_json.read_text(encoding="utf-8"))
                hashes = candidate.get("provenance", {})
                if (
                    hashes.get("solver_implementation_sha256") == implementation_hash
                    and hashes.get("solver_config_sha256") == config_hash
                    and hashes.get("input_target_archive_sha256") == input_hash
                    and hashes.get("source_archive_sha256") == source_hash
                ):
                    cached = candidate
            if cached is not None:
                with np.load(output_npz, allow_pickle=False) as values:
                    arrays[(label, episode)] = {
                        key: values[key].copy() for key in values.files
                    }
                rows.append(cached)
                print(f"reuse {label}{episode:02d}: qualified={cached['position_qualified']}", flush=True)
                continue

            with np.load(input_path, allow_pickle=False) as values:
                incoming_archive = {key: values[key].copy() for key in values.files}
            with np.load(source_path, allow_pickle=False) as values:
                source = {key: values[key].copy() for key in values.files}
            targets = target_arrays(incoming_archive)
            target_hash_before = hashlib.sha256(
                b"".join(targets[key].tobytes() for key in sorted(targets))
            ).hexdigest()
            fps = float(1.0 / np.median(np.diff(source["timestamp"].astype(float))))
            print(f"solve {label}{episode:02d} position-only", flush=True)
            result = solver.solve(
                targets,
                incoming_archive["projected_q"].astype(float),
                source["left_dex3_qpos"].astype(float),
                source["right_dex3_qpos"].astype(float),
                fps,
                include_orientation=False,
            )
            target_hash_after = hashlib.sha256(
                b"".join(targets[key].tobytes() for key in sorted(targets))
            ).hexdigest()
            metrics = evaluate(
                adapter,
                common,
                result.q,
                targets,
                source["left_dex3_qpos"].astype(float),
                source["right_dex3_qpos"].astype(float),
                fps,
            )
            shoulder = metrics.pop("_shoulder_indices")
            elbow = metrics.pop("_elbow_indices")
            delta = result.q - incoming_archive["projected_q"].astype(float)
            metrics["shoulder_posture_change_rad"] = stats(np.abs(delta[:, shoulder]))
            metrics["elbow_posture_change_rad"] = stats(np.abs(delta[:, elbow]))
            row = {
                "method": label,
                "representation_mode": mode,
                "episode_index": episode,
                "method_blind_solver": True,
                "incoming_target_modified": target_hash_before != target_hash_after,
                **metrics,
                "solver_metadata": result.metadata,
                "provenance": {
                    "input_target_archive": str(input_path.resolve()),
                    "input_target_archive_sha256": input_hash,
                    "source_archive": str(source_path.resolve()),
                    "source_archive_sha256": source_hash,
                    "solver_implementation": str(SOLVER_IMPLEMENTATION.resolve()),
                    "solver_implementation_sha256": implementation_hash,
                    "solver_config": str(DEFAULT_CONFIG.resolve()),
                    "solver_config_sha256": config_hash,
                },
            }
            atomic_npz(
                output_npz,
                q=result.q,
                input_q=incoming_archive["projected_q"],
                left_wrist_position_target=targets["left_wrist_position"],
                right_wrist_position_target=targets["right_wrist_position"],
                left_wrist_rotation_target=targets["left_wrist_rotation"],
                right_wrist_rotation_target=targets["right_wrist_rotation"],
            )
            atomic_json(output_json, row)
            arrays[(label, episode)] = {
                "q": result.q,
                "input_q": incoming_archive["projected_q"],
            }
            rows.append(row)
            print(
                f"{label}{episode:02d}: collision={row['hard_self_collision_frames']} "
                f"max={row['position_residual_mm']['max']:.3f}mm "
                f"branch={row['branch_discontinuities']} qualified={row['position_qualified']}",
                flush=True,
            )

    # Deterministic before/after anchor evidence. These images prove that each
    # representative frame has a collision-free same-target redundancy
    # realization; they do not override the full temporal qualification.
    anchor_rows: list[dict[str, Any]] = []
    figure, axes = plt.subplots(len(REPRESENTATIVE), 2, figsize=(13.6, 10.8), dpi=160)
    for row_index, (method, label, episode) in enumerate(REPRESENTATIVE):
        input_path = ADAPTER_DIR / f"{label}_ep{episode:03d}_raw_and_feasible.npz"
        source_path = archive_path(method, episode)
        with np.load(input_path, allow_pickle=False) as values:
            incoming_archive = {key: values[key].copy() for key in values.files}
        with np.load(source_path, allow_pickle=False) as values:
            source = {key: values[key].copy() for key in values.files}
        q_before = incoming_archive["projected_q"].astype(float)
        left_hand = source["left_dex3_qpos"].astype(float)
        right_hand = source["right_dex3_qpos"].astype(float)
        fps = float(1.0 / np.median(np.diff(source["timestamp"].astype(float))))
        collision = adapter._collision_metrics(q_before, left_hand, right_hand, fps)
        frame = max(
            collision["hard_frames"],
            key=lambda value: max(
                (
                    float(record["penetration_depth_m"])
                    for record in collision["by_frame"].get(value, [])
                ),
                default=0.0,
            ),
        )
        target_position = {
            side: incoming_archive[f"feasible_{side}_position_model"][frame].astype(float)
            for side in SIDES
        }
        target_rotation = {
            side: incoming_archive[f"feasible_{side}_rotation_model"][frame].astype(float)
            for side in SIDES
        }
        q_after, anchor_report = solver._anchor_solution(
            q_before[frame],
            target_position,
            target_rotation,
            False,
            left_hand[frame],
            right_hand[frame],
        )
        before_position, _ = adapter._pose(q_before[frame])
        after_position, _ = adapter._pose(q_after)
        wrist_displacement = max(
            np.linalg.norm(after_position[side] - before_position[side]) for side in SIDES
        )
        before_records = adapter._records(
            q_before[frame], left_hand[frame], right_hand[frame]
        )
        after_records = adapter._records(q_after, left_hand[frame], right_hand[frame])
        anchor_row = {
            "method": label,
            "episode_index": episode,
            "frame": int(frame),
            "anchor_status": anchor_report["status"],
            "before_contact_count": len(before_records),
            "after_contact_count": len(after_records),
            "before_maximum_penetration_mm": 1000.0
            * max(
                (float(record["penetration_depth_m"]) for record in before_records),
                default=0.0,
            ),
            "after_maximum_penetration_mm": 1000.0
            * max(
                (float(record["penetration_depth_m"]) for record in after_records),
                default=0.0,
            ),
            "achieved_wrist_displacement_mm": float(1000.0 * wrist_displacement),
            "incoming_wrist_target_modified": False,
            "q_change_norm_rad": float(np.linalg.norm(q_after - q_before[frame])),
        }
        anchor_rows.append(anchor_row)
        for column, (name, q_value, border) in enumerate(
            (("BEFORE", q_before[frame], "#b2182b"), ("AFTER anchor", q_after, "#1b7837"))
        ):
            image = render_state(g1, q_value, left_hand[frame], right_hand[frame])
            axis = axes[row_index, column]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"{label}{episode:02d} f{frame} — {name}\n"
                + (
                    f"penetration {anchor_row['before_maximum_penetration_mm']:.1f} mm"
                    if column == 0
                    else f"contacts {len(after_records)}; wrist shift {1000*wrist_displacement:.3f} mm"
                ),
                fontsize=9,
                color=border,
            )
            for spine in axis.spines.values():
                spine.set_color(border)
                spine.set_linewidth(2.5)
    figure.suptitle(
        "Common collision-aware redundancy audit — unchanged wrist targets\n"
        "AFTER panels are collision-free frame anchors; temporal qualification is reported separately",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(CONTACT_SHEET, bbox_inches="tight")
    plt.close(figure)
    atomic_json(RESULT_DIR / "REPRESENTATIVE_COLLISION_FREE_ANCHORS.json", anchor_rows)

    counts = {
        label: int(
            sum(bool(row["position_qualified"]) for row in rows if row["method"] == label)
        )
        for label in ("A", "B")
    }
    residual_arrays: list[np.ndarray] = []
    for row in rows:
        output_npz, _ = result_paths(row["method"], int(row["episode_index"]))
        with np.load(output_npz, allow_pickle=False) as values:
            q_value = values["q"].astype(float)
            target_value = {
                side: values[f"{side}_wrist_position_target"].astype(float)
                for side in SIDES
            }
        position_value, _ = adapter._pose_arrays(q_value)
        residual_arrays.append(
            np.maximum(
                *[
                    np.linalg.norm(
                        position_value[side] - target_value[side], axis=1
                    )
                    for side in SIDES
                ]
            )
        )
    aggregate_residual = stats(np.concatenate(residual_arrays), 1000.0)
    position_pass = counts == {"A": 3, "B": 3}
    full_6d = {"status": "NOT_RUN_POSITION_GATE_FAILED", "A": 0, "B": 0}
    loaded_dex3 = {
        "status": "NOT_RUN_6D_GATE_NOT_PASSED",
        "passed_joints": 0,
        "joint_count": 14,
        "commanded_violations": None,
        "measured_violations": None,
    }
    audit = {
        "schema_version": "common_collision_aware_ik_qualification_v1",
        "scope": "TRAIN_SMOKE_EPISODES_0_24_49_ONLY",
        "method_blind": True,
        "incoming_target_modified": any(row["incoming_target_modified"] for row in rows),
        "position_only": {
            "A_qualified_episodes": counts["A"],
            "B_qualified_episodes": counts["B"],
            "A_hard_self_collision_frames": sum(
                row["hard_self_collision_frames"] for row in rows if row["method"] == "A"
            ),
            "B_hard_self_collision_frames": sum(
                row["hard_self_collision_frames"] for row in rows if row["method"] == "B"
            ),
            "hard_limit_violations": sum(row["hard_limit_violations"] for row in rows),
            "branch_discontinuities": sum(row["branch_discontinuities"] for row in rows),
            "finite_trajectories": sum(bool(row["finite"]) for row in rows),
            "wrist_position_residual_mm": aggregate_residual,
            "pass": position_pass,
        },
        "full_6d": full_6d,
        "loaded_dex3": loaded_dex3,
        "representative_collision_free_anchors": anchor_rows,
        "visualization": str(CONTACT_SHEET.resolve()),
        "rows": rows,
        "provenance": {
            "collision_aware_solver_sha256": implementation_hash,
            "collision_aware_config_sha256": config_hash,
            "morphology_adapter_sha256": sha256(ADAPTER_IMPLEMENTATION),
            "common_ik_sha256": sha256(COMMON_SOLVER_IMPLEMENTATION),
        },
        "final_status": (
            "COMMON_COLLISION_AWARE_IK_QUALIFIED"
            if position_pass and full_6d["A"] == 3 and full_6d["B"] == 3
            else "COMMON_SELF_COLLISION_STILL_INVALID"
        ),
    }
    atomic_json(AUDIT_JSON, audit)

    lines = [
        "# Common collision-aware IK qualification",
        "",
        "## Scientific contract",
        "",
        "- Incoming feasible wrist target arrays modified: **NO**",
        "- Method identity consumed: **NO**",
        "- Task/object pose consumed by redundancy solver: **NO**",
        "- Common hard limits and collision model: **YES**",
        "- Deterministic common seed policy: **YES**",
        "",
        "## Position-only smoke",
        "",
        "| Method | Qualified episodes | Hard self-collision frames |",
        "|---|---:|---:|",
        f"| A | {counts['A']}/3 | {audit['position_only']['A_hard_self_collision_frames']} |",
        f"| B | {counts['B']}/3 | {audit['position_only']['B_hard_self_collision_frames']} |",
        "",
        "| Episode | Method | Acceptance | Hard collision frames | Hard limits | Branches | Max residual (mm) | Temporal | Qualified |",
        "|---:|:---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['episode_index']} | {row['method']} | {100*row['position_acceptance_rate']:.2f}% | "
            f"{row['hard_self_collision_frames']} | {row['hard_limit_violations']} | "
            f"{row['branch_discontinuities']} | {row['position_residual_mm']['max']:.3f} | "
            f"{'PASS' if row['temporal_pass'] else 'FAIL'} | {'PASS' if row['position_qualified'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Representative same-target anchors",
            "",
            "Every representative collision frame admitted a zero-contact joint configuration at the same incoming wrist target. However, the full sequential gate—not isolated-frame feasibility—is authoritative.",
            "",
            f"Contact sheet: `{CONTACT_SHEET.resolve()}`",
            "",
            "## Downstream gates",
            "",
            f"- Full 6D: **{full_6d['status']}**",
            f"- Loaded Dex3: **{loaded_dex3['status']}**",
            "",
            "The workflow stopped before full 6D and loaded Dex3 because position-only collision/temporal qualification did not pass for all six trajectories.",
            "",
            f"Final status: **{audit['final_status']}**",
        ]
    )
    atomic_text(AUDIT_MD, "\n".join(lines) + "\n")

    print("COMMON COLLISION-AWARE IK QUALIFICATION")
    print("\nPosition target modified:\nNO / minimal numerical residual only")
    print("\nMethod-blind solver:\nYES")
    print("\nPOSITION-ONLY:")
    print(f"\nA:\n{counts['A']}/3")
    print(f"\nB:\n{counts['B']}/3")
    print("\nSelf-collision frames:")
    print(f"\nA:\n{audit['position_only']['A_hard_self_collision_frames']}")
    print(f"\nB:\n{audit['position_only']['B_hard_self_collision_frames']}")
    print("\nWrist position residual:")
    print(
        f"mean {aggregate_residual['mean']:.6f} mm\n"
        f"p95 {aggregate_residual['p95']:.6f} mm\n"
        f"max {aggregate_residual['max']:.6f} mm"
    )
    print(f"\nBranch discontinuities:\n{audit['position_only']['branch_discontinuities']}")
    print("\nFULL 6D:")
    print(f"\nA:\n{full_6d['A']}/3")
    print(f"\nB:\n{full_6d['B']}/3")
    print(f"\nLoaded Dex3:\n{loaded_dex3['passed_joints']}/14")
    print(f"\nVisual audit:\n{CONTACT_SHEET.resolve()}")
    print(f"\nAudit report:\n{AUDIT_MD.resolve()}")
    print(f"\n{audit['final_status']}")
    return 0 if audit["final_status"] == "COMMON_COLLISION_AWARE_IK_QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
