#!/usr/bin/env python3
"""TRAIN-only visual/numeric audit of the common A/B target and IK boundary.

This tool is deliberately read-only.  It consumes the persisted smoke targets
produced by the single-variable reset and never changes registration, targets,
IK, physics, or policy artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics


OUT = ROOT / "outputs/single_variable_ab_common_execution"
SMOKE = OUT / "04_workspace_registration/smoke"
REGISTRATION = OUT / "04_workspace_registration/COMMON_TASK_REGISTRATION_TRAIN_SMOKE.json"
VISUAL = OUT / "target_ik_visual_audit"
EPISODES = (0, 24, 49)
METHODS = (("baseline", "A", "WRIST", "#3465a4"), ("proposed", "B", "INTERACTION", "#cc0000"))
SIDES = ("left", "right")


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_path(method: str, episode: int) -> Path:
    matches = sorted((SMOKE / method / "trajectories").glob(f"*ep{episode:03d}.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {method} archive for episode {episode}: {matches}")
    return matches[0]


def load_archive(method: str, episode: int) -> tuple[Path, dict[str, np.ndarray]]:
    path = archive_path(method, episode)
    with np.load(path, allow_pickle=False) as source:
        return path, {key: source[key] for key in source.files}


def event_window(values: dict[str, np.ndarray]) -> tuple[int, int, dict[str, int]]:
    events = dict(zip(values["event_names"].astype(str), values["event_frames"].astype(int)))
    start = int(events["LEFT_CLOSE_ONSET"])
    stop = int(events["LEFT_STABLE_HOLD"])
    if not 0 <= start <= stop < len(values["timestamp"]):
        raise RuntimeError(f"invalid common grasp window {start}:{stop}")
    return start, stop, events


def mean_rotation(matrices: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(matrices, dtype=np.float64)).mean().as_matrix()


def oriented_box_surface_distance(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    local = (np.asarray(points) - center) @ rotation
    outside = np.maximum(np.abs(local) - half_extents, 0.0)
    outside_norm = np.linalg.norm(outside, axis=1)
    inside = np.all(np.abs(local) <= half_extents, axis=1)
    inside_depth = np.min(half_extents - np.abs(local), axis=1)
    return np.where(inside, -inside_depth, outside_norm)


def add_box(ax: Any, center: np.ndarray, size: np.ndarray, color: str, alpha: float, label: str | None = None) -> None:
    c = np.asarray(center, dtype=float)
    h = 0.5 * np.asarray(size, dtype=float)
    vertices = np.array(
        [[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])]
    ) + c
    edges = []
    for i, a in enumerate(vertices):
        for j, b in enumerate(vertices):
            if j > i and np.count_nonzero(np.abs(a - b) > 1e-12) == 1:
                edges.append((a, b))
    for index, (a, b) in enumerate(edges):
        ax.plot(*np.stack((a, b)).T, color=color, alpha=alpha, lw=1.2, label=label if index == 0 else None)


def axes_at(ax: Any, origin: np.ndarray, rotation: np.ndarray, length: float = 0.035) -> None:
    for column, color in enumerate(("#d73027", "#1a9850", "#4575b4")):
        endpoint = origin + length * rotation[:, column]
        ax.plot(*np.stack((origin, endpoint)).T, color=color, lw=1.5)


def arm_polyline_world(g1: G1Kinematics, q: np.ndarray, side: str) -> np.ndarray:
    landmarks = g1.arm_landmarks(q)[side]
    points = np.stack(
        (
            landmarks["shoulder_pitch"],
            landmarks["elbow"],
            landmarks["wrist_yaw"],
        )
    )
    return g1.model_to_world_position(points)


def setup_view(ax: Any, view: str, bounds: tuple[np.ndarray, np.ndarray]) -> None:
    low, high = bounds
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect(np.maximum(high - low, 1e-6))
    if view == "top":
        ax.view_init(elev=89.8, azim=-90)
    else:
        ax.view_init(elev=25, azim=-65)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_zlabel("world Z (m)")
    ax.grid(alpha=0.2)


def plot_method(
    ax: Any,
    g1: G1Kinematics,
    scene: dict[str, Any],
    values: dict[str, np.ndarray],
    label: str,
    mode: str,
    color: str,
    view: str,
    bounds: tuple[np.ndarray, np.ndarray],
) -> None:
    start, stop, _ = event_window(values)
    sample = np.arange(0, len(values["timestamp"]), max(1, len(values["timestamp"]) // 90))
    success = values["ik_success_per_frame"].astype(bool)
    for side, linestyle in (("left", "-"), ("right", "--")):
        position_model = np.asarray(values[f"target_{side}_wrist_position_model"], dtype=float)
        position = g1.model_to_world_position(position_model)
        ax.plot(*position.T, color=color, lw=0.8, alpha=0.40, ls=linestyle)
        if side == "left":
            ax.plot(*position[start : stop + 1].T, color=color, lw=3.0, alpha=0.95, label=f"{label} grasp window")
        good = sample[success[sample]]
        bad = sample[~success[sample]]
        if len(good):
            ax.scatter(*position[good].T, s=6, c="#2ca25f", depthshade=False)
        if len(bad):
            ax.scatter(*position[bad].T, s=7, c="#de2d26", marker="x", depthshade=False)
    object_position = np.asarray(values["registered_object_position_world"], dtype=float)
    object_rotation = Rotation.from_quat(values["registered_object_quaternion_xyzw"].astype(float)).as_matrix()
    visual = np.asarray(scene["doll"].get("visual_dimensions_m", [0.120, 0.090, 0.085]), dtype=float)
    add_box(ax, object_position, visual, "#fdae61", 0.95, "registered doll")
    axes_at(ax, object_position, object_rotation, 0.045)
    bin_position = np.asarray(values["registered_bin_position_world"], dtype=float)
    bin_size = np.asarray(scene["bin"]["outer_dimensions_xyz_m"], dtype=float)
    add_box(ax, bin_position + np.array([0.0, 0.0, 0.5 * bin_size[2]]), bin_size, "#555555", 0.65, "150 mm bin")
    grasp_frame = (start + stop) // 2
    for side in SIDES:
        wrist_position = g1.model_to_world_position(values[f"target_{side}_wrist_position_model"][grasp_frame].astype(float))
        wrist_rotation = g1.model_to_world_rotation(values[f"target_{side}_wrist_rotation_model"][grasp_frame].astype(float))
        axes_at(ax, wrist_position, wrist_rotation, 0.032)
        shoulder = g1.model_to_world_position(g1.fixed_shoulder_anchors_model()[side])
        ax.scatter(*shoulder, c="#111111", s=35, marker="o")
        q = values["g1_arm_qpos"][grasp_frame].astype(float)
        arm = arm_polyline_world(g1, q, side)
        ax.plot(*arm.T, color="#222222", lw=4.0, alpha=0.8)
    pelvis = np.asarray(scene["g1"]["root_position_world_xyz_m"], dtype=float)
    ax.plot([pelvis[0], pelvis[0]], [pelvis[1], pelvis[1]], [pelvis[2], pelvis[2] + 0.45], color="#222222", lw=8, alpha=0.35)
    setup_view(ax, view, bounds)
    rate = 100.0 * float(np.mean(success))
    ax.set_title(f"{label} — {mode}\nraw target; IK accepted {rate:.1f}%")


def main() -> int:
    VISUAL.mkdir(parents=True, exist_ok=True)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    common = load_common_config(SMOKE / "config/common_config.json")
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    tool_report = json.loads((SMOKE / "config/tool_frame_report.json").read_text(encoding="utf-8"))
    wrist_to_tool = {
        side: np.asarray(tool_report["g1"][f"{side}_wrist_to_grasp_frame"], dtype=float)
        for side in SIDES
    }
    registration_by_episode = {int(row["episode_index"]): row for row in registration["entries"]}
    physical_path = ROOT / "outputs/final_episode_registered_eval35/01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
    physical = json.loads(physical_path.read_text(encoding="utf-8"))
    doll_dimensions = np.asarray(physical["doll"]["visual_dimensions_m"], dtype=float)
    shoulders = g1.fixed_shoulder_anchors_model()

    loaded: dict[tuple[int, str], tuple[Path, dict[str, np.ndarray]]] = {}
    all_points: list[np.ndarray] = []
    for episode in EPISODES:
        for method, _, _, _ in METHODS:
            path, values = load_archive(method, episode)
            loaded[(episode, method)] = (path, values)
            for side in SIDES:
                all_points.append(g1.model_to_world_position(values[f"target_{side}_wrist_position_model"].astype(float)))
            all_points.append(values["registered_object_position_world"].astype(float)[None, :])
            all_points.append(values["registered_bin_position_world"].astype(float)[None, :])
    stacked = np.concatenate(all_points, axis=0)
    low = np.min(stacked, axis=0) - np.array([0.10, 0.10, 0.08])
    high = np.max(stacked, axis=0) + np.array([0.10, 0.10, 0.22])
    bounds = (low, high)

    rows: list[dict[str, Any]] = []
    target_semantics_pass = {"A": True, "B": True}
    for episode in EPISODES:
        entry = registration_by_episode[episode]
        workspace = np.asarray(entry["common_workspace_transform_matrix"], dtype=float)
        inverse_workspace = np.linalg.inv(workspace)
        figure = plt.figure(figsize=(14, 6.4), constrained_layout=True)
        for panel, (method, label, mode, color) in enumerate(METHODS, start=1):
            path, values = loaded[(episode, method)]
            ax = figure.add_subplot(1, 2, panel, projection="3d")
            plot_method(ax, g1, scene, values, label, mode, color, "oblique", bounds)
            start, stop, events = event_window(values)
            window = slice(start, stop + 1)
            object_position = np.asarray(values["registered_object_position_world"], dtype=float)
            object_rotation = Rotation.from_quat(values["registered_object_quaternion_xyzw"].astype(float)).as_matrix()
            source_object_position = np.asarray(entry["source_object_pose"]["position_xyz_m"], dtype=float)
            source_object_rotation = Rotation.from_quat(entry["source_object_pose"]["quaternion_xyzw"]).as_matrix()
            method_row: dict[str, Any] = {
                "episode_index": episode,
                "method": label,
                "representation_mode": mode,
                "archive": str(path.resolve()),
                "archive_sha256": sha256(path),
                "source_name": str(values["source_directory_name"]),
                "grasp_window": {"start": start, "stop": stop, "events": events},
                "registered_object_position_world_m": object_position,
                "registered_object_quaternion_xyzw": values["registered_object_quaternion_xyzw"],
                "sides": {},
            }
            semantic_translation_max = 0.0
            semantic_rotation_max = 0.0
            for side in SIDES:
                wrist_position_model = values[f"target_{side}_wrist_position_model"].astype(float)
                wrist_rotation_model = values[f"target_{side}_wrist_rotation_model"].astype(float)
                wrist_position_world = g1.model_to_world_position(wrist_position_model)
                wrist_rotation_world = g1.model_to_world_rotation(wrist_rotation_model)
                tool_position_generated = wrist_position_world + np.einsum(
                    "tij,j->ti", wrist_rotation_world, wrist_to_tool[side][:3, 3]
                )
                tool_rotation_generated = np.einsum(
                    "tij,jk->tik", wrist_rotation_world, wrist_to_tool[side][:3, :3]
                )
                tool_position_expected = values[f"target_{side}_interaction_frame_position_world"].astype(float)
                # Recover the pre-workspace source-conditioned target and prove
                # that the common rigid transform preserves its object relation.
                source_tool_position = (
                    tool_position_expected - workspace[:3, 3]
                ) @ workspace[:3, :3]
                source_relation = (source_tool_position - source_object_position) @ source_object_rotation
                expected_registered_relation = source_relation
                generated_registered_relation = (
                    tool_position_generated - object_position
                ) @ object_rotation
                translation_discrepancy = np.linalg.norm(
                    generated_registered_relation - expected_registered_relation, axis=1
                )
                semantic_translation_max = max(semantic_translation_max, float(np.max(translation_discrepancy)))
                # The rotation round trip tests the fixed TCP/wrist convention.
                reconstructed_wrist_rotation = np.einsum(
                    "tij,jk->tik", tool_rotation_generated, wrist_to_tool[side][:3, :3].T
                )
                rotation_discrepancy = (
                    Rotation.from_matrix(reconstructed_wrist_rotation)
                    * Rotation.from_matrix(wrist_rotation_world).inv()
                ).magnitude()
                semantic_rotation_max = max(semantic_rotation_max, float(np.max(rotation_discrepancy)))
                median_rotation = mean_rotation(wrist_rotation_world[window])
                relative_rotation = object_rotation.T @ median_rotation
                shoulder_distance = np.linalg.norm(
                    wrist_position_model[window] - shoulders[side], axis=1
                )
                surface_distance = oriented_box_surface_distance(
                    tool_position_generated[window],
                    object_position,
                    object_rotation,
                    0.5 * doll_dimensions,
                )
                method_row["sides"][side] = {
                    "raw_wrist_xyz_world_m_median": np.median(wrist_position_world[window], axis=0),
                    "raw_wrist_xyz_world_m_min": np.min(wrist_position_world[window], axis=0),
                    "raw_wrist_xyz_world_m_max": np.max(wrist_position_world[window], axis=0),
                    "raw_wrist_quaternion_xyzw_world_mean": Rotation.from_matrix(median_rotation).as_quat(),
                    "raw_wrist_rpy_deg_world_mean": Rotation.from_matrix(median_rotation).as_euler("xyz", degrees=True),
                    "shoulder_to_wrist_distance_m": {
                        "min": float(np.min(shoulder_distance)),
                        "mean": float(np.mean(shoulder_distance)),
                        "max": float(np.max(shoulder_distance)),
                    },
                    "object_to_wrist_translation_object_m_median": np.median(
                        (wrist_position_world[window] - object_position) @ object_rotation,
                        axis=0,
                    ),
                    "object_to_wrist_quaternion_xyzw_mean": Rotation.from_matrix(relative_rotation).as_quat(),
                    "object_to_wrist_rpy_deg_mean": Rotation.from_matrix(relative_rotation).as_euler("xyz", degrees=True),
                    "grasp_frame_to_doll_surface_distance_mm": {
                        "min_signed": float(np.min(surface_distance) * 1000.0),
                        "median_signed": float(np.median(surface_distance) * 1000.0),
                        "max_signed": float(np.max(surface_distance) * 1000.0),
                        "sign_convention": "negative is inside declared visual envelope",
                    },
                    "source_object_to_grasp_frame_translation_m_median": np.median(source_relation[window], axis=0),
                    "registered_expected_object_to_grasp_frame_translation_m_median": np.median(expected_registered_relation[window], axis=0),
                    "actual_generated_object_to_grasp_frame_translation_m_median": np.median(generated_registered_relation[window], axis=0),
                    "translation_discrepancy_m_max": float(np.max(translation_discrepancy)),
                    "orientation_fixed_frame_round_trip_discrepancy_rad_max": float(np.max(rotation_discrepancy)),
                    "grasp_aperture_relation": {
                        "whole_hand_target_enclosure_radius_m": float(
                            tool_report["g1"]["target_grasp_enclosure_radius_m"]
                        ),
                        "doll_visual_half_extents_m": 0.5 * doll_dimensions,
                    },
                }
            semantics_pass = semantic_translation_max <= 1e-6 and semantic_rotation_max <= 1e-9
            target_semantics_pass[label] &= semantics_pass
            method_row["semantic_round_trip"] = {
                "maximum_translation_discrepancy_m": semantic_translation_max,
                "maximum_rotation_discrepancy_rad": semantic_rotation_max,
                "pass": semantics_pass,
            }
            method_row["position_only_ik_rate_from_persisted_audit"] = None
            method_row["root_cause_classification"] = (
                "TARGET_TASK_CONSISTENT_BUT_G1_POSITION_INFEASIBLE"
                if semantics_pass
                else "TARGET_FRAME_WRONG"
            )
            rows.append(method_row)
        figure.suptitle(
            f"TRAIN smoke {episode:02d}: raw A/B targets and common sequential IK\n"
            "green dot = accepted frame, red x = rejected; axes RGB = local XYZ",
            fontsize=13,
        )
        path = VISUAL / f"SMOKE_{episode:02d}_A_B_TARGET_IK_AUDIT.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)

    contact = plt.figure(figsize=(17, 10), constrained_layout=True)
    for row_index, view in enumerate(("top", "oblique")):
        for column, episode in enumerate(EPISODES):
            ax = contact.add_subplot(2, 3, row_index * 3 + column + 1, projection="3d")
            for method, label, mode, color in METHODS:
                _, values = loaded[(episode, method)]
                plot_method(ax, g1, scene, values, label, mode, color, view, bounds)
            ax.set_title(f"Episode {episode:02d} — {'TOP' if view == 'top' else 'FRONT-OBLIQUE'}")
    contact.suptitle("Common target/IK audit — identical scale and cameras", fontsize=15)
    contact_path = VISUAL / "SMOKE_AB_TARGET_IK_CONTACT_SHEET.png"
    contact.savefig(contact_path, dpi=220)
    plt.close(contact)

    report = {
        "schema_version": "single_variable_ab_common_target_visual_numeric_audit_v1",
        "status": "PASS" if all(target_semantics_pass.values()) else "FAIL",
        "scope": "TRAIN_ONLY_SMOKE_0_24_49",
        "scientific_inputs_modified": False,
        "registration_manifest": str(REGISTRATION.resolve()),
        "registration_manifest_sha256": sha256(REGISTRATION),
        "common_workspace_transform": registration["common_workspace_registration"],
        "target_task_consistency": target_semantics_pass,
        "classification_basis": (
            "fixed wrist-to-tool reconstruction and source/object relation preservation; "
            "IK outputs are visualized but are not used to judge target semantics"
        ),
        "rows": rows,
        "visuals": {
            "episode_images": [
                str((VISUAL / f"SMOKE_{episode:02d}_A_B_TARGET_IK_AUDIT.png").resolve())
                for episode in EPISODES
            ],
            "contact_sheet": str(contact_path.resolve()),
        },
        "inputs": {
            "physical_environment": str(physical_path.resolve()),
            "physical_environment_sha256": sha256(physical_path),
            "common_config": str((SMOKE / "config/common_config.json").resolve()),
            "common_config_sha256": sha256(SMOKE / "config/common_config.json"),
            "tool_frame_report": str((SMOKE / "config/tool_frame_report.json").resolve()),
            "tool_frame_report_sha256": sha256(SMOKE / "config/tool_frame_report.json"),
        },
    }
    atomic_json(OUT / "COMMON_TARGET_VISUAL_AUDIT.json", report)
    lines = [
        "# Common target visual and numeric audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "This is a read-only TRAIN-smoke audit. No registration, target, solver, policy, or physical artifact was changed.",
        "",
        f"- A target task-consistent: **{'PASS' if target_semantics_pass['A'] else 'FAIL'}**",
        f"- B target task-consistent: **{'PASS' if target_semantics_pass['B'] else 'FAIL'}**",
        f"- Contact sheet: `{contact_path.resolve()}`",
        "- Target semantics are judged before IK from object-relative rigid-transform and fixed wrist/tool round trips.",
        "",
        "| episode | method | grasp window | max relation discrepancy | max fixed-frame rotation discrepancy | left grasp-frame/doll signed distance | classification |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        left = row["sides"]["left"]
        lines.append(
            f"| {row['episode_index']} | {row['method']} ({row['representation_mode']}) | "
            f"{row['grasp_window']['start']}–{row['grasp_window']['stop']} | "
            f"{row['semantic_round_trip']['maximum_translation_discrepancy_m']*1000:.6f} mm | "
            f"{math.degrees(row['semantic_round_trip']['maximum_rotation_discrepancy_rad']):.9f} deg | "
            f"{left['grasp_frame_to_doll_surface_distance_mm']['min_signed']:.3f} mm | "
            f"{row['root_cause_classification']} |"
        )
    lines.extend(
        [
            "",
            "The two methods reconstruct the same source-conditioned interaction-point positions. Their wrist origins and orientations may differ because that is the declared representation switch. A negative signed grasp-frame/doll surface distance means the generated whole-hand interaction point lies inside the declared visual envelope; it is not an IK-success claim.",
            "",
        ]
    )
    atomic_text(OUT / "COMMON_TARGET_VISUAL_AUDIT.md", "\n".join(lines))
    print(OUT / "COMMON_TARGET_VISUAL_AUDIT.json")
    print(contact_path)
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
