"""Model-FK diagnostic renders for the semantic primitive candidate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aloha_g1_dataset_v1.core import (
    HandMapper,
    inverse_transform,
    physical_pinch_frame,
    raw_contact_position,
    raw_wrist_pose,
)

from .common import V2_ROOT


COLORS = {"THUMB": "tab:orange", "INDEX": "tab:blue", "THIRD": "tab:gray"}


def _local_points(runtime: Any, v1: Mapping[str, Any], side: str, q: np.ndarray) -> dict[str, Any]:
    arm = np.asarray(v1["nominal_g1_arm_q"], dtype=np.float64)
    left = q if side == "left" else runtime.open_hand_q["left"]
    right = q if side == "right" else runtime.open_hand_q["right"]
    runtime.assign(arm, left, right)
    wrist = raw_wrist_pose(runtime, side)
    inverse = inverse_transform(wrist)
    labels = tuple(v1["target_frames"][f"{side}_physical_pinch_contacts"])
    pinch = inverse @ physical_pinch_frame(runtime, side, labels)
    points: dict[str, np.ndarray] = {"WRIST": np.zeros(3)}
    palm_world = wrist @ np.asarray(v1["target_frames"][f"{side}_wrist_to_palm"])
    points["PALM"] = (inverse @ palm_world)[:3, 3]
    for role in ("A", "B", "C"):
        spec = runtime.contacts[f"{side}_{role}"]
        world = raw_contact_position(runtime, f"{side}_{role}")
        local = inverse[:3, :3] @ world + inverse[:3, 3]
        digit = "THIRD" if role == "C" else (
            "THUMB" if "thumb" in spec.link else "INDEX"
        )
        points[digit] = local
    return {"points": points, "pinch": pinch}


def _equal_3d(axis: Any, values: np.ndarray) -> None:
    center = 0.5 * (np.min(values, axis=0) + np.max(values, axis=0))
    radius = max(0.07, 0.6 * float(np.max(np.ptp(values, axis=0))))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _draw_state(
    axis: Any,
    state: Mapping[str, Any],
    title: str,
    object_aperture: float | None = None,
) -> None:
    points = state["points"]
    palm = points["PALM"]
    values = [points["WRIST"], palm]
    axis.plot(
        [points["WRIST"][0], palm[0]],
        [points["WRIST"][1], palm[1]],
        [points["WRIST"][2], palm[2]],
        color="black",
        linewidth=3,
    )
    for digit in ("THUMB", "INDEX", "THIRD"):
        point = points[digit]
        values.append(point)
        axis.plot(
            [palm[0], point[0]],
            [palm[1], point[1]],
            [palm[2], point[2]],
            color=COLORS[digit],
            linewidth=2,
        )
        axis.scatter(*point, color=COLORS[digit], s=35, label=digit)
    pinch = state["pinch"]
    center = pinch[:3, 3]
    values.append(center)
    axis.scatter(*center, marker="x", color="magenta", s=55, label="pinch center")
    for column, color, name in ((0, "red", "approach"), (1, "green", "closing"), (2, "purple", "lateral")):
        vector = 0.035 * pinch[:3, column]
        axis.quiver(*center, *vector, color=color, linewidth=1.5, label=name)
    if object_aperture is not None:
        closing = pinch[:3, 1]
        surfaces = [center - 0.5 * object_aperture * closing, center + 0.5 * object_aperture * closing]
        for value in surfaces:
            values.append(value)
            axis.scatter(*value, marker="s", color="limegreen", s=28)
        axis.plot(
            [surfaces[0][0], surfaces[1][0]],
            [surfaces[0][1], surfaces[1][1]],
            [surfaces[0][2], surfaces[1][2]],
            color="limegreen",
            linestyle=":",
            label="class thickness",
        )
    array = np.asarray(values)
    _equal_3d(axis, array)
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("wrist x [m]")
    axis.set_ylabel("wrist y [m]")
    axis.set_zlabel("wrist z [m]")


def _render_semantic_side(
    path: Path,
    runtime: Any,
    v1: Mapping[str, Any],
    side: str,
    states: Mapping[str, np.ndarray],
    aperture: float,
) -> None:
    names = ("OPEN", "PREGRASP", "GRASP", "HOLD", "RELEASE")
    figure = plt.figure(figsize=(18, 4))
    for index, phase in enumerate(names, start=1):
        axis = figure.add_subplot(1, len(names), index, projection="3d")
        state = _local_points(runtime, v1, side, np.asarray(states[phase]))
        _draw_state(axis, state, f"{side.upper()} {phase}", aperture if phase in {"GRASP", "HOLD"} else None)
    handles, labels = figure.axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=7, fontsize=8)
    figure.suptitle(
        f"{side} semantic thumb-index states | TARGET_OBJECT_CLASS_GEOMETRY",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _historical_grasp_q(runtime: Any, v1: Mapping[str, Any], side: str) -> np.ndarray:
    mapper = HandMapper(dict(v1), runtime)
    return mapper.proposed_primitives[side]["GRASP"].copy()


def render_all(
    output_root: Path,
    runtime: Any,
    v1: Mapping[str, Any],
    config: Mapping[str, Any],
    primitives: Mapping[str, Any],
    selected_neutrals: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    states = {
        side: {
            phase: np.asarray(value, dtype=np.float64).copy()
            for phase, value in primitives["sides"][side]["states"].items()
        }
        for side in ("left", "right")
    }
    for side in ("left", "right"):
        indices = np.asarray(primitives["sides"][side]["third_indices"], dtype=np.int64)
        for value in states[side].values():
            value[indices] = np.asarray(selected_neutrals[side])
    render_root = output_root / "renders"
    _render_semantic_side(
        render_root / "left_phone_states.png",
        runtime,
        v1,
        "left",
        states["left"],
        float(config["object_class_geometry"]["left"]["target_surface_aperture_m"]),
    )
    _render_semantic_side(
        render_root / "right_accessory_states.png",
        runtime,
        v1,
        "right",
        states["right"],
        float(config["object_class_geometry"]["right"]["target_surface_aperture_m"]),
    )

    figure = plt.figure(figsize=(12, 6))
    for column, side in enumerate(("left", "right"), start=1):
        historical = _historical_grasp_q(runtime, v1, side)
        active_open = states[side]["GRASP"].copy()
        indices = np.asarray(primitives["sides"][side]["third_indices"], dtype=np.int64)
        active_open[indices] = runtime.open_hand_q[side][indices]
        selected = states[side]["GRASP"]
        axis = figure.add_subplot(1, 2, column, projection="3d")
        for q, label, color in (
            (historical, "v1 phase-coupled third", "tab:red"),
            (active_open, "active open third", "tab:blue"),
            (selected, "v2.1 selected global third", "tab:green"),
        ):
            state = _local_points(runtime, v1, side, q)
            point = state["points"]["THIRD"]
            palm = state["points"]["PALM"]
            axis.plot(
                [palm[0], point[0]],
                [palm[1], point[1]],
                [palm[2], point[2]],
                color=color,
                linewidth=2,
                label=label,
            )
            axis.scatter(*point, color=color, s=45)
        values = np.asarray(
            [
                _local_points(runtime, v1, side, q)["points"]["THIRD"]
                for q in (historical, active_open, selected)
            ]
        )
        _equal_3d(axis, values)
        axis.set_title(f"{side} third neutral comparison")
        axis.legend(fontsize=8)
    figure.suptitle("Third finger is globally fixed and non-task")
    figure.tight_layout()
    figure.savefig(render_root / "third_neutral_comparison.png", dpi=170)
    plt.close(figure)

    exact_payload = np.load(V2_ROOT / "development/dex3_solution.npz", allow_pickle=False)
    exact_q = exact_payload["dex3_q"].astype(np.float64)
    comparison = {
        "Current Proposed v1": _historical_grasp_q(runtime, v1, "left"),
        "Exact-contact v2 ablation": exact_q,
        "Feasible semantic v2.1": states["left"]["GRASP"],
    }
    figure = plt.figure(figsize=(14, 5))
    for index, (label, q) in enumerate(comparison.items(), start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        state = _local_points(runtime, v1, "left", q)
        _draw_state(
            axis,
            state,
            label,
            float(config["object_class_geometry"]["left"]["target_surface_aperture_m"]),
        )
    figure.suptitle("Same nominal frozen wrist: v1 vs exact-contact ablation vs v2.1")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(render_root / "hand_v1_vs_v2_vs_v2_1.png", dpi=170)
    plt.close(figure)
    return {
        "files": [
            str(render_root / "left_phone_states.png"),
            str(render_root / "right_accessory_states.png"),
            str(render_root / "third_neutral_comparison.png"),
            str(render_root / "hand_v1_vs_v2_vs_v2_1.png"),
        ],
        "object_geometry_label": "TARGET_OBJECT_CLASS_GEOMETRY",
        "source_object_pose_used": False,
    }


__all__ = ["render_all"]
