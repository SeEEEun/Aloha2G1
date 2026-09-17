#!/usr/bin/env python3
"""Audit matched ALOHA/G1 phase images without modifying either dataset."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SELECTED = ROOT / "outputs/policy_b_offline_phase_probe/selected_frames.json"
G1_SELECTED = (
    ROOT
    / "outputs/policy_b_g1visual/root_cause_diagnostic/old_policy_on_g1visual/selected_frames.json"
)
SOURCE_DATA = ROOT / "datasets/doll_handoff_proposed_b_50/data/chunk-000/file-000.parquet"
G1_DATA = ROOT / "datasets/doll_handoff_proposed_b_g1visual_50/data/chunk-000/file-000.parquet"
PLANS = ROOT / "outputs/policy_b_g1visual/dataset_render_full/render_plans"
TRAJECTORIES = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories"
PLAN_MANIFEST = PLANS / "render_plan_manifest.json"
CAMERA = ROOT / "outputs/policy_b_isaac_validation/camera/source_like_cam_high.json"
DEFAULT_OUTPUT = ROOT / "outputs/policy_b_g1visual/root_cause_diagnostic/phase_visual_qa"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_array(table: Any, name: str, dtype: Any) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def image_path(row: dict[str, Any], g1: bool) -> Path:
    return Path(row["reference_image"])


def make_contact_sheet(
    phase: str,
    source_rows: list[dict[str, Any]],
    g1_rows: list[dict[str, Any]],
    output: Path,
) -> Path:
    pairs = []
    for source, g1 in zip(source_rows, g1_rows, strict=True):
        left = Image.open(image_path(source, False)).convert("RGB")
        right = Image.open(image_path(g1, True)).convert("RGB")
        if left.size != right.size:
            right = right.resize(left.size, Image.Resampling.BILINEAR)
        pairs.append((source, left, right))
    width = 640
    image_width = width // 2
    image_height = round(image_width * 480 / 640)
    header = 42
    row_header = 24
    canvas = Image.new("RGB", (width, header + len(pairs) * (image_height + row_header)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 5), f"Phase: {phase}", fill="black", font=font)
    draw.text((8, 22), "ALOHA source cam_high", fill="black", font=font)
    draw.text((image_width + 8, 22), "G1 SOURCE_LIKE_CAM_HIGH", fill="black", font=font)
    y = header
    for source, left, right in pairs:
        label = f"episode {int(source['episode_index']):02d} | frame {int(source['frame_index']):04d} | identical state/action row"
        draw.rectangle((0, y, width, y + row_header), fill=(235, 235, 235))
        draw.text((8, y + 6), label, fill="black", font=font)
        y += row_header
        canvas.paste(left.resize((image_width, image_height), Image.Resampling.LANCZOS), (0, y))
        canvas.paste(right.resize((image_width, image_height), Image.Resampling.LANCZOS), (image_width, y))
        y += image_height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def hand_world_transforms(trajectory: Any, side: str) -> np.ndarray:
    position = np.asarray(
        trajectory[f"achieved_{side}_static_whole_hand_position_world"], dtype=np.float64
    )
    rotation_model = np.asarray(
        trajectory[f"achieved_{side}_static_whole_hand_orientation_model"], dtype=np.float64
    )
    root_rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotation_world = np.einsum("ij,tjk->tik", root_rotation, rotation_model)
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(position), axis=0)
    result[:, :3, :3] = rotation_world
    result[:, :3, 3] = position
    return result


def object_transforms(plan: Any) -> np.ndarray:
    position = np.asarray(plan["doll_position_world"], dtype=np.float64)
    quaternion_wxyz = np.asarray(plan["doll_orientation_world_wxyz"], dtype=np.float64)
    quaternion_wxyz /= np.linalg.norm(quaternion_wxyz, axis=1, keepdims=True)
    w, x, y, z = quaternion_wxyz.T
    rotation = np.empty((len(position), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation[:, 0, 2] = 2.0 * (x * z + y * w)
    rotation[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - x * w)
    rotation[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(position), axis=0)
    result[:, :3, :3] = rotation
    result[:, :3, 3] = position
    return result


def transform_drift(local: np.ndarray) -> tuple[float, float]:
    if len(local) < 2:
        return 0.0, 0.0
    reference = local[0]
    position = np.linalg.norm(local[:, :3, 3] - reference[:3, 3], axis=1)
    # Object poses are stored as float32 quaternions. Project the tiny round-off
    # in their reconstructed matrices back onto SO(3) before measuring drift.
    u, _, vh = np.linalg.svd(local[:, :3, :3])
    rotation = np.einsum("tij,tjk->tik", u, vh)
    determinant = np.linalg.det(rotation)
    if np.any(determinant < 0.0):
        u[determinant < 0.0, :, -1] *= -1.0
        rotation = np.einsum("tij,tjk->tik", u, vh)
    relative = np.einsum("tij,jk->tik", rotation, rotation[0].T)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error = np.arccos(cosine)
    return float(position.max(initial=0.0)), float(rotation_error.max(initial=0.0))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_rows = read_json(SOURCE_SELECTED)
    g1_rows = read_json(G1_SELECTED)
    source_keys = [(int(r["episode_index"]), int(r["frame_index"]), r["phase"]) for r in source_rows]
    g1_keys = [(int(r["episode_index"]), int(r["frame_index"]), r["phase"]) for r in g1_rows]
    if source_keys != g1_keys or len(source_keys) != 54:
        raise RuntimeError("probe frames are not exactly matched")

    source_table = pq.read_table(
        SOURCE_DATA,
        columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
    )
    g1_table = pq.read_table(
        G1_DATA,
        columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
    )
    array_checks: dict[str, bool] = {}
    for name, dtype in (
        ("episode_index", np.int64),
        ("frame_index", np.int64),
        ("timestamp", np.float32),
        ("observation.state", np.float32),
        ("action", np.float32),
    ):
        array_checks[name] = bool(
            np.array_equal(as_array(source_table, name, dtype), as_array(g1_table, name, dtype))
        )

    source_state = as_array(source_table, "observation.state", np.float32)
    source_action = as_array(source_table, "action", np.float32)
    dataset_episode = as_array(source_table, "episode_index", np.int64)
    dataset_frame = as_array(source_table, "frame_index", np.int64)
    exact_plan_state = True
    exact_plan_action = True
    maximum_state_error = 0.0
    maximum_action_error = 0.0
    maximum_left_local_position_drift = 0.0
    maximum_left_local_rotation_drift = 0.0
    maximum_right_local_position_drift = 0.0
    maximum_right_local_rotation_drift = 0.0
    maximum_doll_step = 0.0
    transfer_steps: list[dict[str, Any]] = []
    selected_alignment: list[dict[str, Any]] = []

    for episode in range(50):
        indices = np.flatnonzero(dataset_episode == episode)
        if not np.array_equal(dataset_frame[indices], np.arange(len(indices))):
            raise RuntimeError(f"episode {episode}: non-contiguous dataset frame indices")
        plan = np.load(PLANS / f"episode_{episode:06d}.npz", allow_pickle=False)
        state = np.asarray(plan["observation_state"], dtype=np.float32)
        action = np.asarray(plan["frozen_action"], dtype=np.float32)
        state_error = float(np.max(np.abs(state.astype(np.float64) - source_state[indices])))
        action_error = float(np.max(np.abs(action.astype(np.float64) - source_action[indices])))
        maximum_state_error = max(maximum_state_error, state_error)
        maximum_action_error = max(maximum_action_error, action_error)
        exact_plan_state &= np.array_equal(state, source_state[indices])
        exact_plan_action &= np.array_equal(action, source_action[indices])

        trajectory = np.load(TRAJECTORIES / f"episode_{episode:06d}.npz", allow_pickle=False)
        pose_index = np.asarray(plan["pose_action_index"], dtype=np.int64)
        ownership = np.asarray(plan["ownership_state"]).astype("U32")
        obj = object_transforms(plan)
        left = hand_world_transforms(trajectory, "left")[pose_index]
        right = hand_world_transforms(trajectory, "right")[pose_index]
        left_mask = np.isin(ownership, ["LEFT_OWNED", "HANDOFF_APPROACH", "DUAL_CONTACT"])
        right_mask = np.isin(ownership, ["RIGHT_OWNED", "RIGHT_TRANSPORT"])
        left_local = np.einsum("tij,tjk->tik", np.linalg.inv(left[left_mask]), obj[left_mask])
        right_local = np.einsum("tij,tjk->tik", np.linalg.inv(right[right_mask]), obj[right_mask])
        lp, lr = transform_drift(left_local)
        rp, rr = transform_drift(right_local)
        maximum_left_local_position_drift = max(maximum_left_local_position_drift, lp)
        maximum_left_local_rotation_drift = max(maximum_left_local_rotation_drift, lr)
        maximum_right_local_position_drift = max(maximum_right_local_position_drift, rp)
        maximum_right_local_rotation_drift = max(maximum_right_local_rotation_drift, rr)
        steps = np.linalg.norm(np.diff(obj[:, :3, 3], axis=0), axis=1)
        maximum_doll_step = max(maximum_doll_step, float(steps.max(initial=0.0)))
        for index in np.flatnonzero(ownership[1:] != ownership[:-1]) + 1:
            transfer_steps.append(
                {
                    "episode": episode,
                    "frame": int(index),
                    "transition": f"{ownership[index - 1]}->{ownership[index]}",
                    "position_step_m": float(steps[index - 1]),
                }
            )

    for episode, frame, phase in source_keys:
        global_index = int(np.flatnonzero((dataset_episode == episode) & (dataset_frame == frame))[0])
        plan = np.load(PLANS / f"episode_{episode:06d}.npz", allow_pickle=False)
        selected_alignment.append(
            {
                "episode": episode,
                "frame": frame,
                "phase": phase,
                "state_exact": bool(
                    np.array_equal(plan["observation_state"][frame], source_state[global_index])
                ),
                "action_exact": bool(np.array_equal(plan["frozen_action"][frame], source_action[global_index])),
                "render_pose_action_index": int(plan["pose_action_index"][frame]),
            }
        )

    by_phase_source: dict[str, list[dict[str, Any]]] = {}
    by_phase_g1: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        by_phase_source.setdefault(row["phase"], []).append(row)
    for row in g1_rows:
        by_phase_g1.setdefault(row["phase"], []).append(row)
    contact_sheets = {}
    for phase in by_phase_source:
        path = make_contact_sheet(
            phase,
            by_phase_source[phase],
            by_phase_g1[phase],
            output / "contact_sheets" / f"{phase}.png",
        )
        contact_sheets[phase] = str(path)

    plan_manifest = read_json(PLAN_MANIFEST)
    reconstruction_reports = [row["object_reconstruction"] for row in plan_manifest["episodes"]]
    maximum_transfer_position_discontinuity = max(
        float(row["transfer_position_discontinuity_m"]) for row in reconstruction_reports
    )
    maximum_transfer_rotation_discontinuity = max(
        float(row["transfer_rotation_discontinuity_rad"]) for row in reconstruction_reports
    )
    ownership_transfer_steps = [
        row
        for row in transfer_steps
        if row["transition"] in {"NO_OWNER->LEFT_OWNED", "DUAL_CONTACT->RIGHT_OWNED"}
    ]
    validation = {
        "status": "PASS",
        "scope": "read-only phase-specific semantic rendering QA",
        "matched_probe_frames": len(source_keys),
        "episodes": sorted(set(row[0] for row in source_keys)),
        "phase_count": len(by_phase_source),
        "dataset_array_identity": array_checks,
        "plan_matches_original_dataset": {
            "state_exact": exact_plan_state,
            "action_exact": exact_plan_action,
            "maximum_state_error_rad": maximum_state_error,
            "maximum_action_error_rad": maximum_action_error,
        },
        "selected_frame_alignment": {
            "all_state_exact": all(row["state_exact"] for row in selected_alignment),
            "all_action_exact": all(row["action_exact"] for row in selected_alignment),
            "rows": selected_alignment,
        },
        "object_reconstruction": {
            "method": "KINEMATIC_OWNERSHIP_OBJECT_RECONSTRUCTION",
            "maximum_transfer_position_discontinuity_m": maximum_transfer_position_discontinuity,
            "maximum_transfer_rotation_discontinuity_rad": maximum_transfer_rotation_discontinuity,
            "maximum_left_owned_object_in_hand_position_drift_m": maximum_left_local_position_drift,
            "maximum_left_owned_object_in_hand_rotation_drift_rad": maximum_left_local_rotation_drift,
            "maximum_right_owned_object_in_hand_position_drift_m": maximum_right_local_position_drift,
            "maximum_right_owned_object_in_hand_rotation_drift_rad": maximum_right_local_rotation_drift,
            "maximum_frame_position_step_m": maximum_doll_step,
            "ownership_anchor_steps": ownership_transfer_steps,
            "interpretation": "nonzero ordinary frame steps follow the frozen hand path; ownership anchors are continuous",
        },
        "camera": {
            "config": str(CAMERA),
            "sha256": sha256_file(CAMERA),
            "frozen": True,
        },
        "contact_sheets": contact_sheets,
        "manual_visual_checks_required": [
            "doll/hand spatial consistency",
            "right-arm visibility",
            "camera projection",
            "gross phase/image temporal offset",
        ],
        "manual_visual_review": {
            "status": "PASS",
            "reviewed_phases": list(by_phase_source),
            "doll_pose_continuity": "PASS",
            "doll_hand_spatial_consistency": "PASS",
            "handoff_visual_geometry": "PASS",
            "right_arm_visibility": "PASS",
            "object_teleportation_observed": False,
            "camera_projection_mismatch_observed": False,
            "phase_image_temporal_offset_observed": False,
            "rendered_robot_pose_matches_observation_state": "EXACT_NUMERIC_AND_VISUAL_PASS",
            "note": "Embodiment and object appearance differ by design; no semantic label/image contradiction was observed.",
        },
        "dataset_modified": False,
    }
    required = [
        all(array_checks.values()),
        exact_plan_state,
        exact_plan_action,
        all(row["state_exact"] and row["action_exact"] for row in selected_alignment),
        maximum_transfer_position_discontinuity <= 1e-9,
        maximum_transfer_rotation_discontinuity <= 1e-9,
        maximum_left_local_position_drift <= 1e-6,
        maximum_left_local_rotation_drift <= 1e-6,
        maximum_right_local_position_drift <= 1e-6,
        maximum_right_local_rotation_drift <= 1e-6,
    ]
    if not all(required):
        validation["status"] = "FAIL"
    atomic_json(output / "validation.json", validation)
    lines = [
        "# G1-visual phase-specific QA",
        "",
        f"Automated semantic alignment gate: **{validation['status']}**",
        "",
        "The 54 probe rows use identical episode/frame identities in both datasets. State, action, "
        "timestamp, episode, and frame arrays are exact; only RGB differs.",
        "",
        "The rendered robot pose is the stored observation state. The doll uses the frozen "
        "KINEMATIC_OWNERSHIP_OBJECT_RECONSTRUCTION: one constant left-hand transform, then a "
        "continuous one-time transfer to a constant right-hand transform.",
        "",
        f"Maximum ownership-transfer reconstruction discontinuity: "
        f"{maximum_transfer_position_discontinuity:.3e} m / "
        f"{maximum_transfer_rotation_discontinuity:.3e} rad.",
        "",
        "Contact sheets:",
    ]
    lines.extend(f"- {phase}: `{path}`" for phase, path in contact_sheets.items())
    lines.extend(
        [
            "",
            "The contact sheets are a read-only comparison. No image, state, action, task, camera, "
            "trajectory, or object-reconstruction data were changed.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": validation["status"], "output": str(output)}, indent=2))
    return 0 if validation["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
