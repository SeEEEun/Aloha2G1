#!/usr/bin/env python3
"""Classify saved Doll-Handoff smoke contacts from active MuJoCo geometry.

This is a read-only replay audit.  It does not solve IK, alter Cartesian targets,
move scene geometry, package data, or train a policy.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    atomic_csv,
    load_common_config,
    load_scene,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402


AUDIT_ROOT = REPOSITORY / "outputs/doll_handoff_retargeting/natural_arm_audit"
DEFAULT_INPUT = AUDIT_ROOT / "final_after_common_natural_arm_candidate"
DEFAULT_CSV = AUDIT_ROOT / "final_contact_classification.csv"
DEFAULT_MARKDOWN = AUDIT_ROOT / "final_contact_classification.md"
EPISODES = (0, 24, 49)


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, identifier: int) -> str:
    return mujoco.mj_id2name(model, object_type, int(identifier)) or (
        f"{object_type.name.lower()}_{identifier}"
    )


def _is_finger(body: str) -> bool:
    return "hand_" in body and any(
        token in body for token in ("thumb", "index", "middle")
    )


def _is_palm(body: str) -> bool:
    return "hand_palm" in body or "wrist_yaw" in body


def _is_torso(body: str) -> bool:
    return any(token in body for token in ("torso", "pelvis", "waist"))


def _is_arm(body: str) -> bool:
    return any(
        token in body
        for token in (
            "shoulder",
            "upper_arm",
            "elbow",
            "forearm",
            "wrist",
        )
    )


def classify(body_1: str, body_2: str) -> tuple[str, str]:
    """Return the requested physics class and an explicit disposition."""
    bodies = (body_1, body_2)
    sides = {
        side
        for side in ("left", "right")
        if any(body.startswith(f"{side}_") for body in bodies)
    }
    if any(_is_torso(body) for body in bodies) and any(_is_arm(body) for body in bodies):
        return "ARM_TORSO_INVALID", "INVALID_SELF_PENETRATION"
    if len(sides) == 2 and all(_is_finger(body) for body in bodies):
        return "FINGER_FINGER_CONTACT", "DISTAL_SELF_CONTACT_REQUIRES_REVIEW"
    if len(sides) == 2 and any(_is_finger(body) for body in bodies) and any(
        _is_palm(body) for body in bodies
    ):
        return "FINGER_PALM_CONTACT", "DISTAL_SELF_CONTACT_REQUIRES_REVIEW"
    if len(sides) == 2 and all(_is_palm(body) for body in bodies):
        return "PALM_PALM_INVALID", "INVALID_SELF_PENETRATION"
    if len(sides) == 2:
        return "CROSS_ARM_INVALID", "INVALID_SELF_PENETRATION"
    return "OTHER_INVALID", "INVALID_SELF_PENETRATION"


def _vector_fields(prefix: str, value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        f"{prefix}_x": float(value[0]),
        f"{prefix}_y": float(value[1]),
        f"{prefix}_z": float(value[2]),
    }


def _rows(input_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common_path = input_root / "config/common_config.json"
    common = load_common_config(common_path)
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    tolerance = float(common["validation"]["collision_penetration_tolerance_m"])
    world_rotation = np.asarray(g1.root_pose[:3, :3], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    frame_counts: dict[int, set[int]] = {episode: set() for episode in EPISODES}

    for episode in EPISODES:
        stable_id = f"doll_handoff_20260820_ep{episode:03d}"
        trajectory = input_root / "proposed/trajectories" / f"{stable_id}.npz"
        with np.load(trajectory, allow_pickle=False) as saved:
            values = {key: np.asarray(saved[key]) for key in saved.files}
        for frame, (arm, left_hand, right_hand) in enumerate(
            zip(
                values["g1_arm_qpos"],
                values["left_dex3_qpos"],
                values["right_dex3_qpos"],
            )
        ):
            g1.assign(arm, left_hand, right_hand)
            for contact_index, contact in enumerate(g1.data.contact):
                distance = float(contact.dist)
                if distance >= -abs(tolerance):
                    continue
                geom_ids = (int(contact.geom1), int(contact.geom2))
                body_ids = tuple(int(g1.model.geom_bodyid[value]) for value in geom_ids)
                if body_ids[0] == body_ids[1]:
                    continue
                geom_names = tuple(
                    _name(g1.model, mujoco.mjtObj.mjOBJ_GEOM, value)
                    for value in geom_ids
                )
                body_names = tuple(
                    _name(g1.model, mujoco.mjtObj.mjOBJ_BODY, value)
                    for value in body_ids
                )
                classification, disposition = classify(*body_names)

                # Internal same-side hand contacts are not part of the residual
                # cross-hand/body collision set used by the pipeline.
                sides = {
                    side
                    for side in ("left", "right")
                    if any(name.startswith(f"{side}_") for name in body_names)
                }
                if len(sides) == 1 and all(
                    "hand_" in name or "wrist" in name for name in body_names
                ):
                    continue
                torso = any(_is_torso(name) for name in body_names)
                arm = any(_is_arm(name) for name in body_names)
                if len(sides) != 2 and not (torso and arm):
                    continue

                position_model = np.asarray(contact.pos, dtype=np.float64).copy()
                normal_model = np.asarray(contact.frame[:3], dtype=np.float64).copy()
                position_world = g1.model_to_world_position(position_model)
                normal_world = world_rotation @ normal_model
                ownership = str(values["ownership_state"][frame])
                handoff_context = ownership in {
                    "HANDOFF_APPROACH",
                    "DUAL_CONTACT",
                    "RIGHT_OWNED",
                }
                row: dict[str, Any] = {
                    "episode_index": episode,
                    "stable_episode_id": stable_id,
                    "source_directory_name": str(values["source_directory_name"]),
                    "frame": frame,
                    "source_frame_index": int(values["source_frame_index"][frame]),
                    "timestamp_s": float(values["timestamp"][frame]),
                    "ownership_state": ownership,
                    "left_hand_phase": str(values["left_hand_phase"][frame]),
                    "right_hand_phase": str(values["right_hand_phase"][frame]),
                    "handoff_context": handoff_context,
                    "classification": classification,
                    "disposition": disposition,
                    "body_1": body_names[0],
                    "body_2": body_names[1],
                    "geom_1": geom_names[0],
                    "geom_2": geom_names[1],
                    "mujoco_contact_index": contact_index,
                    "signed_contact_distance_m": distance,
                    "penetration_depth_m": -distance,
                    **_vector_fields("contact_position_model_m", position_model),
                    **_vector_fields("contact_position_world_m", position_world),
                    **_vector_fields("contact_normal_model", normal_model),
                    **_vector_fields("contact_normal_world", normal_world),
                }
                rows.append(row)
                frame_counts[episode].add(frame)

    class_records = collections.Counter(row["classification"] for row in rows)
    class_frames = {
        name: len(
            {
                (int(row["episode_index"]), int(row["frame"]))
                for row in rows
                if row["classification"] == name
            }
        )
        for name in sorted(class_records)
    }
    summary = {
        "input_root": str(input_root),
        "collision_penetration_tolerance_m": tolerance,
        "episodes": {
            f"ep{episode:03d}": {
                "contact_records": sum(
                    int(row["episode_index"]) == episode for row in rows
                ),
                "contact_frames": len(frame_counts[episode]),
                "maximum_penetration_depth_m": max(
                    (
                        float(row["penetration_depth_m"])
                        for row in rows
                        if int(row["episode_index"]) == episode
                    ),
                    default=0.0,
                ),
            }
            for episode in EPISODES
        },
        "classification_record_counts": dict(class_records),
        "classification_frame_counts": class_frames,
        "invalid_proximal_or_palm_frame_count": len(
            {
                (int(row["episode_index"]), int(row["frame"]))
                for row in rows
                if row["classification"]
                in {
                    "PALM_PALM_INVALID",
                    "ARM_TORSO_INVALID",
                    "CROSS_ARM_INVALID",
                    "OTHER_INVALID",
                }
            }
        ),
        "distal_hand_contact_review_frame_count": len(
            {
                (int(row["episode_index"]), int(row["frame"]))
                for row in rows
                if row["classification"]
                in {"FINGER_FINGER_CONTACT", "FINGER_PALM_CONTACT"}
            }
        ),
        "benign_proximity_reclassified_from_penetration_count": 0,
        "interpretation": (
            "Every listed row is a MuJoCo contact deeper than the configured "
            "penetration tolerance. Handoff context is reported but does not turn "
            "penetration into benign proximity. Distal finger contact is kept "
            "separate from catastrophic palm/proximal-arm/body penetration."
        ),
    }
    return rows, summary


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Final natural-arm contact classification",
        "",
        "This is a read-only replay of the saved Proposed-B natural-arm candidate "
        "against the active G1 + Dex3 MuJoCo collision model.",
        "",
        f"- Input: `{summary['input_root']}`",
        f"- Penetration tolerance: `{summary['collision_penetration_tolerance_m']:.9f} m`",
        "- Cartesian targets changed: **NO**",
        "- Episode/phase-specific correction used: **NO**",
        "- Benign proximity inferred from penetrating contacts: **NO**",
        "",
        "## Per-episode summary",
        "",
        "| episode | contact records | unique contact frames | max penetration |",
        "|---|---:|---:|---:|",
    ]
    for episode, value in summary["episodes"].items():
        lines.append(
            f"| {episode} | {value['contact_records']} | {value['contact_frames']} | "
            f"{1000.0 * value['maximum_penetration_depth_m']:.3f} mm |"
        )
    lines.extend(
        [
            "",
            "## Classification summary",
            "",
            "| classification | contact records | unique frames |",
            "|---|---:|---:|",
        ]
    )
    for name, count in summary["classification_record_counts"].items():
        lines.append(
            f"| {name} | {count} | "
            f"{summary['classification_frame_counts'][name]} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "`FINGER_FINGER_CONTACT` and `FINGER_PALM_CONTACT` are retained as "
            "distal handoff-contact review items. They are not conflated with "
            "`PALM_PALM_INVALID`, `CROSS_ARM_INVALID`, or `ARM_TORSO_INVALID`. "
            "Conversely, a handoff phase label does not make a positive penetration "
            "depth physically benign.",
            "",
            "The CSV contains one row per MuJoCo contact point, including body/link "
            "pair, geom pair, frame, ownership/hand phase, model/world contact "
            "position, penetration depth, and the MuJoCo contact-frame normal.",
            "",
            "## Residual records",
            "",
            "| ep | frame | ownership | classification | link pair | depth mm | position world m |",
            "|---:|---:|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        pair = f"{row['body_1']} ↔ {row['body_2']}"
        position = (
            f"({row['contact_position_world_m_x']:.4f}, "
            f"{row['contact_position_world_m_y']:.4f}, "
            f"{row['contact_position_world_m_z']:.4f})"
        )
        lines.append(
            f"| {row['episode_index']:03d} | {row['frame']} | "
            f"{row['ownership_state']} | {row['classification']} | {pair} | "
            f"{1000.0 * row['penetration_depth_m']:.3f} | {position} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    rows, summary = _rows(args.input_root.resolve())
    if rows:
        atomic_csv(args.csv.resolve(), rows)
    else:
        atomic_csv(
            args.csv.resolve(),
            [],
            fieldnames=("episode_index", "frame", "classification"),
        )
    args.markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.markdown.resolve().write_text(_markdown(rows, summary), encoding="utf-8")
    summary_path = args.markdown.resolve().with_suffix(".json")
    temporary = summary_path.with_suffix(summary_path.suffix + ".incomplete")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
