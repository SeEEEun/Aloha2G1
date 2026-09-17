#!/usr/bin/env python3
"""Thin Isaac-state adapter for :mod:`common_execution_layer`.

The adapter reads measured rigid-body poses and contact forces.  It does not
write simulator state and exposes no arm/wrist target API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.common_execution_layer import (
    CommonDex3ExecutionLayer,
    Dex3Primitive,
    ExecutionDecision,
    ExecutionSnapshot,
    load_common_grasp_intent,
    load_frozen_evaluator,
    pose_matrix,
    read_json,
    sha256_file,
    whole_hand_pose_from_pad_centers,
)


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PHYSICS_CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
PHYSICAL_ENVIRONMENT = (
    ROOT
    / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
)
COMMON_PHYSICAL_CONTROLLER = (
    ROOT
    / "outputs/final_contact_constrained_eval/03_freeze/FINAL_COMMON_EXECUTION_CONTROLLER.json"
)
WHOLE_HAND_GEOMETRY = (
    ROOT / "configs/doll_handoff_retargeting/dex3_whole_hand.sim.json"
)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_execution_freeze(
    manifest_path: Path, evaluator_sha256: str
) -> dict[str, Any]:
    manifest = read_json(manifest_path.resolve())
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("common execution layer is not FROZEN")
    if manifest.get("frozen_evaluator_sha256") != evaluator_sha256:
        raise RuntimeError("execution/evaluator freeze SHA256 binding mismatch")
    files = manifest.get("implementation_files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("execution freeze has no implementation file list")
    for row in files:
        path = Path(row["path"]).resolve()
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"common execution implementation hash drift: {path}")
    return manifest


class IsaacCommonExecutionRuntime:
    EVENT_FIELDS = (
        "RAW_POLICY_COMMAND",
        "POLICY_SAFE_COMMAND",
        "EXECUTED_COMMAND",
        "MEASURED_Q",
        "COMMON_CONTROLLER_OVERRIDE_MASK",
        "COMMON_CONTROLLER_OVERRIDE_MASK_ARM",
        "COMMON_CONTROLLER_OVERRIDE_MASK_WRIST",
        "COMMON_CONTROLLER_OVERRIDE_MASK_DEX3",
        "COMMON_EXECUTION_PHASE",
        "FROZEN_GRASPABILITY_MARGIN",
        "FROZEN_GRASPABILITY_ELIGIBLE",
        "COMMON_EXECUTION_EVENTS",
    )

    def __init__(
        self,
        controller: CommonDex3ExecutionLayer,
        joint_names: Sequence[str],
        whole_hand_geometry: Mapping[str, Any],
        evaluator_manifest_sha256: str,
        execution_manifest_sha256: str,
    ) -> None:
        self.controller = controller
        self.joint_names = tuple(str(value) for value in joint_names)
        self.geometry = whole_hand_geometry
        self.evaluator_manifest_sha256 = evaluator_manifest_sha256
        self.execution_manifest_sha256 = execution_manifest_sha256
        self.current_decision: ExecutionDecision | None = None
        self.current_frame = -1

    @property
    def event_field_names(self) -> tuple[str, ...]:
        return self.EVENT_FIELDS

    def snapshot(
        self,
        measured_q_rad: np.ndarray,
        object_pose_xyzw: np.ndarray,
        body_names: Sequence[str],
        body_positions_world_m: np.ndarray,
        body_quaternions_xyzw: np.ndarray,
        records: Mapping[str, list[Any]],
    ) -> ExecutionSnapshot:
        lookup = {str(name): index for index, name in enumerate(body_names)}
        whole: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            centers: dict[str, np.ndarray] = {}
            for role in ("A", "B", "C"):
                row = self.geometry[side][role]
                digit = str(row["digit_chain"])
                link = str(row["distal_link"])
                if link not in lookup:
                    raise RuntimeError(f"Isaac body mapping lacks {link}")
                index = lookup[link]
                body = pose_matrix(
                    body_positions_world_m[index], body_quaternions_xyzw[index]
                )
                local = np.asarray(row["local_position_xyz_m"], dtype=np.float64)
                centers[digit] = body[:3, 3] + body[:3, :3] @ local
            whole[side] = whole_hand_pose_from_pad_centers(centers)
        pose = np.asarray(object_pose_xyzw, dtype=np.float64)
        if pose.shape != (7,):
            raise RuntimeError("invalid Isaac object pose")
        object_world = pose_matrix(pose[:3], pose[3:7])

        def latest(name: str) -> float:
            values = records.get(name, [])
            return float(values[-1]) if values else 0.0

        force = {
            side: {
                digit: latest(f"{side}_{digit}_force_n")
                for digit in ("thumb", "index", "middle")
            }
            for side in ("left", "right")
        }
        previous_support: dict[str, bool] | None = None
        control_frames = records.get("control_frame", [])
        if control_frames:
            prior_frame = int(control_frames[-1])
            start = len(control_frames) - 1
            while start > 0 and int(control_frames[start - 1]) == prior_frame:
                start -= 1

            def all_force(side: str, minimum_digits: int) -> bool:
                values = np.column_stack(
                    [
                        np.asarray(records[f"{side}_{digit}_force_n"][start:], dtype=np.float64)
                        for digit in ("thumb", "index", "middle")
                    ]
                )
                meaningful = np.count_nonzero(
                    values >= self.controller.primitive.force_threshold_n, axis=1
                )
                table = np.asarray(
                    records["table_contact_force_n"][start:], dtype=np.float64
                )
                return bool(
                    len(values)
                    and np.all(meaningful >= minimum_digits)
                    and np.all(
                        table <= self.controller.primitive.maximum_table_force_n
                    )
                )

            previous_support = {
                "left_two_table_free": all_force("left", 2),
                "left_three_table_free": all_force("left", 3),
                "right_three_table_free": all_force("right", 3),
                "right_two_table_free": all_force("right", 2),
            }
        return ExecutionSnapshot(
            measured_q_rad=np.asarray(measured_q_rad, dtype=np.float64),
            object_world=object_world,
            whole_hand_world=whole,
            digit_force_n=force,
            table_force_n=latest("table_contact_force_n"),
            previous_control_frame_support=previous_support,
        )

    def step(self, frame: int, snapshot: ExecutionSnapshot) -> np.ndarray:
        self.current_frame = int(frame)
        self.current_decision = self.controller.step(frame, snapshot)
        return self.current_decision.executed_command

    def event_values(self, measured_q_rad: np.ndarray) -> dict[str, Any]:
        if self.current_decision is None or self.current_frame < 0:
            raise RuntimeError("common execution decision missing")
        row = self.current_decision
        frame = self.current_frame
        return {
            "RAW_POLICY_COMMAND": self.controller.raw[frame],
            "POLICY_SAFE_COMMAND": self.controller.safe[frame],
            "EXECUTED_COMMAND": row.executed_command,
            "MEASURED_Q": np.asarray(measured_q_rad, dtype=np.float64),
            "COMMON_CONTROLLER_OVERRIDE_MASK": row.override_mask,
            "COMMON_CONTROLLER_OVERRIDE_MASK_ARM": row.arm_override_mask,
            "COMMON_CONTROLLER_OVERRIDE_MASK_WRIST": row.wrist_override_mask,
            "COMMON_CONTROLLER_OVERRIDE_MASK_DEX3": row.dex3_override_mask,
            "COMMON_EXECUTION_PHASE": row.phase,
            "FROZEN_GRASPABILITY_MARGIN": row.graspability_margin,
            "FROZEN_GRASPABILITY_ELIGIBLE": row.left_envelope_eligible,
            "COMMON_EXECUTION_EVENTS": "|".join(row.events),
        }

    def summary(self) -> dict[str, Any]:
        result = self.controller.summary()
        result.update(
            {
                "evaluator_manifest_sha256": self.evaluator_manifest_sha256,
                "execution_manifest_sha256": self.execution_manifest_sha256,
                "raw_policy_command_logged": True,
                "executed_command_logged": True,
                "measured_q_logged": True,
                "component_and_phase_override_masks_logged": True,
            }
        )
        return result

    def write_summary(self, path: Path) -> None:
        _atomic_json(path, self.summary())


def build_runtime(
    command_path: Path,
    policy_safe_command: np.ndarray,
    joint_names: Sequence[str],
) -> IsaacCommonExecutionRuntime:
    evaluator_manifest_value = os.environ.get("COMMON_EXEC_EVALUATOR_MANIFEST")
    execution_manifest_value = os.environ.get("COMMON_EXEC_FREEZE_MANIFEST")
    if not evaluator_manifest_value or not execution_manifest_value:
        raise RuntimeError("common execution frozen manifest environment is missing")
    evaluator = load_frozen_evaluator(Path(evaluator_manifest_value))
    execution_manifest_path = Path(execution_manifest_value).resolve()
    verify_execution_freeze(execution_manifest_path, evaluator.evaluator_sha256)
    with np.load(command_path.resolve(), allow_pickle=False) as archive:
        required = {
            "raw_policy_command",
            "stable_episode_id",
            "method",
            "commanded_q_rad",
            "joint_names",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"physical ACT command lacks: {sorted(missing)}")
        raw = np.asarray(archive["raw_policy_command"], dtype=np.float64)
        stable_episode_id = str(np.asarray(archive["stable_episode_id"]).item())
        method_code = str(np.asarray(archive["method"]).item()).lower()
        archived_safe = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        archived_names = archive["joint_names"].astype(str).tolist()
    safe = np.asarray(policy_safe_command, dtype=np.float64)
    if not np.array_equal(archived_safe, safe):
        raise RuntimeError("loaded engine command differs from frozen policy-safe command")
    if archived_names != list(joint_names):
        raise RuntimeError("common execution named joint order mismatch")
    if method_code not in {"a", "b"}:
        raise RuntimeError("unknown frozen ACT method identity")
    intent = load_common_grasp_intent(
        evaluator.common_intent_artifact, stable_episode_id, len(safe)
    )
    primitive = Dex3Primitive.from_frozen_dependencies(
        read_json(PHYSICS_CONFIG),
        read_json(PHYSICAL_ENVIRONMENT),
        read_json(COMMON_PHYSICAL_CONTROLLER),
    )
    controller = CommonDex3ExecutionLayer(
        evaluator.envelope,
        primitive,
        intent,
        raw,
        safe,
        f"ACT-{method_code.upper()}40",
    )
    return IsaacCommonExecutionRuntime(
        controller=controller,
        joint_names=joint_names,
        whole_hand_geometry=read_json(WHOLE_HAND_GEOMETRY),
        evaluator_manifest_sha256=evaluator.manifest_sha256,
        execution_manifest_sha256=sha256_file(execution_manifest_path),
    )


__all__ = ["IsaacCommonExecutionRuntime", "build_runtime", "verify_execution_freeze"]
