#!/usr/bin/env python3
"""Isaac measured-state adapter for classifier-free direct physical EVAL35."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.common_execution_isaac_runtime import (
    COMMON_PHYSICAL_CONTROLLER,
    PHYSICAL_ENVIRONMENT,
    PHYSICS_CONFIG,
    WHOLE_HAND_GEOMETRY,
    IsaacCommonExecutionRuntime,
)
from tools.common_execution_layer import Dex3Primitive, read_json, sha256_file
from tools.direct_physical_execution_layer import (
    DirectPhysicalDex3ExecutionLayer,
    authoritative_joint_limits,
)


JOINT_CONTRACT = Path(
    "/home/jbnu/aloha_g1_dataset/outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
)


def verify_direct_freeze(path: Path) -> dict[str, Any]:
    manifest = read_json(path.resolve())
    qualification = os.environ.get("DIRECT_EXECUTION_QUALIFICATION_MODE") == "1"
    allowed = (
        {"PRE_EVAL35_QUALIFICATION_PROVISIONAL"}
        if qualification
        else {"FROZEN_BEFORE_EVAL35"}
    )
    if manifest.get("status") not in allowed:
        raise RuntimeError("direct physical EVAL35 layer is not frozen")
    if manifest.get("graspability_classifier_used") is not False:
        raise RuntimeError("direct freeze unexpectedly enables a classifier")
    for row in manifest.get("files", []):
        dependency = Path(row["path"]).resolve()
        if not dependency.is_file() or sha256_file(dependency) != row["sha256"]:
            raise RuntimeError(f"direct EVAL35 frozen dependency drift: {dependency}")
    return manifest


class DirectIsaacExecutionRuntime(IsaacCommonExecutionRuntime):
    EVENT_FIELDS = (
        "RAW_POLICY_COMMAND",
        "POLICY_SAFE_COMMAND",
        "PROJECTED_POLICY_COMMAND",
        "EXECUTED_COMMAND",
        "MEASURED_Q",
        "COMMON_CONTROLLER_OVERRIDE_MASK",
        "COMMON_CONTROLLER_OVERRIDE_MASK_ARM",
        "COMMON_CONTROLLER_OVERRIDE_MASK_WRIST",
        "COMMON_CONTROLLER_OVERRIDE_MASK_DEX3",
        "COMMON_ARM_HARD_LIMIT_PROJECTION_MASK",
        "DIRECT_COMMON_EXECUTION_PHASE",
        "DIRECT_COMMON_TASK_INTENT",
        "DIRECT_COMMON_EXECUTION_EVENTS",
    )

    def __init__(
        self,
        *args: Any,
        direct_freeze_sha256: str,
        initial_q_rad: np.ndarray,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.direct_freeze_sha256 = direct_freeze_sha256
        self.initial_q_rad = np.asarray(initial_q_rad, dtype=np.float64)
        if self.initial_q_rad.shape != (28,) or not np.isfinite(self.initial_q_rad).all():
            raise RuntimeError("invalid common physical initial q")

    def event_values(self, measured_q_rad: np.ndarray) -> dict[str, Any]:
        if self.current_decision is None or self.current_frame < 0:
            raise RuntimeError("direct execution decision missing")
        row = self.current_decision
        frame = self.current_frame
        projected_policy = self.controller.safe[frame].copy()
        projected_policy[:14] = np.clip(
            projected_policy[:14],
            self.controller.arm_hard_limit_projector.lower_rad,
            self.controller.arm_hard_limit_projector.upper_rad,
        )
        return {
            "RAW_POLICY_COMMAND": self.controller.raw[frame],
            "POLICY_SAFE_COMMAND": self.controller.safe[frame],
            "PROJECTED_POLICY_COMMAND": projected_policy,
            "EXECUTED_COMMAND": row.executed_command,
            "MEASURED_Q": np.asarray(measured_q_rad, dtype=np.float64),
            "COMMON_CONTROLLER_OVERRIDE_MASK": row.override_mask,
            "COMMON_CONTROLLER_OVERRIDE_MASK_ARM": row.arm_override_mask,
            "COMMON_CONTROLLER_OVERRIDE_MASK_WRIST": row.wrist_override_mask,
            "COMMON_CONTROLLER_OVERRIDE_MASK_DEX3": row.dex3_override_mask,
            "COMMON_ARM_HARD_LIMIT_PROJECTION_MASK": (
                self.controller.arm_hard_limit_projection_masks[frame]
            ),
            "DIRECT_COMMON_EXECUTION_PHASE": row.phase,
            "DIRECT_COMMON_TASK_INTENT": self.controller.intent[frame],
            "DIRECT_COMMON_EXECUTION_EVENTS": "|".join(row.events),
        }

    def summary(self) -> dict[str, Any]:
        result = self.controller.summary()
        result.update(
            {
                "direct_execution_freeze_sha256": self.direct_freeze_sha256,
                "raw_policy_command_logged": True,
                "projected_policy_command_logged": True,
                "executed_command_logged": True,
                "measured_q_logged": True,
                "component_override_masks_logged": True,
            }
        )
        return result


def build_runtime(
    command_path: Path,
    policy_safe_command: np.ndarray,
    joint_names: Sequence[str],
) -> DirectIsaacExecutionRuntime:
    freeze_value = os.environ.get("DIRECT_EVAL35_FREEZE_MANIFEST")
    if not freeze_value:
        raise RuntimeError("direct EVAL35 freeze manifest environment is missing")
    freeze_path = Path(freeze_value).resolve()
    verify_direct_freeze(freeze_path)
    with np.load(command_path.resolve(), allow_pickle=False) as archive:
        required = {
            "raw_policy_command",
            "stable_episode_id",
            "method",
            "commanded_q_rad",
            "joint_names",
            "common_task_intent",
            "common_initial_q_rad",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"direct physical ACT command lacks: {sorted(missing)}")
        raw = np.asarray(archive["raw_policy_command"], dtype=np.float64)
        method_code = str(np.asarray(archive["method"]).item()).lower()
        archived_safe = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        archived_names = archive["joint_names"].astype(str).tolist()
        intent = archive["common_task_intent"].astype(str)
        initial_q = np.asarray(archive["common_initial_q_rad"], dtype=np.float64)
        standardized_initial_grasp = bool(
            np.asarray(archive["standardized_initial_grasp"]).item()
        ) if "standardized_initial_grasp" in archive.files else False
        standardized_hold = (
            np.asarray(
                archive["standardized_left_hold_target_q_rad"], dtype=np.float64
            )
            if "standardized_left_hold_target_q_rad" in archive.files
            else None
        )
    safe = np.asarray(policy_safe_command, dtype=np.float64)
    if not np.array_equal(archived_safe, safe):
        raise RuntimeError("engine command differs from frozen policy-safe command")
    if archived_names != list(joint_names):
        raise RuntimeError("direct execution named joint order mismatch")
    if method_code not in {"a", "b"}:
        raise RuntimeError("unknown ACT method identity")
    primitive = Dex3Primitive.from_frozen_dependencies(
        read_json(PHYSICS_CONFIG),
        read_json(PHYSICAL_ENVIRONMENT),
        read_json(COMMON_PHYSICAL_CONTROLLER),
    )
    joint_contract = read_json(JOINT_CONTRACT)
    joint_lower, joint_upper, authoritative_names = authoritative_joint_limits(
        joint_contract
    )
    if tuple(archived_names) != authoritative_names:
        raise RuntimeError("runtime command does not match authoritative joint order")
    controller = DirectPhysicalDex3ExecutionLayer(
        primitive,
        intent,
        raw,
        safe,
        f"ACT-{method_code.upper()}40",
        joint_lower[:14],
        joint_upper[:14],
        joint_lower[14:],
        joint_upper[14:],
        standardized_initial_grasp=standardized_initial_grasp,
        standardized_initial_q_rad=initial_q if standardized_initial_grasp else None,
        standardized_left_hold_target_rad=(
            standardized_hold if standardized_initial_grasp else None
        ),
    )
    if standardized_initial_grasp:
        if standardized_hold is None:
            raise RuntimeError("standardized-grasp archive lacks the qualified HOLD target")
        if np.any(initial_q[21:28] < joint_lower[21:28]) or np.any(
            initial_q[21:28] > joint_upper[21:28]
        ):
            raise RuntimeError("standardized-grasp RIGHT initial q violates a hard limit")
    else:
        expected_initial_dex3 = np.concatenate((controller.left_open, controller.right_open))
        if not np.allclose(initial_q[14:28], expected_initial_dex3, rtol=0.0, atol=1.0e-7):
            raise RuntimeError("common initial Dex3 q is not the frozen limit-safe OPEN state")
    return DirectIsaacExecutionRuntime(
        controller=controller,
        joint_names=joint_names,
        whole_hand_geometry=read_json(WHOLE_HAND_GEOMETRY),
        evaluator_manifest_sha256="NO_CLASSIFIER_OR_ATLAS",
        execution_manifest_sha256=sha256_file(freeze_path),
        direct_freeze_sha256=sha256_file(freeze_path),
        initial_q_rad=initial_q,
    )


__all__ = ["DirectIsaacExecutionRuntime", "build_runtime", "verify_direct_freeze"]
