#!/usr/bin/env python3
"""Shared Isaac controller contract for Policy-B validation and limit tests.

This module contains parameters only.  It has no task, policy, camera, dataset,
or real-robot command path.  Both the policy rollout runner and the independent
Dex3 boundary characterization construct their actuators from this exact data.
"""

from __future__ import annotations

from typing import Any


CONTROL_FPS = 30.0
PHYSICS_DT = 1.0 / 120.0

CONTROLLER_CONTRACT: dict[str, Any] = {
    "schema_version": "g1_policy_b_isaac_controller_contract_v1",
    "simulation": {
        "physics_dt_s": PHYSICS_DT,
        "control_fps": CONTROL_FPS,
        "physics_substeps_per_control_frame": 4,
        "device": "cuda:0",
        "use_fabric": True,
    },
    "actuators": {
        "fixed_base_legs": {
            "joint_names_expr": [
                r"(left|right)_(hip|knee|ankle)_.*_joint",
                r"(left|right)_knee_joint",
            ],
            "effort_limit_sim": 139.0,
            "velocity_limit_sim": 32.0,
            "stiffness": 200.0,
            "damping": 10.0,
        },
        "fixed_base_waist": {
            "joint_names_expr": [r"waist_.*_joint"],
            "effort_limit_sim": 35.0,
            "velocity_limit_sim": 30.0,
            "stiffness": 1000.0,
            "damping": 40.0,
        },
        "arms": {
            "joint_names_expr": [
                r"(left|right)_(shoulder|wrist)_.*_joint",
                r"(left|right)_elbow_joint",
            ],
            "effort_limit_sim": 25.0,
            "velocity_limit_sim": 12.0,
            "stiffness": 1000.0,
            "damping": 40.0,
        },
        "dex3": {
            "joint_names_expr": [r"(left|right)_hand_.*_joint"],
            "effort_limit_sim": 2.5,
            "velocity_limit_sim": 12.0,
            "stiffness": 100.0,
            "damping": 4.0,
        },
    },
}


def build_implicit_actuators(implicit_actuator_cls: Any) -> dict[str, Any]:
    """Instantiate Isaac Lab actuator configs from the frozen plain-data contract."""

    return {
        name: implicit_actuator_cls(**values)
        for name, values in CONTROLLER_CONTRACT["actuators"].items()
    }


__all__ = [
    "CONTROL_FPS",
    "PHYSICS_DT",
    "CONTROLLER_CONTRACT",
    "build_implicit_actuators",
]
