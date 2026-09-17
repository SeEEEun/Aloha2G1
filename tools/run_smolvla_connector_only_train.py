#!/usr/bin/env python3
"""Run LeRobot training with only SmolVLM's installed visual connector trainable.

This launcher changes no model architecture or checkpoint tensor. It resolves the
connector through the concrete installed SmolVLA object graph, freezes all policy
parameters, and re-enables exactly that visual modality-projection module before
the optimizer is constructed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def config_path_from_argv() -> Path:
    for index, token in enumerate(sys.argv[1:]):
        if token.startswith("--config_path="):
            return Path(token.split("=", 1)[1]).resolve()
        if token == "--config_path" and index + 2 <= len(sys.argv[1:]):
            return Path(sys.argv[1:][index + 1]).resolve()
    raise RuntimeError("connector-only launcher requires --config_path")


CONFIG_PATH = config_path_from_argv()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
AUDIT_PATH = Path(CONFIG["output_dir"]).resolve().parent / "R2_trainable_parameter_audit.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def connector_only_optim_params(self: SmolVLAPolicy):
    for parameter in self.parameters():
        parameter.requires_grad_(False)

    # Resolve the visual bridge by its actual object relationship in the
    # installed SmolVLA implementation, not by a guessed substring selection.
    connector = self.model.vlm_with_expert.vlm.model.connector
    connector_parameter_ids = {id(parameter) for parameter in connector.parameters()}
    if not connector_parameter_ids:
        raise RuntimeError("installed SmolVLA connector has no parameters")
    selected = []
    selected_names = []
    selected_shapes = []
    for name, parameter in self.named_parameters():
        if id(parameter) in connector_parameter_ids:
            parameter.requires_grad_(True)
            selected.append(parameter)
            selected_names.append(name)
            selected_shapes.append(list(parameter.shape))
    if {id(parameter) for parameter in selected} != connector_parameter_ids:
        raise RuntimeError("could not map every connector parameter into policy.named_parameters()")
    expected_names = [
        "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"
    ]
    scalar_count = sum(parameter.numel() for parameter in selected)
    if selected_names != expected_names or scalar_count != 11_796_480:
        raise RuntimeError(
            f"installed connector identity changed: names={selected_names}, scalars={scalar_count}"
        )
    nonselected_trainable = [
        name
        for name, parameter in self.named_parameters()
        if parameter.requires_grad and id(parameter) not in connector_parameter_ids
    ]
    if nonselected_trainable:
        raise RuntimeError(f"non-connector parameters remain trainable: {nonselected_trainable}")
    atomic_json(
        AUDIT_PATH,
        {
            "status": "PASS",
            "selection_method": "resolved policy.model.vlm_with_expert.vlm.model.connector object",
            "trainable_parameter_names": selected_names,
            "trainable_parameter_shapes": selected_shapes,
            "trainable_scalar_count": scalar_count,
            "all_non_connector_parameters_frozen": True,
            "frozen_semantic_components": [
                "vision encoder",
                "language transformer",
                "action expert/decoder",
                "state projection",
                "action input/output projections",
                "action-time MLP",
            ],
            "architecture_changed": False,
            "config": str(CONFIG_PATH),
        },
    )
    return selected


SmolVLAPolicy.get_optim_params = connector_only_optim_params


if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    main()
