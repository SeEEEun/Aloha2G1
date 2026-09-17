#!/usr/bin/env python3
"""Instrument the contact-constrained engine with direct Dex3-only execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ENGINE = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
DEFAULT_FREEZE = ROOT / "outputs/final_direct_physical_eval35/00_freeze/DIRECT_EVAL35_FREEZE_MANIFEST.json"
AUTHORITATIVE_JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
DEX3_JOINT_STOP_INSET_RAD = "0.005"


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instrument(source: str) -> tuple[str, dict[str, int]]:
    substitutions = [
        (
            "build_runtime",
            '''    if commands.shape[1] != 28 or not np.isfinite(commands).all():\n        raise RuntimeError("invalid scripted commands")\n''',
            '''    if commands.shape[1] != 28 or not np.isfinite(commands).all():\n        raise RuntimeError("invalid scripted commands")\n    from tools.direct_physical_execution_isaac_runtime import build_runtime\n    common_execution = build_runtime(command_path, commands, names)\n''',
        ),
        (
            "add_trace_fields",
            '''    bin_contact_records: dict[str, list[Any]] = {\n''',
            '''    for common_field in common_execution.event_field_names:\n        if common_field in records:\n            raise RuntimeError(f"direct execution trace field collision: {common_field}")\n        records[common_field] = []\n    bin_contact_records: dict[str, list[Any]] = {\n''',
        ),
        (
            "common_initial_state",
            '''    target[0, joint_ids] = torch.as_tensor(commands[0], device=robot.device, dtype=torch.float32)\n''',
            '''    target[0, joint_ids] = torch.as_tensor(common_execution.initial_q_rad, device=robot.device, dtype=torch.float32)\n''',
        ),
        (
            "causal_step",
            '''    for control_frame, (command, label) in enumerate(zip(commands, stages, strict=True)):\n        if (\n''',
            '''    for control_frame, (command, label) in enumerate(zip(commands, stages, strict=True)):\n        common_snapshot = common_execution.snapshot(\n            measured_q_rad=numpy(robot.data.joint_pos)[0, joint_ids].astype(np.float64),\n            object_pose_xyzw=numpy(doll.data.root_pose_w)[0].astype(np.float64),\n            body_names=body_names,\n            body_positions_world_m=numpy(robot.data.body_pos_w)[0].astype(np.float64),\n            body_quaternions_xyzw=numpy(robot.data.body_quat_w)[0].astype(np.float64),\n            records=records,\n        )\n        command = common_execution.step(control_frame, common_snapshot)\n        if (\n''',
        ),
        (
            "event_values",
            '''            for key, value in values.items():\n                records[key].append(value)\n''',
            '''            values.update(common_execution.event_values(measured))\n            for key, value in values.items():\n                records[key].append(value)\n''',
        ),
        (
            "runtime_summary_file",
            '''    os.replace(temporary, log_path)\n    bin_contact_path: Path | None = None\n''',
            '''    os.replace(temporary, log_path)\n    common_execution.write_summary(output_dir / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")\n    bin_contact_path: Path | None = None\n''',
        ),
        (
            "runtime_summary_result",
            '''        "policy_or_checkpoint_used": False,\n''',
            '''        "direct_common_execution_layer": common_execution.summary(),\n        "policy_or_checkpoint_used": True,\n''',
        ),
    ]
    counts: dict[str, int] = {}
    result = source
    for name, old, new in substitutions:
        count = result.count(old)
        counts[name] = count
        if count != 1:
            raise RuntimeError(f"engine instrumentation point {name}: {count} != 1")
        result = result.replace(old, new, 1)
    compile(result, str(ENGINE), "exec")
    return result, counts


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--direct-freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--qualification-mode", action="store_true")
    parser.add_argument("--validate-patch-only", action="store_true")
    known, remaining = parser.parse_known_args()
    source = ENGINE.read_text(encoding="utf-8")
    instrumented, counts = instrument(source)
    report = {
        "engine": str(ENGINE),
        "engine_sha256": sha256_file(ENGINE),
        "instrumentation_counts": counts,
        "instrumented_source_compiles": True,
    }
    if known.validate_patch_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if not known.direct_freeze_manifest.is_file():
        raise FileNotFoundError(f"direct EVAL35 freeze unavailable: {known.direct_freeze_manifest}")
    os.environ["DIRECT_EVAL35_FREEZE_MANIFEST"] = str(known.direct_freeze_manifest.resolve())
    if known.qualification_mode:
        os.environ["DIRECT_EXECUTION_QUALIFICATION_MODE"] = "1"
    if "--dex3-hard-limit-contract" in remaining or "--dex3-hard-limit-inset-rad" in remaining:
        raise RuntimeError("direct execution owns the common Dex3 joint-stop contract")
    sys.argv = [
        str(ENGINE),
        *remaining,
        "--dex3-hard-limit-contract",
        str(AUTHORITATIVE_JOINT_CONTRACT),
        "--dex3-hard-limit-inset-rad",
        DEX3_JOINT_STOP_INSET_RAD,
    ]
    namespace = {"__name__": "__main__", "__file__": str(ENGINE), "__package__": None}
    exec(compile(instrumented, str(ENGINE), "exec"), namespace, namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
