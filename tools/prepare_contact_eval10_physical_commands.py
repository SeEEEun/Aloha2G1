#!/usr/bin/env python3
"""Prepare immutable, method-preserving EVAL10 physical command artifacts.

The executed command is the frozen ACT deployment-safe trajectory itself.  The
common-controller override mask is therefore exactly zero: this avoids erasing
the A/B policy distinction and provides the strictest physical ACT evaluation.
Source-derived semantic labels are audit/scoring labels only and never change q.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
BASE = ROOT / "outputs/final_contact_constrained_eval/04_eval10_preparation"
ACT_ROOT = BASE / "act_trajectories"
EVAL10 = BASE / "EVAL10_RETARGETING_MANIFEST.json"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
OUT = ROOT / "outputs/final_contact_constrained_eval/05_act_ab_results/commands"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_stages(reference: Path, frames: int) -> np.ndarray:
    with np.load(reference, allow_pickle=False) as archive:
        owner = archive["ownership_state"].astype(str)
        left_phase = archive["left_hand_phase"].astype(str)
        right_phase = archive["right_hand_phase"].astype(str)
    if len(owner) != frames:
        raise RuntimeError(f"semantic reference length mismatch: {reference}")
    stages = np.full(frames, "OPEN", dtype="U32")
    stages[(owner == "NO_OWNER") & (left_phase == "PRESHAPE")] = "PRESHAPE"
    left_owned = np.flatnonzero(owner == "LEFT_OWNED")
    if len(left_owned):
        start = int(left_owned[0])
        stages[left_owned] = "LEFT_TRANSPORT"
        stages[start : min(frames, start + 30)] = "GRAVITY_RETENTION"
        stages[min(frames, start + 30) : min(frames, start + 60)] = "LEFT_LIFT_5CM"
    stages[owner == "HANDOFF_APPROACH"] = "LEFT_TRANSPORT"
    stages[owner == "DUAL_CONTACT"] = "RIGHT_THREE_DIGIT_VERIFICATION"
    stages[owner == "RIGHT_OWNED"] = "RIGHT_POST_RELEASE_RETENTION"
    stages[owner == "RIGHT_TRANSPORT"] = "RIGHT_TRANSPORT"
    stages[owner == "RELEASED"] = "RIGHT_RELEASE"
    # These phase labels are recorded for provenance; no q command depends on them.
    if not np.any(stages == "GRAVITY_RETENTION") or not np.any(stages == "RIGHT_RELEASE"):
        raise RuntimeError(f"incomplete semantic sequence: {reference}")
    return stages


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    eval10 = read_json(EVAL10)
    heldout = read_json(HELDOUT)
    new_rows = {int(row["eval_index"]): row for row in eval10["new_unseen_2"]}
    records = []
    for method in ("a", "b"):
        batch = read_json(ACT_ROOT / f"act_{method}40/BATCH_MANIFEST.json")
        for entry in batch["entries"]:
            index = int(entry["eval_index"])
            act_path = Path(entry["trajectory"])
            if sha256_file(act_path) != entry["trajectory_sha256"]:
                raise RuntimeError(f"ACT cache hash drift: {act_path}")
            with np.load(act_path, allow_pickle=False) as archive:
                raw = np.asarray(archive["raw_temporal_ensemble_action"], dtype=np.float32)
                executed = np.asarray(archive["deployment_safe_action"], dtype=np.float32)
                names = archive["joint_names"].astype(str)
            frames = len(executed)
            if index < 8:
                reference = Path(heldout["entries"][index]["b_trajectory_path"])
            else:
                reference = Path(new_rows[index]["b"]["final_retargeted_source"])
            stages = semantic_stages(reference, frames)
            destination = output / f"act_{method}40/eval_{index:02d}_{entry['stable_episode_id']}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npz.incomplete")
            override = np.zeros_like(executed, dtype=bool)
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    commanded_q_rad=executed,
                    stage=stages,
                    joint_names=names,
                    control_fps_hz=np.asarray(30.0),
                    raw_policy_command=raw,
                    executed_common_controller_command=executed,
                    common_controller_override_mask=override,
                    runtime_right_three_digit_gate_required=np.asarray(False),
                    method=np.asarray(method),
                    eval_index=np.asarray(index),
                    provenance=np.asarray(entry["provenance"]),
                    stable_episode_id=np.asarray(entry["stable_episode_id"]),
                    semantic_reference=np.asarray(str(reference.resolve())),
                    direct_policy_execution=np.asarray(True),
                )
            os.replace(temporary, destination)
            records.append(
                {
                    "method": f"ACT-{method.upper()}40",
                    "eval_index": index,
                    "provenance": entry["provenance"],
                    "stable_episode_id": entry["stable_episode_id"],
                    "frames": frames,
                    "raw_act_trajectory": str(act_path),
                    "raw_act_trajectory_sha256": sha256_file(act_path),
                    "physical_command": str(destination.resolve()),
                    "physical_command_sha256": sha256_file(destination),
                    "semantic_reference": str(reference.resolve()),
                    "semantic_reference_sha256": sha256_file(reference),
                    "common_controller_override_scalar_count": 0,
                    "common_controller_override_fraction": 0.0,
                    "arm_intervention_fraction": 0.0,
                    "wrist_intervention_fraction": 0.0,
                    "dex3_intervention_fraction": 0.0,
                    "raw_to_executed_rmse_rad": float(np.sqrt(np.mean((raw - executed) ** 2))),
                    "deployment_safety_projection_is_frozen_common": True,
                }
            )
    manifest = {
        "schema_version": "contact_constrained_eval10_physical_command_v1",
        "status": "PASS",
        "evaluation_set": "EVAL10",
        "policy_behavior_preserved": True,
        "common_local_primitive_override_used": False,
        "reason": (
            "No post-freeze method-preserving primitive-entry adapter was present in "
            "the authoritative controller artifact. Executing the frozen ACT-E1 "
            "deployment-safe predictions directly avoids replacing policy motion."
        ),
        "semantic_labels_affect_commands": False,
        "identical_execution_path_for_a_b": True,
        "per_episode_or_method_tuning": False,
        "records": records,
    }
    path = output.parent / "PHYSICAL_COMMAND_MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "commands": len(records), "manifest": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
