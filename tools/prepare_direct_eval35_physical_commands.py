#!/usr/bin/env python3
"""Create immutable classifier-free ACT-A/B EVAL35 physical commands."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_execution_isaac_runtime import (
    COMMON_PHYSICAL_CONTROLLER,
    PHYSICAL_ENVIRONMENT,
    PHYSICS_CONFIG,
)
from tools.common_execution_layer import Dex3Primitive
from tools.direct_physical_execution_layer import (
    DEX3_HARD_LIMIT_GUARD_RAD as EXECUTION_DEX3_HARD_LIMIT_GUARD_RAD,
    authoritative_dex3_limits,
)

OUT = ROOT / "outputs/final_direct_physical_eval35"
PREP = OUT / "00_preparation"
BASE_ACT = ROOT / "outputs/final_contact_constrained_eval/04_eval10_preparation/act_trajectories"
NEW_ACT = PREP / "act_trajectories"
EVAL35 = ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer/EVAL35_MANIFEST.json"
INTENT_MANIFEST = PREP / "COMMON_SOURCE_TASK_INTENT_EVAL35.json"
INTENT_NPZ = PREP / "COMMON_SOURCE_TASK_INTENT_EVAL35.npz"
INITIAL = ROOT / "outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json"
MANIFEST = PREP / "DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"

# Preparation-time source of truth for the qualified common Dex3 safety inset.
# Keep this explicit so a stale cached command bundle cannot silently inherit an
# obsolete numerical-interior value.
AUTHORITATIVE_DEX3_HARD_LIMIT_GUARD_RAD = 5.0e-3


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def main() -> int:
    if not np.isclose(
        EXECUTION_DEX3_HARD_LIMIT_GUARD_RAD,
        AUTHORITATIVE_DEX3_HARD_LIMIT_GUARD_RAD,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("preparation/runtime Dex3 hard-limit guard mismatch")
    if MANIFEST.is_file():
        cached = read_json(MANIFEST)
        if cached.get("status") != "FROZEN_BEFORE_PHYSICAL_ROLLOUT" or len(cached.get("records", [])) != 70:
            raise RuntimeError("existing direct EVAL35 command manifest invalid")
        if not np.isclose(
            float(cached.get("dex3_hard_limit_guard_rad", float("nan"))),
            AUTHORITATIVE_DEX3_HARD_LIMIT_GUARD_RAD,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("cached direct EVAL35 commands use a stale Dex3 hard-limit guard")
        for row in cached["records"]:
            if sha256_file(Path(row["physical_command"])) != row["physical_command_sha256"]:
                raise RuntimeError(f"command hash drift: {row['physical_command']}")
        print(json.dumps({"status": cached["status"], "cache_hit": str(MANIFEST)}, indent=2))
        return 0
    eval35 = read_json(EVAL35)
    intent_manifest = read_json(INTENT_MANIFEST)
    initial = read_json(INITIAL)
    if eval35.get("status") != "PASS_IDENTITY_FROZEN" or eval35.get("source_count") != 35:
        raise RuntimeError("EVAL35 identity not frozen")
    if intent_manifest.get("status") != "FROZEN_BEFORE_ROLLOUT" or len(intent_manifest["records"]) != 35:
        raise RuntimeError("common source task intent not frozen")
    if initial.get("status") != "COMMON_G1_POLICY_INITIAL_STATE_AB_V1" or initial.get("policy_specific") is not False:
        raise RuntimeError("common A/B initial state contract unavailable")
    initial_q = np.asarray(initial["full_28d_initial_q_rad"], dtype=np.float32)
    if initial_q.shape != (28,):
        raise RuntimeError("invalid common initial q")
    joint_contract = read_json(JOINT_CONTRACT)
    dex3_lower, dex3_upper, dex3_names = authoritative_dex3_limits(joint_contract)
    primitive = Dex3Primitive.from_frozen_dependencies(
        read_json(PHYSICS_CONFIG),
        read_json(PHYSICAL_ENVIRONMENT),
        read_json(COMMON_PHYSICAL_CONTROLLER),
    )
    command_lower = dex3_lower + AUTHORITATIVE_DEX3_HARD_LIMIT_GUARD_RAD
    command_upper = dex3_upper - AUTHORITATIVE_DEX3_HARD_LIMIT_GUARD_RAD
    common_open = np.concatenate((primitive.left_open, primitive.right_open))
    common_open = np.clip(common_open, command_lower, command_upper).astype(np.float32)
    # Preserve the authoritative common arm/root task-ready state exactly, while
    # placing both Dex3 hands in the same limit-safe common OPEN state used from
    # frame zero by the physical execution layer.
    initial_q[14:28] = common_open
    identities = {int(row["eval_index"]): row for row in eval35["eval_entries"]}
    intent_rows = {int(row["eval_index"]): row for row in intent_manifest["records"]}
    with np.load(INTENT_NPZ, allow_pickle=False) as archive:
        intents = {index: archive[f"eval_{index:02d}"].astype(str) for index in range(35)}
    records: list[dict[str, Any]] = []
    for method in ("a", "b"):
        base = read_json(BASE_ACT / f"act_{method}40/BATCH_MANIFEST.json")
        new = read_json(NEW_ACT / f"act_{method}40/BATCH_MANIFEST.json")
        if base.get("status") != "PASS" or base.get("episodes") != 10:
            raise RuntimeError(f"ACT-{method.upper()} base10 unavailable")
        if new.get("status") != "PASS" or new.get("episodes") != 25:
            raise RuntimeError(f"ACT-{method.upper()} NEW25 unavailable")
        if base["checkpoint_model_sha256"] != new["checkpoint_model_sha256"]:
            raise RuntimeError(f"ACT-{method.upper()} checkpoint drift between partitions")
        entries = sorted(base["entries"] + new["entries"], key=lambda row: int(row["eval_index"]))
        if [int(row["eval_index"]) for row in entries] != list(range(35)):
            raise RuntimeError(f"ACT-{method.upper()} trajectories not exact EVAL35")
        for entry in entries:
            index = int(entry["eval_index"])
            identity = identities[index]
            if entry["stable_episode_id"] != identity["stable_episode_id"]:
                raise RuntimeError(f"ACT/EVAL35 identity mismatch at {method}:{index}")
            trajectory = Path(entry["trajectory"])
            if sha256_file(trajectory) != entry["trajectory_sha256"]:
                raise RuntimeError(f"ACT trajectory hash drift: {trajectory}")
            with np.load(trajectory, allow_pickle=False) as archive:
                raw = np.asarray(archive["raw_temporal_ensemble_action"], dtype=np.float32)
                safe = np.asarray(archive["deployment_safe_action"], dtype=np.float32)
                names = archive["joint_names"].astype(str)
                checkpoint_sha = str(np.asarray(archive["checkpoint_model_sha256"]).item())
            intent = intents[index]
            if raw.shape != safe.shape or safe.shape != (int(identity["frames"]), 28):
                raise RuntimeError(f"ACT command shape mismatch at {method}:{index}")
            if intent.shape != (len(safe),) or sha256_array(intent) != intent_rows[index]["timeline_sha256"]:
                raise RuntimeError(f"common intent mismatch at {index}")
            destination = PREP / "physical_commands" / f"act_{method}40/eval_{index:02d}_{identity['stable_episode_id']}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".npz.incomplete")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, commanded_q_rad=safe, stage=intent, joint_names=names,
                    control_fps_hz=np.asarray(30.0), raw_policy_command=raw,
                    policy_safe_command=safe, common_task_intent=intent,
                    common_initial_q_rad=initial_q,
                    runtime_right_three_digit_gate_required=np.asarray(False),
                    method=np.asarray(method), eval_index=np.asarray(index),
                    provenance=np.asarray(identity["provenance"]),
                    stable_episode_id=np.asarray(identity["stable_episode_id"]),
                    checkpoint_model_sha256=np.asarray(checkpoint_sha),
                    pregrasp_classifier_used=np.asarray(False),
                    wrist_distance_gate_used=np.asarray(False),
                    arm_rescue_allowed=np.asarray(False), wrist_rescue_allowed=np.asarray(False),
                )
            os.replace(temporary, destination)
            records.append({
                "method": f"ACT-{method.upper()}40", "eval_index": index,
                "stable_episode_id": identity["stable_episode_id"],
                "provenance": identity["provenance"], "frames": len(safe),
                "checkpoint_model_sha256": checkpoint_sha,
                "act_trajectory": str(trajectory.resolve()),
                "act_trajectory_sha256": sha256_file(trajectory),
                "physical_command": str(destination.resolve()),
                "physical_command_sha256": sha256_file(destination),
                "raw_policy_arm_sha256": sha256_array(raw[:, :14]),
                "policy_safe_arm_sha256": sha256_array(safe[:, :14]),
                "common_task_intent_sha256": sha256_array(intent),
                "common_initial_q_sha256": sha256_array(initial_q),
                "common_initial_dex3_state": "LIMIT_SAFE_COMMON_OPEN",
                "pregrasp_classifier_used": False, "wrist_distance_gate_used": False,
                "arm_rescue_allowed": False, "wrist_rescue_allowed": False,
                "dex3_common_primitive_allowed": True,
            })
    if len(records) != 70:
        raise RuntimeError("direct command set is not 70 rollouts")
    # Same source-derived opportunity for the paired A/B commands.
    for index in range(35):
        pair = [row for row in records if row["eval_index"] == index]
        if len(pair) != 2 or pair[0]["common_task_intent_sha256"] != pair[1]["common_task_intent_sha256"]:
            raise RuntimeError(f"A/B intent mismatch at {index}")
    value = {
        "schema_version": "direct_physical_eval35_command_manifest_v1",
        "status": "FROZEN_BEFORE_PHYSICAL_ROLLOUT", "evaluation_set": "EVAL35",
        "rollouts": 70, "common_initial_state": str(INITIAL.resolve()),
        "common_initial_state_sha256": sha256_file(INITIAL),
        "common_initial_arm_state_preserved_exactly": True,
        "common_initial_dex3_state": "LIMIT_SAFE_COMMON_OPEN",
        "common_initial_dex3_q_rad": common_open.tolist(),
        "authoritative_joint_limit_contract": str(JOINT_CONTRACT.resolve()),
        "authoritative_joint_limit_contract_sha256": sha256_file(JOINT_CONTRACT),
        "authoritative_dex3_joint_names": list(dex3_names),
        "authoritative_dex3_lower_rad": dex3_lower.tolist(),
        "authoritative_dex3_upper_rad": dex3_upper.tolist(),
        "dex3_hard_limit_guard_rad": AUTHORITATIVE_DEX3_HARD_LIMIT_GUARD_RAD,
        "common_task_intent_manifest": str(INTENT_MANIFEST.resolve()),
        "common_task_intent_manifest_sha256": sha256_file(INTENT_MANIFEST),
        "pregrasp_classifier_used": False, "graspability_atlas_used": False,
        "wrist_distance_gate_used": False, "arm_rescue_allowed": False,
        "wrist_rescue_allowed": False, "method_specific_arm_motion_preserved": True,
        "records": records,
    }
    MANIFEST.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "commands": 70, "manifest": str(MANIFEST)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
