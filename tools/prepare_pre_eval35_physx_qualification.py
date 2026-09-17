#!/usr/bin/env python3
"""Prepare deterministic non-eval PhysX smoke and scripted regression inputs."""

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

from tools.build_eval35_common_task_intent import INTENT_EVENTS, make_timeline  # noqa: E402
from tools.common_execution_isaac_runtime import (  # noqa: E402
    COMMON_PHYSICAL_CONTROLLER,
    PHYSICAL_ENVIRONMENT,
    PHYSICS_CONFIG,
)
from tools.common_execution_layer import Dex3Primitive  # noqa: E402
from tools.direct_physical_execution_layer import (  # noqa: E402
    DEX3_HARD_LIMIT_GUARD_RAD,
    authoritative_dex3_limits,
)


OUT = ROOT / "outputs/final_direct_physical_eval35/00_pre_eval35_execution_freeze"
QUAL = OUT / "physx_qualification"
COMMANDS = QUAL / "commands"
TRAIN40 = ROOT / "outputs/paper_core_ab/train40_manifest.json"
HELDOUT8 = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
INITIAL = ROOT / "outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
SCRIPTED_SOURCE = ROOT / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm/bin_calibrated_full_command.npz"
PREP_MANIFEST = QUAL / "PREPARED_INPUTS.json"
PROVISIONAL = QUAL / "PROVISIONAL_EXECUTION_MANIFEST.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def event_dict(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["event_names"].astype(str)
        frames = archive["event_frames"].astype(np.int64)
    return {str(name): int(frame) for name, frame in zip(names, frames, strict=True)}


def command(path: Path, canonical_names: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["replay_joint_names"].astype(str).tolist()
        values = np.asarray(archive["replay_named_joint_qpos"], dtype=np.float32)
    lookup = {name: index for index, name in enumerate(names)}
    if set(lookup) != set(canonical_names):
        raise RuntimeError("TRAIN40 qualification joint mapping mismatch")
    return values[:, [lookup[name] for name in canonical_names]]


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main() -> int:
    QUAL.mkdir(parents=True, exist_ok=True)
    train = read_json(TRAIN40)
    heldout_ids = {str(row["stable_episode_id"]) for row in read_json(HELDOUT8)["entries"]}
    # This rule is declared/materialized before any smoke outcome is observed.
    selected = sorted(
        train["entries"], key=lambda row: (int(row["final_dataset_index"]), str(row["stable_episode_id"]))
    )[:2]
    if len(selected) != 2 or any(str(row["stable_episode_id"]) in heldout_ids for row in selected):
        raise RuntimeError("deterministic TRAIN40 smoke selection invalid")
    contract = read_json(JOINT_CONTRACT)
    canonical_names = [str(name) for name in contract["joint_names"]]
    lower, upper, dex3_names = authoritative_dex3_limits(contract)
    primitive = Dex3Primitive.from_frozen_dependencies(
        read_json(PHYSICS_CONFIG), read_json(PHYSICAL_ENVIRONMENT), read_json(COMMON_PHYSICAL_CONTROLLER)
    )
    open_q = np.clip(
        np.concatenate((primitive.left_open, primitive.right_open)),
        lower + DEX3_HARD_LIMIT_GUARD_RAD,
        upper - DEX3_HARD_LIMIT_GUARD_RAD,
    ).astype(np.float32)
    common_initial = np.asarray(read_json(INITIAL)["full_28d_initial_q_rad"], dtype=np.float32)
    common_initial[14:28] = open_q
    records = []
    for smoke_index, row in enumerate(selected):
        a_path = Path(row["a_trajectory_path"])
        b_path = Path(row["b_trajectory_path"])
        a_events = event_dict(a_path)
        timeline = make_timeline(
            int(row["frames"]), {name: a_events[name] for name in INTENT_EVENTS}
        )
        for method, source in (("ACT-A40", a_path), ("ACT-B40", b_path)):
            values = command(source, canonical_names)
            path = COMMANDS / f"{method.lower().replace('-', '_')}_train_smoke_{smoke_index:02d}_{row['stable_episode_id']}.npz"
            save_npz(
                path,
                commanded_q_rad=values,
                stage=timeline,
                joint_names=np.asarray(canonical_names),
                control_fps_hz=np.asarray(30.0),
                raw_policy_command=values,
                policy_safe_command=values,
                common_task_intent=timeline,
                common_initial_q_rad=common_initial,
                method=np.asarray("a" if method == "ACT-A40" else "b"),
                stable_episode_id=np.asarray(row["stable_episode_id"]),
                qualification_only=np.asarray(True),
                evaluation_episode=np.asarray(False),
                source_kind=np.asarray("TRAIN40_POLICY_FORMAT_SUPERVISION_STREAM"),
                runtime_right_three_digit_gate_required=np.asarray(False),
            )
            records.append(
                {
                    "method_stream": method,
                    "smoke_index": smoke_index,
                    "stable_episode_id": row["stable_episode_id"],
                    "training_final_dataset_index": int(row["final_dataset_index"]),
                    "command": str(path.resolve()),
                    "command_sha256": sha256_file(path),
                    "frames": len(values),
                    "source_trajectory": str(source.resolve()),
                    "source_trajectory_sha256": sha256_file(source),
                    "evaluation_episode": False,
                }
            )

    with np.load(SCRIPTED_SOURCE, allow_pickle=False) as archive:
        scripted = {key: np.asarray(archive[key]) for key in archive.files}
    scripted_names = scripted["joint_names"].astype(str).tolist()
    if scripted_names != canonical_names:
        raise RuntimeError("scripted regression command joint order is not authoritative")
    original = np.asarray(scripted["commanded_q_rad"], dtype=np.float64)
    corrected = original.copy()
    corrected[:, 14:28] = np.clip(
        corrected[:, 14:28],
        lower + DEX3_HARD_LIMIT_GUARD_RAD,
        upper - DEX3_HARD_LIMIT_GUARD_RAD,
    )
    scripted_path = COMMANDS / "scripted_full_task_limit_safe_regression.npz"
    scripted["commanded_q_rad"] = corrected.astype(np.float32)
    scripted["joint_limit_guard_rad"] = np.asarray(DEX3_HARD_LIMIT_GUARD_RAD)
    scripted["joint_limit_contract_sha256"] = np.asarray(sha256_file(JOINT_CONTRACT))
    scripted["qualification_only"] = np.asarray(True)
    save_npz(scripted_path, **scripted)
    regression = {
        "command": str(scripted_path.resolve()),
        "command_sha256": sha256_file(scripted_path),
        "source_command": str(SCRIPTED_SOURCE.resolve()),
        "source_command_sha256": sha256_file(SCRIPTED_SOURCE),
        "arm_command_exact": bool(np.array_equal(corrected[:, :14], original[:, :14])),
        "changed_dex3_scalar_count": int(np.count_nonzero(corrected[:, 14:] != original[:, 14:])),
        "maximum_dex3_change_rad": float(np.max(np.abs(corrected[:, 14:] - original[:, 14:]), initial=0.0)),
        "dex3_hard_limit_violations": int(np.count_nonzero((corrected[:, 14:] < lower) | (corrected[:, 14:] > upper))),
        "frames": len(corrected),
    }
    value = {
        "schema_version": "pre_eval35_physx_qualification_inputs_v1",
        "status": "PREPARED_BEFORE_PHYSX_OUTCOMES",
        "evaluation_data_used": False,
        "selection_rule": "first two TRAIN40 entries sorted by frozen final_dataset_index then stable_episode_id",
        "selected_train_episodes": [row["stable_episode_id"] for row in selected],
        "smoke_records": records,
        "scripted_regression": regression,
        "authoritative_joint_limit_contract": str(JOINT_CONTRACT.resolve()),
        "authoritative_joint_limit_contract_sha256": sha256_file(JOINT_CONTRACT),
        "dex3_joint_names": list(dex3_names),
        "common_initial_dex3_q_rad": open_q.tolist(),
    }
    atomic_json(PREP_MANIFEST, value)

    implementation_files = [
        ROOT / "tools/direct_physical_execution_layer.py",
        ROOT / "tools/direct_physical_execution_isaac_runtime.py",
        ROOT / "tools/run_direct_physical_execution_isaac.py",
        ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py",
        JOINT_CONTRACT,
        ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json",
        PHYSICS_CONFIG,
        PHYSICAL_ENVIRONMENT,
        COMMON_PHYSICAL_CONTROLLER,
        PREP_MANIFEST,
        *[Path(row["command"]) for row in records],
    ]
    provisional = {
        "schema_version": "pre_eval35_qualification_provisional_execution_v1",
        "status": "PRE_EVAL35_QUALIFICATION_PROVISIONAL",
        "graspability_classifier_used": False,
        "evaluation_rollout_permitted": False,
        "qualification_only": True,
        "files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in implementation_files
        ],
    }
    atomic_json(PROVISIONAL, provisional)
    print(
        json.dumps(
            {
                "status": value["status"],
                "selected_train_episodes": value["selected_train_episodes"],
                "smoke_commands": len(records),
                "scripted_regression_command": regression,
                "provisional_manifest": str(PROVISIONAL),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
