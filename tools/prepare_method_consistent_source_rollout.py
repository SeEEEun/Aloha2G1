#!/usr/bin/env python3
"""Freeze the reopened Experiment-3 method-consistent initialization contract.

This program is pre-inference only.  It reads each HELDOUT8 method dataset,
records the exact episode ``observation.state[0]`` in named 28-D order, and
preserves the prior common-state diagnostic and Table-3 files.  It never runs a
policy, changes a trajectory, or projects an initial state.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tools.paper_core_source_rollout_common import SafetyAudit, frozen_interfaces


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PAPER = ROOT / "outputs/paper_core_ab"
OUTPUT = PAPER / "method_consistent_source_video_rollout"
CONTRACT = OUTPUT / "METHOD_CONSISTENT_INITIALIZATION_CONTRACT.json"
FREEZE = OUTPUT / "METHOD_CONSISTENT_INITIALIZATION_CONTRACT.sha256.json"
EVALUATION = OUTPUT / "METHOD_CONSISTENT_EVALUATION_CONTRACT.json"
PRESERVED = OUTPUT / "preserved_common_state_diagnostic"
HELDOUT = PAPER / "heldout8_manifest.json"
SEMANTIC = ROOT / "configs/paper_semantic_task_sequence_v1.json"
EXECUTION = ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout/SELECTED_ACT_EXECUTION_CONFIG.json"
PROJECTION = ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"
PREVIOUS_ROOT = PAPER / "source_conditioned_rollout"

DATASETS = {
    "a": ROOT / "datasets/doll_handoff_fair_a_heldout8",
    "b": ROOT / "datasets/doll_handoff_proposed_b_heldout8",
}
METHOD_LABELS = {"a": "ACT-A40", "b": "ACT-B40"}
EXPECTED_EXECUTION_SHA256 = "656f6f474f6981c5b0c0417895b91ad924d171031bb66b26e4640b54bc863b64"
EXPECTED_PROJECTION_SHA256 = "05078d0038ab6defaa8a0f56b1f38b752cc92892856996fecc05b75fcce27cf2"


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=json_default,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != json.loads(
            json.dumps(value, allow_nan=False, default=json_default)
        ):
            raise RuntimeError(f"refusing to change frozen artifact: {path}")
        return
    atomic_json(path, value)


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def dataset_arrays(root: Path) -> tuple[list[str], dict[int, dict[str, Any]], dict[str, Any]]:
    info_path = root / "meta/info.json"
    packaging_path = root / "meta/g1_packaging_manifest.json"
    info = read_json(info_path)
    packaging = read_json(packaging_path)
    names = list(info["features"]["observation.state"]["names"])
    if names != list(info["features"]["action"]["names"]) or len(names) != 28:
        raise RuntimeError(f"invalid named 28-D interface: {root}")
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if len(files) != 1:
        raise RuntimeError(f"expected one HELDOUT8 parquet: {root}")
    table = pq.read_table(
        files,
        columns=["observation.state", "action", "episode_index", "frame_index"],
    )
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    episode = np.asarray(table["episode_index"], dtype=np.int64)
    frame = np.asarray(table["frame_index"], dtype=np.int64)
    if state.shape != action.shape or state.shape[1:] != (28,) or not np.isfinite(state).all():
        raise RuntimeError(f"invalid held-out state/action arrays: {root}")
    rows: dict[int, dict[str, Any]] = {}
    for output_episode in range(8):
        indices = np.flatnonzero((episode == output_episode) & (frame == 0))
        if len(indices) != 1:
            raise RuntimeError(f"episode {output_episode} does not have exactly one frame 0: {root}")
        index = int(indices[0])
        if not np.array_equal(state[index], action[index]):
            raise RuntimeError(f"state[0] != action[0] for episode {output_episode}: {root}")
        rows[output_episode] = {
            "q": state[index].copy(),
            "state_sha256": sha256_array(state[index]),
            "action_sha256": sha256_array(action[index]),
            "global_dataset_row": index,
        }
    return names, rows, {
        "dataset_root": str(root.resolve()),
        "info": artifact(info_path),
        "packaging_manifest": artifact(packaging_path),
        "data_parquet": artifact(files[0]),
        "state_array_sha256": packaging["state_array_sha256"],
        "action_array_sha256": packaging["action_array_sha256"],
        "episode_mapping": packaging["episode_mapping"],
        "status": packaging["status"],
    }


def preserve_previous() -> dict[str, Any]:
    if not PREVIOUS_ROOT.is_dir():
        raise FileNotFoundError(PREVIOUS_ROOT)
    inventory = []
    for path in sorted(value for value in PREVIOUS_ROOT.rglob("*") if value.is_file()):
        inventory.append(
            {
                "relative_path": str(path.relative_to(PREVIOUS_ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    snapshots = {
        "table3_source_conditioned_rollout.csv": PAPER / "tables/table3_source_conditioned_rollout.csv",
        "table3_source_conditioned_rollout.md": PAPER / "tables/table3_source_conditioned_rollout.md",
        "table3_source_conditioned_rollout.json": PAPER / "tables/table3_source_conditioned_rollout.json",
        "paper_core_blocked_report.json": PAPER / "paper_core_blocked_report.json",
        "PAPER_CORE_BLOCKED_REPORT.md": PAPER / "PAPER_CORE_BLOCKED_REPORT.md",
        "common_state_experiment3_result.json": PREVIOUS_ROOT / "experiment3_result.json",
    }
    preserved_files = PRESERVED / "files"
    preserved_files.mkdir(parents=True, exist_ok=True)
    snapshot_records = {}
    for name, source in snapshots.items():
        source_record = artifact(source)
        target = preserved_files / name
        if target.exists():
            if sha256_file(target) != source_record["sha256"]:
                raise RuntimeError(f"preserved snapshot changed: {target}")
        else:
            shutil.copy2(source, target)
        snapshot_records[name] = {
            "source": source_record,
            "snapshot": artifact(target),
        }
    return {
        "status": "PRESERVED",
        "previous_output_root": str(PREVIOUS_ROOT),
        "previous_output_file_count": len(inventory),
        "previous_output_inventory_canonical_sha256": canonical_sha256(inventory),
        "previous_output_inventory": inventory,
        "snapshots": snapshot_records,
        "will_not_be_overwritten": True,
    }


def main() -> None:
    heldout = read_json(HELDOUT)
    if heldout.get("status") != "PASS" or heldout.get("episode_count") != 8:
        raise RuntimeError("HELDOUT8 manifest is not frozen and valid")
    if sha256_file(EXECUTION) != EXPECTED_EXECUTION_SHA256:
        raise RuntimeError("official ACT-E1 execution config changed")
    if sha256_file(PROJECTION) != EXPECTED_PROJECTION_SHA256:
        raise RuntimeError("common deployment safety adapter changed")
    semantic = read_json(SEMANTIC)
    if semantic.get("status") != "FROZEN_BEFORE_SOURCE_CONDITIONED_ACT_RESULTS":
        raise RuntimeError("semantic evaluator is not pre-frozen")

    names, lower, upper, _ = frozen_interfaces()
    dataset = {}
    frame0 = {}
    for method in ("a", "b"):
        method_names, rows, dataset[method] = dataset_arrays(DATASETS[method])
        if method_names != names:
            raise RuntimeError(f"{method} held-out named order differs from authoritative order")
        frame0[method] = rows
    safety = SafetyAudit(names, lower, upper)

    tolerances = read_json(PAPER / "COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json")[
        "tolerances"
    ]
    dex3_initial_cap = float(
        tolerances["measured_dex3_hard_limit_excursion_diagnostic_cap_rad"]
    )
    entries = []
    for output_episode, source in enumerate(heldout["entries"]):
        methods = {}
        for method in ("a", "b"):
            q = frame0[method][output_episode]["q"]
            lower_excess = np.maximum(lower - q, 0.0)
            upper_excess = np.maximum(q - upper, 0.0)
            excess = np.maximum(lower_excess, upper_excess)
            hard = excess > 1e-9
            arm_excess = float(np.max(excess[:14]))
            dex3_excess = float(np.max(excess[14:]))
            reset_limit_eligible = arm_excess <= 1e-9 and dex3_excess <= dex3_initial_cap
            collision = safety.collision(q[None])
            if not reset_limit_eligible or collision["invalid_hard_self_collision_incidence"]:
                raise RuntimeError(
                    f"method {method} episode {output_episode} frozen state[0] is not reset-feasible"
                )
            mapping = dataset[method]["episode_mapping"][output_episode]
            if (
                int(mapping["output_episode_index"]) != output_episode
                or int(mapping["final_dataset_index"]) != int(source["final_dataset_index"])
                or mapping["stable_episode_id"] != source["stable_episode_id"]
            ):
                raise RuntimeError("held-out dataset/source identity mapping changed")
            methods[method] = {
                "method": METHOD_LABELS[method],
                "dataset": dataset[method]["dataset_root"],
                "global_dataset_row": frame0[method][output_episode]["global_dataset_row"],
                "initial_q_rad": q,
                "initial_q_float32_sha256": frame0[method][output_episode]["state_sha256"],
                "action0_float32_sha256": frame0[method][output_episode]["action_sha256"],
                "state0_equals_action0_bit_exact": True,
                "initial_state_projection_applied": False,
                "hard_limit_violation_count": int(np.count_nonzero(hard)),
                "maximum_arm_hard_limit_excursion_rad": arm_excess,
                "maximum_dex3_hard_limit_excursion_rad": dex3_excess,
                "preexisting_measured_dex3_excursion_cap_rad": dex3_initial_cap,
                "reset_limit_eligible_under_preexisting_measured_state_contract": reset_limit_eligible,
                "microscopic_dex3_excursion_logged_not_commanded": bool(
                    arm_excess <= 1e-9 and 0.0 < dex3_excess <= dex3_initial_cap
                ),
                "hard_collision_incidence": collision["invalid_hard_self_collision_incidence"],
                "collision_category_frame_counts": collision["frame_counts"],
            }
        entries.append(
            {
                "heldout_output_episode": output_episode,
                "source_final_episode": int(source["final_dataset_index"]),
                "stable_episode_id": source["stable_episode_id"],
                "source_recording_id": source["original_source_recording_id"],
                "frames": int(source["frames"]),
                "source_video": artifact(Path(source["source_rgb_identity"]["canonical_video_path"])),
                "methods": methods,
            }
        )

    previous = preserve_previous()
    contract = {
        "schema_version": "paper_core_method_consistent_initialization_v1",
        "status": "METHOD_CONSISTENT_INITIALIZATION_FROZEN_BEFORE_INFERENCE",
        "experiment_name": "METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT",
        "rule": {
            "act_a_initial_qpos": "Fair-A HELDOUT8 observation.state[0] for the matched source episode",
            "act_b_initial_qpos": "Proposed-B HELDOUT8 observation.state[0] for the matched source episode",
            "deterministic": True,
            "episode_specific": True,
            "method_consistent": True,
            "policy_prediction_or_performance_used": False,
            "initial_q_modified_or_projected": False,
            "state_semantics": "frame 0 observation.state equals the same method's absolute action[0]",
        },
        "joint_order": names,
        "episode_count_per_method": 8,
        "settle_duration_seconds": 1.0,
        "tolerances": tolerances,
        "frame0_gate": {
            "reference": "exact frozen method/episode observation.state[0], not a shared midpoint and not a projected value",
            "maximum_joint_step_rad": 0.15,
            "threshold_source": str(ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"),
            "failure_status": "FRAME0_INELIGIBLE",
            "execute_on_failure": False,
        },
        "identical_across_methods": [
            "source episode identity",
            "complete source ALOHA RGB video bytes",
            "source timestamps and unpaused clock",
            "G1 root pose",
            "Isaac scene/task geometry",
            "30 Hz control rate",
            "official ACT-E1 temporal ensemble coefficient 0.01",
            "common deployment safety adapter",
            "hard limits",
            "collision model",
            "evaluation metrics",
        ],
        "execution_config": artifact(EXECUTION),
        "common_deployment_projection": artifact(PROJECTION),
        "heldout_manifest": artifact(HELDOUT),
        "datasets": dataset,
        "entries": entries,
        "previous_common_state_result": previous,
        "real_hardware": "DISABLED",
    }
    freeze_json(CONTRACT, contract)
    freeze_json(
        FREEZE,
        {
            "schema_version": "paper_core_method_consistent_initialization_freeze_v1",
            "status": "FROZEN",
            "contract": str(CONTRACT),
            "contract_sha256": sha256_file(CONTRACT),
            "entries_canonical_sha256": canonical_sha256(entries),
        },
    )

    frozen_inputs = {
        "table1": [
            artifact(PAPER / f"tables/table1_full50_retargeting.{suffix}")
            for suffix in ("csv", "md", "json")
        ],
        "table2": [
            artifact(PAPER / f"tables/table2_heldout_act_prediction.{suffix}")
            for suffix in ("csv", "md", "json")
        ],
        "act_a_selected_checkpoint": {
            "path": str(PAPER / "act_a40/train/checkpoints/100000/pretrained_model"),
            "model_sha256": "7e9fe737c3fd8ad3919cf3887dad58a732f6651e84e1eab7c1af8267ee16912c",
        },
        "act_b_selected_checkpoint": {
            "path": str(PAPER / "act_b40/train/checkpoints/020000/pretrained_model"),
            "model_sha256": "4c3c52a853cc242c6ba97fa6fa8d2dde96f6d65e737e291cfb99265f3c1b5198",
        },
    }
    evaluation = {
        "schema_version": "paper_core_method_consistent_evaluation_contract_v1",
        "status": "FROZEN_BEFORE_METHOD_CONSISTENT_INFERENCE",
        "experiment_name": "METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT",
        "episodes": "all HELDOUT8 for both ACT-A40 and ACT-B40",
        "initialization_contract": artifact(CONTRACT),
        "semantic_contract": artifact(SEMANTIC),
        "geometry_and_handoff_metrics": "unchanged definitions from outputs/paper_core_ab/source_rollout_evaluation_contract.json",
        "semantic_task_sequence": "frozen configs/paper_semantic_task_sequence_v1.json; same detector for A/B",
        "relative_path_length": "candidate combined bilateral whole-hand path divided by frozen source combined bilateral interaction-frame path",
        "SWPE": "semantic success * L_ref / max(L_pred,L_ref); explicitly semantic, never physical",
        "primary_pairing": "all exact source identities; no performance-based exclusions",
        "ineligible_rule": "FRAME0_INELIGIBLE remains in denominator and is never executed",
        "table3_update_rule": "update outputs/paper_core_ab/tables/table3_source_conditioned_rollout only if all 16 method/episode rollouts PASS; otherwise preserve prior files byte-exact",
        "frozen_inputs": frozen_inputs,
        "tables_1_and_2_must_remain_byte_exact": True,
        "real_hardware": False,
    }
    freeze_json(EVALUATION, evaluation)
    print(
        json.dumps(
            {
                "status": contract["status"],
                "contract": str(CONTRACT),
                "contract_sha256": sha256_file(CONTRACT),
                "evaluation_contract": str(EVALUATION),
                "evaluation_contract_sha256": sha256_file(EVALUATION),
                "episodes": len(entries),
                "previous_common_state_file_count": previous["previous_output_file_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
