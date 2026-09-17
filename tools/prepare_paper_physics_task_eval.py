#!/usr/bin/env python3
"""Freeze all pre-existing inputs before rigid-proxy calibration or policy inference."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/paper_physics_task_eval"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EXPERIMENT2 = ROOT / "outputs/paper_core_ab/offline_heldout8/experiment2_result.json"
CONFIG = ROOT / "configs/paper_eval_doll_rigid_proxy_v1.json"
SUCCESS = ROOT / "configs/paper_physical_success_definition_v1.json"
REQUIRED_DIRS = (
    "environment_calibration",
    "frozen_environment",
    "teacher_forced_trajectories",
    "preflight",
    "physics_replay",
    "success_metrics",
    "videos",
    "figures",
    "tables",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def canonical_hash(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['sha256']} {row['bytes']} {row['path']}" for row in sorted(records, key=lambda x: x["path"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        unexpected = [path for path in OUTPUT.rglob("*") if path.is_file()]
        if unexpected:
            raise FileExistsError(f"refusing to reuse populated experiment root: {OUTPUT}")
    heldout = read_json(HELDOUT)
    experiment2 = read_json(EXPERIMENT2)
    indices = [int(row["final_dataset_index"]) for row in heldout["entries"]]
    if indices != [2, 13, 23, 27, 28, 31, 37, 40]:
        raise RuntimeError(f"heldout identities changed: {indices}")
    if heldout.get("status") != "PASS" or experiment2.get("status") != "PASS":
        raise RuntimeError("paper-core heldout/Experiment-2 result is not frozen PASS")

    for name in REQUIRED_DIRS:
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)

    paper = ROOT / "outputs/paper_core_ab"
    common_records = [
        record(paper / "common48_manifest.json"),
        record(paper / "train40_manifest.json"),
        record(HELDOUT),
        record(EXPERIMENT2),
        record(paper / "preparation_result.json"),
    ]
    table_records = [
        record(path)
        for stem in ("table1_full50_retargeting", "table2_heldout_act_prediction")
        for path in sorted((paper / "tables").glob(f"{stem}.*"))
    ]
    if len(table_records) != 6:
        raise RuntimeError("expected exactly three frozen files for each of Tables 1 and 2")

    checkpoint_records: dict[str, Any] = {}
    checkpoint_files: list[dict[str, Any]] = []
    for method in ("a", "b"):
        selection = experiment2["methods"][method]["checkpoint_selection"]
        checkpoint = Path(selection["selected_checkpoint"]).resolve()
        files = [record(path) for path in sorted(checkpoint.iterdir()) if path.is_file()]
        model = next(row for row in files if Path(row["path"]).name == "model.safetensors")
        if model["sha256"] != selection["selected_model_sha256"]:
            raise RuntimeError(f"ACT-{method.upper()} selected checkpoint changed")
        checkpoint_files.extend(files)
        checkpoint_records[method] = {
            "label": f"ACT-{method.upper()}40",
            "path": str(checkpoint),
            "selected_step": int(selection["selected_step"]),
            "model_sha256": model["sha256"],
            "files": files,
        }

    source_and_trajectory: list[dict[str, Any]] = []
    episode_identity = []
    for output_episode, entry in enumerate(heldout["entries"]):
        source = record(Path(entry["source_rgb_identity"]["canonical_video_path"]))
        a_trajectory = record(Path(entry["a_trajectory_path"]))
        b_trajectory = record(Path(entry["b_trajectory_path"]))
        if source["sha256"] != entry["source_rgb_identity"]["canonical_video_sha256"]:
            raise RuntimeError("heldout source RGB changed")
        source_and_trajectory.extend((source, a_trajectory, b_trajectory))
        episode_identity.append(
            {
                "heldout_output_episode": output_episode,
                "source_final_episode": int(entry["final_dataset_index"]),
                "stable_episode_id": entry["stable_episode_id"],
                "frames": int(entry["frames"]),
                "source_video": source,
                "fair_a_trajectory": a_trajectory,
                "proposed_b_trajectory": b_trajectory,
            }
        )

    scene_records = [
        record(ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"),
        record(ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_scene.usda"),
        record(ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"),
        record(ROOT / "configs/doll_handoff_retargeting/dex3_whole_hand.sim.json"),
        record(ROOT / "outputs/dataset_a_final50_retargeting/config/tool_frame_report.json"),
        record(ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"),
        record(ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"),
    ]
    new_contracts = [record(CONFIG), record(SUCCESS)]
    all_records = [
        *common_records,
        *table_records,
        *checkpoint_files,
        *source_and_trajectory,
        *scene_records,
        *new_contracts,
    ]
    manifest = {
        "schema_version": "paper_physics_task_eval_preexperiment_freeze_v1",
        "status": "FROZEN_BEFORE_POLICY_INDEPENDENT_CALIBRATION_AND_BEFORE_POLICY_PHYSICS_RESULTS",
        "experiment_name": "PHYSICS_BASED_POLICY_TRAJECTORY_TASK_EVALUATION",
        "heldout_source_indices": indices,
        "episode_count": 8,
        "methods": checkpoint_records,
        "episodes": episode_identity,
        "paper_core_manifests_and_results": common_records,
        "frozen_tables_1_and_2": table_records,
        "scene_and_safety_dependencies": scene_records,
        "prepolicy_physics_contracts": new_contracts,
        "canonical_inventory_sha256": canonical_hash(all_records),
        "policy_physics_results_consulted": False,
        "retargeting_modified": False,
        "checkpoints_modified": False,
        "tables_1_and_2_modified": False,
        "real_hardware": False,
        "required_output_directories": [str(OUTPUT / name) for name in REQUIRED_DIRS],
    }
    atomic_json(OUTPUT / "PREEXPERIMENT_FREEZE_MANIFEST.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "output": str(OUTPUT),
        "inventory_sha256": manifest["canonical_inventory_sha256"],
        "act_a": checkpoint_records["a"]["model_sha256"],
        "act_b": checkpoint_records["b"]["model_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
