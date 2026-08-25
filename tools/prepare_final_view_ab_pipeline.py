#!/usr/bin/env python3
"""Audit and materialize the equal A/B final-helmet relabeling command plan.

This is deliberately a dry planner: it does not render, train, run a rollout,
or touch any dataset/checkpoint.  Operational steps stay gated on one frozen
FROZEN_HELMET_D455_FINAL camera config and structurally valid A and B inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from deployment_camera_config import camera_manifest_record, load_camera_config, sha256_file  # noqa: E402


REQUIRED_INTERFACES = {
    "prepare_plan": ("tools/prepare_doll_handoff_g1visual_render_plan.py", ("--dataset-variant", "--camera-config", "--source-dataset", "--trajectories")),
    "renderer": ("tools/render_doll_handoff_g1visual_dataset.py", ("--dataset-variant", "--camera-config", "--plans", "--output")),
    "package": ("tools/package_doll_handoff_g1visual_lerobot.py", ("--dataset-variant", "--source", "--renders", "--destination")),
    "readback": ("tools/validate_doll_handoff_dataset_b_lerobot.py", ("--dataset", "--output", "--repo-id")),
    "paired_rehearsal": ("tools/build_paired_visual_rehearsal_dataset.py", ("--policy-variant", "--camera-config", "--source-dataset", "--final-view-dataset")),
    "adaptation": ("tools/prepare_equal_ab_final_view_adaptation.py", ("--camera-config", "--pipeline-config")),
    "phase_probe": ("tools/probe_policy_9phase.py", ("--policy-variant", "--dataset", "--checkpoint", "--trajectories")),
    "rollout": ("tools/run_policy_isaac_doll_handoff.py", ()),
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_gate(path: Path, sources: list[dict[str, Any]], shared: dict[str, Any]) -> dict[str, Any]:
    if not path.is_dir():
        return {"ready": False, "path": str(path), "reason": "DATASET_NOT_AVAILABLE"}
    episodes = pq.read_table(path / "meta/episodes/chunk-000/file-000.parquet")
    data = pq.read_table(path / "data/chunk-000/file-000.parquet")
    lengths = [int(row["length"]) for row in episodes.to_pylist()]
    expected_lengths = [int(row["source_frame_count"]) for row in sources]
    state_type = data.schema.field("observation.state").type
    action_type = data.schema.field("action").type
    checks = {
        "episodes_50": episodes.num_rows == int(shared["episodes"]),
        "frames_exact": data.num_rows == int(shared["frames"]),
        "source_episode_frame_identity": lengths == expected_lengths,
        "state_28": getattr(state_type, "list_size", None) == int(shared["state_dimension"]),
        "action_28": getattr(action_type, "list_size", None) == int(shared["action_dimension"]),
    }
    return {"ready": all(checks.values()), "path": str(path), "checks": checks}


def command(*parts: Any) -> str:
    return " ".join(str(part) for part in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", type=Path, default=ROOT / "configs/final_view_ab_pipeline.json")
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pre_mount_readiness/final_view_ab/pipeline_dry_run.json")
    args = parser.parse_args()
    config = read_json(args.pipeline_config)
    manifest_path = ROOT / config["common_source_manifest"]
    if sha256_file(manifest_path) != config["common_source_manifest_sha256"]:
        raise RuntimeError("final common 50-source manifest hash changed")
    sources = read_json(manifest_path)["episodes"]
    camera_path = args.camera_config or ROOT / config["camera_config"]
    camera = load_camera_config(camera_path, purpose="A/B final-view pipeline dry-run", allow_pending=True)
    interface_checks = {}
    for name, (relative, flags) in REQUIRED_INTERFACES.items():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        interface_checks[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "required_cli_tokens": {flag: flag in text for flag in flags},
            "pass": path.is_file() and all(flag in text for flag in flags),
        }
    if not all(row["pass"] for row in interface_checks.values()):
        raise RuntimeError("one or more final-view pipeline interfaces are missing")

    shared = config["shared_contract"]
    variant_reports: dict[str, Any] = {}
    for variant in ("A", "B"):
        row = config["dataset_variants"][variant]
        source_dataset = (ROOT / row["source_dataset"]).resolve()
        final_dataset = (ROOT / row["final_view_dataset"]).resolve()
        plan_dir = (ROOT / f"outputs/final_helmet_view/policy_{variant.lower()}/render_plans").resolve()
        render_dir = (ROOT / f"outputs/final_helmet_view/policy_{variant.lower()}/renders").resolve()
        paired = (ROOT / row["paired_dataset"]).resolve()
        checkpoint = (ROOT / row["source_checkpoint"]).resolve()
        trajectories = (ROOT / row["frozen_trajectories"]).resolve()
        variant_reports[variant] = {
            "source_gate": dataset_gate(source_dataset, sources, shared),
            "source_checkpoint_ready": (checkpoint / "model.safetensors").is_file(),
            "final_view_dataset_exists": final_dataset.exists(),
            "commands": {
                "1_prepare_phase_consistent_render_plan": command(
                    "/home/jbnu/miniconda3/envs/isaaclab6/bin/python", ROOT / "tools/prepare_doll_handoff_g1visual_render_plan.py",
                    "--dataset-variant", variant, "--source-dataset", source_dataset,
                    "--source-manifest", manifest_path, "--trajectories", trajectories,
                    "--camera-config", camera.path, "--output", plan_dir,
                ),
                "2_camera_config_driven_render": command(
                    "/home/jbnu/miniconda3/envs/isaaclab6/bin/python", ROOT / "tools/render_doll_handoff_g1visual_dataset.py",
                    "--dataset-variant", variant, "--camera-config", camera.path,
                    "--plans", plan_dir, "--output", render_dir, "--episodes", "all", "--headless",
                ),
                "3_lerobot_package_and_label_identity": command(
                    "/home/jbnu/miniconda3/envs/isaaclab6/bin/python", ROOT / "tools/package_doll_handoff_g1visual_lerobot.py",
                    "--dataset-variant", variant, "--source", source_dataset,
                    "--renders", render_dir, "--destination", final_dataset,
                ),
                "4_lerobot_readback": command(
                    "/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python", ROOT / "tools/validate_doll_handoff_dataset_b_lerobot.py",
                    "--dataset", final_dataset, "--repo-id", f"local/doll_handoff_policy_{variant.lower()}_final_helmet_50",
                    "--output", ROOT / f"outputs/final_helmet_view/policy_{variant.lower()}/lerobot_readback.json",
                ),
                "5_equal_paired_rehearsal_dataset": command(
                    "/home/jbnu/miniconda3/envs/isaaclab6/bin/python", ROOT / "tools/build_paired_visual_rehearsal_dataset.py",
                    "--policy-variant", variant, "--source-dataset", source_dataset,
                    "--final-view-dataset", final_dataset, "--camera-config", camera.path,
                    "--destination", paired,
                ),
                "6_common_9phase_probe": command(
                    "/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python", ROOT / "tools/probe_policy_9phase.py",
                    "--policy-variant", variant, "--dataset", final_dataset,
                    "--trajectories", trajectories, "--checkpoint", checkpoint,
                    "--source-manifest", manifest_path,
                    "--output", ROOT / f"outputs/final_helmet_view/policy_{variant.lower()}/phase_probe",
                ),
                "7_identical_closed_loop_rollout": command(
                    "/home/jbnu/miniconda3/envs/isaaclab6/bin/python", ROOT / "tools/run_policy_isaac_doll_handoff.py",
                    "--policy-variant", variant, "--camera-config", camera.path,
                    "--stage", "full-task",
                    "--checkpoint-override", f"<POLICY_{variant}_SELECTED_ADAPTED_CHECKPOINT>",
                    "--checkpoint-model-sha256", f"<POLICY_{variant}_SELECTED_MODEL_SHA256>",
                    "--checkpoint-training-step", "<SELECTED_STEP_500_1000_OR_1500>",
                    "--output-root", ROOT / f"outputs/final_helmet_view/policy_{variant.lower()}/rollout", "--headless",
                ),
            },
        }
    dataset_a_failure = ROOT / "outputs/pre_mount_readiness/dataset_a/dataset_a_validation.json"
    dataset_a_result = read_json(dataset_a_failure) if dataset_a_failure.is_file() else None
    both_sources = all(variant_reports[v]["source_gate"]["ready"] for v in ("A", "B"))
    both_checkpoints = all(variant_reports[v]["source_checkpoint_ready"] for v in ("A", "B"))
    ready = camera.is_final_helmet and both_sources and both_checkpoints
    if ready:
        status = "READY_FOR_MANUAL_FINAL_VIEW_PIPELINE_EXECUTION"
    elif dataset_a_result and dataset_a_result.get("status", "").startswith("HARD_FAIL"):
        status = "PREPARED_BLOCKED_BY_DATASET_A_HARD_FAIL_AND_FINAL_CAMERA"
    else:
        status = "PREPARED_DISABLED_PENDING_GATES"
    report = {
        "schema_version": "final_helmet_view_equal_ab_pipeline_dry_run_v1",
        "status": status,
        "camera": camera_manifest_record(camera),
        "final_camera_resolved": camera.is_final_helmet,
        "common_source_manifest": str(manifest_path),
        "common_source_manifest_sha256": sha256_file(manifest_path),
        "source_episode_ids": [row["stable_episode_id"] for row in sources],
        "interface_checks": interface_checks,
        "shared_contract": shared,
        "variants": variant_reports,
        "a_b_same_camera_config_sha256": True,
        "a_b_same_renderer": True,
        "a_b_same_frame_count": True,
        "a_b_same_task_schema_strategy_selection_rule": True,
        "adaptation_plan_command": command(
            "/home/jbnu/miniconda3/envs/isaaclab6/bin/python", ROOT / "tools/prepare_equal_ab_final_view_adaptation.py",
            "--pipeline-config", args.pipeline_config.resolve(), "--camera-config", camera.path,
        ),
        "final_50_episode_render_executed": False,
        "training_executed": False,
        "rollout_executed": False,
        "real_robot_invoked": False,
        "dataset_a_gate_report": str(dataset_a_failure) if dataset_a_result else None,
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps({
        "status": status,
        "final_camera_resolved": camera.is_final_helmet,
        "dataset_a_ready": variant_reports["A"]["source_gate"]["ready"],
        "dataset_b_ready": variant_reports["B"]["source_gate"]["ready"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
