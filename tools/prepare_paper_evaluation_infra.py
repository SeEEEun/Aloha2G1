#!/usr/bin/env python3
"""Generate CPU-only paper-evaluation contracts, NA tables, and readiness records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.calibrate_dex3_rigid_doll_grasp import validate_config
from tools.evaluation.contracts import (
    AUTHORITATIVE_REFERENCES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CANONICAL_PHASES,
    FPS,
    sha256_file,
)
from tools.evaluation.io import atomic_json
from tools.evaluation.physical_success import event_log_schema
from tools.evaluation.tables import generate_tables
from tools.freeze_doll_handoff_rigid_proxy import pending_manifest


DEFAULT_OUTPUT = ROOT / "outputs/paper_metrics"
CONFIG = ROOT / "configs/doll_handoff_rigid_proxy_v1.json"
CONFIG_MANIFEST = ROOT / "configs/doll_handoff_rigid_proxy_v1.sha256.json"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def bundle_contract() -> dict:
    return {
        "schema_version": "paper_evaluation_bundle_contract_v1",
        "bundle_schema_version": "paper_evaluation_bundle_v1",
        "identity_rule": "A and B must have identical unique source_episode_id sets; unmatched comparison raises an error",
        "fps_hz": FPS,
        "evaluation_modes": ["retargeting", "offline_act", "source_conditioned", "physical"],
        "success_kinds": {
            "non_physical": "SEMANTIC_SUCCESS",
            "physical": "PHYSICAL_SUCCESS",
            "never_mix": True,
        },
        "canonical_phase_order": list(CANONICAL_PHASES),
        "episode_manifest_fields": {
            "source_episode_id": "required exact frozen identity",
            "arrays_path": "required NPZ path",
            "fps": "optional; must equal 30",
            "annotations": "event frames plus explicit authoritativeness flags",
            "feasibility": "existing validated diagnostics; no collision recomputation",
            "provenance": "frame/geometry/orientation authority declarations",
        },
        "supported_npz_arrays": {
            "reference_{left,right}_wrist_position_m": "float [T,3]",
            "candidate_{left,right}_wrist_position_m": "float [T,3]",
            "reference_{left,right}_wrist_rotation": "optional authoritative float [T,3,3]",
            "candidate_{left,right}_wrist_rotation": "optional float [T,3,3]",
            "reference_{left,right}_whole_hand_position_m": "float [T,3]",
            "candidate_{left,right}_whole_hand_position_m": "float [T,3]",
            "reference_action_rad/candidate_action_rad": "float [T,28]",
            "reference_action_chunk_rad/candidate_action_chunk_rad": "float [N,K,28]",
            "action_chunk_valid_lengths": "optional integer [N], excludes padded tail frames",
            "candidate_q_rad": "float [T,28]",
            "candidate_q_chunk_rad": "float [N,K,28], derivatives never cross chunk boundaries",
            "feasibility_projection_m": "float [T] or [T,arm]; magnitudes only",
        },
        "missing_input_rule": "return status=NA with reason; never fabricate reference/contact semantics",
        "authoritative_references": {key: str(value) for key, value in AUTHORITATIVE_REFERENCES.items()},
    }


def metric_catalog() -> dict:
    ready = {
        "Task success": "READY",
        "PCS": "READY",
        "Wrist": "READY",
        "Whole-hand": "READY",
        "Bimanual": "READY",
        "Handoff ordering": "READY",
        "Action RMSE": "READY",
        "NRMSE": "READY",
        "RPL": "READY",
        "SWPE": "READY",
        "Jerk": "READY",
        "Feasibility": "READY",
    }
    return {
        "schema_version": "paper_metric_catalog_v1",
        "status": "READY",
        "metrics": ready,
        "timing_hz": FPS,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "interval": "paired episode bootstrap percentile 95% CI of mean B-A",
            "supplementary_test": "exact two-sided paired sign test",
        },
        "units": {
            "Cartesian position": "mm",
            "SO(3) geodesic orientation": "deg",
            "action prediction": "rad",
            "NRMSE": "percent of authoritative joint range",
            "path lengths": "m",
            "handoff margin": "frames and s",
        },
    }


def future_commands() -> str:
    python = "/home/jbnu/miniconda3/envs/isaaclab6/bin/python"
    cpu_python = "/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python"
    config = str(CONFIG)
    root = str(ROOT)
    lines = [
        "# Deferred Isaac/GPU commands",
        "",
        "Do not run these commands until the active ACT experiment has ended and GPU use is explicitly authorized.",
        "The physical batch also refuses to start if `nvidia-smi` reports an existing compute process.",
        "",
        "## 1. Rigid-object physics smoke",
        "",
        "The MEDIUM candidate is used only as the middle-of-range smoke material; this does not select the frozen material.",
        "",
        "```bash",
        f"{python} {root}/tools/run_doll_handoff_rigid_proxy_smoke.py --config {config} --material MEDIUM --output-dir {root}/outputs/paper_metrics/rigid_proxy/smoke_medium --headless",
        "```",
        "",
        "## 2. Left fixed-grasp calibration (exact bounded LOW/MEDIUM/HIGH set)",
        "",
        "```bash",
    ]
    for material in ("LOW", "MEDIUM", "HIGH"):
        lines.append(
            f"{python} {root}/tools/calibrate_dex3_rigid_doll_grasp.py run-isaac --config {config} --side left --material {material} --output-dir {root}/outputs/paper_metrics/rigid_proxy/calibration/{material}/left --headless"
        )
    lines.extend(["```", "", "## 3. Right fixed-grasp calibration (same set and primitive)", "", "```bash"])
    for material in ("LOW", "MEDIUM", "HIGH"):
        lines.append(
            f"{python} {root}/tools/calibrate_dex3_rigid_doll_grasp.py run-isaac --config {config} --side right --material {material} --output-dir {root}/outputs/paper_metrics/rigid_proxy/calibration/{material}/right --headless"
        )
    result_paths = " ".join(
        f"{root}/outputs/paper_metrics/rigid_proxy/calibration/{material}/{side}/calibration_result.json"
        for material in ("LOW", "MEDIUM", "HIGH")
        for side in ("left", "right")
    )
    summary = f"{root}/outputs/paper_metrics/rigid_proxy/calibration_summary.json"
    lines.extend(
        [
            "```",
            "",
            "## 4. Summarize and freeze",
            "",
            "The predeclared rule selects the lowest-friction candidate with artifact-free passes for both hands. If none passes, freezing fails and policy evaluation remains disabled.",
            "",
            "```bash",
            f"{python} {root}/tools/calibrate_dex3_rigid_doll_grasp.py summarize --results {result_paths} --output {summary}",
            f"{python} {root}/tools/freeze_doll_handoff_rigid_proxy.py freeze-summary --config {config} --summary {summary} --output-config {config} --manifest {root}/configs/doll_handoff_rigid_proxy_v1.sha256.json",
            "```",
            "",
            "## 5. Serial ACT-A/B physical rollouts",
            "",
            "This executes all eight held-out source identities for A and B with one shared runner/config and no rendering.",
            "",
            "```bash",
            f"{python} {root}/tools/run_all_paper_physical_doll_rollouts.py --config {config} --config-manifest {root}/configs/doll_handoff_rigid_proxy_v1.sha256.json --output-root {root}/outputs/paper_metrics/physical_rollouts",
            "```",
            "",
            "## 6. CPU scoring after results exist",
            "",
            "```bash",
            f"CUDA_VISIBLE_DEVICES='' {cpu_python} {root}/tools/evaluate_paper_metrics.py paper-core-offline",
            f"CUDA_VISIBLE_DEVICES='' {cpu_python} {root}/tools/evaluate_paper_metrics.py paper-core-source",
            f"CUDA_VISIBLE_DEVICES='' {cpu_python} {root}/tools/evaluate_paper_metrics.py physical-batch --rollout-root {root}/outputs/paper_metrics/physical_rollouts",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    static = validate_config(CONFIG)
    atomic_json(output / "rigid_proxy/static_validation.json", static)
    atomic_json(CONFIG_MANIFEST, pending_manifest(CONFIG))
    atomic_json(output / "contracts/evaluation_bundle_v1.json", bundle_contract())
    atomic_json(output / "contracts/physical_event_log_v1.json", event_log_schema())
    atomic_json(output / "contracts/metric_catalog_v1.json", metric_catalog())
    paired_paths = {
        "table1_retargeting_quality": output / "retargeting/paired_statistics.json",
        "table2_downstream_act_policy": output / "offline_act/paired_statistics.json",
        "table3_source_conditioned_rollout": output / "source_conditioned/paired_statistics.json",
        "table4_isaac_physical_task_success": output / "physical/paired_statistics.json",
    }
    paired = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in paired_paths.items()
        if path.is_file()
    }
    table_manifest = generate_tables(output / "tables", paired)
    atomic_json(output / "tables/table_manifest.json", table_manifest)
    _atomic_text(output / "FUTURE_GPU_COMMANDS.md", future_commands())
    primitive_status = {}
    config_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected_primitive_config_sha = (
        config_payload.get("freeze", {}).get("candidate_config_sha256")
        if bool(config_payload.get("freeze", {}).get("frozen"))
        else static["config_sha256"]
    )
    for side in ("left", "right"):
        path = output / f"rigid_proxy/primitives/{side}_fixed_grasp_primitive.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            ready = (
                row.get("status") == "READY"
                and row.get("config_sha256") == expected_primitive_config_sha
                and Path(row.get("trajectory", "")).is_file()
                and sha256_file(Path(row["trajectory"])) == row.get("trajectory_sha256")
            )
        else:
            ready = False
        primitive_status[side] = "READY" if ready else "NOT_READY"
    infrastructure_ready = all(value == "READY" for value in primitive_status.values())
    atomic_json(
        output / "infrastructure_readiness.json",
        {
            "schema_version": "paper_evaluation_infrastructure_readiness_v1",
            "status": (
                "PAPER_METRICS_AND_PHYSICS_EVAL_INFRA_READY"
                if infrastructure_ready
                else "PAPER_EVAL_INFRA_BLOCKED"
            ),
            "metric_suite": metric_catalog()["metrics"],
            "paired_statistics": {
                "status": "READY",
                "seed": BOOTSTRAP_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
            },
            "paper_tables": {"1": "READY", "2": "READY", "3": "READY", "4_optional": "READY"},
            "rigid_proxy": {
                "config": str(CONFIG),
                "config_sha256": static["config_sha256"],
                "status": "PRECALIBRATION_NOT_FROZEN",
                "policy_evaluation_allowed": False,
            },
            "fixed_grasp_primitives": primitive_status,
            "isaac_execution": "NOT_STARTED_BY_DESIGN",
            "gpu_used": False,
            "current_paper_core_job_touched": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
