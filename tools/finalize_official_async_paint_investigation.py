#!/usr/bin/env python3
"""Finalize the bounded async/PAINT investigation after its mandatory stop gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/policy_execution_stability_official_async_paint"
AUDIT_PATH = OUTPUT / "single_raw_chunk/analysis/single_raw_chunk_smoothness.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    conclusion = audit["conclusion"]
    if conclusion["single_raw_chunk_smooth"] is not False:
        raise RuntimeError("single-chunk stop condition was not established")
    for key in (
        "stop_before_official_async_queue",
        "stop_before_paint",
    ):
        if conclusion[key] is not True:
            raise RuntimeError(f"mandatory stop flag is not set: {key}")

    raw_video_root = OUTPUT / "single_raw_chunk/comparison"
    comparison_videos = {
        path.stem: record(path) for path in sorted(raw_video_root.glob("*.mp4"))
    }
    if len(comparison_videos) != 8:
        raise RuntimeError("expected four normal-speed and four slow-motion raw-chunk videos")

    single_runs = {}
    for row in audit["records"]:
        label = row["label"]
        stage = OUTPUT / "single_raw_chunk" / label / "stage1_object_free_single"
        noise = Path(row["flow_noise"]["path"])
        single_runs[label] = {
            "stage_report": record(stage / "stage_report.json"),
            "raw_chunks": record(stage / "inference_chunks.npz"),
            "rollout_trace": record(stage / "rollout_trace.npz"),
            "episode_persistent_flow_noise": record(noise),
            "flow_noise_tensor_sha256": row["flow_noise"]["sha256"],
            "raw_chunk_sha256": row["raw_chunk_sha256"],
            "inference_calls": row["inference_calls"],
            "executed_frames": row["executed_frames"],
            "replanning": row["replanning"],
        }

    manifest = {
        "schema_version": "final_bounded_official_async_paint_investigation_v1",
        "status": "BLOCKED_BY_POLICY_LEVEL_NONSMOOTHNESS",
        "decision_tree_stop_stage": "FIXED_NOISE_SINGLE_RAW_CHUNK_SMOOTHNESS",
        "single_raw_chunk_smooth": False,
        "single_raw_chunk_audit": record(AUDIT_PATH),
        "single_raw_chunk_report": record(
            OUTPUT / "single_raw_chunk/analysis/single_raw_chunk_smoothness.md"
        ),
        "single_raw_chunk_plots": {
            "positions": record(
                OUTPUT / "single_raw_chunk/analysis/single_raw_chunk_arm_positions.png"
            ),
            "velocity": record(
                OUTPUT / "single_raw_chunk/analysis/single_raw_chunk_arm_velocity.png"
            ),
        },
        "single_raw_chunk_runs": single_runs,
        "raw_single_chunk_comparison_videos": comparison_videos,
        "preservation_manifest": record(OUTPUT / "preservation_manifest.json"),
        "all_prior_execution_adapters": "DIAGNOSTIC_ONLY_NOT_REAL_ROBOT_APPROVED",
        "official_async_parity": {
            "formal_audit_completed": False,
            "reason": "NOT_RUN_AFTER_MANDATORY_SINGLE_RAW_CHUNK_STOP",
        },
        "official_async_s1": {
            "run": False,
            "reason": "FORBIDDEN_BY_SINGLE_RAW_CHUNK_STOP_RULE",
        },
        "official_async_s2": {
            "run": False,
            "reason": "FORBIDDEN_BY_SINGLE_RAW_CHUNK_STOP_RULE",
        },
        "paint_compatibility": {
            "tested": False,
            "reason": "FORBIDDEN_BY_SINGLE_RAW_CHUNK_STOP_RULE",
        },
        "paint_pilot": {
            "run": False,
            "reason": "FORBIDDEN_BY_SINGLE_RAW_CHUNK_STOP_RULE",
        },
        "selected_execution_adapter": None,
        "final_execution_decision": "NONE",
        "next_method_class": conclusion["next_method_class"],
        "deferred_methods_not_implemented": [
            "Legato retraining",
            "ABPolicy B-spline replacement",
            "A2C2 correction-head training",
            "REMAC training",
            "full action-representation redesign",
        ],
        "policy_b_retrained_or_modified": False,
        "dataset_b_modified": False,
        "camera_modified": False,
        "retargeted_actions_modified": False,
        "policy_a_touched": False,
        "real_hardware_safety_readiness": "BLOCKED",
        "real_robot_command_allowed": False,
        "real_g1_or_dex3_commands_sent": False,
        "implementation": {
            str((ROOT / "tools/policy_b_inference_worker.py").resolve()): sha256_file(
                ROOT / "tools/policy_b_inference_worker.py"
            ),
            str((ROOT / "tools/run_policy_b_isaac_doll_handoff.py").resolve()): sha256_file(
                ROOT / "tools/run_policy_b_isaac_doll_handoff.py"
            ),
            str((ROOT / "tools/analyze_policy_b_single_raw_chunk_smoothness.py").resolve()): sha256_file(
                ROOT / "tools/analyze_policy_b_single_raw_chunk_smoothness.py"
            ),
            str((ROOT / "tools/prepare_official_async_paint_investigation.py").resolve()): sha256_file(
                ROOT / "tools/prepare_official_async_paint_investigation.py"
            ),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
    }

    lines = [
        "# Final bounded smooth-execution investigation",
        "",
        "## SINGLE RAW CHUNK",
        "",
        "smooth: **NO**",
        "",
        "internal jitter: present in all three fixed-observation/fixed-noise chunks without replanning. Mean arm reversal rates were 12.988/s (initial), 12.770/s (left approach), and 14.431/s (doll plateau). The plateau chunk reached 0.098468 rad/frame, 89.601 rad/s², and 5208.112 rad/s³; its Isaac replay reached 29.610 N table contact.",
        "",
        "## OFFICIAL ASYNC PARITY",
        "",
        "queue implementation: NOT IMPLEMENTED — mandatory single-chunk stop",
        "",
        "observation filter: NOT IMPLEMENTED",
        "",
        "must_go: NOT IMPLEMENTED",
        "",
        "absolute timestamp alignment: NOT IMPLEMENTED",
        "",
        "official aggregator reuse: NOT IMPLEMENTED",
        "",
        "## S1 G=0.5",
        "",
        "policy calls: NOT_RUN",
        "",
        "skipped observations: NOT_RUN",
        "",
        "queue behavior: NOT_RUN",
        "",
        "low-motion metrics: NOT_RUN",
        "",
        "visible oscillation: NOT_RUN",
        "",
        "table contact: NOT_RUN",
        "",
        "## S2 G=0.7",
        "",
        "policy calls: NOT_RUN",
        "",
        "skipped observations: NOT_RUN",
        "",
        "queue behavior: NOT_RUN",
        "",
        "low-motion metrics: NOT_RUN",
        "",
        "visible oscillation: NOT_RUN",
        "",
        "table contact: NOT_RUN",
        "",
        "## BEST OFFICIAL-ASYNC RESULT",
        "",
        "selected: NONE",
        "",
        "reason: the raw policy representation failed the prerequisite smoothness gate before queue semantics could be evaluated.",
        "",
        "## PAINT COMPATIBILITY",
        "",
        "tested: NO",
        "",
        "SmolVLA initial-noise injection: NOT_AUDITED_AFTER_STOP",
        "",
        "prefix locality: NOT_TESTED",
        "",
        "pilot run: NOT_RUN",
        "",
        "## PAINT RESULT",
        "",
        "visible oscillation: NOT_RUN",
        "",
        "low-motion metrics: NOT_RUN",
        "",
        "table contact: NOT_RUN",
        "",
        "latency: NOT_RUN",
        "",
        "## FINAL EXECUTION DECISION",
        "",
        "selected adapter: **NONE**",
        "",
        "REAL HARDWARE: **BLOCKED**",
        "",
        "NEXT METHOD CLASS IF BLOCKED: policy-side training-time prefix/action conditioning, final helmet-D455 XR adaptation, or a smooth action representation.",
        "",
        "Final status: `BLOCKED_BY_POLICY_LEVEL_NONSMOOTHNESS`",
    ]
    report_path = OUTPUT / "final_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["final_report"] = record(report_path)
    manifest_path = OUTPUT / "FINAL_OFFICIAL_ASYNC_PAINT_INVESTIGATION_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
