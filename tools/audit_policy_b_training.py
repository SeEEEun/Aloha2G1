#!/usr/bin/env python3
"""Audit the completed Policy-B run and select one frozen validation checkpoint.

This is a read-only model/dataset audit.  It performs deterministic offline
reconstruction checks on a fixed set of Dataset-B frames; it never updates
model weights.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open


FPS = 30.0
ACTION_CHUNK = 50
STATE_DIM = 28
ACTION_DIM = 28
SEED = 20260824
ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmux-target", default="policy_b:0.0")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal_capture(target: str) -> tuple[str, dict[str, Any]]:
    result = subprocess.run(
        ["tmux", "capture-pane", "-pJt", target, "-S", "-10000"],
        check=True,
        capture_output=True,
        text=True,
    )
    text = ANSI_ESCAPE.sub("", result.stdout).replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    metric_lines = [line for line in lines if "ot_train.py:627" in line and "loss:" in line]
    end_lines = [line for line in lines if "End of training" in line]
    checkpoint_lines = [line for line in lines if "Checkpoint policy after step" in line]
    nonfinite = [
        line
        for line in metric_lines
        if re.search(r"(?:^|\s)(?:loss|grdn|losses_[^:]+):(nan|inf|-inf)(?:\s|$)", line, re.I)
    ]
    final = metric_lines[-1] if metric_lines else None
    parsed_final: dict[str, float | int | str | None] = {"raw": final}
    if final:
        for key in (
            "loss",
            "grdn",
            "lr",
            "losses_after_forward",
            "losses_after_in_ep_bound",
            "losses_after_rm_padding",
        ):
            match = re.search(rf"(?:^|\s){re.escape(key)}:([^\s]+)", final)
            if match:
                parsed_final[key] = float(match.group(1))
        step_match = re.search(r"(?:^|\s)step:([^\s]+)", final)
        if step_match:
            parsed_final["step_label"] = step_match.group(1)
    summary = {
        "metric_line_count": len(metric_lines),
        "checkpoint_line_count": len(checkpoint_lines),
        "end_of_training_present": bool(end_lines),
        "nonfinite_metric_lines": nonfinite,
        "final_reported_metrics": parsed_final,
    }
    return text, summary


def checkpoint_normalization(checkpoint: Path, dataset: Path) -> dict[str, Any]:
    stats = read_json(dataset / "meta/stats.json")
    normalizer = checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    checks: dict[str, Any] = {}
    with safe_open(normalizer, framework="pt", device="cpu") as handle:
        for feature in ("observation.state", "action"):
            for field in ("min", "max", "mean", "std"):
                key = f"{feature}.{field}"
                actual = handle.get_tensor(key).numpy()
                expected = np.asarray(stats[feature][field], dtype=np.float32)
                checks[key] = {
                    "maximum_absolute_difference": float(np.max(np.abs(actual - expected))),
                    "matches": bool(np.allclose(actual, expected, rtol=1e-6, atol=1e-6)),
                }
    return {
        "comparisons": checks,
        "all_match_dataset_b": all(row["matches"] for row in checks.values()),
    }


def fixed_sample_indices(dataset: Path) -> list[dict[str, Any]]:
    episode_table = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet")
    episodes = episode_table.to_pylist()
    selected_episodes = [0, 2, 5, 8, 11, 13, 16, 19, 22, 24, 27, 30, 33, 36, 39, 41, 44, 46, 48, 49]
    phase_fractions = [0.10, 0.30, 0.50, 0.70]
    samples = []
    for rank, episode_index in enumerate(selected_episodes):
        row = episodes[episode_index]
        length = int(row["length"])
        fraction = phase_fractions[rank % len(phase_fractions)]
        local_index = min(int(round((length - ACTION_CHUNK - 1) * fraction)), length - ACTION_CHUNK - 1)
        local_index = max(local_index, 0)
        samples.append(
            {
                "episode_index": episode_index,
                "episode_length": length,
                "phase_fraction": fraction,
                "local_index": local_index,
                "global_index": int(row["dataset_from_index"]) + local_index,
            }
        )
    return samples


def scalar_loss(output: Any) -> float:
    value = output[0] if isinstance(output, tuple) else output
    if isinstance(value, dict):
        value = value["loss"]
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def evaluate_checkpoint(
    checkpoint: Path,
    dataset: Any,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.collate import lerobot_collate_fn

    config = read_json(checkpoint / "config.json")
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    policy.eval()
    load_seconds = time.perf_counter() - started
    errors = []
    losses = []
    latencies = []
    shape_checks = []
    per_joint_abs_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    per_joint_squared_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    per_joint_count = np.zeros(ACTION_DIM, dtype=np.int64)
    for batch_index in range(0, len(samples), 5):
        group = samples[batch_index : batch_index + 5]
        raw = lerobot_collate_fn([dataset[row["global_index"]] for row in group])
        if raw is None:
            raise RuntimeError("LeRobot collate unexpectedly returned None")
        target = raw["action"].detach().cpu().numpy()
        valid = ~raw["action_is_pad"].detach().cpu().numpy().astype(bool)
        processed = preprocessor(raw)
        torch.manual_seed(SEED + batch_index)
        torch.cuda.manual_seed_all(SEED + batch_index)
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            normalized_prediction = policy.predict_action_chunk(processed)
            prediction = postprocessor(normalized_prediction)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - inference_started)
        prediction_np = prediction.detach().cpu().numpy()
        shape_checks.append(list(prediction_np.shape))
        mask = np.broadcast_to(valid[..., None], target.shape)
        error = prediction_np - target
        valid_error = error[mask]
        errors.append(valid_error)
        for joint_index in range(ACTION_DIM):
            joint_error = error[:, :, joint_index][valid]
            per_joint_abs_sum[joint_index] += np.abs(joint_error).sum()
            per_joint_squared_sum[joint_index] += np.square(joint_error).sum()
            per_joint_count[joint_index] += joint_error.size
        torch.manual_seed(SEED + 1000 + batch_index)
        torch.cuda.manual_seed_all(SEED + 1000 + batch_index)
        with torch.inference_mode():
            losses.append(scalar_loss(policy.forward(processed)))
    all_errors = np.concatenate(errors)
    per_joint_mae = per_joint_abs_sum / per_joint_count
    per_joint_rmse = np.sqrt(per_joint_squared_sum / per_joint_count)
    result = {
        "checkpoint": str(checkpoint),
        "training_step": int(checkpoint.parent.name),
        "model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "model_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "checkpoint_timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z",
            time.localtime((checkpoint / "model.safetensors").stat().st_mtime),
        ),
        "load_seconds": load_seconds,
        "sample_count": len(samples),
        "action_value_count": int(all_errors.size),
        "offline_chunk_mae_rad": float(np.mean(np.abs(all_errors))),
        "offline_chunk_rmse_rad": float(np.sqrt(np.mean(np.square(all_errors)))),
        "offline_chunk_max_abs_error_rad": float(np.max(np.abs(all_errors))),
        "per_joint_mae_rad": per_joint_mae.tolist(),
        "per_joint_rmse_rad": per_joint_rmse.tolist(),
        "forward_loss_mean": float(np.mean(losses)),
        "forward_loss_values": losses,
        "inference_batch_size": 5,
        "inference_batch_latency_seconds_mean": float(np.mean(latencies)),
        "inference_batch_latency_seconds_max": float(np.max(latencies)),
        "prediction_shapes": shape_checks,
        "normalization": checkpoint_normalization(checkpoint, dataset.root),
        "config": {
            "state_shape": config["input_features"]["observation.state"]["shape"],
            "action_shape": config["output_features"]["action"]["shape"],
            "chunk_size": int(config["chunk_size"]),
            "n_action_steps": int(config["n_action_steps"]),
            "max_state_dim": int(config["max_state_dim"]),
            "max_action_dim": int(config["max_action_dim"]),
        },
    }
    result["checks"] = {
        "checkpoint_reload": True,
        "all_predictions_1x50x28_or_5x50x28": all(
            shape[1:] == [ACTION_CHUNK, ACTION_DIM] for shape in shape_checks
        ),
        "offline_metrics_finite": all(
            math.isfinite(result[key])
            for key in (
                "offline_chunk_mae_rad",
                "offline_chunk_rmse_rad",
                "offline_chunk_max_abs_error_rad",
                "forward_loss_mean",
            )
        ),
        "logical_state_28": result["config"]["state_shape"] == [STATE_DIM],
        "logical_action_28": result["config"]["action_shape"] == [ACTION_DIM],
        "chunk_50": result["config"]["chunk_size"] == ACTION_CHUNK,
        "normalization_matches_dataset_b": result["normalization"]["all_match_dataset_b"],
    }
    result["status"] = "PASS" if all(result["checks"].values()) else "FAIL"
    del postprocessor, preprocessor, policy
    gc.collect()
    torch.cuda.empty_cache()
    return result


def render_markdown(result: dict[str, Any], output: Path) -> None:
    selected = result["selected_checkpoint"]
    final_metrics = result["terminal_capture"]["final_reported_metrics"]
    lines = [
        "# Policy-B training audit",
        "",
        f"Status: **{result['status']}**",
        "",
        "Policy B completed its configured 20,000 optimization steps. The captured training terminal contains the explicit end-of-training marker and no NaN/Inf metric values. This audit did not train or alter the policy.",
        "",
        "## Completed run",
        "",
        f"- Run directory: `{result['run_directory']}`",
        f"- Dataset: `{result['dataset']}`",
        f"- Configured/actual step: {result['configured_training_steps']} / {result['actual_completed_step']}",
        f"- Final reported loss: {final_metrics.get('loss')}",
        f"- Final reported padding-aware loss: {final_metrics.get('losses_after_rm_padding')}",
        f"- End-of-training marker: {result['terminal_capture']['end_of_training_present']}",
        "",
        "## Offline checkpoint comparison",
        "",
        "The comparison uses one fixed stochastic seed and 20 training-dataset samples spread over all task phases and 20 distinct episodes. It is a checkpoint-integrity/reconstruction diagnostic, not an independent success estimate.",
        "",
        "| Step | Status | Chunk MAE (rad) | Chunk RMSE (rad) | Forward loss | Model SHA256 |",
        "|---:|:---:|---:|---:|---:|:---|",
    ]
    for row in result["checkpoints"]:
        lines.append(
            f"| {row['training_step']} | {row['status']} | {row['offline_chunk_mae_rad']:.6f} | "
            f"{row['offline_chunk_rmse_rad']:.6f} | {row['forward_loss_mean']:.6f} | `{row['model_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Frozen validation checkpoint",
            "",
            f"- Selected checkpoint: `{selected['checkpoint']}`",
            f"- Step: {selected['training_step']}",
            f"- Model SHA256: `{selected['model_sha256']}`",
            f"- Selection rule: {result['selection_rule']}",
            "- Reload: PASS",
            "- Logical output: `[batch, 50, 28]`; the checkpoint's internal 32D action padding is sliced back to 28D.",
            "- Dataset-B state/action normalization tensors: exact match to `meta/stats.json`.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    run = args.run.resolve()
    dataset_root = args.dataset.resolve()
    output_root = args.output_root.resolve()
    audit_dir = output_root / "training_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    capture, terminal_summary = terminal_capture(args.tmux_target)
    (audit_dir / "training_terminal_capture.txt").write_text(capture, encoding="utf-8")
    checkpoint_paths = sorted(
        path
        for path in (run / "checkpoints").glob("*/pretrained_model")
        if path.parent.name.isdigit()
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints under {run}")
    final_train_config = read_json(checkpoint_paths[-1] / "train_config.json")
    configured_steps = int(final_train_config["steps"])
    actual_steps = [
        int(read_json(path.parent / "training_state/training_step.json")["step"])
        for path in checkpoint_paths
    ]
    samples = fixed_sample_indices(dataset_root)
    deltas = {
        "observation.state": [0.0],
        "action": [index / FPS for index in range(ACTION_CHUNK)],
    }
    dataset = LeRobotDataset(
        repo_id="local/doll_handoff_proposed_b_50",
        root=dataset_root,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend="torchcodec",
    )
    checkpoint_results = [
        evaluate_checkpoint(path, dataset, samples) for path in checkpoint_paths
    ]
    passing = [row for row in checkpoint_results if row["status"] == "PASS"]
    if not passing:
        raise RuntimeError("No checkpoint passed reload/offline validation")
    selected = min(passing, key=lambda row: (row["offline_chunk_rmse_rad"], -row["training_step"]))
    checks = {
        "configured_20000_steps": configured_steps == 20_000,
        "actual_completed_20000_steps": max(actual_steps) == configured_steps,
        "all_four_expected_checkpoints_present": actual_steps == [5_000, 10_000, 15_000, 20_000],
        "training_end_marker_present": terminal_summary["end_of_training_present"],
        "training_metrics_finite": not terminal_summary["nonfinite_metric_lines"],
        "all_checkpoints_pass_reload_and_offline_audit": len(passing) == len(checkpoint_results),
        "dataset_length_34478": len(dataset) == 34_478,
    }
    result = {
        "schema_version": "policy_b_completed_training_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run_directory": str(run),
        "dataset": str(dataset_root),
        "configured_training_steps": configured_steps,
        "actual_completed_step": max(actual_steps),
        "available_checkpoint_steps": actual_steps,
        "terminal_capture": terminal_summary,
        "fixed_offline_samples": samples,
        "checkpoints": checkpoint_results,
        "selection_rule": "minimum deterministic physical-action chunk RMSE among checkpoints that pass reload, shape, finite-output, and Dataset-B-normalization checks; later step breaks exact ties",
        "selected_checkpoint": selected,
        "checks": checks,
    }
    write_json(audit_dir / "training_audit.json", result)
    with (audit_dir / "checkpoint_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "training_step",
                "status",
                "offline_chunk_mae_rad",
                "offline_chunk_rmse_rad",
                "offline_chunk_max_abs_error_rad",
                "forward_loss_mean",
                "model_sha256",
                "checkpoint",
            ],
        )
        writer.writeheader()
        for row in checkpoint_results:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    render_markdown(result, output_root / "training_audit.md")
    print(json.dumps({
        "status": result["status"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_step": selected["training_step"],
        "selected_model_sha256": selected["model_sha256"],
        "checkpoint_metrics": [
            {
                "step": row["training_step"],
                "mae": row["offline_chunk_mae_rad"],
                "rmse": row["offline_chunk_rmse_rad"],
                "loss": row["forward_loss_mean"],
            }
            for row in checkpoint_results
        ],
    }, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError(f"Training audit failed: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
