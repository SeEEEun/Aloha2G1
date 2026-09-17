#!/usr/bin/env python3
"""Finalize ACT-B training, reload, and immutable-dataset health evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/policy_b_act"
TRAIN = OUTPUT / "train"
LOG = OUTPUT / "training_stdout.log"
LAUNCH = OUTPUT / "training_launch.json"
CONFIG = OUTPUT / "config/train_config.json"
IDENTITY_BEFORE = OUTPUT / "audit/dataset_identity_before_training.json"
DATASET = ROOT / "datasets/doll_handoff_proposed_b_50"
EXPECTED_STEPS = (20_000, 40_000, 60_000, 80_000, 100_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-step", type=int, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def dataset_tree() -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    rows = []
    for path in sorted(item for item in DATASET.rglob("*") if item.is_file()):
        relative = path.relative_to(DATASET).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        rows.append({"path": relative, "bytes": size, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), rows


def parse_training_metrics(text: str) -> list[dict[str, Any]]:
    # TQDM prefixes the INFO record on the same carriage-return line.
    pattern = re.compile(
        r"step:(?P<step>\d+)(?P<suffix>K?)\s+smpl:[^\s]+.*?"
        r"loss:(?P<loss>[-+0-9.eE]+)\s+grdn:(?P<grad>[-+0-9.eE]+)\s+"
        r"lr:(?P<lr>[-+0-9.eE]+).*?"
        r"l1_loss:(?P<l1>[-+0-9.eE]+)\s+kld_loss:(?P<kld>[-+0-9.eE]+)"
    )
    rows = []
    for match in pattern.finditer(text.replace("\r", "\n")):
        step_label = match.group("step") + match.group("suffix")
        step_approx = int(match.group("step")) * (1000 if match.group("suffix") else 1)
        rows.append(
            {
                "step_log_label": step_label,
                "step_approximate_from_rounded_log_label": step_approx,
                "loss": float(match.group("loss")),
                "gradient_norm": float(match.group("grad")),
                "learning_rate": float(match.group("lr")),
                "l1_loss": float(match.group("l1")),
                "kld_loss": float(match.group("kld")),
            }
        )
    if not rows:
        raise RuntimeError("no structured training metrics found")
    return rows


def checkpoint_record(step: int) -> dict[str, Any]:
    root = TRAIN / "checkpoints" / f"{step:06d}"
    pretrained = root / "pretrained_model"
    model = pretrained / "model.safetensors"
    required = (
        model,
        pretrained / "config.json",
        pretrained / "policy_preprocessor.json",
        pretrained / "policy_postprocessor.json",
        root / "training_state/optimizer_param_groups.json",
        root / "training_state/training_step.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    return {
        "step": step,
        "checkpoint_root": str(root),
        "pretrained_model": str(pretrained),
        "required_files_present": not missing,
        "missing_required_files": missing,
        "model_sha256": sha256_file(model) if model.is_file() else None,
    }


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    args = parse_args()
    launch = json.loads(LAUNCH.read_text(encoding="utf-8"))
    if launch.get("status") != "COMPLETE" or int(launch.get("return_code", -1)) != 0:
        raise RuntimeError(f"ACT-B training process did not complete cleanly: {launch}")
    if not args.selection_report.is_file():
        raise FileNotFoundError(args.selection_report)
    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    if int(selection["selected_checkpoint_step"]) != args.selected_step:
        raise RuntimeError("selected checkpoint argument differs from selection report")

    log_text = LOG.read_text(encoding="utf-8")
    rows = parse_training_metrics(log_text)
    numeric_values = [
        float(value)
        for row in rows
        for key, value in row.items()
        if key not in {"step_log_label", "step_approximate_from_rounded_log_label"}
    ]
    metrics_finite = all(math.isfinite(value) for value in numeric_values)
    checkpoint_rows = [checkpoint_record(step) for step in EXPECTED_STEPS]
    if not metrics_finite or not all(row["required_files_present"] for row in checkpoint_rows):
        raise RuntimeError("training finite/checkpoint-save gate failed")

    checkpoint = (
        TRAIN / "checkpoints" / f"{args.selected_step:06d}" / "pretrained_model"
    )
    policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    policy.eval()
    dataset = LeRobotDataset(
        "local/doll_handoff_proposed_b_50", root=DATASET, video_backend="torchcodec"
    )
    sample = dataset[16634]
    processed = preprocessor(
        {
            "observation.images.cam_high": sample["observation.images.cam_high"].clone(),
            "observation.state": sample["observation.state"].clone(),
        }
    )
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(processed)
        physical = postprocessor(normalized)
    output = physical.detach().float().cpu().numpy()
    reload_pass = output.shape == (1, 50, 28) and bool(np.isfinite(output).all())
    if not reload_pass:
        raise RuntimeError(f"selected checkpoint reload/inference gate failed: {output.shape}")

    before = json.loads(IDENTITY_BEFORE.read_text(encoding="utf-8"))
    after_hash, after_files = dataset_tree()
    files_exact = after_files == before["files"]
    tree_exact = after_hash == before["dataset_tree_sha256"]
    if not files_exact or not tree_exact:
        raise RuntimeError("Dataset B changed during ACT-B training/evaluation")
    after_identity = {
        "schema_version": "act_b_dataset_identity_after_training_v1",
        "status": "PASS",
        "dataset_root": str(DATASET),
        "dataset_tree_sha256": after_hash,
        "identical_to_before_training_manifest": files_exact and tree_exact,
        "before_training_manifest": str(IDENTITY_BEFORE),
        "before_training_manifest_sha256": sha256_file(IDENTITY_BEFORE),
        "ACTION_ARRAYS_UNCHANGED": "YES",
        "STATE_ARRAYS_UNCHANGED": "YES",
        "EPISODE_COUNT": before["episodes"],
        "FRAME_COUNT": before["frames"],
        "files": after_files,
    }
    atomic_json(OUTPUT / "audit/dataset_identity_after_training.json", after_identity)

    report = {
        "schema_version": "act_b_training_health_v1",
        "status": "PASS",
        "training_process_status": launch["status"],
        "return_code": launch["return_code"],
        "elapsed_seconds": launch["elapsed_seconds"],
        "configured_steps": 100_000,
        "structured_log_records": len(rows),
        "all_logged_losses_finite": all(
            math.isfinite(float(row["loss"])) and math.isfinite(float(row["l1_loss"])) and math.isfinite(float(row["kld_loss"]))
            for row in rows
        ),
        "all_logged_gradient_norms_finite": all(
            math.isfinite(float(row["gradient_norm"])) for row in rows
        ),
        "nan_detected_in_structured_metrics": False,
        "inf_detected_in_structured_metrics": False,
        "final_logged_metrics": rows[-1],
        "minimum_logged_loss": float(min(float(row["loss"]) for row in rows)),
        "maximum_logged_gradient_norm": float(max(float(row["gradient_norm"]) for row in rows)),
        "checkpoints": checkpoint_rows,
        "periodic_checkpoint_save": True,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_step": args.selected_step,
        "selected_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "strict_checkpoint_reload": True,
        "reload_eval_mode": not policy.training,
        "reload_output_shape": list(output.shape),
        "reload_output_finite": bool(np.isfinite(output).all()),
        "training_config": str(CONFIG),
        "training_config_sha256": sha256_file(CONFIG),
        "selection_report": str(args.selection_report.resolve()),
        "selection_report_sha256": sha256_file(args.selection_report),
        "dataset_identity_after_training": str(
            OUTPUT / "audit/dataset_identity_after_training.json"
        ),
    }
    atomic_json(OUTPUT / "training_health.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
