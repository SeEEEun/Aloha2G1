#!/usr/bin/env python3
"""Record one ACT-B checkpoint selection from the bounded three-checkpoint audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
COMPARISON = ROOT / "outputs/policy_b_act/checkpoint_evaluation/checkpoint_comparison.json"
OUTPUT = ROOT / "outputs/policy_b_act/selected_checkpoint.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--rationale", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    audit = json.loads(COMPARISON.read_text(encoding="utf-8"))
    results = audit["results"]
    matches = [row for row in results if int(row["checkpoint_step"]) == args.step]
    if len(matches) != 1:
        raise RuntimeError(
            f"step {args.step} is not exactly one of bounded evaluated steps {audit['evaluated_steps']}"
        )
    selected = matches[0]
    checkpoint = Path(selected["checkpoint"])
    actual_hash = sha256_file(checkpoint / "model.safetensors")
    if actual_hash != selected["model_sha256"]:
        raise RuntimeError("evaluated checkpoint hash changed")
    payload = {
        "schema_version": "act_b_checkpoint_selection_v1",
        "status": "PASS",
        "selected_checkpoint_step": args.step,
        "selected_checkpoint": str(checkpoint),
        "selected_model_sha256": actual_hash,
        "selection_rationale": args.rationale,
        "training_loss_used_as_sole_criterion": False,
        "criteria": {
            "dataset_b_action_prediction_accuracy": selected["overall_accuracy"],
            "semantic_phase_behavior": selected["phase_score"],
            "raw_chunk_mechanical_smoothness": selected[
                "representative_raw_chunk_dynamics"
            ]["aggregates"]["act_raw"],
        },
        "bounded_checkpoint_count": audit["evaluated_checkpoint_count"],
        "bounded_evaluated_steps": audit["evaluated_steps"],
        "comparison_report": str(COMPARISON),
        "comparison_report_sha256": sha256_file(COMPARISON),
        "phase_prediction_artifact": selected["prediction_artifact"],
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
