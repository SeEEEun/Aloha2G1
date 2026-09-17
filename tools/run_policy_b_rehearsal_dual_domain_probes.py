#!/usr/bin/env python3
"""Run the fixed 9-phase probe on both visual domains at every rehearsal checkpoint."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
PROBE = ROOT / "tools/probe_policy_b_g1visual_phases.py"
DATASETS = {
    "ALOHA": ROOT / "datasets/doll_handoff_proposed_b_50",
    "G1VISUAL": ROOT / "datasets/doll_handoff_proposed_b_g1visual_50",
}
RUNS = {
    "R1": ROOT / "outputs/policy_b_g1visual/rehearsal/training/R1_paired_expert_001500",
    "R2": ROOT / "outputs/policy_b_g1visual/rehearsal/training/R2_paired_visual_connector_001500",
}
OUTPUT = ROOT / "outputs/policy_b_g1visual/rehearsal/probes"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_probe(
    experiment: str, step: int, checkpoint: Path, model_hash: str, domain: str, dataset: Path
) -> dict[str, Any]:
    output = OUTPUT / experiment / f"{step:06d}" / domain
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(PROBE),
        "--dataset",
        str(dataset),
        "--checkpoint",
        str(checkpoint),
        "--expected-model-sha256",
        model_hash,
        "--output",
        str(output),
    ]
    log_path = output / "console.log"
    print(f"START {experiment} step={step} domain={domain}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if "G1-visual probe" in line or '"status"' in line:
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    if return_code not in (0, 2):
        raise RuntimeError(f"phase probe crashed ({return_code}); see {log_path}")
    decision = json.loads((output / "phase_learning_decision.json").read_text(encoding="utf-8"))
    predictions = np.load(output / "probe_predictions.npz", allow_pickle=False)
    error = predictions["policy_prediction"].astype(np.float64) - predictions[
        "authoritative_target"
    ].astype(np.float64)
    phase_score = sum(
        bool(row["policy_behavior_present"]) for row in decision["phase_results"].values()
    )
    result = {
        "experiment": experiment,
        "step": step,
        "domain": domain,
        "checkpoint": str(checkpoint),
        "model_sha256": model_hash,
        "phase_score": phase_score,
        "phase_total": 9,
        "arm_full_chunk_rmse_rad": float(np.sqrt(np.mean(np.square(error[:, :, :14])))),
        "dex3_full_chunk_rmse_rad": float(np.sqrt(np.mean(np.square(error[:, :, 14:])))),
        "phase_pass": {
            phase: bool(row["policy_behavior_present"])
            for phase, row in decision["phase_results"].items()
        },
        "relevant_evidence": {
            phase: row["relevant_group_evidence"]
            for phase, row in decision["phase_results"].items()
        },
        "probe_output": str(output),
        "return_code": return_code,
    }
    print(
        f"DONE {experiment} step={step} domain={domain} score={phase_score}/9 "
        f"arm={result['arm_full_chunk_rmse_rad']:.6f} dex3={result['dex3_full_chunk_rmse_rad']:.6f}",
        flush=True,
    )
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for experiment, run in RUNS.items():
        for step in (500, 1000, 1500):
            checkpoint = run / "checkpoints" / f"{step:06d}" / "pretrained_model"
            model_file = checkpoint / "model.safetensors"
            if not model_file.is_file():
                raise FileNotFoundError(model_file)
            model_hash = sha256_file(model_file)
            for domain, dataset in DATASETS.items():
                results.append(
                    run_probe(experiment, step, checkpoint, model_hash, domain, dataset)
                )
    fieldnames = [
        "experiment",
        "step",
        "domain",
        "phase_score",
        "phase_total",
        "arm_full_chunk_rmse_rad",
        "dex3_full_chunk_rmse_rad",
        "model_sha256",
        "checkpoint",
    ]
    with (OUTPUT / "dual_domain_checkpoint_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    atomic_json(
        OUTPUT / "dual_domain_checkpoint_results.json",
        {
            "status": "PASS",
            "metric_definition": {
                "arm_rmse": "full 50-step chunk over both 7D arms and all 54 probes",
                "dex3_rmse": "full 50-step chunk over both 7D hands and all 54 probes",
                "phase_gate": "unchanged established 6/6 per-episode direction/cosine rule",
            },
            "results": results,
        },
    )
    print(f"ALL_PROBES_COMPLETE {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
