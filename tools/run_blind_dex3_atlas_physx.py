#!/usr/bin/env python3
"""PhysX-label prepared blind Dex3 atlas samples, preserving every result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
ENGINE = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
DEFAULT_ATLAS = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator/atlas_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def longest(mask: np.ndarray, dt: float) -> float:
    mask = np.asarray(mask, dtype=bool)
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * dt)


def physical_label(run: Path, sample: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((run / "trial_result.json").read_text(encoding="utf-8"))
    with np.load(run / "event_log.npz", allow_pickle=False) as archive:
        stage = archive["stage"].astype(str)
        dt = float(protocol["environment"]["physics_dt_s"])
        threshold = float(protocol["physical_experiment"]["meaningful_digit_force_n"])
        three = (
            (archive["thumb_force_n"] >= threshold)
            & (archive["index_force_n"] >= threshold)
            & (archive["middle_force_n"] >= threshold)
        )
        support_scope = np.isin(stage, ["POWER_GRASP", "GRAVITY_RETENTION", "LIFT_5CM", "HOLD_ELEVATED"])
        simultaneous_s = longest(three & support_scope, dt)
        table_free = archive["table_contact_force_n"] <= float(
            protocol["physical_experiment"]["maximum_table_force_for_elevated_n"]
        )
        hand_support = (
            (archive["thumb_force_n"] >= threshold)
            | (archive["index_force_n"] >= threshold)
            | (archive["middle_force_n"] >= threshold)
        )
        elevated_table_free_s = longest(
            (stage == "HOLD_ELEVATED") & table_free & hand_support, dt
        )
        elevated_three_digit_s = longest(
            (stage == "HOLD_ELEVATED") & table_free & three, dt
        )
        peak_object_speed = float(
            np.max(np.linalg.norm(archive["object_linear_velocity_m_s"], axis=1), initial=0.0)
        )
    physical = protocol["physical_experiment"]
    checks = {
        "simultaneous_three_digit_support": simultaneous_s
        >= float(physical["minimum_simultaneous_three_digit_support_s"]),
        "table_free_retention": elevated_table_free_s
        >= float(physical["minimum_table_free_retention_s"]),
        "three_individually_meaningful": bool(result["three_meaningful_digit_contacts"]),
        "retention": result["retention"]["status"] == "PASS",
        "lift": result["lift"]["status"] == "PASS",
        "artifact_free": result["artifact_checks"]["status"] == "PASS",
    }
    label = all(checks.values())
    return {
        "schema_version": "blind_dex3_physical_label_v1",
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "stratum": sample["stratum"],
        "physically_graspable": label,
        "checks": checks,
        "simultaneous_three_digit_support_s": simultaneous_s,
        "elevated_table_free_retention_s": elevated_table_free_s,
        "elevated_table_free_three_digit_s": elevated_three_digit_s,
        "peak_object_speed_m_s": peak_object_speed,
        "digit_results": result["digits"],
        "retention": result["retention"],
        "lift": result["lift"],
        "artifact_checks": result["artifact_checks"],
        "engine_status": result["status"],
        "command": sample["command"],
        "command_sha256": sample["command_sha256"],
        "event_log": str(run / "event_log.npz"),
        "event_log_sha256": sha256(run / "event_log.npz"),
        "trial_result": str(run / "trial_result.json"),
        "trial_result_sha256": sha256(run / "trial_result.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    parser.add_argument("--sample-id", action="append")
    args = parser.parse_args()
    atlas = args.atlas_root.resolve()
    manifest_path = atlas / f"ROUND{args.round}_COMMAND_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = Path(manifest["blind_protocol"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    records = {row["sample_id"]: row for row in manifest["records"]}
    selected = list(manifest["selected_sample_ids"])
    if args.sample_id:
        requested = set(args.sample_id)
        selected = [sample_id for sample_id in selected if sample_id in requested]
        if len(selected) != len(requested):
            raise RuntimeError("requested sample is not in the predeclared selected set")
    run_root = atlas / f"round_{args.round}_physx"
    run_root.mkdir(parents=True, exist_ok=True)
    labels: list[dict[str, Any]] = []
    for ordinal, sample_id in enumerate(selected, start=1):
        sample = records[sample_id]
        run = run_root / sample_id
        label_path = run / "PHYSICAL_LABEL.json"
        if label_path.is_file():
            labels.append(json.loads(label_path.read_text(encoding="utf-8")))
            print(f"[{ordinal}/{len(selected)}] {sample_id}: cached")
            continue
        if run.exists() and (run / "event_log.npz").is_file() and (run / "trial_result.json").is_file():
            label = physical_label(run, sample, protocol)
            label["wall_seconds"] = 0.0
            label["recovered_after_labeler_only_infrastructure_error"] = True
            dump(label_path, label)
            labels.append(label)
            print(
                f"[{ordinal}/{len(selected)}] {sample_id}: recovered complete physics, "
                f"{'POSITIVE' if label['physically_graspable'] else 'NEGATIVE'}"
            )
            continue
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(f"incomplete nonempty run requires audit, refusing overwrite: {run}")
        run.mkdir(parents=True, exist_ok=True)
        command = [
            str(ISAAC),
            str(ENGINE),
            "--config",
            str(CONFIG),
            "--side",
            "left",
            "--geometry",
            "FROZEN_COMPRESSED_SHORT_55",
            "--profile",
            "P14",
            "--output-dir",
            str(run),
            "--scripted-command-path",
            sample["command"],
            "--object-spawn-side",
            "left",
            "--bin-height-m",
            "0.150",
            "--bin-rim-bevel-m",
            "0.003",
            "--headless",
        ]
        dump(
            run / "INVOCATION.json",
            {
                "sample_id": sample_id,
                "command": command,
                "engine_sha256": sha256(ENGINE),
                "physics_config_sha256": sha256(CONFIG),
                "blind_protocol": str(protocol_path),
                "blind_protocol_sha256": sha256(protocol_path),
                "ab_artifacts_accessed": False,
            },
        )
        started = time.monotonic()
        with (run / "engine.log").open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode not in (0, 2):
            raise RuntimeError(
                f"infrastructure failure for {sample_id}: exit {process.returncode}; "
                f"see {run / 'engine.log'}"
            )
        if not (run / "event_log.npz").is_file() or not (run / "trial_result.json").is_file():
            raise RuntimeError(f"missing complete physics artifacts for {sample_id}")
        label = physical_label(run, sample, protocol)
        label["wall_seconds"] = float(time.monotonic() - started)
        dump(label_path, label)
        labels.append(label)
        dump(
            atlas / f"ROUND{args.round}_PHYSICS_PROGRESS.json",
            {
                "schema_version": f"blind_dex3_round{args.round}_progress_v1",
                "completed": len(labels),
                "total": len(selected),
                "positive": sum(row["physically_graspable"] for row in labels),
                "negative": sum(not row["physically_graspable"] for row in labels),
                "last_completed": sample_id,
                "status": "COMPLETE" if len(labels) == len(selected) else "IN_PROGRESS",
            },
        )
        print(
            f"[{ordinal}/{len(selected)}] {sample_id}: "
            f"{'POSITIVE' if label['physically_graspable'] else 'NEGATIVE'} "
            f"three={label['simultaneous_three_digit_support_s']:.3f}s "
            f"table_free={label['elevated_table_free_retention_s']:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
