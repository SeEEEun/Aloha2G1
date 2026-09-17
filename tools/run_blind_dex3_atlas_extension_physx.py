#!/usr/bin/env python3
"""PhysX-label the preregistered representation-neutral atlas extension."""

from __future__ import annotations

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
OUT = ROOT / "outputs/final_representation_neutral_eval/00_frozen_evaluator_v2"
ATLAS = OUT / "atlas_extension"
PROTOCOL = OUT / "BLIND_ATLAS_EXTENSION_PROTOCOL.json"


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
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * dt)


def label_run(run: Path, sample: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((run / "trial_result.json").read_text(encoding="utf-8"))
    rule = protocol["physical_label"]
    dt = 1.0 / 240.0
    with np.load(run / "event_log.npz", allow_pickle=False) as archive:
        stage = archive["stage"].astype(str)
        force = float(rule["meaningful_digit_force_n"])
        three = (
            (archive["thumb_force_n"] >= force)
            & (archive["index_force_n"] >= force)
            & (archive["middle_force_n"] >= force)
        )
        support_scope = np.isin(
            stage,
            ["POWER_GRASP", "GRAVITY_RETENTION", "LIFT_5CM", "HOLD_ELEVATED"],
        )
        simultaneous = longest(three & support_scope, dt)
        table_free = archive["table_contact_force_n"] <= float(
            rule["maximum_table_force_for_elevated_n"]
        )
        any_support = (
            (archive["thumb_force_n"] >= force)
            | (archive["index_force_n"] >= force)
            | (archive["middle_force_n"] >= force)
        )
        table_free_retention = longest(
            (stage == "HOLD_ELEVATED") & table_free & any_support, dt
        )
        three_table_free = longest(
            (stage == "HOLD_ELEVATED") & table_free & three, dt
        )
        peak_speed = float(
            np.max(
                np.linalg.norm(archive["object_linear_velocity_m_s"], axis=1),
                initial=0.0,
            )
        )
    checks = {
        "simultaneous_three_digit_support": simultaneous
        >= float(rule["minimum_simultaneous_three_digit_support_s"]),
        "table_free_retention": table_free_retention
        >= float(rule["minimum_table_free_retention_s"]),
        "three_individually_meaningful": bool(result["three_meaningful_digit_contacts"]),
        "retention": result["retention"]["status"] == "PASS",
        "lift": result["lift"]["status"] == "PASS",
        "artifact_free": result["artifact_checks"]["status"] == "PASS",
    }
    return {
        "schema_version": "blind_dex3_physical_label_extension_v2",
        "sample_id": sample["sample_id"],
        "family": sample["family"],
        "positive_seed": sample["positive_seed"],
        "physically_graspable": all(checks.values()),
        "checks": checks,
        "simultaneous_three_digit_support_s": simultaneous,
        "elevated_table_free_retention_s": table_free_retention,
        "elevated_table_free_three_digit_s": three_table_free,
        "peak_object_speed_m_s": peak_speed,
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
        "ab_artifacts_accessed": False,
    }


def main() -> int:
    manifest_path = ATLAS / "EXTENSION_COMMAND_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(PROTOCOL):
        raise RuntimeError("extension protocol changed after command generation")
    records = {row["sample_id"]: row for row in manifest["records"]}
    selected = manifest["selected_sample_ids"]
    run_root = ATLAS / "physx"
    run_root.mkdir(parents=True, exist_ok=True)
    completed_labels: list[dict[str, Any]] = []
    for ordinal, sample_id in enumerate(selected, start=1):
        sample = records[sample_id]
        run = run_root / sample_id
        label_path = run / "PHYSICAL_LABEL.json"
        if label_path.is_file():
            completed_labels.append(json.loads(label_path.read_text(encoding="utf-8")))
            print(f"[{ordinal}/{len(selected)}] {sample_id}: cached", flush=True)
            continue
        if run.exists() and (run / "event_log.npz").is_file() and (run / "trial_result.json").is_file():
            label = label_run(run, sample, protocol)
            label["wall_seconds"] = 0.0
            label["recovered_complete_physics"] = True
            dump(label_path, label)
            completed_labels.append(label)
            continue
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(f"incomplete nonempty physics run: {run}")
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
                "protocol_sha256": sha256(PROTOCOL),
                "common_execution_layer_freeze_sha256": manifest[
                    "common_execution_layer_freeze_sha256"
                ],
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
                f"Isaac infrastructure failure for {sample_id}: {process.returncode}"
            )
        if not (run / "event_log.npz").is_file() or not (run / "trial_result.json").is_file():
            raise RuntimeError(f"missing physics trace for {sample_id}")
        label = label_run(run, sample, protocol)
        label["wall_seconds"] = float(time.monotonic() - started)
        dump(label_path, label)
        completed_labels.append(label)
        positive = sum(row["physically_graspable"] for row in completed_labels)
        dump(
            ATLAS / "PHYSICS_PROGRESS.json",
            {
                "schema_version": "blind_dex3_atlas_extension_progress_v2",
                "status": "COMPLETE"
                if len(completed_labels) == len(selected)
                else "IN_PROGRESS",
                "completed": len(completed_labels),
                "total": len(selected),
                "positive": positive,
                "negative": len(completed_labels) - positive,
                "last_completed": sample_id,
                "ab_artifacts_accessed": False,
            },
        )
        print(
            f"[{ordinal}/{len(selected)}] {sample_id}: "
            f"{'POSITIVE' if label['physically_graspable'] else 'NEGATIVE'} "
            f"three={label['simultaneous_three_digit_support_s']:.3f}s "
            f"table_free={label['elevated_table_free_retention_s']:.3f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
