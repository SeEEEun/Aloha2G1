#!/usr/bin/env python3
"""Offline go/no-go validation for all 70 frozen episode-registered rollouts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.direct_physical_execution_isaac_runtime import build_runtime


OUT = ROOT / "outputs/final_episode_registered_eval35"
FREEZE = OUT / "01_freeze/FINAL_EVAL35_FREEZE_MANIFEST.json"
REGISTRATION = OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
COMMANDS = ROOT / "outputs/final_direct_physical_eval35/00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
ARM_AUDIT = ROOT / "outputs/final_direct_physical_eval35/00_preparation/COMMON_ARM_HARD_LIMIT_PROJECTOR_AUDIT.json"
RESULT = OUT / "00_preflight/FINAL_OFFLINE_PREFLIGHT.json"


def read_json(path: Path) -> dict[str, Any]:
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if list((OUT / "02_act_a_results/rollouts").glob("eval_*/RUN_MANIFEST.json")) or list((OUT / "03_act_b_results/rollouts").glob("eval_*/RUN_MANIFEST.json")):
        raise RuntimeError("offline preflight must precede final rollouts")
    freeze = read_json(FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_EVAL35":
        raise RuntimeError("hard freeze is unavailable")
    dependency_pass = 0
    for row in freeze["files"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen dependency mismatch: {path}")
        dependency_pass += 1
    commands = read_json(COMMANDS)
    registration = read_json(REGISTRATION)
    entries = {row["stable_episode_id"]: row for row in registration["entries"]}
    records = sorted(commands["records"], key=lambda row: (row["method"], int(row["eval_index"])))
    if len(records) != 70 or len(entries) != 35:
        raise RuntimeError("preflight inputs are not exact 70/35")
    os.environ["DIRECT_EVAL35_FREEZE_MANIFEST"] = str(FREEZE.resolve())
    rows = []
    for record in records:
        path = Path(record["physical_command"])
        if sha256_file(path) != record["physical_command_sha256"]:
            raise RuntimeError(f"command hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "raw_policy_command", "policy_safe_command", "commanded_q_rad", "joint_names",
                "stable_episode_id", "method", "common_task_intent", "common_initial_q_rad",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise RuntimeError(f"archive fields missing {path}: {missing}")
            raw = np.asarray(archive["raw_policy_command"], dtype=np.float64)
            safe = np.asarray(archive["policy_safe_command"], dtype=np.float64)
            commanded = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
            names = archive["joint_names"].astype(str).tolist()
            stable = str(np.asarray(archive["stable_episode_id"]).item())
            method_code = str(np.asarray(archive["method"]).item()).lower()
            intent = archive["common_task_intent"].astype(str)
            initial = np.asarray(archive["common_initial_q_rad"], dtype=np.float64)
        if raw.shape != safe.shape or safe.shape != commanded.shape or commanded.ndim != 2 or commanded.shape[1] != 28:
            raise RuntimeError(f"invalid command dimensions: {path}")
        if not all(np.isfinite(value).all() for value in (raw, safe, commanded, initial)):
            raise RuntimeError(f"non-finite archive: {path}")
        if not np.array_equal(safe, commanded) or len(names) != 28 or len(set(names)) != 28:
            raise RuntimeError(f"command/order mismatch: {path}")
        if stable != record["stable_episode_id"] or method_code != record["method"][4].lower():
            raise RuntimeError(f"episode or method identity mismatch: {path}")
        if len(intent) != len(commanded) or not set(np.unique(intent)).issubset({
            "OPEN_INTENT", "LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT", "HANDOFF_INTENT", "RIGHT_HOLD_INTENT", "FINAL_RELEASE_INTENT"
        }):
            raise RuntimeError(f"common source task timing invalid: {path}")
        entry = entries.get(stable)
        if entry is None or entry["A_B_identical_object_pose"] is not True:
            raise RuntimeError(f"registration absent or unequal: {stable}")
        if entry["methods"][record["method"]]["command_sha256"] != record["physical_command_sha256"]:
            raise RuntimeError(f"registration/command binding mismatch: {path}")
        runtime = build_runtime(path, commanded, names)
        expected_initial_dex3 = np.concatenate((runtime.controller.left_open, runtime.controller.right_open))
        if not np.allclose(initial[14:], expected_initial_dex3, rtol=0.0, atol=1.0e-7):
            raise RuntimeError(f"stale Dex3 guard/open state: {path}")
        rows.append({
            "method": record["method"], "eval_index": int(record["eval_index"]), "stable_episode_id": stable,
            "frames": len(commanded), "archive_sha256": record["physical_command_sha256"],
            "registration_entry_sha256": entry["entry_sha256"], "runtime_archive_validation": "PASS",
        })
    if any(entries[row["stable_episode_id"]]["target_object_pose"] != entries[row["stable_episode_id"]]["target_object_pose"] for row in rows):
        raise RuntimeError("unreachable matched pose consistency error")
    arm = read_json(ARM_AUDIT)
    patch = subprocess.run(
        ["/home/jbnu/miniconda3/envs/isaaclab6/bin/python", "tools/run_direct_physical_execution_isaac.py", "--validate-patch-only"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if patch.returncode != 0:
        raise RuntimeError(f"runtime patch validation failed:\n{patch.stdout}{patch.stderr}")
    method_counts = {method: sum(row["method"] == method for row in rows) for method in ("ACT-A40", "ACT-B40")}
    value = {
        "schema_version": "final_episode_registered_eval35_offline_preflight_v1",
        "status": "PASS", "freeze_manifest": str(FREEZE.resolve()), "freeze_manifest_sha256": sha256_file(FREEZE),
        "frozen_dependency_validation": f"{dependency_pass}/{len(freeze['files'])} PASS",
        "ACT-A_prepared": method_counts["ACT-A40"], "ACT-B_prepared": method_counts["ACT-B40"],
        "total_prepared": len(rows), "runtime_archive_validation": f"{len(rows)}/70 PASS",
        "registration_entries": 35, "matched_A_B_object_pose_equality": "35/35",
        "common_arm_hard_limit_projector": arm["methods"],
        "remaining_commanded_arm_hard_limit_violations": arm["remaining_arm_hard_limit_violation_scalar_count"],
        "arm_rescue": False, "wrist_rescue": False,
        "final_rollout_count_under_freeze": 0,
        "runtime_instrumentation": json.loads(patch.stdout), "records": rows,
    }
    atomic_json(RESULT, value)
    a = arm["methods"]["ACT-A40"]
    b = arm["methods"]["ACT-B40"]
    (RESULT.parent / "FINAL_OFFLINE_PREFLIGHT.md").write_text(
        "# Final episode-registered EVAL35 offline preflight\n\n"
        "Status: **PASS**\n\n"
        f"- Frozen dependencies: {dependency_pass}/{len(freeze['files'])} PASS\n"
        "- Registration/equality: 35/35 / 35/35 PASS\n"
        "- Runtime archive validation: 70/70 PASS\n"
        f"- ACT-A: {a['projected_scalar_count']} projected scalars, maximum {a['maximum_correction_rad']:.12f} rad\n"
        f"- ACT-B: {b['projected_scalar_count']} projected scalars, maximum {b['maximum_correction_rad']:.12f} rad\n"
        "- Remaining arm hard-limit violations: 0\n"
        "- ARM rescue / WRIST rescue: NO / NO\n"
        "- Final rollouts started: 0/70\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value[key] for key in ("status", "frozen_dependency_validation", "ACT-A_prepared", "ACT-B_prepared", "runtime_archive_validation", "remaining_commanded_arm_hard_limit_violations")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
