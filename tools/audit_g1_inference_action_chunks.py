#!/usr/bin/env python3
"""Offline hard-limit, temporal and collision audit for logged raw 50x28 chunks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from doll_handoff_retargeting.models import G1Kinematics  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety-freeze", type=Path, default=ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json")
    args = parser.parse_args()
    with np.load(args.chunks, allow_pickle=False) as archive:
        chunks = np.asarray(archive["raw_action_chunks_50x28"], dtype=np.float64)
        names = archive["joint_names"].astype(str).tolist()
    if chunks.ndim != 3 or chunks.shape[1:] != (50, 28) or not np.isfinite(chunks).all():
        raise RuntimeError(f"raw action chunks must be finite [N,50,28], got {chunks.shape}")
    safety = json.loads(args.safety_freeze.read_text(encoding="utf-8"))
    specs = safety["joint_specs"]
    if [row["joint_name"] for row in specs] != names:
        raise RuntimeError("logged names differ from frozen common hard-limit order")
    lower = np.asarray([row["minimum"] for row in specs])
    upper = np.asarray([row["maximum"] for row in specs])
    below = lower - chunks
    above = chunks - upper
    excess = np.maximum(below, above)
    common = json.loads((ROOT / "configs/doll_handoff_retargeting/common_config.template.json").read_text())
    scene = json.loads((ROOT / "isaaclab_doll_handoff_scene/scene_layout.json").read_text())
    g1 = G1Kinematics(common, scene)
    collision_rows = []
    maximum_step = 0.0
    for index, chunk in enumerate(chunks):
        geometry = g1.trajectory_geometry(
            chunk[:, :14], chunk[:, 14:21], chunk[:, 21:28],
            float(common["validation"]["collision_penetration_tolerance_m"]),
        )
        counts = {key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()}
        invalid = sum(counts.get(key, 0) for key in ("ARM_TORSO", "CROSS_ARM", "WRIST_OR_PALM_TORSO", "OTHER"))
        collision_rows.append({"chunk_index": index, "category_frame_counts": counts, "invalid_frame_incidence": invalid})
        maximum_step = max(maximum_step, float(np.linalg.norm(np.diff(chunk, axis=0), axis=1).max(initial=0.0)))
    report = {
        "schema_version": "g1_inference_raw_action_preflight_v1",
        "status": "PASS" if int(np.count_nonzero(excess > 0)) == 0 and sum(row["invalid_frame_incidence"] for row in collision_rows) == 0 else "WARNING_RAW_POLICY_OUTPUT_REQUIRES_REVIEW",
        "raw_actions_modified": False,
        "chunks": len(chunks),
        "hard_limit_violation_scalar_count": int(np.count_nonzero(excess > 0)),
        "maximum_hard_limit_excess_rad": float(np.max(excess, initial=0.0)),
        "maximum_within_chunk_step_norm_rad": maximum_step,
        "collision": collision_rows,
        "collision_invalid_frame_incidence": sum(row["invalid_frame_incidence"] for row in collision_rows),
        "command_authorization": "NONE_INFERENCE_ONLY",
    }
    atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "chunks", "hard_limit_violation_scalar_count", "collision_invalid_frame_incidence")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
