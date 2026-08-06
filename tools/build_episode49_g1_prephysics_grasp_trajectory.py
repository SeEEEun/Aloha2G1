#!/usr/bin/env python3
"""Build episode-49 pre-physics trajectory only after static grasp validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/home/jbnu/aloha_g1_dataset")
STATIC = ROOT / "converted_runs/g1_dex3_static_phone_grasp/static_grasp_report.json"
STATIC_NPZ = STATIC.with_name("selected_static_grasp.npz")
SOURCE = ROOT / ("evaluation/smolvla_episode49_temporal_consensus/"
                 "episode_000049_temporal_consensus.npz")
OUT = ROOT / ("converted_runs/smolvla_20k_episode49_g1_prephysics_grasp/"
              "g1_episode49_prephysics_grasp_trajectory.npz")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    report = json.loads(STATIC.read_text())
    if (report.get("verdict") != "G1_STATIC_PHONE_GRASP_FOUND"
            or not report.get("safety_pass") or not STATIC_NPZ.exists()):
        print("G1_STATIC_PHONE_GRASP_NOT_FOUND")
        print("Temporal trajectory generation is fail-closed. No NPZ or videos were created.")
        print("G1_PREPHYSICS_GRASP_SAFETY_BLOCKED")
        return 2
    # This guard is intentionally explicit: the current search did not produce
    # a valid static endpoint, so reaching implementation would be unsafe.
    raise RuntimeError(
        "Validated static endpoint exists but temporal solver implementation "
        "has not been reviewed in this run; refusing to fabricate a trajectory.")


if __name__ == "__main__":
    raise SystemExit(main())
