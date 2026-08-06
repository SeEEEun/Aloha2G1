#!/usr/bin/env python3
"""Kinematic GUI viewer for a validated refined static Dex3 phone grasp."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import find_g1_dex3_static_phone_grasp as old  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grasp", type=Path, default=ROOT / (
        "converted_runs/g1_dex3_static_phone_grasp_refined/"
        "selected_static_grasp_refined.npz"))
    a = p.parse_args()
    with np.load(a.grasp, allow_pickle=False) as z:
        qpos = z["full_qpos"].astype(float)
        pose = z["phone_proxy_pose"].astype(float)
    rpy = Rotation.from_quat(pose[3:][[1, 2, 3, 0]]).as_euler("xyz")
    model, _ = old.expanded_phone_model(pose[:3], rpy)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    from mujoco import viewer
    with viewer.launch_passive(model, data) as window:
        window.cam.lookat[:] = [.22, 0, 1.0]
        window.cam.distance = 1.2
        window.cam.azimuth = 180
        window.cam.elevation = -8
        print("qpos + mj_forward static viewer; no actuator control or mj_step.")
        while window.is_running():
            mujoco.mj_forward(model, data)
            window.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
