#!/usr/bin/env python3
"""Add renderer parity arrays to the generated v16 NPZ without changing q."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from aloha_g1_v15.kinematics import ActiveG1Dex3  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_contact_carrier_v16"
TRAJECTORY = OUT / "arm_dex3_coupled_trajectory.npz"
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
MAPPING = ROOT / "configs/dex3_abc_finger_mapping.sim.json"
PALM = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def main() -> int:
    with np.load(TRAJECTORY, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    arm = payload["g1_arm_q"]
    left = payload["left_dex3_q"]
    right = payload["right_dex3_q"]
    runtime = ActiveG1Dex3(MODEL, MAPPING, PALM, payload["g1_root"])
    left_palm = np.empty((len(arm), 3))
    right_palm = np.empty_like(left_palm)
    contacts = {name: np.empty((len(arm), 3)) for name in runtime.contacts}
    for index in range(len(arm)):
        runtime.assign(arm[index], left[index], right[index])
        left_palm[index] = runtime.palm_pose("left")[:3, 3]
        right_palm[index] = runtime.palm_pose("right")[:3, 3]
        for name in contacts:
            contacts[name][index] = runtime.contact_pose(name)[0]
    payload["achieved_left_position"] = left_palm
    payload["achieved_right_position"] = right_palm
    q_before = payload["controlled_q"].copy()
    atomic_npz(TRAJECTORY, payload)
    with np.load(TRAJECTORY, allow_pickle=False) as archive:
        if not np.array_equal(archive["controlled_q"], q_before):
            raise RuntimeError("render-input preparation changed controlled q")
    atomic_npz(OUT / "left_dex3_trajectory.npz", {
        "joint_names": np.asarray(runtime.hand_joint_names["left"]), "q": left,
        "contact_A": contacts["left_A"], "contact_B": contacts["left_B"], "contact_C": contacts["left_C"],
    })
    atomic_npz(OUT / "right_dex3_trajectory.npz", {
        "joint_names": np.asarray(runtime.hand_joint_names["right"]), "q": right,
        "contact_A": contacts["right_A"], "contact_B": contacts["right_B"], "contact_C": contacts["right_C"],
    })
    print("V16_RENDER_INPUTS_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
