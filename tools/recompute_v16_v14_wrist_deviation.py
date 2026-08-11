#!/usr/bin/env python3
"""Compute the actual achieved-v16 versus v14 wrist-pose deviation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from aloha_g1_v15.kinematics import ActiveG1Dex3  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_contact_carrier_v16"
TRAJECTORY = OUT / "arm_dex3_coupled_trajectory.npz"
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
MAPPING = ROOT / "configs/dex3_abc_finger_mapping.sim.json"
PALM = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "percentile_95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }


def main() -> int:
    with np.load(TRAJECTORY, allow_pickle=False) as archive:
        v16_arm = archive["g1_arm_q"].copy()
        v14_arm = archive["v14_reference_arm_q"].copy()
        root = archive["g1_root"].copy()
    runtime = ActiveG1Dex3(MODEL, MAPPING, PALM, root)
    open_left = runtime.open_hand_q["left"]
    open_right = runtime.open_hand_q["right"]
    poses = {
        key: np.empty((len(v16_arm), 4, 4), dtype=np.float64)
        for key in ("v14_left", "v14_right", "v16_left", "v16_right")
    }
    for index in range(len(v16_arm)):
        runtime.assign(v14_arm[index], open_left, open_right)
        poses["v14_left"][index] = runtime.wrist_pose("left")
        poses["v14_right"][index] = runtime.wrist_pose("right")
        runtime.assign(v16_arm[index], open_left, open_right)
        poses["v16_left"][index] = runtime.wrist_pose("left")
        poses["v16_right"][index] = runtime.wrist_pose("right")
    result = json.loads((OUT / "v14_deviation_metrics.json").read_text(encoding="utf-8"))
    for side in ("left", "right"):
        reference = poses[f"v14_{side}"]
        achieved = poses[f"v16_{side}"]
        translation = np.linalg.norm(achieved[:, :3, 3] - reference[:, :3, 3], axis=1)
        rotation = Rotation.from_matrix(
            np.einsum("tij,tkj->tik", achieved[:, :3, :3], reference[:, :3, :3])
        ).magnitude()
        result[f"{side}_wrist_translation_deviation_from_v14_m"] = distribution(translation)
        result[f"{side}_wrist_orientation_deviation_from_v14_rad"] = distribution(rotation)
    result["legacy_carrier_origin_distance_fields_deprecated"] = [
        "left_wrist_translation_m", "right_wrist_translation_m"
    ]
    result["actual_v14_wrist_deviation_recomputed"] = True
    target = OUT / "v14_deviation_metrics.json"
    temporary = target.with_suffix(".json.incomplete")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print("V16_V14_WRIST_DEVIATION_RECOMPUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
