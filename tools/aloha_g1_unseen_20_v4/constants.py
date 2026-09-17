"""Immutable paths and source identity contract for the unseen-20 audit."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "raw_recordings"
SOURCE_DATASET_ROOT = ROOT / "lerobot_magsafe_20_cam_high_v3_unseen_20260813"
ORIGINAL_DATASET_ROOT = ROOT / "lerobot_magsafe_50_cam_high_v3"
OUTPUT_ROOT = ROOT / "outputs/g1_unseen_20_v4"
CAMERA_KEY = "observation.images.cam_high"
TASK = (
    "Remove the MagSafe accessory from the phone and place the phone on the "
    "MagSafe charger."
)
REPO_ID = "local/magsafe_aloha_20_cam_high_v3_unseen_20260813"

# This tuple is the experiment's predeclared source allowlist. Discovery may
# verify it, but execution never expands it with a glob.
RAW_RECORDING_NAMES = (
    "GoPark_20260813_105846",
    "GoPark_20260813_110203",
    "GoPark_20260813_113606",
    "GoPark_20260813_114407",
    "GoPark_20260813_120500",
    "GoPark_20260813_120821",
    "GoPark_20260813_120959",
    "GoPark_20260813_121153",
    "GoPark_20260813_121413",
    "GoPark_20260813_122109",
    "GoPark_20260813_122319",
    "GoPark_20260813_122459",
    "GoPark_20260813_122658",
    "GoPark_20260813_122953",
    "GoPark_20260813_123248",
    "GoPark_20260813_123713",
    "GoPark_20260813_123900",
    "GoPark_20260813_124203",
    "GoPark_20260813_124631",
    "GoPark_20260813_124904",
)

EXPECTED_FROZEN_SHA256 = {
    "common_arm_v2": "48d9fe29503091fed1bdc4eeb359f349d4a188ad05c368d32c09cdecee60acab",
    "feasibility_v3": "167a2ade3ffe694fb958119d68d0ff83f3187f5983d5fe5221e284e8eaf09bd0",
    "proposed_hand_v2_1": "811eba1591131671d787bb714c86e75b643cd2a13cd87ecddbb486cdd912bbd3",
    "collision_v4": "b2ece14bfd673bac177ee833f94b0fb0805dd74c00e23c9da31a92c0cc046261",
}

FROZEN_PATHS = {
    "common_arm_v2": ROOT
    / "outputs/g1_dataset_retargeting_arm_v2/candidates/frozen_common_arm_v2_config.json",
    "feasibility_v3": ROOT
    / "outputs/g1_dataset_feasibility_v3/solver/frozen_feasibility_v3_config.json",
    "proposed_hand_v2_1": ROOT
    / "outputs/g1_dataset_retargeting_hand_v2_1/config/proposed_hand_v2_1_candidate.json",
    "collision_v4": ROOT
    / "outputs/g1_dataset_collision_v4/calibration/frozen_global_collision_v4_config.json",
}

ORIGINAL_V4_ROOT = ROOT / "outputs/g1_dataset_collision_v4"
ORIGINAL_ID_PREFIX = "old50"
UNSEEN_ID_PREFIX = "new20"


def stable_source_id(prefix: str, episode_id: int) -> str:
    return f"{prefix}:{int(episode_id):03d}"


if len(RAW_RECORDING_NAMES) != 20 or len(set(RAW_RECORDING_NAMES)) != 20:
    raise RuntimeError("unseen source allowlist must contain exactly 20 unique names")

