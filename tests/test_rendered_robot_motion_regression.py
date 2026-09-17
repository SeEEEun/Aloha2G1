from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from rendered_robot_motion_guard import (  # noqa: E402
    StaticRobotRenderError,
    assert_moving_q_has_moving_robot_masks,
)


OUT = ROOT / (
    "outputs/scene_registered_retargeting/"
    "current_layout_ep49_execution_quality_v17_2_renderfix"
)


def test_guard_rejects_static_masks_for_moving_q() -> None:
    with pytest.raises(StaticRobotRenderError, match="STATIC_DESPITE_MOVING_Q"):
        assert_moving_q_has_moving_robot_masks(
            0.5,
            {"overview": {
                "robot_masks_identical_at_all_keyframes": True,
                "maximum_keyframe_mask_xor_pixels": 0,
                "robot_mask_nonempty_all_frames": True,
            }},
        )


@pytest.mark.parametrize(
    ("artifact", "motion_key"),
    [
        ("kinematic_execution_render_parity.json", "requested_q_motion_max_peak_to_peak_rad"),
        ("render_parity_full_task_diagnostic_paper_white.json", "target_motion_max_peak_to_peak_rad"),
        ("gui_execution_render_parity.json", "target_motion_max_peak_to_peak_rad"),
    ],
)
def test_v17_2_renderfix_masks_move_with_q(artifact: str, motion_key: str) -> None:
    payload = json.loads((OUT / artifact).read_text(encoding="utf-8"))
    assert_moving_q_has_moving_robot_masks(
        float(payload[motion_key]), payload["rendered_motion"]
    )
