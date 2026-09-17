from __future__ import annotations

import numpy as np

from tools.common_execution_layer import (
    ARM_INDICES,
    DEX3_INDICES,
    LEFT_DEX3,
    RIGHT_DEX3,
    WRIST_INDICES,
    Dex3Primitive,
    ExecutionSnapshot,
)
from tools.direct_physical_execution_layer import (
    CommonArmHardLimitProjector,
    DirectPhysicalDex3ExecutionLayer,
)


DEX3_LOWER = np.full(14, -1.0)
DEX3_UPPER = np.full(14, 1.0)
ARM_LOWER = np.full(14, -1.0)
ARM_UPPER = np.full(14, 1.0)


def primitive() -> Dex3Primitive:
    return Dex3Primitive(
        left_open=np.zeros(7), left_preshape=np.full(7, 0.25),
        left_full_close=np.full(7, 0.75), right_open=np.zeros(7),
        right_preshape=np.full(7, -0.25), right_full_close=np.full(7, -0.75),
        preshape_frames=2, close_frames=2, release_frames=2,
        force_threshold_n=0.015, left_retention_frames=2,
        right_verification_frames=2, right_retention_frames=2,
        maximum_table_force_n=0.02, bin_center_xy_m=np.asarray([0.7, 0.1]),
        bin_opening_xy_m=np.asarray([0.2, 0.2]), bin_bottom_z_m=0.79,
        bin_rim_z_m=0.95,
    )


def snapshot(
    left: float = 0.0,
    right: float = 0.0,
    measured_q: np.ndarray | None = None,
    table_force_n: float = 0.0,
    object_z_m: float = 0.0,
) -> ExecutionSnapshot:
    object_world = np.eye(4)
    object_world[2, 3] = object_z_m
    return ExecutionSnapshot(
        measured_q_rad=(np.zeros(28) if measured_q is None else measured_q),
        object_world=object_world,
        whole_hand_world={"left": np.eye(4), "right": np.eye(4)},
        digit_force_n={
            "left": {digit: left for digit in ("thumb", "index", "middle")},
            "right": {digit: right for digit in ("thumb", "index", "middle")},
        }, table_force_n=table_force_n,
    )


def layer(intent: list[str]) -> DirectPhysicalDex3ExecutionLayer:
    raw = np.linspace(0.0, 0.2, len(intent) * 28).reshape(len(intent), 28)
    return DirectPhysicalDex3ExecutionLayer(
        primitive(), intent, raw, raw.copy(), "ACT-A40",
        ARM_LOWER, ARM_UPPER, DEX3_LOWER, DEX3_UPPER
    )


def test_left_closes_on_intent_without_any_geometric_gate() -> None:
    value = layer(["OPEN_INTENT", "LEFT_CLOSE_INTENT", "LEFT_CLOSE_INTENT"])
    value.step(0, snapshot())
    decision = value.step(1, snapshot())
    assert value.left_trigger == 1
    assert "LEFT_CLOSE_INTENT_START" in decision.events


def test_common_open_is_used_before_intent_for_both_hands() -> None:
    value = layer(["OPEN_INTENT"])
    decision = value.step(0, snapshot())
    np.testing.assert_array_equal(decision.executed_command[LEFT_DEX3], value.left_open)
    np.testing.assert_array_equal(decision.executed_command[RIGHT_DEX3], value.right_open)


def test_transition_starts_from_previous_executed_target_not_lagged_measured_q() -> None:
    value = layer(["OPEN_INTENT", "LEFT_CLOSE_INTENT"])
    previous = value.step(0, snapshot())
    lagged = np.zeros(28)
    lagged[LEFT_DEX3] = 0.95
    current = value.step(1, snapshot(measured_q=lagged))
    np.testing.assert_array_equal(
        current.executed_command[LEFT_DEX3], previous.executed_command[LEFT_DEX3]
    )


def test_arm_and_wrist_are_bitwise_preserved() -> None:
    value = layer(["LEFT_CLOSE_INTENT"] * 5)
    for frame in range(5):
        decision = value.step(frame, snapshot())
        np.testing.assert_array_equal(decision.executed_command[ARM_INDICES], value.safe[frame, ARM_INDICES])
        np.testing.assert_array_equal(decision.executed_command[WRIST_INDICES], value.safe[frame, WRIST_INDICES])
        assert not decision.arm_override_mask.any()
        assert not decision.wrist_override_mask.any()


def test_common_arm_hard_limit_projector_is_nearest_bound_and_not_rescue() -> None:
    projector = CommonArmHardLimitProjector(ARM_LOWER, ARM_UPPER)
    projected, changed = projector.project(
        np.asarray([1.2, -1.3, *([0.25] * 12)], dtype=np.float64)
    )
    np.testing.assert_array_equal(projected[:2], np.asarray([1.0, -1.0]))
    np.testing.assert_array_equal(changed[:2], np.asarray([True, True]))
    assert not changed[2:].any()

    safe = np.zeros((1, 28), dtype=np.float64)
    safe[0, 5] = 1.2
    value = DirectPhysicalDex3ExecutionLayer(
        primitive(), ["OPEN_INTENT"], safe.copy(), safe, "ACT-A40",
        ARM_LOWER, ARM_UPPER, DEX3_LOWER, DEX3_UPPER
    )
    decision = value.step(0, snapshot())
    assert decision.executed_command[5] == 1.0
    assert value.arm_hard_limit_projection_masks[0][5]
    assert not decision.arm_override_mask.any()
    assert not decision.wrist_override_mask.any()
    summary = value.summary()
    assert summary["common_arm_hard_limit_projection_scalar_count"] == 1
    assert summary["common_wrist_hard_limit_projection_scalar_count"] == 1
    assert np.isclose(summary["maximum_common_arm_hard_limit_correction_rad"], 0.2)
    assert summary["arm_rescue_used"] is False
    assert summary["wrist_rescue_used"] is False


def test_handoff_closure_gets_full_opportunity_but_release_is_interlocked() -> None:
    value = layer(
        ["LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT"]
        + ["HANDOFF_INTENT"] * 14
    )
    value.step(0, snapshot())
    value.step(1, snapshot(left=0.02))
    value.step(2, snapshot(left=0.02))
    assert value.right_trigger == 2
    assert value.left_release_frame is None
    # Incidental contact while moving to preshape is provisional only.
    value.step(3, snapshot(left=0.02, right=0.02))
    assert value.contact_latched_targets["right"] == {}
    measured = np.zeros(28)
    measured[RIGHT_DEX3] = -0.5
    for frame in range(4, 16):
        value.step(frame, snapshot(left=0.02, right=0.02, measured_q=measured))
    assert value.grasp_confirmed_frame["right"] == 11
    assert value.table_support_loss_frame["right"] == 12
    assert value.grasp_confirmed_frame["right"] < value.table_support_loss_frame["right"]
    assert value.left_release_frame == 12


def test_transient_close_contact_does_not_permanently_latch() -> None:
    value = layer(["HANDOFF_INTENT"] * 8)
    measured = np.zeros(28)
    measured[RIGHT_DEX3] = -0.5
    value.step(0, snapshot(right=0.02, measured_q=measured))
    value.step(1, snapshot(right=0.02, measured_q=measured))
    assert value.contact_latched_targets["right"] == {}
    value.step(2, snapshot(right=0.02, measured_q=measured))
    value.step(3, snapshot(right=0.0, measured_q=measured))
    value.step(4, snapshot(right=0.02, measured_q=measured))
    assert value.grasp_confirmed_frame["right"] is None
    assert value.contact_latched_targets["right"] == {}


def test_lift_state_requires_confirm_then_table_support_loss() -> None:
    value = layer(["HANDOFF_INTENT"] * 18)
    measured = np.zeros(28)
    measured[RIGHT_DEX3] = -0.5
    for frame in range(13):
        value.step(
            frame,
            snapshot(
                right=0.02,
                measured_q=measured,
                table_force_n=0.2,
                object_z_m=0.0,
            ),
        )
    assert value.grasp_confirmed_frame["right"] == 9
    assert value.table_support_loss_frame["right"] is None
    assert value.lift_start_frame["right"] is None
    assert value.grasp_state["right"] == "HOLD"
    for frame in range(13, 16):
        value.step(
            frame,
            snapshot(
                right=0.02,
                measured_q=measured,
                table_force_n=0.0,
                object_z_m=0.006,
            ),
        )
    assert value.table_support_loss_frame["right"] == 13
    assert value.lift_start_frame["right"] == 15
    assert value.grasp_confirmed_frame["right"] < value.table_support_loss_frame["right"]
    assert value.table_support_loss_frame["right"] < value.lift_start_frame["right"]


def test_final_release_is_source_intent_driven_not_bin_position_gated() -> None:
    value = layer(["HANDOFF_INTENT", "FINAL_RELEASE_INTENT"])
    value.step(0, snapshot())
    decision = value.step(1, snapshot())
    assert value.right_release_frame == 1
    assert "FINAL_RELEASE_INTENT_START" in decision.events


def test_every_interpolated_dex3_command_is_bounded_and_arm_wrist_unchanged() -> None:
    intent = (
        ["OPEN_INTENT"] * 3
        + ["LEFT_CLOSE_INTENT"] * 8
        + ["LEFT_HOLD_INTENT"] * 5
        + ["HANDOFF_INTENT"] * 12
        + ["RIGHT_HOLD_INTENT"] * 5
        + ["FINAL_RELEASE_INTENT"] * 8
    )
    value = layer(intent)
    measured = np.linspace(-0.95, 0.95, 28)
    for frame in range(len(intent)):
        decision = value.step(
            frame, snapshot(left=0.02, right=0.02, measured_q=measured)
        )
        dex3 = decision.executed_command[DEX3_INDICES]
        assert np.all(dex3 >= DEX3_LOWER)
        assert np.all(dex3 <= DEX3_UPPER)
        np.testing.assert_array_equal(
            decision.executed_command[ARM_INDICES], value.safe[frame, ARM_INDICES]
        )
        np.testing.assert_array_equal(
            decision.executed_command[WRIST_INDICES], value.safe[frame, WRIST_INDICES]
        )
