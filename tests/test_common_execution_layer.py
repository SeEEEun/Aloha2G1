from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.common_execution_layer import (
    ARM_INDICES,
    DEX3_INDICES,
    WRIST_INDICES,
    CommonDex3ExecutionLayer,
    Dex3Primitive,
    ExecutionSnapshot,
    FrozenUnionBallEnvelope,
    _EnvelopeCenter,
)
from tools import run_final_common_execution_eval35 as eval35_runner


SHA = "a" * 64


def primitive() -> Dex3Primitive:
    return Dex3Primitive(
        left_open=np.zeros(7),
        left_preshape=np.full(7, 0.25),
        left_full_close=np.full(7, 0.75),
        right_open=np.zeros(7),
        right_preshape=np.full(7, -0.25),
        right_full_close=np.full(7, -0.75),
        preshape_frames=2,
        close_frames=2,
        release_frames=2,
        force_threshold_n=0.015,
        left_retention_frames=2,
        right_verification_frames=2,
        right_retention_frames=2,
        maximum_table_force_n=0.02,
        bin_center_xy_m=np.asarray([0.7, 0.1]),
        bin_opening_xy_m=np.asarray([0.2, 0.2]),
        bin_bottom_z_m=0.79,
        bin_rim_z_m=0.95,
    )


def envelope() -> FrozenUnionBallEnvelope:
    return FrozenUnionBallEnvelope(
        centers=(
            _EnvelopeCenter(np.asarray([0.0, 0.0, 0.0]), np.eye(3), 0.25),
            _EnvelopeCenter(np.asarray([0.03, 0.0, 0.0]), np.eye(3), 0.25),
        ),
        translation_normalization_m=0.01,
        rotation_normalization_deg=10.0,
        evaluator_sha256=SHA,
    )


def snapshot(
    left_x: float,
    left_force: float = 0.0,
    right_force: float = 0.0,
    object_position: tuple[float, float, float] = (0.0, 0.0, 0.85),
) -> ExecutionSnapshot:
    object_world = np.eye(4)
    object_world[:3, 3] = object_position
    left = object_world.copy()
    left[0, 3] += left_x
    right = object_world.copy()
    right[1, 3] += 0.2
    return ExecutionSnapshot(
        measured_q_rad=np.zeros(28),
        object_world=object_world,
        whole_hand_world={"left": left, "right": right},
        digit_force_n={
            "left": {digit: left_force for digit in ("thumb", "index", "middle")},
            "right": {digit: right_force for digit in ("thumb", "index", "middle")},
        },
        table_force_n=0.0,
    )


def controller(frames: int = 20) -> CommonDex3ExecutionLayer:
    raw = np.linspace(0.0, 0.1, frames * 28).reshape(frames, 28)
    safe = raw.copy()
    # Deliberate common safety-only delta; it must be preserved, not treated as
    # an execution-layer arm override.
    safe[:, 0] += 1.0e-4
    intent = np.full(frames, "CLOSE_INTENT")
    return CommonDex3ExecutionLayer(
        envelope(), primitive(), intent, raw, safe, "ACT-A40"
    )


def test_full_union_region_accepts_more_than_one_center() -> None:
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.03
    between = np.eye(4)
    between[0, 3] = 0.015
    assert envelope().margin(first) >= 0.0
    assert envelope().margin(second) >= 0.0
    assert envelope().margin(between) < 0.0


def test_outside_envelope_never_rescues_arm_wrist_or_fingers() -> None:
    layer = controller(5)
    for frame in range(5):
        decision = layer.step(frame, snapshot(left_x=0.1))
        np.testing.assert_array_equal(decision.executed_command, layer.safe[frame])
        assert not decision.override_mask.any()
    assert layer.summary()["events"]["left_envelope_entry_frame"] is None


def test_grasp_entry_overrides_only_dex3_and_preserves_policy_arm_motion() -> None:
    layer = controller(8)
    for frame in range(8):
        decision = layer.step(frame, snapshot(left_x=0.03))
        np.testing.assert_array_equal(
            decision.executed_command[ARM_INDICES], layer.safe[frame, ARM_INDICES]
        )
        np.testing.assert_array_equal(
            decision.executed_command[WRIST_INDICES], layer.safe[frame, WRIST_INDICES]
        )
        assert not decision.arm_override_mask.any()
        assert not decision.wrist_override_mask.any()
        assert np.array_equal(
            decision.override_mask[DEX3_INDICES], decision.dex3_override_mask[DEX3_INDICES]
        )
    assert layer.left_trigger == 0
    np.testing.assert_allclose(layer.trace[3].executed_command[14:21], 0.75)


def test_handoff_requires_real_receiving_contact_and_support_interlock() -> None:
    layer = controller(16)
    # Trigger and establish stable LEFT ownership.
    layer.step(0, snapshot(left_x=0.0))
    layer.step(1, snapshot(left_x=0.0, left_force=0.02))
    layer.step(2, snapshot(left_x=0.0, left_force=0.02))
    assert layer.left_owned_frame == 2
    # No right contact: no receiving-hand realization.
    layer.step(3, snapshot(left_x=0.0, left_force=0.02))
    assert layer.right_trigger is None
    # Real receiving contact starts the same local primitive.
    layer.step(4, snapshot(left_x=0.0, left_force=0.02, right_force=0.02))
    assert layer.right_trigger == 4
    # Giving hand remains closed until the frozen two-frame support gate passes.
    layer.step(5, snapshot(left_x=0.0, left_force=0.02, right_force=0.02))
    assert layer.left_release_frame == 5
    assert "GIVING_HAND_RELEASE_START" in layer.trace[5].events


def test_release_waits_for_right_ownership_and_frozen_bin_region() -> None:
    layer = controller(20)
    rows = [
        snapshot(0.0),
        snapshot(0.0, 0.02),
        snapshot(0.0, 0.02),
        snapshot(0.0, 0.02, 0.02),
        snapshot(0.0, 0.02, 0.02),
        snapshot(0.0, 0.02, 0.02),
        snapshot(0.0, 0.02, 0.02),
        snapshot(0.0, 0.0, 0.02),
    ]
    for frame, row in enumerate(rows):
        layer.step(frame, row)
    assert layer.right_owned_frame is not None
    assert layer.right_release_frame is None
    outside = snapshot(0.0, 0.0, 0.02, (0.4, 0.1, 0.85))
    layer.step(8, outside)
    assert layer.right_release_frame is None
    inside = snapshot(0.0, 0.0, 0.02, (0.7, 0.1, 0.85))
    layer.step(9, inside)
    assert layer.right_release_frame == 9
    assert layer.trace[9].phase == "RELEASE"


def test_eval35_identity_is_eval10_plus_all_25_evaluation_only_sources() -> None:
    manifest = json.loads(eval35_runner.EVAL35.read_text(encoding="utf-8"))
    base = json.loads(eval35_runner.BASE_EVAL10.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS_IDENTITY_FROZEN"
    assert manifest["evaluation_set"] == "EVAL35"
    assert manifest["eval_entries"][:10] == base["eval_entries"]
    new_rows = manifest["new_20260902_evaluation_only"]
    expected = sorted(
        path.name
        for path in (eval35_runner.ROOT / "raw_recordings").glob("GoPark_20260902_*")
        if path.is_dir()
    )
    assert len(expected) == len(new_rows) == 25
    assert [row["source_name"] for row in new_rows] == expected
    assert [row["eval_index"] for row in new_rows] == list(range(10, 35))
    for row in new_rows:
        assert row["evaluation_only"] is True
        for key in (
            "evaluator_calibration_allowed",
            "evaluator_tuning_allowed",
            "training_allowed",
            "checkpoint_selection_allowed",
            "controller_tuning_allowed",
            "outcome_used_for_episode_selection",
        ):
            assert row[key] is False


def test_eval35_aggregation_requires_and_reports_35_per_method(
    tmp_path: Path, monkeypatch: object
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(eval35_runner, "OUT", tmp_path)
    monkeypatch.setattr(eval35_runner, "EXECUTION_FREEZE", freeze)
    results = []
    for method in ("ACT-A40", "ACT-B40"):
        for index in range(35):
            passed = index < (1 if method == "ACT-A40" else 2)
            results.append(
                {
                    "method": method,
                    "eval_index": index,
                    "stable_episode_id": f"episode_{index:02d}",
                    "provenance": "TEST",
                    "outcomes": {stage: passed for stage in eval35_runner.STAGES},
                    "first_failure_stage": None if passed else "PHYSICAL_READINESS",
                }
            )
    eval35_runner.aggregate(results, SimpleNamespace(evaluator_sha256=SHA))
    a = json.loads((tmp_path / "ACT_A_EVAL35_FINAL_PHYSICAL.json").read_text())
    b = json.loads((tmp_path / "ACT_B_EVAL35_FINAL_PHYSICAL.json").read_text())
    assert a["stage_success_count"]["FULL_TASK_SUCCESS"] == 1
    assert b["stage_success_count"]["FULL_TASK_SUCCESS"] == 2
    assert a["stage_success_percent"]["FULL_TASK_SUCCESS"] == 100.0 / 35.0
    assert b["stage_success_percent"]["FULL_TASK_SUCCESS"] == 200.0 / 35.0
    report = (tmp_path / "ACT_AB_FULL_TASK_SUCCESS.md").read_text()
    assert "1/35 = 2.9%" in report
    assert "2/35 = 5.7%" in report
