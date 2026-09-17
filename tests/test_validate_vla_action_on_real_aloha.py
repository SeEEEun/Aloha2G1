from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import validate_vla_action_on_real_aloha as validator  # noqa: E402


def source() -> np.ndarray:
    return validator.load_source_action()[0]


def fake_arm(ip: str) -> SimpleNamespace:
    return SimpleNamespace(ip=ip, model="V0_FOLLOWER", min_time_to_move_multiplier=16)


def fake_config(order=("left", "right")) -> SimpleNamespace:
    arms = {side: fake_arm("192.168.1.5" if side == "left" else "192.168.1.4") for side in order}
    return SimpleNamespace(follower_arms=arms, leader_arms={"unused": object()}, cameras={"unused": object()})


def approved_authorization() -> validator.HardwareAuthorization:
    return validator.HardwareAuthorization(True, True, True, ())


def reviewed_safety() -> dict:
    return {
        "status": "REVIEWED",
        "hardware_replay_allowed": True,
        "controller_limits_verified": True,
        "gripper_limits_verified": True,
        "connect_motion_approved": True,
        "disconnect_motion_approved": True,
        "max_start_position_error": 0.1,
        "max_tracking_error": 0.1,
        "tracking_error_duration": 0.2,
        "max_joint_step": 1.0,
        "max_joint_velocity": 30.0,
        "max_joint_acceleration": 1000.0,
        "gripper_policy": "REJECT",
        "left_gripper_min": 0.0,
        "left_gripper_max": 0.044,
        "right_gripper_min": 0.0,
        "right_gripper_max": 0.044,
        "max_loop_overrun": 0.01,
        "operator_estop_confirmed": True,
        "workspace_clear_confirmed": True,
    }


def verified_stop_record() -> dict:
    return {
        "verified": True,
        "mechanism": "mock power cut",
        "location": "outside mock workspace",
        "cuts_left_follower": True,
        "cuts_right_follower": True,
        "single_operator_action": True,
        "reachable_outside_workspace": True,
        "power_loss_arm_behavior": "REMAINS_SUPPORTED",
        "area_below_both_arms_clear": True,
        "power_loss_behavior_accepted": True,
        "verified_by": "mock operator",
        "verification_date": "2026-08-10",
        "notes": "offline unit-test fixture; not a real hardware assertion",
    }


def test_01_source_action_shape_is_exact() -> None:
    assert source().shape == (990, 14)


def test_02_source_action_is_finite() -> None:
    assert np.isfinite(source()).all()


def test_03_exact_14d_channel_order() -> None:
    assert validator.JOINTS == [
        *(f"left_joint_{i}" for i in range(7)),
        *(f"right_joint_{i}" for i in range(7)),
    ]
    assert validator.UNITS == ["rad"] * 6 + ["m"] + ["rad"] * 6 + ["m"]


def test_04_cpu_torch_float32_conversion() -> None:
    tensor = validator.to_cpu_float32_tensor(source()[0])
    assert tensor.shape == (14,)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"


def test_05_follower_dictionary_order_is_asserted() -> None:
    backend = validator.ALOHAHardwareBackend(lambda _: fake_config())
    summary = backend.build_config()
    assert summary["follower_order"] == ["left", "right"]
    assert backend.config.leader_arms == {} and backend.config.cameras == {}
    wrong = validator.ALOHAHardwareBackend(lambda _: fake_config(("right", "left")))
    with pytest.raises(validator.Blocked, match="BLOCKED_FOLLOWER_ORDER"):
        wrong.build_config()


def test_06_invalid_shape_is_rejected() -> None:
    with pytest.raises(validator.Blocked, match="shape"):
        validator.validate_action14(np.zeros(13))


def test_07_nan_and_inf_are_rejected() -> None:
    for bad in (np.nan, np.inf, -np.inf):
        action = np.zeros(14)
        action[4] = bad
        with pytest.raises(validator.Blocked, match="NaN/Inf"):
            validator.validate_action14(action)


def test_08_negative_gripper_is_blocked_under_reject() -> None:
    action = np.zeros(14)
    action[6] = -1e-4
    with pytest.raises(validator.Blocked, match="left gripper"):
        validator.prepare_action_for_limits(
            action,
            validator.mock_controller_limits(gripper_min=0.0),
            validator.GripperPolicy.REJECT,
        )


def test_09_mock_controller_limit_violation_is_rejected() -> None:
    action = np.zeros(14)
    action[2] = 11.0
    with pytest.raises(validator.Blocked, match="left arm"):
        validator.prepare_action_for_limits(action, validator.mock_controller_limits())


def test_10_source_npz_and_array_remain_unchanged() -> None:
    action, integrity = validator.load_source_action()
    before = validator.array_sha256(action)
    validator.gripper_policy_audit(action)
    validator.run_mock_offline(action, 0.1, 30.0)
    validator.assert_source_unchanged(validator.DEFAULT_INPUT, action, integrity)
    assert validator.array_sha256(action) == before
    assert integrity["source_action_unchanged"] is True


def test_11_minimum_jerk_transition_is_finite() -> None:
    start = np.zeros(14)
    target = np.ones(14)
    transition = validator.minimum_jerk_transition(start, target, 1.0, 30.0, True)
    assert np.isfinite(transition).all()


def test_12_transition_starts_at_mock_actual_state() -> None:
    action = source()
    actual = action[0].astype(float).copy()
    actual[:6] += 0.02
    transition = validator.minimum_jerk_transition(actual, action[0], 1.0, 30.0, True)
    assert np.array_equal(transition[0], actual)


def test_13_transition_ends_at_source_arm_start() -> None:
    action = source()
    actual = action[0].astype(float).copy()
    actual[:6] += 0.02
    actual[7:13] -= 0.02
    transition = validator.minimum_jerk_transition(actual, action[0], 1.0, 30.0, True)
    arm_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    assert np.allclose(transition[-1, arm_indices], action[0, arm_indices])


def test_14_transition_holds_grippers() -> None:
    start = np.zeros(14)
    start[[6, 13]] = [0.02, 0.03]
    target = np.ones(14)
    transition = validator.minimum_jerk_transition(start, target, 1.0, 30.0, True)
    assert np.all(transition[:, 6] == 0.02)
    assert np.all(transition[:, 13] == 0.03)


def test_15_hardware_backend_connect_rejects_unreviewed_authorization(monkeypatch) -> None:
    backend = validator.ALOHAHardwareBackend(lambda _: fake_config())
    backend.build_config()
    imported = []
    monkeypatch.setattr(__import__("builtins"), "__import__", lambda *args, **kwargs: imported.append(args[0]))
    with pytest.raises(validator.Blocked, match="HARDWARE EXECUTION REFUSED"):
        backend.connect(validator.HardwareAuthorization(False, False, False, ("not reviewed",)))
    assert imported == []


def test_16_execute_hardware_flag_alone_is_insufficient() -> None:
    preflight, authorization = validator.evaluate_preflight(
        source_valid=True,
        config_valid=True,
        follower_order_valid=True,
        safety={},
        safety_reasons=["reviewed safety config missing"],
        stop_verification={},
        stop_verification_reasons=["BLOCKED_NO_VERIFIED_OPERATOR_STOP"],
        workspace_ack=False,
        estop_ack=False,
        connect_ack=False,
        disconnect_ack=False,
        execute_hardware=True,
        dry_run=False,
        hardware_confirmation=None,
        shutdown_confirmation=None,
    )
    assert preflight["status"] == "BLOCKED"
    assert authorization.connection_allowed is False


def test_17_stop_command_loop_does_not_shutdown() -> None:
    backend = validator.MockALOHAHardwareBackend(np.zeros(14))
    backend.connect()
    backend.stop_command_loop()
    assert backend.command_loop_stopped is True
    assert backend.shutdown_called is False
    assert backend.connected is True


def test_18_normal_shutdown_is_separate() -> None:
    backend = validator.MockALOHAHardwareBackend(np.zeros(14))
    backend.connect()
    backend.stop_command_loop()
    backend.normal_shutdown(approved_authorization())
    assert backend.shutdown_called is True
    assert backend.connected is False


def test_19_offline_cli_never_calls_real_build_or_connect(tmp_path, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(validator.ALOHAHardwareBackend, "build_config", lambda *_: called.append("build"))
    monkeypatch.setattr(validator.ALOHAHardwareBackend, "connect", lambda *_: called.append("connect"))
    assert validator.main(["--dry-run", "--output-dir", str(tmp_path)]) == 0
    assert called == []
    assert (tmp_path / "hardware_backend_audit.json").exists()


def test_20_full_990_frame_dry_run_completes() -> None:
    action = source()
    result, timing = validator.run_mock_offline(action, 0.1, 30.0)
    assert result["status"] == "PASS"
    assert result["source_frames_processed"] == 990
    assert result["commands_recorded"] == 994
    assert len(timing) == result["commands_recorded"]


def test_21_tracking_error_gate_is_persistent() -> None:
    monitor = validator.TrackingErrorMonitor(0.1, 0.2)
    monitor.check(np.zeros(14), np.ones(14), 0.1)
    with pytest.raises(validator.Blocked, match="PERSISTENT_TRACKING_ERROR"):
        monitor.check(np.zeros(14), np.ones(14), 0.1)


def test_22_timing_instrumentation_has_required_fields() -> None:
    result, timing = validator.run_mock_offline(source(), 0.1, 30.0)
    assert result["timing_rows"] == len(timing)
    assert {
        "scheduled_timestamp_ns",
        "actual_host_send_timestamp_ns",
        "state_read_timestamp_ns",
        "loop_period_ns",
        "loop_overrun_ns",
        "command_age_ns",
    } <= timing[0].keys()


def test_23_reviewed_safety_can_authorize_connection_but_not_before_start_state() -> None:
    safety = reviewed_safety()
    preflight, authorization = validator.evaluate_preflight(
        source_valid=True,
        config_valid=True,
        follower_order_valid=True,
        safety=safety,
        safety_reasons=[],
        stop_verification=verified_stop_record(),
        stop_verification_reasons=[],
        workspace_ack=True,
        estop_ack=True,
        connect_ack=True,
        disconnect_ack=True,
        execute_hardware=True,
        dry_run=False,
        hardware_confirmation=validator.HARDWARE_CONFIRMATION,
        shutdown_confirmation=validator.SHUTDOWN_CONFIRMATION,
        start_state_valid=False,
    )
    assert authorization.connection_allowed is True
    assert authorization.replay_allowed is False
    assert preflight["status"] == "BLOCKED"


def test_24_controller_limit_clamp_only_changes_transient_copy() -> None:
    original = np.zeros(14)
    original[[6, 13]] = [-1e-4, 0.05]
    prepared = validator.prepare_action_for_limits(
        original,
        validator.mock_controller_limits(gripper_min=0.0, gripper_max=0.044),
        validator.GripperPolicy.CLAMP_TO_CONTROLLER_LIMIT,
    )
    assert np.array_equal(original[[6, 13]], [-1e-4, 0.05])
    assert np.array_equal(prepared[[6, 13]], [0.0, 0.044])


def inspection_authorization() -> tuple[dict, validator.InspectionAuthorization]:
    return validator.evaluate_inspection_authorization(
        config_valid=True,
        follower_order_valid=True,
        workspace_clear_confirmed=True,
        operator_estop_confirmed=True,
        stop_verification=verified_stop_record(),
        stop_verification_reasons=[],
        connect_motion_approved=True,
        disconnect_motion_approved=True,
        physical_left_right_verified=True,
        inspection_confirmation=validator.INSPECTION_CONFIRMATION,
    )


class InspectionMock(validator.MockALOHAHardwareBackend):
    def __init__(self):
        super().__init__(np.arange(14, dtype=float) / 100, validator.mock_controller_limits())
        self.connect_calls = 0
        self.state_read_calls = 0
        self.limit_read_calls = 0

    def connect_for_inspection(self, authorization=None):
        self.connect_calls += 1
        super().connect_for_inspection(authorization)

    def read_state(self):
        self.state_read_calls += 1
        return super().read_state()

    def read_controller_limits(self):
        self.limit_read_calls += 1
        return super().read_controller_limits()

    def send_action(self, *args, **kwargs):
        raise AssertionError("inspection must never call send_action")


def run_mock_inspection(tmp_path, backend=None):
    record, authorization = inspection_authorization()
    backend = backend or InspectionMock()
    output, result = validator.run_hardware_inspection(
        backend=backend,
        authorization=authorization,
        authorization_record=record,
        config_summary={"follower_order": ["left", "right"]},
        output_root=tmp_path,
        inspection_name="inspection_mock_test",
    )
    return backend, output, result


def test_25_inspection_never_calls_send_or_replay_and_queries_read_apis(tmp_path) -> None:
    backend, output, result = run_mock_inspection(tmp_path)
    assert backend.connect_calls == 1
    assert backend.state_read_calls == 2
    assert backend.limit_read_calls == 1
    assert backend.commands == []
    assert result["send_action_called"] is False
    assert result["goal_position_command_called"] is False
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["forbidden_operations"]["trajectory_loop_entered"] is False
    assert manifest["forbidden_operations"]["optimized_action_loaded"] is False


def test_26_inspection_writes_required_success_outputs(tmp_path) -> None:
    _, output, _ = run_mock_inspection(tmp_path)
    required = {
        "inspection_report.md",
        "inspection_result.json",
        "persistent_config_snapshot.json",
        "current_state_before_or_after_connect.json",
        "controller_limits_left.json",
        "controller_limits_right.json",
        "gripper_limit_comparison.json",
        "authorization_record.json",
        "run_manifest.json",
    }
    assert required <= {path.name for path in output.iterdir()}


def test_27_inspection_authorization_is_separate_and_required() -> None:
    record, authorization = validator.evaluate_inspection_authorization(
        config_valid=True,
        follower_order_valid=True,
        workspace_clear_confirmed=True,
        operator_estop_confirmed=True,
        stop_verification=verified_stop_record(),
        stop_verification_reasons=[],
        connect_motion_approved=True,
        disconnect_motion_approved=True,
        physical_left_right_verified=False,
        inspection_confirmation=validator.INSPECTION_CONFIRMATION,
    )
    assert record["requires_replay_safety_config"] is False
    assert authorization.inspection_allowed is False
    assert authorization.normal_shutdown_allowed is False


def test_28_execute_and_hardware_inspect_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        validator.parser().parse_args(["--execute-hardware", "--hardware-inspect"])


def test_29_failed_inspection_connect_is_not_retried(tmp_path) -> None:
    class FailedConnect(InspectionMock):
        def connect_for_inspection(self, authorization=None):
            del authorization
            self.connect_calls += 1
            raise RuntimeError("mock connect failure")

    record, authorization = inspection_authorization()
    backend = FailedConnect()
    with pytest.raises(validator.InspectionFailure, match="INSPECTION_CONNECTION_FAILED"):
        validator.run_hardware_inspection(
            backend=backend,
            authorization=authorization,
            authorization_record=record,
            config_summary={"follower_order": ["left", "right"]},
            output_root=tmp_path,
            inspection_name="inspection_connect_failure",
        )
    assert backend.connect_calls == 1


def test_30_failed_limit_read_aborts_and_never_sends(tmp_path) -> None:
    class FailedLimits(InspectionMock):
        def read_controller_limits(self):
            self.limit_read_calls += 1
            raise RuntimeError("mock limit failure")

    record, authorization = inspection_authorization()
    backend = FailedLimits()
    with pytest.raises(validator.InspectionFailure, match="INSPECTION_LIMIT_QUERY_FAILED"):
        validator.run_hardware_inspection(
            backend=backend,
            authorization=authorization,
            authorization_record=record,
            config_summary={"follower_order": ["left", "right"]},
            output_root=tmp_path,
            inspection_name="inspection_limit_failure",
        )
    assert backend.limit_read_calls == 1
    assert backend.commands == []
    assert backend.shutdown_called is True


def test_31_nonfinite_inspection_state_is_rejected(tmp_path) -> None:
    class NonfiniteState(InspectionMock):
        def read_state(self):
            self.state_read_calls += 1
            state = self.state.copy()
            state[3] = np.nan
            return validator.StateSample(state, 123)

    record, authorization = inspection_authorization()
    backend = NonfiniteState()
    with pytest.raises(validator.InspectionFailure, match="BLOCKED_INVALID_HARDWARE_DATA"):
        validator.run_hardware_inspection(
            backend=backend,
            authorization=authorization,
            authorization_record=record,
            config_summary={"follower_order": ["left", "right"]},
            output_root=tmp_path,
            inspection_name="inspection_nonfinite_state",
        )
    assert backend.commands == []


def test_32_successful_inspection_keeps_replay_blocked(tmp_path) -> None:
    _, output, result = run_mock_inspection(tmp_path)
    assert result["inspection_status"] == "REAL_HARDWARE_INSPECTION_COMPLETE"
    assert result["replay_status"] == "REPLAY_STILL_BLOCKED"
    saved = json.loads((output / "inspection_result.json").read_text())
    assert saved["replay_status"] == "REPLAY_STILL_BLOCKED"


def test_33_gripper_limits_are_extracted_without_nominal_zero_assumption(tmp_path) -> None:
    _, output, _ = run_mock_inspection(tmp_path)
    comparison = json.loads((output / "gripper_limit_comparison.json").read_text())
    assert comparison["nominal_zero_assumed"] is False
    assert comparison["sides"]["left"]["left_gripper_controller_min"] == -0.001
    assert comparison["sides"]["right"]["right_gripper_controller_min"] == -0.001
    assert comparison["sides"]["left"]["comparison"] == "SOURCE_GRIPPER_WITHIN_CONTROLLER_LIMIT"


def test_34_inspection_order_failure_blocks_authorization() -> None:
    _, authorization = validator.evaluate_inspection_authorization(
        config_valid=True,
        follower_order_valid=False,
        workspace_clear_confirmed=True,
        operator_estop_confirmed=True,
        stop_verification=verified_stop_record(),
        stop_verification_reasons=[],
        connect_motion_approved=True,
        disconnect_motion_approved=True,
        physical_left_right_verified=True,
        inspection_confirmation=validator.INSPECTION_CONFIRMATION,
    )
    assert authorization.inspection_allowed is False
    assert any("FOLLOWER_ORDER_VALID" in reason for reason in authorization.reasons)


def test_35_denied_inspection_cli_never_loads_source_or_connects(monkeypatch, tmp_path) -> None:
    called = []
    monkeypatch.setattr(
        validator,
        "_static_config",
        lambda backend: ({"follower_order": ["left", "right"]}, True, True, []),
    )
    monkeypatch.setattr(
        validator,
        "load_source_action",
        lambda *_: (_ for _ in ()).throw(AssertionError("inspection loaded source action")),
    )
    monkeypatch.setattr(
        validator.ALOHAHardwareBackend,
        "connect_for_inspection",
        lambda *_: called.append("connect"),
    )
    with pytest.raises(validator.Blocked, match="BLOCKED_NO_VERIFIED_OPERATOR_STOP"):
        validator.main(
            ["--hardware-inspect", "--inspection-output-dir", str(tmp_path)]
        )
    assert called == []


def test_36_nonfinite_controller_limit_is_rejected(tmp_path) -> None:
    class NonfiniteLimits(InspectionMock):
        def read_controller_limits(self):
            limits = super().read_controller_limits()
            limits["right"][2]["velocity_max"] = np.inf
            return limits

    record, authorization = inspection_authorization()
    with pytest.raises(validator.InspectionFailure, match="BLOCKED_INVALID_HARDWARE_DATA"):
        validator.run_hardware_inspection(
            backend=NonfiniteLimits(),
            authorization=authorization,
            authorization_record=record,
            config_summary={"follower_order": ["left", "right"]},
            output_root=tmp_path,
            inspection_name="inspection_nonfinite_limit",
        )


def test_37_default_physical_stop_record_is_fail_closed() -> None:
    record, reasons = validator.load_stop_verification(
        validator.DEFAULT_STOP_VERIFICATION
    )
    assert record["verified"] is False
    assert record["cuts_left_follower"] is True
    assert record["cuts_right_follower"] is True
    assert record["single_operator_action"] is False
    assert record["simultaneous_both_followers_cut"] is False
    assert record["power_loss_arm_behavior"] is None
    assert record["power_loss_arm_behavior_report"] == "UNKNOWN_NOT_TESTED"
    assert validator.verified_operator_stop(record, reasons) is False
    assert reasons[0] == "BLOCKED_NO_VERIFIED_OPERATOR_STOP"


def test_38_legacy_estop_boolean_alone_never_authorizes_inspection() -> None:
    record, reasons = validator.load_stop_verification(
        validator.DEFAULT_STOP_VERIFICATION
    )
    result, authorization = validator.evaluate_inspection_authorization(
        config_valid=True,
        follower_order_valid=True,
        workspace_clear_confirmed=True,
        operator_estop_confirmed=True,
        stop_verification=record,
        stop_verification_reasons=reasons,
        connect_motion_approved=True,
        disconnect_motion_approved=True,
        physical_left_right_verified=True,
        inspection_confirmation=validator.INSPECTION_CONFIRMATION,
    )
    assert authorization.inspection_allowed is False
    assert result["states"]["VERIFIED_OPERATOR_STOP"] is False
    assert result["legacy_operator_estop_flag"] == "IGNORED_NOT_AUTHORIZATION"


def test_39_complete_mock_stop_record_validates(tmp_path) -> None:
    path = tmp_path / "stop.json"
    path.write_text(json.dumps(verified_stop_record()))
    record, reasons = validator.load_stop_verification(path)
    assert reasons == []
    assert validator.verified_operator_stop(record, reasons) is True


def test_40_significant_power_loss_drop_is_never_authorized(tmp_path) -> None:
    unsafe = verified_stop_record()
    unsafe["power_loss_arm_behavior"] = "DROPS_SIGNIFICANTLY"
    path = tmp_path / "unsafe_stop.json"
    path.write_text(json.dumps(unsafe))
    record, reasons = validator.load_stop_verification(path)
    assert validator.verified_operator_stop(record, reasons) is False
    assert any("drop significantly" in reason for reason in reasons)


def test_41_legacy_estop_boolean_alone_never_authorizes_replay() -> None:
    stop_record, stop_reasons = validator.load_stop_verification(
        validator.DEFAULT_STOP_VERIFICATION
    )
    preflight, authorization = validator.evaluate_preflight(
        source_valid=True,
        config_valid=True,
        follower_order_valid=True,
        safety=reviewed_safety(),
        safety_reasons=[],
        stop_verification=stop_record,
        stop_verification_reasons=stop_reasons,
        workspace_ack=True,
        estop_ack=True,
        connect_ack=True,
        disconnect_ack=True,
        execute_hardware=True,
        dry_run=False,
        hardware_confirmation=validator.HARDWARE_CONFIRMATION,
        shutdown_confirmation=validator.SHUTDOWN_CONFIRMATION,
        start_state_valid=True,
    )
    assert authorization.connection_allowed is False
    assert authorization.replay_allowed is False
    assert preflight["states"]["VERIFIED_OPERATOR_STOP"] is False


def low_level_authorization(
    stage: str = "B", *, motion: bool = False
) -> validator.CharacterizationAuthorization:
    return validator.CharacterizationAuthorization(
        connection_allowed=True,
        motion_allowed=motion,
        close_transport_allowed=True,
        stage=stage,
        reasons=(),
        states={},
    )


def stage_a_authorization(
    **overrides,
) -> tuple[dict, validator.StageAInspectionAuthorization]:
    values = {
        "config_valid": True,
        "follower_order_verified": True,
        "workspace_clear_confirmed": True,
        "physical_left_right_verified": True,
        "operator_present_confirmed": True,
        "left_power_switch_reachable": True,
        "right_power_switch_reachable": True,
        "acknowledge_idle_brake_connect_may_move": True,
        "acknowledge_idle_cleanup_command": True,
        "stage_a_confirmation": validator.STAGE_A_CONFIRMATION,
    }
    values.update(overrides)
    return validator.evaluate_stage_a_authorization(**values)


def resolved_watchdog() -> validator.WatchdogConfig:
    return validator.WatchdogConfig(
        max_tracking_error_arm_rad=0.1,
        max_tracking_error_gripper_m=0.01,
        tracking_error_duration_sec=0.1,
        max_command_step_arm_rad=0.1,
        max_command_step_gripper_m=0.01,
        max_command_velocity_arm_rad_s=10.0,
        max_command_velocity_gripper_m_s=0.5,
        max_state_age_sec=0.1,
        max_state_read_duration_sec=0.1,
        max_command_call_duration_sec=0.1,
        max_loop_overrun_sec=0.01,
        controller_position_tolerance=0.01,
    )


class MockClock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        self.now += 1_000
        return self.now

    def sleep(self, seconds):
        self.now += int(seconds * 1e9)


def test_42_low_level_connect_uses_configure_false_and_no_position_target() -> None:
    backend, drivers = validator.make_mock_low_level_backend()
    _, authorization = stage_a_authorization()
    result = backend.connect_stage_a_idle_without_home(authorization)
    assert result["home_command_sent"] is False
    assert result["goal_position_sent"] is False
    assert result["physical_motion_free_claimed"] is False
    for driver in drivers.values():
        assert driver.calls[0][-1] is False  # configure(clear_error=False)
        assert all(call[0] != "set_all_positions" for call in driver.calls)
        assert all(call[0] not in {"home", "sleep", "Goal_Position"} for call in driver.calls)


def test_43_low_level_connect_asserts_left_right_order_before_driver_construction() -> None:
    constructed = []
    wrong = fake_config(("right", "left"))
    backend = validator.StationaryAlohaLowLevelBackend(
        config_factory=lambda _: wrong,
        driver_factory=lambda side: constructed.append(side),
        idle_mode="idle",
        position_mode="position",
        model_mapping={"V0_FOLLOWER": ("model", "end")},
    )
    with pytest.raises(validator.Blocked, match="BLOCKED_FOLLOWER_ORDER"):
        backend.build_config()
    assert constructed == []


def test_44_brake_abort_confirms_both_seven_joint_idle_readbacks() -> None:
    backend, drivers = validator.make_mock_low_level_backend()
    backend.connect_stage_a_idle_without_home(stage_a_authorization()[1])
    result = backend.brake_abort("MOCK_USER_ABORT")
    assert result["status"] == "SOFTWARE_BRAKE_ABORT_CONFIRMED"
    assert all(result["sides"][side]["idle_confirmed"] for side in validator.FOLLOWER_ORDER)
    assert all(len(result["sides"][side]["api_readback"]) == 7 for side in validator.FOLLOWER_ORDER)
    assert all(any(call == ("set_all_modes", "idle") for call in driver.calls) for driver in drivers.values())


def test_45_brake_abort_attempts_right_if_left_fails_and_never_cleans_up() -> None:
    backend, drivers = validator.make_mock_low_level_backend(fail_idle_side="left")
    backend.connect_stage_a_idle_without_home(stage_a_authorization()[1])
    result = backend.brake_abort("MOCK_LEFT_FAILURE")
    assert result["status"] == "SOFTWARE_BRAKE_ABORT_PARTIAL"
    assert result["sides"]["left"]["idle_confirmed"] is False
    assert result["sides"]["right"]["idle_confirmed"] is True
    assert result["sides"]["right"]["idle_requested"] is True
    assert all(call[0] != "cleanup" for driver in drivers.values() for call in driver.calls)
    assert result["disconnect_called"] is False


def test_46_low_level_cleanup_is_separately_gated_and_has_no_position_call() -> None:
    backend, drivers = validator.make_mock_low_level_backend()
    authorization = stage_a_authorization()[1]
    backend.connect_stage_a_idle_without_home(authorization)
    backend.brake_abort("MOCK_COMPLETE")
    denied = validator.StageAInspectionAuthorization(True, False, (), {})
    with pytest.raises(validator.Blocked, match="TRANSPORT CLOSE REFUSED"):
        backend.close_transport_without_home(denied)
    result = backend.close_transport_without_home(authorization)
    assert result["status"] == "NO_HOME_TRANSPORT_CLOSED"
    assert result["high_level_disconnect_called"] is False
    assert all(call[0] != "set_all_positions" for driver in drivers.values() for call in driver.calls)


@pytest.mark.parametrize("key,source_name", [("q", "Q_KEY"), ("Q", "Q_KEY"), (" ", "SPACE_KEY")])
def test_47_q_and_space_set_abort_flag_only(key, source_name) -> None:
    abort = validator.AbortController(lambda: 123)
    assert abort.request_from_key(key) is True
    assert abort.requested is True
    assert abort.reason == "USER_ABORT"
    assert abort.source == source_name
    assert abort.requested_host_monotonic_ns == 123


@pytest.mark.parametrize("signum", [validator.signal.SIGINT, validator.signal.SIGTERM])
def test_48_sigint_and_sigterm_handler_only_set_flag(signum) -> None:
    abort = validator.AbortController(lambda: 456)
    abort.signal_handler(signum, None)
    assert abort.requested is True
    assert abort.source == validator.signal.Signals(signum).name


def watchdog_check(**overrides):
    values = {
        "command14": np.zeros(14),
        "actual14": np.zeros(14),
        "previous_command14": None,
        "dt_sec": 1 / 30,
        "state_age_sec": 0.0,
        "state_read_duration_sec": 0.0,
        "command_call_duration_sec": 0.0,
        "loop_overrun_sec": 0.0,
    }
    values.update(overrides)
    validator.RuntimeWatchdog(
        resolved_watchdog(), validator.mock_controller_limits()
    ).check(**values)


def test_49_watchdog_nonfinite_state_aborts() -> None:
    bad = np.zeros(14)
    bad[4] = np.nan
    with pytest.raises(validator.WatchdogAbort, match="NONFINITE_OR_INVALID"):
        watchdog_check(actual14=bad)


def test_50_watchdog_state_timeout_and_loop_overrun_abort() -> None:
    with pytest.raises(validator.WatchdogAbort, match="STATE_STALE"):
        watchdog_check(state_age_sec=0.11)
    with pytest.raises(validator.WatchdogAbort, match="LOOP_OVERRUN"):
        watchdog_check(loop_overrun_sec=0.011)


def test_51_watchdog_joint_limit_and_controller_error_abort() -> None:
    command = np.zeros(14)
    command[0] = 11.0
    with pytest.raises(validator.WatchdogAbort, match="COMMAND_OUTSIDE_CONTROLLER_LIMIT"):
        watchdog_check(command14=command)
    with pytest.raises(validator.WatchdogAbort, match="CONTROLLER_ERROR"):
        watchdog_check(controller_fault=True, controller_error_information={"left": "fault"})


def test_52_watchdog_tracking_error_requires_persistence() -> None:
    watchdog = validator.RuntimeWatchdog(
        resolved_watchdog(), validator.mock_controller_limits()
    )
    command = np.zeros(14)
    command[0] = 0.2
    base = dict(
        command14=command,
        actual14=np.zeros(14),
        previous_command14=None,
        dt_sec=0.05,
        state_age_sec=0.0,
        state_read_duration_sec=0.0,
        command_call_duration_sec=0.0,
        loop_overrun_sec=0.0,
    )
    watchdog.check(**base)
    with pytest.raises(validator.WatchdogAbort, match="PERSISTENT_ARM_TRACKING_ERROR"):
        watchdog.check(**base)


def test_53_manual_abort_watchdog_stops_before_command_path() -> None:
    abort = validator.AbortController()
    abort.request("USER_ABORT", "Q_KEY")
    with pytest.raises(validator.WatchdogAbort, match="MANUAL_ABORT"):
        watchdog_check(abort_controller=abort)


def test_54_generated_characterization_starts_ends_at_current_and_holds_grippers() -> None:
    initial = np.arange(14, dtype=float) / 100
    trajectory, sides = validator.generate_characterization_trajectory(
        initial,
        stage="B",
        active_side="left",
        joint_indices=(2,),
        amplitude_rad=0.001,
        duration_sec=0.2,
    )
    assert sides == ("left",)
    assert np.array_equal(trajectory[0], initial)
    assert np.array_equal(trajectory[-1], initial)
    assert np.array_equal(
        trajectory[:, [6, 13]], np.repeat(initial[[6, 13]][None], len(trajectory), axis=0)
    )


def test_55_stage_b_mock_does_not_load_source_and_commands_only_selected_follower(monkeypatch) -> None:
    monkeypatch.setattr(
        validator,
        "load_source_action",
        lambda *_: (_ for _ in ()).throw(AssertionError("characterization loaded source")),
    )
    backend, drivers = validator.make_mock_low_level_backend()
    clock = MockClock()
    result = validator.run_tracking_characterization_stage(
        backend=backend,
        authorization=low_level_authorization("B", motion=True),
        stage="B",
        active_side="left",
        joint_indices=(0,),
        amplitude_rad=0.001,
        duration_sec=0.1,
        watchdog_config=resolved_watchdog(),
        reviewed_controller_limits=validator.mock_controller_limits(),
        minimum_controller_limit_margin=0.0,
        abort_controller=validator.AbortController(clock),
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )
    assert result["failure"] is None
    assert result["optimized_action_loaded"] is False
    left_position_calls = [
        call for call in drivers["left"].calls if call[0] == "set_joint_position"
    ]
    right_position_calls = [
        call for call in drivers["right"].calls if call[0] == "set_joint_position"
    ]
    assert len(left_position_calls) > 0
    assert all(call[1] == 0 for call in left_position_calls)
    assert right_position_calls == []
    assert all(
        call[0]
        not in {"set_all_positions", "set_arm_positions", "set_gripper_position"}
        for driver in drivers.values()
        for call in driver.calls
    )
    assert result["gripper_position_target_sent"] is False
    assert result["brake_abort"]["status"] == "SOFTWARE_BRAKE_ABORT_CONFIRMED"


def test_56_stage_a_mock_reads_state_limits_but_never_position_commands() -> None:
    backend, drivers = validator.make_mock_low_level_backend()
    result = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert result["status"] == "STAGE_A_INSPECTION_COMPLETE"
    assert result["position_command_audit"]["invariant_confirmed"] is True
    assert result["state14"]["state14"] == pytest.approx(np.zeros(14))
    assert set(result["controller_limits"]) == {"left", "right"}
    assert all(call[0] != "set_all_positions" for driver in drivers.values() for call in driver.calls)


def test_57_characterization_hard_failure_requests_brake_and_stops_more_commands() -> None:
    backend, drivers = validator.make_mock_low_level_backend()
    drivers["left"].state[0] = np.nan
    result = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert result["status"] == "STAGE_A_STATE_READ_FAILED"
    # connect succeeded, invalid first state caused no position target, then both idle were attempted.
    assert result["brake_abort"]["status"] == "SOFTWARE_BRAKE_ABORT_CONFIRMED"
    assert all(call[0] != "set_all_positions" for driver in drivers.values() for call in driver.calls)
    assert all(call[0] != "cleanup" for driver in drivers.values() for call in driver.calls)
    assert result["transport_close"]["status"] == "NOT_CALLED_OPERATOR_INTERVENTION_REQUIRED"


def test_58_tracking_metrics_estimates_known_positive_lag() -> None:
    command = np.sin(np.linspace(0, 4 * np.pi, 120))
    actual = np.concatenate(([command[0]] * 3, command[:-3]))
    samples = []
    for index in range(len(command)):
        q = np.zeros(14)
        a = np.zeros(14)
        q[0], a[0] = command[index], actual[index]
        samples.append(
            {
                "command_q": q.tolist(),
                "actual_q": a.tolist(),
                "loop_period_sec": 1 / 30,
                "loop_overrun_sec": 0.0,
            }
        )
    metrics = validator.compute_tracking_metrics(samples)
    assert metrics["per_joint"][0]["estimated_lag_samples"] == 3
    assert metrics["per_joint"][0]["command_actual_correlation"] > 0.99


def test_59_full_source_controller_limit_audit_counts_gripper_violations() -> None:
    limits = validator.mock_controller_limits(gripper_min=0.0, gripper_max=0.05)
    audit = validator.compare_source_to_controller_limits(source(), limits)
    assert audit["arm_violation_count"] == 0
    assert audit["gripper_violation_count"] > 0
    assert audit["sides"]["left"][6]["classification"] == "SOURCE_GRIPPER_OUTSIDE_CONTROLLER_LIMITS"


def test_60_stage_a_cli_uses_separate_gate_without_source_or_high_level_connect(
    monkeypatch, tmp_path
) -> None:
    calls = []
    monkeypatch.setattr(
        validator,
        "_static_config",
        lambda backend: (
            {"follower_order": ["left", "right"], "followers": {}},
            True,
            True,
            [],
        ),
    )
    monkeypatch.setattr(
        validator,
        "run_stage_a_idle_inspection",
        lambda **kwargs: calls.append(kwargs["authorization"]) or {"failure": None},
    )
    monkeypatch.setattr(validator, "write_stage_a_inspection_run", lambda *a, **k: tmp_path)
    monkeypatch.setattr(validator.ForegroundAbortInput, "start", lambda self: None)
    monkeypatch.setattr(validator.ForegroundAbortInput, "close", lambda self: None)
    monkeypatch.setattr(validator.ForegroundAbortInput, "poll", lambda self: None)
    monkeypatch.setattr(
        validator,
        "load_source_action",
        lambda *_: (_ for _ in ()).throw(AssertionError("Stage A loaded source")),
    )
    monkeypatch.setattr(
        validator.ALOHAHardwareBackend,
        "connect",
        lambda *_: (_ for _ in ()).throw(AssertionError("high-level connect called")),
    )
    assert validator.main(
        [
            "--hardware-characterize-tracking",
            "--characterization-stage",
            "A",
            "--workspace-clear-confirmed",
            "--physical-left-right-verified",
            "--operator-present-confirmed",
            "--left-power-switch-reachable",
            "--right-power-switch-reachable",
            "--acknowledge-idle-brake-connect-may-move",
            "--acknowledge-idle-cleanup-command",
            "--stage-a-confirmation",
            validator.STAGE_A_CONFIRMATION,
        ]
    ) == 0
    assert len(calls) == 1
    assert calls[0].risk_class is validator.RISK_CLASS_A


def test_61_characterization_mode_is_mutually_exclusive_with_replay_and_inspection() -> None:
    with pytest.raises(SystemExit):
        validator.parser().parse_args(
            ["--hardware-characterize-tracking", "--execute-hardware"]
        )
    with pytest.raises(SystemExit):
        validator.parser().parse_args(
            ["--hardware-characterize-tracking", "--hardware-inspect"]
        )


def test_62_offline_software_audit_preserves_source_and_uses_no_vendor_factory(tmp_path, monkeypatch) -> None:
    before = validator.sha(validator.DEFAULT_INPUT)
    monkeypatch.setattr(
        validator.StationaryAlohaLowLevelBackend,
        "_resolve_vendor",
        lambda *_: (_ for _ in ()).throw(AssertionError("vendor factory resolved")),
    )
    result = validator.run_software_safety_offline_audit(
        validator.DEFAULT_INPUT, tmp_path
    )
    assert result["real_hardware_connection"] is False
    assert result["real_mode_change"] is False
    assert validator.sha(validator.DEFAULT_INPUT) == before
    assert (tmp_path / "connection_path_audit.json").exists()


def test_63_unresolved_watchdog_template_cannot_authorize_motion() -> None:
    result, authorization = validator.evaluate_characterization_authorization(
        stage="B",
        config_valid=True,
        follower_order_valid=True,
        workspace_clear_confirmed=True,
        physical_left_right_verified=True,
        stop_verification=verified_stop_record(),
        stop_verification_reasons=[],
        acknowledge_idle_brake_connect_may_move=True,
        acknowledge_idle_cleanup_command=True,
        characterization_confirmation=validator.CHARACTERIZATION_CONFIRMATION,
        characterization_config={
            "status": "REVIEWED",
            "hardware_replay_allowed": False,
            "controller_limits_verified": True,
            "watchdog": {
                name: None for name in validator.WatchdogConfig.__dataclass_fields__
            },
        },
        characterization_config_reasons=[],
        prior_stage_approved=True,
    )
    assert authorization.connection_allowed is False
    assert authorization.motion_allowed is False
    assert result["states"]["WATCHDOG_THRESHOLDS_REVIEWED"] is False


def test_64_abort_requested_during_connect_brakes_after_connect_without_state_or_motion() -> None:
    abort = validator.AbortController()
    backend, drivers = validator.make_mock_low_level_backend()
    original_connect = backend.connect_stage_a_idle_without_home

    def connect_then_abort(authorization):
        result = original_connect(authorization)
        abort.request("USER_ABORT", "SIGINT")
        return result

    backend.connect_stage_a_idle_without_home = connect_then_abort
    result = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
        abort_controller=abort,
    )
    assert result["status"] == "STAGE_A_USER_ABORT"
    assert result["failure"]["code"] == "STAGE_A_USER_ABORT"
    assert result["brake_abort"]["status"] == "SOFTWARE_BRAKE_ABORT_CONFIRMED"
    assert all(call[0] != "get_all_positions" for driver in drivers.values() for call in driver.calls)
    assert all(call[0] != "set_all_positions" for driver in drivers.values() for call in driver.calls)
    assert all(call[0] != "cleanup" for driver in drivers.values() for call in driver.calls)


def test_65_high_level_backend_brake_reaches_low_level_drivers_without_disconnect(monkeypatch) -> None:
    mode = object()
    monkeypatch.setitem(sys.modules, "trossen_arm", SimpleNamespace(Mode=SimpleNamespace(idle=mode)))

    class Driver:
        def __init__(self, fail=False):
            self.fail = fail
            self.calls = []
            self.modes = [mode] * 7

        def set_all_modes(self, requested):
            self.calls.append(("set_all_modes", requested))
            if self.fail:
                raise RuntimeError("idle failed")

        def get_modes(self):
            self.calls.append(("get_modes",))
            return self.modes

    left, right = Driver(fail=True), Driver()
    robot = SimpleNamespace(
        is_connected=True,
        follower_arms={
            "left": SimpleNamespace(driver=left),
            "right": SimpleNamespace(driver=right),
        },
    )
    backend = validator.ALOHAHardwareBackend()
    backend.robot = robot
    backend.command_loop_stopped = False
    result = backend.brake_abort("MOCK_RUNTIME_FAILURE")
    assert backend.command_loop_stopped is True
    assert result["status"] == "SOFTWARE_BRAKE_ABORT_PARTIAL"
    assert result["sides"]["right"]["idle_confirmed"] is True
    assert result["disconnect_called"] is False
    assert right.calls[0][0] == "set_all_modes"


def test_66_watchdog_read_command_timeouts_step_and_velocity_abort() -> None:
    with pytest.raises(validator.WatchdogAbort, match="STATE_READ_TIMEOUT"):
        watchdog_check(state_read_duration_sec=0.11)
    with pytest.raises(validator.WatchdogAbort, match="COMMAND_CALL_TIMEOUT"):
        watchdog_check(command_call_duration_sec=0.11)
    step_command = np.zeros(14)
    step_command[0] = 0.11
    with pytest.raises(validator.WatchdogAbort, match="COMMAND_STEP_ARM"):
        watchdog_check(
            command14=step_command,
            previous_command14=np.zeros(14),
        )
    velocity_command = np.zeros(14)
    velocity_command[0] = 0.05
    with pytest.raises(validator.WatchdogAbort, match="COMMAND_VELOCITY"):
        watchdog_check(
            command14=velocity_command,
            previous_command14=np.zeros(14),
            dt_sec=0.001,
        )


def test_67_stage_a_authorizes_without_verified_operator_stop() -> None:
    record, authorization = stage_a_authorization()
    assert authorization.connection_allowed is True
    assert authorization.motion_allowed is False
    assert authorization.risk_class is validator.RISK_CLASS_A
    assert record["states_not_required"]["VERIFIED_OPERATOR_STOP"] == "NOT_REQUIRED_FOR_STAGE_A"
    assert record["motion_allowed"] is False


@pytest.mark.parametrize(
    "field,state_name",
    [
        ("workspace_clear_confirmed", "WORKSPACE_CLEAR_CONFIRMED"),
        ("operator_present_confirmed", "OPERATOR_PRESENT_CONFIRMED"),
        ("left_power_switch_reachable", "LEFT_POWER_SWITCH_REACHABLE"),
        ("right_power_switch_reachable", "RIGHT_POWER_SWITCH_REACHABLE"),
        ("low_level_no_home_path_verified", "LOW_LEVEL_NO_HOME_PATH_VERIFIED"),
        ("mode_idle_semantics_verified", "MODE_IDLE_SEMANTICS_VERIFIED"),
    ],
)
def test_68_stage_a_required_condition_failures_are_closed(field, state_name) -> None:
    record, authorization = stage_a_authorization(**{field: False})
    assert authorization.connection_allowed is False
    assert record["states"][state_name] is False
    assert any(state_name in reason for reason in authorization.reasons)


def test_69_stage_a_confirmation_is_distinct_from_motion_confirmation() -> None:
    _, authorization = stage_a_authorization(
        stage_a_confirmation=validator.CHARACTERIZATION_CONFIRMATION
    )
    assert authorization.connection_allowed is False
    assert validator.RISK_CLASS_A.value == "READ_ONLY_IDLE_INSPECTION"
    assert validator.RISK_CLASS_MOTION.value == "MOTION"


def test_70_stage_a_driver_guard_blocks_position_before_raw_driver_call() -> None:
    backend, drivers = validator.make_mock_low_level_backend()
    backend.connect_stage_a_idle_without_home(stage_a_authorization()[1])
    with pytest.raises(
        validator.Blocked, match="BLOCKED_STAGE_A_POSITION_COMMAND_VIOLATION"
    ):
        backend.drivers["left"].set_all_positions([0.0] * 7, 1.0, False)
    assert all(call[0] != "set_all_positions" for call in drivers["left"].calls)
    audit = backend.stage_a_position_command_audit()
    assert audit["position_commands"] == 1
    assert audit["invariant_confirmed"] is False


def test_71_stage_a_runner_never_loads_source_sends_action_or_enters_loop(monkeypatch) -> None:
    monkeypatch.setattr(
        validator,
        "load_source_action",
        lambda *_: (_ for _ in ()).throw(AssertionError("Stage A loaded source")),
    )
    backend, drivers = validator.make_mock_low_level_backend()
    result = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    audit = result["position_command_audit"]
    assert audit["optimized_action_loaded"] is False
    assert audit["trajectory_loop_entered"] is False
    assert audit["send_action_calls"] == 0
    assert audit["goal_position_commands"] == 0
    assert all(
        call[0] not in {"set_all_positions", "Goal_Position", "send_action"}
        for driver in drivers.values()
        for call in driver.calls
    )


def test_72_stage_a_reads_modes_state_limits_and_extracts_grippers() -> None:
    initial = np.arange(14, dtype=float) / 1000
    backend, drivers = validator.make_mock_low_level_backend(initial)
    result = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert result["status"] == "STAGE_A_INSPECTION_COMPLETE"
    assert result["state14"]["state14"] == pytest.approx(initial)
    for side in validator.FOLLOWER_ORDER:
        calls = [call[0] for call in drivers[side].calls]
        assert calls.count("get_modes") >= 3  # configure check, post-read check, brake readback
        assert calls.count("get_all_positions") == 1
        assert calls.count("get_joint_limits") == 1
        assert result["modes"]["after_reads"][side] == ["idle"] * 7
        assert result["gripper_limits"]["sides"][side]["joint_index"] == 6


def test_73_stage_a_writes_dedicated_required_outputs(tmp_path) -> None:
    backend, _ = validator.make_mock_low_level_backend()
    auth_record, authorization = stage_a_authorization()
    result = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=authorization,
        operator_stop_verified=False,
    )
    output = validator.write_stage_a_inspection_run(
        tmp_path,
        result,
        auth_record,
        {"follower_order": ["left", "right"]},
        run_name="mock_stage_a",
    )
    required = {
        "stage_a_report.md",
        "stage_a_result.json",
        "authorization_record.json",
        "persistent_config_snapshot.json",
        "left_modes.json",
        "right_modes.json",
        "state14.json",
        "controller_limits_left.json",
        "controller_limits_right.json",
        "gripper_limits.json",
        "position_command_audit.json",
        "run_manifest.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    audit = json.loads((output / "position_command_audit.json").read_text())
    assert audit["position_commands"] == 0
    assert audit["goal_position_commands"] == 0
    assert audit["optimized_action_loaded"] is False
    assert audit["trajectory_loop_entered"] is False


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("B", "STAGE_B_BLOCKED_NO_VERIFIED_OPERATOR_STOP"),
        ("C", "STAGE_C_BLOCKED_NO_VERIFIED_OPERATOR_STOP"),
        ("D", "STAGE_D_BLOCKED_NO_VERIFIED_OPERATOR_STOP"),
    ],
)
def test_74_motion_stages_still_require_verified_operator_stop(stage, expected) -> None:
    stop_record, stop_reasons = validator.load_stop_verification(
        validator.DEFAULT_STOP_VERIFICATION
    )
    watchdog = {
        name: getattr(resolved_watchdog(), name)
        for name in validator.WatchdogConfig.__dataclass_fields__
    }
    record, authorization = validator.evaluate_characterization_authorization(
        stage=stage,
        config_valid=True,
        follower_order_valid=True,
        workspace_clear_confirmed=True,
        physical_left_right_verified=True,
        stop_verification=stop_record,
        stop_verification_reasons=stop_reasons,
        acknowledge_idle_brake_connect_may_move=True,
        acknowledge_idle_cleanup_command=True,
        characterization_confirmation=validator.CHARACTERIZATION_CONFIRMATION,
        characterization_config={
            "status": "REVIEWED",
            "hardware_replay_allowed": False,
            "controller_limits_verified": True,
            "watchdog": watchdog,
        },
        characterization_config_reasons=[],
        prior_stage_approved=True,
    )
    assert authorization.connection_allowed is False
    assert authorization.motion_allowed is False
    assert record["status"] == "BLOCKED_NO_VERIFIED_OPERATOR_STOP"
    assert validator.stage_a_motion_gate_statuses(False)[f"stage_{stage.lower()}"] == expected


def test_75_stage_a_success_keeps_every_motion_status_blocked() -> None:
    backend, _ = validator.make_mock_low_level_backend()
    result = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert result["successful_stage_a_authorizes_motion"] is False
    assert result["motion_gate_statuses"] == {
        "stage_b": "STAGE_B_BLOCKED_NO_VERIFIED_OPERATOR_STOP",
        "stage_c": "STAGE_C_BLOCKED_NO_VERIFIED_OPERATOR_STOP",
        "stage_d": "STAGE_D_BLOCKED_NO_VERIFIED_OPERATOR_STOP",
        "vla_replay": "VLA_REPLAY_BLOCKED_NO_VERIFIED_OPERATOR_STOP",
    }


def test_76_stage_a_configure_failure_codes_are_side_specific() -> None:
    left_backend, left_drivers = validator.make_mock_low_level_backend()
    left_drivers["left"].configure = lambda *args: (_ for _ in ()).throw(
        RuntimeError("left configure failed")
    )
    left = validator.run_stage_a_idle_inspection(
        backend=left_backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert left["status"] == "STAGE_A_LEFT_CONFIGURE_FAILED"
    assert all(call[0] != "configure" for call in left_drivers["right"].calls)

    right_backend, right_drivers = validator.make_mock_low_level_backend()
    right_drivers["right"].configure = lambda *args: (_ for _ in ()).throw(
        RuntimeError("right configure failed")
    )
    right = validator.run_stage_a_idle_inspection(
        backend=right_backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert right["status"] == "STAGE_A_RIGHT_CONFIGURE_FAILED"
    assert any(call[0] == "configure" for call in right_drivers["left"].calls)


def test_77_stage_a_mode_and_limit_failures_are_categorized() -> None:
    mode_backend, mode_drivers = validator.make_mock_low_level_backend()
    mode_drivers["left"].get_modes = lambda: (_ for _ in ()).throw(
        RuntimeError("mode read failed")
    )
    mode_result = validator.run_stage_a_idle_inspection(
        backend=mode_backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert mode_result["status"] == "STAGE_A_MODE_READ_FAILED"

    limit_backend, limit_drivers = validator.make_mock_low_level_backend()
    limit_drivers["right"].get_joint_limits = lambda: (_ for _ in ()).throw(
        RuntimeError("limit read failed")
    )
    limit_result = validator.run_stage_a_idle_inspection(
        backend=limit_backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert limit_result["status"] == "STAGE_A_LIMIT_READ_FAILED"


def test_78_stage_a_idle_mismatch_is_blocked_without_corrective_position() -> None:
    backend, drivers = validator.make_mock_low_level_backend()
    drivers["left"].get_modes = lambda: ["position"] * 7
    result = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert result["status"] == "BLOCKED_STAGE_A_IDLE_READBACK"
    assert all(call[0] != "set_all_positions" for driver in drivers.values() for call in driver.calls)


def test_79_stage_a_offline_source_audit_uses_saved_limits_and_preserves_npz(tmp_path) -> None:
    before = validator.sha(validator.DEFAULT_INPUT)
    backend, _ = validator.make_mock_low_level_backend()
    auth_record, authorization = stage_a_authorization()
    stage_a = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=authorization,
        operator_stop_verified=False,
    )
    output = validator.write_stage_a_inspection_run(
        tmp_path,
        stage_a,
        auth_record,
        {"follower_order": ["left", "right"]},
        run_name="source_audit",
    )
    audit = validator.run_stage_a_source_limit_audit(output)
    assert audit["source_action_unchanged"] is True
    assert validator.sha(validator.DEFAULT_INPUT) == before
    assert (output / "controller_limit_source_comparison.json").exists()
    assert (output / "gripper_source_comparison.json").exists()
    comparison = json.loads((output / "controller_limit_source_comparison.json").read_text())
    assert comparison["source_shape"] == [990, 14]
    assert len(comparison["sides"]["left"]) == 7
    assert len(comparison["sides"]["right"]) == 7


def test_80_vla_replay_remains_blocked_by_current_physical_stop() -> None:
    stop_record, stop_reasons = validator.load_stop_verification(
        validator.DEFAULT_STOP_VERIFICATION
    )
    preflight, authorization = validator.evaluate_preflight(
        source_valid=True,
        config_valid=True,
        follower_order_valid=True,
        safety=reviewed_safety(),
        safety_reasons=[],
        stop_verification=stop_record,
        stop_verification_reasons=stop_reasons,
        workspace_ack=True,
        estop_ack=True,
        connect_ack=True,
        disconnect_ack=True,
        execute_hardware=True,
        dry_run=False,
        hardware_confirmation=validator.HARDWARE_CONFIRMATION,
        shutdown_confirmation=validator.SHUTDOWN_CONFIRMATION,
        start_state_valid=True,
    )
    assert preflight["states"]["VERIFIED_OPERATOR_STOP"] is False
    assert authorization.connection_allowed is False
    assert authorization.replay_allowed is False


def test_81_stage_a_never_reaches_manipulator_robot_connect_path(monkeypatch) -> None:
    monkeypatch.setattr(
        validator.ALOHAHardwareBackend,
        "_construct_and_connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ManipulatorRobot.connect path reached")
        ),
    )
    backend, _ = validator.make_mock_low_level_backend()
    result = validator.run_stage_a_idle_inspection(
        backend=backend,
        authorization=stage_a_authorization()[1],
        operator_stop_verified=False,
    )
    assert result["status"] == "STAGE_A_INSPECTION_COMPLETE"
    assert result["position_command_audit"]["high_level_connect_called"] is False
    assert result["position_command_audit"]["manipulator_robot_constructed"] is False


def test_82_real_stage_a_limits_are_loaded_as_authoritative() -> None:
    limits, authority = validator.load_authoritative_stage_a_limits()
    assert authority["status"] == "AUTHORITATIVE_REAL_STAGE_A_CONTROLLER_LIMITS_LOADED"
    assert authority["follower_order"] == ["left", "right"]
    assert authority["arm_violation_count"] == 0
    assert limits["left"][6]["position_min"] == 0.0
    assert limits["right"][6]["position_min"] == 0.0
    assert limits["left"][6]["position_max"] == pytest.approx(0.03999999910593033)
    assert limits["right"][6]["position_max"] == pytest.approx(0.03999999910593033)


def test_83_hardware_adapter_saturates_only_grippers() -> None:
    adapter = validator.StageAHardwareCommandAdapter.from_stage_a_directory()
    original = source()[0].copy()
    original[6] = -0.0001
    original[13] = 0.01
    arms_before = original[validator.StageAHardwareCommandAdapter.ARM_INDICES].tobytes()
    command, record = adapter.adapt(original, source_frame=0)
    assert command[6] == 0.0
    assert command[13] == original[13]
    assert command[validator.StageAHardwareCommandAdapter.ARM_INDICES].tobytes() == arms_before
    assert record["channels"]["left"]["reason"] == "BELOW_CONTROLLER_MINIMUM"
    assert record["channels"]["right"]["reason"] == "IN_RANGE_UNCHANGED"
    assert record["arm_channels_modified"] == 0
    assert original[6] == pytest.approx(-0.0001)


def test_84_hardware_adapter_saturates_above_real_gripper_max() -> None:
    adapter = validator.StageAHardwareCommandAdapter.from_stage_a_directory()
    original = source()[0].copy()
    original[6] = 0.05
    original[13] = 0.06
    command, record = adapter.adapt(original)
    expected = 0.03999999910593033
    assert command[6] == pytest.approx(expected)
    assert command[13] == pytest.approx(expected)
    assert record["channels"]["left"]["reason"] == "ABOVE_CONTROLLER_MAXIMUM"
    assert record["channels"]["right"]["reason"] == "ABOVE_CONTROLLER_MAXIMUM"


def test_85_complete_saturation_audit_has_exact_counts_and_preserves_source() -> None:
    before = validator.sha(validator.DEFAULT_INPUT)
    source_action = source()
    array_before = validator.array_sha256(source_action)
    adapter = validator.StageAHardwareCommandAdapter.from_stage_a_directory()
    audit, command_copy = validator.audit_gripper_hardware_saturation(
        source_action, adapter
    )
    assert audit["sides"]["left"]["saturated_frame_count"] == 139
    assert audit["sides"]["right"]["saturated_frame_count"] == 697
    assert audit["sides"]["left"]["maximum_saturation_correction"] == pytest.approx(
        0.00017970101907849312
    )
    assert audit["sides"]["right"]["maximum_saturation_correction"] == pytest.approx(
        0.00014547863975167274
    )
    assert audit["arm_channels_byte_identical"] is True
    assert audit["arm_channel_modification_count"] == 0
    assert np.array_equal(
        source_action[:, validator.StageAHardwareCommandAdapter.ARM_INDICES],
        command_copy[:, validator.StageAHardwareCommandAdapter.ARM_INDICES],
    )
    assert validator.array_sha256(source_action) == array_before
    assert validator.sha(validator.DEFAULT_INPUT) == before


def test_86_stage_b_recommendation_is_tiny_and_unapproved() -> None:
    limits, authority = validator.load_authoritative_stage_a_limits()
    recommendation = validator.recommend_first_stage_b_config(
        authority["state14"], limits, stage_a_authority=authority, source=source()
    )
    stage = recommendation["stages"]["B"]
    assert recommendation["status"] == "UNAPPROVED_DEFAULT"
    assert recommendation["real_motion_authorized"] is False
    assert stage["active_side"] == "left"
    assert stage["joint_indices"] == [0]
    assert stage["amplitude_rad"] == 0.005
    assert stage["duration_sec"] == 2.0
    assert stage["rate_hz"] == 30.0
    assert recommendation["recommendation"][
        "expected_peak_command_velocity_rad_s"
    ] == pytest.approx(0.005 * np.pi / 2.0)


def test_87_stage_b_real_state_mock_commands_only_left_joint_0_and_no_gripper() -> None:
    limits, authority = validator.load_authoritative_stage_a_limits()
    initial = np.asarray(authority["state14"])
    backend, drivers = validator.make_mock_low_level_backend(
        initial, controller_limits=limits
    )
    clock = MockClock()
    result = validator.run_tracking_characterization_stage(
        backend=backend,
        authorization=low_level_authorization("B", motion=True),
        stage="B",
        active_side="left",
        joint_indices=(0,),
        amplitude_rad=0.005,
        duration_sec=2.0,
        rate_hz=30.0,
        watchdog_config=resolved_watchdog(),
        reviewed_controller_limits=limits,
        minimum_controller_limit_margin=0.1,
        abort_controller=validator.AbortController(clock),
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )
    assert result["failure"] is None
    assert result["samples_recorded"] == 61
    assert np.array_equal(result["samples"][0]["command_q"], initial)
    assert np.array_equal(result["samples"][-1]["command_q"], initial)
    assert result["gripper_position_target_sent"] is False
    assert all(
        np.array_equal(np.asarray(row["command_q"])[[6, 13]], initial[[6, 13]])
        for row in result["samples"]
    )
    left_joint = [call for call in drivers["left"].calls if call[0] == "set_joint_position"]
    assert left_joint and all(call[1] == 0 for call in left_joint)
    assert not any(call[0] == "set_joint_position" for call in drivers["right"].calls)
    assert not any(
        call[0] in {"set_all_positions", "set_arm_positions", "set_gripper_position"}
        for driver in drivers.values()
        for call in driver.calls
    )


def test_88_stage_b_manual_abort_requests_two_sided_brake_without_cleanup() -> None:
    limits, authority = validator.load_authoritative_stage_a_limits()
    backend, drivers = validator.make_mock_low_level_backend(
        authority["state14"], controller_limits=limits
    )
    abort = validator.AbortController()
    polls = {"count": 0}

    def poll():
        polls["count"] += 1
        if polls["count"] == 3:
            abort.request("USER_ABORT", "SPACE_KEY")

    result = validator.run_tracking_characterization_stage(
        backend=backend,
        authorization=low_level_authorization("B", motion=True),
        stage="B",
        active_side="left",
        joint_indices=(0,),
        amplitude_rad=0.005,
        duration_sec=2.0,
        watchdog_config=resolved_watchdog(),
        reviewed_controller_limits=limits,
        minimum_controller_limit_margin=0.1,
        abort_controller=abort,
        key_poller=poll,
    )
    assert result["failure"]["code"] == "MANUAL_ABORT_BEFORE_COMMAND"
    assert result["brake_abort"]["status"] == "SOFTWARE_BRAKE_ABORT_CONFIRMED"
    assert all(result["brake_abort"]["sides"][side]["idle_requested"] for side in validator.FOLLOWER_ORDER)
    assert result["transport_close"]["status"] == "NOT_CALLED_OPERATOR_INTERVENTION_REQUIRED"
    assert all(call[0] != "cleanup" for driver in drivers.values() for call in driver.calls)
    assert result["brake_abort"]["home_called"] is False
    assert result["brake_abort"]["sleep_called"] is False
    assert result["brake_abort"]["disconnect_called"] is False


def test_89_stage_b_presend_watchdog_abort_brakes_before_violating_target() -> None:
    limits, authority = validator.load_authoritative_stage_a_limits()
    backend, drivers = validator.make_mock_low_level_backend(
        authority["state14"], controller_limits=limits
    )
    strict = resolved_watchdog()
    strict = validator.WatchdogConfig(
        **{
            **strict.__dict__,
            "max_command_step_arm_rad": 1e-9,
        }
    )
    clock = MockClock()
    result = validator.run_tracking_characterization_stage(
        backend=backend,
        authorization=low_level_authorization("B", motion=True),
        stage="B",
        active_side="left",
        joint_indices=(0,),
        amplitude_rad=0.005,
        duration_sec=0.2,
        watchdog_config=strict,
        reviewed_controller_limits=limits,
        minimum_controller_limit_margin=0.1,
        abort_controller=validator.AbortController(clock),
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )
    assert result["failure"]["code"] == "COMMAND_STEP_ARM"
    assert result["brake_abort"]["status"] == "SOFTWARE_BRAKE_ABORT_CONFIRMED"
    calls = [call for call in drivers["left"].calls if call[0] == "set_joint_position"]
    # One current-position seed plus frame zero; the violating next target was rejected pre-send.
    assert len(calls) == 2


def test_90_stage_b_scheduler_and_metrics_include_required_timing() -> None:
    limits, authority = validator.load_authoritative_stage_a_limits()
    backend, _ = validator.make_mock_low_level_backend(
        authority["state14"], controller_limits=limits
    )
    clock = MockClock()
    result = validator.run_tracking_characterization_stage(
        backend=backend,
        authorization=low_level_authorization("B", motion=True),
        stage="B",
        active_side="left",
        joint_indices=(0,),
        amplitude_rad=0.005,
        duration_sec=0.2,
        rate_hz=30.0,
        watchdog_config=resolved_watchdog(),
        reviewed_controller_limits=limits,
        minimum_controller_limit_margin=0.1,
        abort_controller=validator.AbortController(clock),
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )
    scheduled = np.asarray([row["scheduled_timestamp_ns"] for row in result["samples"]])
    assert np.all(np.diff(scheduled) == int(round(1e9 / 30.0)))
    assert result["timing_model"]["controller_goal_time_sec"] == pytest.approx(16 / 30)
    selected = result["selected_joint_tracking_metrics"]
    assert "mae" in selected and "rmse" in selected
    assert "peak_absolute_command_velocity" in selected
    assert "peak_absolute_actual_velocity" in selected
    assert result["tracking_metrics"]["loop"]["mean_frequency_hz"] == pytest.approx(
        30.0, rel=1e-3
    )


def test_91_stage_b_offline_bundle_uses_no_vendor_or_network(tmp_path, monkeypatch) -> None:
    before = validator.sha(validator.DEFAULT_INPUT)
    monkeypatch.setattr(
        validator.StationaryAlohaLowLevelBackend,
        "_resolve_vendor",
        lambda *_: (_ for _ in ()).throw(AssertionError("vendor factory resolved")),
    )
    manifest = validator.run_stage_b_preparation_offline(
        output=tmp_path,
    )
    assert manifest["status"] == "STAGE_B_PREPARATION_COMPLETE_OFFLINE"
    assert manifest["real_hardware_connection"] is False
    assert manifest["real_motor_command"] is False
    assert validator.sha(validator.DEFAULT_INPUT) == before
    required = {
        "gripper_saturation_audit.json",
        "hardware_command_adapter.json",
        "recommended_stage_b_config.json",
        "mock_stage_b_tracking.json",
        "mock_brake_abort.json",
        "report.md",
        "commands.sh",
        "run_manifest.json",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}


def test_92_watchdog_checks_actual_controller_velocity() -> None:
    actual = np.zeros(14)
    actual[0] = 0.5
    with pytest.raises(
        validator.WatchdogAbort, match="ACTUAL_VELOCITY_CONTROLLER_LIMIT"
    ):
        watchdog_check(
            actual14=actual,
            previous_actual14=np.zeros(14),
            dt_sec=0.001,
        )


def test_93_stage_b_writer_creates_required_future_run_files(tmp_path) -> None:
    limits, authority = validator.load_authoritative_stage_a_limits()
    initial = np.asarray(authority["state14"])
    backend, _ = validator.make_mock_low_level_backend(
        initial, controller_limits=limits
    )
    clock = MockClock()
    result = validator.run_tracking_characterization_stage(
        backend=backend,
        authorization=low_level_authorization("B", motion=True),
        stage="B",
        active_side="left",
        joint_indices=(0,),
        amplitude_rad=0.001,
        duration_sec=0.1,
        watchdog_config=resolved_watchdog(),
        reviewed_controller_limits=limits,
        minimum_controller_limit_margin=0.1,
        abort_controller=validator.AbortController(clock),
        clock_ns=clock,
        sleep_fn=clock.sleep,
    )
    output = validator.write_stage_b_tracking_run(
        tmp_path,
        result,
        {"status": "MOCK_AUTHORIZATION"},
        {"status": "MOCK_CONFIG"},
        authority,
        run_name="mock_stage_b",
    )
    required = {
        "stage_b_report.md",
        "stage_b_result.json",
        "initial_state.json",
        "controller_limits_snapshot.json",
        "characterization_config.json",
        "tracking_samples.csv",
        "tracking_metrics.json",
        "abort_log.json",
        "brake_abort_result.json",
        "timing_metrics.json",
        "run_manifest.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["optimized_action_loaded"] is False
    assert manifest["gripper_position_target_sent"] is False


def test_94_mock_eventual_vla_path_uses_stage_a_adapter_and_runtime_watchdog(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(validator.ForegroundAbortInput, "start", lambda self: None)
    monkeypatch.setattr(validator.ForegroundAbortInput, "poll", lambda self: None)
    monkeypatch.setattr(validator.ForegroundAbortInput, "close", lambda self: None)
    source_action = source()
    source_hash = validator.array_sha256(source_action)
    limits, _ = validator.load_authoritative_stage_a_limits()
    backend = validator.MockALOHAHardwareBackend(source_action[0], limits)
    safety = reviewed_safety()
    safety.update(
        {
            "controller_limits": limits,
            "gripper_policy": "CLAMP_TO_CONTROLLER_LIMIT",
            "left_gripper_min": limits["left"][6]["position_min"],
            "left_gripper_max": limits["left"][6]["position_max"],
            "right_gripper_min": limits["right"][6]["position_min"],
            "right_gripper_max": limits["right"][6]["position_max"],
            "max_loop_overrun": 1.0,
        }
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        authoritative_stage_a_dir=validator.DEFAULT_AUTHORITATIVE_STAGE_A_RESULT,
        start_frame=0,
        end_frame=1,
        start_transition_seconds=0.1,
        command_hz=30.0,
        speed_scale=1.0,
        workspace_clear_confirmed=True,
        operator_estop_confirmed=False,
        acknowledge_connect_moves_home=True,
        acknowledge_disconnect_moves_home_sleep=True,
        hardware_confirmation=validator.HARDWARE_CONFIRMATION,
        shutdown_confirmation=validator.SHUTDOWN_CONFIRMATION,
    )
    authorization = validator.HardwareAuthorization(True, True, True, ())
    result = validator.run_real_short_replay(
        args,
        source_action,
        backend,
        {},
        authorization,
        safety,
        verified_stop_record(),
        [],
    )
    assert result["runtime_watchdog_active"] is True
    assert result["gripper_policy"] == "GRIPPER_ONLY_SATURATION_TO_REAL_STAGE_A_LIMITS"
    assert backend.commands
    assert all(command["action14"][6] >= 0.0 for command in backend.commands)
    assert all(command["action14"][13] >= 0.0 for command in backend.commands)
    assert validator.array_sha256(source_action) == source_hash
