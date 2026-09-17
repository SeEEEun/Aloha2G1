#!/usr/bin/env python3
"""Classifier-free common execution for the final direct physical A/B test.

The method-specific ACT command supplies every arm and wrist target.  This
module applies the already frozen P14 Dex3 realization at a common timeline
derived from the matched source demonstration and one method-blind nearest-bound
hard-limit projector to the 14 arm/wrist command channels.  There is deliberately
no IK, smoothing, rescue, pose, distance, atlas, readiness, method, or episode
gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from tools.common_execution_layer import (
    ARM_INDICES,
    DEX3_INDICES,
    LEFT_DEX3,
    RIGHT_DEX3,
    WRIST_INDICES,
    Dex3Primitive,
    ExecutionDecision,
    ExecutionSnapshot,
    _transition,
)


DIRECT_INTENTS = {
    "OPEN_INTENT",
    "LEFT_CLOSE_INTENT",
    "LEFT_HOLD_INTENT",
    "HANDOFF_INTENT",
    "RIGHT_HOLD_INTENT",
    "FINAL_RELEASE_INTENT",
}

# This is a numerical guard band, not a learned/tuned grasp parameter.  It is
# applied symmetrically to every Dex3 command using only the authoritative hard
# limits.  The value is small enough (0.2865 deg) to leave the physical P14
# realization unchanged for all practical purposes while keeping both the
# commanded target and the finite-gain PhysX response inside a measured hard
# stop.  It is one method- and joint-independent safety inset.
DEX3_HARD_LIMIT_GUARD_RAD = 5.0e-3
G1_ARM_COMMAND_INDICES = np.arange(0, 14, dtype=np.int64)
# The frozen Dex3 drive has kp=100 and effort_limit=2.5.  A 0.025 rad target
# offset is therefore the largest preload that does not, by itself, request
# more than the actuator's 2.5 Nm effort ceiling.  This is a hardware/drive
# bound shared by both methods, not an outcome-selected grasp parameter.
CONTACT_SEEKING_PRELOAD_MAX_RAD = 2.5e-2
CONTACT_DEBOUNCE_FRAMES = 3
GRASP_CONFIRMATION_RELATIVE_TRANSLATION_M = 1.5e-2
GRASP_LIFT_START_DELTA_M = 5.0e-3
DIGIT_LOCAL_INDICES = {
    "thumb": np.asarray([0, 1, 2], dtype=np.int64),
    "middle": np.asarray([3, 4], dtype=np.int64),
    "index": np.asarray([5, 6], dtype=np.int64),
}


def authoritative_joint_limits(
    joint_contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Read the authoritative 28-D joint-limit contract in canonical order."""

    specs = joint_contract.get("joint_specs")
    names = joint_contract.get("joint_names")
    if not isinstance(specs, list) or not isinstance(names, list) or len(names) != 28:
        raise ValueError("authoritative joint contract is malformed")
    by_index = {int(row["index"]): row for row in specs}
    if set(range(28)) != set(by_index):
        raise ValueError("authoritative joint limits are incomplete")
    ordered_names = tuple(str(names[index]) for index in range(28))
    for index, name in enumerate(ordered_names):
        if str(by_index[index].get("joint_name")) != name:
            raise ValueError("authoritative joint-order mismatch")
    lower = np.asarray(
        [float(by_index[index]["minimum"]) for index in range(28)],
        dtype=np.float64,
    )
    upper = np.asarray(
        [float(by_index[index]["maximum"]) for index in range(28)],
        dtype=np.float64,
    )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
        raise ValueError("authoritative joint-limit interval is invalid")
    return lower, upper, ordered_names


def authoritative_dex3_limits(
    joint_contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Read the measured 14-Dex3 limit contract in canonical 28-D order."""

    lower, upper, names = authoritative_joint_limits(joint_contract)
    return lower[14:].copy(), upper[14:].copy(), names[14:]


@dataclass(frozen=True)
class CommonArmHardLimitProjector:
    """Method-blind nearest-bound projection for the 14 G1 arm/wrist channels."""

    lower_rad: np.ndarray
    upper_rad: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower_rad, dtype=np.float64)
        upper = np.asarray(self.upper_rad, dtype=np.float64)
        if lower.shape != (14,) or upper.shape != (14,):
            raise ValueError("arm hard limits must have shape (14,)")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("arm hard limits contain non-finite values")
        if np.any(lower >= upper):
            raise ValueError("invalid arm hard-limit interval")
        object.__setattr__(self, "lower_rad", lower.copy())
        object.__setattr__(self, "upper_rad", upper.copy())

    def project(self, command_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        command = np.asarray(command_rad, dtype=np.float64)
        if command.shape[-1:] != (14,) or not np.isfinite(command).all():
            raise ValueError("arm command must be finite with trailing dimension 14")
        projected = np.clip(command, self.lower_rad, self.upper_rad)
        changed = projected != command
        return projected, changed


class DirectPhysicalDex3ExecutionLayer:
    """Causal method-blind finger controller with no pre-grasp classifier."""

    def __init__(
        self,
        primitive: Dex3Primitive,
        common_task_intent: Sequence[str],
        raw_policy_command: np.ndarray,
        policy_safe_command: np.ndarray,
        method: str,
        arm_lower_rad: np.ndarray,
        arm_upper_rad: np.ndarray,
        dex3_lower_rad: np.ndarray,
        dex3_upper_rad: np.ndarray,
        hard_limit_guard_rad: float = DEX3_HARD_LIMIT_GUARD_RAD,
        standardized_initial_grasp: bool = False,
        standardized_initial_q_rad: np.ndarray | None = None,
        standardized_left_hold_target_rad: np.ndarray | None = None,
    ) -> None:
        if method not in {"ACT-A40", "ACT-B40"}:
            raise ValueError("direct layer accepts only ACT-A40/ACT-B40")
        raw = np.asarray(raw_policy_command, dtype=np.float64)
        safe = np.asarray(policy_safe_command, dtype=np.float64)
        intent = np.asarray(common_task_intent).astype(str)
        if raw.shape != safe.shape or raw.ndim != 2 or raw.shape[1] != 28:
            raise ValueError("ACT command must have shape (T, 28)")
        if intent.shape != (len(raw),):
            raise ValueError("common intent and command lengths differ")
        if not set(np.unique(intent)).issubset(DIRECT_INTENTS):
            raise ValueError("unknown direct common task intent")
        if not np.isfinite(raw).all() or not np.isfinite(safe).all():
            raise ValueError("ACT command contains non-finite values")
        lower = np.asarray(dex3_lower_rad, dtype=np.float64)
        upper = np.asarray(dex3_upper_rad, dtype=np.float64)
        if lower.shape != (14,) or upper.shape != (14,):
            raise ValueError("Dex3 hard limits must have shape (14,)")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("Dex3 hard limits contain non-finite values")
        if np.any(lower >= upper):
            raise ValueError("invalid Dex3 hard-limit interval")
        guard = float(hard_limit_guard_rad)
        if not np.isfinite(guard) or guard < 0.0:
            raise ValueError("invalid Dex3 hard-limit guard")
        if np.any(lower + guard >= upper - guard):
            raise ValueError("Dex3 hard-limit guard collapses an interval")
        self.primitive = primitive
        self.intent = intent
        self.raw = raw
        self.safe = safe
        self.method = method
        self.arm_hard_limit_projector = CommonArmHardLimitProjector(
            arm_lower_rad, arm_upper_rad
        )
        self.dex3_hard_lower_rad = lower.copy()
        self.dex3_hard_upper_rad = upper.copy()
        self.dex3_command_lower_rad = lower + guard
        self.dex3_command_upper_rad = upper - guard
        self.hard_limit_guard_rad = guard
        self.standardized_initial_grasp = bool(standardized_initial_grasp)
        endpoints = np.concatenate(
            (
                primitive.left_open,
                primitive.left_preshape,
                primitive.left_full_close,
                primitive.right_open,
                primitive.right_preshape,
                primitive.right_full_close,
            )
        )
        endpoint_lower = np.concatenate(
            (
                *([self.dex3_command_lower_rad[:7]] * 3),
                *([self.dex3_command_lower_rad[7:]] * 3),
            )
        )
        endpoint_upper = np.concatenate(
            (
                *([self.dex3_command_upper_rad[:7]] * 3),
                *([self.dex3_command_upper_rad[7:]] * 3),
            )
        )
        projected = np.clip(endpoints, endpoint_lower, endpoint_upper)
        self.endpoint_limit_projection_scalar_count = int(
            np.count_nonzero(projected != endpoints)
        )
        split = np.split(projected, 6)
        (
            self.left_open,
            self.left_preshape,
            self.left_full_close,
            self.right_open,
            self.right_preshape,
            self.right_full_close,
        ) = (value.copy() for value in split)
        self.frame_limit_saturation_scalar_count = 0
        self.left_trigger: int | None = None
        self.left_start_q: np.ndarray | None = None
        self.left_owned_frame: int | None = None
        self.right_trigger: int | None = None
        self.right_start_q: np.ndarray | None = None
        self.right_support_frame: int | None = None
        self.left_release_frame: int | None = None
        self.left_release_start_q: np.ndarray | None = None
        self.right_owned_frame: int | None = None
        self.right_release_frame: int | None = None
        self.right_release_start_q: np.ndarray | None = None
        self.contact_latched_targets: dict[str, dict[str, np.ndarray]] = {
            "left": {},
            "right": {},
        }
        self.contact_latch_frames: dict[str, dict[str, int]] = {
            "left": {},
            "right": {},
        }
        self.grasp_state: dict[str, str] = {"left": "OPEN", "right": "OPEN"}
        self.first_digit_contact_frame: dict[str, dict[str, int | None]] = {
            side: {digit: None for digit in DIGIT_LOCAL_INDICES}
            for side in ("left", "right")
        }
        self.close_digit_contact_frame: dict[str, dict[str, int | None]] = {
            side: {digit: None for digit in DIGIT_LOCAL_INDICES}
            for side in ("left", "right")
        }
        self.digit_contact_debounce: dict[str, dict[str, int]] = {
            side: {digit: 0 for digit in DIGIT_LOCAL_INDICES}
            for side in ("left", "right")
        }
        self.digit_contact_loss: dict[str, dict[str, int]] = {
            side: {digit: 0 for digit in DIGIT_LOCAL_INDICES}
            for side in ("left", "right")
        }
        self.enclosure_candidate_frame: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self.grasp_confirmed_frame: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self.table_support_loss_frame: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self.lift_start_frame: dict[str, int | None] = {"left": None, "right": None}
        self.grasp_confirmed_object_z_m: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self.enclosure_relative_translation_m: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }
        self.enclosure_debounce: dict[str, int] = {"left": 0, "right": 0}
        self.table_free_debounce: dict[str, int] = {"left": 0, "right": 0}
        self.preload_start_frame: dict[str, int | None] = {
            "left": None,
            "right": None,
        }
        self.contact_loss_events: dict[str, list[dict[str, Any]]] = {
            "left": [],
            "right": [],
        }
        self.contact_latch_start_targets: dict[str, dict[str, np.ndarray]] = {
            "left": {},
            "right": {},
        }
        self.maximum_relative_translation_during_confirmation_m: dict[str, float] = {
            "left": 0.0,
            "right": 0.0,
        }
        self.left_support_counter = 0
        self.right_three_counter = 0
        self.right_retention_counter = 0
        self.last_frame = -1
        self.trace: list[ExecutionDecision] = []
        self.arm_hard_limit_projection_masks: list[np.ndarray] = []
        self.arm_hard_limit_projection_corrections: list[np.ndarray] = []

        self.standardized_left_hold_target_rad: np.ndarray | None = None
        if self.standardized_initial_grasp:
            initial = np.asarray(standardized_initial_q_rad, dtype=np.float64)
            hold = np.asarray(standardized_left_hold_target_rad, dtype=np.float64)
            if initial.shape != (28,) or not np.isfinite(initial).all():
                raise ValueError("standardized initial q must be finite with shape (28,)")
            if hold.shape != (7,) or not np.isfinite(hold).all():
                raise ValueError("standardized LEFT hold target must be finite with shape (7,)")
            if str(intent[0]) != "LEFT_HOLD_INTENT":
                raise ValueError("standardized-grasp execution must begin in LEFT_HOLD_INTENT")
            self.standardized_left_hold_target_rad = self._bounded(hold, 0)
            if not np.allclose(
                initial[14:21],
                np.clip(initial[14:21], self.dex3_hard_lower_rad[:7], self.dex3_hard_upper_rad[:7]),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("standardized LEFT grasp state violates an authoritative limit")
            # The state is restored from a persisted, table-free, contact-qualified
            # physical HOLD sample.  This seeds only controller state; continued
            # retention is resolved anew by contact dynamics in every rollout.
            self.left_trigger = 0
            self.left_start_q = initial[14:21].copy()
            self.left_owned_frame = 0
            self.grasp_confirmed_frame["left"] = 0
            self.table_support_loss_frame["left"] = 0
            self.lift_start_frame["left"] = 0
            self.preload_start_frame["left"] = 0
            self.grasp_state["left"] = "HOLD"
            for digit, indices in DIGIT_LOCAL_INDICES.items():
                self.contact_latched_targets["left"][digit] = (
                    self.standardized_left_hold_target_rad[indices].copy()
                )
                self.contact_latch_start_targets["left"][digit] = (
                    self.standardized_left_hold_target_rad[indices].copy()
                )
                self.contact_latch_frames["left"][digit] = 0

    @staticmethod
    def _three_digit(force: Mapping[str, float], threshold: float) -> bool:
        return all(
            float(force.get(digit, 0.0)) >= threshold
            for digit in ("thumb", "index", "middle")
        )

    @staticmethod
    def _two_digit(force: Mapping[str, float], threshold: float) -> bool:
        return sum(
            float(force.get(digit, 0.0)) >= threshold
            for digit in ("thumb", "index", "middle")
        ) >= 2

    def _bounded(self, value: np.ndarray, side_offset: int) -> np.ndarray:
        lower = self.dex3_command_lower_rad[side_offset : side_offset + 7]
        upper = self.dex3_command_upper_rad[side_offset : side_offset + 7]
        result = np.clip(np.asarray(value, dtype=np.float64), lower, upper)
        self.frame_limit_saturation_scalar_count += int(
            np.count_nonzero(result != value)
        )
        return result

    def _previous_executed_dex3(self, indices: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        if not self.trace:
            return fallback.copy()
        return np.asarray(self.trace[-1].executed_command[indices], dtype=np.float64).copy()

    @staticmethod
    def _relative_translation(snapshot: ExecutionSnapshot, side: str) -> np.ndarray:
        hand = np.asarray(snapshot.whole_hand_world[side], dtype=np.float64)
        obj = np.asarray(snapshot.object_world, dtype=np.float64)
        return hand[:3, :3].T @ (obj[:3, 3] - hand[:3, 3])

    def _digit_moved_beyond_preshape(
        self, side: str, measured: np.ndarray, digit: str
    ) -> bool:
        indices = DIGIT_LOCAL_INDICES[digit]
        preshape = self.left_preshape if side == "left" else self.right_preshape
        full_close = self.left_full_close if side == "left" else self.right_full_close
        direction = np.sign(full_close[indices] - preshape[indices])
        signed_motion = (measured[indices] - preshape[indices]) * direction
        return bool(np.max(signed_motion, initial=0.0) > 1.0e-4)

    def _advance_grasp_state(
        self,
        side: str,
        frame: int,
        snapshot: ExecutionSnapshot,
        trigger: int | None,
        release_active: bool,
    ) -> list[str]:
        """Advance a contact-only grasp state machine without changing arm motion."""

        events: list[str] = []
        force = snapshot.digit_force_n[side]
        for digit in DIGIT_LOCAL_INDICES:
            if (
                self.first_digit_contact_frame[side][digit] is None
                and float(force.get(digit, 0.0)) >= self.primitive.force_threshold_n
            ):
                self.first_digit_contact_frame[side][digit] = frame
                events.append(f"{side.upper()}_{digit.upper()}_FIRST_DOLL_CONTACT")

        if release_active:
            if self.grasp_state[side] != "RELEASE":
                self.grasp_state[side] = "RELEASE"
                events.append(f"{side.upper()}_GRASP_STATE_RELEASE")
            return events
        if trigger is None:
            self.grasp_state[side] = "OPEN"
            return events

        elapsed = frame - trigger
        if (
            elapsed < self.primitive.preshape_frames
            and self.grasp_confirmed_frame[side] is None
        ):
            # PRESHAPE contacts are deliberately provisional. They cannot latch,
            # debounce toward confirmation, or stop the nominal preshape motion.
            if self.grasp_state[side] != "PRESHAPE":
                self.grasp_state[side] = "PRESHAPE"
                events.append(f"{side.upper()}_GRASP_STATE_PRESHAPE")
            for digit in DIGIT_LOCAL_INDICES:
                self.digit_contact_debounce[side][digit] = 0
                self.digit_contact_loss[side][digit] = 0
            self.enclosure_debounce[side] = 0
            self.table_free_debounce[side] = 0
            self.enclosure_relative_translation_m[side] = None
            return events

        if self.grasp_confirmed_frame[side] is None:
            if self.grasp_state[side] in {"OPEN", "PRESHAPE"}:
                self.grasp_state[side] = "PROGRESSIVE_CLOSE"
                events.append(f"{side.upper()}_GRASP_STATE_PROGRESSIVE_CLOSE")
            for digit in DIGIT_LOCAL_INDICES:
                meaningful = (
                    float(force.get(digit, 0.0)) >= self.primitive.force_threshold_n
                )
                if meaningful:
                    self.digit_contact_debounce[side][digit] += 1
                    if self.close_digit_contact_frame[side][digit] is None:
                        self.close_digit_contact_frame[side][digit] = frame
                else:
                    self.digit_contact_debounce[side][digit] = 0

            side_offset = 0 if side == "left" else 7
            measured = np.asarray(
                snapshot.measured_q_rad[14 + side_offset : 21 + side_offset],
                dtype=np.float64,
            )
            persistent = {
                digit: (
                    self.digit_contact_debounce[side][digit]
                    >= CONTACT_DEBOUNCE_FRAMES
                    and self._digit_moved_beyond_preshape(side, measured, digit)
                )
                for digit in DIGIT_LOCAL_INDICES
            }
            opposing = bool(
                persistent["thumb"]
                and (persistent["index"] or persistent["middle"])
            )
            relative = self._relative_translation(snapshot, side)
            reference = self.enclosure_relative_translation_m[side]
            bounded = True
            if reference is not None:
                displacement = float(np.linalg.norm(relative - reference))
                self.maximum_relative_translation_during_confirmation_m[side] = max(
                    self.maximum_relative_translation_during_confirmation_m[side],
                    displacement,
                )
                bounded = displacement <= GRASP_CONFIRMATION_RELATIVE_TRANSLATION_M

            if opposing and bounded:
                if reference is None:
                    self.enclosure_relative_translation_m[side] = relative.copy()
                self.enclosure_debounce[side] += 1
            else:
                # A superficial/transient brush cannot become a permanent latch.
                if self.grasp_state[side] == "GRASP_CONFIRM":
                    events.append(f"{side.upper()}_TRANSIENT_CONTACT_UNLATCH")
                self.enclosure_debounce[side] = 0
                self.table_free_debounce[side] = 0
                self.enclosure_relative_translation_m[side] = None
                self.grasp_state[side] = "PROGRESSIVE_CLOSE"

            if self.enclosure_debounce[side] >= CONTACT_DEBOUNCE_FRAMES:
                if self.enclosure_candidate_frame[side] is None:
                    self.enclosure_candidate_frame[side] = frame
                    events.append(f"{side.upper()}_OPPOSING_ENCLOSURE_CANDIDATE")
                self.grasp_state[side] = "GRASP_CONFIRM"

            # Confirm the closed-hand enclosure before the scheduled arm lift.
            # Requiring table support loss here would create a causal deadlock:
            # the object cannot become table-free until the unchanged arm lift
            # begins, while the arm lift is required to begin only after grasp
            # confirmation.  We therefore require a second full debounce window
            # of persistent opposing contact and bounded hand/object relative
            # motion.  Table support loss remains a mandatory post-confirmation
            # mechanical validation and is required before entering LIFT below.
            if (
                self.grasp_state[side] == "GRASP_CONFIRM"
                and self.enclosure_debounce[side] >= 2 * CONTACT_DEBOUNCE_FRAMES
            ):
                self.grasp_confirmed_frame[side] = frame
                self.grasp_confirmed_object_z_m[side] = float(snapshot.object_world[2, 3])
                self.preload_start_frame[side] = frame
                self.grasp_state[side] = "PRELOAD"
                full_close = self.left_full_close if side == "left" else self.right_full_close
                nominal = self._left_target(frame) if side == "left" else self._right_target(frame)
                for digit, indices in DIGIT_LOCAL_INDICES.items():
                    if persistent[digit]:
                        preload = np.clip(
                            full_close[indices] - nominal[indices],
                            -CONTACT_SEEKING_PRELOAD_MAX_RAD,
                            CONTACT_SEEKING_PRELOAD_MAX_RAD,
                        )
                        self.contact_latched_targets[side][digit] = (
                            nominal[indices] + preload
                        )
                        previous_target = self._previous_executed_dex3(
                            LEFT_DEX3 if side == "left" else RIGHT_DEX3,
                            self.left_preshape if side == "left" else self.right_preshape,
                        )
                        self.contact_latch_start_targets[side][digit] = (
                            previous_target[indices].copy()
                        )
                        self.contact_latch_frames[side][digit] = frame
                        events.append(
                            f"{side.upper()}_{digit.upper()}_CONFIRMED_PRELOAD_LATCH"
                        )
                events.extend(
                    (
                        f"{side.upper()}_GRASP_CONFIRMED",
                        f"{side.upper()}_GRASP_STATE_PRELOAD",
                    )
                )
            elif self.grasp_state[side] not in {"GRASP_CONFIRM", "PRELOAD"}:
                self.grasp_state[side] = "PROGRESSIVE_CLOSE"
            return events

        # Confirmed grasp: maintain only a bounded joint-space preload. A digit
        # that loses real doll contact is unlatched and allowed to close again.
        side_offset = 0 if side == "left" else 7
        measured = np.asarray(
            snapshot.measured_q_rad[14 + side_offset : 21 + side_offset],
            dtype=np.float64,
        )
        full_close = self.left_full_close if side == "left" else self.right_full_close
        nominal = self._left_target(frame) if side == "left" else self._right_target(frame)
        for digit, indices in DIGIT_LOCAL_INDICES.items():
            meaningful = float(force.get(digit, 0.0)) >= self.primitive.force_threshold_n
            if meaningful:
                self.digit_contact_loss[side][digit] = 0
                self.digit_contact_debounce[side][digit] += 1
                if (
                    digit not in self.contact_latched_targets[side]
                    and self.digit_contact_debounce[side][digit]
                    >= CONTACT_DEBOUNCE_FRAMES
                    and self._digit_moved_beyond_preshape(side, measured, digit)
                ):
                    preload = np.clip(
                        full_close[indices] - nominal[indices],
                        -CONTACT_SEEKING_PRELOAD_MAX_RAD,
                        CONTACT_SEEKING_PRELOAD_MAX_RAD,
                    )
                    self.contact_latched_targets[side][digit] = nominal[indices] + preload
                    previous_target = self._previous_executed_dex3(
                        LEFT_DEX3 if side == "left" else RIGHT_DEX3,
                        self.left_preshape if side == "left" else self.right_preshape,
                    )
                    self.contact_latch_start_targets[side][digit] = (
                        previous_target[indices].copy()
                    )
                    self.contact_latch_frames[side][digit] = frame
                    events.append(f"{side.upper()}_{digit.upper()}_CONTACT_RELATCH")
            else:
                self.digit_contact_debounce[side][digit] = 0
                self.digit_contact_loss[side][digit] += 1
                if (
                    digit in self.contact_latched_targets[side]
                    and self.digit_contact_loss[side][digit] >= CONTACT_DEBOUNCE_FRAMES
                ):
                    del self.contact_latched_targets[side][digit]
                    self.contact_latch_start_targets[side].pop(digit, None)
                    self.contact_loss_events[side].append({"digit": digit, "frame": frame})
                    events.append(f"{side.upper()}_{digit.upper()}_CONTACT_LOST_UNLATCH")

        table_free = float(snapshot.table_force_n) <= self.primitive.maximum_table_force_n
        if table_free:
            self.table_free_debounce[side] += 1
        else:
            self.table_free_debounce[side] = 0
        if (
            self.table_support_loss_frame[side] is None
            and self.table_free_debounce[side] >= CONTACT_DEBOUNCE_FRAMES
        ):
            self.table_support_loss_frame[side] = frame - CONTACT_DEBOUNCE_FRAMES + 1
            events.append(f"{side.upper()}_TABLE_SUPPORT_LOST")

        if self.grasp_state[side] == "PRELOAD":
            assert self.preload_start_frame[side] is not None
            if frame - int(self.preload_start_frame[side]) >= CONTACT_DEBOUNCE_FRAMES:
                self.grasp_state[side] = "HOLD"
                events.append(f"{side.upper()}_GRASP_STATE_HOLD")
        elif self.grasp_state[side] == "HOLD":
            start_z = self.grasp_confirmed_object_z_m[side]
            if (
                start_z is not None
                and self.table_support_loss_frame[side] is not None
                and float(snapshot.object_world[2, 3]) - start_z
                >= GRASP_LIFT_START_DELTA_M
            ):
                self.grasp_state[side] = "LIFT"
                self.lift_start_frame[side] = frame
                events.append(f"{side.upper()}_GRASP_STATE_LIFT")
        return events

    def _left_target(self, frame: int) -> np.ndarray:
        if self.left_trigger is None or self.left_start_q is None:
            return self.left_open.copy()
        p = self.primitive
        if self.left_release_frame is not None:
            if self.left_release_start_q is None:
                raise RuntimeError("missing left release state")
            return _transition(
                self.left_release_start_q,
                self.left_open,
                frame - self.left_release_frame,
                p.release_frames,
            )
        if self.standardized_initial_grasp:
            assert self.standardized_left_hold_target_rad is not None
            return self.standardized_left_hold_target_rad.copy()
        elapsed = frame - self.left_trigger
        if elapsed < p.preshape_frames:
            return _transition(
                self.left_start_q, self.left_preshape, elapsed, p.preshape_frames
            )
        elapsed -= p.preshape_frames
        if elapsed < p.close_frames:
            return _transition(
                self.left_preshape, self.left_full_close, elapsed, p.close_frames
            )
        return self.left_full_close.copy()

    def _right_target(self, frame: int) -> np.ndarray:
        if self.right_trigger is None or self.right_start_q is None:
            return self.right_open.copy()
        p = self.primitive
        if self.right_release_frame is not None:
            if self.right_release_start_q is None:
                raise RuntimeError("missing right release state")
            return _transition(
                self.right_release_start_q,
                self.right_open,
                frame - self.right_release_frame,
                p.release_frames,
            )
        elapsed = frame - self.right_trigger
        if elapsed < p.preshape_frames:
            return _transition(
                self.right_start_q, self.right_preshape, elapsed, p.preshape_frames
            )
        elapsed -= p.preshape_frames
        if elapsed < p.close_frames:
            return _transition(
                self.right_preshape, self.right_full_close, elapsed, p.close_frames
            )
        return self.right_full_close.copy()

    def _contact_seek(
        self,
        side: str,
        frame: int,
        snapshot: ExecutionSnapshot,
        nominal_target: np.ndarray,
        release_active: bool,
    ) -> tuple[np.ndarray, list[str]]:
        """Apply confirmed preload; all earlier contacts remain provisional."""
        if release_active or self.grasp_confirmed_frame[side] is None:
            return nominal_target, []
        target = np.asarray(nominal_target, dtype=np.float64).copy()
        for digit, local_indices in DIGIT_LOCAL_INDICES.items():
            if digit in self.contact_latched_targets[side]:
                start = self.contact_latch_start_targets[side][digit]
                target[local_indices] = _transition(
                    start,
                    self.contact_latched_targets[side][digit],
                    frame - self.contact_latch_frames[side][digit],
                    CONTACT_DEBOUNCE_FRAMES,
                )
            else:
                # A confirmed digit that subsequently loses contact resumes
                # closing, but by no more than the frozen progressive-close rate.
                previous = self._previous_executed_dex3(
                    LEFT_DEX3 if side == "left" else RIGHT_DEX3,
                    self.left_preshape if side == "left" else self.right_preshape,
                )
                preshape = self.left_preshape if side == "left" else self.right_preshape
                full_close = self.left_full_close if side == "left" else self.right_full_close
                maximum_step = (
                    np.abs(full_close[local_indices] - preshape[local_indices])
                    / max(1, self.primitive.close_frames)
                )
                target[local_indices] = previous[local_indices] + np.clip(
                    target[local_indices] - previous[local_indices],
                    -maximum_step,
                    maximum_step,
                )
        return target, []

    def _phase(self, frame: int) -> str:
        value = str(self.intent[frame])
        return {
            "OPEN_INTENT": "APPROACH",
            "LEFT_CLOSE_INTENT": "GRASP",
            "LEFT_HOLD_INTENT": "LEFT_HOLD",
            "HANDOFF_INTENT": "HANDOFF",
            "RIGHT_HOLD_INTENT": "TRANSPORT",
            "FINAL_RELEASE_INTENT": "RELEASE",
        }[value]

    def step(self, frame: int, snapshot: ExecutionSnapshot) -> ExecutionDecision:
        if frame != self.last_frame + 1 or frame >= len(self.safe):
            raise RuntimeError("direct controller requires one ordered call per frame")
        self.last_frame = frame
        p = self.primitive
        intent = str(self.intent[frame])
        base = self.safe[frame].copy()
        events: list[str] = []

        if self.standardized_initial_grasp and frame == 0:
            self.grasp_confirmed_object_z_m["left"] = float(
                snapshot.object_world[2, 3]
            )
            events.extend(
                (
                    "STANDARDIZED_INITIAL_LEFT_GRASP_RESTORED",
                    "LEFT_GRASP_CONFIRMED_AT_INITIALIZATION",
                    "LEFT_PHYSICAL_OWNERSHIP_AT_INITIALIZATION",
                )
            )

        # No geometric, wrist-distance, atlas, or classifier condition exists.
        if self.left_trigger is None and intent == "LEFT_CLOSE_INTENT":
            self.left_trigger = frame
            self.left_start_q = self._previous_executed_dex3(
                LEFT_DEX3, self.left_open
            )
            events.append("LEFT_CLOSE_INTENT_START")

        # The same source-derived handoff opportunity is granted to both methods,
        # even when the receiving hand has not contacted the doll.  Physics then
        # determines whether closure actually acquires it.
        if self.right_trigger is None and intent == "HANDOFF_INTENT":
            self.right_trigger = frame
            self.right_start_q = self._previous_executed_dex3(
                RIGHT_DEX3, self.right_open
            )
            events.append("RIGHT_HANDOFF_CLOSE_INTENT_START")

        # Release is source-intent driven. There is no position/bin rescue gate.
        if intent == "FINAL_RELEASE_INTENT":
            if self.right_trigger is not None and self.right_release_frame is None:
                self.right_release_frame = frame
                self.right_release_start_q = self._previous_executed_dex3(
                    RIGHT_DEX3, self.right_full_close
                )
                events.append("FINAL_RELEASE_INTENT_START")
            elif (
                self.right_trigger is None
                and self.left_trigger is not None
                and self.left_release_frame is None
            ):
                self.left_release_frame = frame
                self.left_release_start_q = self._previous_executed_dex3(
                    LEFT_DEX3, self.left_full_close
                )
                events.append("LEFT_STANDALONE_RELEASE_INTENT_START")

        events.extend(
            self._advance_grasp_state(
                "left",
                frame,
                snapshot,
                self.left_trigger,
                self.left_release_frame is not None,
            )
        )
        events.extend(
            self._advance_grasp_state(
                "right",
                frame,
                snapshot,
                self.right_trigger,
                self.right_release_frame is not None,
            )
        )

        left_force = snapshot.digit_force_n["left"]
        right_force = snapshot.digit_force_n["right"]
        table_free = float(snapshot.table_force_n) <= p.maximum_table_force_n
        previous = snapshot.previous_control_frame_support
        left_two = (
            bool(previous["left_two_table_free"])
            if previous is not None
            else self._two_digit(left_force, p.force_threshold_n) and table_free
        )
        if self.grasp_confirmed_frame["left"] is not None and left_two:
            self.left_support_counter += 1
        else:
            self.left_support_counter = 0
        if (
            self.left_owned_frame is None
            and self.left_support_counter >= p.left_retention_frames
        ):
            self.left_owned_frame = frame
            events.append("LEFT_PHYSICAL_OWNERSHIP")

        right_support = (
            bool(previous["right_two_table_free"])
            if previous is not None
            else self._two_digit(right_force, p.force_threshold_n) and table_free
        )
        if self.grasp_confirmed_frame["right"] is not None and right_support:
            self.right_three_counter += 1
        else:
            self.right_three_counter = 0
        if (
            self.right_support_frame is None
            and self.right_three_counter >= p.right_verification_frames
        ):
            self.right_support_frame = frame
            self.left_release_frame = frame
            self.left_release_start_q = self._previous_executed_dex3(
                LEFT_DEX3, self.left_full_close
            )
            events.extend(
                ("RIGHT_RETENTION_SUPPORT_CONFIRMED", "GIVING_HAND_RELEASE_START")
            )

        left_release_complete = bool(
            self.left_release_frame is not None
            and frame - self.left_release_frame >= p.release_frames - 1
        )
        right_two = (
            bool(previous["right_two_table_free"])
            if previous is not None
            else self._two_digit(right_force, p.force_threshold_n) and table_free
        )
        if left_release_complete and right_two:
            self.right_retention_counter += 1
        else:
            self.right_retention_counter = 0
        if (
            self.right_owned_frame is None
            and self.right_retention_counter >= p.right_retention_frames
        ):
            self.right_owned_frame = frame
            events.append("RIGHT_PHYSICAL_OWNERSHIP")

        executed = base.copy()
        projected_arm, projected_arm_mask = self.arm_hard_limit_projector.project(
            base[G1_ARM_COMMAND_INDICES]
        )
        executed[G1_ARM_COMMAND_INDICES] = projected_arm
        arm_limit_mask = np.zeros(28, dtype=bool)
        arm_limit_mask[G1_ARM_COMMAND_INDICES] = projected_arm_mask
        arm_limit_correction = np.zeros(28, dtype=np.float64)
        arm_limit_correction[G1_ARM_COMMAND_INDICES] = np.abs(
            projected_arm - base[G1_ARM_COMMAND_INDICES]
        )
        self.arm_hard_limit_projection_masks.append(arm_limit_mask)
        self.arm_hard_limit_projection_corrections.append(arm_limit_correction)
        left_target, left_seek_events = self._contact_seek(
            "left",
            frame,
            snapshot,
            self._left_target(frame),
            self.left_release_frame is not None,
        )
        right_target, right_seek_events = self._contact_seek(
            "right",
            frame,
            snapshot,
            self._right_target(frame),
            self.right_release_frame is not None,
        )
        events.extend(left_seek_events)
        events.extend(right_seek_events)
        executed[LEFT_DEX3] = self._bounded(left_target, 0)
        executed[RIGHT_DEX3] = self._bounded(right_target, 7)
        arm = executed[G1_ARM_COMMAND_INDICES]
        if np.any(arm < self.arm_hard_limit_projector.lower_rad) or np.any(
            arm > self.arm_hard_limit_projector.upper_rad
        ):
            raise RuntimeError("COMMON EXECUTION LAYER INVALID: arm hard limit")
        dex3 = executed[DEX3_INDICES]
        if np.any(dex3 < self.dex3_hard_lower_rad) or np.any(
            dex3 > self.dex3_hard_upper_rad
        ):
            raise RuntimeError("COMMON EXECUTION LAYER INVALID: Dex3 hard limit")
        expected_arm = np.clip(
            base[G1_ARM_COMMAND_INDICES],
            self.arm_hard_limit_projector.lower_rad,
            self.arm_hard_limit_projector.upper_rad,
        )
        if not np.array_equal(executed[G1_ARM_COMMAND_INDICES], expected_arm):
            raise RuntimeError("COMMON EXECUTION LAYER INVALID: non-projector arm change")
        override = executed != base
        arm_mask = np.zeros(28, dtype=bool)
        wrist_mask = np.zeros(28, dtype=bool)
        dex3_mask = np.zeros(28, dtype=bool)
        # Arm/wrist rescue masks remain zero.  Mandatory componentwise hard-limit
        # projection is logged separately in arm_hard_limit_projection_masks.
        dex3_mask[DEX3_INDICES] = override[DEX3_INDICES]
        decision = ExecutionDecision(
            executed_command=executed,
            override_mask=override,
            arm_override_mask=arm_mask,
            wrist_override_mask=wrist_mask,
            dex3_override_mask=dex3_mask,
            phase=self._phase(frame),
            graspability_margin=float("nan"),
            left_envelope_eligible=False,
            events=tuple(events),
        )
        self.trace.append(decision)
        return decision

    def summary(self) -> dict[str, Any]:
        arm_rescue_masks = (
            np.stack([row.arm_override_mask for row in self.trace])
            if self.trace
            else np.zeros((0, 28), dtype=bool)
        )
        wrist_rescue_masks = (
            np.stack([row.wrist_override_mask for row in self.trace])
            if self.trace
            else np.zeros((0, 28), dtype=bool)
        )
        dex3_masks = (
            np.stack([row.dex3_override_mask for row in self.trace])
            if self.trace
            else np.zeros((0, 28), dtype=bool)
        )
        safety = self.safe[: len(self.trace)] - self.raw[: len(self.trace)]
        arm_limit_masks = (
            np.stack(self.arm_hard_limit_projection_masks)
            if self.arm_hard_limit_projection_masks
            else np.zeros((0, 28), dtype=bool)
        )
        arm_limit_corrections = (
            np.stack(self.arm_hard_limit_projection_corrections)
            if self.arm_hard_limit_projection_corrections
            else np.zeros((0, 28), dtype=np.float64)
        )
        return {
            "schema_version": "direct_physical_dex3_runtime_summary_v1",
            "method": self.method,
            "standardized_initial_grasp": self.standardized_initial_grasp,
            "standardized_initial_grasp_counted_as_method_success": False,
            "pregrasp_classifier_used": False,
            "graspability_atlas_used": False,
            "wrist_distance_gate_used": False,
            "frames": len(self.trace),
            "events": {
                "left_close_intent_frame": self.left_trigger,
                "left_physical_ownership_frame": self.left_owned_frame,
                "right_handoff_close_intent_frame": self.right_trigger,
                "right_retention_support_frame": self.right_support_frame,
                "right_three_digit_support_frame": self.right_support_frame,
                "giving_hand_release_frame": self.left_release_frame,
                "right_physical_ownership_frame": self.right_owned_frame,
                "final_release_intent_frame": self.right_release_frame,
                "contact_seek_latch_frames": self.contact_latch_frames,
                "first_digit_doll_contact_frames": self.first_digit_contact_frame,
                "first_close_phase_digit_contact_frames": self.close_digit_contact_frame,
                "opposing_enclosure_candidate_frames": self.enclosure_candidate_frame,
                "grasp_confirmed_frames": self.grasp_confirmed_frame,
                "table_support_loss_frames": self.table_support_loss_frame,
                "lift_start_frames": self.lift_start_frame,
                "contact_loss_unlatch_events": self.contact_loss_events,
            },
            "grasp_state_machine": [
                "OPEN",
                "PRESHAPE",
                "PROGRESSIVE_CLOSE",
                "GRASP_CONFIRM",
                "PRELOAD",
                "HOLD",
                "LIFT",
                "RELEASE",
            ],
            "final_grasp_state": dict(self.grasp_state),
            "preshape_contact_is_provisional": True,
            "preshape_permanent_latch_disabled": True,
            "transient_contact_may_unlatch": True,
            "contact_debounce_frames": CONTACT_DEBOUNCE_FRAMES,
            "mechanical_grasp_confirmation": {
                "requires_doll_filtered_contact": True,
                "requires_thumb_and_opposing_digit": True,
                "requires_digit_motion_beyond_preshape": True,
                "requires_bounded_object_hand_relative_translation": True,
                "relative_translation_bound_m": GRASP_CONFIRMATION_RELATIVE_TRANSLATION_M,
                "requires_table_support_loss_before_confirmation": False,
                "requires_table_support_loss_before_lift_state": True,
                "confirmation_debounce_frames_after_candidate": CONTACT_DEBOUNCE_FRAMES,
                "maximum_relative_translation_observed_m": dict(
                    self.maximum_relative_translation_during_confirmation_m
                ),
            },
            "lift_before_confirmed_grasp": False,
            "contact_seeking_close": True,
            "contact_seeking_scope": "DEX3_DIGITS_ONLY",
            "contact_seeking_preload_max_rad": CONTACT_SEEKING_PRELOAD_MAX_RAD,
            "contact_seek_contact_threshold_n": self.primitive.force_threshold_n,
            "contacted_digit_action": "after mechanical confirmation only: debounced reversible latch plus bounded preload toward frozen P14 full-close",
            "no_contact_digit_action": "continue progressive frozen P14 close to the safe limit",
            "physical_support_topology": "at least two contacted digits with table-free retention; topology recorded separately",
            "arm_common_override_scalar_count": int(
                np.count_nonzero(arm_rescue_masks[:, ARM_INDICES])
                if len(arm_rescue_masks)
                else 0
            ),
            "wrist_common_override_scalar_count": int(
                np.count_nonzero(wrist_rescue_masks[:, WRIST_INDICES])
                if len(wrist_rescue_masks)
                else 0
            ),
            "common_arm_hard_limit_projector": "NEAREST_VALID_VALUE_COMPONENTWISE",
            "common_arm_hard_limit_projector_same_for_a_b": True,
            "common_arm_hard_limit_projection_scalar_count": int(
                np.count_nonzero(arm_limit_masks[:, G1_ARM_COMMAND_INDICES])
                if len(arm_limit_masks)
                else 0
            ),
            "common_wrist_hard_limit_projection_scalar_count": int(
                np.count_nonzero(arm_limit_masks[:, WRIST_INDICES])
                if len(arm_limit_masks)
                else 0
            ),
            "maximum_common_arm_hard_limit_correction_rad": float(
                np.max(
                    arm_limit_corrections[:, G1_ARM_COMMAND_INDICES], initial=0.0
                )
                if len(arm_limit_corrections)
                else 0.0
            ),
            "arm_hard_lower_rad": self.arm_hard_limit_projector.lower_rad.tolist(),
            "arm_hard_upper_rad": self.arm_hard_limit_projector.upper_rad.tolist(),
            "arm_hard_limits_enforced_every_commanded_frame": True,
            "arm_hard_limit_violation_scalar_count_after_projection": int(
                sum(
                    np.count_nonzero(
                        (row.executed_command[G1_ARM_COMMAND_INDICES]
                         < self.arm_hard_limit_projector.lower_rad)
                        | (row.executed_command[G1_ARM_COMMAND_INDICES]
                           > self.arm_hard_limit_projector.upper_rad)
                    )
                    for row in self.trace
                )
            ),
            "dex3_common_override_scalar_count": int(
                np.count_nonzero(dex3_masks[:, DEX3_INDICES])
                if len(dex3_masks)
                else 0
            ),
            "dex3_hard_limits_enforced_every_commanded_frame": True,
            "dex3_hard_limit_guard_rad": self.hard_limit_guard_rad,
            "dex3_hard_lower_rad": self.dex3_hard_lower_rad.tolist(),
            "dex3_hard_upper_rad": self.dex3_hard_upper_rad.tolist(),
            "endpoint_limit_projection_scalar_count": self.endpoint_limit_projection_scalar_count,
            "frame_limit_saturation_scalar_count": self.frame_limit_saturation_scalar_count,
            "maximum_raw_to_policy_safe_arm_delta_rad": float(
                np.max(np.abs(safety[:, ARM_INDICES]), initial=0.0)
            ),
            "maximum_raw_to_policy_safe_wrist_delta_rad": float(
                np.max(np.abs(safety[:, WRIST_INDICES]), initial=0.0)
            ),
            "arm_rescue_used": False,
            "wrist_rescue_used": False,
            "object_motion_commanded": False,
            "direct_state_write_during_execution": False,
        }


__all__ = [
    "CommonArmHardLimitProjector",
    "DEX3_HARD_LIMIT_GUARD_RAD",
    "DIRECT_INTENTS",
    "G1_ARM_COMMAND_INDICES",
    "CONTACT_SEEKING_PRELOAD_MAX_RAD",
    "DIGIT_LOCAL_INDICES",
    "DirectPhysicalDex3ExecutionLayer",
    "authoritative_dex3_limits",
    "authoritative_joint_limits",
]
