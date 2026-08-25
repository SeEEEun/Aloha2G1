#!/usr/bin/env python3
"""Disabled advisory Dex3 contact state-machine skeleton; it has no command interface."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any


class ContactState(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CONTACT_DETECTED = "CONTACT_DETECTED"
    HOLD = "HOLD"
    RELEASE = "RELEASE"


class ContactFeedbackGraspAdapter:
    """Compute advisory states only after a locally measured real calibration."""

    REQUIRED_STATUS = "REAL_DEX3_CONTACT_CALIBRATION_COMPLETE"

    def __init__(self, calibration_file: str | Path):
        self.path = Path(calibration_file).resolve()
        self.calibration: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        self.state = ContactState.OPEN
        self.active = False

    def activate(self) -> None:
        value = self.calibration
        checks = {
            "schema": value.get("schema_version") == "dex3_contact_calibration_v1",
            "real_status": value.get("status") == self.REQUIRED_STATUS,
            "force_units_unassigned": value.get("force_units") == "NOT_ASSIGNED",
            "real_source": value.get("source_recording") not in (None, "", "NOT_RECORDED"),
            "left_annotation": value.get("left_sensor_index_annotation") not in (None, "", "NOT_ANNOTATED"),
            "right_annotation": value.get("right_sensor_index_annotation") not in (None, "", "NOT_ANNOTATED"),
            "contact_rule": isinstance(value.get("contact_delta_detection_rule"), dict),
            "release_rule": isinstance(value.get("release_delta_detection_rule"), dict),
            "explicit_authorization": value.get("execution_adapter_activation_allowed") is True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"ContactFeedbackGraspAdapter remains DISABLED: {checks}")
        self.active = True

    def observe(self, raw_press_sensor_state: Any) -> ContactState:
        if not self.active:
            raise RuntimeError("ContactFeedbackGraspAdapter is DISABLED until real calibration activation")
        # State transition logic is intentionally not implemented before real
        # sensor-index annotation and delta calibration exist.
        raise NotImplementedError("calibration-specific advisory transition logic is not implemented")


__all__ = ["ContactFeedbackGraspAdapter", "ContactState"]
