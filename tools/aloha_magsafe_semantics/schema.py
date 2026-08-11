"""Versioned semantic timeline data structures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .event_names import CONFIDENCE_CLASSES, PROGRESS_NAMES, REQUIRED_EVENTS, SCHEMA_NAME


@dataclass(frozen=True)
class EventRecord:
    event_name: str
    action_index: int | None
    action_time_sec: float | None
    observed_frame: int | None
    observed_time_sec: float | None
    confidence: float
    confidence_class: str
    evidence: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    score_difference: float | None = None

    def __post_init__(self) -> None:
        if self.confidence_class not in CONFIDENCE_CLASSES:
            raise ValueError(f"invalid confidence class: {self.confidence_class}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")


@dataclass
class SemanticTimeline:
    trajectory_length: int
    timestamps: np.ndarray
    events: dict[str, EventRecord]
    sample_arrays: dict[str, np.ndarray]
    detector_config_hash: str
    trajectory_hash: str
    fk_model_hash: str
    task_geometry_hash: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_name: str = SCHEMA_NAME
    schema_version: int = 1

    @property
    def start_index(self) -> int:
        return 0

    @property
    def end_index(self) -> int:
        return self.trajectory_length - 1

    def event(self, name: str) -> EventRecord:
        if name not in self.events:
            raise KeyError(f"semantic event not present: {name}")
        return self.events[name]

    def interval(self, start_name: str, end_name: str) -> tuple[int, int]:
        start = self.event(start_name).action_index
        end = self.event(end_name).action_index
        if start is None or end is None:
            raise ValueError(f"unresolved semantic interval: {start_name} -> {end_name}")
        return int(start), int(end)

    def progress(self, name: str) -> np.ndarray:
        key = name if name.endswith("_progress") else f"{name}_progress"
        if name in PROGRESS_NAMES:
            key = f"{name}_progress"
        if key not in self.sample_arrays:
            raise KeyError(f"semantic progress not present: {name}")
        return self.sample_arrays[key]

    def mandatory_complete(self) -> bool:
        return all(name in self.events and self.events[name].action_index is not None for name in REQUIRED_EVENTS)

    def high_medium_complete(self) -> bool:
        return self.mandatory_complete() and all(
            self.events[name].confidence_class in ("HIGH", "MEDIUM") for name in REQUIRED_EVENTS
        )

    def to_dict(self, include_sample_arrays: bool = False) -> dict[str, Any]:
        result = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "trajectory_length": self.trajectory_length,
            "time_range_sec": [float(self.timestamps[0]), float(self.timestamps[-1])],
            "source_type": self.source_type,
            "events": [asdict(self.events[name]) for name in REQUIRED_EVENTS if name in self.events]
            + [asdict(record) for name, record in self.events.items() if name not in REQUIRED_EVENTS],
            "sample_array_keys": sorted(self.sample_arrays),
            "provenance": {
                "detector_config_hash": self.detector_config_hash,
                "trajectory_hash": self.trajectory_hash,
                "FK_model_hash": self.fk_model_hash,
                "task_geometry_hash": self.task_geometry_hash,
                "reference_timeline_used_for_detection": False,
            },
            "validation": {
                "mandatory_complete": self.mandatory_complete(),
                "high_medium_complete": self.high_medium_complete(),
            },
            "metadata": self.metadata,
        }
        if include_sample_arrays:
            result["sample_arrays"] = {key: value.tolist() for key, value in self.sample_arrays.items()}
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any], sample_arrays: dict[str, np.ndarray] | None = None) -> "SemanticTimeline":
        events = {row["event_name"]: EventRecord(**row) for row in payload["events"]}
        provenance = payload["provenance"]
        if sample_arrays is None:
            sample_arrays = {key: np.asarray(value) for key, value in payload.get("sample_arrays", {}).items()}
        length = int(payload["trajectory_length"])
        time_range = payload.get("time_range_sec", [0.0, float(length - 1)])
        timestamps = np.linspace(float(time_range[0]), float(time_range[1]), length)
        return cls(
            trajectory_length=length,
            timestamps=timestamps,
            events=events,
            sample_arrays=sample_arrays,
            detector_config_hash=provenance["detector_config_hash"],
            trajectory_hash=provenance["trajectory_hash"],
            fk_model_hash=provenance["FK_model_hash"],
            task_geometry_hash=provenance["task_geometry_hash"],
            source_type=payload["source_type"],
            metadata=payload.get("metadata", {}),
            schema_name=payload.get("schema_name", SCHEMA_NAME),
            schema_version=int(payload.get("schema_version", 1)),
        )

