#!/usr/bin/env python3
"""Policy-independent causal action-plan stitching for 30 Hz joint commands.

This module contains no task, episode, phase, object, or policy-specific logic.
It never mutates the supplied raw policy chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StitchingResult:
    stitched_execution_plan: np.ndarray
    audit: dict[str, Any]


class CommonCausalActionStitcher:
    """Turn raw/RTC model plans into a causally continuous execution plan."""

    def __init__(self, method: str, crossfade_window_frames: int = 5):
        if method not in {"naive", "rtc", "crossfade"}:
            raise ValueError(f"unsupported causal stitching method {method!r}")
        if crossfade_window_frames < 2:
            raise ValueError("crossfade window must contain at least two frames")
        self.method = method
        self.crossfade_window_frames = int(crossfade_window_frames)

    @staticmethod
    def _minimum_jerk_weight(u: np.ndarray) -> np.ndarray:
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    def stitch(
        self,
        *,
        raw_policy_chunk: np.ndarray,
        model_stitched_plan: np.ndarray,
        previous_remaining_plan: np.ndarray,
        last_commanded_action: np.ndarray | None,
        measured_qpos: np.ndarray,
    ) -> StitchingResult:
        raw = np.asarray(raw_policy_chunk, dtype=np.float64)
        model_plan = np.asarray(model_stitched_plan, dtype=np.float64)
        previous = np.asarray(previous_remaining_plan, dtype=np.float64).reshape(-1, raw.shape[1])
        measured = np.asarray(measured_qpos, dtype=np.float64)
        if raw.shape != (50, 28) or model_plan.shape != raw.shape or measured.shape != (28,):
            raise ValueError("stitcher requires raw/model (50,28) and measured (28,) arrays")
        if not np.isfinite(raw).all() or not np.isfinite(model_plan).all() or not np.isfinite(measured).all():
            raise ValueError("stitcher input contains NaN/Inf")
        if last_commanded_action is not None:
            last_commanded = np.asarray(last_commanded_action, dtype=np.float64)
            if last_commanded.shape != (28,) or not np.isfinite(last_commanded).all():
                raise ValueError("last commanded action must be finite (28,)")
        else:
            last_commanded = None

        stitched = model_plan.copy()
        weights = np.empty(0, dtype=np.float64)
        anchor_source = "NOT_APPLICABLE"
        anchored_previous = np.empty((0, 28), dtype=np.float64)
        if self.method == "crossfade" and last_commanded is not None and len(previous):
            window = min(self.crossfade_window_frames, len(previous), len(stitched))
            # Translate the old remaining plan so its first row is exactly the
            # latest measured configuration. This preserves the old plan's
            # relative motion while grounding every replan in causal feedback.
            anchored_previous = previous[:window] + (measured - previous[0])[None, :]
            u = np.arange(1, window + 1, dtype=np.float64) / float(window)
            weights = self._minimum_jerk_weight(u)
            stitched[:window] = (
                (1.0 - weights[:, None]) * anchored_previous
                + weights[:, None] * raw[:window]
            )
            anchor_source = "LATEST_MEASURED_QPOS"

        correction = stitched - raw
        first_command_delta = (
            stitched[0] - last_commanded
            if last_commanded is not None
            else stitched[0] - measured
        )
        audit = {
            "method": self.method,
            "causal": True,
            "future_observations_or_policy_calls_used": False,
            "offline_temporal_consensus_used": False,
            "raw_policy_chunk_preserved_bitwise": bool(
                np.array_equal(raw, np.asarray(raw_policy_chunk, dtype=np.float64))
            ),
            "previous_remaining_plan_rows": int(len(previous)),
            "crossfade_window_frames": (
                self.crossfade_window_frames if self.method == "crossfade" else 0
            ),
            "applied_crossfade_frames": int(len(weights)),
            "minimum_jerk_weights": weights,
            "anchor_source": anchor_source,
            "last_commanded_action": last_commanded,
            "measured_qpos_at_replan": measured,
            "maximum_abs_raw_to_stitched_correction_rad": float(
                np.max(np.abs(correction))
            ),
            "mean_abs_raw_to_stitched_correction_rad": float(
                np.mean(np.abs(correction))
            ),
            "first_stitched_command_delta_max_abs_rad": float(
                np.max(np.abs(first_command_delta))
            ),
            "phase_episode_task_specific_logic": False,
        }
        return StitchingResult(stitched_execution_plan=stitched, audit=audit)
