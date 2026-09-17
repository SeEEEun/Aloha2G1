"""One pooled, smoothed, hysteretic and debounced doll-handoff event detector."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from .common import EVENT_AUDIT, SIDES, atomic_csv, atomic_json, scalar_stats
from .source import SourceEpisode, SourceRepository


ANOMALY_NAMES = (
    "NO_LEFT_GRASP",
    "NO_RIGHT_GRASP",
    "LEFT_RELEASE_BEFORE_RIGHT_GRASP",
    "NO_LEFT_RELEASE",
    "NO_RIGHT_FINAL_RELEASE",
    "AMBIGUOUS_HANDOFF",
)


@dataclass(frozen=True)
class EpisodeEvents:
    episode_index: int
    source_name: str
    frames: dict[str, int | None]
    transitions: dict[str, list[dict[str, Any]]]
    smoothed_gripper: dict[str, np.ndarray]
    binary_labels: dict[str, np.ndarray]
    semantic_labels: dict[str, np.ndarray]
    ownership_labels: np.ndarray
    inter_hand_distance_m: np.ndarray
    handoff_window: tuple[int, int]
    anomalies: tuple[str, ...]
    source_semantic_valid: bool

    def ownership_state(self, frame: int) -> str:
        return str(self.ownership_labels[int(frame)])

    def handoff_label(self, frame: int) -> str:
        approach = int(self.frames.get("HANDOFF_APPROACH") or 0)
        right_grasp = self.frames.get("RIGHT_GRASP")
        left_release = self.frames.get("LEFT_RELEASE")
        final_release = self.frames.get("RIGHT_FINAL_RELEASE")
        if right_grasp is None or left_release is None:
            return "AMBIGUOUS_HANDOFF"
        if approach <= frame < right_grasp:
            return "HANDOFF_APPROACH"
        if frame == right_grasp:
            return "RIGHT_GRASP"
        if right_grasp < frame < left_release:
            return "DUAL_HOLD"
        if frame == left_release:
            return "LEFT_RELEASE"
        if final_release is None or left_release < frame < final_release:
            return "RIGHT_HOLD"
        return ""


def _two_means(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centers = np.quantile(values, [0.2, 0.8]).astype(np.float64)
    for _ in range(100):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = np.asarray([np.mean(values[labels == index]) for index in range(2)])
        if np.allclose(updated, centers, rtol=0.0, atol=1e-12):
            centers = updated
            break
        centers = updated
    centers.sort()
    return float(centers[0]), float(centers[1])


class EventAuditor:
    def __init__(
        self,
        common: Mapping[str, Any],
        scene: Mapping[str, Any],
        sources: SourceRepository,
        output: Path = EVENT_AUDIT,
    ):
        self.common = common
        self.scene = scene
        self.sources = sources
        self.output = output
        self.detector_config: dict[str, Any] = {}
        self.events: dict[int, EpisodeEvents] = {}
        self.fk_cache: dict[int, dict[str, Any]] = {}

    def derive_config(self) -> dict[str, Any]:
        cfg = self.common["event_detector"]
        pooled = {side: [] for side in SIDES}
        for episode_index in self.sources.valid_episode_indices():
            episode = self.sources.episode(episode_index)
            pooled["left"].append(episode.action[:, 6])
            pooled["right"].append(episode.action[:, 13])
        sides: dict[str, Any] = {}
        for side in SIDES:
            values = np.concatenate(pooled[side])
            closed, opened = _two_means(values)
            gap = opened - closed
            close_threshold = closed + float(cfg["closed_cluster_fraction"]) * gap
            open_threshold = closed + float(cfg["open_cluster_fraction"]) * gap
            if not closed < close_threshold < open_threshold < opened:
                raise RuntimeError(f"invalid pooled {side} hysteresis thresholds")
            sides[side] = {
                "pooled_sample_count": int(len(values)),
                "pooled_min_m": float(np.min(values)),
                "pooled_max_m": float(np.max(values)),
                "closed_cluster_center_m": closed,
                "open_cluster_center_m": opened,
                "close_enter_threshold_m": close_threshold,
                "open_enter_threshold_m": open_threshold,
                "increasing_is_open": True,
            }
        self.detector_config = {
            "schema_version": "doll_handoff_common_event_detector_v1",
            "algorithm": "command-aperture median smoothing + pooled two-cluster hysteresis + debounce + one pooled command-to-measured-state lag",
            "signal_key": cfg["signal_key"],
            "event_frame_reference": "observation.state / measured-FK time",
            "pooled_command_to_measured_state_lag_frames": int(
                self.sources.manifest[
                    "pooled_action_to_measured_state_gripper_lag_audit"
                ]["best_state_lag_frames"]
            ),
            "smoothing_window_frames": int(cfg["smoothing_window_frames"]),
            "debounce_frames": int(cfg["debounce_frames"]),
            "stable_hold_frames": int(cfg["stable_hold_frames"]),
            "pregrasp_duration_sec": float(cfg["pregrasp_duration_sec"]),
            "threshold_scope": cfg["threshold_scope"],
            "episode_specific_thresholds": False,
            "manual_event_frames": False,
            "sides": sides,
        }
        return self.detector_config

    def _transitions(self, signal: np.ndarray, side: str) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
        cfg = self.detector_config
        window = int(cfg["smoothing_window_frames"])
        smoothed = median_filter(np.asarray(signal, dtype=np.float64), size=window, mode="nearest")
        side_cfg = cfg["sides"][side]
        close_threshold = float(side_cfg["close_enter_threshold_m"])
        open_threshold = float(side_cfg["open_enter_threshold_m"])
        debounce = int(cfg["debounce_frames"])
        midpoint = 0.5 * (close_threshold + open_threshold)
        state = "CLOSED" if smoothed[0] <= midpoint else "OPEN"
        labels = np.empty(len(smoothed), dtype="U8")
        labels[:] = state
        transitions: list[dict[str, Any]] = []
        index = 1
        while index < len(smoothed):
            if state == "CLOSED":
                changed = (
                    index + debounce <= len(smoothed)
                    and np.all(smoothed[index : index + debounce] >= open_threshold)
                )
                next_state = "OPEN"
            else:
                changed = (
                    index + debounce <= len(smoothed)
                    and np.all(smoothed[index : index + debounce] <= close_threshold)
                )
                next_state = "CLOSED"
            if changed:
                transitions.append(
                    {
                        "from": state,
                        "to": next_state,
                        "onset_frame": int(index),
                        "confirmed_frame": int(index + debounce - 1),
                        "smoothed_value_at_onset_m": float(smoothed[index]),
                    }
                )
                labels[index:] = next_state
                state = next_state
                index += debounce
            else:
                index += 1
        lag = int(cfg["pooled_command_to_measured_state_lag_frames"])
        for row in transitions:
            row["signal_key"] = cfg["signal_key"]
            row["aligned_onset_frame"] = min(
                len(smoothed) - 1, int(row["onset_frame"]) + lag
            )
            row["aligned_confirmed_frame"] = min(
                len(smoothed) - 1, int(row["confirmed_frame"]) + lag
            )
        if lag > 0:
            labels = np.concatenate(
                (np.full(lag, labels[0], dtype=labels.dtype), labels[:-lag])
            )
        return smoothed, transitions, labels

    @staticmethod
    def _event_frame(
        transitions: list[dict[str, Any]],
        index: int,
        expected_from: str,
        expected_to: str,
        field: str,
    ) -> int | None:
        if index >= len(transitions):
            return None
        row = transitions[index]
        if row["from"] != expected_from or row["to"] != expected_to:
            return None
        return int(row[field])

    def detect_episode(self, episode: SourceEpisode, fk: dict[str, Any]) -> EpisodeEvents:
        if not self.detector_config:
            self.derive_config()
        smoothed: dict[str, np.ndarray] = {}
        transitions: dict[str, list[dict[str, Any]]] = {}
        binary: dict[str, np.ndarray] = {}
        for side, channel in (("left", 6), ("right", 13)):
            smoothed[side], transitions[side], binary[side] = self._transitions(
                episode.action[:, channel], side
            )
        frames: dict[str, int | None] = {
            "LEFT_CLOSE_ONSET": self._event_frame(
                transitions["left"], 1, "OPEN", "CLOSED", "aligned_onset_frame"
            ),
            "LEFT_GRASP": self._event_frame(
                transitions["left"], 1, "OPEN", "CLOSED", "aligned_confirmed_frame"
            ),
            "RIGHT_CLOSE_ONSET": self._event_frame(
                transitions["right"], 1, "OPEN", "CLOSED", "aligned_onset_frame"
            ),
            "RIGHT_GRASP": self._event_frame(
                transitions["right"], 1, "OPEN", "CLOSED", "aligned_confirmed_frame"
            ),
            "LEFT_RELEASE": self._event_frame(
                transitions["left"], 2, "CLOSED", "OPEN", "aligned_onset_frame"
            ),
            "RIGHT_FINAL_RELEASE": self._event_frame(
                transitions["right"], 2, "CLOSED", "OPEN", "aligned_onset_frame"
            ),
        }
        hold_frames = int(self.detector_config["stable_hold_frames"])
        frames["LEFT_STABLE_HOLD"] = (
            min(len(episode.action) - 1, int(frames["LEFT_GRASP"]) + hold_frames)
            if frames["LEFT_GRASP"] is not None
            else None
        )
        frames["RIGHT_STABLE_HOLD"] = (
            min(len(episode.action) - 1, int(frames["RIGHT_GRASP"]) + hold_frames)
            if frames["RIGHT_GRASP"] is not None
            else None
        )
        frames["RIGHT_HOLD"] = frames["RIGHT_STABLE_HOLD"]
        frames["RIGHT_ACQUIRE"] = frames["RIGHT_GRASP"]

        left_tcp = np.asarray(fk["left_tcp_position_world"])
        right_tcp = np.asarray(fk["right_tcp_position_world"])
        inter_hand = np.linalg.norm(right_tcp - left_tcp, axis=1)
        minimum_index = int(np.argmin(inter_hand))
        frames["MIN_INTER_HAND_DISTANCE"] = minimum_index

        initial_right_open = self._event_frame(
            transitions["right"], 0, "CLOSED", "OPEN", "aligned_confirmed_frame"
        )
        right_grasp = frames["RIGHT_GRASP"]
        if initial_right_open is not None and right_grasp is not None and initial_right_open < right_grasp:
            segment = inter_hand[initial_right_open : right_grasp + 1]
            approach = int(initial_right_open + np.argmax(segment))
        elif right_grasp is not None:
            approach = max(0, int(right_grasp) - int(round(0.3 * episode.fps)))
        else:
            approach = 0
        frames["HANDOFF_APPROACH"] = approach

        left_release = frames["LEFT_RELEASE"]
        if left_release is not None:
            limit = min(
                len(inter_hand),
                int(left_release)
                + int(round(float(self.common["event_detector"]["handoff_post_release_search_sec"]) * episode.fps))
                + 1,
            )
            base = inter_hand[int(left_release)]
            candidates = np.flatnonzero(inter_hand[int(left_release) : limit] >= base + 0.05)
            end = int(left_release) + int(candidates[0]) if len(candidates) else limit - 1
        else:
            end = min(len(inter_hand) - 1, approach)

        anomalies: list[str] = []
        if frames["LEFT_GRASP"] is None:
            anomalies.append("NO_LEFT_GRASP")
        if frames["RIGHT_GRASP"] is None:
            anomalies.append("NO_RIGHT_GRASP")
        if frames["LEFT_RELEASE"] is None:
            anomalies.append("NO_LEFT_RELEASE")
        if frames["RIGHT_FINAL_RELEASE"] is None:
            anomalies.append("NO_RIGHT_FINAL_RELEASE")
        if (
            frames["RIGHT_GRASP"] is not None
            and frames["LEFT_RELEASE"] is not None
            and int(frames["LEFT_RELEASE"]) <= int(frames["RIGHT_GRASP"])
        ):
            anomalies.append("LEFT_RELEASE_BEFORE_RIGHT_GRASP")
        expected_sequence = [("CLOSED", "OPEN"), ("OPEN", "CLOSED"), ("CLOSED", "OPEN"), ("OPEN", "CLOSED")]
        transition_valid = all(
            len(transitions[side]) == 4
            and [(row["from"], row["to"]) for row in transitions[side]] == expected_sequence
            for side in SIDES
        )
        if (
            not transition_valid
            or inter_hand[minimum_index] > 3.0 * float(self.scene["doll"]["diameter_m"])
        ):
            anomalies.append("AMBIGUOUS_HANDOFF")

        semantic: dict[str, np.ndarray] = {}
        pregrasp_frames = int(round(float(self.detector_config["pregrasp_duration_sec"]) * episode.fps))
        for side in SIDES:
            labels = np.full(len(episode.action), "OPEN", dtype="U12")
            close_onset = frames[f"{side.upper()}_CLOSE_ONSET"]
            grasp = frames[f"{side.upper()}_GRASP"]
            stable = frames[f"{side.upper()}_STABLE_HOLD"]
            release = frames["LEFT_RELEASE" if side == "left" else "RIGHT_FINAL_RELEASE"]
            if close_onset is not None:
                pre_start = max(0, int(close_onset) - pregrasp_frames)
                labels[pre_start : int(close_onset)] = "PRESHAPE"
                grasp_end = int(stable if stable is not None else grasp or close_onset)
                labels[int(close_onset) : grasp_end] = "GRASP"
                if stable is not None and release is not None:
                    labels[int(stable) : int(release)] = "HOLD"
                if release is not None:
                    release_confirm = self._event_frame(
                        transitions[side], 2, "CLOSED", "OPEN", "aligned_confirmed_frame"
                    )
                    labels[int(release) : int(release_confirm or release) + 1] = "RELEASE"
            semantic[side] = labels

        ownership = np.full(len(episode.action), "NO_OWNER", dtype="U20")
        left_grasp = frames["LEFT_GRASP"]
        right_grasp = frames["RIGHT_GRASP"]
        left_release = frames["LEFT_RELEASE"]
        final_release = frames["RIGHT_FINAL_RELEASE"]
        if left_grasp is not None:
            ownership[int(left_grasp) :] = "LEFT_OWNED"
        approach_start = max(
            int(left_grasp or 0), int(frames["HANDOFF_APPROACH"] or 0)
        )
        if right_grasp is not None and approach_start < int(right_grasp):
            ownership[approach_start : int(right_grasp)] = "HANDOFF_APPROACH"
        if right_grasp is not None:
            dual_end = int(left_release) if left_release is not None else len(ownership)
            ownership[int(right_grasp) : dual_end] = "DUAL_CONTACT"
        if left_release is not None:
            right_owned_end = min(len(ownership), max(int(left_release), int(end)) + 1)
            ownership[int(left_release) : right_owned_end] = "RIGHT_OWNED"
            release_start = int(final_release) if final_release is not None else len(ownership)
            ownership[right_owned_end:release_start] = "RIGHT_TRANSPORT"
        if final_release is not None:
            ownership[int(final_release) :] = "RELEASED"

        return EpisodeEvents(
            episode_index=episode.record.episode_index,
            source_name=episode.record.source_name,
            frames=frames,
            transitions=transitions,
            smoothed_gripper=smoothed,
            binary_labels=binary,
            semantic_labels=semantic,
            ownership_labels=ownership,
            inter_hand_distance_m=inter_hand,
            handoff_window=(approach, end),
            anomalies=tuple(anomalies),
            source_semantic_valid=not anomalies,
        )

    def _plot(self, episode: SourceEpisode, event: EpisodeEvents) -> Path:
        plot_dir = self.output / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        path = plot_dir / f"episode_{episode.record.episode_index:03d}_{episode.record.source_name}.png"
        time = episode.timestamps
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for side, color in (("left", "tab:green"), ("right", "tab:blue")):
            channel = 6 if side == "left" else 13
            axes[0].plot(time, episode.state[:, channel], color=color, alpha=0.22, lw=0.7, ls=":", label=f"{side} measured")
            axes[0].plot(time, episode.action[:, channel], color=color, alpha=0.42, lw=0.8, label=f"{side} command")
            axes[0].plot(time, event.smoothed_gripper[side], color=color, lw=1.2, label=f"{side} smoothed")
            side_cfg = self.detector_config["sides"][side]
            axes[0].axhline(side_cfg["close_enter_threshold_m"], color=color, ls=":", lw=0.8)
            axes[0].axhline(side_cfg["open_enter_threshold_m"], color=color, ls="--", lw=0.8)
        axes[0].set_ylabel("gripper [m]")
        axes[0].legend(ncol=2, fontsize=8)
        axes[1].plot(time, event.inter_hand_distance_m, color="tab:purple", lw=1.1)
        axes[1].set_ylabel("TCP distance [m]")
        phase_codes = {"OPEN": 0, "PRESHAPE": 1, "GRASP": 2, "HOLD": 3, "RELEASE": 4}
        for offset, side in enumerate(SIDES):
            values = np.asarray([phase_codes[value] + 0.08 * offset for value in event.semantic_labels[side]])
            axes[2].plot(time, values, label=side, lw=1.1)
        axes[2].set_yticks(list(phase_codes.values()), list(phase_codes))
        axes[2].set_ylabel("semantic phase")
        axes[2].set_xlabel("source time [s]")
        axes[2].legend(fontsize=8)
        event_colors = {
            "LEFT_GRASP": "green",
            "RIGHT_GRASP": "blue",
            "LEFT_RELEASE": "orange",
            "RIGHT_FINAL_RELEASE": "red",
        }
        for name, color in event_colors.items():
            frame = event.frames[name]
            if frame is None:
                continue
            for axis in axes:
                axis.axvline(time[int(frame)], color=color, ls="--", lw=0.8, alpha=0.8)
            axes[0].text(time[int(frame)], axes[0].get_ylim()[1], name, rotation=90, va="top", fontsize=7, color=color)
        fig.suptitle(
            f"{episode.record.source_name} | event detector | anomalies={','.join(event.anomalies) or 'NONE'}"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        self.derive_config()
        atomic_json(self.output / "detector_config.json", self.detector_config)
        rows: list[dict[str, Any]] = []
        all_events: dict[str, Any] = {}
        for episode_index in self.sources.valid_episode_indices():
            episode = self.sources.episode(episode_index)
            fk = self.sources.aloha.fk(episode.state)
            self.fk_cache[episode_index] = fk
            event = self.detect_episode(episode, fk)
            self.events[episode_index] = event
            plot = self._plot(episode, event)
            dual_hold = (
                int(event.frames["LEFT_RELEASE"]) - int(event.frames["RIGHT_GRASP"])
                if event.frames["LEFT_RELEASE"] is not None and event.frames["RIGHT_GRASP"] is not None
                else 0
            )
            row = {
                "episode_index": episode_index,
                "stable_episode_id": episode.record.stable_episode_id,
                "source_name": episode.record.source_name,
                **{name.lower() + "_frame": value for name, value in event.frames.items()},
                "minimum_source_inter_hand_distance_m": float(np.min(event.inter_hand_distance_m)),
                "minimum_source_inter_hand_distance_frame": int(np.argmin(event.inter_hand_distance_m)),
                "minimum_source_inter_hand_distance_time_sec": float(
                    episode.timestamps[int(np.argmin(event.inter_hand_distance_m))]
                ),
                "dual_hold_frames": dual_hold,
                "dual_hold_sec": dual_hold / episode.fps,
                "right_grasp_before_left_release": bool(dual_hold > 0),
                "source_semantic_valid": event.source_semantic_valid,
                "anomalies": ";".join(event.anomalies),
                "plot": str(plot),
            }
            rows.append(row)
            all_events[str(episode_index)] = {
                "source_name": episode.record.source_name,
                "frames": event.frames,
                "transitions": event.transitions,
                "handoff_window": event.handoff_window,
                "ownership_sequence": list(dict.fromkeys(event.ownership_labels.tolist())),
                "ownership_claim_scope": (
                    "SOURCE_DERIVED_INTERACTION_SEMANTICS_NOT_MEASURED_OBJECT_CONTACT"
                ),
                "anomalies": event.anomalies,
                "source_semantic_valid": event.source_semantic_valid,
            }
        atomic_csv(self.output / "per_episode_events.csv", rows)
        atomic_json(self.output / "events.json", all_events)
        anomaly_counts = {
            name: sum(name in event.anomalies for event in self.events.values())
            for name in ANOMALY_NAMES
        }
        aggregate = {
            "schema_version": "doll_handoff_event_aggregate_v1",
            "episode_count": len(rows),
            "source_semantic_valid_count": sum(event.source_semantic_valid for event in self.events.values()),
            "source_semantic_invalid_count": sum(not event.source_semantic_valid for event in self.events.values()),
            "anomaly_counts": anomaly_counts,
            "right_grasp_before_left_release_count": sum(
                bool(row["right_grasp_before_left_release"]) for row in rows
            ),
            "dual_hold_frames": scalar_stats(np.asarray([row["dual_hold_frames"] for row in rows])),
            "dual_hold_sec": scalar_stats(np.asarray([row["dual_hold_sec"] for row in rows])),
            "minimum_source_inter_hand_distance_m": scalar_stats(
                np.asarray([row["minimum_source_inter_hand_distance_m"] for row in rows])
            ),
            "manual_event_frames": 0,
            "episode_specific_thresholds": 0,
        }
        atomic_json(self.output / "aggregate_event_statistics.json", aggregate)
        return aggregate


__all__ = ["ANOMALY_NAMES", "EpisodeEvents", "EventAuditor"]
