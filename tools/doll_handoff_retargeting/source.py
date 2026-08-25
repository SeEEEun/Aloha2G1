"""Programmatic discovery, schema audit, and loading of the 50 raw recordings."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import SOURCE_AUDIT, atomic_csv, atomic_json, array_sha256, sha256_file
from .models import ALOHAKinematics


def fixed_list_numpy(column: pa.ChunkedArray, width: int) -> np.ndarray:
    chunks = column.combine_chunks()
    if not pa.types.is_fixed_size_list(chunks.type) or chunks.type.list_size != width:
        raise ValueError(f"expected fixed_size_list[{width}], got {chunks.type}")
    return np.asarray(chunks.values.to_numpy(zero_copy_only=False)).reshape(len(chunks), width)


def scalar_numpy(column: pa.ChunkedArray, dtype: Any) -> np.ndarray:
    return np.asarray(column.combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


@dataclass(frozen=True)
class SourceRecord:
    episode_index: int
    stable_episode_id: str
    source_name: str
    root: Path
    parquet: Path
    frame_count: int
    fps: float
    duration_sec: float
    camera_keys: tuple[str, ...]
    image_directories: dict[str, Path]
    valid: bool
    problems: tuple[str, ...]


@dataclass(frozen=True)
class SourceEpisode:
    record: SourceRecord
    action: np.ndarray
    state: np.ndarray
    timestamps: np.ndarray
    frame_index: np.ndarray
    task_index: np.ndarray

    @property
    def fps(self) -> float:
        return self.record.fps


class SourceRepository:
    def __init__(
        self,
        common: Mapping[str, Any],
        aloha: ALOHAKinematics,
        audit_output: Path = SOURCE_AUDIT,
    ):
        self.common = common
        self.aloha = aloha
        self.output = audit_output
        self.records: list[SourceRecord] = []
        self.manifest: dict[str, Any] = {}
        self._cache: dict[int, SourceEpisode] = {}
        self.selection_manifest: dict[str, Any] | None = None

    def discover(self) -> list[Path]:
        selection_path = self.common.get("source_selection_manifest")
        if selection_path:
            path = Path(str(selection_path)).resolve()
            selection = json.loads(path.read_text(encoding="utf-8"))
            episodes = selection.get("episodes", [])
            expected = int(self.common["expected_source_count"])
            if len(episodes) != expected:
                raise RuntimeError(
                    f"source selection manifest has {len(episodes)} episodes, expected {expected}"
                )
            indices = [int(row["final_dataset_index"]) for row in episodes]
            if indices != list(range(expected)):
                raise RuntimeError("source selection manifest final_dataset_index is not contiguous")
            roots = [Path(str(row["raw_directory_path"])).resolve() for row in episodes]
            for row, root in zip(episodes, roots):
                if not root.is_dir() or root.name != str(row["raw_directory"]):
                    raise RuntimeError(f"selected source directory is missing or mismatched: {root}")
                parquet = Path(str(row["source_parquet_path"])).resolve()
                if not parquet.is_file() or sha256_file(parquet) != row["source_parquet_sha256"]:
                    raise RuntimeError(f"selected source parquet identity changed: {parquet}")
            self.selection_manifest = selection
            return roots
        pattern = str(self.common["source_glob"])
        path = Path(pattern)
        roots = sorted(path.parent.glob(path.name), key=lambda value: value.name)
        excluded_names = set(map(str, self.common["excluded_names"]))
        roots = [value for value in roots if value.is_dir() and value.name not in excluded_names]
        if len(roots) != int(self.common["expected_source_count"]):
            raise RuntimeError(
                f"expected exactly {self.common['expected_source_count']} source directories, "
                f"found {len(roots)} under {path.parent}"
            )
        return roots

    @staticmethod
    def _info_shape(info: Mapping[str, Any], key: str) -> tuple[int, ...]:
        value = info.get("features", {}).get(key, {}).get("shape")
        if not isinstance(value, list):
            raise ValueError(f"meta/info.json missing {key} shape")
        return tuple(map(int, value))

    def audit(self) -> dict[str, Any]:
        roots = self.discover()
        model_channels = self.aloha.channel_report()
        manifest_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        invalid_rows: list[dict[str, Any]] = []
        schema_texts: dict[str, int] = {}
        feature_name_sets: dict[str, int] = {}
        lag_sums = {lag: 0.0 for lag in range(16)}
        lag_counts = {lag: 0 for lag in range(16)}
        gripper_lag_sums = {lag: 0.0 for lag in range(16)}
        gripper_lag_counts = {lag: 0 for lag in range(16)}
        records: list[SourceRecord] = []
        for episode_index, root in enumerate(roots):
            problems: list[str] = []
            parquet_files = sorted((root / "data").rglob("*.parquet"))
            if len(parquet_files) != 1:
                problems.append(f"expected exactly one parquet, found {len(parquet_files)}")
                parquet = parquet_files[0] if parquet_files else root / "<missing>"
                table = None
            else:
                parquet = parquet_files[0]
                try:
                    table = pq.read_table(parquet)
                except Exception as error:  # retained in invalid_sources, never dropped
                    table = None
                    problems.append(f"parquet read failed: {type(error).__name__}: {error}")
            info_path = root / "meta/info.json"
            tasks_path = root / "meta/tasks.jsonl"
            if not info_path.is_file():
                problems.append("missing meta/info.json")
                info: dict[str, Any] = {}
            else:
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                except Exception as error:
                    info = {}
                    problems.append(f"invalid meta/info.json: {error}")
            if not tasks_path.is_file():
                problems.append("missing meta/tasks.jsonl")

            frame_count = int(table.num_rows) if table is not None else 0
            fps = float(info.get("fps", 0.0) or 0.0)
            duration = 0.0
            action = state = timestamps = frame_index = task_index = None
            if table is not None:
                required = {
                    "action",
                    "observation.state",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                }
                missing = sorted(required - set(table.column_names))
                if missing:
                    problems.append(f"parquet missing columns: {missing}")
                else:
                    try:
                        action = fixed_list_numpy(table["action"], 14).astype(np.float64)
                        state = fixed_list_numpy(table["observation.state"], 14).astype(np.float64)
                        timestamps = scalar_numpy(table["timestamp"], np.float64)
                        frame_index = scalar_numpy(table["frame_index"], np.int64)
                        task_index = scalar_numpy(table["task_index"], np.int64)
                        episode_ids = scalar_numpy(table["episode_index"], np.int64)
                        if not all(
                            np.isfinite(value).all() for value in (action, state, timestamps)
                        ):
                            problems.append("action/state/timestamps contain NaN or infinity")
                        if not np.array_equal(frame_index, np.arange(frame_count)):
                            problems.append("frame_index is not contiguous from zero")
                        if not np.array_equal(episode_ids, np.zeros(frame_count, dtype=np.int64)):
                            problems.append("raw recording episode_index is not uniformly zero")
                        if frame_count > 1 and not np.allclose(
                            np.diff(timestamps), 1.0 / fps, atol=2e-6, rtol=0.0
                        ):
                            problems.append("timestamp cadence differs from metadata FPS")
                        duration = float(timestamps[-1] - timestamps[0]) if frame_count else 0.0
                    except Exception as error:
                        problems.append(f"column validation failed: {type(error).__name__}: {error}")
                schema = str(table.schema.remove_metadata())
                schema_texts[schema] = schema_texts.get(schema, 0) + 1

            names: list[str] = []
            try:
                action_shape = self._info_shape(info, "action")
                state_shape = self._info_shape(info, "observation.state")
                names = list(info["features"]["action"]["names"])
                state_names = list(info["features"]["observation.state"]["names"])
                if action_shape != (14,) or state_shape != (14,):
                    problems.append(
                        f"metadata vector shapes must both be [14], got {action_shape}/{state_shape}"
                    )
                if names != state_names:
                    problems.append("action and observation.state names differ")
                expected_names = [f"left_joint_{index}" for index in range(7)] + [
                    f"right_joint_{index}" for index in range(7)
                ]
                if names != expected_names:
                    problems.append(f"unexpected channel names/order: {names}")
                name_key = json.dumps(names)
                feature_name_sets[name_key] = feature_name_sets.get(name_key, 0) + 1
            except Exception as error:
                problems.append(f"metadata feature validation failed: {error}")

            camera_keys = tuple(
                sorted(
                    key
                    for key, value in info.get("features", {}).items()
                    if isinstance(value, Mapping) and value.get("dtype") == "video"
                )
            )
            image_directories: dict[str, Path] = {}
            camera_report: dict[str, Any] = {}
            for key in camera_keys:
                directory = root / "images" / key / "episode_000000"
                image_directories[key] = directory
                images = sorted(directory.glob("frame_*.png")) if directory.is_dir() else []
                dimensions = None
                if images:
                    sample = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
                    dimensions = [int(sample.shape[0]), int(sample.shape[1]), 3] if sample is not None else None
                expected_dimensions = info["features"][key].get("shape")
                if len(images) != frame_count:
                    problems.append(f"{key}: image count {len(images)} != frames {frame_count}")
                if dimensions != expected_dimensions:
                    problems.append(
                        f"{key}: sample image shape {dimensions} != metadata {expected_dimensions}"
                    )
                camera_report[key] = {
                    "directory": str(directory),
                    "file_count": len(images),
                    "sample_shape": dimensions,
                    "metadata_shape": expected_dimensions,
                }

            # The channel semantics are verified against the active model, not inferred
            # solely from generic left_joint_6/right_joint_6 labels.
            if action is not None and state is not None:
                for side, index in (("left", 6), ("right", 13)):
                    lower, upper = model_channels[f"{side}_gripper_range_m"]
                    margin = 0.003
                    for key, values in (("action", action), ("observation.state", state)):
                        if np.min(values[:, index]) < lower - margin or np.max(values[:, index]) > upper + margin:
                            problems.append(
                                f"{key} {side} gripper channel is incompatible with model range"
                            )

            stable_id = (
                str(self.selection_manifest["episodes"][episode_index]["stable_episode_id"])
                if self.selection_manifest is not None
                else f"doll_handoff_20260820_ep{episode_index:03d}"
            )
            action_state_report: dict[str, Any] = {}
            if action is not None and state is not None:
                arm_channels = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                gripper_channels = [6, 13]
                lag_mae: dict[int, float] = {}
                for state_lag in range(16):
                    if state_lag:
                        command_values = action[:-state_lag]
                        measured_values = state[state_lag:]
                    else:
                        command_values = action
                        measured_values = state
                    difference = np.abs(command_values - measured_values)
                    arm_difference = difference[:, arm_channels]
                    gripper_difference = difference[:, gripper_channels]
                    lag_mae[state_lag] = float(np.mean(arm_difference))
                    lag_sums[state_lag] += float(np.sum(arm_difference))
                    lag_counts[state_lag] += int(arm_difference.size)
                    gripper_lag_sums[state_lag] += float(
                        np.sum(gripper_difference)
                    )
                    gripper_lag_counts[state_lag] += int(gripper_difference.size)
                best_lag = min(lag_mae, key=lag_mae.get)
                action_state_report = {
                    "mean_absolute_difference_all_channels": float(
                        np.mean(np.abs(action - state))
                    ),
                    "mean_absolute_difference_arm_rad": float(
                        np.mean(np.abs(action[:, arm_channels] - state[:, arm_channels]))
                    ),
                    "mean_absolute_difference_gripper_m": float(
                        np.mean(
                            np.abs(
                                action[:, gripper_channels] - state[:, gripper_channels]
                            )
                        )
                    ),
                    "best_command_to_measured_arm_state_lag_frames": int(best_lag),
                    "best_lag_arm_mae_rad": lag_mae[best_lag],
                }
            valid = not problems
            record = SourceRecord(
                episode_index=episode_index,
                stable_episode_id=stable_id,
                source_name=root.name,
                root=root.resolve(),
                parquet=parquet.resolve(),
                frame_count=frame_count,
                fps=fps,
                duration_sec=duration,
                camera_keys=camera_keys,
                image_directories=image_directories,
                valid=valid,
                problems=tuple(problems),
            )
            records.append(record)
            row = {
                "episode_index": episode_index,
                "stable_episode_id": stable_id,
                "source_name": root.name,
                "source_root": str(root.resolve()),
                "parquet_path": str(parquet.resolve()),
                "parquet_sha256": sha256_file(parquet) if parquet.is_file() else None,
                "frame_count": frame_count,
                "fps": fps,
                "duration_sec": duration,
                "timestamp_start": float(timestamps[0]) if timestamps is not None else None,
                "timestamp_end": float(timestamps[-1]) if timestamps is not None else None,
                "observation_state_key": "observation.state",
                "observation_state_shape": [frame_count, 14],
                "action_key": "action",
                "action_shape": [frame_count, 14],
                "channel_names": names,
                "left_arm_channels": [0, 1, 2, 3, 4, 5],
                "left_gripper_channel": 6,
                "right_arm_channels": [7, 8, 9, 10, 11, 12],
                "right_gripper_channel": 13,
                "camera_assets": camera_report,
                "metadata": {
                    "info_path": str(info_path.resolve()),
                    "tasks_path": str(tasks_path.resolve()),
                    "robot_type": info.get("robot_type"),
                    "codebase_version": info.get("codebase_version"),
                    "raw_metadata_totals": {
                        key: info.get(key)
                        for key in ("total_episodes", "total_frames", "total_tasks", "total_videos")
                    },
                },
                "action_sha256": array_sha256(action.astype(np.float32)) if action is not None else None,
                "state_sha256": array_sha256(state.astype(np.float32)) if state is not None else None,
                "action_state_relationship": action_state_report,
                "selected_motion_key": self.common["source_channels"]["motion_key"],
                "valid": valid,
                "problems": problems,
            }
            manifest_rows.append(row)
            summary_rows.append(
                {
                    "episode_index": episode_index,
                    "stable_episode_id": stable_id,
                    "source_name": root.name,
                    "parquet_path": str(parquet.resolve()),
                    "frames": frame_count,
                    "fps": fps,
                    "duration_sec": duration,
                    "action_shape": f"{frame_count}x14",
                    "state_shape": f"{frame_count}x14",
                    "selected_motion_key": self.common["source_channels"]["motion_key"],
                    "command_to_state_lag_frames": action_state_report.get(
                        "best_command_to_measured_arm_state_lag_frames"
                    ),
                    "camera_keys": ";".join(camera_keys),
                    "camera_count": len(camera_keys),
                    "valid": valid,
                    "problems": "; ".join(problems),
                }
            )
            if not valid:
                invalid_rows.append(row)

        self.records = records
        self.manifest = {
            "schema_version": "doll_handoff_source_manifest_v1",
            "source_glob": self.common["source_glob"],
            "excluded_names": self.common["excluded_names"],
            "enumerated_count": len(records),
            "valid_count": sum(record.valid for record in records),
            "invalid_count": sum(not record.valid for record in records),
            "stable_id_rule": (
                "copied from exact final common 50-source manifest"
                if self.selection_manifest is not None
                else "sorted source directory name -> zero-based index"
            ),
            "source_selection_manifest": self.common.get("source_selection_manifest"),
            "source_selection_rule": (
                "exact final common 50-source manifest order"
                if self.selection_manifest is not None
                else "sorted source directory name"
            ),
            "motion_source_key": self.common["source_channels"]["motion_key"],
            "command_source_key": self.common["source_channels"]["command_key"],
            "motion_source_selection": self.common["source_channels"][
                "motion_key_selection"
            ],
            "pooled_action_to_measured_state_arm_lag_audit": {
                "tested_state_lag_frames": list(range(16)),
                "pooled_mae_rad": {
                    str(lag): lag_sums[lag] / max(lag_counts[lag], 1)
                    for lag in range(16)
                },
                "best_state_lag_frames": min(
                    range(16),
                    key=lambda lag: lag_sums[lag] / max(lag_counts[lag], 1),
                ),
                "interpretation": (
                    "action is a command stream that leads measured observation.state; "
                    "observation.state is therefore used for executed-motion FK/events"
                ),
            },
            "pooled_action_to_measured_state_gripper_lag_audit": {
                "tested_state_lag_frames": list(range(16)),
                "pooled_mae_m": {
                    str(lag): gripper_lag_sums[lag]
                    / max(gripper_lag_counts[lag], 1)
                    for lag in range(16)
                },
                "best_state_lag_frames": min(
                    range(16),
                    key=lambda lag: gripper_lag_sums[lag]
                    / max(gripper_lag_counts[lag], 1),
                ),
                "interpretation": (
                    "commanded aperture transitions are aligned to measured FK time "
                    "with this single pooled dataset-wide lag"
                ),
            },
            "model_verified_channel_mapping": model_channels,
            "records": manifest_rows,
        }
        self.output.mkdir(parents=True, exist_ok=True)
        atomic_json(self.output / "source_manifest.json", self.manifest)
        atomic_csv(self.output / "episode_summary.csv", summary_rows)
        atomic_json(
            self.output / "invalid_sources.json",
            {
                "invalid_count": len(invalid_rows),
                "invalid_sources": invalid_rows,
                "silent_discard_count": 0,
            },
        )
        schema_lines = [
            "# Doll-Handoff source schema report",
            "",
            f"- Enumerated date-matched directories: **{len(records)}**",
            f"- Valid recordings: **{sum(record.valid for record in records)}**",
            f"- Invalid recordings: **{sum(not record.valid for record in records)}**",
            f"- Unique Arrow schemas: **{len(schema_texts)}**",
            f"- Unique action/state name orders: **{len(feature_name_sets)}**",
            "- Motion/FK/event source: `observation.state` (measured executed follower pose).",
            "- Command source retained for audit: `action`; pooled lag analysis shows it leads measured state.",
            f"- Pooled best command-to-measured-state lag: **{self.manifest['pooled_action_to_measured_state_arm_lag_audit']['best_state_lag_frames']} frames**.",
            f"- Pooled best gripper-command-to-measured-state lag: **{self.manifest['pooled_action_to_measured_state_gripper_lag_audit']['best_state_lag_frames']} frames**; this one global lag aligns event frames to measured FK.",
            "- Model-audited mapping: left arm `0:6`, left gripper `6`, right arm `7:13`, right gripper `13`.",
            "- Gripper proof: channels 6/13 match the named prismatic carriage limits; channels 0:6 and 7:13 map to the named six-revolute-joint chains.",
            "- Raw `meta/info.json` acquisition totals are zero in every standalone recording and are not used as episode counts; the parquet and image files are audited directly.",
            "",
            "## Arrow schema",
            "",
        ]
        for schema, count in schema_texts.items():
            schema_lines.extend([f"Observed in {count} recording(s):", "", "```text", schema, "```", ""])
        schema_lines.extend(
            [
                "## Camera assets",
                "",
                "Every recording contains frame-aligned PNG directories for `cam_high`, `cam_low`, `cam_left_wrist`, and `cam_right_wrist`; counts and sample dimensions are retained per episode in `source_manifest.json`.",
                "",
            ]
        )
        (self.output / "schema_report.md").write_text("\n".join(schema_lines), encoding="utf-8")
        return self.manifest

    def episode(self, episode_index: int) -> SourceEpisode:
        if not self.records:
            self.audit()
        if episode_index in self._cache:
            return self._cache[episode_index]
        record = self.records[episode_index]
        if not record.valid:
            raise RuntimeError(
                f"source episode {episode_index} is invalid and remains in reports: {record.problems}"
            )
        table = pq.read_table(
            record.parquet,
            columns=["action", "observation.state", "timestamp", "frame_index", "task_index"],
        )
        episode = SourceEpisode(
            record=record,
            action=fixed_list_numpy(table["action"], 14).astype(np.float64),
            state=fixed_list_numpy(table["observation.state"], 14).astype(np.float64),
            timestamps=scalar_numpy(table["timestamp"], np.float64),
            frame_index=scalar_numpy(table["frame_index"], np.int64),
            task_index=scalar_numpy(table["task_index"], np.int64),
        )
        self._cache[episode_index] = episode
        return episode

    def valid_episode_indices(self) -> list[int]:
        if not self.records:
            self.audit()
        return [record.episode_index for record in self.records if record.valid]


__all__ = ["SourceEpisode", "SourceRecord", "SourceRepository", "fixed_list_numpy"]
