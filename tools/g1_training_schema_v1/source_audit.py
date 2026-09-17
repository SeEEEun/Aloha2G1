from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .constants import FPS, IMAGE_KEY, TASK_TEXT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


@dataclass(frozen=True)
class SourceEpisode:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    data_chunk_index: int
    data_file_index: int
    video_chunk_index: int
    video_file_index: int
    video_from_timestamp: float
    video_to_timestamp: float
    tasks: tuple[str, ...]
    raw: dict[str, Any]


class SourceDatasetIndex:
    """Read-only index over the authoritative source LeRobot v3 dataset."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.info_path = self.root / "meta/info.json"
        if not self.info_path.is_file():
            raise FileNotFoundError(f"missing LeRobot metadata: {self.info_path}")
        self.info = load_json(self.info_path)
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError(f"expected LeRobot v3.0, got {self.info.get('codebase_version')!r}")
        if float(self.info.get("fps", -1)) != FPS:
            raise ValueError(f"expected {FPS} fps source, got {self.info.get('fps')!r}")
        self._episode_rows = self._read_episode_rows()
        self.episodes = {int(row["episode_index"]): self._parse_episode(row) for row in self._episode_rows}
        expected = list(range(int(self.info["total_episodes"])))
        if sorted(self.episodes) != expected:
            raise ValueError("source episode_index values are not contiguous 0..N-1")

    def _read_episode_rows(self) -> list[dict[str, Any]]:
        paths = sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet"))
        if not paths:
            raise FileNotFoundError("no meta/episodes parquet files")
        rows: list[dict[str, Any]] = []
        for path in paths:
            rows.extend(pq.read_table(path).to_pylist())
        return sorted(rows, key=lambda row: int(row["episode_index"]))

    @staticmethod
    def _parse_episode(row: dict[str, Any]) -> SourceEpisode:
        key = f"videos/{IMAGE_KEY}"
        return SourceEpisode(
            episode_index=int(row["episode_index"]),
            length=int(row["length"]),
            dataset_from_index=int(row["dataset_from_index"]),
            dataset_to_index=int(row["dataset_to_index"]),
            data_chunk_index=int(row["data/chunk_index"]),
            data_file_index=int(row["data/file_index"]),
            video_chunk_index=int(row[f"{key}/chunk_index"]),
            video_file_index=int(row[f"{key}/file_index"]),
            video_from_timestamp=float(row[f"{key}/from_timestamp"]),
            video_to_timestamp=float(row[f"{key}/to_timestamp"]),
            tasks=tuple(row["tasks"]),
            raw=row,
        )

    def data_path(self, episode: SourceEpisode) -> Path:
        return self.root / self.info["data_path"].format(
            chunk_index=episode.data_chunk_index, file_index=episode.data_file_index
        )

    def video_path(self, episode: SourceEpisode) -> Path:
        return self.root / self.info["video_path"].format(
            video_key=IMAGE_KEY,
            chunk_index=episode.video_chunk_index,
            file_index=episode.video_file_index,
        )

    def read_episode_arrays(self, episode_id: int) -> dict[str, np.ndarray]:
        episode = self.episodes[episode_id]
        table = pq.read_table(self.data_path(episode))
        ep_col = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        selected = np.flatnonzero(ep_col == episode_id)
        if selected.size != episode.length:
            raise ValueError(
                f"source episode {episode_id} frame count {selected.size} != metadata {episode.length}"
            )
        first, last = int(selected[0]), int(selected[-1])
        if last - first + 1 != episode.length:
            raise ValueError(f"source episode {episode_id} rows are not contiguous")
        part = table.slice(first, episode.length)
        arrays: dict[str, np.ndarray] = {}
        for key in ["observation.state", "action"]:
            arrays[key] = np.asarray(part[key].to_pylist(), dtype=np.float32)
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            arrays[key] = np.asarray(part[key].to_numpy())
        expected_frames = np.arange(episode.length, dtype=np.int64)
        if not np.array_equal(arrays["frame_index"], expected_frames):
            raise ValueError(f"source episode {episode_id} frame_index does not reset at zero")
        expected_ts = expected_frames / FPS
        if not np.allclose(arrays["timestamp"], expected_ts, rtol=0.0, atol=1e-5):
            raise ValueError(f"source episode {episode_id} timestamps are not frame_index/fps")
        return arrays

    def validate_rgb_reference(self, episode_id: int) -> Path:
        path = self.video_path(self.episodes[episode_id])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"invalid source RGB video reference: {path}")
        return path


def build_source_audit(root: str | Path) -> dict[str, Any]:
    source = SourceDatasetIndex(root)
    info = source.info
    task_table = pq.read_table(source.root / "meta/tasks.parquet").to_pylist()
    data_paths = sorted((source.root / "data").glob("chunk-*/file-*.parquet"))
    video_paths = sorted((source.root / "videos" / IMAGE_KEY).glob("chunk-*/*.mp4"))
    first = source.read_episode_arrays(0)
    last = source.read_episode_arrays(max(source.episodes))
    tasks = [row.get("__index_level_0__") for row in task_table]
    if tasks != [TASK_TEXT]:
        raise ValueError(f"unexpected task metadata: {tasks}")
    return {
        "audit_status": "PASS",
        "root": str(source.root),
        "dataset_version": info["codebase_version"],
        "robot_type": info["robot_type"],
        "counts": {
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "tasks": info["total_tasks"],
            "video_shards": len(video_paths),
        },
        "features": info["features"],
        "rgb_keys": [IMAGE_KEY],
        "state_dimension": info["features"]["observation.state"]["shape"][0],
        "action_dimension": info["features"]["action"]["shape"][0],
        "fps": info["fps"],
        "timestamp_convention": "episode-local float32 frame_index / 30; resets to 0 each episode",
        "episode_index_convention": "int64 contiguous 0..49",
        "frame_index_convention": "int64 contiguous 0..T-1, reset per episode",
        "global_index_convention": "int64 contiguous 0..50301 across episodes",
        "task_storage": {"path": "meta/tasks.parquet", "rows": task_table, "episode_tasks": "meta/episodes tasks list"},
        "video_storage": {"template": info["video_path"], "paths": [str(p.relative_to(source.root)) for p in video_paths]},
        "parquet_storage": {"template": info["data_path"], "paths": [str(p.relative_to(source.root)) for p in data_paths]},
        "normalization_metadata": {
            "path": "meta/stats.json",
            "present": (source.root / "meta/stats.json").is_file(),
            "stat_keys": ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"],
        },
        "boundary_spot_checks": {
            "episode_0_length": len(first["timestamp"]),
            "episode_49_length": len(last["timestamp"]),
            "timestamps_and_indices_valid": True,
        },
        "sha256": {
            "meta/info.json": sha256_file(source.root / "meta/info.json"),
            "meta/stats.json": sha256_file(source.root / "meta/stats.json"),
            "meta/tasks.parquet": sha256_file(source.root / "meta/tasks.parquet"),
            **{str(p.relative_to(source.root)): sha256_file(p) for p in data_paths + video_paths},
        },
    }
