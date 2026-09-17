#!/usr/bin/env python3
"""Lossless CPU-only Dataset-B -> HDF5 -> RLDS converter for UniFoLM.

Images are decoded once from the authoritative H.264 stream and stored as
lossless HDF5 uint8 arrays.  The RLDS builder encodes those arrays as PNG, so
RLDS readback can be pixel-exact to the converter's source decode.  State,
action, timestamps, indices, source identity, and the frozen task are retained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

import numpy as np


# Prevent TensorFlow or any transitive import from seeing the ACT-owned GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.unifolm_g1_dex3_28d_adapter import (  # noqa: E402
    ACTION_DIM,
    JOINT_NAMES,
    STATE_DIM,
    TASK_INSTRUCTION,
    validate_joint_mapping,
)


SCHEMA_VERSION = "unifolm_g1_dex3_28d_rlds_v1"
DEFAULT_DATASET = PROJECT_ROOT / "datasets/doll_handoff_proposed_b_50"
DEFAULT_SEED = 280328


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash relative paths, sizes, and file bytes in deterministic order."""
    root = root.resolve()
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(file_hash.encode("ascii") + b"\n")
        files.append({"path": relative, "size": size, "sha256": file_hash})
    return digest.hexdigest(), files


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fixed_list_to_numpy(column: Any, dim: int, dtype: np.dtype) -> np.ndarray:
    combined = column.combine_chunks()
    values = combined.values.to_numpy(zero_copy_only=False)
    result = np.asarray(values, dtype=dtype).reshape(len(combined), dim)
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    video_chunk_index: int
    video_file_index: int
    video_from_timestamp: float
    video_to_timestamp: float
    source_recording_identity: str


class DatasetBIndex:
    def __init__(self, root: Path) -> None:
        import pyarrow.parquet as pq

        self.root = root.resolve()
        info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        packaging = json.loads((self.root / "meta/g1_packaging_manifest.json").read_text(encoding="utf-8"))
        validate_joint_mapping(info["features"]["observation.state"]["names"], range(STATE_DIM))
        if info["features"]["action"]["names"] != list(JOINT_NAMES):
            raise ValueError("Dataset-B action joint order differs from the authoritative 28D order")
        if info["total_episodes"] != 50 or info["total_frames"] != 34478 or info["fps"] != 30:
            raise ValueError("Dataset-B totals differ from the authoritative contract")
        if packaging["task_instruction"] != TASK_INSTRUCTION:
            raise ValueError("Dataset-B task instruction differs from the frozen instruction")

        data_files = sorted((self.root / "data").rglob("*.parquet"))
        tables = [pq.read_table(path) for path in data_files]
        if len(tables) != 1:
            import pyarrow as pa

            table = pa.concat_tables(tables)
        else:
            table = tables[0]
        order = np.asarray(table["index"].combine_chunks().to_numpy())
        if not np.array_equal(order, np.arange(len(order), dtype=order.dtype)):
            raise ValueError("Dataset-B global index is not contiguous and ordered")
        self.state = fixed_list_to_numpy(table["observation.state"], STATE_DIM, np.float32)
        self.action = fixed_list_to_numpy(table["action"], ACTION_DIM, np.float32)
        self.timestamp = np.asarray(table["timestamp"].combine_chunks().to_numpy(), dtype=np.float32)
        self.frame_index = np.asarray(table["frame_index"].combine_chunks().to_numpy(), dtype=np.int64)
        self.episode_index = np.asarray(table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64)
        self.index = np.asarray(table["index"].combine_chunks().to_numpy(), dtype=np.int64)
        self.task_index = np.asarray(table["task_index"].combine_chunks().to_numpy(), dtype=np.int64)

        episode_files = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        episode_tables = [pq.read_table(path) for path in episode_files]
        if len(episode_tables) != 1:
            import pyarrow as pa

            episode_table = pa.concat_tables(episode_tables)
        else:
            episode_table = episode_tables[0]
        episode_dict = episode_table.to_pydict()
        source_by_episode = {
            int(item["final_dataset_episode_index"]): item["source_raw_episode"]
            for item in packaging["episode_alignment"]
        }
        self.episodes: list[EpisodeRecord] = []
        for row in range(episode_table.num_rows):
            episode_index = int(episode_dict["episode_index"][row])
            self.episodes.append(EpisodeRecord(
                episode_index=episode_index,
                length=int(episode_dict["length"][row]),
                dataset_from_index=int(episode_dict["dataset_from_index"][row]),
                dataset_to_index=int(episode_dict["dataset_to_index"][row]),
                video_chunk_index=int(episode_dict["videos/observation.images.cam_high/chunk_index"][row]),
                video_file_index=int(episode_dict["videos/observation.images.cam_high/file_index"][row]),
                video_from_timestamp=float(episode_dict["videos/observation.images.cam_high/from_timestamp"][row]),
                video_to_timestamp=float(episode_dict["videos/observation.images.cam_high/to_timestamp"][row]),
                source_recording_identity=source_by_episode[episode_index],
            ))
        self.episodes.sort(key=lambda episode: episode.episode_index)
        if [episode.episode_index for episode in self.episodes] != list(range(50)):
            raise ValueError("Dataset-B episode indices are not exactly 0..49")
        if sum(episode.length for episode in self.episodes) != 34478:
            raise ValueError("Dataset-B per-episode lengths do not sum to 34,478")

    def video_path(self, episode: EpisodeRecord) -> Path:
        return (
            self.root
            / "videos/observation.images.cam_high"
            / f"chunk-{episode.video_chunk_index:03d}"
            / f"file-{episode.video_file_index:03d}.mp4"
        )

    def episode_arrays(self, episode: EpisodeRecord) -> dict[str, np.ndarray]:
        start, stop = episode.dataset_from_index, episode.dataset_to_index
        arrays = {
            "state": self.state[start:stop],
            "action": self.action[start:stop],
            "timestamp": self.timestamp[start:stop],
            "frame_index": self.frame_index[start:stop],
            "episode_index": self.episode_index[start:stop],
            "index": self.index[start:stop],
            "task_index": self.task_index[start:stop],
        }
        if any(len(value) != episode.length for value in arrays.values()):
            raise ValueError(f"episode {episode.episode_index}: numeric slice length mismatch")
        if not np.array_equal(arrays["frame_index"], np.arange(episode.length)):
            raise ValueError(f"episode {episode.episode_index}: frame index mismatch")
        if not np.all(arrays["episode_index"] == episode.episode_index):
            raise ValueError(f"episode {episode.episode_index}: episode index mismatch")
        return arrays


def iter_video_rgb(video_path: Path) -> Iterator[tuple[int, np.ndarray, int, float]]:
    import av

    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            pts = -1 if frame.pts is None else int(frame.pts)
            seconds = float("nan") if frame.time is None else float(frame.time)
            yield frame_index, rgb, pts, seconds


def _bytes_from_h5(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def convert_episode_to_hdf5(index: DatasetBIndex, episode: EpisodeRecord, output_path: Path) -> dict[str, Any]:
    import h5py

    arrays = index.episode_arrays(episode)
    if not np.isfinite(arrays["state"]).all() or not np.isfinite(arrays["action"]).all():
        raise ValueError(f"episode {episode.episode_index}: state/action contains NaN or Inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    image_digest = hashlib.sha256()
    decoded_frames = 0
    video_path = index.video_path(episode)
    with h5py.File(temporary, "w", libver="latest") as target:
        target.attrs.update({
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode.episode_index,
            "source_recording_identity": episode.source_recording_identity,
            "source_video_path": str(video_path),
            "fps": 30,
            "frame_count": episode.length,
            "image_semantic": "observation.images.cam_high decoded RGB24",
            "state_semantic": "q_target[max(t-1,0)]",
            "action_semantic": "absolute q_target[t]",
        })
        observations = target.create_group("observations")
        images = observations.create_group("images")
        image_dataset = images.create_dataset(
            "cam_high",
            shape=(episode.length, 480, 640, 3),
            dtype=np.uint8,
            chunks=(1, 480, 640, 3),
            compression="gzip",
            compression_opts=4,
        )
        observations.create_dataset("qpos", data=arrays["state"], dtype=np.float32, compression="gzip")
        target.create_dataset("action", data=arrays["action"], dtype=np.float32, compression="gzip")
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            target.create_dataset(key, data=arrays[key], compression="gzip")
        target.create_dataset("video_pts", shape=(episode.length,), dtype=np.int64, compression="gzip")
        target.create_dataset("video_timestamp", shape=(episode.length,), dtype=np.float64, compression="gzip")
        text_dtype = h5py.string_dtype(encoding="utf-8")
        target.create_dataset("language_instruction", data=TASK_INSTRUCTION, dtype=text_dtype)
        target.create_dataset("joint_names", data=np.asarray(JOINT_NAMES, dtype=object), dtype=text_dtype)

        for frame_index, rgb, pts, seconds in iter_video_rgb(video_path):
            if frame_index >= episode.length:
                raise ValueError(f"episode {episode.episode_index}: video has extra frames")
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"episode {episode.episode_index}: unexpected RGB shape/dtype {rgb.shape}/{rgb.dtype}")
            image_dataset[frame_index] = rgb
            target["video_pts"][frame_index] = pts
            target["video_timestamp"][frame_index] = seconds
            image_digest.update(rgb.tobytes(order="C"))
            decoded_frames += 1
        if decoded_frames != episode.length:
            raise ValueError(
                f"episode {episode.episode_index}: decoded {decoded_frames} frames, expected {episode.length}"
            )
        target.attrs["decoded_rgb_sha256"] = image_digest.hexdigest()
        target.flush()
    os.replace(temporary, output_path)
    report = verify_hdf5_episode(index, episode, output_path, inspect_count=10, verify_source_images=True)
    report["hdf5_path"] = str(output_path.resolve())
    report["hdf5_sha256"] = sha256_file(output_path)
    report["hdf5_size_bytes"] = output_path.stat().st_size
    return report


def verify_hdf5_episode(
    index: DatasetBIndex,
    episode: EpisodeRecord,
    hdf5_path: Path,
    inspect_count: int = 10,
    verify_source_images: bool = True,
) -> dict[str, Any]:
    import h5py

    arrays = index.episode_arrays(episode)
    with h5py.File(hdf5_path, "r") as converted:
        state = converted["observations/qpos"][:]
        action = converted["action"][:]
        timestamp = converted["timestamp"][:]
        frame_index = converted["frame_index"][:]
        episode_index = converted["episode_index"][:]
        global_index = converted["index"][:]
        task_index = converted["task_index"][:]
        video_pts = converted["video_pts"][:]
        video_timestamp = converted["video_timestamp"][:]
        task = _bytes_from_h5(converted["language_instruction"][()])
        joints = tuple(_bytes_from_h5(value) for value in converted["joint_names"][:])
        image_dataset = converted["observations/images/cam_high"]
        state_error = float(np.max(np.abs(state.astype(np.float64) - arrays["state"].astype(np.float64))))
        action_error = float(np.max(np.abs(action.astype(np.float64) - arrays["action"].astype(np.float64))))
        timestamp_error = float(np.max(np.abs(timestamp.astype(np.float64) - arrays["timestamp"].astype(np.float64))))
        video_timestamp_alignment_error = float(
            np.max(np.abs(video_timestamp - timestamp.astype(np.float64)))
        )
        checks = {
            "frame_count": len(state) == episode.length == image_dataset.shape[0],
            "state_shape": state.shape == (episode.length, STATE_DIM),
            "action_shape": action.shape == (episode.length, ACTION_DIM),
            "state_exact": np.array_equal(state, arrays["state"]),
            "action_exact": np.array_equal(action, arrays["action"]),
            "timestamp_exact": np.array_equal(timestamp, arrays["timestamp"]),
            "frame_index_exact": np.array_equal(frame_index, arrays["frame_index"]),
            "episode_index_exact": np.array_equal(episode_index, arrays["episode_index"]),
            "global_index_exact": np.array_equal(global_index, arrays["index"]),
            "task_index_exact": np.array_equal(task_index, arrays["task_index"]),
            "metadata_episode_exact": int(converted.attrs["episode_id"]) == episode.episode_index,
            "metadata_frame_count_exact": int(converted.attrs["frame_count"]) == episode.length,
            "metadata_fps_exact": int(converted.attrs["fps"]) == 30,
            "metadata_source_identity_exact": (
                _bytes_from_h5(converted.attrs["source_recording_identity"])
                == episode.source_recording_identity
            ),
            "metadata_source_video_exact": (
                _bytes_from_h5(converted.attrs["source_video_path"])
                == str(index.video_path(episode))
            ),
            "task_exact": task == TASK_INSTRUCTION,
            "joint_order_exact": joints == JOINT_NAMES,
            "state_finite": bool(np.isfinite(state).all()),
            "action_finite": bool(np.isfinite(action).all()),
        }
        image_source_digest = hashlib.sha256()
        image_converted_digest = hashlib.sha256()
        image_mismatches = 0
        if verify_source_images:
            source_pts = []
            source_video_timestamps = []
            for decoded_index, rgb, pts, seconds in iter_video_rgb(index.video_path(episode)):
                converted_rgb = image_dataset[decoded_index]
                source_pts.append(pts)
                source_video_timestamps.append(seconds)
                image_source_digest.update(rgb.tobytes(order="C"))
                image_converted_digest.update(converted_rgb.tobytes(order="C"))
                if not np.array_equal(rgb, converted_rgb):
                    image_mismatches += 1
            checks["image_exact"] = image_mismatches == 0
            checks["video_pts_exact"] = np.array_equal(video_pts, np.asarray(source_pts, dtype=np.int64))
            checks["video_timestamp_exact"] = np.array_equal(
                video_timestamp, np.asarray(source_video_timestamps, dtype=np.float64)
            )
            checks["dataset_video_timestamp_aligned"] = video_timestamp_alignment_error <= 1e-6
        else:
            for start in range(0, episode.length, 32):
                batch = image_dataset[start : min(start + 32, episode.length)]
                image_converted_digest.update(np.ascontiguousarray(batch).tobytes(order="C"))

        random = np.random.default_rng(DEFAULT_SEED + episode.episode_index)
        inspect_indices = sorted(random.choice(episode.length, size=min(inspect_count, episode.length), replace=False).tolist())
        inspected = []
        for local_index in inspect_indices:
            rgb = image_dataset[local_index]
            inspected.append({
                "episode_index": episode.episode_index,
                "frame_index": int(frame_index[local_index]),
                "global_index": int(converted["index"][local_index]),
                "timestamp": float(timestamp[local_index]),
                "rgb_shape": list(rgb.shape),
                "rgb_dtype": str(rgb.dtype),
                "rgb_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
                "state_shape": list(state[local_index].shape),
                "action_shape": list(action[local_index].shape),
                "task": task,
                "state_nan": bool(np.isnan(state[local_index]).any()),
                "state_inf": bool(np.isinf(state[local_index]).any()),
                "action_nan": bool(np.isnan(action[local_index]).any()),
                "action_inf": bool(np.isinf(action[local_index]).any()),
            })
    if not all(checks.values()):
        raise ValueError(f"episode {episode.episode_index} HDF5 verification failed: {checks}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "episode_index": episode.episode_index,
        "source_recording_identity": episode.source_recording_identity,
        "frame_count": episode.length,
        "fps": 30,
        "task": TASK_INSTRUCTION,
        "checks": checks,
        "max_abs_state_error": state_error,
        "max_abs_action_error": action_error,
        "max_abs_timestamp_error": timestamp_error,
        "max_abs_dataset_vs_video_timestamp_error": video_timestamp_alignment_error,
        "image_mismatch_count": image_mismatches,
        "source_decoded_rgb_sha256": image_source_digest.hexdigest() if verify_source_images else None,
        "converted_rgb_sha256": image_converted_digest.hexdigest(),
        "inspected_frames": inspected,
    }


def parse_episode_selection(selection: str, count: int) -> list[int]:
    if selection == "all":
        return list(range(count))
    values: set[int] = set()
    for part in selection.split(","):
        if "-" in part:
            start_text, stop_text = part.split("-", 1)
            values.update(range(int(start_text), int(stop_text) + 1))
        else:
            values.add(int(part))
    result = sorted(values)
    if not result or result[0] < 0 or result[-1] >= count:
        raise ValueError(f"invalid episode selection {selection!r}")
    return result


def convert_hdf5_command(args: argparse.Namespace) -> None:
    dataset = DatasetBIndex(args.dataset)
    selection = parse_episode_selection(args.episodes, len(dataset.episodes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for episode_index in selection:
        episode = dataset.episodes[episode_index]
        output_path = args.output_dir / f"episode_{episode_index:06d}.hdf5"
        if output_path.exists() and not args.overwrite:
            report = verify_hdf5_episode(dataset, episode, output_path, inspect_count=10, verify_source_images=True)
            report["hdf5_path"] = str(output_path.resolve())
            report["hdf5_sha256"] = sha256_file(output_path)
            report["hdf5_size_bytes"] = output_path.stat().st_size
        else:
            report = convert_episode_to_hdf5(dataset, episode, output_path)
        reports.append(report)
        print(f"episode {episode_index:02d}: PASS {episode.length} frames {output_path.stat().st_size} bytes", flush=True)
    converted_hash, files = tree_hash(args.output_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "cpu_only": True,
        "source_dataset": str(dataset.root),
        "selected_episodes": selection,
        "episode_count": len(selection),
        "frame_count": sum(report["frame_count"] for report in reports),
        "state_identity": all(report["checks"]["state_exact"] for report in reports),
        "action_identity": all(report["checks"]["action_exact"] for report in reports),
        "timestamp_identity": all(report["checks"]["timestamp_exact"] for report in reports),
        "image_identity": all(report["checks"]["image_exact"] for report in reports),
        "dataset_tree_sha256": converted_hash,
        "files": files,
        "episodes": reports,
    }
    write_json(args.report, payload)
    print(json.dumps({key: payload[key] for key in ("status", "episode_count", "frame_count", "dataset_tree_sha256")}, indent=2))


def make_rlds_builder(hdf5_paths: Sequence[Path], data_dir: Path) -> Any:
    import h5py
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    class G1Dex3_28d(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Lossless Dataset-B G1 arms + Dex3 28D conversion."}
        # The builder is defined by this project-side executable rather than an
        # importable Python package, so give TFDS an explicit metadata root.
        pkg_dir_path = Path(__file__).resolve().parent

        def _info(self) -> Any:
            return self.dataset_info_from_configs(features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(480, 640, 3), dtype=np.uint8, encoding_format="png"
                        ),
                        "state": tfds.features.Tensor(shape=(STATE_DIM,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(ACTION_DIM,), dtype=np.float32),
                    "timestamp": tfds.features.Scalar(dtype=np.float32),
                    "frame_index": tfds.features.Scalar(dtype=np.int64),
                    "episode_index": tfds.features.Scalar(dtype=np.int64),
                    "index": tfds.features.Scalar(dtype=np.int64),
                    "task_index": tfds.features.Scalar(dtype=np.int64),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "episode_index": tfds.features.Scalar(dtype=np.int64),
                    "source_recording_identity": tfds.features.Text(),
                    "source_hdf5_path": tfds.features.Text(),
                    "frame_count": tfds.features.Scalar(dtype=np.int64),
                    "fps": tfds.features.Scalar(dtype=np.int64),
                }),
            }))

        def _split_generators(self, dl_manager: Any) -> dict[str, Sequence[Path]]:
            del dl_manager
            return {"train": self._generate_examples(hdf5_paths)}

        def _generate_examples(self, paths: Sequence[Path]) -> Iterator[tuple[str, dict[str, Any]]]:
            for path in paths:
                with h5py.File(path, "r") as source:
                    episode_index = int(source.attrs["episode_id"])
                    frame_count = int(source.attrs["frame_count"])
                    source_identity = _bytes_from_h5(source.attrs["source_recording_identity"])
                    task = _bytes_from_h5(source["language_instruction"][()])
                    steps = []
                    for frame_index in range(frame_count):
                        steps.append({
                            "observation": {
                                "image": source["observations/images/cam_high"][frame_index],
                                "state": source["observations/qpos"][frame_index],
                            },
                            "action": source["action"][frame_index],
                            "timestamp": source["timestamp"][frame_index],
                            "frame_index": source["frame_index"][frame_index],
                            "episode_index": source["episode_index"][frame_index],
                            "index": source["index"][frame_index],
                            "task_index": source["task_index"][frame_index],
                            "discount": np.float32(1.0),
                            "is_first": frame_index == 0,
                            "is_last": frame_index == frame_count - 1,
                            "is_terminal": frame_index == frame_count - 1,
                            "language_instruction": task,
                        })
                yield f"episode_{episode_index:06d}", {
                    "steps": steps,
                    "episode_metadata": {
                        "episode_index": episode_index,
                        "source_recording_identity": source_identity,
                        "source_hdf5_path": str(path.resolve()),
                        "frame_count": frame_count,
                        "fps": 30,
                    },
                }

    return G1Dex3_28d(data_dir=str(data_dir))


def build_rlds_command(args: argparse.Namespace) -> None:
    hdf5_paths = sorted(args.hdf5_dir.glob("episode_*.hdf5"))
    if not hdf5_paths:
        raise ValueError(f"no HDF5 episodes found under {args.hdf5_dir}")
    builder = make_rlds_builder(hdf5_paths, args.output_root)
    import tensorflow_datasets as tfds

    if args.overwrite and Path(builder.data_dir).exists():
        raise ValueError("destructive RLDS overwrite is intentionally unsupported; choose a fresh output root")
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(try_download_gcs=False)
    )
    print(f"RLDS prepared: {builder.data_dir}", flush=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "cpu_only": True,
        "hdf5_dir": str(args.hdf5_dir.resolve()),
        "hdf5_episode_count": len(hdf5_paths),
        "rlds_dir": str(Path(builder.data_dir).resolve()),
    }
    dataset_hash, files = tree_hash(Path(builder.data_dir))
    payload["dataset_tree_sha256"] = dataset_hash
    payload["files"] = files
    write_json(args.report, payload)


def verify_rlds_command(args: argparse.Namespace) -> None:
    import h5py
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder_from_directory(str(args.rlds_dir))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    hdf5_by_episode = {
        int(path.stem.split("_")[-1]): path for path in sorted(args.hdf5_dir.glob("episode_*.hdf5"))
    }
    episode_reports = []
    total_frames = 0
    dataset_state_digest = hashlib.sha256()
    dataset_action_digest = hashlib.sha256()
    dataset_timestamp_digest = hashlib.sha256()
    dataset_image_digest = hashlib.sha256()
    canonical_numeric_bytes: dict[int, tuple[bytes, bytes, bytes]] = {}
    for serialized_episode in dataset:
        metadata = serialized_episode["episode_metadata"]
        episode_index = int(metadata["episode_index"].numpy())
        path = hdf5_by_episode[episode_index]
        state_digest = hashlib.sha256()
        action_digest = hashlib.sha256()
        image_digest = hashlib.sha256()
        states, actions, timestamps, frames, episode_indices = [], [], [], [], []
        global_indices, task_indices, tasks = [], [], []
        first_flags, last_flags, terminal_flags = [], [], []
        with h5py.File(path, "r") as source:
            source_state = source["observations/qpos"][:]
            source_action = source["action"][:]
            source_timestamp = source["timestamp"][:]
            source_episode_index = source["episode_index"][:]
            source_global_index = source["index"][:]
            source_task_index = source["task_index"][:]
            source_identity = _bytes_from_h5(source.attrs["source_recording_identity"])
            source_image_digest = hashlib.sha256()
            for start in range(0, len(source_state), 32):
                batch = source["observations/images/cam_high"][start : min(start + 32, len(source_state))]
                source_image_digest.update(np.ascontiguousarray(batch).tobytes(order="C"))
            for step in serialized_episode["steps"]:
                state = step["observation"]["state"].numpy()
                action = step["action"].numpy()
                image = step["observation"]["image"].numpy()
                states.append(state)
                actions.append(action)
                timestamps.append(step["timestamp"].numpy())
                frames.append(int(step["frame_index"].numpy()))
                episode_indices.append(int(step["episode_index"].numpy()))
                global_indices.append(int(step["index"].numpy()))
                task_indices.append(int(step["task_index"].numpy()))
                tasks.append(step["language_instruction"].numpy().decode("utf-8"))
                first_flags.append(bool(step["is_first"].numpy()))
                last_flags.append(bool(step["is_last"].numpy()))
                terminal_flags.append(bool(step["is_terminal"].numpy()))
                state_digest.update(np.ascontiguousarray(state).tobytes(order="C"))
                action_digest.update(np.ascontiguousarray(action).tobytes(order="C"))
                image_digest.update(np.ascontiguousarray(image).tobytes(order="C"))
            states_array = np.asarray(states, dtype=np.float32)
            actions_array = np.asarray(actions, dtype=np.float32)
            timestamps_array = np.asarray(timestamps, dtype=np.float32)
            canonical_numeric_bytes[episode_index] = (
                states_array.tobytes(order="C"),
                actions_array.tobytes(order="C"),
                timestamps_array.tobytes(order="C"),
            )
            checks = {
                "frame_count": len(states) == len(source_state),
                "state_shape": states_array.shape == source_state.shape,
                "action_shape": actions_array.shape == source_action.shape,
                "state_exact": np.array_equal(states_array, source_state),
                "action_exact": np.array_equal(actions_array, source_action),
                "timestamp_exact": np.array_equal(timestamps_array, source_timestamp),
                "frame_index_exact": frames == list(range(len(source_state))),
                "episode_index_exact": episode_indices == source_episode_index.tolist(),
                "global_index_exact": global_indices == source_global_index.tolist(),
                "task_index_exact": task_indices == source_task_index.tolist(),
                "task_exact": len(set(tasks)) == 1 and tasks[0] == TASK_INSTRUCTION,
                "terminal_flags_exact": (
                    first_flags == [True] + [False] * (len(source_state) - 1)
                    and last_flags == [False] * (len(source_state) - 1) + [True]
                    and terminal_flags == [False] * (len(source_state) - 1) + [True]
                ),
                "metadata_episode_exact": int(metadata["episode_index"].numpy()) == episode_index,
                "metadata_frame_count_exact": int(metadata["frame_count"].numpy()) == len(source_state),
                "metadata_fps_exact": int(metadata["fps"].numpy()) == 30,
                "metadata_source_identity_exact": (
                    metadata["source_recording_identity"].numpy().decode("utf-8") == source_identity
                ),
                "image_exact": image_digest.hexdigest() == source_image_digest.hexdigest(),
                "state_finite": bool(np.isfinite(states_array).all()),
                "action_finite": bool(np.isfinite(actions_array).all()),
            }
            state_error = float(np.max(np.abs(states_array.astype(np.float64) - source_state.astype(np.float64))))
            action_error = float(np.max(np.abs(actions_array.astype(np.float64) - source_action.astype(np.float64))))
            timestamp_error = float(np.max(np.abs(timestamps_array.astype(np.float64) - source_timestamp.astype(np.float64))))
        if not all(checks.values()):
            raise ValueError(f"RLDS episode {episode_index} verification failed: {checks}")
        episode_reports.append({
            "episode_index": episode_index,
            "frame_count": len(states),
            "checks": checks,
            "max_abs_state_error": state_error,
            "max_abs_action_error": action_error,
            "max_abs_timestamp_error": timestamp_error,
            "state_sha256": state_digest.hexdigest(),
            "action_sha256": action_digest.hexdigest(),
            "decoded_rgb_sha256": image_digest.hexdigest(),
        })
        total_frames += len(states)
        print(f"RLDS episode {episode_index:02d}: PASS {len(states)} frames", flush=True)
    episode_reports.sort(key=lambda report: report["episode_index"])
    if [report["episode_index"] for report in episode_reports] != sorted(hdf5_by_episode):
        raise ValueError("RLDS episode IDs differ from the HDF5 manifest")
    for report in episode_reports:
        state_bytes, action_bytes, timestamp_bytes = canonical_numeric_bytes[report["episode_index"]]
        dataset_state_digest.update(state_bytes)
        dataset_action_digest.update(action_bytes)
        dataset_timestamp_digest.update(timestamp_bytes)
        dataset_image_digest.update(bytes.fromhex(report["decoded_rgb_sha256"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "cpu_only": True,
        "rlds_dir": str(args.rlds_dir.resolve()),
        "episode_count": len(episode_reports),
        "frame_count": total_frames,
        "state_identity": all(item["checks"]["state_exact"] for item in episode_reports),
        "action_identity": all(item["checks"]["action_exact"] for item in episode_reports),
        "timestamp_identity": all(item["checks"]["timestamp_exact"] for item in episode_reports),
        "image_identity": all(item["checks"]["image_exact"] for item in episode_reports),
        "task_exact": all(item["checks"]["task_exact"] for item in episode_reports),
        "all_index_fields_exact": all(
            item["checks"][key]
            for item in episode_reports
            for key in ("frame_index_exact", "episode_index_exact", "global_index_exact", "task_index_exact")
        ),
        "all_episode_metadata_exact": all(
            item["checks"][key]
            for item in episode_reports
            for key in (
                "metadata_episode_exact",
                "metadata_frame_count_exact",
                "metadata_fps_exact",
                "metadata_source_identity_exact",
            )
        ),
        "state_sha256": dataset_state_digest.hexdigest(),
        "action_sha256": dataset_action_digest.hexdigest(),
        "timestamp_sha256": dataset_timestamp_digest.hexdigest(),
        "decoded_rgb_episode_hashes_sha256": dataset_image_digest.hexdigest(),
        "episodes": episode_reports,
    }
    dataset_hash, files = tree_hash(args.rlds_dir)
    payload["dataset_tree_sha256"] = dataset_hash
    payload["files"] = files
    write_json(args.report, payload)
    print(json.dumps({key: payload[key] for key in ("status", "episode_count", "frame_count", "dataset_tree_sha256")}, indent=2))


def hash_command(args: argparse.Namespace) -> None:
    digest, files = tree_hash(args.root)
    payload = {"root": str(args.root.resolve()), "tree_sha256": digest, "files": files}
    if args.report:
        write_json(args.report, payload)
    print(json.dumps(payload, indent=2))


def _statistics(values: np.ndarray, include_mask: bool = False) -> dict[str, Any]:
    result = {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }
    if include_mask:
        result["mask"] = [True] * ACTION_DIM
    return result


def audit_source_command(args: argparse.Namespace) -> None:
    dataset = DatasetBIndex(args.dataset)
    episode_checks = []
    for episode in dataset.episodes:
        arrays = dataset.episode_arrays(episode)
        expected_timestamps = np.arange(episode.length, dtype=np.float32) / np.float32(30.0)
        checks = {
            "frame0_state_equals_action0": bool(np.array_equal(arrays["state"][0], arrays["action"][0])),
            "lag1_state_equals_previous_action": bool(np.array_equal(arrays["state"][1:], arrays["action"][:-1])),
            "timestamp_matches_frame_over_30": bool(np.array_equal(arrays["timestamp"], expected_timestamps)),
            "state_finite": bool(np.isfinite(arrays["state"]).all()),
            "action_finite": bool(np.isfinite(arrays["action"]).all()),
        }
        if not all(checks.values()):
            raise ValueError(f"source episode {episode.episode_index} failed: {checks}")
        episode_checks.append({
            "episode_index": episode.episode_index,
            "source_recording_identity": episode.source_recording_identity,
            "frame_count": episode.length,
            "checks": checks,
        })
    dataset_digest, files = tree_hash(dataset.root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "cpu_only": True,
        "dataset": str(dataset.root),
        "dataset_tree_sha256": dataset_digest,
        "file_count": len(files),
        "episode_count": len(dataset.episodes),
        "frame_count": len(dataset.state),
        "fps": 30,
        "state_shape": list(dataset.state.shape),
        "action_shape": list(dataset.action.shape),
        "state_sha256": hashlib.sha256(dataset.state.tobytes(order="C")).hexdigest(),
        "action_sha256": hashlib.sha256(dataset.action.tobytes(order="C")).hexdigest(),
        "timestamp_sha256": hashlib.sha256(dataset.timestamp.tobytes(order="C")).hexdigest(),
        "task_instruction": TASK_INSTRUCTION,
        "joint_names": list(JOINT_NAMES),
        "mapped_indices": list(range(STATE_DIM)),
        "state_semantic": "frame 0: q_target[0]; t > 0: q_target[t-1]",
        "action_semantic": "absolute q_target[t] joint-position target",
        "metadata_conflict": {
            "meta/g1_training_contract.json": "claims same-row q_target[t] and is stale",
            "authoritative_and_verified": "meta/g1_packaging_manifest.json plus numeric lag-1 equality",
        },
        "episodes": episode_checks,
    }
    write_json(args.report, payload)
    stats = {
        "g1_dex3_28d": {
            "action": _statistics(dataset.action, include_mask=True),
            "proprio": _statistics(dataset.state),
            "num_transitions": len(dataset.state),
            "num_trajectories": len(dataset.episodes),
            "normalization_type": "bounds",
            "joint_names": list(JOINT_NAMES),
        }
    }
    write_json(args.statistics, stats)
    print(json.dumps({
        "status": payload["status"],
        "episodes": payload["episode_count"],
        "frames": payload["frame_count"],
        "dataset_tree_sha256": payload["dataset_tree_sha256"],
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hdf5_parser = subparsers.add_parser("convert-hdf5")
    hdf5_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    hdf5_parser.add_argument("--output-dir", type=Path, required=True)
    hdf5_parser.add_argument("--episodes", default="all", help="all, comma list, or inclusive ranges")
    hdf5_parser.add_argument("--report", type=Path, required=True)
    hdf5_parser.add_argument("--overwrite", action="store_true")
    hdf5_parser.set_defaults(func=convert_hdf5_command)

    rlds_parser = subparsers.add_parser("build-rlds")
    rlds_parser.add_argument("--hdf5-dir", type=Path, required=True)
    rlds_parser.add_argument("--output-root", type=Path, required=True)
    rlds_parser.add_argument("--report", type=Path, required=True)
    rlds_parser.add_argument("--overwrite", action="store_true")
    rlds_parser.set_defaults(func=build_rlds_command)

    verify_parser = subparsers.add_parser("verify-rlds")
    verify_parser.add_argument("--hdf5-dir", type=Path, required=True)
    verify_parser.add_argument("--rlds-dir", type=Path, required=True)
    verify_parser.add_argument("--report", type=Path, required=True)
    verify_parser.set_defaults(func=verify_rlds_command)

    hash_parser = subparsers.add_parser("hash-tree")
    hash_parser.add_argument("--root", type=Path, required=True)
    hash_parser.add_argument("--report", type=Path)
    hash_parser.set_defaults(func=hash_command)

    source_parser = subparsers.add_parser("audit-source")
    source_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    source_parser.add_argument("--report", type=Path, required=True)
    source_parser.add_argument("--statistics", type=Path, required=True)
    source_parser.set_defaults(func=audit_source_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
