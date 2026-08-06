#!/usr/bin/env python3
"""Safely rebuild 50 single-episode LeRobot v2.1 datasets as one v3.0 dataset."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# Hugging Face Datasets otherwise writes lock/cache files under the user's
# read-only ~/.cache. This process-private cache never touches LeRobot's cache.
_PROCESS_HF_CACHE = Path("/tmp") / f"magsafe_lerobot_hf_datasets_cache_{os.getpid()}"
os.environ.setdefault("HF_DATASETS_CACHE", str(_PROCESS_HF_CACHE))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_SOURCE = Path("/home/jbnu/aloha_g1_dataset/raw_recordings")
DEFAULT_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/lerobot_magsafe_50_cam_high_v3")
DEFAULT_REPO_ID = "local/magsafe_aloha_50_cam_high_v3"
DEFAULT_CAMERA = "observation.images.cam_high"
DEFAULT_TASK = (
    "Remove the MagSafe accessory from the phone and place the phone on the MagSafe charger."
)
DEFAULT_REPORT = Path(
    "/home/jbnu/aloha_g1_dataset/reports/magsafe_lerobot_v3_build_report.json"
)
DEFAULT_MANIFEST = Path(
    "/home/jbnu/aloha_g1_dataset/reports/magsafe_lerobot_v3_manifest.csv"
)
JOINT_NAMES = [
    *(f"left_joint_{i}" for i in range(7)),
    *(f"right_joint_{i}" for i in range(7)),
]
EXPECTED_EPISODES = 50
EXPECTED_IMAGE_SHAPE = (480, 640, 3)
LOG = logging.getLogger("build_magsafe_lerobot_v3")
_LAST_NUMBER = re.compile(r"(\d+)(?!.*\d)")


class ValidationError(RuntimeError):
    """Input or output failed a required validation."""


@dataclass
class SourceEpisode:
    output_episode_index: int
    source_folder: str
    source_parquet: str
    source_frame_count: int
    cam_high_png_count: int
    source_first_timestamp: float
    source_last_timestamp: float
    first_frame_index: int
    last_frame_index: int
    image_paths: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--camera-key", default=DEFAULT_CAMERA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def fail(source: Path, reason: str) -> ValidationError:
    return ValidationError(f"{source}: {reason}")


def natural_frame_number(path: Path) -> int:
    match = _LAST_NUMBER.search(path.stem)
    if match is None:
        raise fail(path, "PNG filename contains no numeric frame index")
    return int(match.group(1))


def fixed_float32_14(field: pa.Field) -> bool:
    return (
        pa.types.is_fixed_size_list(field.type)
        and field.type.list_size == 14
        and pa.types.is_float32(field.type.value_type)
    )


def read_vectors(table: pa.Table, key: str, source: Path) -> np.ndarray:
    try:
        values = table[key].combine_chunks().values.to_numpy(zero_copy_only=False)
    except Exception as exc:
        raise fail(source, f"cannot read {key}: {exc}") from exc
    array = np.asarray(values, dtype=np.float32).reshape(len(table), 14)
    if not np.isfinite(array).all():
        bad = np.argwhere(~np.isfinite(array))[0].tolist()
        raise fail(source, f"{key} contains NaN/inf at row={bad[0]}, element={bad[1]}")
    return array


def feature_contract(info: dict[str, Any], source: Path, camera_key: str) -> None:
    features = info.get("features")
    if not isinstance(features, dict):
        raise fail(source, "meta/info.json has no features object")
    for key in ("observation.state", "action"):
        ft = features.get(key)
        if not isinstance(ft, dict):
            raise fail(source, f"missing feature {key}")
        if ft.get("dtype") != "float32" or ft.get("shape") != [14]:
            raise fail(source, f"{key} feature is not float32 shape [14]: {ft}")
        if ft.get("names") != JOINT_NAMES:
            raise fail(source, f"{key} joint names/order mismatch")
    camera = features.get(camera_key)
    if not isinstance(camera, dict) or camera.get("shape") != [480, 640, 3]:
        raise fail(source, f"{camera_key} metadata shape is not [480, 640, 3]")


def validate_sources(source_root: Path, camera_key: str) -> tuple[list[SourceEpisode], int, str]:
    if not source_root.is_dir():
        raise ValidationError(f"{source_root}: source root is not a directory")
    folders = sorted(p for p in source_root.glob("GoPark_*") if p.is_dir())
    if len(folders) != EXPECTED_EPISODES:
        raise ValidationError(
            f"{source_root}: expected {EXPECTED_EPISODES} GoPark_* directories, found {len(folders)}"
        )

    records: list[SourceEpisode] = []
    fps_values: set[int] = set()
    robot_types: set[str] = set()
    reference_contract: tuple[Any, ...] | None = None

    for episode_index, folder in enumerate(folders):
        parquets = sorted(folder.glob("data/**/episode_*.parquet"))
        if len(parquets) != 1:
            raise fail(folder, f"expected exactly 1 parquet, found {len(parquets)}")
        parquet = parquets[0]
        info_path = folder / "meta" / "info.json"
        if not info_path.is_file():
            raise fail(folder, "missing meta/info.json")
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise fail(folder, f"invalid meta/info.json: {exc}") from exc

        robot_type = info.get("robot_type")
        fps = info.get("fps")
        if not isinstance(robot_type, str) or not robot_type:
            raise fail(folder, "robot_type is missing")
        if not isinstance(fps, int) or fps <= 0:
            raise fail(folder, f"invalid fps: {fps!r}")
        robot_types.add(robot_type)
        fps_values.add(fps)
        feature_contract(info, folder, camera_key)
        contract = (
            tuple(info["features"]["observation.state"]["names"]),
            tuple(info["features"]["action"]["names"]),
            info["features"]["observation.state"]["dtype"],
            tuple(info["features"]["observation.state"]["shape"]),
            info["features"]["action"]["dtype"],
            tuple(info["features"]["action"]["shape"]),
        )
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise fail(folder, "state/action feature contract differs from earlier sources")

        pf = pq.ParquetFile(parquet)
        row_count = pf.metadata.num_rows
        if row_count < 1:
            raise fail(folder, "parquet has zero rows")
        schema = pf.schema_arrow
        for key in ("observation.state", "action"):
            try:
                field = schema.field(key)
            except KeyError as exc:
                raise fail(folder, f"parquet missing {key}") from exc
            if not fixed_float32_14(field):
                raise fail(folder, f"parquet {key} is not fixed-size float32 list[14]: {field.type}")

        table = pq.read_table(
            parquet,
            columns=["observation.state", "action", "timestamp", "frame_index"],
        )
        read_vectors(table, "observation.state", folder)
        read_vectors(table, "action", folder)
        timestamps = np.asarray(table["timestamp"].combine_chunks().to_numpy(), dtype=np.float64)
        frame_indices = np.asarray(table["frame_index"].combine_chunks().to_numpy(), dtype=np.int64)
        if not np.isfinite(timestamps).all():
            raise fail(folder, "timestamp contains NaN/inf")
        if row_count > 1:
            if not np.all(np.diff(timestamps) > 0):
                raise fail(folder, "timestamp is not strictly increasing")
            if not np.all(np.diff(frame_indices) > 0):
                raise fail(folder, "frame_index is not strictly increasing")
            expected = 1.0 / fps
            if not np.allclose(np.diff(timestamps), expected, atol=1e-3, rtol=0):
                worst = float(np.max(np.abs(np.diff(timestamps) - expected)))
                raise fail(folder, f"timestamp cadence differs from {fps} FPS (max error {worst})")

        image_dirs = [p for p in (folder / "images" / camera_key).glob("episode_*") if p.is_dir()]
        if len(image_dirs) != 1:
            raise fail(folder, f"expected exactly 1 {camera_key} PNG directory, found {len(image_dirs)}")
        pngs = list(image_dirs[0].glob("*.png"))
        numbered = [(natural_frame_number(p), p) for p in pngs]
        numbered.sort(key=lambda item: (item[0], item[1].name))
        numbers = [item[0] for item in numbered]
        if len(numbers) != len(set(numbers)):
            raise fail(folder, "duplicate numeric PNG frame index")
        if numbers and numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise fail(folder, f"missing PNG frame index in range {numbers[0]}..{numbers[-1]}")
        if len(numbered) != row_count:
            raise fail(folder, f"PNG count {len(numbered)} != parquet rows {row_count}")
        if numbers != frame_indices.tolist():
            raise fail(folder, "PNG numeric frame indices do not exactly match parquet frame_index")
        for _, image_path in numbered:
            try:
                with Image.open(image_path) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != (640, 480):
                        raise fail(
                            folder,
                            f"{image_path.name} is mode={image.mode}, size={image.size}; expected RGB 640x480",
                        )
            except ValidationError:
                raise
            except Exception as exc:
                raise fail(folder, f"cannot open PNG {image_path}: {exc}") from exc

        records.append(
            SourceEpisode(
                output_episode_index=episode_index,
                source_folder=str(folder),
                source_parquet=str(parquet),
                source_frame_count=row_count,
                cam_high_png_count=len(numbered),
                source_first_timestamp=float(timestamps[0]),
                source_last_timestamp=float(timestamps[-1]),
                first_frame_index=int(frame_indices[0]),
                last_frame_index=int(frame_indices[-1]),
                image_paths=[str(item[1]) for item in numbered],
            )
        )
        LOG.info("valid source %02d: %s (%d frames)", episode_index, folder.name, row_count)

    if len(fps_values) != 1:
        raise ValidationError(f"source FPS values conflict: {sorted(fps_values)}")
    if len(robot_types) != 1:
        raise ValidationError(f"source robot_type values conflict: {sorted(robot_types)}")
    fps = next(iter(fps_values))
    if fps != 30:
        raise ValidationError(f"expected 30 FPS, found {fps}")
    return records, fps, next(iter(robot_types))


def output_features(camera_key: str) -> dict[str, dict[str, Any]]:
    return {
        camera_key: {
            "dtype": "video",
            # LeRobot 0.4.0 validate_frame compares numpy.shape (a tuple)
            # directly with this value. JSON serialization still emits arrays.
            "shape": EXPECTED_IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": JOINT_NAMES,
        },
        "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
    }


def build_dataset(
    records: list[SourceEpisode],
    temp_root: Path,
    repo_id: str,
    task: str,
    camera_key: str,
    fps: int,
    robot_type: str,
) -> None:
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=output_features(camera_key),
        root=temp_root,
        robot_type=robot_type,
        use_videos=True,
    )
    for record in records:
        table = pq.read_table(
            record.source_parquet,
            columns=["observation.state", "action"],
        )
        states = read_vectors(table, "observation.state", Path(record.source_folder))
        actions = read_vectors(table, "action", Path(record.source_folder))
        for row, image_name in enumerate(record.image_paths):
            with Image.open(image_name) as image:
                image_array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            dataset.add_frame(
                {
                    camera_key: image_array,
                    "observation.state": states[row].copy(),
                    "action": actions[row].copy(),
                    "task": task,
                }
            )
        dataset.save_episode()
        LOG.info("saved output episode %02d (%d frames)", record.output_episode_index, len(table))
    dataset.finalize()
    LOG.info("dataset.finalize() completed")


def tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def ffprobe_json(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def rate_value(text: str) -> float:
    numerator, denominator = text.split("/")
    return float(numerator) / float(denominator)


def validate_output(
    root: Path,
    repo_id: str,
    records: list[SourceEpisode],
    task: str,
    camera_key: str,
    fps: int,
) -> dict[str, Any]:
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    expected_frames = sum(record.source_frame_count for record in records)
    meta = dataset.meta
    if dataset.num_episodes != 50 or meta.total_episodes != 50:
        raise fail(root, f"output episodes={meta.total_episodes}, expected 50")
    if len(dataset) != expected_frames or meta.total_frames != expected_frames:
        raise fail(root, f"output frames={meta.total_frames}, expected {expected_frames}")
    if meta.total_tasks != 1:
        raise fail(root, f"output tasks={meta.total_tasks}, expected 1")
    for key in ("observation.state", "action"):
        ft = dataset.features[key]
        if (
            ft["dtype"] != "float32"
            or tuple(ft["shape"]) != (14,)
            or ft["names"] != JOINT_NAMES
        ):
            raise fail(root, f"invalid output feature {key}: {ft}")
    camera_ft = dataset.features[camera_key]
    if camera_ft["dtype"] != "video" or tuple(camera_ft["shape"]) != (480, 640, 3):
        raise fail(root, f"invalid output camera feature: {camera_ft}")
    task_names = list(meta.tasks.index)
    if task_names != [task]:
        raise fail(root, f"output tasks mismatch: {task_names!r}")

    starts: list[int] = []
    cursor = 0
    for record in records:
        starts.append(cursor)
        cursor += record.source_frame_count

    decoded_boundary_count = 0
    for episode_index, record in enumerate(records):
        for local_index in sorted({0, record.source_frame_count - 1}):
            item = dataset[starts[episode_index] + local_index]
            image = tensor_numpy(item[camera_key])
            if image.shape != (3, 480, 640):
                raise fail(root, f"episode {episode_index} frame {local_index} decoded shape={image.shape}")
            if not np.isfinite(image).all() or float(image.min()) < 0 or float(image.max()) > 1:
                raise fail(root, f"episode {episode_index} frame {local_index} decoded range invalid")
            if item["task"] != task:
                raise fail(root, f"episode {episode_index} frame {local_index} task mismatch")
            decoded_boundary_count += 1

    comparisons: list[dict[str, Any]] = []
    for episode_index in (0, 24, 49):
        record = records[episode_index]
        source = pq.read_table(
            record.source_parquet, columns=["observation.state", "action"]
        )
        states = read_vectors(source, "observation.state", Path(record.source_folder))
        actions = read_vectors(source, "action", Path(record.source_folder))
        points = sorted({0, record.source_frame_count // 2, record.source_frame_count - 1})
        for local_index in points:
            item = dataset[starts[episode_index] + local_index]
            state_error = float(
                np.max(np.abs(tensor_numpy(item["observation.state"]) - states[local_index]))
            )
            action_error = float(np.max(np.abs(tensor_numpy(item["action"]) - actions[local_index])))
            if state_error > 1e-6 or action_error > 1e-6:
                raise fail(
                    root,
                    f"episode {episode_index} frame {local_index} value mismatch "
                    f"(state={state_error}, action={action_error})",
                )
            comparisons.append(
                {
                    "episode": episode_index,
                    "local_frame": local_index,
                    "state_max_abs_error": state_error,
                    "action_max_abs_error": action_error,
                }
            )

    data_shards = sorted(root.glob("data/chunk-*/file-*.parquet"))
    video_shards = sorted(root.glob(f"videos/{camera_key}/chunk-*/file-*.mp4"))
    if not data_shards or not video_shards:
        raise fail(root, "missing official data or video shards")
    probed_frames = 0
    probe_results: list[dict[str, Any]] = []
    for video in video_shards:
        probe = ffprobe_json(video)
        streams = probe.get("streams", [])
        if len(streams) != 1 or streams[0].get("codec_type") != "video":
            raise fail(video, "ffprobe found no unique video stream")
        stream = streams[0]
        if (stream.get("width"), stream.get("height")) != (640, 480):
            raise fail(video, f"ffprobe resolution is {stream.get('width')}x{stream.get('height')}")
        measured_fps = rate_value(stream.get("avg_frame_rate") or stream["r_frame_rate"])
        if not math.isclose(measured_fps, fps, abs_tol=1e-6):
            raise fail(video, f"ffprobe FPS={measured_fps}, expected {fps}")
        count_text = stream.get("nb_read_frames") or stream.get("nb_frames")
        if count_text in (None, "N/A"):
            raise fail(video, "ffprobe could not count frames")
        count = int(count_text)
        probed_frames += count
        probe_results.append({"path": str(video.relative_to(root)), "frames": count})
    if probed_frames != expected_frames:
        raise fail(root, f"ffprobe total video frames={probed_frames}, expected {expected_frames}")

    return {
        "dataset_load_success": True,
        "boundary_frames_decoded": decoded_boundary_count,
        "state_action_comparisons": comparisons,
        "ffprobe": probe_results,
        "ffprobe_total_frames": probed_frames,
        "data_shards": [str(path.relative_to(root)) for path in data_shards],
        "video_shards": [str(path.relative_to(root)) for path in video_shards],
    }


def write_manifest(path: Path, records: list[SourceEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "output_episode_index",
        "source_folder",
        "source_parquet",
        "source_frame_count",
        "cam_high_png_count",
        "source_first_timestamp",
        "source_last_timestamp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            writer.writerow({key: row[key] for key in fields})


def lerobot_version() -> str:
    import lerobot

    return str(getattr(lerobot, "__version__", "unknown"))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        print("OUTPUT_ALREADY_EXISTS")
        return 2
    if args.camera_key != DEFAULT_CAMERA:
        LOG.warning("non-default camera key requested: %s", args.camera_key)

    records, fps, robot_type = validate_sources(source_root, args.camera_key)
    total_frames = sum(record.source_frame_count for record in records)
    LOG.info(
        "source validation passed: %d valid / 0 invalid, %d total frames, fps=%d, robot_type=%s",
        len(records),
        total_frames,
        fps,
        robot_type,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": "success",
                    "valid_episodes": len(records),
                    "invalid_episodes": 0,
                    "total_frames": total_frames,
                    "fps": fps,
                    "robot_type": robot_type,
                },
                indent=2,
            )
        )
        return 0

    temp_root = output_root.parent / f".{output_root.name}.tmp_{os.getpid()}"
    if temp_root.exists():
        raise ValidationError(f"{temp_root}: temporary path already exists")
    renamed = False
    try:
        build_dataset(
            records, temp_root, args.repo_id, args.task, args.camera_key, fps, robot_type
        )
        temp_validation = validate_output(
            temp_root, args.repo_id, records, args.task, args.camera_key, fps
        )
        if output_root.exists():
            print("OUTPUT_ALREADY_EXISTS")
            raise ValidationError(f"{output_root}: appeared during build")
        temp_root.rename(output_root)
        renamed = True
        final_validation = validate_output(
            output_root, args.repo_id, records, args.task, args.camera_key, fps
        )
        write_manifest(args.manifest.resolve(), records)
        report = {
            "execution_time": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "python_executable": sys.executable,
            "lerobot_version": lerobot_version(),
            "source_root": str(source_root),
            "output_root": str(output_root),
            "repo_id": args.repo_id,
            "camera_key": args.camera_key,
            "task_string": args.task,
            "source_folder_count": len(records),
            "valid_episode_count": len(records),
            "total_frame_count": total_frames,
            "fps": fps,
            "robot_type": robot_type,
            "state_dimension": 14,
            "action_dimension": 14,
            "image_shape": list(EXPECTED_IMAGE_SHAPE),
            "source_frame_counts": [
                {
                    "source_folder": Path(record.source_folder).name,
                    "frames": record.source_frame_count,
                }
                for record in records
            ],
            "output_features": output_features(args.camera_key),
            "output_data_shards": final_validation["data_shards"],
            "output_video_shards": final_validation["video_shards"],
            "final_validation": final_validation,
            "pre_rename_validation": temp_validation,
            "errors": [],
            "warnings": [],
        }
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"build": "success", **report}, indent=2))
        return 0
    except Exception:
        if not renamed and temp_root.exists():
            shutil.rmtree(temp_root)
            LOG.info("removed this run's temporary directory: %s", temp_root)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        LOG.error("%s", exc)
        raise SystemExit(1) from exc
    finally:
        if (
            os.environ.get("HF_DATASETS_CACHE") == str(_PROCESS_HF_CACHE)
            and _PROCESS_HF_CACHE.exists()
        ):
            shutil.rmtree(_PROCESS_HF_CACHE)
