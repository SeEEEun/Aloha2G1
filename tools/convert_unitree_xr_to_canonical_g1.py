#!/usr/bin/env python3
"""Validate official Unitree XR recordings and create the canonical 28D dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from deployment_camera_config import load_camera_config, sha256_file  # noqa: E402


JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_index_0_joint", "right_hand_index_1_joint",
]
PARTS = ("left_arm", "right_arm", "left_ee", "right_ee")
PART_LENGTHS = (7, 7, 7, 7)
DEFAULT_TASK = "Pick up the doll with the left hand, handoff it to the right hand, and place it in the trash bin."


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _qpos(sample: dict[str, Any], stream: str) -> np.ndarray:
    values = []
    for part, length in zip(PARTS, PART_LENGTHS, strict=True):
        try:
            value = sample[stream][part]["qpos"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"missing {stream}.{part}.qpos") from error
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (length,) or not np.isfinite(array).all():
            raise ValueError(f"{stream}.{part}.qpos must be finite length {length}")
        values.append(array)
    return np.concatenate(values)


def _timestamp(sample: dict[str, Any], index: int, fps: float) -> tuple[int, str]:
    for key in ("host_monotonic_timestamp_ns", "timestamp_ns", "capture_timestamp_ns"):
        if key in sample and sample[key] is not None:
            return int(sample[key]), key
    if "timestamp" in sample and sample["timestamp"] is not None:
        return int(round(float(sample["timestamp"]) * 1e9)), "timestamp_seconds"
    return int(round(index / fps * 1e9)), "DERIVED_FROM_IDX_AND_RECORDED_FPS"


def validate_episode(path: Path, camera_key: str) -> tuple[dict[str, Any] | None, list[str]]:
    problems = []
    try:
        payload = read_json(path / "data.json")
    except Exception as error:
        return None, [f"data.json read failed: {type(error).__name__}: {error}"]
    fps = float(payload.get("info", {}).get("image", {}).get("fps", 0.0))
    if fps <= 0:
        problems.append("missing positive info.image.fps")
        fps = 30.0
    data = payload.get("data", [])
    if not data:
        problems.append("episode has no samples")
    states, actions, timestamps, timestamp_sources, image_paths = [], [], [], [], []
    previous = None
    for index, sample in enumerate(data):
        if int(sample.get("idx", -1)) != index:
            problems.append(f"sample {index}: idx is not contiguous")
        try:
            states.append(_qpos(sample, "states"))
            actions.append(_qpos(sample, "actions"))
        except ValueError as error:
            problems.append(f"sample {index}: {error}")
            continue
        relative = sample.get("colors", {}).get(camera_key)
        image = path / str(relative) if relative else None
        if image is None or not image.is_file():
            problems.append(f"sample {index}: missing final helmet RGB key {camera_key}")
        else:
            decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != (480, 640, 3):
                problems.append(f"sample {index}: RGB must decode as 640x480x3")
            image_paths.append(image)
        stamp, source = _timestamp(sample, index, fps)
        if previous is not None and stamp <= previous:
            problems.append(f"sample {index}: timestamp is not strictly increasing")
        previous = stamp
        timestamps.append(stamp)
        timestamp_sources.append(source)
    if problems:
        return None, problems
    task = str(payload.get("text", {}).get("goal") or DEFAULT_TASK)
    result = {
        "source": path,
        "fps": fps,
        "task": task,
        "state": np.stack(states),
        "action": np.stack(actions),
        "timestamp_ns": np.asarray(timestamps, dtype=np.int64),
        "timestamp_sources": timestamp_sources,
        "images": image_paths,
        "source_payload": payload,
    }
    return result, []


def make_simulation_fixture(root: Path, episodes: int = 3, frames: int = 12) -> None:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite simulation fixture: {root}")
    for episode in range(episodes):
        directory = root / "doll_handoff" / f"episode_{episode:04d}"
        colors = directory / "colors"
        colors.mkdir(parents=True)
        rows = []
        for frame in range(frames):
            phase = (episode * frames + frame) / 30.0
            state = np.sin(np.arange(28, dtype=np.float32) * 0.07 + phase) * 0.1
            action = state + 0.002
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            image[..., 0] = 25 + episode * 15
            image[..., 1] = np.arange(640, dtype=np.uint8)[None, :]
            image[..., 2] = 80 + frame * 4
            relative = Path("colors") / f"{frame:06d}_color_0.jpg"
            cv2.putText(image, f"XR SIM ep{episode} frame{frame}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(str(directory / relative), image)

            def split(value: np.ndarray) -> dict[str, Any]:
                return {
                    "left_arm": {"qpos": value[0:7].tolist()},
                    "right_arm": {"qpos": value[7:14].tolist()},
                    "left_ee": {"qpos": value[14:21].tolist()},
                    "right_ee": {"qpos": value[21:28].tolist()},
                }

            rows.append(
                {
                    "idx": frame,
                    "host_monotonic_timestamp_ns": 10_000_000_000 + (episode * frames + frame) * 33_333_333,
                    "colors": {"color_0": str(relative)},
                    "states": split(state),
                    "actions": split(action),
                }
            )
        atomic_json(
            directory / "data.json",
            {
                "info": {"version": "SIMULATION", "image": {"width": 640, "height": 480, "fps": 30}},
                "text": {"goal": DEFAULT_TASK},
                "data": rows,
            },
        )


def discover(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("data.json"))


def canonical_sidecars(
    episodes: list[dict[str, Any]], output: Path, camera_serial: str, calibration_hash: str, synthetic: bool
) -> list[dict[str, Any]]:
    directory = output / "canonical_episodes"
    directory.mkdir(parents=True)
    rows = []
    for index, episode in enumerate(episodes):
        path = directory / f"episode_{index:06d}.npz"
        with path.with_suffix(".npz.incomplete").open("wb") as stream:
            np.savez_compressed(
                stream,
                state_28d=episode["state"].astype(np.float32),
                action_28d=episode["action"].astype(np.float32),
                timestamp_ns=episode["timestamp_ns"],
                joint_names=np.asarray(JOINT_NAMES, dtype="U64"),
                task=np.asarray(episode["task"]),
            )
        os.replace(path.with_suffix(".npz.incomplete"), path)
        metadata = {
            "schema_version": "canonical_g1_xr_episode_v1",
            "episode_index": index,
            "official_xr_source": str(episode["source"]),
            "frames": len(episode["state"]),
            "fps": episode["fps"],
            "state_semantic": "MEASURED_G1_DEX3_28D",
            "action_semantic": "G1_DEX3_COMMAND_28D",
            "joint_names": JOINT_NAMES,
            "task": episode["task"],
            "camera_key": "observation.images.cam_high",
            "camera_serial": camera_serial,
            "camera_calibration_sha256": calibration_hash,
            "timestamp_sources": sorted(set(episode["timestamp_sources"])),
            "synthetic": synthetic,
            "npz": str(path),
            "npz_sha256": sha256_file(path),
        }
        metadata_path = path.with_suffix(".json")
        atomic_json(metadata_path, metadata)
        rows.append(metadata)
    return rows


def package_lerobot(episodes: list[dict[str, Any]], destination: Path, repo_id: str) -> dict[str, Any]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.import_utils import get_safe_default_video_backend
    except ImportError as error:
        raise RuntimeError("run this conversion with the lerobot-smolvla Python environment") from error
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite LeRobot XR dataset: {destination}")
    features = {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (28,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (28,), "names": JOINT_NAMES},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=destination,
        fps=30,
        robot_type="unitree_g1_dex3_xr_canonical",
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        video_backend=get_safe_default_video_backend(),
    )
    for episode in episodes:
        if not np.isclose(episode["fps"], 30.0):
            raise RuntimeError("current policy contract requires XR recordings at exactly 30 Hz")
        for frame in range(len(episode["state"])):
            bgr = cv2.imread(str(episode["images"][frame]), cv2.IMREAD_COLOR)
            dataset.add_frame(
                {
                    "observation.images.cam_high": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    "observation.state": episode["state"][frame],
                    "action": episode["action"][frame],
                    "task": episode["task"],
                }
            )
        dataset.save_episode()
    dataset.finalize()
    readback = LeRobotDataset(
        repo_id=repo_id,
        root=destination,
        download_videos=False,
        video_backend=get_safe_default_video_backend(),
    )
    reads = []
    offsets = np.cumsum([0] + [len(row["state"]) for row in episodes[:-1]])
    for index, (offset, episode) in enumerate(zip(offsets, episodes, strict=True)):
        item = readback[int(offset + len(episode["state"]) // 2)]
        checks = {
            "image": tuple(item["observation.images.cam_high"].shape) == (3, 480, 640),
            "state": tuple(item["observation.state"].shape) == (28,),
            "action": tuple(item["action"].shape) == (28,),
            "task": item["task"] == episode["task"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"XR LeRobot readback failed for episode {index}: {checks}")
        reads.append({"episode_index": index, "checks": checks})
    return {
        "status": "PASS",
        "dataset": str(destination),
        "episodes": len(episodes),
        "frames": len(readback),
        "readback": reads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/doll_handoff_xr_canonical")
    parser.add_argument("--camera-key", default="color_0")
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--camera-serial")
    parser.add_argument("--exclude-invalid", action="store_true")
    parser.add_argument("--reference-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260825)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.reference_fraction < 1.0:
        parser.error("--reference-fraction must be between zero and one")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite XR bridge output: {args.output}")
    args.output.mkdir(parents=True)
    if args.simulate:
        fixture = args.output / "official_xr_simulation_fixture"
        make_simulation_fixture(fixture)
        input_root = fixture
        camera = load_camera_config(args.camera_config, purpose="XR simulation dry-run", allow_pending=True)
        camera_serial = "SIMULATION_ONLY_NOT_A_CAMERA_SERIAL"
    else:
        if args.input is None or args.camera_serial is None:
            parser.error("physical-recording conversion requires --input and --camera-serial")
        input_root = args.input.resolve()
        camera = load_camera_config(args.camera_config, purpose="real XR conversion")
        camera_serial = args.camera_serial
        configured_serial = str(camera.raw["device"]["serial"])
        if configured_serial != camera_serial:
            raise RuntimeError(f"XR camera serial mismatch: {camera_serial} != {configured_serial}")
    calibration_hash = sha256_file(camera.path)

    valid = []
    invalid = []
    for path in discover(input_root):
        episode, problems = validate_episode(path, args.camera_key)
        if problems:
            invalid.append({"source": str(path), "problems": problems})
        else:
            valid.append(episode)
    exclusion = {
        "schema_version": "xr_bad_episode_exclusion_manifest_v1",
        "status": "NO_BAD_EPISODES" if not invalid else "BAD_EPISODES_PRESENT",
        "invalid_episode_count": len(invalid),
        "invalid_episodes": invalid,
        "exclusion_applied": bool(invalid and args.exclude_invalid),
        "silent_exclusion_count": 0,
    }
    atomic_json(args.output / "bad_episode_exclusion_manifest.json", exclusion)
    if invalid and not args.exclude_invalid:
        raise RuntimeError("invalid XR episodes found; review the exclusion manifest and rerun with --exclude-invalid")
    if not valid:
        raise RuntimeError("no structurally valid XR episodes")

    metadata = canonical_sidecars(valid, args.output, camera_serial, calibration_hash, args.simulate)
    indices = list(range(len(valid)))
    random.Random(args.split_seed).shuffle(indices)
    reference_count = max(1, round(len(indices) * args.reference_fraction)) if len(indices) > 1 else 0
    reference = sorted(indices[:reference_count])
    adaptation = sorted(indices[reference_count:])
    split = {
        "schema_version": "xr_adaptation_reference_split_v1",
        "seed": args.split_seed,
        "reference_fraction_requested": args.reference_fraction,
        "adaptation_episode_indices": adaptation,
        "reference_episode_indices": reference,
        "same_split_required_for_policy_a_and_policy_b": True,
    }
    atomic_json(args.output / "adaptation_reference_split.json", split)
    lerobot = package_lerobot(valid, args.dataset.resolve(), args.repo_id)
    manifest = {
        "schema_version": "unitree_xr_canonical_g1_bridge_v1",
        "status": "PASS_SIMULATION_DRY_RUN" if args.simulate else "PASS_REAL_RECORDING_CONVERSION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_reference_commits": {
            "xr_teleoperate": "845b25a32f7febedf220e830952a7134897adb9d",
            "unitree_lerobot": "41c2805742de879ddab2d8d6beaeaf215f876395",
        },
        "input": str(input_root),
        "camera_source_key": args.camera_key,
        "output_camera_key": "observation.images.cam_high",
        "camera_serial": camera_serial,
        "camera_calibration": str(camera.path),
        "camera_calibration_sha256": calibration_hash,
        "state": "measured G1/Dex3 28D",
        "action": "G1/Dex3 command 28D",
        "joint_names": JOINT_NAMES,
        "timestamps": "preserved when present, otherwise explicitly derived from idx/fps",
        "episodes": metadata,
        "bad_episode_exclusion_manifest": str(args.output / "bad_episode_exclusion_manifest.json"),
        "split_manifest": str(args.output / "adaptation_reference_split.json"),
        "lerobot": lerobot,
        "real_collection": "NOT_STARTED" if args.simulate else "IMPORTED_EXISTING_RECORDINGS",
    }
    atomic_json(args.output / "xr_bridge_manifest.json", manifest)
    atomic_json(args.dataset / "meta/xr_bridge_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "episodes": len(valid), "frames": lerobot["frames"], "dataset": str(args.dataset)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
