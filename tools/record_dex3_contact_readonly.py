#!/usr/bin/env python3
"""Subscribe to the two authoritative Dex3 state topics and record raw fields only."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOPICS = {"left": "rt/lf/dex3/left/state", "right": "rt/lf/dex3/right/state"}
JOINT_NAMES = {
    side: [f"{side}_hand_{name}_joint" for name in (
        "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"
    )]
    for side in ("left", "right")
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_message(message: Any) -> dict[str, np.ndarray]:
    motors = list(message.motor_state)
    sensors = list(message.press_sensor_state)
    if len(motors) != 7:
        raise RuntimeError(f"Dex3 HandState has {len(motors)} motors, expected 7")
    if len(sensors) != 9:
        raise RuntimeError(f"Dex3 HandState has {len(sensors)} pressure sensor records, expected 9")
    pressure = np.asarray([sensor.pressure for sensor in sensors], dtype=np.float32)
    temperature = np.asarray([sensor.temperature for sensor in sensors], dtype=np.float32)
    if pressure.shape != (9, 12) or temperature.shape != (9, 12):
        raise RuntimeError(f"raw pressure/temperature shape changed: {pressure.shape}/{temperature.shape}")
    result = {
        "q": np.asarray([motor.q for motor in motors], dtype=np.float32),
        "dq": np.asarray([motor.dq for motor in motors], dtype=np.float32),
        "ddq": np.asarray([motor.ddq for motor in motors], dtype=np.float32),
        "tau_est": np.asarray([motor.tau_est for motor in motors], dtype=np.float32),
        "motor_mode": np.asarray([motor.mode for motor in motors], dtype=np.uint8),
        "motor_temperature_raw": np.asarray([motor.temperature for motor in motors], dtype=np.int16),
        "press_raw": pressure,
        "press_temperature_raw": temperature,
        "press_lost_raw": np.asarray([sensor.lost for sensor in sensors], dtype=np.uint32),
        "press_reserve_raw": np.asarray([sensor.reserve for sensor in sensors], dtype=np.uint32),
    }
    if not all(np.isfinite(value).all() for key, value in result.items() if value.dtype.kind == "f"):
        raise RuntimeError("Dex3 HandState contains NaN/Inf")
    return result


class ReadOnlySubscribers:
    """Exactly two state subscribers. This class has no write-side dependency."""

    def __init__(self, interface: str):
        sdk = Path("/home/jbnu/jaeyoung/unitree/unitree_sdk2_python")
        if str(sdk) not in sys.path:
            sys.path.insert(0, str(sdk))
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_

        ChannelFactoryInitialize(0, interface)
        self.lock = threading.Lock()
        self.rows: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
        self.errors: list[str] = []
        self.start_ns = time.monotonic_ns()
        self.subscribers = []
        for side in ("left", "right"):
            subscriber = ChannelSubscriber(TOPICS[side], HandState_)
            subscriber.Init(lambda message, selected=side: self._receive(selected, message), 100)
            self.subscribers.append(subscriber)

    def _receive(self, side: str, message: Any) -> None:
        monotonic_ns = time.monotonic_ns()
        wall_ns = time.time_ns()
        try:
            parsed = parse_message(copy.deepcopy(message))
            with self.lock:
                sequence = len(self.rows[side])
                self.rows[side].append(
                    {
                        "receive_sequence": sequence,
                        "host_monotonic_timestamp_ns": monotonic_ns,
                        "host_wall_timestamp_ns": wall_ns,
                        **parsed,
                    }
                )
        except Exception as error:
            with self.lock:
                self.errors.append(f"{side}: {type(error).__name__}: {error}")

    def snapshot(self) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
        with self.lock:
            return copy.deepcopy(self.rows), list(self.errors)


def synthetic_rows(duration: float, rate: float) -> dict[str, list[dict[str, Any]]]:
    count = max(2, int(round(duration * rate)))
    result = {"left": [], "right": []}
    for side_index, side in enumerate(("left", "right")):
        for index in range(count):
            base = 100.0 + side_index * 8.0 + np.arange(9)[:, None] * 2.0 + np.arange(12)[None, :] * 0.1
            delta = 4.0 * np.sin(index * 0.06 + np.arange(9)[:, None] * 0.2)
            result[side].append(
                {
                    "receive_sequence": index,
                    "host_monotonic_timestamp_ns": 1_000_000_000 + int(index / rate * 1e9),
                    "host_wall_timestamp_ns": 2_000_000_000 + int(index / rate * 1e9),
                    "q": np.sin(index * 0.01 + np.arange(7)) * 0.1,
                    "dq": np.cos(index * 0.01 + np.arange(7)) * 0.001,
                    "ddq": np.zeros(7),
                    "tau_est": np.zeros(7),
                    "motor_mode": np.zeros(7, dtype=np.uint8),
                    "motor_temperature_raw": np.zeros((7, 2), dtype=np.int16),
                    "press_raw": (base + delta).astype(np.float32),
                    "press_temperature_raw": np.full((9, 12), 25.0, dtype=np.float32),
                    "press_lost_raw": np.zeros(9, dtype=np.uint32),
                    "press_reserve_raw": np.zeros(9, dtype=np.uint32),
                }
            )
    return result


def stack(rows: list[dict[str, Any]], key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    value = np.asarray([row[key] for row in rows], dtype=dtype)
    if value.shape != (len(rows), *shape):
        raise RuntimeError(f"{key} stack has shape {value.shape}, expected {(len(rows), *shape)}")
    return value


def timing(rows: list[dict[str, Any]], target_rate: float) -> dict[str, Any]:
    timestamps = stack(rows, "host_monotonic_timestamp_ns", (), np.int64)
    gaps = np.diff(timestamps) / 1e9
    duration = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0.0
    rate = (len(timestamps) - 1) / duration if duration > 0 else 0.0
    estimated_missing = int(np.sum(np.maximum(np.rint(gaps * target_rate).astype(int) - 1, 0)))
    return {
        "received_messages": len(rows),
        "duration_s": duration,
        "receive_rate_hz": rate,
        "maximum_interarrival_gap_s": float(gaps.max(initial=0.0)),
        "estimated_dropout_count_from_interarrival_gaps": estimated_missing,
        "transport_sequence_available": False,
        "dropout_is_estimate": True,
    }


def save(output: Path, rows: dict[str, list[dict[str, Any]]], errors: list[str], target_rate: float, synthetic: bool, interface: str) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite read-only recording: {output}")
    if any(len(rows[side]) < 2 for side in ("left", "right")):
        raise RuntimeError("both authoritative state topics must provide at least two messages")
    arrays: dict[str, np.ndarray] = {}
    per_hand = {}
    fields = {
        "q": ((7,), np.float32), "dq": ((7,), np.float32), "ddq": ((7,), np.float32),
        "tau_est": ((7,), np.float32), "motor_mode": ((7,), np.uint8),
        "motor_temperature_raw": ((7, 2), np.int16), "press_raw": ((9, 12), np.float32),
        "press_temperature_raw": ((9, 12), np.float32), "press_lost_raw": ((9,), np.uint32),
        "press_reserve_raw": ((9,), np.uint32), "receive_sequence": ((), np.int64),
        "host_monotonic_timestamp_ns": ((), np.int64), "host_wall_timestamp_ns": ((), np.int64),
    }
    for side in ("left", "right"):
        for key, (shape, dtype) in fields.items():
            arrays[f"{side}_{key}"] = stack(rows[side], key, shape, dtype)
        per_hand[side] = timing(rows[side], target_rate)
    output.parent.mkdir(parents=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, output)
    manifest = {
        "schema_version": "dex3_contact_readonly_recording_v1",
        "status": "PASS_SYNTHETIC_READONLY" if synthetic else "PASS_REAL_READONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recording": str(output),
        "recording_sha256": sha256_file(output),
        "synthetic": synthetic,
        "network_interface": interface,
        "host": socket.gethostname(),
        "topics": TOPICS,
        "message_type": "unitree_hg.msg.dds_.HandState_",
        "joint_names": JOINT_NAMES,
        "motor_fields": ["q", "dq", "ddq", "tau_est", "mode", "temperature_raw"],
        "press_sensor_records_per_hand": 9,
        "raw_values_per_press_sensor_record": 12,
        "press_fields": ["pressure_raw", "temperature_raw", "lost_raw", "reserve_raw"],
        "pressure_units": "RAW_DEVICE_VALUES_FORCE_UNITS_NOT_ASSIGNED",
        "per_hand_timing": per_hand,
        "callback_errors": errors,
        "read_only": True,
        "command_path": "ABSENT",
        "mode_switch": "ABSENT",
        "real_hand_motion_requested": False,
        "official_sdk2_reference": {
            "commit": "f29ee9f234851e9e79f75102c0f9e83008d8fdd1",
            "example": "example/g1/dex3/g1_dex3_example.cpp",
        },
    }
    atomic_json(output.with_suffix(".manifest.json"), manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-interface", default="lo")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--expected-rate", type=float, default=100.0)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0 or args.expected_rate <= 0:
        parser.error("duration and expected rate must be positive")
    if args.simulate:
        rows = synthetic_rows(args.duration, args.expected_rate)
        errors: list[str] = []
    else:
        reader = ReadOnlySubscribers(args.network_interface)
        time.sleep(args.duration)
        rows, errors = reader.snapshot()
    save(args.output.resolve(), rows, errors, args.expected_rate, args.simulate, args.network_interface)
    print(json.dumps({"status": "PASS_SYNTHETIC_READONLY" if args.simulate else "PASS_REAL_READONLY", "recording": str(args.output.resolve()), "received": {side: len(rows[side]) for side in rows}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
