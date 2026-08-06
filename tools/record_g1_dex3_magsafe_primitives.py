#!/usr/bin/env python3
"""Read-only Unitree G1 + Dex3 predefined grasp primitive recorder.

This module deliberately imports no command messages, publishers, or command
clients.  It subscribes only to the verified Unitree state topics.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL_OUTPUT = ROOT / "configs/dex3_magsafe_grasp_primitives.real.json"
DEFAULT_SIM_OUTPUT = ROOT / "configs/dex3_magsafe_grasp_primitives.sim.json"
PRIMITIVES = (
    "LEFT_PHONE_OPEN", "LEFT_PHONE_PREGRASP", "LEFT_PHONE_GRASP",
    "LEFT_PHONE_HOLD", "LEFT_PHONE_RELEASE", "RIGHT_ACCESSORY_OPEN",
    "RIGHT_ACCESSORY_PREGRASP", "RIGHT_ACCESSORY_GRASP",
    "RIGHT_ACCESSORY_HOLD", "RIGHT_ACCESSORY_RELEASE",
)

# Motor order is the order used by Unitree's g1_dex3_example.cpp.  Names and
# limits are the corresponding joints in the supplied g1_with_hands.xml.
DEX3_NAMES = {
    "left": [f"left_hand_{n}_joint" for n in (
        "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")],
    "right": [f"right_hand_{n}_joint" for n in (
        "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")],
}
DEX3_LIMITS = {
    "left": [[-1.0472, 1.0472], [-0.724312, 1.0472], [0.0, 1.74533], [-1.5708, 0.0],
             [-1.74533, 0.0], [-1.5708, 0.0], [-1.74533, 0.0]],
    "right": [[-1.0472, 1.0472], [-1.0472, 0.724312], [-1.74533, 0.0], [0.0, 1.5708],
              [0.0, 1.74533], [0.0, 1.5708], [0.0, 1.74533]],
}
ARM_INDEX_NAME = [
    (15, "left_shoulder_pitch_joint"), (16, "left_shoulder_roll_joint"),
    (17, "left_shoulder_yaw_joint"), (18, "left_elbow_joint"),
    (19, "left_wrist_roll_joint"), (20, "left_wrist_pitch_joint"),
    (21, "left_wrist_yaw_joint"), (22, "right_shoulder_pitch_joint"),
    (23, "right_shoulder_roll_joint"), (24, "right_shoulder_yaw_joint"),
    (25, "right_elbow_joint"), (26, "right_wrist_roll_joint"),
    (27, "right_wrist_pitch_joint"), (28, "right_wrist_yaw_joint"),
]
HAND_TOPICS = {"left": "rt/lf/dex3/left/state", "right": "rt/lf/dex3/right/state"}
LOWSTATE_TOPIC = "rt/lowstate"
STD_WARNING_RAD = 0.02


def banner() -> None:
    print("\n" + "=" * 70)
    print("READ-ONLY PRIMITIVE RECORDER")
    print("THIS PROGRAM DOES NOT COMMAND THE ROBOT")
    print("=" * 70 + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_document(interface: str, simulation: bool = False) -> dict[str, Any]:
    return {
        "source": "simulation_placeholder" if simulation else "real_robot_recording",
        "authoritative_for_real_robot": not simulation,
        "real_robot_command_allowed": not simulation,
        "generated_by": str(Path(__file__).resolve()), "generated_at": utc_now(),
        "schema_version": 1, "robot": "Unitree G1", "hand": "Dex3", "units": "radian",
        "joint_names": {"left_dex3": DEX3_NAMES["left"], "right_dex3": DEX3_NAMES["right"]},
        "joint_limits": {"left_dex3": DEX3_LIMITS["left"], "right_dex3": DEX3_LIMITS["right"]},
        "arm_joint_names": [name for _, name in ARM_INDEX_NAME],
        "roles": {"left": "phone", "right": "magsafe_accessory"},
        "capture": {
            "authoritative_data": "Dex3 hand qpos only",
            "full_arm_snapshot_usage": "diagnostic only; never an arm trajectory",
            "source_machine": socket.gethostname(), "platform": platform.platform(),
            "network_interface": interface,
            "state_topics": {"g1": LOWSTATE_TOPIC, **HAND_TOPICS},
            "state_source_provenance": [
                "/home/jbnu/jaeyoung/unitree/unitree_sdk2/example/g1/dex3/g1_dex3_example.cpp",
                "/home/jbnu/jaeyoung/unitree/unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py",
                "/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml",
            ],
        },
        "primitives": {name: {} for name in PRIMITIVES},
    }


class ReadOnlyStateReader:
    """The only live-robot object: three DDS subscribers and no writers."""
    def __init__(self, interface: str):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_
        except ImportError as exc:
            raise RuntimeError("unitree_sdk2py is required for live recording") from exc
        ChannelFactoryInitialize(0, interface)
        self._lock = threading.Lock()
        self._states: dict[str, tuple[float, Any] | None] = {"left": None, "right": None, "g1": None}
        self._subs = [
            ChannelSubscriber(HAND_TOPICS["left"], HandState_),
            ChannelSubscriber(HAND_TOPICS["right"], HandState_),
            ChannelSubscriber(LOWSTATE_TOPIC, LowState_),
        ]
        self._subs[0].Init(lambda msg: self._put("left", msg), 10)
        self._subs[1].Init(lambda msg: self._put("right", msg), 10)
        self._subs[2].Init(lambda msg: self._put("g1", msg), 10)

    def _put(self, key: str, msg: Any) -> None:
        with self._lock:
            self._states[key] = (time.monotonic(), copy.deepcopy(msg))

    def snapshot(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                states = dict(self._states)
            now = time.monotonic()
            if all(v is not None and now - v[0] <= timeout for v in states.values()):
                left, right, g1 = states["left"][1], states["right"][1], states["g1"][1]
                if len(left.motor_state) < 7 or len(right.motor_state) < 7 or len(g1.motor_state) < 29:
                    raise RuntimeError("state message has fewer motors than expected")
                return {
                    "left": [float(left.motor_state[i].q) for i in range(7)],
                    "right": [float(right.motor_state[i].q) for i in range(7)],
                    "arm": [float(g1.motor_state[i].q) for i, _ in ARM_INDEX_NAME],
                    "robot_state": {
                        "mode_machine": int(getattr(g1, "mode_machine", 0)),
                        "mode_pr": int(getattr(g1, "mode_pr", 0)),
                    },
                }
            time.sleep(0.005)
        raise TimeoutError(f"fresh state not received within {timeout:.3f}s")


class DryRunStateReader:
    def __init__(self):
        self._primitive_index = 0
        self._sample_index = 0

    def begin(self, primitive: str) -> None:
        self._primitive_index = PRIMITIVES.index(primitive)
        self._sample_index = 0

    def snapshot(self, timeout: float) -> dict[str, Any]:
        del timeout
        side = "left" if self._primitive_index < 5 else "right"
        stage = self._primitive_index % 5
        ratio = [0.05, 0.30, 0.72, 0.70, 0.12][stage]
        out = {}
        for hand in ("left", "right"):
            lo, hi = np.asarray(DEX3_LIMITS[hand], float).T
            r = ratio if hand == side else 0.08
            phase = 0.0008 * np.sin(self._sample_index * 0.31 + np.arange(7))
            out[hand] = (lo + (hi - lo) * r + phase).tolist()
        out["arm"] = (np.array([0.2, 0.2, 0, 1.28, 0, 0, 0,
                                0.2, -0.2, 0, 1.28, 0, 0, 0]) +
                      0.0002 * math.sin(self._sample_index * 0.2)).tolist()
        out["robot_state"] = {"mode_machine": 0, "mode_pr": 0, "dry_run": True}
        self._sample_index += 1
        return out


def capture(reader: Any, primitive: str, duration: float, timeout: float,
            interface: str, note: str) -> dict[str, Any]:
    if hasattr(reader, "begin"):
        reader.begin(primitive)
    samples, start = [], time.monotonic()
    period = 0.01
    while time.monotonic() - start < duration:
        tick = time.monotonic()
        samples.append(reader.snapshot(timeout))
        time.sleep(max(0.0, period - (time.monotonic() - tick)))
    discard = min(len(samples), int(math.ceil(0.5 / period)))
    kept = samples[discard:]
    if not kept:
        raise RuntimeError("no samples remain after the first 0.5 seconds were discarded")
    side = "left" if primitive.startswith("LEFT_") else "right"
    hand = np.asarray([s[side] for s in kept], dtype=float)
    arm = np.asarray([s["arm"] for s in kept], dtype=float)
    if not np.isfinite(hand).all() or not np.isfinite(arm).all():
        raise ValueError("NaN/Inf detected; capture refused")
    median, mean, std = np.median(hand, axis=0), hand.mean(axis=0), hand.std(axis=0)
    lo, hi = np.asarray(DEX3_LIMITS[side], dtype=float).T
    margin = np.minimum(median - lo, hi - median)
    if np.any(margin < 0):
        raise ValueError(f"joint limit violation; margins={margin.tolist()}")
    if float(std.max()) > STD_WARNING_RAD:
        answer = input(f"WARNING: max joint std={std.max():.6f} rad. Save capture? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            raise RuntimeError("capture rejected after high-variance warning")
    wrist_ids = list(range(4, 7)) if side == "left" else list(range(11, 14))
    return {
        "authoritative_side": side, "joint_names": DEX3_NAMES[side],
        "qpos": median.tolist(), "sample_median": median.tolist(),
        "sample_mean": mean.tolist(), "sample_std": std.tolist(),
        "sample_min": hand.min(axis=0).tolist(), "sample_max": hand.max(axis=0).tolist(),
        "joint_limit_margin": margin.tolist(), "sample_count": len(kept),
        "total_sample_count": len(samples), "discarded_sample_count": discard,
        "sample_duration_s": duration, "sample_rate_target_hz": 100,
        "full_arm_snapshot": {
            "diagnostic_only": True, "joint_names": [n for _, n in ARM_INDEX_NAME],
            "qpos_median": np.median(arm, axis=0).tolist(),
            "sample_mean": arm.mean(axis=0).tolist(), "sample_std": arm.std(axis=0).tolist(),
            "sample_min": arm.min(axis=0).tolist(), "sample_max": arm.max(axis=0).tolist(),
        },
        "wrist_snapshot": {
            "joint_names": [[n for _, n in ARM_INDEX_NAME][i] for i in wrist_ids],
            "qpos_median": np.median(arm[:, wrist_ids], axis=0).tolist(),
        },
        "robot_mode_state": samples[-1]["robot_state"], "captured_at": utc_now(),
        "timestamp": time.time(), "source_machine": socket.gethostname(),
        "network_interface": interface, "operator_note": note,
    }


def validate(doc: dict[str, Any], require_all: bool = True) -> list[str]:
    errors, notes = [], []
    if doc.get("joint_names", {}).get("left_dex3") != DEX3_NAMES["left"]:
        errors.append("left Dex3 joint names/order mismatch")
    if doc.get("joint_names", {}).get("right_dex3") != DEX3_NAMES["right"]:
        errors.append("right Dex3 joint names/order mismatch")
    primitives = doc.get("primitives", {})
    for name in PRIMITIVES:
        item = primitives.get(name)
        if not item:
            if require_all: errors.append(f"missing primitive: {name}")
            continue
        side = "left" if name.startswith("LEFT_") else "right"
        q = np.asarray(item.get("qpos", []), float)
        if item.get("authoritative_side") != side: errors.append(f"{name}: wrong authoritative side")
        if item.get("joint_names") != DEX3_NAMES[side]: errors.append(f"{name}: joint names/order mismatch")
        if q.shape != (7,): errors.append(f"{name}: qpos shape {q.shape}, expected (7,)"); continue
        if not np.isfinite(q).all(): errors.append(f"{name}: NaN/Inf")
        lo, hi = np.asarray(DEX3_LIMITS[side], float).T
        if np.any(q < lo) or np.any(q > hi): errors.append(f"{name}: joint-limit violation")
    for side, prefix in (("left", "LEFT_PHONE"), ("right", "RIGHT_ACCESSORY")):
        a, b = primitives.get(prefix + "_OPEN"), primitives.get(prefix + "_GRASP")
        if a and b and np.array_equal(np.asarray(a["qpos"]), np.asarray(b["qpos"])):
            errors.append(f"{side}: OPEN and GRASP are exactly identical")
        g, h = primitives.get(prefix + "_GRASP"), primitives.get(prefix + "_HOLD")
        if g and h:
            notes.append(f"{side} GRASP/HOLD max abs difference: "
                         f"{np.max(np.abs(np.asarray(g['qpos'])-np.asarray(h['qpos']))):.8f} rad")
    if notes:
        print("Validation notes:")
        for note in notes: print(f"  - {note}")
    return errors


def inspect(doc: dict[str, Any]) -> None:
    print(f"schema={doc.get('schema_version')} robot={doc.get('robot')} hand={doc.get('hand')}")
    for name in PRIMITIVES:
        item = doc.get("primitives", {}).get(name, {})
        if not item: print(f"  {name}: MISSING"); continue
        print(f"  {name}: n={item['sample_count']} max_std={max(item['sample_std']):.6f} "
              f"min_margin={min(item['joint_limit_margin']):.6f} qpos={item['qpos']}")
    errors = validate(doc, require_all=True)
    print("VALID" if not errors else "INVALID:\n  - " + "\n  - ".join(errors))


def load_or_new(path: Path, interface: str, simulation: bool = False) -> dict[str, Any]:
    if not path.exists(): return empty_document(interface, simulation)
    with path.open() as f: doc = json.load(f)
    return doc


def atomic_save(path: Path, doc: dict[str, Any]) -> None:
    errors = validate(doc, require_all=True)
    if errors: raise ValueError("save validation failed:\n  - " + "\n  - ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
        backup.write_bytes(path.read_bytes())
        print(f"Existing file backed up to {backup}")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f: json.dump(doc, f, indent=2, allow_nan=False); f.write("\n")
    os.replace(temp, path)
    print(f"Saved {path}")


def assert_real_robot_primitive_config(doc: dict[str, Any]) -> None:
    """Fail-closed gate for any existing/future real robot backend."""
    if (doc.get("source") == "simulation_placeholder"
            or not doc.get("authoritative_for_real_robot", False)
            or not doc.get("real_robot_command_allowed", False)):
        raise RuntimeError("SIMULATION PRIMITIVES CANNOT BE SENT TO THE REAL ROBOT")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network-interface", default="eth0")
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--record", nargs="*", choices=(*PRIMITIVES, "ALL"), metavar="PRIMITIVE")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--timeout", type=float, default=1.0)
    p.add_argument("--sample-duration", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args(); banner()
    path = (args.output or (DEFAULT_SIM_OUTPUT if args.dry_run else DEFAULT_REAL_OUTPUT)).expanduser().resolve()
    if args.dry_run and path.name.endswith(".real.json"):
        raise ValueError("dry-run output must not use the .real.json suffix")
    if args.sample_duration <= 0.5: raise ValueError("--sample-duration must be greater than 0.5s")
    if args.timeout <= 0: raise ValueError("--timeout must be positive")
    if args.inspect:
        if not path.exists(): raise FileNotFoundError(path)
        with path.open() as f: inspect(json.load(f))
        return 0
    doc = load_or_new(path, args.network_interface, args.dry_run)
    reader = DryRunStateReader() if args.dry_run else ReadOnlyStateReader(args.network_interface)

    def record_one(name: str) -> None:
        print(f"\n{name}: use the existing teleop to pose the robot, then press Enter.")
        if not (args.dry_run and not sys.stdin.isatty()): input()
        note = "dry-run synthetic sample" if args.dry_run else input("Operator note (optional): ").strip()
        doc["primitives"][name] = capture(reader, name, args.sample_duration, args.timeout,
                                           args.network_interface, note)
        print(f"Recorded {name}")

    if args.record is not None:
        selected = list(PRIMITIVES) if not args.record or "ALL" in args.record else args.record
        for name in selected: record_one(name)
        atomic_save(path, doc)
        return 0
    while True:
        print("\n" + "\n".join(f"{i}. Record {n}" for i, n in enumerate(PRIMITIVES, 1)))
        print("11. Inspect recorded primitives\n12. Save and exit\nq. Abort without overwriting")
        choice = input("> ").strip().lower()
        if choice == "q": print("Aborted; output was not changed."); return 0
        if choice == "11": inspect(doc); continue
        if choice == "12": atomic_save(path, doc); return 0
        if choice.isdigit() and 1 <= int(choice) <= 10: record_one(PRIMITIVES[int(choice)-1]); continue
        print("Invalid selection")


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (RuntimeError, ValueError, TimeoutError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
