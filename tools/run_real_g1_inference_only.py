#!/usr/bin/env python3
"""D455 + measured named state + Policy A/B -> raw 50x28 logs; never commands G1."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from deployment_camera_config import camera_manifest_record, load_camera_config, sha256_file  # noqa: E402
from helmet_d455.realsense import start_pipeline  # noqa: E402
from helmet_d455.parent_pose import ParentLinkPoseAdapter  # noqa: E402


ARM_INDEX = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
STATE_TOPICS = {
    "g1": "rt/lowstate",
    "left": "rt/lf/dex3/left/state",
    "right": "rt/lf/dex3/right/state",
}
AUTHKEY = b"policy-b-isaac-local-v1"
POLICY_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
PREFLIGHT_PYTHON = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class D455InputAdapter:
    def __init__(self, camera: Any):
        raw = camera.raw
        color = raw["streams"]["color"]
        depth = raw["streams"]["depth"]
        self.active = start_pipeline(
            serial=str(raw["device"]["serial"]),
            color=(int(color["width_px"]), int(color["height_px"]), int(color["fps"])),
            depth=(int(depth["width_px"]), int(depth["height_px"]), int(depth["fps"])),
        )

    def read(self) -> tuple[np.ndarray, dict[str, Any]]:
        frames = self.active.pipeline.wait_for_frames(5000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("D455 color frame unavailable")
        rgb = np.asanyarray(color.get_data()).copy()
        return rgb, {
            "device_timestamp_ms": float(color.get_timestamp()),
            "device_frame_number": int(color.get_frame_number()),
            "timestamp_domain": str(color.get_frame_timestamp_domain()),
            "host_monotonic_timestamp_ns": time.monotonic_ns(),
            "host_wall_timestamp_ns": time.time_ns(),
        }

    def close(self) -> None:
        self.active.stop()


class NamedG1StateAdapter:
    """Three state subscribers and no write-side object."""

    def __init__(self, interface: str):
        sdk = Path("/home/jbnu/jaeyoung/unitree/unitree_sdk2_python")
        sys.path.insert(0, str(sdk))
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

        ChannelFactoryInitialize(0, interface)
        self.lock = threading.Lock()
        self.latest: dict[str, tuple[int, int, Any] | None] = {key: None for key in STATE_TOPICS}
        types = {"g1": LowState_, "left": HandState_, "right": HandState_}
        self.subscribers = []
        for key in ("g1", "left", "right"):
            subscriber = ChannelSubscriber(STATE_TOPICS[key], types[key])
            subscriber.Init(lambda message, selected=key: self._receive(selected, message), 10)
            self.subscribers.append(subscriber)

    def _receive(self, key: str, message: Any) -> None:
        with self.lock:
            self.latest[key] = (time.monotonic_ns(), time.time_ns(), copy.deepcopy(message))

    def read(self, timeout: float = 0.2) -> tuple[np.ndarray, dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                latest = dict(self.latest)
            if all(value is not None for value in latest.values()):
                now = time.monotonic_ns()
                if all(now - value[0] <= int(timeout * 1e9) for value in latest.values() if value):
                    g1 = latest["g1"][2]
                    left = latest["left"][2]
                    right = latest["right"][2]
                    if len(g1.motor_state) < 29 or len(left.motor_state) != 7 or len(right.motor_state) != 7:
                        raise RuntimeError("G1/Dex3 state message dimensions changed")
                    state = np.asarray(
                        [g1.motor_state[index].q for index in ARM_INDEX]
                        + [motor.q for motor in left.motor_state]
                        + [motor.q for motor in right.motor_state],
                        dtype=np.float32,
                    )
                    return state, {
                        "topic_receive_monotonic_ns": {key: int(value[0]) for key, value in latest.items() if value},
                        "topic_receive_wall_ns": {key: int(value[1]) for key, value in latest.items() if value},
                        "topics": STATE_TOPICS,
                    }
            time.sleep(0.002)
        raise TimeoutError("fresh named G1/Dex3 state_28d was not received")


class SimulationInputs:
    def __init__(self, names: list[str]):
        self.names = names
        self.index = 0

    def read(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[..., 0] = np.arange(640, dtype=np.uint8)[None, :]
        image[..., 1] = 40
        cv2.putText(image, f"INFERENCE ONLY {self.index}", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        state = np.sin(np.arange(28) * 0.1 + self.index * 0.03).astype(np.float32) * 0.05
        stamp = time.monotonic_ns()
        self.index += 1
        return image, state, {"host_monotonic_timestamp_ns": stamp, "synthetic": True}, {"host_monotonic_timestamp_ns": stamp, "synthetic": True}


class PolicyProcess:
    def __init__(self, checkpoint: Path, output: Path):
        token = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
        self.socket = Path(f"/tmp/g1inf_{os.getpid()}_{token}.sock")
        self.ready = output / "policy_worker_ready.json"
        self.log_stream = (output / "policy_worker.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(POLICY_PYTHON), str(ROOT / "tools/policy_b_inference_worker.py"), "--checkpoint", str(checkpoint), "--socket", str(self.socket), "--ready", str(self.ready)],
            cwd=ROOT, stdout=self.log_stream, stderr=subprocess.STDOUT, text=True,
        )
        deadline = time.monotonic() + 90.0
        while not self.ready.is_file():
            if self.process.poll() is not None:
                raise RuntimeError(f"policy worker exited; inspect {output / 'policy_worker.log'}")
            if time.monotonic() >= deadline:
                self.process.terminate()
                raise TimeoutError("policy worker readiness timeout")
            time.sleep(0.1)
        self.connection = Client(str(self.socket), family="AF_UNIX", authkey=AUTHKEY)

    def infer(self, rgb: np.ndarray, state: np.ndarray, task: str) -> tuple[np.ndarray, dict[str, Any]]:
        self.connection.send({"command": "infer", "rgb": rgb, "state": state, "task": task})
        response = self.connection.recv()
        if response.get("status") != "PASS":
            raise RuntimeError(f"policy inference failed: {response}")
        action = np.asarray(response.pop("action"), dtype=np.float32)
        return action[0], response

    def close(self) -> None:
        self.connection.send({"command": "shutdown"})
        self.connection.recv()
        self.connection.close()
        self.process.wait(timeout=30)
        self.log_stream.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("A", "B"), required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-interface", default="lo")
    parser.add_argument("--locked-parent-pose-id")
    parser.add_argument(
        "--parent-fk-json",
        type=Path,
        help="read-only timestamp-aligned task_from_parent bridge for a movable camera parent",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--task", default="Pick up the doll with the left hand, handoff it to the right hand, and place it in the trash bin.")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--mock-policy", action="store_true")
    args = parser.parse_args()
    if args.mock_policy and not args.simulation:
        parser.error("--mock-policy is restricted to --simulation")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite inference log: {args.output}")
    args.output.mkdir(parents=True)
    safety_path = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    names = list(safety["joint_names"])
    if args.simulation:
        camera = load_camera_config(args.camera_config, purpose="inference simulation dry-run", allow_pending=True)
        simulated = SimulationInputs(names)
        d455 = state_adapter = None
        parent_pose = None
    else:
        camera = load_camera_config(args.camera_config, purpose=f"real-G1 Policy {args.policy} inference")
        parent_pose = ParentLinkPoseAdapter(
            camera,
            locked_parent_pose_id=args.locked_parent_pose_id,
            fk_json=args.parent_fk_json,
        )
        d455 = D455InputAdapter(camera)
        state_adapter = NamedG1StateAdapter(args.network_interface)
        simulated = None
    defaults = json.loads((ROOT / "configs/real_g1_inference_only.json").read_text())
    checkpoint = (args.checkpoint or ROOT / defaults["policy_checkpoints"][args.policy]).resolve()
    if not args.mock_policy and not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Policy {args.policy} checkpoint is unavailable: {checkpoint}")
    policy = None if args.mock_policy else PolicyProcess(checkpoint, args.output)
    chunks, states, camera_times, state_times, parent_pose_rows, inference_rows = [], [], [], [], [], []
    try:
        for index in range(args.iterations):
            if simulated is not None:
                rgb, state, camera_time, state_time = simulated.read()
            else:
                rgb, camera_time = d455.read()
                state, state_time = state_adapter.read()
                parent_pose_rows.append(
                    parent_pose.read(camera_time["host_monotonic_timestamp_ns"])
                )
            if rgb.shape != (480, 640, 3) or state.shape != (28,) or not np.isfinite(state).all():
                raise RuntimeError("malformed RGB/state input")
            if policy is None:
                chunk = np.repeat(state[None, :], 50, axis=0)
                inference = {"status": "MOCK_POLICY_SIMULATION_ONLY", "request_index": index}
            else:
                chunk, inference = policy.infer(rgb, state, args.task)
            if chunk.shape != (50, 28) or not np.isfinite(chunk).all():
                raise RuntimeError("policy returned malformed raw action chunk")
            image_path = args.output / "rgb" / f"frame_{index:06d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            chunks.append(chunk)
            states.append(state)
            camera_times.append(camera_time)
            state_times.append(state_time)
            inference_rows.append({"index": index, "completed_monotonic_ns": time.monotonic_ns(), "rgb": str(image_path), **inference})
    finally:
        if policy is not None:
            policy.close()
        if d455 is not None:
            d455.close()
    arrays = args.output / "inference_log.npz"
    with arrays.with_suffix(".npz.incomplete").open("wb") as stream:
        np.savez_compressed(
            stream,
            raw_action_chunks_50x28=np.asarray(chunks, dtype=np.float32),
            measured_state_28d=np.asarray(states, dtype=np.float32),
            joint_names=np.asarray(names, dtype="U64"),
        )
    os.replace(arrays.with_suffix(".npz.incomplete"), arrays)
    preflight = args.output / "raw_action_preflight.json"
    subprocess.run(
        [str(PREFLIGHT_PYTHON), str(ROOT / "tools/audit_g1_inference_action_chunks.py"), "--chunks", str(arrays), "--output", str(preflight)],
        cwd=ROOT, check=True,
    )
    manifest = {
        "schema_version": "real_g1_inference_only_log_v1",
        "status": "INFERENCE_ONLY_NO_COMMAND_PUBLISHER",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulation": args.simulation,
        "mock_policy": args.mock_policy,
        "policy_variant": args.policy,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": None if args.mock_policy else sha256_file(checkpoint / "model.safetensors"),
        "dataset_normalization": "loaded from checkpoint saved preprocessor/postprocessor" if not args.mock_policy else "not exercised by mock",
        "camera": camera_manifest_record(camera),
        "state_joint_names": names,
        "state_topics": STATE_TOPICS,
        "iterations": len(chunks),
        "raw_action_shape_per_iteration": [50, 28],
        "log": str(arrays),
        "log_sha256": sha256_file(arrays),
        "camera_timestamps": camera_times,
        "state_timestamps": state_times,
        "camera_parent_pose": (
            parent_pose_rows
            if not args.simulation
            else [{"mode": "SIMULATION_PENDING_CAMERA_NOT_OPERATIONAL"} for _ in chunks]
        ),
        "inference_timestamps": inference_rows,
        "action_limit_and_collision_preflight": str(preflight),
        "command_publisher": "ABSENT",
        "robot_command_client": "ABSENT",
        "local_policy_ipc": "authenticated Unix-domain socket; inference/shutdown messages only",
        "mode_switch": "ABSENT",
        "robot_motion": "NOT_REQUESTED",
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "policy": args.policy, "iterations": len(chunks), "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
