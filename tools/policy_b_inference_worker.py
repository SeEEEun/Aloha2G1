#!/usr/bin/env python3
"""Persistent local-only Policy-A/Policy-B SmolVLA inference worker.

The worker exists solely to keep the validated LeRobot/Torch environment
separate from Isaac Lab's Torch build.  It listens on a Unix-domain socket,
has no network or robot transport code, and exposes only RGB+state+task to
action-chunk inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from multiprocessing.connection import Listener
from pathlib import Path
import random
import signal
import time
from typing import Any

import numpy as np
import torch


EXPECTED_RGB_SHAPE = (480, 640, 3)
EXPECTED_STATE_SHAPE = (28,)
EXPECTED_ACTION_SHAPE = (1, 50, 28)
AUTHKEY = b"policy-b-isaac-local-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--causal-stitching-method",
        choices=("naive", "rtc", "crossfade"),
        default="naive",
        help="Model-side causal plan stitching. Raw unconditioned chunks are always returned separately.",
    )
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--rtc-guidance-weight", type=float, default=5.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=("exp", "linear"),
        default="exp",
    )
    parser.add_argument(
        "--fixed-flow-noise",
        action="store_true",
        help=(
            "Reuse one seed-derived flow-noise tensor for every inference. This is a "
            "diagnostic sampling mode; every resulting raw chunk remains logged."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    """Persist a NumPy array without exposing a partially written artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def main() -> int:
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.rtc import RTCConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    socket_path = args.socket.resolve()
    ready_path = args.ready.resolve()
    if socket_path.exists():
        socket_path.unlink()
    if ready_path.exists():
        ready_path.unlink()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("SmolVLA policy worker requires CUDA")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config["input_features"]["observation.state"]["shape"] != [28]:
        raise RuntimeError("policy worker expected a 28D state interface")
    if config["output_features"]["action"]["shape"] != [28]:
        raise RuntimeError("policy worker expected a 28D action interface")
    if int(config["chunk_size"]) != 50 or int(config["n_action_steps"]) != 50:
        raise RuntimeError("policy worker expected a 50-row action chunk")
    load_started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    policy.eval()
    if not 1 <= args.execution_horizon < EXPECTED_ACTION_SHAPE[1]:
        raise RuntimeError("execution horizon must be in 1..49")
    rtc_config = None
    if args.causal_stitching_method == "rtc":
        schedule = {
            "exp": RTCAttentionSchedule.EXP,
            "linear": RTCAttentionSchedule.LINEAR,
        }[args.rtc_prefix_attention_schedule]
        rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=args.execution_horizon,
            max_guidance_weight=args.rtc_guidance_weight,
            prefix_attention_schedule=schedule,
            debug=False,
        )
        policy.config.rtc_config = rtc_config
        policy.init_rtc_processor()
    load_seconds = time.perf_counter() - load_started
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    listener = Listener(str(socket_path), family="AF_UNIX", authkey=AUTHKEY)
    fixed_noise: torch.Tensor | None = None
    fixed_noise_record: dict[str, Any] | None = None
    if args.fixed_flow_noise:
        fixed_noise = policy.model.sample_noise(
            (1, 50, int(policy.config.max_action_dim)), next(policy.parameters()).device
        ).detach().clone()
        fixed_noise_array = fixed_noise.float().cpu().numpy()
        fixed_noise_path = ready_path.parent / "episode_persistent_flow_noise.npy"
        atomic_npy(fixed_noise_path, fixed_noise_array)
        fixed_noise_record = {
            "path": str(fixed_noise_path),
            "seed": int(args.seed),
            "shape": list(fixed_noise_array.shape),
            "dtype": str(fixed_noise_array.dtype),
            "sha256": hashlib.sha256(fixed_noise_array.tobytes()).hexdigest(),
            "file_sha256": sha256_file(fixed_noise_path),
            "lifetime": "EPISODE_PERSISTENT_REUSED_FOR_EVERY_INFERENCE",
            "best_action_mode_guaranteed": False,
        }
    ready = {
        "status": "READY",
        "pid": os.getpid(),
        "checkpoint": str(checkpoint),
        "model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "load_seconds": load_seconds,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "interface": "RGB_UINT8_480x640x3 + STATE_FLOAT32_28 + TASK -> ACTION_FLOAT32_1x50x28",
        "transport": "LOCAL_UNIX_DOMAIN_SOCKET_ONLY",
        "real_robot_transport_available": False,
        "causal_stitching": {
            "method": args.causal_stitching_method,
            "model_side_rtc_enabled": rtc_config is not None,
            "runner_side_crossfade_expected": args.causal_stitching_method == "crossfade",
            "execution_horizon": args.execution_horizon,
            "rtc_guidance_weight": args.rtc_guidance_weight if rtc_config else None,
            "rtc_prefix_attention_schedule": args.rtc_prefix_attention_schedule if rtc_config else None,
            "simulation_inference_delay_frames": "REQUEST_SUPPLIED_PER_INFERENCE",
            "simulation_inference_delay_reason": (
                "Latency-aware RTC audits pass the measured causal queue delay explicitly; "
                "non-RTC calls use zero."
            ),
            "future_observations_used": False,
            "raw_unconditioned_chunk_retained": True,
            "flow_noise_mode": (
                "FIXED_SEED_DERIVED_TENSOR_REUSED"
                if args.fixed_flow_noise
                else "FRESH_SEQUENTIAL_SEED_STREAM"
            ),
            "episode_persistent_flow_noise": fixed_noise_record,
        },
    }
    atomic_json(ready_path, ready)
    connection = listener.accept()
    request_index = 0
    previous_stitched_normalized: torch.Tensor | None = None
    previous_stitched_action: np.ndarray | None = None
    try:
        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "shutdown":
                connection.send({"status": "SHUTDOWN_ACK"})
                break
            if command != "infer":
                connection.send({"status": "ERROR", "error": f"unknown command {command!r}"})
                continue
            started = time.perf_counter()
            rgb = np.asarray(request["rgb"])
            state = np.asarray(request["state"], dtype=np.float32)
            task = request["task"]
            inference_delay = int(request.get("inference_delay", 0))
            if rgb.shape != EXPECTED_RGB_SHAPE or rgb.dtype != np.uint8:
                raise RuntimeError(f"RGB must be uint8 {EXPECTED_RGB_SHAPE}; got {rgb.dtype} {rgb.shape}")
            if state.shape != EXPECTED_STATE_SHAPE or not np.isfinite(state).all():
                raise RuntimeError(f"state must be finite {EXPECTED_STATE_SHAPE}; got {state.shape}")
            if not isinstance(task, str) or not task.strip():
                raise RuntimeError("task must be one non-empty string")
            if not 0 <= inference_delay < EXPECTED_ACTION_SHAPE[1]:
                raise RuntimeError(
                    "inference_delay must be an integer number of action rows in 0..49"
                )
            image_tensor = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .contiguous()
                .to(dtype=torch.float32)
                .div_(255.0)
            )
            raw = {
                "observation.images.cam_high": image_tensor,
                "observation.state": torch.from_numpy(state.copy()),
                "task": task,
            }
            processed = preprocessor(raw)
            action_device = processed["observation.state"].device
            if args.fixed_flow_noise:
                if fixed_noise is None:
                    raise RuntimeError("episode-persistent flow noise was not initialized")
                noise = fixed_noise.clone()
            else:
                noise = policy.model.sample_noise(
                    (1, 50, int(policy.config.max_action_dim)), action_device
                )
            previous_remaining_plan = (
                previous_stitched_action[args.execution_horizon :].copy()
                if previous_stitched_action is not None
                else np.empty((0, 28), dtype=np.float32)
            )
            previous_remaining_normalized = (
                previous_stitched_normalized[:, args.execution_horizon :].clone()
                if previous_stitched_normalized is not None
                else None
            )
            torch.cuda.synchronize()
            raw_inference_started = time.perf_counter()
            # RTC needs a local enable_grad block in its denoiser, so no_grad is
            # intentional here; torch.inference_mode would prohibit that block.
            with torch.no_grad():
                raw_normalized = policy.predict_action_chunk(
                    processed,
                    noise=noise,
                    inference_delay=0,
                    prev_chunk_left_over=None,
                    execution_horizon=args.execution_horizon,
                )
                raw_prediction = postprocessor(raw_normalized.clone())
            torch.cuda.synchronize()
            raw_inference_seconds = time.perf_counter() - raw_inference_started
            rtc_inference_seconds = 0.0
            if rtc_config is not None and previous_remaining_normalized is not None:
                torch.cuda.synchronize()
                rtc_inference_started = time.perf_counter()
                with torch.no_grad():
                    stitched_normalized = policy.predict_action_chunk(
                        processed,
                        noise=noise,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=previous_remaining_normalized,
                        execution_horizon=args.execution_horizon,
                    )
                    stitched_prediction = postprocessor(stitched_normalized.clone())
                torch.cuda.synchronize()
                rtc_inference_seconds = time.perf_counter() - rtc_inference_started
            else:
                stitched_normalized = raw_normalized.clone()
                stitched_prediction = raw_prediction.clone()
            action = raw_prediction.detach().float().cpu().numpy()
            stitched_action = stitched_prediction.detach().float().cpu().numpy()
            if action.shape != EXPECTED_ACTION_SHAPE:
                raise RuntimeError(f"policy output shape {action.shape}, expected {EXPECTED_ACTION_SHAPE}")
            if stitched_action.shape != EXPECTED_ACTION_SHAPE:
                raise RuntimeError(
                    f"stitched output shape {stitched_action.shape}, expected {EXPECTED_ACTION_SHAPE}"
                )
            if not np.isfinite(action).all() or not np.isfinite(stitched_action).all():
                raise RuntimeError("raw or stitched policy output contains NaN/Inf")
            previous_stitched_normalized = stitched_normalized.detach().clone()
            previous_stitched_action = stitched_action[0].copy()
            inference_seconds = raw_inference_seconds + rtc_inference_seconds
            connection.send(
                {
                    "status": "PASS",
                    "request_index": request_index,
                    "action": action,
                    "stitched_action": stitched_action,
                    "previous_remaining_plan": previous_remaining_plan,
                    "action_shape": list(action.shape),
                    "action_minimum": float(action.min()),
                    "action_maximum": float(action.max()),
                    "inference_seconds": inference_seconds,
                    "raw_inference_seconds": raw_inference_seconds,
                    "rtc_inference_seconds": rtc_inference_seconds,
                    "rtc_inference_delay_frames": (
                        inference_delay if rtc_config is not None else 0
                    ),
                    "request_total_seconds": time.perf_counter() - started,
                    "input_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "input_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                    "prediction_finite": bool(np.isfinite(action).all()),
                    "stitched_prediction_finite": bool(np.isfinite(stitched_action).all()),
                    "causal_stitching_method": args.causal_stitching_method,
                    "previous_remaining_plan_rows": int(len(previous_remaining_plan)),
                    "raw_and_stitched_identical": bool(np.array_equal(action, stitched_action)),
                    "prediction_tensor_after_internal_padding_slicing": [1, 50, 28],
                    "flow_noise_mode": (
                        "FIXED_SEED_DERIVED_TENSOR_REUSED"
                        if args.fixed_flow_noise
                        else "FRESH_SEQUENTIAL_SEED_STREAM"
                    ),
                    "flow_noise_sha256": hashlib.sha256(
                        noise.detach().float().cpu().numpy().tobytes()
                    ).hexdigest(),
                }
            )
            request_index += 1
    finally:
        connection.close()
        listener.close()
        if socket_path.exists():
            socket_path.unlink()
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
