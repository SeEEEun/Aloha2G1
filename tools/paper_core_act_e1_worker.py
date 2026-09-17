#!/usr/bin/env python3
"""GPU inference worker for paired source-video-conditioned ACT rollouts.

Every request performs one raw 50x28 ACT prediction.  The command is produced
by the installed official ``ACTTemporalEnsembler.update`` with coefficient
0.01, then the official ACT postprocessor.  Raw and ensembled outputs are
returned separately.  There is no controller, smoothing, language, DDS, or
real-hardware transport in this process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import numpy as np
import torch


AUTHKEY = b"paper-core-act-ab-source-v1"
TEMPORAL_ENSEMBLE_COEFF = 0.01
CHUNK_SIZE = 50
ACTION_DIM = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from lerobot.policies.factory import make_pre_post_processors

    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    model_path = checkpoint / "model.safetensors"
    actual_sha = sha256_file(model_path)
    if actual_sha != args.checkpoint_sha256:
        raise RuntimeError(f"checkpoint hash mismatch: {actual_sha}")
    base_policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    if (
        base_policy.config.chunk_size != CHUNK_SIZE
        or base_policy.config.n_action_steps != CHUNK_SIZE
        or base_policy.config.temporal_ensemble_coeff is not None
        or base_policy.config.action_feature.shape != (ACTION_DIM,)
    ):
        raise RuntimeError("selected paper ACT checkpoint violates the common raw-policy contract")

    e1_config = copy.deepcopy(base_policy.config)
    e1_config.n_action_steps = 1
    e1_config.temporal_ensemble_coeff = TEMPORAL_ENSEMBLE_COEFF
    policy = ACTPolicy(e1_config)
    loaded = policy.load_state_dict(base_policy.state_dict(), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"strict E1 weight reload failed: {loaded}")
    policy.to(next(base_policy.parameters()).device)
    del base_policy
    torch.cuda.empty_cache()
    if not isinstance(policy.temporal_ensembler, ACTTemporalEnsembler):
        raise RuntimeError("installed official ACTTemporalEnsembler was not constructed")
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    dropout_active = [
        name
        for name, module in policy.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.training
    ]
    if dropout_active:
        raise RuntimeError(f"dropout active in eval worker: {dropout_active}")

    socket_path = args.socket.resolve()
    if socket_path.exists() or socket_path.is_socket():
        socket_path.unlink()
    ready = {
        "schema_version": "paper_core_act_e1_worker_v1",
        "status": "READY",
        "pid": os.getpid(),
        "method": args.method.upper(),
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": actual_sha,
        "execution": "ACT_E1_TEMPORAL_ENSEMBLE",
        "implementation": "installed official ACTPolicy.predict_action_chunk + ACTTemporalEnsembler.update + official postprocessor",
        "implementation_equivalence": "the two operations inside official ACTPolicy.select_action when temporal_ensemble_coeff is non-null; invoked directly once to preserve the exact raw query without duplicate inference",
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": 1,
        "temporal_ensemble_coeff": TEMPORAL_ENSEMBLE_COEFF,
        "temporal_ensembler_class": "ACTTemporalEnsembler",
        "custom_averaging": False,
        "raw_policy_query_preserved": True,
        "raw_policy_query_shape": [CHUNK_SIZE, ACTION_DIM],
        "controlled_inputs": ["source ALOHA cam_high RGB", "measured Isaac 28D state"],
        "language_input": False,
        "dropout_modules_active": dropout_active,
        "real_hardware_transport": False,
    }
    listener = Listener(str(socket_path), family="AF_UNIX", authkey=AUTHKEY)
    atomic_json(args.ready.resolve(), ready)
    connection = listener.accept()
    call_index = 0
    try:
        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "shutdown":
                connection.send({"status": "PASS", "calls": call_index})
                break
            if command == "reset":
                policy.reset()
                call_index = 0
                connection.send({"status": "PASS"})
                continue
            if command != "infer":
                connection.send({"status": "FAIL", "error": f"unknown command {command!r}"})
                continue
            rgb = np.asarray(request["rgb"], dtype=np.uint8)
            state = np.asarray(request["state"], dtype=np.float32)
            if rgb.shape != (480, 640, 3) or state.shape != (ACTION_DIM,):
                connection.send(
                    {"status": "FAIL", "error": f"input shapes rgb={rgb.shape} state={state.shape}"}
                )
                continue
            if not np.isfinite(state).all():
                connection.send({"status": "FAIL", "error": "non-finite state"})
                continue
            image = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .contiguous()
                .float()
                .div_(255.0)
            )
            processed = preprocessor(
                {
                    "observation.images.cam_high": image,
                    "observation.state": torch.from_numpy(state.copy()),
                }
            )
            start = time.perf_counter()
            with torch.inference_mode():
                normalized_chunk = policy.predict_action_chunk(processed)
                # This is exactly the official temporal branch of
                # ACTPolicy.select_action, without a second deterministic query.
                normalized_action = policy.temporal_ensembler.update(normalized_chunk)
                physical_chunk = postprocessor(normalized_chunk.clone())
                physical_action = postprocessor(normalized_action.clone())
            elapsed = time.perf_counter() - start
            chunk = normalized_chunk[0].detach().float().cpu().numpy().astype(np.float32)
            raw_chunk = physical_chunk[0].detach().float().cpu().numpy().astype(np.float32)
            normalized = normalized_action[0].detach().float().cpu().numpy().astype(np.float32)
            action = physical_action[0].detach().float().cpu().numpy().astype(np.float32)
            if (
                chunk.shape != (CHUNK_SIZE, ACTION_DIM)
                or raw_chunk.shape != (CHUNK_SIZE, ACTION_DIM)
                or normalized.shape != (ACTION_DIM,)
                or action.shape != (ACTION_DIM,)
                or not all(np.isfinite(value).all() for value in (chunk, raw_chunk, normalized, action))
            ):
                connection.send({"status": "FAIL", "error": "malformed/non-finite ACT result"})
                continue
            connection.send(
                {
                    "status": "PASS",
                    "call_index": call_index,
                    "normalized_chunk": chunk,
                    "raw_chunk": raw_chunk,
                    "normalized_action": normalized,
                    "raw_ensembled_action": action,
                    "inference_seconds": elapsed,
                    "rgb_sha256": sha256_array(rgb),
                    "state_sha256": sha256_array(state),
                    "raw_chunk_sha256": sha256_array(raw_chunk),
                    "raw_ensembled_action_sha256": sha256_array(action),
                }
            )
            call_index += 1
    finally:
        connection.close()
        listener.close()
        if socket_path.exists() or socket_path.is_socket():
            socket_path.unlink()


if __name__ == "__main__":
    main()
