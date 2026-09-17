#!/usr/bin/env python3
"""Cross ACT-B frame-0 RGB and state captures to isolate branch sensitivity."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout/frame0_cross_input_isolation"
CHECKPOINT = ROOT / "outputs/policy_b_act/train/checkpoints/100000/pretrained_model"
EXPECTED_SHA = "a8d0a0d437421a1a0100f4f8c8ed67b5f6cd7ef0d383b1ebba0e6e45ada4d078"
CAPTURE_ROOT = ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    if sha256_file(CHECKPOINT / "model.safetensors") != EXPECTED_SHA:
        raise RuntimeError("ACT-B checkpoint changed")
    images = {
        "settled_render": cv2.cvtColor(
            cv2.imread(str(CAPTURE_ROOT / "frame0_common_visual/policy_input_frame0.png")),
            cv2.COLOR_BGR2RGB,
        ),
        "resnapped_render": cv2.cvtColor(
            cv2.imread(str(CAPTURE_ROOT / "frame0_common_visual_resnap/policy_input_frame0.png")),
            cv2.COLOR_BGR2RGB,
        ),
    }
    with np.load(CAPTURE_ROOT / "frame0_common_visual/frame0_arrays.npz", allow_pickle=False) as archive:
        states = {
            "settled_state": archive["isaac_inference_state"].astype(np.float32),
            "nominal_state": archive["nominal_common_initial"].astype(np.float32),
        }
        names = archive["joint_names"].astype(str)

    policy = ACTPolicy.from_pretrained(CHECKPOINT, local_files_only=True, strict=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(CHECKPOINT))
    records = []
    chunks = {}
    for image_label, rgb in images.items():
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        for state_label, state in states.items():
            processed = preprocessor(
                {
                    "observation.images.cam_high": image.clone(),
                    "observation.state": torch.from_numpy(state.copy()),
                }
            )
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(processed)
                physical = postprocessor(normalized.clone())
            chunk = physical[0].detach().float().cpu().numpy().astype(np.float32)
            label = f"{image_label}__{state_label}"
            chunks[label] = chunk
            delta = chunk[0, :14].astype(np.float64) - state[:14].astype(np.float64)
            nominal_delta = chunk[0, :14].astype(np.float64) - states["nominal_state"][:14].astype(np.float64)
            records.append(
                {
                    "label": label,
                    "image": image_label,
                    "state": state_label,
                    "rgb_sha256": sha256_array(rgb),
                    "state_sha256": sha256_array(state),
                    "chunk_sha256": sha256_array(chunk),
                    "chunk_shape": list(chunk.shape),
                    "finite": bool(np.isfinite(chunk).all()),
                    "first_arm_branch_norm_vs_controlled_state_rad": float(np.linalg.norm(delta)),
                    "first_arm_norm_vs_nominal_rad": float(np.linalg.norm(nominal_delta)),
                    "frozen_branch_gate_rad": 0.18,
                    "branch_pass": float(np.linalg.norm(delta)) <= 0.18,
                    "first_action_rad": chunk[0],
                }
            )
    np.savez_compressed(OUT / "cross_input_chunks.npz", joint_names=names, **chunks)
    payload = {
        "schema_version": "act_b_frame0_cross_input_isolation_v1",
        "status": "PASS",
        "purpose": "isolate frame-0 RGB effect from 28D proprioceptive-state effect; no Isaac command",
        "checkpoint_model_sha256": EXPECTED_SHA,
        "records": records,
        "real_hardware_transport": False,
        "commands_executed": 0,
    }
    temporary = OUT / "cross_input_report.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=lambda x: x.tolist()) + "\n", encoding="utf-8")
    os.replace(temporary, OUT / "cross_input_report.json")
    for row in records:
        print(row["label"], row["first_arm_branch_norm_vs_controlled_state_rad"], row["branch_pass"])


if __name__ == "__main__":
    main()
