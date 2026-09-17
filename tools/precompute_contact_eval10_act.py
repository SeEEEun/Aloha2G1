#!/usr/bin/env python3
"""Precompute frozen source-conditioned ACT-E1 trajectories for EVAL10.

The first eight entries reuse the original HELDOUT8 identity and dataset state
convention.  Entries 8-9 use the frozen Fair-A/Proposed-B NEW_UNSEEN_2 states
and the original cam_high PNGs.  No simulator, controller, or tuning is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/final_contact_constrained_eval/04_eval10_preparation/act_trajectories"
EVAL10 = ROOT / "outputs/final_contact_constrained_eval/04_eval10_preparation/EVAL10_RETARGETING_MANIFEST.json"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EXPERIMENT2 = ROOT / "outputs/paper_core_ab/offline_heldout8/experiment2_result.json"
EXECUTION = ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout/SELECTED_ACT_EXECUTION_CONFIG.json"
PROJECTION = ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"
DATASETS = {
    "a": ROOT / "datasets/doll_handoff_fair_a_heldout8",
    "b": ROOT / "datasets/doll_handoff_proposed_b_heldout8",
}
EXPECTED = {
    "heldout": "a86181b049d0f521d1167c2b58bc15f3a7cb6ad87ee9a1f634ef02c04adcc710",
    "experiment2": "c3e0c24611997c5a61dcc8f1686dfd9b5a8691e8b9bf2db22f7a0013c62ee3ae",
    "execution": "656f6f474f6981c5b0c0417895b91ad924d171031bb66b26e4640b54bc863b64",
    "projection": "05078d0038ab6defaa8a0f56b1f38b752cc92892856996fecc05b75fcce27cf2",
    "a_parquet": "d689201ecb8dff24a985761028fc6efaaf0f3437159c4df99093c47049168b57",
    "b_parquet": "ce367f7b05fe2946e662aa6977aca15387ef4632890d677a19182cc926c72d2f",
}
FPS = 30.0
DIM = 28
CHUNK = 50
COEFFICIENT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require(path: Path, digest: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != digest:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {digest}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda item: item.tolist()
            if isinstance(item, np.ndarray)
            else item.item()
            if isinstance(item, np.generic)
            else str(item),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def heldout_states(method: str, output_episode: int, frames: int) -> np.ndarray:
    parquet = DATASETS[method] / "data/chunk-000/file-000.parquet"
    require(parquet, EXPECTED[f"{method}_parquet"], f"ACT-{method.upper()} HELDOUT8 parquet")
    table = pq.read_table(parquet, columns=["observation.state", "episode_index", "frame_index"])
    table = table.filter(pc.equal(table["episode_index"], output_episode))
    index = np.asarray(table["frame_index"], dtype=np.int64)
    order = np.argsort(index, kind="stable")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[order]
    if states.shape != (frames, DIM) or not np.array_equal(index[order], np.arange(frames)):
        raise RuntimeError("HELDOUT8 state identity failed")
    return states


def video_frames(path: Path, count: int) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    try:
        for frame in range(count):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"video ended at {frame}/{count}: {path}")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ok, _ = capture.read()
        if ok:
            raise RuntimeError(f"video has more frames than frozen manifest: {path}")
    finally:
        capture.release()


def png_frames(root: Path, count: int) -> Iterator[np.ndarray]:
    paths = [root / f"frame_{frame:06d}.png" for frame in range(count)]
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"incomplete original PNG sequence: {root}")
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not decode {path}")
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    args = parse_args()
    method = args.method
    output_root = args.output_root.resolve() / f"act_{method}40"
    require(HELDOUT, EXPECTED["heldout"], "HELDOUT8")
    require(EXPERIMENT2, EXPECTED["experiment2"], "Experiment-2")
    require(EXECUTION, EXPECTED["execution"], "ACT execution")
    require(PROJECTION, EXPECTED["projection"], "deployment projection")
    eval10 = read_json(EVAL10)
    heldout = read_json(HELDOUT)
    experiment = read_json(EXPERIMENT2)
    if eval10.get("status") != "PASS" or len(eval10.get("eval_entries", [])) != 10:
        raise RuntimeError("EVAL10 retargeting is not frozen/PASS")

    selection = experiment["methods"][method]["checkpoint_selection"]
    checkpoint = Path(selection["selected_checkpoint"]).resolve()
    checkpoint_sha = str(selection["selected_model_sha256"])
    require(checkpoint / "model.safetensors", checkpoint_sha, "selected ACT checkpoint")

    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from lerobot.policies.factory import make_pre_post_processors
    from tools.common_deployment_safety_projection import NamedJointDeploymentSafetyProjector

    base = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    if (
        base.config.chunk_size != CHUNK
        or base.config.n_action_steps != CHUNK
        or base.config.temporal_ensemble_coeff is not None
        or base.config.action_feature.shape != (DIM,)
    ):
        raise RuntimeError("checkpoint violates frozen raw ACT contract")
    config = copy.deepcopy(base.config)
    config.n_action_steps = 1
    config.temporal_ensemble_coeff = COEFFICIENT
    policy = ACTPolicy(config)
    loaded = policy.load_state_dict(base.state_dict(), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"strict reload failed: {loaded}")
    policy.to(next(base.parameters()).device)
    del base
    policy.eval()
    if not isinstance(policy.temporal_ensembler, ACTTemporalEnsembler):
        raise RuntimeError("official ACTTemporalEnsembler was not created")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    projector = NamedJointDeploymentSafetyProjector.from_path(PROJECTION)

    summaries = []
    new_rows = {int(row["eval_index"]): row for row in eval10["new_unseen_2"]}
    for eval_index in range(10):
        entry = eval10["eval_entries"][eval_index]
        frames = int(entry["frames"])
        destination = output_root / f"eval_{eval_index:02d}_{entry['stable_episode_id']}"
        trajectory_path = destination / "full_trajectory.npz"
        manifest_path = destination / "cache_manifest.json"
        if manifest_path.is_file() and trajectory_path.is_file():
            cached = read_json(manifest_path)
            if (
                cached.get("status") != "PASS"
                or cached.get("checkpoint_model_sha256") != checkpoint_sha
                or cached.get("trajectory_sha256") != sha256_file(trajectory_path)
            ):
                raise RuntimeError(f"invalid existing cache: {destination}")
            summaries.append(cached)
            print(f"[CACHE] ACT-{method.upper()}40 EVAL10 {eval_index + 1}/10", flush=True)
            continue

        if eval_index < 8:
            source = heldout["entries"][eval_index]
            if source["stable_episode_id"] != entry["stable_episode_id"]:
                raise RuntimeError("HELDOUT8/EVAL10 identity mismatch")
            states = heldout_states(method, eval_index, frames)
            source_path = Path(source["source_rgb_identity"]["canonical_video_path"]).resolve()
            require(
                source_path,
                source["source_rgb_identity"]["canonical_video_sha256"],
                "HELDOUT8 source RGB",
            )
            rgb_iterator = video_frames(source_path, frames)
            rgb_identity = sha256_file(source_path)
            rgb_storage = "canonical HELDOUT8 MP4"
        else:
            row = new_rows[eval_index]
            state_path = Path(row[method]["evaluation_trajectory"])
            require(state_path, row[method]["evaluation_trajectory_sha256"], "NEW2 retargeted state")
            with np.load(state_path, allow_pickle=False) as payload:
                states = np.asarray(payload["observation_state"], dtype=np.float32)
            source_path = Path(row["source_cam_high"]).resolve()
            rgb_iterator = png_frames(source_path, frames)
            rgb_identity = row["source_cam_high_tree_sha256"]
            rgb_storage = "original frame-aligned cam_high PNG tree"
        if states.shape != (frames, DIM) or not np.isfinite(states).all():
            raise RuntimeError(f"invalid method-specific state EVAL10 index {eval_index}")

        policy.reset()
        normalized_chunks = []
        raw_chunks = []
        normalized_actions = []
        raw_actions = []
        rgb_hashes = []
        durations = []
        started = time.monotonic()
        for frame, rgb in enumerate(rgb_iterator):
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"invalid RGB at {eval_index}:{frame}")
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
                    "observation.state": torch.from_numpy(states[frame].copy()),
                }
            )
            begin = time.perf_counter()
            with torch.inference_mode():
                normalized_chunk = policy.predict_action_chunk(processed)
                normalized_action = policy.temporal_ensembler.update(normalized_chunk)
                physical_chunk = postprocessor(normalized_chunk.clone())
                physical_action = postprocessor(normalized_action.clone())
            durations.append(time.perf_counter() - begin)
            normalized_chunks.append(normalized_chunk[0].detach().float().cpu().numpy())
            raw_chunks.append(physical_chunk[0].detach().float().cpu().numpy())
            normalized_actions.append(normalized_action[0].detach().float().cpu().numpy())
            raw_actions.append(physical_action[0].detach().float().cpu().numpy())
            rgb_hashes.append(sha256_array(rgb))
        if len(raw_actions) != frames:
            raise RuntimeError("source RGB iterator frame count mismatch")
        normalized_chunks_array = np.asarray(normalized_chunks, dtype=np.float32)
        raw_chunks_array = np.asarray(raw_chunks, dtype=np.float32)
        normalized_actions_array = np.asarray(normalized_actions, dtype=np.float32)
        raw_actions_array = np.asarray(raw_actions, dtype=np.float32)
        projection = projector.project(raw_actions_array, global_row_offset=0)
        hard_actions = projection.hard_limit_projected_action.astype(np.float32)
        deployment_actions = projection.deployment_safe_action.astype(np.float32)

        destination.mkdir(parents=True, exist_ok=True)
        temporary = trajectory_path.with_suffix(".npz.incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                raw_act_chunks=raw_chunks_array,
                normalized_act_chunks=normalized_chunks_array,
                raw_temporal_ensemble_action=raw_actions_array,
                normalized_temporal_ensemble_action=normalized_actions_array,
                hard_limit_projected_action=hard_actions,
                deployment_safe_action=deployment_actions,
                method_specific_reference_state=states,
                source_frame_index=np.arange(frames, dtype=np.int64),
                source_timestamp_seconds=np.arange(frames, dtype=np.float64) / FPS,
                source_rgb_sha256=np.asarray(rgb_hashes),
                joint_names=np.asarray(projector.names),
                control_fps_hz=np.asarray(FPS),
                temporal_ensemble_coeff=np.asarray(COEFFICIENT),
                method=np.asarray(method),
                eval_index=np.asarray(eval_index),
                checkpoint_model_sha256=np.asarray(checkpoint_sha),
                source_rgb_identity_sha256=np.asarray(rgb_identity),
            )
        os.replace(temporary, trajectory_path)
        manifest = {
            "schema_version": "contact_constrained_eval10_act_e1_v1",
            "status": "PASS",
            "method": f"ACT-{method.upper()}40",
            "eval_index": eval_index,
            "provenance": entry["provenance"],
            "stable_episode_id": entry["stable_episode_id"],
            "frames": frames,
            "policy_input": "original ALOHA cam_high RGB[t] + frozen method-specific retargeted observation.state[t]",
            "source_rgb": str(source_path),
            "source_rgb_storage": rgb_storage,
            "source_rgb_identity_sha256": rgb_identity,
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(selection["selected_step"]),
            "checkpoint_model_sha256": checkpoint_sha,
            "official_execution": {
                "policy": "ACTPolicy",
                "temporal_ensembler": "ACTTemporalEnsembler",
                "coefficient": COEFFICIENT,
                "chunk_size": CHUNK,
                "n_action_steps": 1,
                "custom_smoothing": False,
            },
            "trajectory": str(trajectory_path.resolve()),
            "trajectory_sha256": sha256_file(trajectory_path),
            "array_sha256": {
                "raw_policy_command": sha256_array(raw_actions_array),
                "deployment_safe_action": sha256_array(deployment_actions),
                "method_specific_reference_state": sha256_array(states),
            },
            "inference": {
                "calls": frames,
                "mean_seconds": float(np.mean(durations)),
                "maximum_seconds": float(np.max(durations)),
                "wall_seconds": time.monotonic() - started,
                "eval_mode": True,
            },
            "training_used": False,
            "checkpoint_selection_changed": False,
            "controller_tuning_used": False,
            "simulator_used": False,
        }
        atomic_json(manifest_path, manifest)
        summaries.append(manifest)
        print(f"[READY] ACT-{method.upper()}40 EVAL10 {eval_index + 1}/10", flush=True)

    summary = {
        "schema_version": "contact_constrained_eval10_act_batch_v1",
        "status": "PASS",
        "method": f"ACT-{method.upper()}40",
        "episodes": 10,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": checkpoint_sha,
        "entries": [
            {
                "eval_index": row["eval_index"],
                "stable_episode_id": row["stable_episode_id"],
                "provenance": row["provenance"],
                "frames": row["frames"],
                "trajectory": row["trajectory"],
                "trajectory_sha256": row["trajectory_sha256"],
            }
            for row in summaries
        ],
    }
    atomic_json(output_root / "BATCH_MANIFEST.json", summary)
    print(json.dumps({"status": "PASS", "method": method, "episodes": 10}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
