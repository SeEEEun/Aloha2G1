#!/usr/bin/env python3
"""Precompute one frozen teacher-forced ACT-A/B held-out trajectory.

The policy input at source frame ``t`` is the original ALOHA cam_high RGB[t]
plus the matching retargeting method's frozen held-out observation.state[t].
The output is the official LeRobot ACT temporal ensemble (coefficient 0.01).
This program has no Isaac, controller, DDS, or real-robot interface.
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
from typing import Any

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EXPERIMENT2 = ROOT / "outputs/paper_core_ab/offline_heldout8/experiment2_result.json"
EXECUTION = ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout/SELECTED_ACT_EXECUTION_CONFIG.json"
PROJECTION = ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"
EXPECTED_HELDOUT_SHA256 = "a86181b049d0f521d1167c2b58bc15f3a7cb6ad87ee9a1f634ef02c04adcc710"
EXPECTED_EXPERIMENT2_SHA256 = "c3e0c24611997c5a61dcc8f1686dfd9b5a8691e8b9bf2db22f7a0013c62ee3ae"
EXPECTED_EXECUTION_SHA256 = "656f6f474f6981c5b0c0417895b91ad924d171031bb66b26e4640b54bc863b64"
EXPECTED_PROJECTION_SHA256 = "05078d0038ab6defaa8a0f56b1f38b752cc92892856996fecc05b75fcce27cf2"
DATASETS = {
    "a": ROOT / "datasets/doll_handoff_fair_a_heldout8",
    "b": ROOT / "datasets/doll_handoff_proposed_b_heldout8",
}
EXPECTED_PARQUET_SHA256 = {
    "a": "d689201ecb8dff24a985761028fc6efaaf0f3437159c4df99093c47049168b57",
    "b": "ce367f7b05fe2946e662aa6977aca15387ef4632890d677a19182cc926c72d2f",
}
CONTROL_FPS = 30.0
CHUNK_SIZE = 50
ACTION_DIM = 28
TEMPORAL_ENSEMBLE_COEFF = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    parser.add_argument("--heldout-episode", type=int, choices=range(8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: expected={expected} actual={actual} path={path}")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_reference_states(method: str, output_episode: int, frames: int) -> tuple[np.ndarray, Path]:
    parquet = DATASETS[method] / "data/chunk-000/file-000.parquet"
    require_sha(parquet, EXPECTED_PARQUET_SHA256[method], f"ACT-{method.upper()} held-out parquet")
    table = pq.read_table(
        parquet,
        columns=["observation.state", "episode_index", "frame_index", "timestamp"],
    )
    table = table.filter(pc.equal(table["episode_index"], output_episode))
    frame_index = np.asarray(table["frame_index"], dtype=np.int64)
    order = np.argsort(frame_index, kind="stable")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[order]
    timestamps = np.asarray(table["timestamp"], dtype=np.float32)[order]
    if (
        states.shape != (frames, ACTION_DIM)
        or not np.array_equal(frame_index[order], np.arange(frames))
        or not np.allclose(timestamps, np.arange(frames) / CONTROL_FPS, atol=1e-5, rtol=0.0)
        or not np.isfinite(states).all()
    ):
        raise RuntimeError("method-specific held-out state sequence violates its frozen contract")
    return states, parquet


def validate_existing(output: Path, identity: dict[str, Any]) -> bool:
    manifest_path = output / "cache_manifest.json"
    trajectory_path = output / "full_trajectory.npz"
    if not manifest_path.is_file() or not trajectory_path.is_file():
        return False
    manifest = read_json(manifest_path)
    checks = (
        manifest.get("status") == "PASS",
        manifest.get("experiment") == "TEACHER_FORCED_SOURCE_VIDEO_POLICY_VISUALIZATION",
        manifest.get("method") == identity["method"],
        manifest.get("heldout_output_episode") == identity["heldout_output_episode"],
        manifest.get("source_final_episode") == identity["source_final_episode"],
        manifest.get("checkpoint_model_sha256") == identity["checkpoint_model_sha256"],
        manifest.get("source_video_sha256") == identity["source_video_sha256"],
        manifest.get("trajectory_sha256") == sha256_file(trajectory_path),
    )
    if not all(checks):
        raise RuntimeError(f"existing cache failed identity/hash validation: {output}")
    print(f"[CACHE HIT] {trajectory_path}", flush=True)
    return True


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    require_sha(HELDOUT, EXPECTED_HELDOUT_SHA256, "HELDOUT8 manifest")
    require_sha(EXPERIMENT2, EXPECTED_EXPERIMENT2_SHA256, "Experiment-2 result")
    require_sha(EXECUTION, EXPECTED_EXECUTION_SHA256, "selected ACT execution")
    require_sha(PROJECTION, EXPECTED_PROJECTION_SHA256, "common deployment adapter")
    heldout = read_json(HELDOUT)
    experiment2 = read_json(EXPERIMENT2)
    execution = read_json(EXECUTION)
    if heldout.get("status") != "PASS" or heldout.get("episode_count") != 8:
        raise RuntimeError("HELDOUT8 manifest is not PASS")
    official = execution["official_lerobot_execution"]
    if (
        execution.get("selection") != "ACT_E1_TEMPORAL_ENSEMBLE"
        or official.get("chunk_size") != CHUNK_SIZE
        or official.get("n_action_steps") != 1
        or official.get("temporal_ensemble_coeff") != TEMPORAL_ENSEMBLE_COEFF
        or official.get("control_fps") != CONTROL_FPS
        or official.get("custom_smoothing")
    ):
        raise RuntimeError("frozen official ACT-E1 execution contract changed")
    entry = heldout["entries"][args.heldout_episode]
    source_episode = int(entry["final_dataset_index"])
    frames = int(entry["frames"])
    source_video = Path(entry["source_rgb_identity"]["canonical_video_path"]).resolve()
    source_video_sha = str(entry["source_rgb_identity"]["canonical_video_sha256"])
    require_sha(source_video, source_video_sha, "original ALOHA source video")
    selection = experiment2["methods"][args.method]["checkpoint_selection"]
    checkpoint = Path(selection["selected_checkpoint"]).resolve()
    checkpoint_sha = str(selection["selected_model_sha256"])
    require_sha(checkpoint / "model.safetensors", checkpoint_sha, "selected ACT model")
    identity = {
        "method": f"ACT-{args.method.upper()}40",
        "heldout_output_episode": args.heldout_episode,
        "source_final_episode": source_episode,
        "checkpoint_model_sha256": checkpoint_sha,
        "source_video_sha256": source_video_sha,
    }
    if validate_existing(output, identity):
        return 0
    if (output / "cache_manifest.json").exists() or (output / "full_trajectory.npz").exists():
        raise RuntimeError("refusing to overwrite an incomplete or mismatched completed cache")
    states, parquet = load_reference_states(args.method, args.heldout_episode, frames)

    atomic_json(
        progress_path,
        {
            "status": "LOADING_ACT",
            **identity,
            "frame": 0,
            "frames": frames,
        },
    )
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from lerobot.policies.factory import make_pre_post_processors
    from tools.common_deployment_safety_projection import NamedJointDeploymentSafetyProjector
    from tools.paper_core_source_rollout_common import frozen_interfaces

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
        raise RuntimeError(f"strict ACT-E1 weight reload failed: {loaded}")
    policy.to(next(base_policy.parameters()).device)
    del base_policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not isinstance(policy.temporal_ensembler, ACTTemporalEnsembler):
        raise RuntimeError("official ACTTemporalEnsembler was not constructed")
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    dropout_active = [
        name for name, module in policy.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.training
    ]
    if dropout_active:
        raise RuntimeError(f"dropout remains active in eval mode: {dropout_active}")

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"could not decode source video: {source_video}")
    if (
        int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) != frames
        or not np.isclose(capture.get(cv2.CAP_PROP_FPS), CONTROL_FPS, atol=1e-6)
        or int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) != 640
        or int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) != 480
    ):
        raise RuntimeError("source video metadata differs from HELDOUT8")

    normalized_chunks: list[np.ndarray] = []
    raw_chunks: list[np.ndarray] = []
    normalized_actions: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    rgb_hashes: list[str] = []
    inference_seconds: list[float] = []
    started = time.monotonic()
    try:
        for frame in range(frames):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"source video ended at frame {frame}/{frames}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"malformed source frame {frame}: {rgb.shape} {rgb.dtype}")
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
            start = time.perf_counter()
            with torch.inference_mode():
                normalized_chunk = policy.predict_action_chunk(processed)
                normalized_action = policy.temporal_ensembler.update(normalized_chunk)
                physical_chunk = postprocessor(normalized_chunk.clone())
                physical_action = postprocessor(normalized_action.clone())
            inference_seconds.append(time.perf_counter() - start)
            values = (
                normalized_chunk[0].detach().float().cpu().numpy().astype(np.float32),
                physical_chunk[0].detach().float().cpu().numpy().astype(np.float32),
                normalized_action[0].detach().float().cpu().numpy().astype(np.float32),
                physical_action[0].detach().float().cpu().numpy().astype(np.float32),
            )
            if (
                values[0].shape != (CHUNK_SIZE, ACTION_DIM)
                or values[1].shape != (CHUNK_SIZE, ACTION_DIM)
                or values[2].shape != (ACTION_DIM,)
                or values[3].shape != (ACTION_DIM,)
                or not all(np.isfinite(value).all() for value in values)
            ):
                raise RuntimeError(f"malformed/non-finite ACT output at frame {frame}")
            normalized_chunks.append(values[0])
            raw_chunks.append(values[1])
            normalized_actions.append(values[2])
            raw_actions.append(values[3])
            rgb_hashes.append(sha256_array(rgb))
            if frame % 10 == 0 or frame + 1 == frames:
                atomic_json(
                    progress_path,
                    {
                        "status": "PRECOMPUTING_ACT_E1",
                        **identity,
                        "frame": frame + 1,
                        "frames": frames,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                print(
                    f"[ACT-{args.method.upper()}40] source ep {source_episode}: "
                    f"{frame + 1}/{frames}",
                    flush=True,
                )
    finally:
        capture.release()

    normalized_chunks_array = np.stack(normalized_chunks).astype(np.float32)
    raw_chunks_array = np.stack(raw_chunks).astype(np.float32)
    normalized_actions_array = np.stack(normalized_actions).astype(np.float32)
    raw_actions_array = np.stack(raw_actions).astype(np.float32)
    projector = NamedJointDeploymentSafetyProjector.from_path(PROJECTION)
    projection = projector.project(raw_actions_array, global_row_offset=0)
    hard_actions = projection.hard_limit_projected_action.astype(np.float32)
    deployment_actions = projection.deployment_safe_action.astype(np.float32)

    names, lower, upper, _ = frozen_interfaces()
    if names != projector.names:
        raise RuntimeError("named joint order differs between frozen interfaces")
    thresholds = read_json(
        ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
    )["unchanged_acceptance"]
    initial_velocity = (
        (states[1].astype(np.float64) - states[0].astype(np.float64)) * CONTROL_FPS
        if frames > 1 else np.zeros(ACTION_DIM, dtype=np.float64)
    )
    first_abort: dict[str, Any] | None = None
    maxima = {
        "maximum_joint_step_rad": 0.0,
        "maximum_velocity_rad_s": 0.0,
        "maximum_acceleration_rad_s2": 0.0,
        "maximum_arm_step_l2_rad": 0.0,
    }
    previous = states[0].astype(np.float64)
    previous_velocity = initial_velocity
    prior_arm_norms: list[float] = []
    for frame, command in enumerate(deployment_actions.astype(np.float64)):
        delta = command - previous
        velocity = delta * CONTROL_FPS
        acceleration = (velocity - previous_velocity) * CONTROL_FPS
        arm_norm = float(np.linalg.norm(delta[:14]))
        local = float(np.median(prior_arm_norms[-10:])) if prior_arm_norms else 0.0
        branch_threshold = max(
            float(thresholds["branch_absolute_step_norm_rad"]),
            float(thresholds["branch_local_multiplier"]) * max(local, 1e-6),
        )
        violation = (command < lower - 1e-9) | (command > upper + 1e-9)
        checks = {
            "finite": bool(np.isfinite(command).all()),
            "command_hard_limits": int(np.count_nonzero(violation)) == 0,
            "adjacent_step": float(np.max(np.abs(delta)))
            <= float(thresholds["maximum_joint_step_rad"]),
            "velocity": float(np.max(np.abs(velocity)))
            <= float(thresholds["maximum_velocity_rad_s"]),
            "acceleration": float(np.max(np.abs(acceleration)))
            <= float(thresholds["maximum_acceleration_rad_s2"]),
            "branch": frame == 0 or arm_norm <= branch_threshold,
        }
        audit = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "command_hard_limit_violation_count": int(np.count_nonzero(violation)),
            "command_hard_limit_violation_joints": [
                names[index] for index in np.flatnonzero(violation)
            ],
            "maximum_joint_step_rad": float(np.max(np.abs(delta))),
            "maximum_velocity_rad_s": float(np.max(np.abs(velocity))),
            "maximum_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
            "arm_step_l2_rad": arm_norm,
            "branch_threshold_rad": branch_threshold,
            "branch_gate_applicable": frame > 0,
        }
        maxima["maximum_joint_step_rad"] = max(
            maxima["maximum_joint_step_rad"], audit["maximum_joint_step_rad"]
        )
        maxima["maximum_velocity_rad_s"] = max(
            maxima["maximum_velocity_rad_s"], audit["maximum_velocity_rad_s"]
        )
        maxima["maximum_acceleration_rad_s2"] = max(
            maxima["maximum_acceleration_rad_s2"], audit["maximum_acceleration_rad_s2"]
        )
        maxima["maximum_arm_step_l2_rad"] = max(
            maxima["maximum_arm_step_l2_rad"], arm_norm
        )
        if audit["status"] != "PASS" and first_abort is None:
            first_abort = {
                "frame": frame,
                "source_time_seconds": frame / CONTROL_FPS,
                "failed_checks": [name for name, passed in audit["checks"].items() if not passed],
                "audit": audit,
            }
        previous = command
        previous_velocity = velocity
        if frame > 0:
            prior_arm_norms.append(arm_norm)

    trajectory_path = output / "full_trajectory.npz"
    atomic_npz(
        trajectory_path,
        raw_act_chunks=raw_chunks_array,
        normalized_act_chunks=normalized_chunks_array,
        raw_temporal_ensemble_action=raw_actions_array,
        normalized_temporal_ensemble_action=normalized_actions_array,
        hard_limit_projected_action=hard_actions,
        deployment_safe_action=deployment_actions,
        method_specific_reference_state=states,
        source_frame_index=np.arange(frames, dtype=np.int64),
        source_timestamp_seconds=np.arange(frames, dtype=np.float64) / CONTROL_FPS,
        source_rgb_sha256=np.asarray(rgb_hashes),
        joint_names=np.asarray(names),
        control_fps_hz=np.asarray(CONTROL_FPS),
        temporal_ensemble_coeff=np.asarray(TEMPORAL_ENSEMBLE_COEFF),
        method=np.asarray(args.method),
        heldout_output_episode=np.asarray(args.heldout_episode),
        source_final_episode=np.asarray(source_episode),
        checkpoint_model_sha256=np.asarray(checkpoint_sha),
        source_video_sha256=np.asarray(source_video_sha),
    )
    trajectory_sha = sha256_file(trajectory_path)
    manifest = {
        "schema_version": "teacher_forced_source_video_act_gui_cache_v1",
        "status": "PASS",
        "experiment": "TEACHER_FORCED_SOURCE_VIDEO_POLICY_VISUALIZATION",
        "claim_control": "human visual diagnostic; physical object success not evaluated",
        **identity,
        "stable_episode_id": entry["stable_episode_id"],
        "source_recording_id": entry["original_source_recording_id"],
        "frames": frames,
        "control_fps_hz": CONTROL_FPS,
        "duration_seconds": (frames - 1) / CONTROL_FPS,
        "policy_input": "original ALOHA cam_high RGB[t] + frozen method-specific held-out observation.state[t]",
        "live_isaac_rgb_used": False,
        "measured_isaac_state_used_for_policy_input": False,
        "helmet_or_head_camera_used": False,
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(selection["selected_step"]),
        "source_video": str(source_video),
        "method_dataset": str(DATASETS[args.method]),
        "method_dataset_parquet": str(parquet),
        "method_dataset_parquet_sha256": EXPECTED_PARQUET_SHA256[args.method],
        "heldout_manifest": str(HELDOUT),
        "heldout_manifest_sha256": EXPECTED_HELDOUT_SHA256,
        "experiment2_result": str(EXPERIMENT2),
        "experiment2_result_sha256": EXPECTED_EXPERIMENT2_SHA256,
        "official_execution": {
            "policy_class": "lerobot.policies.act.modeling_act.ACTPolicy",
            "temporal_ensembler_class": "lerobot.policies.act.modeling_act.ACTTemporalEnsembler",
            "chunk_size": CHUNK_SIZE,
            "n_action_steps": 1,
            "temporal_ensemble_coeff": TEMPORAL_ENSEMBLE_COEFF,
            "custom_averaging": False,
            "custom_smoothing": False,
        },
        "common_deployment_adapter": {
            "path": str(PROJECTION),
            "sha256": EXPECTED_PROJECTION_SHA256,
            "applied_identically_to_a_b": True,
            "arm_outputs_bitwise_preserved": projection.summary["arm_outputs_bitwise_preserved"],
            "summary": projection.summary,
        },
        "preserved_arrays": {
            "raw_act_chunks": list(raw_chunks_array.shape),
            "raw_temporal_ensemble_action": list(raw_actions_array.shape),
            "deployment_safe_action": list(deployment_actions.shape),
            "method_specific_reference_state": list(states.shape),
            "source_timestamps": [frames],
        },
        "array_sha256": {
            "raw_act_chunks": sha256_array(raw_chunks_array),
            "raw_temporal_ensemble_action": sha256_array(raw_actions_array),
            "deployment_safe_action": sha256_array(deployment_actions),
            "method_specific_reference_state": sha256_array(states),
        },
        "safety_preflight": {
            "status": "PASS" if first_abort is None else "ABORT_DECLARED",
            "first_abort": first_abort,
            "maxima": maxima,
            "thresholds_unchanged": thresholds,
            "kinematic_self_collision_audit": "DEFERRED_TO_ISAAC_GUI_ENVIRONMENT_USING_UNCHANGED_SAFETY_AUDIT",
            "playback_rule": "stop before the first declared hard-safety-failing command and hold the last safe physical state",
        },
        "inference": {
            "eval_mode": True,
            "dropout_modules_active": dropout_active,
            "calls": frames,
            "mean_seconds": float(np.mean(inference_seconds)),
            "maximum_seconds": float(np.max(inference_seconds)),
            "total_wall_seconds": time.monotonic() - started,
        },
        "trajectory": str(trajectory_path),
        "trajectory_sha256": trajectory_sha,
        "precompute_tool": str(Path(__file__).resolve()),
        "precompute_tool_sha256": sha256_file(Path(__file__).resolve()),
        "learned_policy_weights_modified": False,
        "dataset_or_retargeting_modified": False,
        "real_robot_interface": False,
    }
    atomic_json(output / "cache_manifest.json", manifest)
    atomic_json(
        progress_path,
        {
            "status": "READY",
            **identity,
            "frame": frames,
            "frames": frames,
            "trajectory": str(trajectory_path),
            "trajectory_sha256": trajectory_sha,
            "safety_preflight": manifest["safety_preflight"]["status"],
        },
    )
    print(json.dumps({"status": "READY", "output": str(output), "frames": frames}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
