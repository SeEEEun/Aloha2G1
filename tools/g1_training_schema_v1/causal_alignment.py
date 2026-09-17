from __future__ import annotations

import numpy as np

from .constants import CHUNK_SIZE, FPS


def validate_episode_timestamps(timestamps: np.ndarray, frame_count: int, fps: float = FPS) -> None:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.shape != (frame_count,):
        raise ValueError(f"timestamp shape {timestamps.shape} != ({frame_count},)")
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps contain non-finite values")
    if frame_count and timestamps[0] != 0.0:
        raise ValueError("episode timestamp must start at zero")
    if frame_count > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps are not strictly monotonic within episode")
    expected = np.arange(frame_count, dtype=np.float64) / fps
    if not np.allclose(timestamps, expected, rtol=0.0, atol=1e-5):
        raise ValueError("timestamps are not frame_index/fps")


def action_chunk_indices(frame_index: int, frame_count: int, chunk_size: int = CHUNK_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Mirror LeRobot's per-episode clamp and action_is_pad convention."""

    if frame_count <= 0 or not 0 <= frame_index < frame_count:
        raise ValueError("invalid frame index/count")
    requested = frame_index + np.arange(chunk_size, dtype=np.int64)
    is_pad = requested >= frame_count
    indices = np.minimum(requested, frame_count - 1)
    return indices, is_pad


def alignment_contract() -> dict:
    return {
        "state_frame": "retargeted target controlled qpos q_target[t] at source RGB row t",
        "action_frame": "absolute controlled-joint target q_target[t] associated with the same row t",
        "causal_relation": "observation.state[t] -> action[t]; no +1 row shift",
        "timestamp_offset_frames": 0,
        "timestamp_offset_seconds": 0.0,
        "no_future_state_leakage": True,
        "source_command_observation_note": (
            "The source row contains same-cycle measured follower q and absolute command q. A separately "
            "approved diagnostic found approximately 7-frame plant response latency; it does not authorize "
            "shifting the recorded training rows."
        ),
        "last_frame_handling": "retain last row; action[t] is valid",
        "episode_boundary_handling": "never query across an episode boundary",
        "action_chunk_start": "current row t",
        "action_chunk_size": CHUNK_SIZE,
        "chunk_indices": "t..t+49",
        "chunk_padding": "clamp out-of-range future indices to episode last row",
        "padding_mask": "action_is_pad is true exactly where t+k >= episode length; masked out of SmolVLA loss",
    }
