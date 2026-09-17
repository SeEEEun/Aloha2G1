from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import ACTION_DIM, STATE_DIM
from .target_contract import RetargetedTrajectory


@dataclass(frozen=True)
class AdaptedEpisode:
    episode_id: int
    observation_state: np.ndarray
    action: np.ndarray
    timestamps: np.ndarray
    internal_state_semantic: str = "retargeted_target_state"


def adapt_target_qpos(trajectory: RetargetedTrajectory) -> AdaptedEpisode:
    """Create target-embodiment state/action without changing trajectory values.

    Both arrays are distinct copies so later consumers cannot mutate one through
    the other.  This intentionally does not use the source ALOHA 14D state.
    """

    q = np.asarray(trajectory.q, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != STATE_DIM or ACTION_DIM != STATE_DIM:
        raise ValueError(f"expected target qpos [T,{STATE_DIM}], got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("target qpos contains non-finite values")
    return AdaptedEpisode(
        episode_id=trajectory.episode_id,
        observation_state=np.array(q, copy=True, dtype=np.float32, order="C"),
        action=np.array(q, copy=True, dtype=np.float32, order="C"),
        timestamps=np.array(trajectory.timestamps, copy=True, dtype=np.float64),
    )
