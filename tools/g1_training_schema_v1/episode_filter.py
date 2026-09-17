from __future__ import annotations

from pathlib import Path
from typing import Any

from .target_contract import EpisodeDecision, discover_episode_dirs, inspect_episode


def audit_episode_root(root: str | Path) -> tuple[dict[int, Path], dict[int, EpisodeDecision]]:
    directories = discover_episode_dirs(root)
    decisions = {episode_id: inspect_episode(path) for episode_id, path in directories.items()}
    return directories, decisions


def accepted_episode_ids(decisions: dict[int, EpisodeDecision]) -> list[int]:
    return sorted(episode_id for episode_id, decision in decisions.items() if decision.accepted)


def matched_episode_intersection(
    decisions_a: dict[int, EpisodeDecision], decisions_b: dict[int, EpisodeDecision]
) -> list[int]:
    return sorted(set(accepted_episode_ids(decisions_a)) & set(accepted_episode_ids(decisions_b)))


def intersection_manifest(
    decisions_a: dict[int, EpisodeDecision], decisions_b: dict[int, EpisodeDecision]
) -> dict[str, Any]:
    accepted_a = accepted_episode_ids(decisions_a)
    accepted_b = accepted_episode_ids(decisions_b)
    matched = matched_episode_intersection(decisions_a, decisions_b)
    return {
        "schema_version": "g1_training_schema_v1",
        "mode": "matched_episode_intersection",
        "dataset_a_accepted_episode_ids": accepted_a,
        "dataset_b_accepted_episode_ids": accepted_b,
        "matched_episode_ids": matched,
        "counts": {"dataset_a": len(accepted_a), "dataset_b": len(accepted_b), "matched": len(matched)},
    }


def select_episode_ids(
    decisions: dict[int, EpisodeDecision],
    mode: str,
    other_decisions: dict[int, EpisodeDecision] | None = None,
) -> list[int]:
    if mode == "native":
        return accepted_episode_ids(decisions)
    if mode == "matched":
        if other_decisions is None:
            raise ValueError("matched mode requires the other A/B input root")
        return matched_episode_intersection(decisions, other_decisions)
    raise ValueError(f"unknown episode-set mode: {mode}")
