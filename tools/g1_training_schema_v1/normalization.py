from __future__ import annotations

from typing import Any, Iterable

import numpy as np

STAT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def feature_stats(array: np.ndarray) -> dict[str, list[Any]]:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("normalization input must be a non-empty [N,D] array")
    if not np.isfinite(array).all():
        raise ValueError("normalization input contains non-finite values")
    values = array.astype(np.float64, copy=False)
    result: dict[str, list[Any]] = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0, ddof=0).tolist(),
        "count": [int(values.shape[0])],
    }
    for q in STAT_QUANTILES:
        result[f"q{int(q * 100):02d}"] = np.quantile(values, q, axis=0).tolist()
    return result


def fit_accepted_normalization(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[Any]]]:
    """Fit only from explicitly accepted training episodes.

    A rejected record is an error rather than something silently ignored.  The
    caller must perform the accepted-set selection first.
    """

    records = list(records)
    if not records:
        raise ValueError("cannot fit normalization without accepted episodes")
    if any(record.get("accepted") is not True for record in records):
        raise ValueError("normalization received a non-accepted episode")
    states = np.concatenate([np.asarray(record["observation.state"], dtype=np.float32) for record in records])
    actions = np.concatenate([np.asarray(record["action"], dtype=np.float32) for record in records])
    return {"observation.state": feature_stats(states), "action": feature_stats(actions)}


def fit_shared_diagnostic_normalization(
    records_a: Iterable[dict[str, Any]], records_b: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Fit one explicitly marked diagnostic stat set across accepted A and B records."""

    combined = list(records_a) + list(records_b)
    return {
        "mode": "SHARED_DIAGNOSTIC_NOT_DEFAULT",
        "statistics": fit_accepted_normalization(combined),
    }


def normalization_contract() -> dict[str, Any]:
    return {
        "default_mode": "per_dataset_accepted_train_set",
        "algorithm": "MEAN_STD",
        "features": ["observation.state", "action"],
        "formula": "normalized = (x - mean) / (population_std + 1e-8)",
        "inverse_formula": "x = normalized * population_std + mean",
        "epsilon": 1e-8,
        "statistical_reduction": "per channel over every frame in accepted training episodes; population std (ddof=0)",
        "fairness": "A and B use the identical algorithm but fit separate numerical statistics by default",
        "failed_episode_rule": "rejected/failed episodes are forbidden inputs to the estimator",
        "dry_run_rule": "no statistics are fit when zero accepted Integrated-v2 episodes are available",
        "diagnostic_shared_mode": (
            "optional only: concatenate the already accepted A and B training sets and fit one shared set; "
            "never the default and must be reported separately"
        ),
        "authoritative_code": {
            "config": "/home/jbnu/lerobot-smolvla/src/lerobot/policies/smolvla/configuration_smolvla.py::SmolVLAConfig.normalization_mapping",
            "processor": "/home/jbnu/lerobot-smolvla/src/lerobot/processor/normalize_processor.py::NormalizeProcessorStep",
        },
    }
