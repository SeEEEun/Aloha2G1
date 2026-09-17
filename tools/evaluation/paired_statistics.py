"""Strict source-identity paired statistics with fixed-seed bootstrap CIs."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, validate_unique_episode_ids


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten JSON-like scalar metrics; strings, arrays, and NA records are skipped."""

    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        if value.get("status") == "NA":
            return result
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(child, path))
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            result[prefix] = numeric
    return result


def _summary(values: np.ndarray) -> dict[str, Any]:
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def _paired_sign_test_two_sided(differences: np.ndarray) -> dict[str, Any]:
    nonzero = differences[np.abs(differences) > 1e-15]
    n = int(nonzero.size)
    if n == 0:
        return {"nonzero_pairs": 0, "positive": 0, "negative": 0, "p_value": 1.0}
    positive = int(np.count_nonzero(nonzero > 0.0))
    negative = n - positive
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return {
        "nonzero_pairs": n,
        "positive": positive,
        "negative": negative,
        "p_value": float(min(1.0, 2.0 * probability)),
        "supplementary_only": True,
        "test": "exact two-sided paired sign test",
    }


def paired_bootstrap_mean_ci(
    differences: Any,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires finite non-empty differences")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    generator = rng if rng is not None else np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(resamples, len(values)))
    estimates = np.mean(values[indices], axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _index(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    entries = list(rows)
    validate_unique_episode_ids(entries)
    return {str(row["source_episode_id"]): row for row in entries}


def paired_statistics(
    a_rows: Iterable[Mapping[str, Any]],
    b_rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    metric_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare only exact A/B source identities and fail on any unmatched ID."""

    a_by_id = _index(a_rows)
    b_by_id = _index(b_rows)
    a_ids = set(a_by_id)
    b_ids = set(b_by_id)
    if a_ids != b_ids:
        raise ValueError(
            "A/B source episode identities differ; unmatched comparison is forbidden; "
            f"only_A={sorted(a_ids - b_ids)}, only_B={sorted(b_ids - a_ids)}"
        )
    ordered_ids = sorted(a_ids)
    flat_a = {key: flatten_numeric(a_by_id[key].get("metrics", {})) for key in ordered_ids}
    flat_b = {key: flatten_numeric(b_by_id[key].get("metrics", {})) for key in ordered_ids}
    if metric_paths is None:
        paths = sorted(set().union(*(set(row) for row in flat_a.values()), *(set(row) for row in flat_b.values())))
    else:
        paths = sorted(set(map(str, metric_paths)))
    rng = np.random.default_rng(seed)
    reports: dict[str, Any] = {}
    for path in paths:
        eligible = [key for key in ordered_ids if path in flat_a[key] and path in flat_b[key]]
        if not eligible:
            reports[path] = {
                "status": "NA",
                "reason": "metric has no complete A/B episode pairs",
                "paired_episode_count": 0,
            }
            continue
        a = np.asarray([flat_a[key][path] for key in eligible], dtype=np.float64)
        b = np.asarray([flat_b[key][path] for key in eligible], dtype=np.float64)
        difference = b - a
        ci_low, ci_high = paired_bootstrap_mean_ci(
            difference, resamples=resamples, rng=rng
        )
        reports[path] = {
            "status": "READY",
            "paired_episode_count": len(eligible),
            "paired_episode_ids": eligible,
            "missing_metric_episode_ids": [key for key in ordered_ids if key not in eligible],
            "A": _summary(a),
            "B": _summary(b),
            "paired_B_minus_A": _summary(difference),
            "paired_bootstrap_95_percent_CI_of_mean_B_minus_A": [ci_low, ci_high],
            "supplementary_paired_sign_test": _paired_sign_test_two_sided(difference),
        }
    return {
        "schema_version": "paper_paired_statistics_v1",
        "status": "READY",
        "pairing": {
            "rule": "exact source_episode_id equality; any unmatched identity is a hard error",
            "paired_episode_count": len(ordered_ids),
            "source_episode_ids": ordered_ids,
            "unmatched_episode_count": 0,
        },
        "bootstrap": {
            "statistic": "paired episode-level mean B-A",
            "confidence_interval": "percentile 95%",
            "seed": int(seed),
            "resamples": int(resamples),
        },
        "metrics": reports,
    }
