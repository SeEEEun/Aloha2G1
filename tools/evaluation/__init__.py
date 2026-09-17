"""Policy-independent paper evaluation for Doll-Handoff A/B experiments.

The package is deliberately NumPy-only.  It does not import a policy, Isaac,
rendering, or robot interfaces.  See :mod:`tools.evaluation.contracts` for the
portable episode-bundle schema.
"""

from .contracts import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CANONICAL_PHASES,
    FPS,
    PHYSICAL_SUCCESS,
    SEMANTIC_SUCCESS,
)
from .metrics import evaluate_episode

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CANONICAL_PHASES",
    "FPS",
    "PHYSICAL_SUCCESS",
    "SEMANTIC_SUCCESS",
    "evaluate_episode",
]
