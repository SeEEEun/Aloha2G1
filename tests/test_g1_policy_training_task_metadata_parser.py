from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.train_g1_policy_pair_matched51_v1 import (
    IntegrityFailure,
    resolve_task_metadata,
    validate_task_metadata_pair,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_A = ROOT / "lerobot_g1_magsafe_matched51_baseline_a_v1"
DATASET_B = ROOT / "lerobot_g1_magsafe_matched51_proposed_b_v1"
TASK = "Remove the MagSafe accessory from the phone and place the phone on the MagSafe charger."


def test_current_index_level_zero_layout_resolves() -> None:
    for root in (DATASET_A, DATASET_B):
        result = resolve_task_metadata(pq.read_table(root / "meta/tasks.parquet"))
        assert result["columns"] == ["task_index", "__index_level_0__"]
        assert result["authoritative_task_column"] == "__index_level_0__"
        assert result["task_values"] == [TASK]
        assert result["task_index_to_text"] == {"0": TASK}


def test_canonical_task_column_resolves() -> None:
    table = pa.table({"task_index": pa.array([0]), "task": pa.array([TASK])})
    result = resolve_task_metadata(table)
    assert result["authoritative_task_column"] == "task"
    assert result["task_index_to_text"] == {"0": TASK}


def test_ambiguous_multiple_string_columns_rejected() -> None:
    table = pa.table(
        {
            "task_index": pa.array([0]),
            "__index_level_0__": pa.array([TASK]),
            "description": pa.array([TASK]),
        }
    )
    with pytest.raises(IntegrityFailure, match="Ambiguous"):
        resolve_task_metadata(table)


def test_missing_string_column_rejected() -> None:
    table = pa.table({"task_index": pa.array([0]), "value": pa.array([7])})
    with pytest.raises(IntegrityFailure, match="No authoritative task string column"):
        resolve_task_metadata(table)


def test_a_b_differing_tasks_rejected() -> None:
    left = resolve_task_metadata(pa.table({"task_index": pa.array([0]), "task": pa.array([TASK])}))
    right = resolve_task_metadata(
        pa.table({"task_index": pa.array([0]), "task": pa.array(["Different task"])})
    )
    with pytest.raises(IntegrityFailure, match="task mappings differ"):
        validate_task_metadata_pair(left, right)


def test_task_index_mismatch_rejected() -> None:
    table = pa.table({"task_index": pa.array([1]), "task": pa.array([TASK])})
    with pytest.raises(IntegrityFailure, match="task_index mapping"):
        resolve_task_metadata(table)
