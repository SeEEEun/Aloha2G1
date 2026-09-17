"""Shared paths, serialization, and integrity helpers for hand repair v2."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = ROOT / "outputs/g1_dataset_retargeting_v1"
V2_ROOT = ROOT / "outputs/g1_dataset_retargeting_hand_v2"
V1_CONFIG = V1_ROOT / "config/aloha_g1_retargeting_v1.json"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(materialized[0])
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str, dict[str, str]]:
    """Hash every regular file without mutating the tree."""
    files = sorted(path for path in root.rglob("*") if path.is_file())
    values = {str(path.relative_to(root)): sha256_file(path) for path in files}
    digest = hashlib.sha256()
    for name, value in values.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), values


def load_v1_config() -> dict[str, Any]:
    return json.loads(V1_CONFIG.read_text(encoding="utf-8"))


def load_v2_config(path: Path | None = None) -> dict[str, Any]:
    source = path or (V2_ROOT / "config/aloha_g1_hand_v2.json")
    return json.loads(source.read_text(encoding="utf-8"))
