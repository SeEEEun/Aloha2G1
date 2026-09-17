from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
GENERIC_RUNTIME = [
    ROOT / "tools/aloha_magsafe_semantics",
    ROOT / "tools/retarget_aloha_trajectory_to_g1.py",
    ROOT / "tools/retarget_episode49_semantic_compat.py",
    ROOT / "tools/v15_semantic_interface.py",
]


def _runtime_files() -> list[Path]:
    result: list[Path] = []
    for entry in GENERIC_RUNTIME:
        result.extend(sorted(entry.rglob("*.py")) if entry.is_dir() else [entry])
    return result


def test_generic_runtime_contains_no_episode_reference_indices() -> None:
    fixture = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    reference_values = {
        int(row["aligned_action_index"])
        for row in fixture["event_mapping"].values()
    }
    failures = []
    for path in _runtime_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in reference_values:
                failures.append({"file": str(path.relative_to(ROOT)), "line": node.lineno, "value": node.value})
    assert not failures, failures


def test_generic_runtime_has_no_literal_semantic_branch_patterns() -> None:
    patterns = (
        re.compile(r"\b(?:frame|action_index)\s*==\s*\d+"),
        re.compile(r"\bphase_knots\s*=\s*\[[^\]]*\d+"),
        re.compile(r"\[\s*\d+\s*:\s*\d+\s*\].*(?:phase|insert|release)", re.IGNORECASE),
    )
    failures = []
    for path in _runtime_files():
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            for match in pattern.finditer(text):
                failures.append({"file": str(path.relative_to(ROOT)), "pattern": pattern.pattern, "text": match.group(0)})
    assert not failures, failures

