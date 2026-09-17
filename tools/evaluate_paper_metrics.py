#!/usr/bin/env python3
"""CPU-only entry point for all policy-independent paper metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT_HINT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(ROOT_HINT))

from tools.evaluation.contracts import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    ROOT,
    authoritative_joint_ranges,
)
from tools.evaluation.io import atomic_json, evaluate_bundle, result_rows
from tools.evaluation.paired_statistics import paired_statistics
from tools.evaluation.paper_core_adapter import (
    evaluate_offline_act,
    evaluate_physical_batch,
    evaluate_source_conditioned,
)
from tools.evaluation.physical_success import detect_physical_success
from tools.evaluation.project_adapter import evaluate_frozen_retargeting
from tools.evaluation.tables import generate_tables


DEFAULT_OUTPUT = ROOT / "outputs/paper_metrics"


def _regenerate_available_tables() -> dict[str, Any]:
    paths = {
        "table1_retargeting_quality": DEFAULT_OUTPUT / "retargeting/paired_statistics.json",
        "table2_downstream_act_policy": DEFAULT_OUTPUT / "offline_act/paired_statistics.json",
        "table3_source_conditioned_rollout": DEFAULT_OUTPUT
        / "source_conditioned/paired_statistics.json",
        "table4_isaac_physical_task_success": DEFAULT_OUTPUT
        / "physical/paired_statistics.json",
    }
    reports = {key: _read(path) for key, path in paths.items() if path.is_file()}
    manifest = generate_tables(DEFAULT_OUTPUT / "tables", reports)
    atomic_json(DEFAULT_OUTPUT / "tables/table_manifest.json", manifest)
    return manifest


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("evaluate-bundle", help="evaluate one portable bundle")
    bundle.add_argument("--bundle", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)

    paired = subparsers.add_parser("paired", help="strictly pair two result JSON files")
    paired.add_argument("--a", type=Path, required=True)
    paired.add_argument("--b", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    paired.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    paired.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)

    retarget = subparsers.add_parser(
        "project-retargeting", help="read-only evaluation of current frozen full-50 A/B"
    )
    retarget.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT / "retargeting")
    retarget.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)

    offline = subparsers.add_parser(
        "paper-core-offline", help="read-only adapter for selected held-out ACT chunks"
    )
    offline.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT / "offline_act")
    offline.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)

    source = subparsers.add_parser(
        "paper-core-source", help="read-only adapter for all source-conditioned rollouts"
    )
    source.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT / "source_conditioned"
    )
    source.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)

    physical_batch = subparsers.add_parser(
        "physical-batch", help="score the complete hash-matched 8x2 physical rollout matrix"
    )
    physical_batch.add_argument("--rollout-root", type=Path, required=True)
    physical_batch.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT / "physical")
    physical_batch.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)

    tables = subparsers.add_parser("tables", help="generate all four paper tables")
    tables.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "tables")
    for index in range(1, 5):
        tables.add_argument(f"--table{index}-paired", type=Path)

    physical = subparsers.add_parser("physical-success", help="score one logged Isaac trace")
    physical.add_argument("--log", type=Path, required=True)
    physical.add_argument("--config", type=Path, required=True)
    physical.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, ranges = authoritative_joint_ranges()
    if args.command == "evaluate-bundle":
        atomic_json(args.output, evaluate_bundle(args.bundle, ranges))
    elif args.command == "paired":
        report = paired_statistics(
            result_rows(_read(args.a)),
            result_rows(_read(args.b)),
            seed=args.seed,
            resamples=args.resamples,
        )
        atomic_json(args.output, report)
    elif args.command == "project-retargeting":
        a, b = evaluate_frozen_retargeting(joint_ranges_rad=ranges)
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_root / "fair_a_episode_metrics.json", a)
        atomic_json(args.output_root / "proposed_b_episode_metrics.json", b)
        paired = paired_statistics(a["episodes"], b["episodes"], resamples=args.resamples)
        atomic_json(args.output_root / "paired_statistics.json", paired)
        _regenerate_available_tables()
    elif args.command in ("paper-core-offline", "paper-core-source"):
        evaluator = (
            evaluate_offline_act
            if args.command == "paper-core-offline"
            else evaluate_source_conditioned
        )
        a, b = evaluator(joint_ranges_rad=ranges)
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_root / "act_a_episode_metrics.json", a)
        atomic_json(args.output_root / "act_b_episode_metrics.json", b)
        report = paired_statistics(a["episodes"], b["episodes"], resamples=args.resamples)
        atomic_json(args.output_root / "paired_statistics.json", report)
        _regenerate_available_tables()
    elif args.command == "physical-batch":
        a, b = evaluate_physical_batch(args.rollout_root, joint_ranges_rad=ranges)
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_root / "act_a_episode_metrics.json", a)
        atomic_json(args.output_root / "act_b_episode_metrics.json", b)
        report = paired_statistics(a["episodes"], b["episodes"], resamples=args.resamples)
        atomic_json(args.output_root / "paired_statistics.json", report)
        _regenerate_available_tables()
    elif args.command == "tables":
        paths = {
            f"table{index}_{suffix}": getattr(args, f"table{index}_paired")
            for index, suffix in (
                (1, "retargeting_quality"),
                (2, "downstream_act_policy"),
                (3, "source_conditioned_rollout"),
                (4, "isaac_physical_task_success"),
            )
        }
        reports = {key: _read(path) for key, path in paths.items() if path is not None}
        manifest = generate_tables(args.output_dir, reports)
        atomic_json(args.output_dir / "table_manifest.json", manifest)
    else:
        config = _read(args.config)
        with np.load(args.log, allow_pickle=False) as archive:
            log = {key: np.asarray(archive[key]) for key in archive.files}
        atomic_json(args.output, detect_physical_success(log, config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
