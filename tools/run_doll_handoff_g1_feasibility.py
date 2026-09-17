#!/usr/bin/env python3
"""Run staged diagnostics and generic G1 feasibility realization for Doll-Handoff."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_feasibility.common import (  # noqa: E402
    OUTPUT_ROOT,
    load_json,
)
from tools.doll_handoff_feasibility.diagnostics import (  # noqa: E402
    build_failure_diagnostics,
)
from tools.doll_handoff_feasibility.evaluate import (  # noqa: E402
    full50_report,
    representative_gate,
)
from tools.doll_handoff_feasibility.solver import (  # noqa: E402
    GenericG1FeasibilityResolver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "diagnose",
            "representative",
            "full50",
            "videos",
            "finalize",
            "all",
        ),
        help="staged execution prevents an all-50 rerun before the representative gate",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def _solve_episodes(
    resolver: GenericG1FeasibilityResolver,
    episodes: list[int],
    reuse_existing: bool = False,
) -> list:
    results = []
    for position, episode in enumerate(episodes, start=1):
        print(
            f"[{position:02d}/{len(episodes):02d}] processing ep{episode:03d}",
            flush=True,
        )
        action = "solving"
        if reuse_existing:
            try:
                result = resolver.load_exported_episode(episode)
                action = "reusing"
            except (FileNotFoundError, KeyError, RuntimeError, ValueError):
                result = resolver.solve_episode(episode, export=True)
        else:
            result = resolver.solve_episode(episode, export=True)
        print(
            f"[{position:02d}/{len(episodes):02d}] {action} ep{episode:03d}",
            flush=True,
        )
        results.append(result)
        metadata = results[-1].metadata
        print(
            "  "
            f"changed={metadata['changed_arm_frame_count']} "
            f"projection={metadata['projection_active_frame_count']} "
            f"source_max={metadata['source_position_residual_m']['max']*1000:.2f}mm",
            flush=True,
        )
    return results


def _run_representative(output_root: Path) -> dict:
    resolver = GenericG1FeasibilityResolver(output_root=output_root)
    episodes = [
        int(value)
        for value in resolver.config["representative_gate"]["episodes"]
    ]
    return representative_gate(resolver, _solve_episodes(resolver, episodes))


def _run_full50(output_root: Path) -> dict:
    gate_path = output_root / "representative_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("representative gate is absent; refusing the full-50 run")
    gate = load_json(gate_path)
    if gate.get("status") != "PASS":
        raise RuntimeError(
            "representative gate did not pass; refusing the full-50 run"
        )
    resolver = GenericG1FeasibilityResolver(output_root=output_root)
    return full50_report(
        resolver,
        _solve_episodes(resolver, list(range(50)), reuse_existing=True),
    )


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    if args.stage == "diagnose":
        result = build_failure_diagnostics(args.output_root)
    elif args.stage == "representative":
        result = _run_representative(args.output_root)
    elif args.stage == "full50":
        result = _run_full50(args.output_root)
    elif args.stage == "videos":
        from tools.doll_handoff_feasibility.render_review import (
            render_representative_reviews,
        )

        result = render_representative_reviews(args.output_root)
    elif args.stage == "finalize":
        from tools.doll_handoff_feasibility.finalize import finalize_review

        result = finalize_review(args.output_root)
    elif args.stage == "all":
        diagnostic = build_failure_diagnostics(args.output_root)
        representative = _run_representative(args.output_root)
        result = {"diagnostic": diagnostic, "representative": representative}
        if representative.get("status") == "PASS":
            result["full50"] = _run_full50(args.output_root)
            from tools.doll_handoff_feasibility.render_review import (
                render_representative_reviews,
            )

            result["videos"] = render_representative_reviews(args.output_root)
            from tools.doll_handoff_feasibility.finalize import finalize_review

            result["freeze"] = finalize_review(args.output_root)
        else:
            result["full50"] = "NOT_RUN_REPRESENTATIVE_GATE_FAILED"
    else:  # pragma: no cover - argparse makes this unreachable
        raise ValueError(args.stage)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
