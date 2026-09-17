#!/usr/bin/env python3
"""Apply the frozen common G1 feasibility resolver to Fair A on CPU."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fair_a_full50_audit.common import OUTPUT_ROOT  # noqa: E402
from tools.fair_a_full50_audit.wrist_resolver import (  # noqa: E402
    RepresentationNeutralWristResolver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        default="0-49",
        help="comma-separated indices/ranges, e.g. 3,4,15-18",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("frozen-exact", "full-pose-common-fix"),
        default="frozen-exact",
    )
    return parser.parse_args()


def episode_indices(specification: str) -> list[int]:
    values: set[int] = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(value) for value in token.split("-", 1))
            values.update(range(first, last + 1))
        else:
            values.add(int(token))
    episodes = sorted(values)
    if not episodes or episodes[0] < 0 or episodes[-1] >= 50:
        raise ValueError("episode specification must be a nonempty subset of 0..49")
    return episodes


def main() -> int:
    args = parse_args()
    episodes = episode_indices(args.episodes)
    if args.mode == "full-pose-common-fix":
        from tools.fair_a_full50_audit.full_pose_resolver import (
            FullPoseCommonWristResolver,
        )

        resolver = FullPoseCommonWristResolver(output_root=args.output_root.resolve())
    else:
        resolver = RepresentationNeutralWristResolver(
            output_root=args.output_root.resolve()
        )
    summary = []
    for position, episode in enumerate(episodes, start=1):
        action = "solve"
        if not args.no_reuse:
            try:
                result = resolver.load_exported_episode(episode)
                action = "reuse"
            except (FileNotFoundError, KeyError, RuntimeError, ValueError):
                result = resolver.solve_episode(episode, export=True)
        else:
            result = resolver.solve_episode(episode, export=True)
        row = {
            "episode": episode,
            "action": action,
            "changed_frames": int(result.metadata["changed_arm_frame_count"]),
            "projection_frames": int(result.metadata["projection_active_frame_count"]),
            "projection_max_m": float(np_max(result.projection_translation_m)),
            "source_position_max_m": float(
                result.metadata["source_position_residual_m"]["max"]
            ),
            "orientation_max_rad": float(
                result.metadata["realized_orientation_residual_rad"]["max"]
            ),
        }
        summary.append(row)
        print(
            f"[{position:02d}/{len(episodes):02d}] ep{episode:03d} {action} "
            f"changed={row['changed_frames']} projection={row['projection_frames']} "
            f"projection_max={1000*row['projection_max_m']:.2f}mm "
            f"source_max={1000*row['source_position_max_m']:.2f}mm "
            f"orientation_max={row['orientation_max_rad']:.3f}rad",
            flush=True,
        )
    print(json.dumps({"episodes": summary}, indent=2))
    return 0


def np_max(value) -> float:
    import numpy as np

    return float(np.max(np.asarray(value), initial=0.0))


if __name__ == "__main__":
    raise SystemExit(main())
