#!/usr/bin/env python3
"""Seal the completed Proposed-B-only 50-episode review artifact tree."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    atomic_json,
    load_json,
    sha256_file,
)


ROOT = REPOSITORY / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
REVIEW = ROOT / "review"


def main() -> int:
    aggregate = load_json(REVIEW / "proposed_b_aggregate_summary.json")
    visual = load_json(REVIEW / "visual_review_manifest.json")
    freeze_path = ROOT / "frozen_approval/freeze_manifest.json"
    freeze = load_json(freeze_path)
    trajectories = sorted((ROOT / "proposed/trajectories").glob("*.npz"))
    metrics = sorted(
        (ROOT / "proposed/metrics").glob(
            "doll_handoff_20260820_ep[0-9][0-9][0-9].json"
        )
    )
    validations = sorted((ROOT / "proposed/metrics").glob("*.validation.json"))
    manifests = sorted((ROOT / "proposed/metrics").glob("*.manifest.json"))
    videos = sorted((REVIEW / "videos").glob("*.mp4"))
    checks = {
        "aggregate_complete": aggregate["status"] == "COMPLETE",
        "aggregate_integrity_pass": bool(aggregate["overall_integrity_pass"]),
        "attempted_50": int(aggregate["attempted_episode_count"]) == 50,
        "trajectory_count_50": len(trajectories) == 50,
        "metric_count_50": len(metrics) == 50,
        "validation_count_50": len(validations) == 50,
        "manifest_count_50": len(manifests) == 50,
        "visual_manifest_pass": visual["status"] == "PASS",
        "all_failures_rendered": bool(visual["every_failure_has_overview_video"]),
        "all_representatives_three_views": bool(
            visual["every_representative_has_overview_top_side"]
        ),
        "rendered_video_count_37": len(videos) == 37,
        "baseline_a_absent": not (ROOT / "baseline").exists(),
        "dataset_packaging_absent": not any(
            (ROOT / name).exists() for name in ("dataset", "dataset_a", "dataset_b")
        ),
        "policy_training_absent": not any(
            (ROOT / name).exists() for name in ("policy", "policy_a", "policy_b", "training")
        ),
        "freeze_scope_proposed_only": freeze["scope"]
        == "PROPOSED_B_ONLY_50_EPISODE_REVIEW",
    }
    if not all(checks.values()):
        raise RuntimeError(
            "cannot seal Proposed-B review: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    entries_by_episode = {
        int(entry["episode_index"]): entry for entry in visual["entries"]
    }
    representatives = {
        f"ep{episode:03d}": entries_by_episode[episode]["outputs"]
        for episode in visual["representative_episode_indices"]
    }
    final_summary = {
        "schema_version": "interaction_centric_proposed_b_50_final_review_v1",
        "status": "PROPOSED_B_50_REVIEW_ARTIFACTS_COMPLETE",
        "root": str(ROOT),
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "implementation_sha256": freeze["implementation_sha256"],
        "scene_sha256": freeze["approved_resolved_file_sha256"]["scene_layout"],
        "attempted": 50,
        "status_counts": aggregate["status_counts"],
        "failure_episode_indices": aggregate["failure_episode_indices"],
        "aggregate_metrics": aggregate["aggregate_metrics"],
        "counts": aggregate["counts"],
        "representative_videos": representatives,
        "visual_review_manifest": str(REVIEW / "visual_review_manifest.json"),
        "per_episode_csv": str(REVIEW / "proposed_b_per_episode.csv"),
        "failure_report": str(REVIEW / "proposed_b_failures.json"),
        "aggregate_report": str(REVIEW / "proposed_b_aggregate_summary.json"),
        "checks": checks,
        "baseline_a": "NOT_RUN_BY_DESIGN",
        "dataset_packaging": "NOT_STARTED_BY_DESIGN",
        "policy_training": "NOT_STARTED_BY_DESIGN",
    }
    atomic_json(REVIEW / "final_summary.json", final_summary)

    excluded = {
        REVIEW / "artifact_manifest.json",
        REVIEW / "PROPOSED_B_50_REVIEW_COMPLETE",
    }
    files = [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path not in excluded
        and path.name != "visual_review_manifest.in_progress.json"
    ]
    file_hashes = {
        str(path.relative_to(ROOT)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    }
    digest = hashlib.sha256()
    for name, item in sorted(file_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\0")
    artifact_manifest = {
        "schema_version": "proposed_b_50_review_artifact_manifest_v1",
        "status": "COMPLETE",
        "root": str(ROOT),
        "file_count": len(file_hashes),
        "total_bytes": sum(item["bytes"] for item in file_hashes.values()),
        "tree_sha256": digest.hexdigest(),
        "files": file_hashes,
    }
    artifact_path = REVIEW / "artifact_manifest.json"
    atomic_json(artifact_path, artifact_manifest)
    artifact_hash = sha256_file(artifact_path)
    (REVIEW / "PROPOSED_B_50_REVIEW_COMPLETE").write_text(
        "PROPOSED_B_50_REVIEW_COMPLETE\n"
        "attempted: 50 / 50\n"
        f"PASS: {aggregate['status_counts']['PASS']}\n"
        f"FAIL_IK: {aggregate['status_counts']['FAIL_IK']}\n"
        f"FAIL_COLLISION: {aggregate['status_counts']['FAIL_COLLISION']}\n"
        f"artifact_manifest: {artifact_path}\n"
        f"artifact_manifest_sha256: {artifact_hash}\n"
        f"artifact_tree_sha256: {artifact_manifest['tree_sha256']}\n"
        "baseline_a: NOT_RUN\n"
        "dataset_packaging: NOT_RUN\n"
        "policy_training: NOT_RUN\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": final_summary["status"],
                "checks": checks,
                "artifact_file_count": artifact_manifest["file_count"],
                "artifact_total_bytes": artifact_manifest["total_bytes"],
                "artifact_tree_sha256": artifact_manifest["tree_sha256"],
                "artifact_manifest_sha256": artifact_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
