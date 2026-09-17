#!/usr/bin/env python3
"""Create a hash manifest or freeze one calibrated rigid-doll material.

The freeze path is intentionally fail-closed.  It accepts only matching LEFT
and RIGHT results from the fixed, policy-independent calibration primitive and
never reads ACT outputs, checkpoints, or policy labels.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation.contracts import sha256_file
from tools.evaluation.io import atomic_json


DEFAULT_CONFIG = ROOT / "configs/doll_handoff_rigid_proxy_v1.json"
DEFAULT_MANIFEST = ROOT / "configs/doll_handoff_rigid_proxy_v1.sha256.json"
ARTIFACT_FLAGS = (
    "penetration_artifact_detected",
    "magnetic_or_sticky_behavior_detected",
    "teleportation_detected",
    "object_constraint_attachment_detected",
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _candidate(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in config["material_candidates"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"material candidate {name!r} is absent or duplicated")
    return dict(matches[0])


def _validate_result(
    path: Path,
    *,
    side: str,
    candidate: Mapping[str, Any],
    config_sha256: str,
    expected_scene: Path,
) -> dict[str, Any]:
    result = _read(path)
    if result.get("schema_version") != "doll_handoff_rigid_proxy_grasp_calibration_v1":
        raise ValueError(f"unexpected calibration schema: {path}")
    expected = {
        "side": side,
        "material_candidate": candidate["name"],
        "config_sha256": config_sha256,
        "status": "PASS",
        "policy_or_checkpoint_used": False,
        "policy_specific_logic": False,
        "object_config_selected_from_policy_result": False,
    }
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{side} calibration result is not freeze-eligible: {mismatches}")
    if result.get(f"{side.upper()}_LIFT_PASS") is not True:
        raise RuntimeError(f"{side} fixed lift did not pass")
    artifacts = result.get("artifact_checks", {})
    flagged = [key for key in ARTIFACT_FLAGS if artifacts.get(key) is not False]
    if flagged:
        raise RuntimeError(f"{side} result has prohibited or unverified artifacts: {flagged}")
    if artifacts.get("object_pose_writes_during_timed_loop") != 0:
        raise RuntimeError(f"{side} result wrote the object pose during the timed trial")
    runtime = result.get("runtime_material", {})
    for key in ("static_friction", "dynamic_friction"):
        if not math.isclose(
            float(runtime.get(key, float("nan"))),
            float(candidate[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{side} runtime material differs from candidate for {key}")
    if bool(runtime.get("source_scene_saved_or_modified", True)):
        raise RuntimeError(f"{side} calibration modified the source scene")
    for key in ("primitive", "event_log"):
        artifact_path = Path(result[key])
        digest_key = f"{key}_sha256"
        if not artifact_path.is_file() or sha256_file(artifact_path) != result.get(digest_key):
            raise RuntimeError(f"{side} {key} is absent or its hash changed")
    scene_path = Path(result["scene"])
    if scene_path.resolve() != expected_scene.resolve():
        raise RuntimeError(f"{side} calibration used a different scene")
    if not scene_path.is_file() or sha256_file(scene_path) != result.get("scene_sha256"):
        raise RuntimeError(f"{side} calibration scene is absent or its hash changed")
    return result


def pending_manifest(config_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    scene_files = {
        key: {
            "path": str(Path(value).resolve()),
            "sha256": sha256_file(Path(value)),
        }
        for key, value in config["source_scene"].items()
        if key in ("layout", "scene_usd", "g1_preview_usd")
    }
    return {
        "schema_version": "doll_handoff_rigid_proxy_sha256_manifest_v1",
        "status": "FROZEN" if bool(config.get("freeze", {}).get("frozen")) else "PRECALIBRATION_NOT_FROZEN",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "selected_material_candidate": config.get("selected_material_candidate"),
        "policy_evaluation_allowed": bool(config.get("policy_evaluation_allowed", False)),
        "calibration_result_files": [],
        "source_scene_files": scene_files,
        "policy_results_consulted": False,
    }


def freeze(
    config_path: Path,
    candidate_name: str,
    left_path: Path,
    right_path: Path,
    output_config: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read(config_path)
    if config.get("schema_version") != "doll_handoff_rigid_proxy_v1":
        raise ValueError("unexpected rigid-proxy schema")
    if bool(config.get("freeze", {}).get("frozen")):
        raise RuntimeError("config is already frozen; refusing to overwrite its calibration record")
    source_sha = sha256_file(config_path)
    candidate = _candidate(config, candidate_name)
    expected_scene = Path(config["source_scene"]["g1_preview_usd"])
    left = _validate_result(
        left_path,
        side="left",
        candidate=candidate,
        config_sha256=source_sha,
        expected_scene=expected_scene,
    )
    right = _validate_result(
        right_path,
        side="right",
        candidate=candidate,
        config_sha256=source_sha,
        expected_scene=expected_scene,
    )
    frozen = deepcopy(config)
    timestamp = datetime.now(timezone.utc).isoformat()
    result_hashes = {
        "left": sha256_file(left_path),
        "right": sha256_file(right_path),
    }
    frozen.update(
        {
            "status": "FROZEN_AFTER_POLICY_INDEPENDENT_BILATERAL_LIFT_CALIBRATION",
            "policy_evaluation_allowed": True,
            "selected_material_candidate": candidate_name,
            "selected_material": candidate,
        }
    )
    frozen["freeze"] = {
        "frozen": True,
        "selected_candidate": candidate_name,
        "left_lift_pass": True,
        "right_lift_pass": True,
        "candidate_config_sha256": source_sha,
        "calibration_result_sha256": result_hashes,
        "calibration_result_paths": {
            "left": str(left_path.resolve()),
            "right": str(right_path.resolve()),
        },
        "selection_basis": "same fixed primitive passed LEFT and RIGHT without prohibited artifacts; no policy result consulted",
        "frozen_at": timestamp,
    }
    atomic_json(output_config, frozen)
    manifest = {
        "schema_version": "doll_handoff_rigid_proxy_sha256_manifest_v1",
        "status": "FROZEN",
        "config": str(output_config.resolve()),
        "config_sha256": sha256_file(output_config),
        "candidate_config_path_at_calibration": str(config_path.resolve()),
        "candidate_config_sha256": source_sha,
        "selected_material_candidate": candidate_name,
        "selected_material": candidate,
        "source_scene_files": {
            key: {
                "path": str(Path(value).resolve()),
                "sha256": sha256_file(Path(value)),
            }
            for key, value in frozen["source_scene"].items()
            if key in ("layout", "scene_usd", "g1_preview_usd")
        },
        "calibration_result_files": {
            "left": {"path": str(left_path.resolve()), "sha256": result_hashes["left"]},
            "right": {"path": str(right_path.resolve()), "sha256": result_hashes["right"]},
        },
        "primitive_files": {
            side: {"path": result["primitive"], "sha256": result["primitive_sha256"]}
            for side, result in (("left", left), ("right", right))
        },
        "event_logs": {
            side: {"path": result["event_log"], "sha256": result["event_log_sha256"]}
            for side, result in (("left", left), ("right", right))
        },
        "left_lift_pass": True,
        "right_lift_pass": True,
        "policy_results_consulted": False,
        "policy_specific_logic": False,
        "frozen_at": timestamp,
    }
    return frozen, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="hash the current candidate/frozen config")
    manifest.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    manifest.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    freeze_parser = sub.add_parser("freeze", help="freeze one bilateral-pass material")
    freeze_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freeze_parser.add_argument("--candidate", choices=("LOW", "MEDIUM", "HIGH"), required=True)
    freeze_parser.add_argument("--left-result", type=Path, required=True)
    freeze_parser.add_argument("--right-result", type=Path, required=True)
    freeze_parser.add_argument("--output-config", type=Path, default=DEFAULT_CONFIG)
    freeze_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    summary_parser = sub.add_parser(
        "freeze-summary",
        help="apply the predeclared lowest-friction bilateral-pass selection rule",
    )
    summary_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    summary_parser.add_argument("--summary", type=Path, required=True)
    summary_parser.add_argument("--output-config", type=Path, default=DEFAULT_CONFIG)
    summary_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "manifest":
        atomic_json(args.output, pending_manifest(args.config))
    else:
        if args.command == "freeze-summary":
            summary = _read(args.summary)
            rule = "LOWEST_FRICTION_CANDIDATE_WITH_BILATERAL_ARTIFACT_FREE_LIFT_PASS"
            if (
                summary.get("status") != "CALIBRATION_COMPLETE"
                or summary.get("predeclared_selection_rule") != rule
                or bool(summary.get("policy_results_consulted"))
            ):
                raise RuntimeError("calibration summary is incomplete or violates the selection contract")
            candidate = summary.get("recommended_candidate_under_predeclared_rule")
            if candidate not in ("LOW", "MEDIUM", "HIGH"):
                raise RuntimeError("no material candidate passed both fixed lifts")
            selected = [
                row
                for row in summary["results"]
                if row["material_candidate"] == candidate
            ]
            by_side = {row["side"]: row for row in selected}
            if set(by_side) != {"left", "right"}:
                raise RuntimeError("selected calibration summary lacks one hand")
            for row in by_side.values():
                result_path = Path(row["result_path"])
                if sha256_file(result_path) != row["result_sha256"]:
                    raise RuntimeError("calibration result changed after summary")
            args.candidate = candidate
            args.left_result = Path(by_side["left"]["result_path"])
            args.right_result = Path(by_side["right"]["result_path"])
        _, manifest = freeze(
            args.config,
            args.candidate,
            args.left_result,
            args.right_result,
            args.output_config,
        )
        atomic_json(args.manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
