"""Read-only adapters for the paper-core ACT artifacts produced by the parallel task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    AUTHORITATIVE_REFERENCES,
    FPS,
    PHYSICAL_SUCCESS,
    ROOT,
    SEMANTIC_SUCCESS,
    sha256_file,
)
from .metrics import aggregate_episode_metrics, evaluate_episode
from .io import evaluate_bundle
from .semantic_sequence import semantic_phase_events


PAPER = ROOT / "outputs/paper_core_ab"
HELDOUT = PAPER / "heldout8_manifest.json"
OFFLINE_RESULT = PAPER / "offline_heldout8/experiment2_result.json"
SOURCE_RESULT = PAPER / "source_conditioned_rollout/experiment3_result.json"
SOURCE_ROOT = PAPER / "source_conditioned_rollout"
SEMANTIC_CONFIG = ROOT / "configs/paper_semantic_task_sequence_v1.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _event_map(payload: Mapping[str, np.ndarray]) -> dict[str, int]:
    return {
        str(name): int(frame)
        for name, frame in zip(
            payload["event_names"].astype(str),
            payload["event_frames"].astype(np.int64),
            strict=True,
        )
    }


def _joint_contract() -> tuple[list[str], np.ndarray, np.ndarray]:
    payload = _read(AUTHORITATIVE_REFERENCES["joint_ranges"])
    names = list(payload["joint_names"])
    specs = payload["joint_specs"]
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    if len(names) != 28 or lower.shape != (28,) or np.any(upper <= lower):
        raise RuntimeError("authoritative joint limit contract changed")
    return names, lower, upper


def _g1_interfaces() -> tuple[Any, np.ndarray, dict[str, np.ndarray]]:
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics

    names, _, _ = _joint_contract()
    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    lookup = {name: index for index, name in enumerate(names)}
    arm = np.asarray([lookup[str(name)] for name in g1.arm_joint_names], dtype=np.int64)
    hands = {
        side: np.asarray([lookup[name] for name in g1.hand_joint_names[side]], dtype=np.int64)
        for side in ("left", "right")
    }
    if len(set(np.concatenate((arm, hands["left"], hands["right"])).tolist())) != 28:
        raise RuntimeError("named G1 FK mapping is not bijective")
    return g1, arm, hands


def _geometry(g1: Any, arm: np.ndarray, hands: Mapping[str, np.ndarray], q: np.ndarray) -> dict[str, Any]:
    tolerance = float(
        _read(AUTHORITATIVE_REFERENCES["feasibility_config"])["unchanged_acceptance"][
            "collision_penetration_tolerance_m"
        ]
    )
    return g1.trajectory_geometry(
        q[:, arm], q[:, hands["left"]], q[:, hands["right"]], tolerance
    )


def _branch_count(chunks: list[np.ndarray]) -> int:
    thresholds = _read(AUTHORITATIVE_REFERENCES["feasibility_config"])["unchanged_acceptance"]
    absolute = float(thresholds["branch_absolute_step_norm_rad"])
    multiplier = float(thresholds["branch_local_multiplier"])
    count = 0
    for chunk in chunks:
        norms = np.linalg.norm(np.diff(chunk[:, :14], axis=0), axis=1)
        for index, value in enumerate(norms):
            local = float(np.median(norms[max(0, index - 9) : min(len(norms), index + 10)]))
            count += int(value > max(absolute, multiplier * max(local, 1e-6)))
    return count


def _result(method: str, mode: str, rows: list[dict[str, Any]], provenance: Mapping[str, Any]) -> dict[str, Any]:
    success_kind = PHYSICAL_SUCCESS if mode == "physical" else SEMANTIC_SUCCESS
    return {
        "schema_version": "paper_evaluation_result_v1",
        "status": "READY",
        "method": "ACT-A40" if method == "a" else "ACT-B40",
        "evaluation_mode": mode,
        "success_kind": success_kind,
        "episode_count": len(rows),
        "source_episode_ids": [row["source_episode_id"] for row in rows],
        "episodes": rows,
        "aggregate": aggregate_episode_metrics(rows),
        "read_only_adapter": True,
        "provenance": dict(provenance),
    }


def evaluate_source_conditioned(*, joint_ranges_rad: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt all eight completed or aborted source-aligned rollouts without exclusion."""

    manifest = _read(HELDOUT)
    semantic_config = _read(SEMANTIC_CONFIG)
    if semantic_config.get("status") != "FROZEN_BEFORE_SOURCE_CONDITIONED_ACT_RESULTS":
        raise RuntimeError("semantic sequence contract is not frozen")
    names, _, _ = _joint_contract()
    rows: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    for output_episode, entry in enumerate(manifest["entries"]):
        reference = _npz(Path(entry["b_trajectory_path"]))
        event_frames = _event_map(reference)
        semantic_arrays = {
            "left_hand_phase": reference["left_hand_phase"],
            "right_hand_phase": reference["right_hand_phase"],
            "ownership_state": reference["ownership_state"],
        }
        for method in ("a", "b"):
            path = SOURCE_ROOT / method / (
                f"heldout_{output_episode:02d}_source_{int(entry['final_dataset_index']):02d}"
            )
            report_path = path / "rollout_report.json"
            derived_path = path / "trajectory_evaluation_arrays.npz"
            evaluation_path = path / "trajectory_evaluation.json"
            if not report_path.is_file() or not derived_path.is_file() or not evaluation_path.is_file():
                raise FileNotFoundError(
                    f"source rollout is not immediately scoreable for exact identity {entry['stable_episode_id']}: {path}"
                )
            report = _read(report_path)
            existing = _read(evaluation_path)
            arrays = _npz(derived_path)
            if arrays["joint_names"].astype(str).tolist() != names:
                raise RuntimeError(f"source rollout named order changed: {path}")
            q = arrays["measured_q"].astype(np.float64)
            target_q = arrays["method_specific_retargeted_target_q"].astype(np.float64)
            if q.shape != target_q.shape or q.ndim != 2 or q.shape[1] != 28:
                raise RuntimeError(f"source rollout q/target shape mismatch: {path}")
            semantic = semantic_phase_events(
                q,
                target_q,
                event_frames=event_frames,
                semantic_arrays=semantic_arrays,
                config=semantic_config,
            )
            ordering = existing["handoff_ordering"]
            feasibility = {
                "hard_ik_failure_count": 0,
                "hard_collision_count": int(existing["collision"]["hard_collision_frame_incidence"])
                + int(existing["table_contact"]["occurrence_count"]),
                "joint_limit_failure_count": int(existing["joint_limits"]["violation_frames"]),
                "branch_discontinuity_count": int(existing["branch_discontinuity_count"]),
                "minimum_clearance_m": None,
                "provenance": f"{evaluation_path}: existing paper-core validated G1 geometry/safety diagnostics",
            }
            episode = {
                "source_episode_id": entry["stable_episode_id"],
                "fps": FPS,
                "evaluation_mode": "source_conditioned",
                "success_kind": SEMANTIC_SUCCESS,
                "arrays": {
                    "reference_left_wrist_position_m": arrays["source_left_wrist_world"],
                    "reference_right_wrist_position_m": arrays["source_right_wrist_world"],
                    "candidate_left_wrist_position_m": arrays["predicted_left_wrist_world"],
                    "candidate_right_wrist_position_m": arrays["predicted_right_wrist_world"],
                    "reference_left_whole_hand_position_m": arrays["source_left_interaction_world"],
                    "reference_right_whole_hand_position_m": arrays["source_right_interaction_world"],
                    "candidate_left_whole_hand_position_m": arrays["predicted_left_whole_hand_world"],
                    "candidate_right_whole_hand_position_m": arrays["predicted_right_whole_hand_world"],
                    "candidate_q_rad": q,
                },
                "annotations": {
                    "candidate_phase_events_frame": semantic["canonical_phase_events_frame"],
                    "phase_events_authoritative": True,
                    "right_acquire_frame": ordering["right_acquire_frame"],
                    "left_release_frame": ordering["left_release_frame"],
                    "handoff_events_authoritative": True,
                    "annotation_scope": semantic["annotation_scope"],
                },
                "feasibility": feasibility,
                "provenance": {
                    "whole_hand_frame_authoritative": True,
                    "whole_hand_frame_definition": str(AUTHORITATIVE_REFERENCES["whole_hand_frame"]),
                    "wrist_orientation_reference_authoritative": False,
                    "source_rollout": str(path),
                    "source_rollout_report_sha256": sha256_file(report_path),
                    "complete_source_rollout": bool(
                        report.get("status") == "PASS"
                        and int(report["executed_frames"]) == int(report["requested_frames"])
                    ),
                    "semantic_phase_detection": semantic,
                },
            }
            rows[method].append(evaluate_episode(episode, joint_ranges_rad=joint_ranges_rad))
    provenance = {
        "source_result": str(SOURCE_RESULT),
        "source_result_sha256": sha256_file(SOURCE_RESULT) if SOURCE_RESULT.is_file() else None,
        "heldout_manifest": str(HELDOUT),
        "heldout_manifest_sha256": sha256_file(HELDOUT),
        "semantic_contract": str(SEMANTIC_CONFIG),
        "semantic_contract_sha256": sha256_file(SEMANTIC_CONFIG),
        "all_eight_identities_required": True,
    }
    return (
        _result("a", "source_conditioned", rows["a"], provenance),
        _result("b", "source_conditioned", rows["b"], provenance),
    )


OFFLINE_BEHAVIOR_MAP = {
    "LEFT_APPROACH": "LEFT_APPROACH",
    "LEFT_GRASP": "LEFT_GRASP",
    "LEFT_TRANSPORT": "LEFT_TRANSPORT",
    "RIGHT_APPROACH": "RIGHT_HANDOFF_APPROACH",
    "DUAL_CONTACT": "DUAL_HAND_CONFIGURATION",
    "RIGHT_OWNED": "RIGHT_OWNED",
    "RIGHT_TRANSPORT": "RIGHT_TRANSPORT",
    "RELEASE": "RELEASE",
}


def evaluate_offline_act(*, joint_ranges_rad: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt selected raw held-out ACT chunks into paired episode metrics."""

    result = _read(OFFLINE_RESULT)
    if result.get("status") != "PASS":
        raise RuntimeError("offline paper-core result is absent or not PASS")
    manifest = _read(HELDOUT)
    names, lower, upper = _joint_contract()
    g1, arm, hands = _g1_interfaces()
    rows: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    for method in ("a", "b"):
        method_result = result["methods"][method]
        prediction_path = Path(method_result["selected_prediction_archive"])
        prediction_archive = _npz(prediction_path)
        prediction = prediction_archive["prediction"].astype(np.float64)
        if prediction.shape != (72, 50, 28):
            raise RuntimeError(f"selected raw prediction shape changed: {prediction_path}")
        if method_result["selected_checkpoint_evaluation"]["prediction_shape"] != [72, 50, 28]:
            raise RuntimeError("selected checkpoint metadata/prediction mismatch")
        behavior_rows = method_result["selected_checkpoint_evaluation"]["phase_score"]["behaviors"]
        ordering_rows = method_result["hand_configuration_and_ordering"]["handoff_ordering"]["per_episode"]
        for output_episode, entry in enumerate(manifest["entries"]):
            selected = np.flatnonzero(prediction_archive["output_episode"] == output_episode)
            if len(selected) != 9:
                raise RuntimeError(f"offline probe count differs from nine for {entry['stable_episode_id']}")
            selected = selected[np.argsort(prediction_archive["frame"][selected])]
            frames = prediction_archive["frame"][selected].astype(np.int64)
            valid = prediction_archive["valid_frames"][selected].astype(np.int64)
            candidate_chunks = prediction[selected]
            trajectory_path = Path(entry[f"{method}_trajectory_path"])
            trajectory = _npz(trajectory_path)
            target_all = trajectory["replay_named_joint_qpos"].astype(np.float64)
            target_chunks = np.zeros_like(candidate_chunks)
            flattened_candidate = []
            reference_wrist = {side: [] for side in ("left", "right")}
            candidate_wrist = {side: [] for side in ("left", "right")}
            reference_hand = {side: [] for side in ("left", "right")}
            candidate_hand = {side: [] for side in ("left", "right")}
            common_a = _npz(Path(entry["a_trajectory_path"]))
            common_b = _npz(Path(entry["b_trajectory_path"]))
            valid_chunks = []
            for local, (frame, length) in enumerate(zip(frames, valid, strict=True)):
                truth = target_all[frame : frame + length]
                if truth.shape != (length, 28):
                    raise RuntimeError("offline target chunk extends past source episode")
                target_chunks[local, :length] = truth
                if length < 50:
                    target_chunks[local, length:] = truth[-1]
                candidate_valid = candidate_chunks[local, :length]
                valid_chunks.append(candidate_valid)
                flattened_candidate.append(candidate_valid)
                geometry = _geometry(g1, arm, hands, candidate_valid)
                for side in ("left", "right"):
                    candidate_wrist[side].append(geometry[f"{side}_wrist_position_world"])
                    candidate_hand[side].append(geometry[f"{side}_grasp_position_world"])
                    reference_wrist[side].append(
                        g1.model_to_world_position(
                            common_a[f"target_{side}_wrist_position_model"][frame : frame + length]
                        )
                    )
                    reference_hand[side].append(
                        common_b[f"source_{side}_interaction_frame_position_world"][frame : frame + length]
                    )
            flat_q = np.concatenate(flattened_candidate)
            all_geometry = _geometry(g1, arm, hands, flat_q)
            hard_flags = np.logical_or.reduce(
                [
                    np.asarray(all_geometry["collision_flags"][key], dtype=bool)
                    for key in ("ARM_TORSO", "CROSS_ARM", "WRIST_OR_PALM_TORSO", "OTHER")
                ]
            )
            violations = (flat_q < lower[None] - 1e-9) | (flat_q > upper[None] + 1e-9)
            source_episode = int(entry["final_dataset_index"])
            phase_events: dict[str, int | None] = {}
            for canonical, behavior in OFFLINE_BEHAVIOR_MAP.items():
                matches = [
                    row
                    for row in behavior_rows[behavior]["per_episode"]
                    if int(row["final_episode"]) == source_episode
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"offline semantic behavior row mismatch: {canonical}")
                phase_events[canonical] = int(matches[0]["frame"]) if matches[0]["success"] else None
            ordering_matches = [
                row for row in ordering_rows if int(row["final_episode"]) == source_episode
            ]
            if len(ordering_matches) != 1:
                raise RuntimeError("offline handoff ordering row mismatch")
            ordering = ordering_matches[0]
            base = int(ordering["dual_probe_frame"])
            right_offset = ordering["right_acquire_predicted_offset"]
            left_offset = ordering["left_release_predicted_offset"]
            episode = {
                "source_episode_id": entry["stable_episode_id"],
                "fps": FPS,
                "evaluation_mode": "offline_act",
                "success_kind": SEMANTIC_SUCCESS,
                "path_efficiency_applicable": False,
                "arrays": {
                    "reference_action_chunk_rad": target_chunks,
                    "candidate_action_chunk_rad": candidate_chunks,
                    "action_chunk_valid_lengths": valid,
                    "candidate_q_chunk_rad": candidate_chunks,
                    **{
                        f"reference_{side}_wrist_position_m": np.concatenate(reference_wrist[side])
                        for side in ("left", "right")
                    },
                    **{
                        f"candidate_{side}_wrist_position_m": np.concatenate(candidate_wrist[side])
                        for side in ("left", "right")
                    },
                    **{
                        f"reference_{side}_whole_hand_position_m": np.concatenate(reference_hand[side])
                        for side in ("left", "right")
                    },
                    **{
                        f"candidate_{side}_whole_hand_position_m": np.concatenate(candidate_hand[side])
                        for side in ("left", "right")
                    },
                },
                "annotations": {
                    "candidate_phase_events_frame": phase_events,
                    "phase_events_authoritative": True,
                    "right_acquire_frame": base + int(right_offset) if right_offset is not None else None,
                    "left_release_frame": base + int(left_offset) if left_offset is not None else None,
                    "handoff_events_authoritative": True,
                    "annotation_scope": "FROZEN_OFFLINE_SEMANTIC_BEHAVIOR_CONTRACT_NOT_PHYSICAL_CONTACT",
                },
                "feasibility": {
                    "hard_ik_failure_count": 0,
                    "hard_collision_count": int(np.count_nonzero(hard_flags)),
                    "joint_limit_failure_count": int(np.count_nonzero(np.any(violations, axis=1))),
                    "branch_discontinuity_count": _branch_count(valid_chunks),
                    "minimum_clearance_m": None,
                    "provenance": "same frozen G1Kinematics collision categories and acceptance thresholds; raw chunks scored independently",
                },
                "smoothness": {
                    "reversal_deadband_rad_s": float(
                        _read(PAPER / "offline_evaluation_contract.json")["raw_smoothness_contract"][
                            "direction_reversal_deadband_rad_s"
                        ]
                    )
                },
                "provenance": {
                    "whole_hand_frame_authoritative": True,
                    "whole_hand_frame_definition": str(AUTHORITATIVE_REFERENCES["whole_hand_frame"]),
                    "wrist_orientation_reference_authoritative": False,
                    "selected_prediction_archive": str(prediction_path),
                    "selected_prediction_archive_sha256": sha256_file(prediction_path),
                    "disjoint_probe_chunks": True,
                },
            }
            rows[method].append(evaluate_episode(episode, joint_ranges_rad=joint_ranges_rad))
    provenance = {
        "offline_result": str(OFFLINE_RESULT),
        "offline_result_sha256": sha256_file(OFFLINE_RESULT),
        "heldout_manifest": str(HELDOUT),
        "heldout_manifest_sha256": sha256_file(HELDOUT),
        "all_eight_identities_required": True,
    }
    return (
        _result("a", "offline_act", rows["a"], provenance),
        _result("b", "offline_act", rows["b"], provenance),
    )


def evaluate_physical_batch(
    rollout_root: Path, *, joint_ranges_rad: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score all 8x2 physical bundles; a missing identity is a hard error."""

    rollout_root = Path(rollout_root).resolve()
    manifest = _read(HELDOUT)
    rows: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    config_hashes: set[str] = set()
    for output_episode, entry in enumerate(manifest["entries"]):
        identity = str(entry["stable_episode_id"])
        for method in ("a", "b"):
            path = rollout_root / method / f"heldout_{output_episode:02d}_{identity}"
            bundle_path = path / "evaluation_bundle.json"
            report_path = path / "rollout_report.json"
            if not bundle_path.is_file() or not report_path.is_file():
                raise FileNotFoundError(
                    f"physical A/B comparison cannot exclude missing identity {identity}: {path}"
                )
            report = _read(report_path)
            config_hashes.add(str(report["rigid_proxy_config_sha256"]))
            evaluated = evaluate_bundle(bundle_path, joint_ranges_rad)
            if evaluated["episode_count"] != 1:
                raise RuntimeError(f"physical rollout bundle must contain one episode: {bundle_path}")
            row = evaluated["episodes"][0]
            if row["source_episode_id"] != identity:
                raise RuntimeError(f"physical bundle source identity mismatch: {bundle_path}")
            row["provenance"].update(
                {
                    "rollout_report": str(report_path),
                    "rollout_report_sha256": sha256_file(report_path),
                    "rollout_status": report["status"],
                    "physics_trace_valid": report["physics_trace_valid"],
                }
            )
            rows[method].append(row)
    if len(config_hashes) != 1:
        raise RuntimeError(f"physical A/B rollouts used different object configs: {config_hashes}")
    provenance = {
        "rollout_root": str(rollout_root),
        "heldout_manifest": str(HELDOUT),
        "heldout_manifest_sha256": sha256_file(HELDOUT),
        "rigid_proxy_config_sha256": next(iter(config_hashes)),
        "all_sixteen_rollouts_required": True,
        "unmatched_or_missing_episode_exclusion_allowed": False,
    }
    return (
        _result("a", "physical", rows["a"], provenance),
        _result("b", "physical", rows["b"], provenance),
    )
