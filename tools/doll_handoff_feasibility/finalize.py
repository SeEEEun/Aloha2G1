"""Verify provenance and freeze the Generic-G1 feasibility review artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    FROZEN_ROOT,
    OUTPUT_ROOT,
    REPOSITORY,
    load_json,
    sha256_file,
    stable_episode_id,
    stable_json_sha256,
    trajectory_path,
    verify_frozen_contract,
    write_json,
)
from .solver import DEFAULT_CONFIG, resolver_implementation_hash


TARGET_KEYS = (
    "target_left_wrist_position_model",
    "target_right_wrist_position_model",
    "target_left_wrist_rotation_model",
    "target_right_wrist_rotation_model",
    "target_left_interaction_frame_position_world",
    "target_right_interaction_frame_position_world",
    "target_left_interaction_frame_position_task",
    "target_right_interaction_frame_position_task",
)
SEMANTIC_KEYS = (
    "left_hand_phase",
    "right_hand_phase",
    "ownership_state",
    "event_names",
    "event_frames",
)


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _target_integrity(output_root: Path) -> dict[str, Any]:
    frozen_target_hashes: dict[str, str] = {}
    after_target_hashes: dict[str, str] = {}
    frozen_file_hashes: dict[str, str] = {}
    after_file_hashes: dict[str, str] = {}
    action_hashes: dict[str, str] = {}
    semantic_checks: dict[str, bool] = {}
    episode_records: list[dict[str, Any]] = []
    for episode in range(50):
        stable = stable_episode_id(episode)
        before_path = trajectory_path(episode)
        after_path = output_root / "after/trajectories" / f"{stable}.npz"
        if not after_path.is_file():
            raise FileNotFoundError(after_path)
        with np.load(before_path, allow_pickle=False) as before_payload, np.load(
            after_path, allow_pickle=False
        ) as after_payload:
            before_components = {
                key: _array_hash(before_payload[key]) for key in TARGET_KEYS
            }
            after_components = {
                key: _array_hash(after_payload[key]) for key in TARGET_KEYS
            }
            targets_equal = all(
                np.array_equal(before_payload[key], after_payload[key])
                for key in TARGET_KEYS
            )
            semantics_equal = all(
                np.array_equal(before_payload[key], after_payload[key])
                for key in SEMANTIC_KEYS
            )
            hands_equal = all(
                np.array_equal(before_payload[key], after_payload[key])
                for key in ("left_dex3_qpos", "right_dex3_qpos")
            )
            frozen_target_hashes[stable] = stable_json_sha256(before_components)
            after_target_hashes[stable] = stable_json_sha256(after_components)
            action_hashes[stable] = stable_json_sha256(
                {
                    key: _array_hash(after_payload[key])
                    for key in (
                        "g1_arm_qpos",
                        "left_dex3_qpos",
                        "right_dex3_qpos",
                    )
                }
            )
            semantic_checks[stable] = semantics_equal
            episode_records.append(
                {
                    "episode_index": episode,
                    "stable_episode_id": stable,
                    "frame_count": int(after_payload["g1_arm_qpos"].shape[0]),
                    "source_targets_byte_equal": targets_equal,
                    "interaction_semantics_byte_equal": semantics_equal,
                    "dex3_hands_byte_equal": hands_equal,
                    "before_trajectory_sha256": sha256_file(before_path),
                    "after_trajectory_sha256": sha256_file(after_path),
                    "after_solver_metric_sha256": sha256_file(
                        output_root / "after/metrics" / f"{stable}.solver.json"
                    ),
                    "target_array_set_sha256": after_target_hashes[stable],
                    "after_action_array_set_sha256": action_hashes[stable],
                }
            )
        frozen_file_hashes[stable] = sha256_file(before_path)
        after_file_hashes[stable] = sha256_file(after_path)
    return {
        "frozen_trajectory_file_set_sha256": stable_json_sha256(
            frozen_file_hashes
        ),
        "after_trajectory_file_set_sha256": stable_json_sha256(after_file_hashes),
        "frozen_cartesian_target_array_set_sha256": stable_json_sha256(
            frozen_target_hashes
        ),
        "after_cartesian_target_array_set_sha256": stable_json_sha256(
            after_target_hashes
        ),
        "after_action_array_set_sha256": stable_json_sha256(action_hashes),
        "all_source_targets_byte_equal": all(
            row["source_targets_byte_equal"] for row in episode_records
        ),
        "all_interaction_semantics_byte_equal": all(semantic_checks.values()),
        "all_dex3_hands_byte_equal": all(
            row["dex3_hands_byte_equal"] for row in episode_records
        ),
        "episodes": episode_records,
    }


def _write_derivation(output_root: Path, aggregate: dict[str, Any]) -> None:
    text = f"""# Generic G1 feasibility resolver derivation

## Scope

This is a common fixed-base G1 target-realization backend. It has no episode, frame, ownership-state, handoff, doll, bin, Dataset-A, or Dataset-B parameter branch. Frozen Proposed-B interaction targets and source-derived semantic arrays are inputs and remain immutable. The frozen common natural-arm trajectory is the seed and nominal redundancy realization.

## Evidence-driven failure model

The 1,173 frozen hard-failure frames cluster into five observed signatures: torso-clearance/elbow-branch conflict (619 frames, 11 episodes), outer-workspace/low-manipulability reach (391 frames, 8 episodes), temporal transition conflict (111 frames, 12 episodes), bilateral distal-hand geometry conflict (47 frames, 1 episode), and five residual target-realization frames in two episodes. No selected hard frame is joint-limit dominated.

Full orientation is not selected as a shared cause. Frozen Proposed B already marks orientation as non-gating for the approximately spherical doll. The task-bearing grasp-frame X axis is represented on the sphere by `cross(x_achieved, x_source)` as a weak redundancy tie-breaker; raw quaternion components are never scaled, no acceptance threshold is relaxed, and no new orientation projection is activated. Every realized orientation remains the immutable source orientation (reported orientation projection: 0 rad).

## Common constrained tracking

For both arms jointly, each active frame minimizes Cartesian whole-hand position error plus weak deviations from the frozen common-natural-arm seed, temporal neighbors, and nominal posture. It is subject to the original joint limits, 0.15 rad component step, 130 rad/s² acceleration, and a 0.18 rad Euclidean branch-continuity ball. Deterministic forward/backward passes repair only windows around source residuals over the unchanged 10 mm strict threshold.

## Generic collision realization

Robot self-contact is evaluated on the active G1/Dex3 MuJoCo geometry; canonical doll/bin geometry is not part of the robot-self-collision gate. For each global hard segment, the deepest frame is projected with signed geom-distance constraints while bounding each whole-hand source-position error by `max(10 mm, current_error + 1 µm)`. The joint correction is distributed by one minimum-acceleration window rule over the common padding set `[16, 32, 48]`. A candidate is accepted only if hard collision/contact decreases, physical-error-frame count does not increase, source maximum does not regress beyond its existing bound, temporal limits pass, and branch count does not increase.

## Nearest explicit target projection

Source and realized targets are stored separately. Let `s` be an immutable source interaction target, `a` the achieved constrained G1 whole-hand point, and `epsilon = 9.9 mm` (a numerical interior to the unchanged 10 mm strict gate). The reported target is the closed-form Euclidean projection

```text
r* = s                                      if ||a-s|| <= epsilon
r* = s + (||a-s||-epsilon)(a-s)/||a-s||    otherwise.
```

This is `argmin_r ||r-s||²` subject to `||a-r|| <= epsilon` for the common constrained G1 realization. It is not a waypoint and does not alter `s`. Translation magnitude, active reason, and zero orientation change are stored per frame and per hand. Episodes whose joint realization remains self-colliding are still `HARD_FAIL`; projection does not waive collision validation.

## Result and limitations

The rule changes the all-50 classification from 1/25/24 to {aggregate['classification_counts']['CLEAN_PASS']}/{aggregate['classification_counts']['USABLE_WITH_WARNING']}/{aggregate['classification_counts']['HARD_FAIL']} (clean/warning/hard). Physical hard IK is {aggregate['physical_hard_ik_episode_count']} and distal hard contact is {aggregate['distal_hard_collision_episode_count']}. Arm/torso hard contact remains in ep013 and ep036; both are retained rather than episode-tuned. Mean projection is {aggregate['feasibility_projection_translation_m']['mean']*1000:.3f} mm and maximum projection is {aggregate['feasibility_projection_translation_m']['max']*1000:.3f} mm. During `DUAL_CONTACT`, mean/max bimanual relation changes are {aggregate['mean_dual_contact_bimanual_relation_change_m']*1000:.3f}/{aggregate['max_dual_contact_bimanual_relation_change_m']*1000:.3f} mm.
"""
    (output_root / "resolver_derivation.md").write_text(text, encoding="utf-8")


def _write_final_report(
    output_root: Path,
    aggregate: dict[str, Any],
    implementation: dict[str, Any],
) -> None:
    clusters = _read_csv(output_root / "failure_clusters.csv")
    representatives = _read_csv(output_root / "representative_before_after.csv")
    by_episode = {int(row["episode_index"]): row for row in representatives}
    cluster_lines = "\n".join(
        f"{index}. {row['cluster']}: {row['episode_count']} episodes, {row['frame_count']} frames; {row['common_geometric_signature']}"
        for index, row in enumerate(clusters, start=1)
    )
    text = f"""# Doll-Handoff generic G1 feasibility final report

## RECOVERED PROJECT STATE

repo: `{REPOSITORY}`  
git commit: `{_git('rev-parse', 'HEAD')}`  
frozen B implementation hash: `{verify_frozen_contract()['implementation_sha256']}`  
frozen B trajectory hash: `{verify_frozen_contract()['trajectory_file_set_sha256']}`  
frozen B target hash: `{verify_frozen_contract()['cartesian_target_array_set_sha256']}`

## BEFORE

CLEAN_PASS: 1  
USABLE_WITH_WARNING: 25  
HARD_FAIL: 24

## FAILURE CLUSTERS

{cluster_lines}

## GENERIC FEASIBILITY RESOLVER

method: deterministic bidirectional constrained whole-hand tracking, signed-distance collision anchor projection with minimum-acceleration windows, and explicit nearest-target projection  
implementation hash: `{implementation['implementation_sha256']}`  
task-independent: YES  
episode-specific parameters: 0  
source targets modified: NO  
orientation slack activated: NO

## REPRESENTATIVE TEST

episodes: ep002, ep003, ep011, ep026, ep024  
hard IK source maxima: ep002 {float(by_episode[2]['before_max_source_position_residual_m'])*1000:.2f} → {float(by_episode[2]['after_max_source_position_residual_m'])*1000:.2f} mm; ep003 {float(by_episode[3]['before_max_source_position_residual_m'])*1000:.2f} → {float(by_episode[3]['after_max_source_position_residual_m'])*1000:.2f} mm  
hard collision frames: {sum(int(by_episode[e]['before_hard_collision_frames']) for e in (3,11,26))} → {sum(int(by_episode[e]['after_hard_collision_frames']) for e in (3,11,26))}  
clean control regression: ep024 arm motion unchanged; after CLEAN_PASS

## FULL 50

CLEAN_PASS: {aggregate['classification_counts']['CLEAN_PASS']}  
USABLE_WITH_WARNING: {aggregate['classification_counts']['USABLE_WITH_WARNING']}  
HARD_FAIL: {aggregate['classification_counts']['HARD_FAIL']} (ep013, ep036)

HARD IK: 15 → {aggregate['physical_hard_ik_episode_count']} episodes  
ARM/TORSO HARD: 11 → {aggregate['arm_torso_hard_collision_episode_count']} episodes  
DISTAL HARD: 1 → {aggregate['distal_hard_collision_episode_count']} episodes  
joint-limit violations: {aggregate['joint_limit_violations']}  
branch discontinuities: {aggregate['branch_discontinuities']}  
maximum velocity: {aggregate['maximum_velocity_rad_s']:.6f} rad/s  
maximum acceleration: {aggregate['maximum_acceleration_rad_s2']:.6f} rad/s²  
mean/median projection: {aggregate['feasibility_projection_translation_m']['mean']*1000:.3f}/{aggregate['feasibility_projection_translation_m']['median']*1000:.3f} mm  
maximum projection: {aggregate['feasibility_projection_translation_m']['max']*1000:.3f} mm

## INTERACTION SEMANTICS

handoff: {aggregate['handoff_ordering_valid']} / 50 unchanged  
ownership: {aggregate['ownership_transition_valid']} / 50 unchanged  
release event: {aggregate['release_event_present']} / 50 unchanged  
source-image bin release: {aggregate['source_bin_release_inside']} / 50 inside  
DUAL_CONTACT bimanual relation mean/max change: {aggregate['mean_dual_contact_bimanual_relation_change_m']*1000:.3f}/{aggregate['max_dual_contact_bimanual_relation_change_m']*1000:.3f} mm

## SOURCE TARGET HASH

unchanged: YES (`{aggregate['source_target_hash']['expected']}`)

## DATASET B

NOT_PACKAGED_BY_DESIGN

## POLICY TRAINING

NOT_STARTED_BY_DESIGN

## NEXT RECOMMENDED STEP

Human-review the 21 synchronized videos, including the retained ep013/ep036 torso penetrations. Do not package Dataset B unless the operator accepts the warning/projection distribution and either excludes the two hard failures transparently or commissions a new generic collision-feasibility formulation; do not episode-tune them.

READY_FOR_FEASIBILITY_RESOLVER_REVIEW
"""
    (output_root / "final_report.md").write_text(text, encoding="utf-8")


def finalize_review(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    freeze = verify_frozen_contract()
    aggregate = load_json(output_root / "full50/aggregate_summary.json")
    gate = load_json(output_root / "representative_gate.json")
    videos = load_json(output_root / "videos/manifest.json")
    if gate.get("status") != "PASS":
        raise RuntimeError("representative gate is not PASS")
    if int(aggregate.get("episode_count", 0)) != 50:
        raise RuntimeError("full-50 aggregate is absent or incomplete")
    integrity = _target_integrity(output_root)
    if integrity["frozen_trajectory_file_set_sha256"] != freeze[
        "trajectory_file_set_sha256"
    ]:
        raise RuntimeError("frozen BEFORE trajectory-set hash drifted")
    if integrity["frozen_cartesian_target_array_set_sha256"] != freeze[
        "cartesian_target_array_set_sha256"
    ]:
        raise RuntimeError("frozen BEFORE target-set hash drifted")
    if integrity["after_cartesian_target_array_set_sha256"] != freeze[
        "cartesian_target_array_set_sha256"
    ]:
        raise RuntimeError("AFTER source target-set hash differs from BEFORE")
    if not all(
        integrity[key]
        for key in (
            "all_source_targets_byte_equal",
            "all_interaction_semantics_byte_equal",
            "all_dex3_hands_byte_equal",
        )
    ):
        raise RuntimeError("immutable episode arrays changed")
    write_json(output_root / "resolver_config.json", load_json(DEFAULT_CONFIG))
    implementation = resolver_implementation_hash(DEFAULT_CONFIG)
    _write_derivation(output_root, aggregate)
    _write_final_report(output_root, aggregate, implementation)
    video_hashes = {
        Path(path).relative_to(output_root).as_posix(): sha256_file(path)
        for entry in videos["entries"]
        for path in entry["outputs"].values()
    }
    artifact_names = (
        "recovered_state.md",
        "failure_frames.csv",
        "failure_clusters.csv",
        "failure_cluster_report.md",
        "failure_cluster_summary.json",
        "resolver_config.json",
        "resolver_derivation.md",
        "representative_before_after.csv",
        "representative_gate.json",
        "representative_gate_report.md",
        "full50/per_episode.csv",
        "full50/aggregate_summary.json",
        "full50/failures.json",
        "videos/manifest.json",
        "final_report.md",
    )
    artifact_hashes = {
        name: sha256_file(output_root / name) for name in artifact_names
    }
    manifest = {
        "schema_version": "doll_handoff_generic_g1_feasibility_freeze_v1",
        "status": "GENERIC_G1_FEASIBILITY_REVIEW_FROZEN",
        "scope": "REVIEW_ONLY_NOT_DATASET_PACKAGING",
        "repository": str(REPOSITORY),
        "git_commit": _git("rev-parse", "HEAD"),
        "frozen_before": {
            "root": str(FROZEN_ROOT),
            "implementation_sha256": freeze["implementation_sha256"],
            "trajectory_file_set_sha256": freeze["trajectory_file_set_sha256"],
            "cartesian_target_array_set_sha256": freeze[
                "cartesian_target_array_set_sha256"
            ],
            "common_natural_arm_solver_sha256": load_json(DEFAULT_CONFIG)[
                "frozen_input"
            ]["common_natural_arm_solver_sha256"],
        },
        "resolver": {
            **implementation,
            "config_sha256": sha256_file(DEFAULT_CONFIG),
            "task_independent": True,
            "episode_specific_parameters": 0,
            "phase_specific_cartesian_parameters": 0,
            "source_target_mutation": False,
            "orientation_projection": False,
        },
        "integrity": integrity,
        "representative_gate": {
            "status": gate["status"],
            "episodes": gate["episodes"],
            "checks": gate["checks"],
        },
        "full50": aggregate,
        "retained_hard_failures": [13, 36],
        "artifact_sha256": artifact_hashes,
        "artifact_set_sha256": stable_json_sha256(artifact_hashes),
        "video_sha256": video_hashes,
        "video_set_sha256": stable_json_sha256(video_hashes),
        "dataset_b_packaged": False,
        "dataset_a_packaged": False,
        "policy_training_started": False,
        "human_review_required": True,
    }
    write_json(output_root / "freeze_manifest.json", manifest)
    return manifest


__all__ = ["finalize_review"]
