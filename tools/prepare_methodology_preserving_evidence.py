#!/usr/bin/env python3
"""Recover immutable evidence for the methodology-preserving completion run.

This tool is read-only with respect to every scientific and physics input.  It
verifies the authoritative hashes, derives descriptive SE(3) provenance from
recorded physics, and creates the new experiment evidence ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.build_doll_handoff_proxy_v2_handoff_gate import hand_model  # noqa: E402
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import authoritative_joint_ranges  # noqa: E402


OUT = ROOT / "outputs/final_methodology_preserving_completion"
EVIDENCE = OUT / "00_evidence"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
SELECTED = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp"
    / "SELECTED_RIGHT_TRANSPORT_GRASP.json"
)
R14_COMMAND = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp/candidates"
    / "R14_RELEASE_D1_CENTERED_DEEP_FAST/right_only_r6_command.npz"
)
R14_EVENT = R14_COMMAND.parent / "physics_r6/event_log.npz"
LEFT_RESULT = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/hand_calibration_v2"
    / "p14_three_digit_preload/trials/left/trial_result.json"
)
RIGHT_RESULT = LEFT_RESULT.parent.parent / "right/trial_result.json"
HANDOFF_COMMAND = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/scripted_full_task/p14_bilateral"
    / "backward_constructed_handoff/B2_PATH_F40"
    / "right_preload_partial_left_relax_exact_endpoint_v4/full"
    / "scripted_full_task_command.npz"
)
HANDOFF_EVENT = HANDOFF_COMMAND.parent / "physics_right_sensor/event_log.npz"
HANDOFF_RESULT = HANDOFF_EVENT.parent / "trial_result.json"
KEYFRAME = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/success_first_common_execution"
    / "post_handoff_keyframe/POST_HANDOFF_KEYFRAME.npz"
)
KEYFRAME_MANIFEST = KEYFRAME.parent / "POST_HANDOFF_KEYFRAME_MANIFEST.json"
ACT_A = ROOT / "outputs/paper_core_ab/act_a40/train/checkpoints/100000/pretrained_model/model.safetensors"
ACT_B = ROOT / "outputs/paper_core_ab/act_b40/train/checkpoints/020000/pretrained_model/model.safetensors"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"

EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    SELECTED: "10d406038795f2f4dfc38ecdfd408de176d63297f05dd4814fc1a4b29392b8c9",
    R14_COMMAND: "5f5db710e407f60d39f8e0138729f820fb79e3a85941caccb596574ef3ed50bf",
    R14_EVENT: "75c4c4d9da333c23f78ebd7c5ee6a3c63e35c786fbe245788c1a92fc80efe02b",
    LEFT_RESULT: "7953c9bba7f11c76658b24196206875188fd6626a8c72d653f60a49e51dc1089",
    RIGHT_RESULT: "7b4de6b2ec8075d596ce58a611bbdc28c2ac6f2c96450fe29acd427fda793e4f",
    HANDOFF_COMMAND: "2a5b1a7a118e3c0fcac33ee2aeab903b0ccf713a11878acf802ea615865728ab",
    HANDOFF_EVENT: "9508e6d8d9e91a9d8955e12d6697145896eb0fb4e7a2457bde828b4b858ab016",
    KEYFRAME: "298f68726b077ee3b8760fdfa178c5f3b81a1d2c84a6ce808395e594a536464f",
    KEYFRAME_MANIFEST: "94c4ab35951f9c3e8262bc323bbf8537ca92861dc1d3c6aa9d56282c1470d4bb",
    ACT_A: "7e9fe737c3fd8ad3919cf3887dad58a732f6651e84e1eab7c1af8267ee16912c",
    ACT_B: "4c3c52a853cc242c6ba97fa6fa8d2dde96f6d65e737e291cfb99265f3c1b5198",
    HELDOUT: "a86181b049d0f521d1167c2b58bc15f3a7cb6ad87ee9a1f634ef02c04adcc710",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda item: item.tolist()
            if isinstance(item, np.ndarray)
            else item.item()
            if isinstance(item, np.generic)
            else str(item),
        )
        + "\n",
    )


def mean_pose(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.median(position, axis=0)
    pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).mean().as_matrix()
    return pose


def longest_duration(mask: np.ndarray, dt: float) -> float:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * dt)


def main() -> int:
    actual = {str(path): sha256_file(path) for path in EXPECTED}
    mismatch = {
        str(path): {"expected": digest, "actual": actual[str(path)]}
        for path, digest in EXPECTED.items()
        if actual[str(path)] != digest
    }
    if mismatch:
        raise RuntimeError(f"immutable evidence changed: {mismatch}")

    config = read_json(CONFIG)
    selected = read_json(SELECTED)
    left = read_json(LEFT_RESULT)
    right = read_json(RIGHT_RESULT)
    handoff = read_json(HANDOFF_RESULT)
    heldout = read_json(HELDOUT)
    if selected.get("status") != "PASS" or selected["repeatability"]["successes"] != 3:
        raise RuntimeError("verified RIGHT transport grasp is not 3/3 PASS")
    if left.get("status") != "PASS" or right.get("status") != "PASS":
        raise RuntimeError("bilateral standalone evidence is not PASS")
    if handoff["runtime_right_three_digit_gate"]["status"] != "PASS":
        raise RuntimeError("recorded positive handoff evidence changed")

    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    names, _ = authoritative_joint_ranges()
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    p1_left = hand_model(g1, names, config, "left", "POWER_GRASP_P1")
    p1_right = hand_model(g1, names, config, "right", "POWER_GRASP_P1")
    right_primitive = Path(config["source_arm_primitives"]["right"])
    with np.load(right_primitive, allow_pickle=False) as archive:
        reference_arm = np.asarray(archive["approach_arm_q_rad"][-1], dtype=np.float64)
    g1.assign(reference_arm, p1_left, p1_right)
    static_tool = np.linalg.inv(g1.wrist_pose("right")) @ g1.whole_hand_grasp_pose("right")

    with np.load(R14_EVENT, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    stage = event["stage"].astype(str)
    endpoint_mask = stage == "HOLD_ELEVATED"
    endpoint_rows = np.flatnonzero(endpoint_mask)[-240:]
    endpoint_object = mean_pose(
        event["object_position_world_m"][endpoint_rows],
        event["object_quaternion_xyzw"][endpoint_rows],
    )
    endpoint_command = event["commanded_q_rad"][endpoint_rows[-1]]
    g1.assign(
        endpoint_command[arm_indices],
        endpoint_command[left_indices],
        endpoint_command[right_indices],
    )
    tool_model, tool_rotation_model, _, _ = g1.static_tool_pose_state("right", static_tool)
    tool_world = np.eye(4, dtype=np.float64)
    tool_world[:3, 3] = g1.model_to_world_position(tool_model)
    tool_world[:3, :3] = g1.model_to_world_rotation(tool_rotation_model)
    object_to_tool = np.linalg.inv(endpoint_object) @ tool_world

    force_threshold = float(config["gates"]["meaningful_digit_force_n"])
    dt = float(config["timing"]["physics_dt_s"])
    stable = stage == "GRAVITY_RETENTION"
    contact = {}
    for digit in ("thumb", "index", "middle"):
        force = event[f"{digit}_force_n"]
        active = stable & (force >= force_threshold)
        contact[digit] = {
            "mean_normal_force_n": float(np.mean(force[stable])),
            "maximum_normal_force_n": float(np.max(force[stable], initial=0.0)),
            "sustained_duration_s": longest_duration(active, dt),
            "mean_contact_point_world_m": np.nanmean(
                event[f"{digit}_contact_point_world_m"][active], axis=0
            ),
            "mean_contact_normal_world": np.nanmean(
                event[f"{digit}_contact_normal_world"][active], axis=0
            ),
        }

    keyframe_manifest = read_json(KEYFRAME_MANIFEST)
    heldout_indices = [int(entry["final_dataset_index"]) for entry in heldout["entries"]]
    evidence = {
        "schema_version": "methodology_preserving_positive_evidence_v1",
        "immutable_hashes": actual,
        "doll_contract": {
            "visual_dimensions_m": config["frozen_doll_contract"]["visual_dimensions_m"],
            "collision_dimensions_m": config["frozen_doll_contract"]["collision_dimensions_m"],
            "mass_kg": config["frozen_doll_contract"]["mass_kg"],
            "static_friction": config["frozen_doll_contract"]["static_friction"],
            "dynamic_friction": config["frozen_doll_contract"]["dynamic_friction"],
            "physics_unchanged": True,
        },
        "left_standalone": {
            "status": left["status"],
            "three_digit": left["three_meaningful_digit_contacts"],
            "retention_s": left["retention"]["measured_contact_s"],
            "measured_lift_m": left["lift"]["measured_m"],
            "elevated_hold_s": left["lift"]["elevated_hold_s"],
            "result": str(LEFT_RESULT),
        },
        "right_standalone": {
            "status": right["status"],
            "three_digit": right["three_meaningful_digit_contacts"],
            "retention_s": right["retention"]["measured_contact_s"],
            "measured_lift_m": right["lift"]["measured_m"],
            "elevated_hold_s": right["lift"]["elevated_hold_s"],
            "result": str(RIGHT_RESULT),
        },
        "verified_right_transport_grasp": {
            "descriptive_label": "VERIFIED_RIGHT_TRANSPORT_GRASP",
            "legacy_candidate_id_for_provenance_only": selected["candidate_id"],
            "right_hand_model_order_names": list(g1.hand_joint_names["right"]),
            "right_hand_7d_rad": selected["right_transport_hold_model_order_7d_rad"],
            "endpoint_object_pose_world": endpoint_object,
            "endpoint_tool_pose_world": tool_world,
            "object_to_tool_se3": object_to_tool,
            "contact_topology": selected["contact_topology"],
            "contact_measurements": contact,
            "palm_support": selected["contact_topology"]["palm_contact"],
            "right_only_metrics": selected["right_only_metrics"],
            "release_controller": selected["release_controller"],
            "repeatability": selected["repeatability"],
            "authoritative_command": str(R14_COMMAND),
            "authoritative_command_sha256": actual[str(R14_COMMAND)],
            "event_log": str(R14_EVENT),
            "event_log_sha256": actual[str(R14_EVENT)],
        },
        "positive_handoff": {
            "command": str(HANDOFF_COMMAND),
            "command_sha256": actual[str(HANDOFF_COMMAND)],
            "event_log": str(HANDOFF_EVENT),
            "event_log_sha256": actual[str(HANDOFF_EVENT)],
            "right_three_digit_gate": handoff["runtime_right_three_digit_gate"],
            "right_retained_after_left_release": True,
            "post_handoff_keyframe": str(KEYFRAME),
            "post_handoff_keyframe_sha256": actual[str(KEYFRAME)],
            "keyframe_provenance": keyframe_manifest,
            "limitation": "The successful ownership endpoint was P14-like and was not yet the verified transport grasp; later conversion lost the doll.",
        },
        "paper_inputs": {
            "act_a_checkpoint": str(ACT_A.parent),
            "act_a_model_sha256": actual[str(ACT_A)],
            "act_b_checkpoint": str(ACT_B.parent),
            "act_b_model_sha256": actual[str(ACT_B)],
            "heldout_manifest": str(HELDOUT),
            "heldout_manifest_sha256": actual[str(HELDOUT)],
            "heldout_final_dataset_indices": heldout_indices,
            "split_seed": int(heldout["split_contract"]["split_seed"]),
        },
        "prohibited_mechanism_used": False,
        "real_robot_used": False,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    atomic_json(EVIDENCE / "RECOVERED_POSITIVE_EVIDENCE.json", evidence)
    markdown = f"""# Recovered positive evidence

All evidence was read from repository artifacts and checked against SHA256 before
the new search. No scientific input, doll parameter, or policy was changed.

## Physical contract

- Visual reference: 120 × 90 × 85 mm; compressed collision proxy:
  115 × 55 × 70 mm; mass: 0.020 kg.
- Static/dynamic friction: 0.90 / 0.75; restitution: 0.
- Physics config SHA256: `{actual[str(CONFIG)]}`.

## Bilateral graspability

- LEFT: PASS, three digits, {left['retention']['measured_contact_s']:.3f} s retention,
  {1000.0 * left['lift']['measured_m']:.3f} mm lift,
  {left['lift']['elevated_hold_s']:.3f} s elevated hold.
- RIGHT: PASS, three digits, {right['retention']['measured_contact_s']:.3f} s retention,
  {1000.0 * right['lift']['measured_m']:.3f} mm lift,
  {right['lift']['elevated_hold_s']:.3f} s elevated hold.

## VERIFIED_RIGHT_TRANSPORT_GRASP

- Exact 7D vector in named model order `{list(g1.hand_joint_names['right'])}`:
  `{selected['right_transport_hold_model_order_7d_rad']}`.
- RIGHT-only grasp → lift → horizontal transport → bin descent → release → settle:
  PASS 3/3 consecutive byte-identical runs.
- Command SHA256: `{actual[str(R14_COMMAND)]}`.
- Event SHA256: `{actual[str(R14_EVENT)]}`.
- Contact topology: opposed thumb/index/middle; palm support was not required.
- The exact elevated endpoint and its object→tool SE(3) are stored in the JSON ledger.

## Genuine positive handoff evidence

- RIGHT established sustained thumb/index/middle support, LEFT fully released,
  and RIGHT retained the elevated doll independently for at least 1.0 s.
- Source command SHA256: `{actual[str(HANDOFF_COMMAND)]}`.
- Source event SHA256: `{actual[str(HANDOFF_EVENT)]}`.
- Limitation: this successful ownership state was P14-like; conversion into the
  verified transport grasp subsequently lost the object.

## Frozen paper inputs

- ACT-A40: `{actual[str(ACT_A)]}`.
- ACT-B40: `{actual[str(ACT_B)]}`.
- HELDOUT8: `{heldout_indices}`, seed `{heldout['split_contract']['split_seed']}`.

The verified transport endpoint is the authoritative first-choice handoff endpoint.
"""
    atomic_text(EVIDENCE / "RECOVERED_POSITIVE_EVIDENCE.md", markdown)
    status = """# Methodology-preserving completion status

Current gate: STAGE_A_OFFLINE_PRESENTATION_SEARCH

- Scientific A/B inputs: hash-verified and unchanged.
- Doll physics: hash-verified and unchanged.
- VERIFIED_RIGHT_TRANSPORT_GRASP: PASS 3/3 and locked as first choice.
- Full task/freeze/ACT evaluation: not yet eligible.
"""
    atomic_text(OUT / "CURRENT_STATUS.md", status)
    print(json.dumps(evidence, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
