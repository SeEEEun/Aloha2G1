#!/usr/bin/env python3
"""Audit the approved Episode-49 timeline and the next fingertip-v3 gate.

This is deliberately non-generative: it does not solve IK, run physics, or
create a trajectory.  It prevents an unrelated thumb/index phone-proxy solver
from being presented as right-middle-finger ring insertion evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_fingertip_semantic_v3"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
MAPPING = ROOT / "configs/dex3_abc_finger_mapping.sim.json"
TIPS = ROOT / "configs/dex3_fingertip_frames.sim.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name: str, value: dict) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    timeline = json.loads(TIMELINE.read_text())
    mapping = json.loads(MAPPING.read_text())
    tips = json.loads(TIPS.read_text())
    events = {x["event"]: x for x in timeline["events"]}
    required = ("left_arm_return_near_home", "task_end")
    for name in required:
        event = events[name]
        assert event["frame"] == 702
        assert event["timestamp_s"] == 23.4
        assert event["source"] == "manual_video_review"
        assert event["approval"] == "APPROVED_BY_USER"
        assert event["scope"] == "EPISODE_49_ONLY"
    assert timeline["frame_range"] == [0, 989] and timeline["fps"] == 30.0
    assert 702 / timeline["fps"] == 23.4
    ordered = sorted(x["frame"] for x in timeline["events"])
    assert all(a <= b for a, b in zip(ordered, ordered[1:]))
    with np.load(SOURCE) as z:
        action = z["optimized_action"]
        timestamp = z["timestamp"]
        assert action.shape == (990, 14) and timestamp.shape == (990,)
        assert np.isfinite(action).all() and np.isfinite(timestamp).all()
        assert float(z["fps"]) == 30.0

    assert mapping["left"]["A"]["digit_chain"] == "thumb"
    assert mapping["left"]["B"]["digit_chain"] == "index"
    assert mapping["left"]["C"]["digit_chain"] == "middle"
    assert mapping["right"]["A"]["digit_chain"] == "index"
    assert mapping["right"]["B"]["digit_chain"] == "thumb"
    assert mapping["right"]["C"]["digit_chain"] == "middle"

    right_c = tips["fingertips"]["right_C"]
    transverse = sorted(float(x) for x in right_c["pad_half_extent_m"])[1:]
    inner_radius = 0.0225
    raw_radial_clearance = inner_radius - max(transverse)
    timeline_gate = {
        "status": "PASS",
        "artifact": str(TIMELINE),
        "sha256": sha(TIMELINE),
        "frame_count": 990,
        "fps": 30.0,
        "new_manual_events": {name: {k: events[name][k] for k in
            ("frame", "timestamp_s", "source", "approval", "scope")} for name in required},
        "chronology_validation": "PASS_NON_DECREASING",
        "equal_frame_702_intentional": True,
        "invented_event_frames": False,
    }
    insertion_gate = {
        "status": "BLOCKED_RIGHT_C_INSERTION",
        "right_C_digit": "middle",
        "ring_inner_radius_m": inner_radius,
        "right_C_pad_transverse_half_extents_m": transverse,
        "raw_geometry_radial_clearance_m": raw_radial_clearance,
        "raw_aperture_fit_precheck": raw_radial_clearance > 0,
        "continuous_collision_checked_insertion_candidate": None,
        "inner_rim_hook_candidate": None,
        "reason": (
            "No repository solver validates right middle-finger continuous annulus insertion, "
            "phone non-penetration, and inner-rim hook contact. Existing static phone-grasp "
            "solver targets thumb/index against a torso-local box and is not valid evidence."
        ),
        "acceptance_gate_relaxed": False,
    }
    write_json("candidate_search_results.json", {
        "status": "BLOCKED_BEFORE_TRAJECTORY_GENERATION",
        "classification": "BLOCKED_RIGHT_C_INSERTION",
        "timeline_gate": timeline_gate,
        "mapping_gate": "PASS",
        "fingertip_frame_gate": "PASS_ACTIVE_COLLISION_GEOMETRY",
        "right_C_ring_insertion_gate": insertion_gate,
    })
    write_json("selected_candidate.json", {
        "selected_candidate": None,
        "status": "FINGERTIP_SEMANTIC_CANDIDATE_NOT_READY",
        "classification": "BLOCKED_RIGHT_C_INSERTION",
        "trajectory_generated": False,
        "ik_run": False,
        "physics_run": False,
        "isaaclab_replay_run": False,
        "authoritative_for_real_robot": False,
        "real_robot_command_allowed": False,
    })
    write_json("run_manifest.json", {
        "status": "BLOCKED_RIGHT_C_INSERTION",
        "diagnostic_only": True,
        "simulation_only": True,
        "timeline_gate": "PASS",
        "timeline_sha256": sha(TIMELINE),
        "source_sha256": sha(SOURCE),
        "outputs_intentionally_not_created": [
            "aloha_fk_source_trajectory.npz", "semantic_contact_anchors.json",
            "phone_pose_trajectory.npz", "accessory_pose_trajectory.npz",
            "fingertip_target_trajectory.npz", "g1_arm_fingertip_semantic_trajectory.npz",
            "g1_full_arm_dex3_fingertip_trajectory.npz", "fixed-object reach video",
            "semantic object-follow video", "Isaac Lab replay",
        ],
        "why_not_created": insertion_gate["reason"],
        "no_dds": True, "no_publisher": True, "no_hardware_client": True,
        "physics_run": False, "real_robot_safety": "NOT_PERFORMED",
    })
    print("TIMELINE_GATE=PASS")
    print("PIPELINE_STATUS=BLOCKED_RIGHT_C_INSERTION")
    print(f"RIGHT_C_RAW_RADIAL_CLEARANCE_M={raw_radial_clearance:.9f}")
    print("TRAJECTORY_GENERATED=false")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
