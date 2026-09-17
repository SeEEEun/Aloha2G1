#!/usr/bin/env python3
"""Render representative raw ACT-A40/B40 chunks before temporal ensembling.

These short kinematic reviews are a policy-quality control for Experiment 2.
They repeat the single held-out source observation that generated each chunk
and apply no temporal ensemble, smoothing, controller, or physics.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.doll_handoff_retargeting.render import ReviewRenderer
    from tools.prepare_paper_core_static_figures import label_source
except ModuleNotFoundError:  # Direct ``python tools/<script>.py`` execution.
    from doll_handoff_retargeting.common import load_common_config, load_scene
    from doll_handoff_retargeting.models import G1Kinematics
    from doll_handoff_retargeting.render import ReviewRenderer
    from prepare_paper_core_static_figures import label_source


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PAPER = ROOT / "outputs/paper_core_ab"
RESULT = PAPER / "offline_heldout8/experiment2_result.json"
MANIFEST = PAPER / "heldout8_manifest.json"
OUTPUT = PAPER / "offline_heldout8/raw_chunk_visual_review"
PHASES = ("initial_approach", "dual_contact_transfer", "release")
FPS = 30.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_selected(result: dict[str, Any], method: str) -> dict[str, np.ndarray]:
    path = Path(result["methods"][method]["selected_prediction_archive"])
    with np.load(path, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    if values["prediction"].shape != (72, 50, 28):
        raise RuntimeError(f"selected prediction shape changed: {path}")
    return values


def source_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, value = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode source frame {index}: {path}")
        return value
    finally:
        capture.release()


def trajectory(
    q: np.ndarray,
    names: list[str],
    g1: G1Kinematics,
) -> dict[str, np.ndarray]:
    lookup = {name: index for index, name in enumerate(names)}
    arm = np.asarray([lookup[str(name)] for name in g1.arm_joint_names], dtype=np.int64)
    hands = {
        side: np.asarray([lookup[name] for name in g1.hand_joint_names[side]], dtype=np.int64)
        for side in ("left", "right")
    }
    if len(set(np.concatenate((arm, hands["left"], hands["right"])).tolist())) != 28:
        raise RuntimeError("named G1 mapping is not bijective")
    grasp = {side: [] for side in ("left", "right")}
    for value in q:
        g1.assign(value[arm], value[hands["left"]], value[hands["right"]])
        for side in ("left", "right"):
            grasp[side].append(
                g1.model_to_world_position(
                    g1.whole_hand_grasp_pose(side)[:3, 3]
                )
            )
    left = np.asarray(grasp["left"], dtype=np.float64)
    right = np.asarray(grasp["right"], dtype=np.float64)
    return {
        "g1_arm_qpos": q[:, arm].astype(np.float64),
        "left_dex3_qpos": q[:, hands["left"]].astype(np.float64),
        "right_dex3_qpos": q[:, hands["right"]].astype(np.float64),
        "achieved_left_physical_grasp_frame_position_world": left,
        "achieved_right_physical_grasp_frame_position_world": right,
        "inter_grasp_frame_distance_m": np.linalg.norm(right - left, axis=1),
        "left_hand_phase": np.asarray(["RAW_ACT"] * len(q)),
        "right_hand_phase": np.asarray(["RAW_ACT"] * len(q)),
        "ik_success_per_frame": np.ones(len(q), dtype=bool),
    }


def relabel_raw_panel(
    image: np.ndarray, title: str, phase: str, offset: int
) -> np.ndarray:
    value = image.copy()
    cv2.rectangle(value, (0, 0), (value.shape[1], 106), (14, 14, 18), -1)
    cv2.putText(
        value,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        value,
        f"{phase} | raw chunk offset {offset:02d}/49 | {offset / FPS:.2f}s",
        (10, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        value,
        "NO temporal ensemble | NO smoothing | NO controller",
        (10, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (80, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        value,
        "kinematic policy-output visualization; IK/contact not scored",
        (10, 99),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (100, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return value


def main() -> None:
    result = read_json(RESULT)
    manifest = read_json(MANIFEST)
    if result["status"] != "PASS":
        raise RuntimeError("Experiment 2 must be complete before raw review rendering")
    final_episode = int(manifest["representative_episode"])
    entry = next(
        row
        for row in manifest["entries"]
        if int(row["final_dataset_index"]) == final_episode
    )
    source_path = Path(entry["source_rgb_identity"]["canonical_video_path"])
    if sha256_file(source_path) != entry["source_rgb_identity"]["canonical_video_sha256"]:
        raise RuntimeError("representative source video changed")
    selected = {method: load_selected(result, method) for method in ("a", "b")}
    names = result["input_audit"]["feature_names"]
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    renderer = ReviewRenderer(common, scene, g1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = {}
    contact_rows = []
    try:
        for phase in PHASES:
            indices = {}
            chunks = {}
            frames = {}
            for method in ("a", "b"):
                matches = np.flatnonzero(
                    (selected[method]["final_episode"].astype(np.int64) == final_episode)
                    & (selected[method]["phase"].astype(str) == phase)
                )
                if len(matches) != 1:
                    raise RuntimeError(f"raw probe is not unique: {method} {phase}")
                indices[method] = int(matches[0])
                chunks[method] = selected[method]["prediction"][matches[0]].astype(np.float64)
                frames[method] = int(selected[method]["frame"][matches[0]])
            if frames["a"] != frames["b"]:
                raise RuntimeError(f"A/B source probe frame differs: {phase}")
            probe_frame = frames["a"]
            source = label_source(
                source_frame(source_path, probe_frame),
                entry["original_source_recording_id"],
                probe_frame,
                probe_frame / FPS,
            )
            trajectories = {
                "a": trajectory(chunks["a"], names, g1),
                "b": trajectory(chunks["b"], names, g1),
            }
            path = OUTPUT / f"raw_act_a_vs_act_b_{phase}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 480)
            )
            if not writer.isOpened():
                raise RuntimeError(f"could not create {path}")
            sampled = {}
            try:
                for offset in range(50):
                    panels = [
                        relabel_raw_panel(
                            renderer.robot_panel(
                            "baseline",
                            trajectories["a"],
                            {"status": "PASS"},
                            {"frames": {}},
                            f"raw ACT-A40 | {phase}",
                            offset,
                            offset / FPS,
                            "overview",
                        ),
                            "RAW ACT-A40",
                            phase,
                            offset,
                        ),
                        relabel_raw_panel(
                            renderer.robot_panel(
                            "proposed",
                            trajectories["b"],
                            {"status": "PASS"},
                            {"frames": {}},
                            f"raw ACT-B40 | {phase}",
                            offset,
                            offset / FPS,
                            "overview",
                        ),
                            "RAW ACT-B40",
                            phase,
                            offset,
                        ),
                    ]
                    combined = np.hstack((source, *panels))
                    writer.write(combined)
                    if offset in (0, 12, 24, 36, 49):
                        sampled[offset] = cv2.resize(
                            combined, (960, 240), interpolation=cv2.INTER_AREA
                        )
            finally:
                writer.release()
            contact_rows.append(np.hstack([sampled[index] for index in (0, 12, 24, 36, 49)]))
            records[phase] = {
                "source_frame": probe_frame,
                "source_timestamp_seconds": probe_frame / FPS,
                "video": str(path),
                "video_sha256": sha256_file(path),
                "frames": 50,
                "duration_seconds": 50 / FPS,
            }
    finally:
        renderer.close()
    contact = OUTPUT / "raw_act_a_vs_act_b_temporal_strips.png"
    if not cv2.imwrite(str(contact), np.vstack(contact_rows)):
        raise RuntimeError(f"could not write {contact}")
    output = {
        "schema_version": "paper_core_raw_act_chunk_visual_review_v1",
        "status": "PASS",
        "representative_source_final_episode": final_episode,
        "representative_selection_rule": manifest[
            "representative_episode_rule_frozen_before_policy_results"
        ],
        "records": records,
        "temporal_strips": {
            "path": str(contact),
            "sha256": sha256_file(contact),
            "sampled_chunk_offsets": [0, 12, 24, 36, 49],
        },
        "raw_prediction": True,
        "temporal_ensemble": False,
        "smoothing": False,
        "physics": False,
        "execution_adapter": False,
        "physical_manipulation_success_claimed": False,
    }
    atomic_json(OUTPUT / "raw_chunk_visual_review_assets.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
