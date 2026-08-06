#!/usr/bin/env python3
"""Replace only the stale G1 panels of the immutable-source v12 comparison."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/jbnu/aloha_g1_dataset")
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12_renderfix"
OLD = V12 / "aloha_to_g1_target_anchored_4panel.mp4"
EXACT = OUT / "isaaclab_position_only_exact_overview_RENDERFIX.mp4"
NULL = OUT / "isaaclab_position_only_nullspace_overview_RENDERFIX.mp4"
OUTPUT = OUT / "aloha_to_g1_target_anchored_4panel_RENDERFIX.mp4"
RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py"
SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
EXACT_NPZ = V12 / "position_only_exact_arm_trajectory.npz"
NULL_NPZ = V12 / "position_only_nullspace_arm_trajectory.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def main() -> int:
    inputs = [OLD, EXACT, NULL]
    counts = {str(path.resolve()): frames(path) for path in inputs}
    if any(value != 990 for value in counts.values()):
        raise RuntimeError(counts)
    exact_runtime = json.loads((OUT / "runtime_exact_review.json").read_text(encoding="utf-8"))
    null_runtime = json.loads((OUT / "runtime_nullspace_review.json").read_text(encoding="utf-8"))
    if exact_runtime["joint_mapping_sha256"] != null_runtime["joint_mapping_sha256"]:
        raise RuntimeError("Exact/Nullspace Isaac joint mapping hash mismatch")
    metadata = {
        "panel_1": "raw cam_high from sealed v12 source panel; +7 action-to-observation convention unchanged",
        "panel_2": "optimized_action Stationary ALOHA replay from sealed v12 source panel",
        "panel_3": "G1 position-only Exact RENDERFIX",
        "panel_4": "G1 position-only Nullspace RENDERFIX",
        "old_v12_comparison_sha256": sha256(OLD),
        "exact_renderfix_sha256": sha256(EXACT),
        "nullspace_renderfix_sha256": sha256(NULL),
        "exact_trajectory_path": str(EXACT_NPZ.resolve()),
        "exact_trajectory_sha256": sha256(EXACT_NPZ),
        "nullspace_trajectory_path": str(NULL_NPZ.resolve()),
        "nullspace_trajectory_sha256": sha256(NULL_NPZ),
        "renderer_sha256": sha256(RENDERER),
        "active_scene_sha256": sha256(SCENE),
        "q_key": "g1_arm_q",
        "joint_mapping_sha256": exact_runtime["joint_mapping_sha256"],
        "frame_count": 990,
        "fps": 7.5,
        "physics_steps": 0,
        "dex3_applied": False,
        "real_robot_command_allowed": False,
    }
    filter_graph = (
        "[0:v]crop=960:540:0:0[raw];"
        "[0:v]crop=960:540:960:0[source];"
        "[1:v]scale=960:540[exact];"
        "[2:v]scale=960:540[null];"
        "[raw][source]hstack=inputs=2[top];"
        "[exact][null]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[out]"
    )
    temporary = OUTPUT.with_name(OUTPUT.stem + ".incomplete.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(OLD), "-i", str(EXACT), "-i", str(NULL),
            "-filter_complex", filter_graph, "-map", "[out]", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", "7.5", "-frames:v", "990",
            "-metadata", "title=ALOHA to G1 target-anchored v12 RENDERFIX",
            "-metadata", "comment=" + json.dumps(metadata, separators=(",", ":")),
            "-movflags", "+faststart", str(temporary),
        ],
        check=True,
    )
    temporary.replace(OUTPUT)
    if frames(OUTPUT) != 990:
        raise RuntimeError("composed frame count mismatch")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
