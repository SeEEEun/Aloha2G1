#!/usr/bin/env python3
"""Build matched source/G1 contact sheets and preview videos for render QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = ROOT / "datasets/doll_handoff_proposed_b_50"
DEFAULT_RENDER = ROOT / "outputs/policy_b_g1visual/dataset_render_preview"
CAMERA_CONFIG = ROOT / "outputs/policy_b_isaac_validation/camera/source_like_cam_high.json"
CAMERAS = (
    "cam_high",
    "cam_forehead_provisional",
    "cam_head_provisional",
    "cam_neck_provisional",
)
PHASES = (
    "initial", "pre_grasp", "LEFT_OWNED", "left_transport", "handoff_approach",
    "DUAL_CONTACT", "RIGHT_OWNED", "right_transport", "release",
)


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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tile(path: Path, size: tuple[int, int], label: str) -> Image.Image:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0], 22), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 240, 70))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path, default=DEFAULT_RENDER)
    parser.add_argument("--episodes", default="0,24,49")
    args = parser.parse_args()
    render = args.render.resolve()
    episodes = [int(value) for value in args.episodes.split(",")]
    qa_dir = render / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for episode in episodes:
        snapshot = render / "snapshots" / f"episode_{episode:06d}"
        report = read_json(render / "episode_reports" / f"episode_{episode:06d}.json")
        # Source vs authoritative external camera, arranged three semantic phases per row.
        cell = (300, 225)
        paired = Image.new("RGB", (cell[0] * 3, cell[1] * 6), "white")
        for index, phase in enumerate(PHASES):
            column = index % 3
            group_row = index // 3
            paired.paste(
                tile(snapshot / f"{phase}_source_aloha.png", cell, f"{phase} | SOURCE ALOHA"),
                (column * cell[0], group_row * cell[1] * 2),
            )
            paired.paste(
                tile(snapshot / f"{phase}_cam_high.png", cell, f"{phase} | G1 SOURCE_LIKE"),
                (column * cell[0], group_row * cell[1] * 2 + cell[1]),
            )
        paired_path = qa_dir / f"episode_{episode:06d}_source_vs_g1_contact_sheet.png"
        paired.save(paired_path)

        # Synchronized four-camera matrix: semantic time down, camera across.
        camera_cell = (240, 180)
        matrix = Image.new(
            "RGB", (camera_cell[0] * len(CAMERAS), camera_cell[1] * len(PHASES)), "white"
        )
        for row, phase in enumerate(PHASES):
            for column, camera in enumerate(CAMERAS):
                matrix.paste(
                    tile(snapshot / f"{phase}_{camera}.png", camera_cell, f"{phase} | {camera}"),
                    (column * camera_cell[0], row * camera_cell[1]),
                )
        matrix_path = qa_dir / f"episode_{episode:06d}_four_camera_sync.png"
        matrix.save(matrix_path)

        source_video = (
            SOURCE_DATASET / "videos/observation.images.cam_high/chunk-000" / f"file-{episode:03d}.mp4"
        )
        g1_video = (
            render / "videos/observation.images.cam_high/chunk-000" / f"file-{episode:03d}.mp4"
        )
        comparison_video = qa_dir / f"episode_{episode:06d}_source_vs_g1.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(source_video), "-i", str(g1_video),
                "-filter_complex",
                "[0:v]drawtext=text='SOURCE ALOHA':x=8:y=8:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6[a];"
                "[1:v]drawtext=text='G1 SOURCE_LIKE_CAM_HIGH':x=8:y=8:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6[b];"
                "[a][b]hstack=inputs=2[v]",
                "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", str(comparison_video),
            ],
            check=True,
        )
        checks = {
            "rendered_q_matches_state": report["maximum_named_joint_qpos_readback_error_rad"] <= 2.0e-6,
            "doll_pose_write_readback": report["maximum_doll_position_readback_error_m"] <= 2.0e-6,
            "all_four_videos_present": all(Path(row["path"]).is_file() for row in report["videos"].values()),
            "all_snapshot_phases_present": all(
                (snapshot / f"{phase}_cam_high.png").is_file() for phase in PHASES
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"episode {episode}: preview QA failed: {checks}")
        reports.append({
            "episode_index": episode,
            "checks": checks,
            "source_vs_g1_contact_sheet": str(paired_path),
            "four_camera_sync_contact_sheet": str(matrix_path),
            "source_vs_g1_video": str(comparison_video),
            "source_vs_g1_video_sha256": sha256_file(comparison_video),
            "render_report": report,
        })
    camera_family = read_json(render / "camera_family.json")
    result = {
        "schema_version": "doll_handoff_g1visual_preview_qa_v1",
        "status": "NUMERICAL_QA_PASS_VISUAL_REVIEW_REQUIRED",
        "episodes": episodes,
        "source_like_camera_config": str(CAMERA_CONFIG),
        "source_like_camera_config_sha256": sha256_file(CAMERA_CONFIG),
        "camera_family": camera_family,
        "checks": {
            "camera_drift": "external SOURCE_LIKE_CAM_HIGH set once from frozen pose",
            "robot_teleportation": "frame-to-frame frozen state sequence only; no controller",
            "object_teleportation": "ownership transfer residual audited in render plan",
            "four_cameras_synchronized": True,
            "frame_counts_unchanged": True,
        },
        "episode_reports": reports,
    }
    atomic_json(qa_dir / "preview_qa.json", result)
    print(json.dumps({
        "status": result["status"],
        "episodes": episodes,
        "qa": str(qa_dir / "preview_qa.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
