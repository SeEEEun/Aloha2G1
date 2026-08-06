#!/usr/bin/env python3
"""Finalize numeric, articulation, and rendered-mesh parity for v12 renderfix."""
from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12_renderfix"
RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py"
OLD_RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12.py"
KEY_FRAMES = [0, 169, 216, 319, 334, 523, 695, 989]
TRAJECTORIES = {
    "exact": V12 / "position_only_exact_arm_trajectory.npz",
    "nullspace": V12 / "position_only_nullspace_arm_trajectory.npz",
}
SCENE_FILES = [
    ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json",
    ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda",
    ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def dump(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def video_probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
            "-show_entries", "format_tags=comment", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    comment = data.get("format", {}).get("tags", {}).get("comment", "{}")
    try:
        metadata = json.loads(comment)
    except json.JSONDecodeError:
        metadata = {"raw_comment": comment}
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "decoded_frames": int(stream["nb_read_frames"]),
        "fps_fraction": stream["r_frame_rate"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "metadata": metadata,
    }


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a).T @ np.asarray(b)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def centroid(mask: np.ndarray):
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return np.array([x.mean(), y.mean()], dtype=float)


def tile(rgb: np.ndarray, label: str, width: int = 320) -> np.ndarray:
    height = int(round(rgb.shape[0] * width / rgb.shape[1]))
    image = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (width, height), interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 120), 1, cv2.LINE_AA)
    return image


def render_audit_for(kind: str) -> tuple[dict, list[np.ndarray], list[np.ndarray]]:
    runtime = json.loads((OUT / f"runtime_{kind}_robot-only.json").read_text())
    data = np.load(OUT / f"rendered_keyframes_{kind}_robot-only.npz")
    labels = runtime["keyframes"][0]["segmentation_info"]["proof"]["idToLabels"]
    robot_ids = [int(key) for key, value in labels.items() if "/World/G1/" in value]
    left_wrist_ids = [
        int(key) for key, value in labels.items()
        if "/World/G1/Asset/left_wrist_yaw_link/" in value
    ]
    right_wrist_ids = [
        int(key) for key, value in labels.items()
        if "/World/G1/Asset/right_wrist_yaw_link/" in value
    ]
    images = [data[f"rgb_proof_{frame}"] for frame in KEY_FRAMES]
    instance = [data[f"instance_proof_{frame}"] for frame in KEY_FRAMES]
    masks = [np.isin(mask, robot_ids) for mask in instance]
    left_wrist_masks = [np.isin(mask, left_wrist_ids) for mask in instance]
    right_wrist_masks = [np.isin(mask, right_wrist_ids) for mask in instance]
    base = masks[0]
    base_rgb = images[0].astype(np.float32)
    rows = []
    for frame, image, mask, left_mask, right_mask in zip(
        KEY_FRAMES, images, masks, left_wrist_masks, right_wrist_masks
    ):
        union = np.logical_or(base, mask)
        intersection = np.logical_and(base, mask)
        difference = np.logical_xor(base, mask)
        robot_delta = np.abs(image.astype(np.float32) - base_rgb)[union]
        rows.append(
            {
                "frame": frame,
                "robot_mask_pixels": int(mask.sum()),
                "robot_mask_xor_pixels_from_frame_0": int(difference.sum()),
                "robot_mask_iou_with_frame_0": float(intersection.sum() / max(1, union.sum())),
                "robot_mask_centroid_xy_px": centroid(mask),
                "robot_mask_centroid_shift_from_frame_0_px": (
                    None if centroid(mask) is None or centroid(base) is None
                    else float(np.linalg.norm(centroid(mask) - centroid(base)))
                ),
                "rgb_mean_abs_difference_inside_robot_union": (
                    float(robot_delta.mean()) if robot_delta.size else 0.0
                ),
                "left_wrist_link_projected_centroid_xy_px": centroid(left_mask),
                "right_wrist_link_projected_centroid_xy_px": centroid(right_mask),
            }
        )

    def maximum_visible_pairwise(rows_, key: str) -> float:
        points = [np.asarray(row[key]) for row in rows_ if row[key] is not None]
        if len(points) < 2:
            return 0.0
        return float(max(np.linalg.norm(a - b) for a in points for b in points))

    max_left_px = maximum_visible_pairwise(rows, "left_wrist_link_projected_centroid_xy_px")
    max_right_px = maximum_visible_pairwise(rows, "right_wrist_link_projected_centroid_xy_px")
    max_xor = max(row["robot_mask_xor_pixels_from_frame_0"] for row in rows)
    all_identical = all(row["robot_mask_xor_pixels_from_frame_0"] == 0 for row in rows[1:])
    payload = {
        "trajectory": kind,
        "robot_instance_ids": robot_ids,
        "left_wrist_instance_ids": left_wrist_ids,
        "right_wrist_instance_ids": right_wrist_ids,
        "keyframes": rows,
        "maximum_robot_mask_xor_pixels": max_xor,
        "minimum_robot_mask_iou": min(row["robot_mask_iou_with_frame_0"] for row in rows),
        "maximum_visible_left_wrist_link_projected_pairwise_movement_px": max_left_px,
        "maximum_visible_right_wrist_link_projected_pairwise_movement_px": max_right_px,
        "keyframe_robot_masks_all_identical": all_identical,
        "left_arm_visible_projected_motion": max_left_px > 5.0,
        "right_arm_visible_projected_motion": max_right_px > 5.0,
        "pass": (not all_identical and max_left_px > 5.0 and max_right_px > 5.0),
    }
    return payload, images, masks


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    keyframe_runtime = {
        kind: json.loads((OUT / f"keyframe_runtime_{kind}.json").read_text())
        for kind in TRAJECTORIES
    }

    write_readback = {
        "status": "PASS",
        "tolerance_rad": 1e-6,
        "trajectories": {},
    }
    articulation = {
        "status": "PASS",
        "palm_position_tolerance_m": 0.003,
        "minimum_required_wrist_motion_m": 0.020,
        "trajectories": {},
    }
    for kind, runtime in keyframe_runtime.items():
        rows = runtime["keyframes"]
        max_readback = max(row["requested_after_render_max_error_rad"] for row in rows)
        mapping_pass = (
            runtime["mapped_joints"] == 14
            and not runtime["missing"]
            and not runtime["duplicates"]
            and not runtime["left_right_swap"]
            and not runtime["arm_order_mismatch"]
        )
        write_readback["trajectories"][kind] = {
            "trajectory_sha256": runtime["trajectory_sha256"],
            "q_key": runtime["q_key"],
            "joint_mapping": runtime["joint_mapping"],
            "joint_mapping_sha256": runtime["joint_mapping_sha256"],
            "mapped_joints": runtime["mapped_joints"],
            "missing": runtime["missing"],
            "duplicates": runtime["duplicates"],
            "maximum_requested_vs_readback_error_rad": max_readback,
            "keyframes": [
                {
                    key: row[key]
                    for key in (
                        "frame", "requested_q", "articulation_write_array", "readback_immediate",
                        "readback_after_scene_update", "readback_after_render",
                        "requested_immediate_max_error_rad", "requested_after_update_max_error_rad",
                        "requested_after_render_max_error_rad",
                    )
                }
                for row in rows
            ],
            "pass": bool(mapping_pass and max_readback <= 1e-6),
        }
        max_left_wrist = max(row["left_wrist_displacement_from_frame_0_m"] for row in rows)
        max_right_wrist = max(row["right_wrist_displacement_from_frame_0_m"] for row in rows)
        max_palm = max(
            max(row["left_palm_vs_numerical_error_m"], row["right_palm_vs_numerical_error_m"])
            for row in rows
        )
        max_rotation = max(
            max(row["left_rotation_vs_numerical_error_deg"], row["right_rotation_vs_numerical_error_deg"])
            for row in rows
        )
        max_usd_body_pos = 0.0
        max_usd_body_rot = 0.0
        for row in rows:
            for side in ("left", "right"):
                usd = row["usd_wrist_after_explicit_link_sync"][side]
                body = row["wrist_state_after_render"][side]
                max_usd_body_pos = max(
                    max_usd_body_pos,
                    float(np.linalg.norm(np.asarray(usd["position_m"]) - np.asarray(body["wrist_position_m"]))),
                )
                max_usd_body_rot = max(
                    max_usd_body_rot,
                    rotation_error_deg(np.asarray(usd["rotation"]), np.asarray(body["wrist_rotation"])),
                )
        parity_pass = (
            max_palm <= 0.003
            and max_left_wrist > 0.020
            and max_right_wrist > 0.020
            and max_usd_body_pos <= 1e-9
            and max_readback <= 1e-6
        )
        articulation["trajectories"][kind] = {
            "maximum_left_wrist_displacement_from_frame_0_m": max_left_wrist,
            "maximum_right_wrist_displacement_from_frame_0_m": max_right_wrist,
            "maximum_numerical_fk_vs_isaac_palm_position_error_m": max_palm,
            "maximum_numerical_fk_vs_isaac_wrist_rotation_error_deg": max_rotation,
            "maximum_explicit_usd_link_vs_isaac_body_position_error_m": max_usd_body_pos,
            "maximum_explicit_usd_link_vs_isaac_body_rotation_error_deg": max_usd_body_rot,
            "physics_steps": runtime["physics_steps"],
            "keyframes": rows,
            "pass": parity_pass,
        }
        if not write_readback["trajectories"][kind]["pass"]:
            write_readback["status"] = "BLOCKED_ISAAC_JOINT_WRITE"
        if not parity_pass:
            articulation["status"] = "BLOCKED_ARTICULATION_UPDATE"

    dump(OUT / "isaac_joint_write_readback_audit.json", write_readback)
    dump(OUT / "keyframe_articulation_parity.json", articulation)

    rendered = {
        "status": "PASS",
        "method": "uncolorized RTX instance-ID masks restricted to actual /World/G1 prims",
        "static_table_object_background_excluded": True,
        "trajectories": {},
    }
    rendered_images = {}
    rendered_masks = {}
    for kind in TRAJECTORIES:
        payload, images, masks = render_audit_for(kind)
        rendered["trajectories"][kind] = payload
        rendered_images[kind] = images
        rendered_masks[kind] = masks
        if not payload["pass"]:
            rendered["status"] = "BLOCKED_RENDERED_MESH_MOTION"

    proof_videos = {
        kind: video_probe(OUT / f"g1_{kind}_robot_only_motion_proof.mp4")
        for kind in TRAJECTORIES
    }
    rendered["robot_only_videos"] = proof_videos
    rendered["all_robot_only_videos_decode_to_990_frames"] = all(
        video["decoded_frames"] == 990 for video in proof_videos.values()
    )
    rendered["exact_and_nullspace_video_sha256_differ"] = (
        proof_videos["exact"]["sha256"] != proof_videos["nullspace"]["sha256"]
    )
    if not rendered["all_robot_only_videos_decode_to_990_frames"]:
        rendered["status"] = "BLOCKED_RENDERED_MESH_MOTION"
    dump(OUT / "rendered_mesh_motion_audit.json", rendered)

    contact_rows = []
    for kind in TRAJECTORIES:
        contact_rows.append(
            np.hstack(
                [tile(image, f"{kind.upper()} | frame {frame}") for image, frame in zip(rendered_images[kind], KEY_FRAMES)]
            )
        )
    cv2.imwrite(str(OUT / "keyframe_robot_only_contact_sheet.png"), np.vstack(contact_rows))

    mask_rows = []
    for kind in TRAJECTORIES:
        base = rendered_masks[kind][0]
        tiles = []
        for image, mask, frame in zip(rendered_images[kind], rendered_masks[kind], KEY_FRAMES):
            marked = image.copy()
            difference = np.logical_xor(base, mask)
            marked[difference] = (0.35 * marked[difference] + 0.65 * np.array([255, 30, 30])).astype(np.uint8)
            tiles.append(tile(marked, f"{kind.upper()} f{frame} | mask XOR {int(difference.sum())}"))
        mask_rows.append(np.hstack(tiles))
    cv2.imwrite(str(OUT / "robot_mask_difference_contact_sheet.png"), np.vstack(mask_rows))

    sealed = json.loads((OUT / "input_hash_audit.json").read_text())
    after = {path: sha256(Path(path)) for path in sealed["files"]}
    unchanged = {path: after[path] == before for path, before in sealed["files"].items()}
    sealed["status"] = "INPUT_HASHES_UNCHANGED_AFTER_RENDERFIX" if all(unchanged.values()) else "INPUT_HASH_MISMATCH"
    sealed["files_after"] = after
    sealed["byte_identical_after"] = unchanged
    sealed["all_sealed_inputs_unchanged"] = all(unchanged.values())
    dump(OUT / "input_hash_audit.json", sealed)

    with np.load(TRAJECTORIES["exact"], allow_pickle=False) as exact_npz, np.load(
        TRAJECTORIES["nullspace"], allow_pickle=False
    ) as null_npz:
        exact_q = exact_npz["g1_arm_q"]
        null_q = null_npz["g1_arm_q"]
        q_difference = np.abs(exact_q - null_q)
        target_keys = [
            "corrected_left_position_scene", "corrected_right_position_scene",
            "corrected_left_rotation_scene", "corrected_right_rotation_scene",
        ]
        targets_identical = {key: np.array_equal(exact_npz[key], null_npz[key]) for key in target_keys}

    review_names = [
        "isaaclab_position_only_exact_overview_RENDERFIX.mp4",
        "isaaclab_position_only_nullspace_overview_RENDERFIX.mp4",
        "isaaclab_position_only_nullspace_side_RENDERFIX.mp4",
        "isaaclab_position_only_nullspace_top_RENDERFIX.mp4",
        "aloha_to_g1_target_anchored_4panel_RENDERFIX.mp4",
    ]
    review_videos = {
        name: video_probe(OUT / name) for name in review_names if (OUT / name).exists()
    }
    sync_pass = (
        write_readback["status"] == "PASS"
        and articulation["status"] == "PASS"
        and rendered["status"] == "PASS"
        and sealed["all_sealed_inputs_unchanged"]
    )
    visual_sync = {
        "status": "PASS" if sync_pass else "FAIL",
        "numeric_q_motion_present": True,
        "requested_q_source_expression": "q[frame] from immutable v12 NPZ g1_arm_q",
        "robot_articulation_frame_expression": "q[frame]",
        "review_marker_frame_expression": "target_left[frame], target_right[frame], achieved_[frame]",
        "robot_only_markers_present": False,
        "robot_only_source_panel_present": False,
        "actual_g1_link_mesh_sync_source": "post-write Isaac articulation body_pos_w/body_quat_w",
        "actual_g1_link_mesh_sync_destination": "/World/G1/Asset/<body_name> in temporary render stage",
        "rtx_stale_frame_fix": {
            "use_fabric": False,
            "rtx_read_transforms_from_fabric": False,
            "explicit_update_articulations_kinematic": True,
            "explicit_actual_g1_link_usd_visual_sync": True,
            "render_context_transform_cadence_reset": True,
        },
        "physics_steps": 0,
        "q_exact_nullspace_comparison": {
            "max_abs_difference_rad": float(q_difference.max()),
            "mean_abs_difference_rad": float(q_difference.mean()),
            "differing_frames": int(np.any(q_difference > 0, axis=1).sum()),
            "differing_joints": int(np.any(q_difference > 0, axis=0).sum()),
        },
        "exact_nullspace_cartesian_targets_identical": targets_identical,
        "renderer_path": str(RENDERER.resolve()),
        "renderer_sha256": sha256(RENDERER),
        "scene_hashes_after": {str(path.resolve()): sha256(path) for path in SCENE_FILES},
        "robot_only_videos": proof_videos,
        "review_videos": review_videos,
        "all_existing_review_videos_decode_to_990_frames": all(
            item["decoded_frames"] == 990 for item in review_videos.values()
        ),
        "target_recomputed": False,
        "ik_recomputed": False,
        "orientation_optimized": False,
        "dex3_applied": False,
        "physics_used": False,
        "dds_publisher_hardware": False,
    }
    dump(OUT / "visual_frame_sync_audit.json", visual_sync)

    old_lines = OLD_RENDERER.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = RENDERER.read_text(encoding="utf-8").splitlines(keepends=True)
    patch = "".join(
        difflib.unified_diff(old_lines, new_lines, fromfile=str(OLD_RENDERER), tofile=str(RENDERER))
    )
    (OUT / "renderer_fix.patch").write_text(patch, encoding="utf-8")

    command_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        "source /home/jbnu/miniconda3/etc/profile.d/conda.sh",
        "conda activate isaaclab6",
        "",
        "# Exact overview GUI",
        "DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p "
        "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py "
        "--trajectory exact --mode review --cameras overview --gui --viz kit",
        "",
        "# Nullspace overview GUI",
        "DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p "
        "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py "
        "--trajectory nullspace --mode review --cameras overview --gui --viz kit",
        "",
        "# Nullspace side GUI",
        "DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p "
        "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py "
        "--trajectory nullspace --mode review --cameras side --gui --viz kit",
        "",
    ]
    commands = OUT / "commands.sh"
    commands.write_text("\n".join(command_lines), encoding="utf-8")
    commands.chmod(0o755)

    final_status = (
        "ISAACLAB_VISUAL_REPLAY_FIXED"
        if sync_pass and len(review_videos) == len(review_names)
        else "ROBOT_ONLY_RENDERED_MESH_GATE_PASS_REVIEW_REGENERATION_PENDING"
        if sync_pass
        else "BLOCKED_RENDERED_MESH_MOTION"
    )
    report = f"""# V12 Isaac Lab articulation visual replay fix

## Status

{final_status}

- Original v12 targets and IK NPZ files were read only; all sealed hashes are unchanged: `{sealed['all_sealed_inputs_unchanged']}`.
- Requested-vs-Isaac readback maximum error: `{max(max(row['requested_after_render_max_error_rad'] for row in keyframe_runtime[k]['keyframes']) for k in keyframe_runtime):.3e}` rad.
- Maximum numerical-FK-vs-Isaac palm position error: `{max(articulation['trajectories'][k]['maximum_numerical_fk_vs_isaac_palm_position_error_m'] for k in articulation['trajectories'])*1000:.3f}` mm.
- Exact maximum wrist displacement: left `{articulation['trajectories']['exact']['maximum_left_wrist_displacement_from_frame_0_m']*1000:.3f}` mm, right `{articulation['trajectories']['exact']['maximum_right_wrist_displacement_from_frame_0_m']*1000:.3f}` mm.
- Nullspace maximum wrist displacement: left `{articulation['trajectories']['nullspace']['maximum_left_wrist_displacement_from_frame_0_m']*1000:.3f}` mm, right `{articulation['trajectories']['nullspace']['maximum_right_wrist_displacement_from_frame_0_m']*1000:.3f}` mm.
- Actual G1 mask maximum XOR: Exact `{rendered['trajectories']['exact']['maximum_robot_mask_xor_pixels']}` pixels, Nullspace `{rendered['trajectories']['nullspace']['maximum_robot_mask_xor_pixels']}` pixels.
- Both robot-only proof videos decode to exactly 990 frames: `{rendered['all_robot_only_videos_decode_to_990_frames']}`.

## Root cause and correction

The numeric articulation state changed, but this Isaac Sim build's RTX geometry-streaming path kept reading stale articulation transforms. The corrected zero-physics-step renderer disables RTX Fabric transform reads and explicitly mirrors the post-write Isaac articulation body poses to the actual `/World/G1/Asset/<body>` link prims in the temporary composed render stage before capture. USD wrist-link poses and Isaac body poses agree to numerical precision.

## Proof videos

- `{OUT / 'g1_exact_robot_only_motion_proof.mp4'}`
- `{OUT / 'g1_nullspace_robot_only_motion_proof.mp4'}`

## Contact sheets

- `{OUT / 'keyframe_robot_only_contact_sheet.png'}`
- `{OUT / 'robot_mask_difference_contact_sheet.png'}`

## Review videos

{os.linesep.join(f'- `{item["path"]}` ({item["decoded_frames"]} frames)' for item in review_videos.values()) if review_videos else '- Regenerated review videos are gated on robot-only parity and are not yet present.'}

## Safety and scope

No target generation, IK, orientation optimization, Dex3 fitting, physics interaction, DDS, publisher, or hardware path was run.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    manifest_files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and not path.name.endswith(".incomplete"):
            manifest_files[str(path.relative_to(OUT))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "status": final_status,
        "renderer_sha256": sha256(RENDERER),
        "original_npz_hashes_unchanged": sealed["all_sealed_inputs_unchanged"],
        "files": manifest_files,
        "no_target_recomputation": True,
        "no_ik_recomputation": True,
        "no_orientation_optimization": True,
        "no_dex3": True,
        "no_physics": True,
        "no_dds_publisher_hardware": True,
    }
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({"status": final_status, "rendered_mesh": rendered["status"]}, indent=2))
    return 0 if sync_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
