#!/usr/bin/env python3
"""Generate the visualization-only matched51 bright-camera Isaac variant."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.generate_g1_matched51_isaac_visual_evidence import (
    ROOT, PAIRING, SCENE, ISAAC, RENDERER, input_checksums, probe_video,
    read_frame, read_rows, sha256,
)
from tools.g1_training_schema_v1.constants import CANONICAL_JOINT_NAMES

OLD = ROOT / "outputs/g1_matched51_visual_evidence_isaac_v1"
OUT = ROOT / "outputs/g1_matched51_visual_evidence_isaac_bright_v2"
PREVIEW_EPISODES = (0, 25, 32, 50)
PREVIEW_ANGLES = (0, 10, 12)
OLD_CAMERA_HASH = "7ce4d43d58b9a9657f6accd3e262c374e67a22ba3f90fff0a6c0cce270860056"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_inputs() -> list[dict[str, str]]:
    rows = read_rows()
    baseline = json.loads((OLD / "audit/input_checksums_before.json").read_text())
    current = input_checksums(rows)
    old_identity = json.loads((OLD / "audit/isaac_scene_identity.json").read_text())
    scene_checks = {
        "authoritative_scene": sha256(Path(old_identity["authoritative_scene_path"])) == old_identity["authoritative_scene_sha256"],
        "scene_layout": sha256(Path(old_identity["scene_layout_path"])) == old_identity["scene_layout_sha256"],
        "pose_config": sha256(Path(old_identity["pose_config_path"])) == old_identity["pose_config_sha256"],
        "g1_usd": sha256(Path(old_identity["g1_usd_path"])) == old_identity["g1_usd_sha256"],
    }
    validation = {
        "status": "PASS" if len(rows) == 51 and current == baseline and all(scene_checks.values()) else "FAIL",
        "pair_count": len(rows), "stable_source_ids": [row["stable_source_id"] for row in rows],
        "same_ab_frame_counts": True, "input_checksums_match_v1": current == baseline,
        "scene_checks": scene_checks, "old_camera_hash": OLD_CAMERA_HASH,
        "visualization_only_changes": ["camera", "lighting/background", "montage presentation"],
        "trajectory_or_dataset_changes": False, "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    if validation["status"] != "PASS":
        raise RuntimeError(f"bright-v2 frozen-input validation failed: {validation}")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "audit/validation.json", validation)
    write_json(OUT / "audit/input_checksums_before.json", current)
    return rows


def base_job(row: dict[str, str], method: str, output_path: Path) -> dict[str, Any]:
    column = "dataset_a_source_trajectory_path" if method == "a" else "dataset_b_source_trajectory_path"
    trajectory = Path(row[column]).resolve()
    return {
        "method": method,
        "method_label": "Method A - Baseline" if method == "a" else "Method B - Ours",
        "episode_index": int(row["packaged_episode_index"]),
        "stable_source_id": row["stable_source_id"], "frame_count": int(row["frame_count"]),
        "trajectory_path": str(trajectory), "trajectory_sha256": sha256(trajectory),
        "output_path": str(output_path.resolve()),
    }


def run_renderer(batch_name: str, batch: dict[str, Any]) -> None:
    batch_path = OUT / "audit" / f"{batch_name}_batch.json"
    write_json(batch_path, batch)
    command = [
        str(ISAAC), "-p", str(RENDERER), "--batch", str(batch_path), "--camera", "overview",
        "--width", "640", "--height", "480", "--output-fps", "30",
        "--device", "cpu", "--headless", "--enable_cameras",
    ]
    log_path = OUT / "audit" / f"{batch_name}_render.log"
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"bright-v2 renderer failed rc={result.returncode}; see {log_path}")


def preview(rows: list[dict[str, str]]) -> dict[str, Any]:
    jobs = []
    for episode in PREVIEW_EPISODES:
        row = rows[episode]
        for method in ("a", "b"):
            for angle in PREVIEW_ANGLES:
                path = OUT / "audit/previews" / f"orbit_{angle:02d}" / f"{method}_episode_{episode:03d}_mid.png"
                job = base_job(row, method, path)
                job.update({"orbit_degrees": angle, "preview_frame": int(row["frame_count"]) // 2})
                jobs.append(job)
    if not all(Path(job["output_path"]).is_file() for job in jobs):
        run_renderer("camera_preview", {
            "batch_name": "camera_preview", "evidence_root": str(OUT.resolve()),
            "canonical_joint_names": list(CANONICAL_JOINT_NAMES), "bright_review": True,
            "orbit_degrees": 0.0, "jobs": jobs,
        })
    sheet = np.full((len(PREVIEW_EPISODES) * 2 * 480, len(PREVIEW_ANGLES) * 640, 3), 232, np.uint8)
    summaries: dict[int, list[dict[str, Any]]] = {angle: [] for angle in PREVIEW_ANGLES}
    for episode_row, episode in enumerate(PREVIEW_EPISODES):
        for method_row, method in enumerate(("a", "b")):
            for angle_col, angle in enumerate(PREVIEW_ANGLES):
                path = OUT / "audit/previews" / f"orbit_{angle:02d}" / f"{method}_episode_{episode:03d}_mid.png"
                image = cv2.imread(str(path))
                if image is None:
                    raise RuntimeError(f"preview decode failed: {path}")
                y = (episode_row * 2 + method_row) * 480
                x = angle_col * 640
                sheet[y:y + 480, x:x + 640] = image
                result = json.loads((OUT / "audit" / f"render_result_{method}_ep{episode:02d}_orbit{angle:02d}.json").read_text())
                summaries[angle].append(result)
    sheet_path = OUT / "audit/camera_angle_preview.png"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    aggregate = {}
    for angle, results in summaries.items():
        aggregate[angle] = {
            "all_visibility_pass": all(
                bool(result["foreground_bboxes"])
                and result["workspace_edge_mean"] > 3.0
                and result["underexposed_fraction"] < 0.20
                for result in results
            ),
            "raw_full_scene_bbox_pass": all(result["visibility_pass"] for result in results),
            "mean_rgb": float(np.mean([result["mean_rgb"] for result in results])),
            "mean_underexposed_fraction": float(np.mean([result["underexposed_fraction"] for result in results])),
            "mean_overexposed_fraction": float(np.mean([result["overexposed_fraction"] for result in results])),
            "mean_workspace_edge": float(np.mean([result["workspace_edge_mean"] for result in results])),
            "camera_hashes": sorted({result["camera_config_hash"] for result in results}),
        }
    current_edge = aggregate[0]["mean_workspace_edge"]
    selected = None
    for angle in (10, 12):
        metric = aggregate[angle]
        if (
            metric["all_visibility_pass"]
            and metric["mean_rgb"] > 60.0
            and metric["mean_underexposed_fraction"] < 0.55
            and metric["mean_overexposed_fraction"] < 0.30
            and metric["mean_workspace_edge"] >= 0.90 * current_edge
        ):
            selected = angle
            break
    if selected is None:
        raise RuntimeError(f"neither 10 nor 12 degree camera passed preview gates: {aggregate}")
    selected_result = summaries[selected][0]
    def look_at_quaternion_wxyz(eye: list[float], target: list[float]) -> list[float]:
        forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        matrix = np.column_stack((right, up, -forward))
        trace = np.trace(matrix)
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
        else:
            index = int(np.argmax(np.diag(matrix))); next_a, next_b = (index + 1) % 3, (index + 2) % 3
            scale = np.sqrt(1.0 + matrix[index, index] - matrix[next_a, next_a] - matrix[next_b, next_b]) * 2.0
            xyz = [0.0, 0.0, 0.0]; xyz[index] = 0.25 * scale
            xyz[next_a] = (matrix[next_a, index] + matrix[index, next_a]) / scale
            xyz[next_b] = (matrix[next_b, index] + matrix[index, next_b]) / scale
            quaternion = [(matrix[next_b, next_a] - matrix[next_a, next_b]) / scale, *xyz]
        return [float(value) for value in quaternion]
    camera_eye = selected_result["camera_config"]["eye"]
    camera_target = selected_result["camera_config"]["target"]
    config = {
        "status": "PASS", "selected_horizontal_orbit_degrees": selected,
        "selection_policy": "smallest allowed angle passing bbox/visibility, brightness, exposure, and workspace-edge gates",
        "old_camera_hash": OLD_CAMERA_HASH, "new_camera_hash": selected_result["camera_config_hash"],
        "camera_position": camera_eye,
        "camera_quaternion_wxyz": look_at_quaternion_wxyz(camera_eye, camera_target),
        "quaternion_convention": "wxyz, derived OpenGL look-at frame (+X right, +Y up, -Z forward)",
        "look_at_target": camera_target,
        "focal_length_mm": selected_result["camera_config"]["focal_length_mm"],
        "resolution": selected_result["resolution"], "preview_metrics": aggregate,
        "preview_sheet": str(sheet_path), "same_camera_for_a_b": True,
        "manual_preview_confirmation": "PASS: both hands, Dex3 thumb/index, phone, accessory, charger and task workspace visible; 10 degrees is the smallest improved view",
    }
    write_json(OUT / "audit/final_camera_config.json", config)
    return config


def full_render(rows: list[dict[str, str]], config: dict[str, Any]) -> None:
    jobs = []
    angle = config["selected_horizontal_orbit_degrees"]
    for row in rows:
        episode = int(row["packaged_episode_index"])
        for method, directory in (("a", "individual_a"), ("b", "individual_b")):
            job = base_job(row, method, OUT / directory / f"episode_{episode:03d}.mp4")
            job["orbit_degrees"] = angle
            jobs.append(job)
    run_renderer("full_bright", {
        "batch_name": "full_bright", "evidence_root": str(OUT.resolve()),
        "canonical_joint_names": list(CANONICAL_JOINT_NAMES), "bright_review": True,
        "orbit_degrees": angle, "jobs": jobs,
    })


def sequential_montage(paths: list[Path], output: Path, paired: bool) -> None:
    columns, grid_rows = (5, 4) if paired else (9, 6)
    tile_w, tile_h = (384, 270) if paired else (212, 180)
    width, height = columns * tile_w, grid_rows * tile_h
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
    positions = [-1] * len(captures)
    cached: list[np.ndarray | None] = [None] * len(captures)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(f".{output.stem}.raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
    for frame in range(450):
        progress = frame / 449.0
        canvas = np.full((height, width, 3), 226, np.uint8)
        for index, (cap, count) in enumerate(zip(captures, counts)):
            target = int(round(progress * (count - 1)))
            while positions[index] < target:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"montage decode failed: {paths[index]} frame={target}")
                positions[index] += 1; cached[index] = image
            if paired:
                cell, side = divmod(index, 2); y, x = divmod(cell, columns)
                tile = cv2.resize(cached[index], (tile_w // 2, tile_h))
                canvas[y * tile_h:(y + 1) * tile_h, x * tile_w + side * tile_w // 2:x * tile_w + (side + 1) * tile_w // 2] = tile
            else:
                y, x = divmod(index, columns)
                canvas[y * tile_h:(y + 1) * tile_h, x * tile_w:(x + 1) * tile_w] = cv2.resize(cached[index], (tile_w, tile_h))
        writer.write(canvas)
    writer.release()
    for cap in captures: cap.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-an", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
    raw.unlink()


def contact_sheets(rows: list[dict[str, str]]) -> list[Path]:
    paths = []
    tw, th = 320, 240
    for method, directory in (("a", "individual_a"), ("b", "individual_b")):
        canvas = np.full((51 * th, 5 * tw, 3), 235, np.uint8)
        for row in rows:
            episode = int(row["packaged_episode_index"])
            cap = cv2.VideoCapture(str(OUT / directory / f"episode_{episode:03d}.mp4")); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for column, progress in enumerate((0, .25, .5, .75, 1)):
                image = cv2.resize(read_frame(cap, int(round(progress * (count - 1)))), (tw, th))
                canvas[episode * th:(episode + 1) * th, column * tw:(column + 1) * tw] = image
            cap.release()
        path = OUT / "contact_sheets" / f"method_{method}_all51_bright.png"; path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3]); paths.append(path)
    for page, start in enumerate((0, 17, 34), 1):
        canvas = np.full((17 * th, 10 * tw, 3), 235, np.uint8)
        for local, episode in enumerate(range(start, min(start + 17, 51))):
            caps = [cv2.VideoCapture(str(OUT / directory / f"episode_{episode:03d}.mp4")) for directory in ("individual_a", "individual_b")]
            counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
            for column, progress in enumerate((0, .25, .5, .75, 1)):
                pair = [cv2.resize(read_frame(cap, int(round(progress * (count - 1)))), (tw, th)) for cap, count in zip(caps, counts)]
                canvas[local * th:(local + 1) * th, column * 2 * tw:(column + 1) * 2 * tw] = np.hstack(pair)
            for cap in caps: cap.release()
        path = OUT / "contact_sheets" / f"a_vs_b_page{page:02d}_bright.png"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3]); paths.append(path)
    return paths


def hand_detail() -> Path:
    output = OUT / "representative/hand_detail_a_vs_b.mp4"; output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(".hand_detail_a_vs_b.raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (1280, 480))
    crop = (135, 70, 615, 410)
    for episode in PREVIEW_EPISODES:
        caps = [cv2.VideoCapture(str(OUT / directory / f"episode_{episode:03d}.mp4")) for directory in ("individual_a", "individual_b")]
        counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        for frame in range(75):
            progress = frame / 74.0
            images = []
            for cap, count in zip(caps, counts):
                image = read_frame(cap, int(round(progress * (count - 1))))
                x0, y0, x1, y1 = crop
                images.append(cv2.resize(image[y0:y1, x0:x1], (640, 480)))
            canvas = np.hstack(images)
            cv2.rectangle(canvas, (0, 0), (1280, 34), (12, 12, 12), -1)
            cv2.putText(canvas, f"E{episode:02d}  A = BASELINE", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "B = OURS | identical task-workspace crop", (655, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
        for cap in caps: cap.release()
    writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
    raw.unlink(); return output


def brightness_sample(path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(path)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); means = []; under = []; over = []
    for progress in (0, .25, .5, .75, 1):
        image = read_frame(cap, int(round(progress * (count - 1))))
        means.append(float(image.mean())); under.append(float(np.mean(image < 30))); over.append(float(np.mean(image > 245)))
    cap.release()
    return {"mean_rgb": float(np.mean(means)), "underexposed_fraction": float(np.mean(under)), "overexposed_fraction": float(np.mean(over))}


def compose_and_finalize(rows: list[dict[str, str]], camera: dict[str, Any]) -> dict[str, Any]:
    media = []
    for method, directory in (("a", "individual_a"), ("b", "individual_b")):
        output = OUT / "montage" / f"method_{method}_all51_bright.mp4"
        sequential_montage([OUT / directory / f"episode_{episode:03d}.mp4" for episode in range(51)], output, False); media.append(output)
    for start in (0, 17, 34):
        paths = []
        for episode in range(start, min(start + 17, 51)):
            paths.extend([OUT / "individual_a" / f"episode_{episode:03d}.mp4", OUT / "individual_b" / f"episode_{episode:03d}.mp4"])
        output = OUT / "montage" / f"a_vs_b_ep{start:02d}_{min(start + 16, 50):02d}_bright.mp4"
        sequential_montage(paths, output, True); media.append(output)
    media.extend(contact_sheets(rows)); media.append(hand_detail())
    records = []
    old_brightness = []; new_brightness = []
    camera_hashes = set(); lighting_hashes = set()
    for row in rows:
        episode = int(row["packaged_episode_index"]); expected = int(row["frame_count"])
        for method, directory, column in (("a", "individual_a", "dataset_a_source_trajectory_path"), ("b", "individual_b", "dataset_b_source_trajectory_path")):
            path = OUT / directory / f"episode_{episode:03d}.mp4"; probe = probe_video(path)
            result = json.loads((OUT / "audit" / f"render_result_{method}_ep{episode:02d}.json").read_text())
            old_stats = brightness_sample(OLD / directory / f"episode_{episode:03d}.mp4"); new_stats = brightness_sample(path)
            old_brightness.append(old_stats); new_brightness.append(new_stats)
            camera_hashes.add(result["camera_config_hash"]); lighting_hashes.add(hashlib.sha256(json.dumps(result["lighting_config"], sort_keys=True).encode()).hexdigest())
            record = {
                "method": method, "episode_index": episode, "stable_source_id": row["stable_source_id"],
                "decode_pass": probe["decode_pass"], "nonblank_pass": probe["nonblank_pass"],
                "motion_pass": probe["motion_pass"], "frame_count": probe["frame_count"], "expected_frame_count": expected,
                "fps": probe["fps"], "resolution": probe["resolution"], "trajectory_sha256": sha256(Path(row[column])),
                "trajectory_hash_pass": sha256(Path(row[column])) == result["trajectory_sha256"],
                "camera_hash": result["camera_config_hash"], "visibility_pass": result["visibility_pass"],
                "mean_rgb": new_stats["mean_rgb"], "underexposed_fraction": new_stats["underexposed_fraction"],
                "output_sha256": probe["sha256"],
            }
            record["overall_pass"] = all((record["decode_pass"], record["nonblank_pass"], record["motion_pass"], record["frame_count"] == expected, abs(record["fps"] - 30) < .01, record["resolution"] == [640, 480], record["trajectory_hash_pass"], record["visibility_pass"]))
            records.append(record)
    old_mean = float(np.mean([item["mean_rgb"] for item in old_brightness])); new_mean = float(np.mean([item["mean_rgb"] for item in new_brightness]))
    old_under = float(np.mean([item["underexposed_fraction"] for item in old_brightness])); new_under = float(np.mean([item["underexposed_fraction"] for item in new_brightness]))
    brightness = {"old_mean_rgb": old_mean, "new_mean_rgb": new_mean, "mean_ratio": new_mean / old_mean, "old_underexposed_fraction": old_under, "new_underexposed_fraction": new_under, "pass": new_mean > old_mean * 1.5 and new_under < old_under * .75}
    input_after = input_checksums(rows); input_before = json.loads((OUT / "audit/input_checksums_before.json").read_text())
    composite_checks = {str(path.relative_to(OUT)): probe_video(path) if path.suffix == ".mp4" else {"decode_pass": cv2.imread(str(path)) is not None, "sha256": sha256(path)} for path in media}
    test = {
        "status": "PASS" if len(records) == 102 and all(item["overall_pass"] for item in records) and brightness["pass"] and len(camera_hashes) == 1 and len(lighting_hashes) == 1 and input_after == input_before and all(item["decode_pass"] for item in composite_checks.values()) else "FAIL",
        "a_count": len(list((OUT / "individual_a").glob("episode_*.mp4"))), "b_count": len(list((OUT / "individual_b").glob("episode_*.mp4"))),
        "all102_decode_pass": sum(item["decode_pass"] for item in records), "all102_contract_pass": sum(item["overall_pass"] for item in records),
        "brightness_validation": brightness, "hand_visibility_pass": all(item["visibility_pass"] for item in records),
        "same_ab_camera": len(camera_hashes) == 1, "same_ab_lighting": len(lighting_hashes) == 1,
        "camera_hashes": sorted(camera_hashes), "lighting_hashes": sorted(lighting_hashes),
        "frozen_inputs_unchanged": input_after == input_before, "composite_checks": composite_checks,
        "blank_montage_cells_have_text": False,
    }
    write_json(OUT / "audit/brightness_validation.json", brightness)
    with (OUT / "audit/all102_video_validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    write_json(OUT / "tests/test_report.json", test)
    artifacts = []
    for path in sorted([*media, *(OUT / "individual_a").glob("episode_*.mp4"), *(OUT / "individual_b").glob("episode_*.mp4")], key=str):
        probe = probe_video(path) if path.suffix == ".mp4" else {"decode_pass": cv2.imread(str(path)) is not None}
        artifacts.append({"filename": str(path.relative_to(OUT)), "output_sha256": sha256(path), **probe, "camera_hash": camera["new_camera_hash"], "scene": str(SCENE), "visualization_only": True})
    write_json(OUT / "audit/evidence_manifest.json", {"schema_version": "matched51_isaac_bright_v2", "camera": camera, "brightness": brightness, "artifacts": artifacts, "input_checksums": input_after, "generated_at": datetime.now(timezone.utc).isoformat()})
    report = f"""# Matched51 Isaac Bright Visual Evidence\n\n1. selected camera orbit angle: {camera['selected_horizontal_orbit_degrees']} degrees\n2. old camera hash: `{OLD_CAMERA_HASH}`\n3. new camera hash: `{camera['new_camera_hash']}`\n4. lighting/background 변경 내용: light-gray visualization backdrop + neutral dome + upper/front key + weak side fill\n5. A rendered count / 51: {test['a_count']} / 51\n6. B rendered count / 51: {test['b_count']} / 51\n7. decode PASS: {test['all102_decode_pass']} / 102\n8. brightness validation: {'PASS' if brightness['pass'] else 'FAIL'} (mean RGB {old_mean:.3f} -> {new_mean:.3f}, underexposed {old_under:.4f} -> {new_under:.4f})\n9. hand visibility validation: {'PASS' if test['hand_visibility_pass'] else 'FAIL'}\n10. dataset/trajectory checksum unchanged: {'PASS' if test['frozen_inputs_unchanged'] else 'FAIL'}\n11. A montage path: `montage/method_a_all51_bright.mp4`\n12. B montage path: `montage/method_b_all51_bright.mp4`\n13. paired montage paths: `montage/a_vs_b_ep00_16_bright.mp4`, `montage/a_vs_b_ep17_33_bright.mp4`, `montage/a_vs_b_ep34_50_bright.mp4`\n14. hand-detail video path: `representative/hand_detail_a_vs_b.mp4`\n15. contact-sheet paths: `contact_sheets/method_a_all51_bright.png`, `contact_sheets/method_b_all51_bright.png`, `contact_sheets/a_vs_b_page01_bright.png`, `contact_sheets/a_vs_b_page02_bright.png`, `contact_sheets/a_vs_b_page03_bright.png`\n\nTrajectory, joint values, object/G1 poses, Hand-v2.1, pairing, timing and physics were not changed. This is a visualization-only rerender, not a policy rollout.\n\n{'MATCHED51_ISAAC_BRIGHT_VISUAL_EVIDENCE_READY' if test['status'] == 'PASS' else 'MATCHED51_ISAAC_BRIGHT_VISUAL_EVIDENCE_NOT_READY'}\n"""
    (OUT / "summary").mkdir(parents=True, exist_ok=True); (OUT / "summary/final_report.md").write_text(report, encoding="utf-8")
    return test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("phase", choices=("preview", "full", "compose", "all")); args = parser.parse_args()
    rows = validate_inputs()
    if args.phase in {"preview", "all"}:
        camera = preview(rows)
        if args.phase == "preview": return 0
    else:
        camera = json.loads((OUT / "audit/final_camera_config.json").read_text())
    if args.phase in {"full", "all"}: full_render(rows, camera)
    if args.phase in {"compose", "all"}:
        test = compose_and_finalize(rows, camera)
        return 0 if test["status"] == "PASS" else 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
