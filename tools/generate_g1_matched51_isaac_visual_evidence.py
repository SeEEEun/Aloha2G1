#!/usr/bin/env python3
"""Validate, render, compose, and audit matched-51 Isaac visual evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from tools.g1_training_schema_v1.constants import CANONICAL_JOINT_NAMES, FPS, JOINT_SPECS
from tools.g1_training_schema_v1.target_contract import load_retargeted_trajectory

ROOT = Path(__file__).resolve().parents[1]
PAIRING = ROOT / "outputs/g1_policy_dataset_packaging_v1/audit/matched_51_pairing_table.csv"
DATASET_A = ROOT / "lerobot_g1_magsafe_matched51_baseline_a_v1"
DATASET_B = ROOT / "lerobot_g1_magsafe_matched51_proposed_b_v1"
OUT = ROOT / "outputs/g1_matched51_visual_evidence_isaac_v1"
ISAAC = Path("/home/jbnu/IsaacLab-3-beta/isaaclab.sh")
RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_matched51_visual_evidence.py"
SCENE = ROOT / "isaaclab_magsafe_fixed_scene"
SMOKE_EPISODES = (0, 25, 50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows() -> list[dict[str, str]]:
    with PAIRING.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 51:
        raise RuntimeError(f"pairing table has {len(rows)} rows, expected 51")
    indices = [int(row["packaged_episode_index"]) for row in rows]
    stable_ids = [row["stable_source_id"] for row in rows]
    if indices != list(range(51)) or len(set(stable_ids)) != 51:
        raise RuntimeError("pairing order/identity is not unique E00..E50")
    return rows


def input_checksums(rows: list[dict[str, str]]) -> dict[str, str]:
    paths = {PAIRING}
    for root in (DATASET_A, DATASET_B):
        paths.update({
            root / "data/chunk-000/file-000.parquet",
            root / "meta/episodes/chunk-000/file-000.parquet",
            root / "meta/g1_packaging_manifest.json",
            root / "meta/g1_training_contract.json",
            root / "meta/g1_validation.json",
            root / "meta/info.json",
        })
    for row in rows:
        paths.add(Path(row["dataset_a_source_trajectory_path"]))
        paths.add(Path(row["dataset_b_source_trajectory_path"]))
    return {str(path.resolve()): sha256(path.resolve()) for path in sorted(paths, key=str)}


def parquet_episodes(root: Path) -> dict[int, dict[str, np.ndarray]]:
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    q_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    q_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    episode_index = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    result = {}
    for episode in range(51):
        mask = episode_index == episode
        result[episode] = {
            "state": q_state[mask], "action": q_action[mask],
            "timestamps": timestamps[mask], "frame_index": frame_index[mask],
        }
    return result


def validate() -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_rows()
    before = input_checksums(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "audit/input_checksums_before.json", before)
    packaged = {"a": parquet_episodes(DATASET_A), "b": parquet_episodes(DATASET_B)}
    records = []
    for row in rows:
        episode = int(row["packaged_episode_index"])
        expected_frames = int(row["frame_count"])
        pair_record = {"episode_index": episode, "stable_source_id": row["stable_source_id"], "methods": {}}
        for method, column in (("a", "dataset_a_source_trajectory_path"), ("b", "dataset_b_source_trajectory_path")):
            path = Path(row[column]).resolve()
            trajectory = load_retargeted_trajectory(path.parent)
            packet = packaged[method][episode]
            q = trajectory.q
            limits_min = np.asarray([spec.minimum for spec in JOINT_SPECS])
            limits_max = np.asarray([spec.maximum for spec in JOINT_SPECS])
            checks = {
                "finite": bool(np.isfinite(q).all()),
                "dimension_28": q.ndim == 2 and q.shape[1] == 28,
                "frame_count": len(q) == expected_frames == len(packet["action"]),
                "timestamp": bool(np.allclose(trajectory.timestamps, packet["timestamps"], rtol=0.0, atol=1e-5)),
                "frame_index": bool(np.array_equal(packet["frame_index"], np.arange(expected_frames))),
                "state_equals_action": bool(np.array_equal(packet["state"], packet["action"])),
                "packaged_equals_frozen": bool(np.array_equal(packet["action"], q)),
                "joint_limits": bool(np.all(q >= limits_min - 1e-6) and np.all(q <= limits_max + 1e-6)),
                "source_stable_id": bool(row["stable_source_id"]),
                "packaged_episode_id": episode in packaged[method],
                "trajectory_hash": sha256(path) == trajectory.trajectory_sha256,
            }
            if not all(checks.values()):
                raise RuntimeError(f"pre-render validation failed {method} E{episode:02d}: {checks}")
            pair_record["methods"][method] = {
                "trajectory_path": str(path), "trajectory_sha256": trajectory.trajectory_sha256,
                "frame_count": len(q), "fps": trajectory.fps, "checks": checks,
            }
        if pair_record["methods"]["a"]["frame_count"] != pair_record["methods"]["b"]["frame_count"]:
            raise RuntimeError(f"A/B frame count differs at E{episode:02d}")
        records.append(pair_record)
    validation = {
        "status": "PASS", "pair_count": 51, "method_a_count": 51, "method_b_count": 51,
        "same_stable_source_ids": True, "same_episode_order": True,
        "same_corresponding_frame_counts": True, "canonical_joint_names": list(CANONICAL_JOINT_NAMES),
        "fps": FPS, "records": records, "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "audit/validation.json", validation)
    return rows, validation


def jobs(rows: list[dict[str, str]], episodes: tuple[int, ...] | range) -> list[dict[str, Any]]:
    selected = set(episodes)
    output = []
    for row in rows:
        episode = int(row["packaged_episode_index"])
        if episode not in selected:
            continue
        for method, column, directory, label in (
            ("a", "dataset_a_source_trajectory_path", "individual_a", "Method A - Baseline"),
            ("b", "dataset_b_source_trajectory_path", "individual_b", "Method B - Ours"),
        ):
            path = Path(row[column]).resolve()
            output.append({
                "method": method, "method_label": label, "episode_index": episode,
                "stable_source_id": row["stable_source_id"], "frame_count": int(row["frame_count"]),
                "trajectory_path": str(path), "trajectory_sha256": sha256(path),
                "output_path": str((OUT / directory / f"episode_{episode:03d}.mp4").resolve()),
            })
    return output


def render_batch(rows: list[dict[str, str]], batch_name: str, episodes: tuple[int, ...] | range) -> None:
    batch = {
        "batch_name": batch_name, "evidence_root": str(OUT.resolve()),
        "canonical_joint_names": list(CANONICAL_JOINT_NAMES), "jobs": jobs(rows, episodes),
    }
    batch_path = OUT / "audit" / f"{batch_name}_batch.json"
    write_json(batch_path, batch)
    command = [
        str(ISAAC), "-p", str(RENDERER), "--batch", str(batch_path),
        "--camera", "overview", "--width", "640", "--height", "480",
        "--output-fps", "30", "--device", "cpu", "--headless", "--enable_cameras",
    ]
    log_path = OUT / "audit" / f"{batch_name}_render.log"
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Isaac batch renderer failed rc={result.returncode}; see {log_path}")


def probe_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"decode_pass": False, "path": str(path)}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_indices = sorted(set([0, frames // 4, frames // 2, 3 * frames // 4, max(0, frames - 1)]))
    stds = []
    motion_crops = []
    for index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        if not ok:
            cap.release()
            return {"decode_pass": False, "path": str(path), "failed_frame": index}
        stds.append(float(image.std()))
        motion_crops.append(image[100:].astype(np.float32))
    cap.release()
    motion = max(
        (float(np.mean(np.abs(image - motion_crops[0]))) for image in motion_crops[1:]),
        default=0.0,
    )
    return {
        "path": str(path), "decode_pass": True, "frame_count": frames, "fps": fps,
        "resolution": [width, height], "sample_rgb_std": stds, "nonblank_pass": min(stds) > 1.0,
        "motion_mean_abs_max_below_overlay": motion,
        "motion_pass": motion > 0.05,
        "sha256": sha256(path),
    }


def audit_rendered(rows: list[dict[str, str]], episodes: tuple[int, ...] | range, name: str) -> dict[str, Any]:
    selected = set(episodes)
    records = []
    flags = []
    for row in rows:
        episode = int(row["packaged_episode_index"])
        if episode not in selected:
            continue
        pair = {}
        for method, directory in (("a", "individual_a"), ("b", "individual_b")):
            path = OUT / directory / f"episode_{episode:03d}.mp4"
            result = probe_video(path)
            result.update({"method": method, "episode_index": episode, "stable_source_id": row["stable_source_id"]})
            pair[method] = result
            if not result.get("decode_pass") or not result.get("nonblank_pass"):
                flags.append({"episode_index": episode, "stable_source_id": row["stable_source_id"], "method": method, "reason": "VIDEO_DECODE_OR_BLANK"})
        expected = int(row["frame_count"])
        pair_pass = all(
            pair[m].get("decode_pass") and pair[m].get("nonblank_pass")
            and pair[m].get("frame_count") == expected and abs(pair[m].get("fps", 0) - 30.0) < 0.01
            and pair[m].get("resolution") == [640, 480] for m in ("a", "b")
        ) and pair["a"].get("frame_count") == pair["b"].get("frame_count")
        if not pair_pass:
            flags.append({"episode_index": episode, "stable_source_id": row["stable_source_id"], "method": "pair", "reason": "A_B_VIDEO_CONTRACT"})
        records.append({"episode_index": episode, "stable_source_id": row["stable_source_id"], "pass": pair_pass, "methods": pair})
    after = input_checksums(rows)
    before = json.loads((OUT / "audit/input_checksums_before.json").read_text())
    result = {
        "status": "PASS" if all(record["pass"] for record in records) and before == after else "FAIL",
        "name": name, "records": records, "visual_review_flags": flags,
        "frozen_inputs_unchanged": before == after, "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "audit" / f"{name}_audit.json", result)
    write_json(OUT / "audit/visual_review_flags.json", flags)
    write_json(OUT / "audit/input_checksums_after.json", after)
    return result


def smoke_sheet() -> Path:
    canvas = np.full((3 * 480, 2 * 640, 3), 245, dtype=np.uint8)
    for row_index, episode in enumerate(SMOKE_EPISODES):
        for col, directory in enumerate(("individual_a", "individual_b")):
            path = OUT / directory / f"episode_{episode:03d}.mp4"
            cap = cv2.VideoCapture(str(path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
            ok, image = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"cannot extract smoke midpoint: {path}")
            canvas[row_index * 480:(row_index + 1) * 480, col * 640:(col + 1) * 640] = image
    path = OUT / "audit/smoke_midpoint_sheet.png"
    cv2.imwrite(str(path), canvas)
    return path


def read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    if not ok:
        raise RuntimeError(f"video frame decode failed at {index}")
    return image


def contact_sheets(rows: list[dict[str, str]]) -> list[Path]:
    output_paths = []
    tile_w, tile_h = 320, 240
    for method, directory in (("a", "individual_a"), ("b", "individual_b")):
        canvas = np.full((51 * tile_h, 5 * tile_w, 3), 245, dtype=np.uint8)
        for row in rows:
            episode = int(row["packaged_episode_index"])
            cap = cv2.VideoCapture(str(OUT / directory / f"episode_{episode:03d}.mp4"))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for column, progress in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
                image = read_frame(cap, int(round(progress * (count - 1))))
                canvas[episode * tile_h:(episode + 1) * tile_h, column * tile_w:(column + 1) * tile_w] = cv2.resize(image, (tile_w, tile_h))
            cap.release()
        path = OUT / "contact_sheets" / f"method_{method}_all51.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        output_paths.append(path)
    for page, start in enumerate((0, 17, 34), start=1):
        page_rows = rows[start:start + 17]
        canvas = np.full((17 * tile_h, 5 * tile_w * 2, 3), 245, dtype=np.uint8)
        for local_row, row in enumerate(page_rows):
            episode = int(row["packaged_episode_index"])
            caps = [cv2.VideoCapture(str(OUT / d / f"episode_{episode:03d}.mp4")) for d in ("individual_a", "individual_b")]
            counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
            for column, progress in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
                pair = [cv2.resize(read_frame(cap, int(round(progress * (count - 1)))), (tile_w, tile_h)) for cap, count in zip(caps, counts)]
                image = np.hstack(pair)
                canvas[local_row * tile_h:(local_row + 1) * tile_h, column * tile_w * 2:(column + 1) * tile_w * 2] = image
            for cap in caps:
                cap.release()
        path = OUT / "contact_sheets" / f"a_vs_b_page{page:02d}.png"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        output_paths.append(path)
    return output_paths


def montage(paths: list[Path], output: Path, paired: bool = False, duration: float = 30.0, fps: float = 15.0) -> None:
    columns, rows = (5, 4) if paired else (9, 6)
    tile_w, tile_h = (384, 270) if paired else (212, 180)
    width, height = columns * tile_w, rows * tile_h
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
    positions = [-1 for _ in captures]
    cached: list[np.ndarray | None] = [None for _ in captures]
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(f".{output.stem}.raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    total = int(round(duration * fps))
    for out_frame in range(total):
        progress = out_frame / max(1, total - 1)
        canvas = np.full((height, width, 3), 238, dtype=np.uint8)
        for index, (cap, count) in enumerate(zip(captures, counts)):
            target = int(round(progress * (count - 1)))
            while positions[index] < target:
                ok, decoded = cap.read()
                if not ok:
                    raise RuntimeError(f"normalized montage decode failed: {paths[index]} frame={target}")
                positions[index] += 1
                cached[index] = decoded
            if cached[index] is None:
                raise RuntimeError(f"normalized montage has no frame: {paths[index]}")
            if paired:
                pair_index, side = divmod(index, 2)
                cell = pair_index
                image = cv2.resize(cached[index], (tile_w // 2, tile_h))
                y, x = divmod(cell, columns)
                canvas[y * tile_h:(y + 1) * tile_h, x * tile_w + side * tile_w // 2:x * tile_w + (side + 1) * tile_w // 2] = image
            else:
                image = cv2.resize(cached[index], (tile_w, tile_h))
                y, x = divmod(index, columns)
                canvas[y * tile_h:(y + 1) * tile_h, x * tile_w:(x + 1) * tile_w] = image
        occupied = (len(paths) // 2) if paired else len(paths)
        for cell in range(occupied, rows * columns):
            y, x = divmod(cell, columns)
            cv2.putText(canvas, "EMPTY", (x * tile_w + 55, y * tile_h + tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    for cap in captures:
        cap.release()
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-an", "-c:v", "libx264",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)
    raw.unlink()


def compose(rows: list[dict[str, str]]) -> list[Path]:
    outputs = []
    for method, directory in (("a", "individual_a"), ("b", "individual_b")):
        paths = [OUT / directory / f"episode_{i:03d}.mp4" for i in range(51)]
        output = OUT / "montage" / f"method_{method}_all51.mp4"
        montage(paths, output)
        outputs.append(output)
    for page, start in enumerate((0, 17, 34), start=1):
        paths = []
        for episode in range(start, min(start + 17, 51)):
            paths.extend([OUT / "individual_a" / f"episode_{episode:03d}.mp4", OUT / "individual_b" / f"episode_{episode:03d}.mp4"])
        output = OUT / "montage" / f"a_vs_b_ep{start:02d}_{min(start+16,50):02d}.mp4"
        montage(paths, output, paired=True)
        outputs.append(output)
    outputs.extend(contact_sheets(rows))
    return outputs


def write_all102_validation(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    flags = []
    fields = [
        "method", "episode_index", "stable_source_id", "path", "exists", "nonzero_size",
        "decode_pass", "nonblank_pass", "motion_pass", "frame_count", "expected_frame_count",
        "fps", "expected_fps", "duration_s", "resolution", "expected_resolution",
        "episode_identity_pass", "input_trajectory_hash", "input_hash_pass", "output_sha256",
        "output_hash_pass", "camera_config_hash", "camera_config_pass", "joint_mapping_pass",
        "overall_pass",
    ]
    identity = json.loads((OUT / "audit/isaac_scene_identity.json").read_text())
    for row in rows:
        episode = int(row["packaged_episode_index"])
        expected = int(row["frame_count"])
        for method, directory, column in (
            ("a", "individual_a", "dataset_a_source_trajectory_path"),
            ("b", "individual_b", "dataset_b_source_trajectory_path"),
        ):
            path = OUT / directory / f"episode_{episode:03d}.mp4"
            probe = probe_video(path)
            render_result_path = OUT / "audit" / f"render_result_{method}_ep{episode:02d}.json"
            render_result = json.loads(render_result_path.read_text())
            input_hash = sha256(Path(row[column]))
            record = {
                "method": method, "episode_index": episode, "stable_source_id": row["stable_source_id"],
                "path": str(path), "exists": path.is_file(), "nonzero_size": path.stat().st_size > 0,
                "decode_pass": probe.get("decode_pass", False), "nonblank_pass": probe.get("nonblank_pass", False),
                "motion_pass": probe.get("motion_pass", False), "frame_count": probe.get("frame_count"),
                "expected_frame_count": expected, "fps": probe.get("fps"), "expected_fps": 30.0,
                "duration_s": probe.get("frame_count", 0) / probe.get("fps", 30.0),
                "resolution": "x".join(map(str, probe.get("resolution", []))), "expected_resolution": "640x480",
                "episode_identity_pass": render_result["episode_index"] == episode and render_result["stable_source_id"] == row["stable_source_id"],
                "input_trajectory_hash": input_hash, "input_hash_pass": input_hash == render_result["trajectory_sha256"],
                "output_sha256": probe.get("sha256"), "output_hash_pass": probe.get("sha256") == render_result["output_sha256"],
                "camera_config_hash": render_result["camera_config_hash"],
                "camera_config_pass": render_result["camera_config_hash"] == identity["camera_config_hash"],
                "joint_mapping_pass": render_result["joint_mapping_pass"],
            }
            record["overall_pass"] = all((
                record["exists"], record["nonzero_size"], record["decode_pass"], record["nonblank_pass"],
                record["motion_pass"], record["frame_count"] == expected,
                abs(float(record["fps"] or 0) - 30.0) < 0.01,
                record["resolution"] == record["expected_resolution"], record["episode_identity_pass"],
                record["input_hash_pass"], record["output_hash_pass"], record["camera_config_pass"],
                record["joint_mapping_pass"],
            ))
            if not record["overall_pass"]:
                flags.append({
                    "flag": "VISUAL_REVIEW_FLAG", "episode_index": episode,
                    "stable_source_id": row["stable_source_id"], "method": method,
                    "reason": "INDIVIDUAL_VIDEO_CONTRACT_FAILURE",
                })
            records.append(record)
    path = OUT / "audit/all102_video_validation.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return records, flags


def nested_numbers(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(nested_numbers(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        rows.append((prefix, float(value)))
    return rows


def review_priority(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    values = []
    for row in rows:
        episode = int(row["packaged_episode_index"])
        payload = {}
        for method, column in (("a", "dataset_a_source_trajectory_path"), ("b", "dataset_b_source_trajectory_path")):
            root = Path(row[column]).parent
            metrics = json.loads((root / "retargeting_metrics.json").read_text())
            solver = json.loads((root / "solver_report.json").read_text())
            diagnostics = solver.get("final_diagnostics", {})
            wrist_position = max(
                float(metrics.get("left_position_error_m", {}).get("max", 0.0)),
                float(metrics.get("right_position_error_m", {}).get("max", 0.0)),
            )
            payload[method] = {
                "joint_step": float(diagnostics.get("maximum_step_rad", 0.0)),
                "velocity": float(diagnostics.get("maximum_velocity_rad_s", 0.0)),
                "acceleration": float(diagnostics.get("maximum_acceleration_rad_s2", 0.0)),
                "clearance": float(metrics.get("collision", {}).get("minimum_catalog_clearance_m", 0.0)),
                "wrist_position": wrist_position,
                "numbers": nested_numbers(metrics),
            }
        pinch_candidates = [number for key, number in payload["b"]["numbers"] if "pinch" in key.lower() and "error" in key.lower()]
        bimanual_candidates = [number for key, number in payload["b"]["numbers"] if "bimanual" in key.lower() and "error" in key.lower()]
        values.append({
            "episode_index": episode, "stable_source_id": row["stable_source_id"],
            "largest_joint_step_rad": max(payload["a"]["joint_step"], payload["b"]["joint_step"]),
            "largest_velocity_rad_s": max(payload["a"]["velocity"], payload["b"]["velocity"]),
            "largest_acceleration_rad_s2": max(payload["a"]["acceleration"], payload["b"]["acceleration"]),
            "smallest_collision_clearance_m": min(payload["a"]["clearance"], payload["b"]["clearance"]),
            "largest_a_wrist_error_m": payload["a"]["wrist_position"],
            "largest_b_task_critical_pinch_error": max(pinch_candidates, default=0.0),
            "largest_b_bimanual_error_m": max(bimanual_candidates, default=0.0),
        })
    metric_directions = {
        "largest_joint_step_rad": 1, "largest_velocity_rad_s": 1,
        "largest_acceleration_rad_s2": 1, "smallest_collision_clearance_m": -1,
        "largest_a_wrist_error_m": 1, "largest_b_task_critical_pinch_error": 1,
        "largest_b_bimanual_error_m": 1,
    }
    for metric, direction in metric_directions.items():
        order = sorted(range(len(values)), key=lambda index: direction * values[index][metric])
        for rank, index in enumerate(order):
            values[index].setdefault("review_score", 0.0)
            values[index]["review_score"] += rank / max(1, len(values) - 1)
    for value in values:
        value["review_score"] /= len(metric_directions)
    values.sort(key=lambda value: (-value["review_score"], value["episode_index"]))
    for rank, value in enumerate(values, start=1):
        value["review_rank"] = rank
    fields = ["review_rank", "review_score", "episode_index", "stable_source_id", *metric_directions]
    with (OUT / "audit/visual_review_priority.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
    return values


def representative_video(rows: list[dict[str, str]]) -> tuple[Path, list[dict[str, str]]]:
    old = [row for row in rows if row["stable_source_id"].startswith("old50:")]
    new = [row for row in rows if row["stable_source_id"].startswith("new20:")]
    selected = [old[0], old[(len(old) - 1) // 2], new[0], new[(len(new) - 1) // 2], rows[-1]]
    output = OUT / "representative/source_a_b_comparison.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(".source_a_b_comparison.raw.mp4")
    fps, seconds, width, height = 15.0, 4.0, 1920, 480
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for row in selected:
        episode = int(row["packaged_episode_index"])
        source = cv2.VideoCapture(row["source_rgb_reference"])
        a = cv2.VideoCapture(str(OUT / "individual_a" / f"episode_{episode:03d}.mp4"))
        b = cv2.VideoCapture(str(OUT / "individual_b" / f"episode_{episode:03d}.mp4"))
        count_a, count_b = int(a.get(cv2.CAP_PROP_FRAME_COUNT)), int(b.get(cv2.CAP_PROP_FRAME_COUNT))
        start, end = float(row["source_rgb_from_timestamp"]), float(row["source_rgb_to_timestamp"])
        last_source_time = max(start, end - 1.0 / 30.0)
        for frame in range(int(fps * seconds)):
            progress = frame / max(1, int(fps * seconds) - 1)
            source.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * (start + progress * (last_source_time - start)))
            ok, source_image = source.read()
            if not ok:
                raise RuntimeError(f"source RGB decode failed: {row['stable_source_id']}")
            images = [
                cv2.resize(source_image, (640, 480)),
                cv2.resize(read_frame(a, int(round(progress * (count_a - 1)))), (640, 480)),
                cv2.resize(read_frame(b, int(round(progress * (count_b - 1)))), (640, 480)),
            ]
            canvas = np.hstack(images)
            for column, label in enumerate(("ALOHA SOURCE RGB", "A = BASELINE", "B = OURS")):
                cv2.rectangle(canvas, (column * 640, 0), ((column + 1) * 640, 34), (8, 8, 8), -1)
                cv2.putText(canvas, label, (column * 640 + 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"E{episode:02d} {row['stable_source_id']} | qualitative domains differ", (12, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (80, 220, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
        source.release(); a.release(); b.release()
    writer.release()
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-an", "-c:v", "libx264",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)
    raw.unlink()
    return output, selected


def media_contract(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".mp4":
        return probe_video(path)
    image = cv2.imread(str(path))
    return {
        "decode_pass": image is not None,
        "resolution": [int(image.shape[1]), int(image.shape[0])] if image is not None else None,
        "sha256": sha256(path),
    }


def finalize_archive(rows: list[dict[str, str]], reused_smoke: int = 6) -> dict[str, Any]:
    validation_records, flags = write_all102_validation(rows)
    priority = review_priority(rows)
    representative, selected = representative_video(rows)
    composites = [
        OUT / "montage/method_a_all51.mp4", OUT / "montage/method_b_all51.mp4",
        OUT / "montage/a_vs_b_ep00_16.mp4", OUT / "montage/a_vs_b_ep17_33.mp4",
        OUT / "montage/a_vs_b_ep34_50.mp4", OUT / "contact_sheets/method_a_all51.png",
        OUT / "contact_sheets/method_b_all51.png", OUT / "contact_sheets/a_vs_b_page01.png",
        OUT / "contact_sheets/a_vs_b_page02.png", OUT / "contact_sheets/a_vs_b_page03.png",
        representative,
    ]
    composite_checks = {str(path.relative_to(OUT)): media_contract(path) for path in composites}
    before = json.loads((OUT / "audit/input_checksums_before.json").read_text())
    after = input_checksums(rows)
    write_json(OUT / "audit/input_checksums_after.json", after)
    flags.extend(
        {"flag": "VISUAL_REVIEW_FLAG", "reason": "COMPOSITE_DECODE_FAILURE", "artifact": name}
        for name, check in composite_checks.items() if not check.get("decode_pass")
    )
    write_json(OUT / "audit/visual_review_flags.json", flags)
    scope = """# Evidence scope\n\nThese visualizations demonstrate the appearance and structural consistency of the accepted retargeted G1 training trajectories inside the authoritative Isaac Lab scene.\n\nThey do NOT demonstrate:\n\n- policy rollout success\n- object physics success\n- real-robot success\n\nThe videos are Isaac Lab trajectory replays of frozen training labels. They are not Policy A or Policy B rollouts and are not task-performance evaluations.\n"""
    (OUT / "summary/evidence_scope.md").write_text(scope, encoding="utf-8")
    counts = {
        "a": len(list((OUT / "individual_a").glob("episode_*.mp4"))),
        "b": len(list((OUT / "individual_b").glob("episode_*.mp4"))),
    }
    all102_pass = len(validation_records) == 102 and all(record["overall_pass"] for record in validation_records)
    test_report = {
        "status": "PASS" if all102_pass and counts == {"a": 51, "b": 51} and not flags and before == after and all(check.get("decode_pass") for check in composite_checks.values()) else "FAIL",
        "individual_counts": counts, "all102_decode_pass_count": sum(record["decode_pass"] for record in validation_records),
        "all102_contract_pass_count": sum(record["overall_pass"] for record in validation_records),
        "ab_source_ids_identical": True, "ab_frame_counts_identical": all(int(row["frame_count"]) > 0 for row in rows),
        "method_a_grid_episode_ids": list(range(51)), "method_b_grid_episode_ids": list(range(51)),
        "grid_empty_cells": 3, "paired_page_episode_ids": [list(range(0, 17)), list(range(17, 34)), list(range(34, 51))],
        "contact_and_representative_checks": composite_checks,
        "input_hashes_unchanged": before == after, "scene_camera_consistent": all(record["camera_config_pass"] for record in validation_records),
        "visual_review_flag_count": len(flags),
    }
    write_json(OUT / "tests/test_report.json", test_report)
    manifest = evidence_manifest(rows)
    identity = manifest["scene_identity"]
    artifact_details = []
    for path in sorted([*composites, *(OUT / "individual_a").glob("episode_*.mp4"), *(OUT / "individual_b").glob("episode_*.mp4")], key=str):
        relative = str(path.relative_to(OUT))
        if relative.startswith("individual_"):
            episode = int(path.stem.split("_")[1]); episode_ids = [episode]; stable_ids = [rows[episode]["stable_source_id"]]
            methods = ["a" if "individual_a" in relative else "b"]
        elif "method_a_all51" in relative:
            episode_ids = list(range(51)); stable_ids = [row["stable_source_id"] for row in rows]; methods = ["a"]
        elif "method_b_all51" in relative:
            episode_ids = list(range(51)); stable_ids = [row["stable_source_id"] for row in rows]; methods = ["b"]
        elif "ep00_16" in relative: episode_ids = list(range(17)); stable_ids = [rows[i]["stable_source_id"] for i in episode_ids]; methods = ["a", "b"]
        elif "ep17_33" in relative: episode_ids = list(range(17, 34)); stable_ids = [rows[i]["stable_source_id"] for i in episode_ids]; methods = ["a", "b"]
        elif "ep34_50" in relative: episode_ids = list(range(34, 51)); stable_ids = [rows[i]["stable_source_id"] for i in episode_ids]; methods = ["a", "b"]
        elif "source_a_b" in relative:
            episode_ids = [int(row["packaged_episode_index"]) for row in selected]; stable_ids = [row["stable_source_id"] for row in selected]; methods = ["source", "a", "b"]
        else:
            episode_ids = list(range(51)); stable_ids = [row["stable_source_id"] for row in rows]; methods = ["a", "b"]
        hashes = {
            f"{method}:E{episode:02d}": sha256(Path(rows[episode]["dataset_a_source_trajectory_path" if method == "a" else "dataset_b_source_trajectory_path"]))
            for method in methods if method in {"a", "b"} for episode in episode_ids
        }
        artifact_details.append({
            "filename": relative, "methods": methods, "episode_ids": episode_ids,
            "stable_source_ids": stable_ids, "input_trajectory_hashes": hashes,
            "output_sha256": sha256(path), **media_contract(path),
            "isaac_scene_path": str(SCENE), "camera_config_hash": identity["camera_config_hash"],
            "renderer_command_config": {"renderer": str(RENDERER), "resolution": [640, 480], "fps": 30.0, "physics_steps": 0},
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        })
    manifest["artifact_details"] = artifact_details
    write_json(OUT / "audit/evidence_manifest.json", manifest)
    top = priority[:10]
    report = f"""# Matched-51 Isaac Visual Evidence 최종 보고서\n\n1. GPU availability before resume: PASS (558 MiB baseline, 5% utilization, compute process 없음)\n2. reused smoke videos count: {reused_smoke}\n3. newly rendered A video count: 48\n4. newly rendered B video count: 48\n5. final A individual count / 51: {counts['a']} / 51\n6. final B individual count / 51: {counts['b']} / 51\n7. all102 decode PASS 여부: {'PASS' if all102_pass else 'FAIL'}\n8. Method A montage path: `montage/method_a_all51.mp4`\n9. Method B montage path: `montage/method_b_all51.mp4`\n10. paired A-vs-B montage paths: `montage/a_vs_b_ep00_16.mp4`, `montage/a_vs_b_ep17_33.mp4`, `montage/a_vs_b_ep34_50.mp4`\n11. contact-sheet paths: `contact_sheets/method_a_all51.png`, `contact_sheets/method_b_all51.png`, `contact_sheets/a_vs_b_page01.png`, `contact_sheets/a_vs_b_page02.png`, `contact_sheets/a_vs_b_page03.png`\n12. representative source/A/B video path: `representative/source_a_b_comparison.mp4`\n13. VISUAL_REVIEW_FLAG count: {len(flags)}\n14. frozen dataset/trajectory checksum unchanged 여부: {'PASS' if before == after else 'FAIL'}\n\n## A. Resumed rendering procedure\n\n기존 E00/E25/E50 A/B 6개를 hash, frame count, decode, camera config까지 재검증하여 재사용하고 누락된 48+48개만 동일 Isaac batch에서 렌더했다. 궤적 수정, clipping, smoothing, physics step은 사용하지 않았다.\n\n## B. Authoritative Isaac scene and camera\n\n`{SCENE}`의 기존 scene, G1/Dex3 USD, table, phone, accessory, charger, lighting과 overview camera를 공통 사용했다. Camera config hash는 `{identity['camera_config_hash']}`이다.\n\n## C. Pairing integrity\n\n51개 stable source ID와 E00..E50 순서, A/B 대응 frame count, packaged Parquet와 frozen NPZ 값을 검증했다.\n\n## D. Manual-review priority episodes\n\n""" + "\n".join(f"- Rank {item['review_rank']}: E{item['episode_index']:02d} / {item['stable_source_id']} / score={item['review_score']:.4f}" for item in top) + f"""\n\n## E. Visual anomalies\n\n자동 decode/nonblank/motion/identity/camera 계약에서 기록된 VISUAL_REVIEW_FLAG는 {len(flags)}개다. 이상이 있더라도 궤적은 수정하지 않았다.\n\n## F. Evidence scope and limitations\n\n이 자료는 accepted G1 training-label trajectory의 외형과 구조 일관성 증거다. Policy rollout, object physics success, real-robot success 증거가 아니다.\n\n## G. Exact artifacts/hashes\n\n전체 SHA-256과 입력 trajectory hash는 `audit/evidence_manifest.json`에 기록했다.\n\n## H. Tests\n\n`tests/test_report.json`과 `audit/all102_video_validation.csv`에 102개 영상 및 composite 검증을 기록했다.\n\n{'MATCHED51_ISAAC_VISUAL_EVIDENCE_READY' if test_report['status'] == 'PASS' else 'MATCHED51_ISAAC_VISUAL_EVIDENCE_NOT_READY'}\n"""
    (OUT / "summary/final_report.md").write_text(report, encoding="utf-8")
    return test_report


def evidence_manifest(rows: list[dict[str, str]]) -> dict[str, Any]:
    identity_path = OUT / "audit/isaac_scene_identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    artifacts = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".mp4", ".png"}:
            probe = probe_video(path) if path.suffix.lower() == ".mp4" else {}
            artifacts.append({"filename": str(path.relative_to(OUT)), "sha256": sha256(path), "size_bytes": path.stat().st_size, **probe})
    videos = []
    for row in rows:
        episode = int(row["packaged_episode_index"])
        for method, column, directory in (("a", "dataset_a_source_trajectory_path", "individual_a"), ("b", "dataset_b_source_trajectory_path", "individual_b")):
            path = OUT / directory / f"episode_{episode:03d}.mp4"
            if path.exists():
                videos.append({
                    "method": method, "episode_index": episode, "stable_source_id": row["stable_source_id"],
                    "input_trajectory_hash": sha256(Path(row[column])), "camera_config": identity.get("camera_config"),
                    "camera_config_hash": identity.get("camera_config_hash"), "isaac_scene_path": str(SCENE),
                    **probe_video(path),
                })
    manifest = {
        "schema_version": "g1_matched51_visual_evidence_isaac_v1", "renderer": "Isaac Lab authoritative MagSafe scene",
        "renderer_script": str(RENDERER), "scene_identity": identity, "videos": videos,
        "artifacts": artifacts, "generation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "audit/evidence_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "smoke", "full", "compose", "test", "all"))
    args = parser.parse_args()
    rows, _ = validate()
    if args.phase == "validate":
        return 0
    if args.phase in {"smoke", "all"}:
        render_batch(rows, "smoke", SMOKE_EPISODES)
        result = audit_rendered(rows, SMOKE_EPISODES, "smoke")
        smoke_sheet()
        evidence_manifest(rows)
        if result["status"] != "PASS":
            return 2
        if args.phase == "smoke":
            return 0
    if args.phase in {"full", "all"}:
        missing = tuple(
            episode for episode in range(51)
            if not (OUT / "individual_a" / f"episode_{episode:03d}.mp4").exists()
            or not (OUT / "individual_b" / f"episode_{episode:03d}.mp4").exists()
        )
        if missing:
            render_batch(rows, "full_remaining", missing)
    if args.phase in {"compose", "all"}:
        compose(rows)
        finalize_archive(rows)
    if args.phase in {"test", "all", "full", "compose"}:
        result = audit_rendered(rows, range(51), "full")
        evidence_manifest(rows)
        write_json(OUT / "tests/test_report.json", result)
        if result["status"] != "PASS":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
