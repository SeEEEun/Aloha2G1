#!/usr/bin/env python3
"""Render the stopped A01-A05/B01-B03 diagnostic from saved PhysX traces only."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_final_grasp_capture_task_frames import DistanceModel
from tools.common_execution_layer import pose_matrix
from tools.direct_physical_execution_layer import authoritative_joint_limits
from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.render_final_episode_registered_physical_evidence import PhysicalRenderer, camera


OUT = ROOT / "outputs/final_episode_registered_eval35"
DEST = OUT / "08_human_visual_diagnosis"
A_ROOT = OUT / "02_act_a_results/rollouts"
B_ROOT = OUT / "03_act_b_results/rollouts"
CONTACT_MODEL = OUT / "01_freeze/FINAL_DOLL_CONTACT_MODEL.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
COMMON = ROOT / "outputs/final_direct_physical_eval35/00_preparation/runtime_frozen_fair_a/config/common_config.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
FREEZE = OUT / "01_freeze/FINAL_EVAL35_FREEZE_MANIFEST.json"

FPS = 30
PANEL_W, PANEL_H = 640, 480
HEADER_H = 64
TRIPTYCH_W, TRIPTYCH_H = 3 * PANEL_W, HEADER_H + PANEL_H
MATCHED_H = HEADER_H + 2 * PANEL_H
VIEWS = ("TOP", "OVERVIEW", "LEFT-HAND CLOSE-UP")
DIGITS = ("thumb", "index", "middle")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_run(root: Path, index: int) -> Path:
    values = [path for path in root.glob(f"eval_{index:02d}_*") if (path / "event_log.npz").is_file()]
    if len(values) != 1:
        raise RuntimeError(f"expected one saved trace for {root.name} index {index}, found {len(values)}")
    return values[0]


def load_trace(path: Path) -> dict[str, Any]:
    with np.load(path / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    control = event["control_frame"].astype(np.int64)
    last_rows = np.flatnonzero(np.r_[np.diff(control) != 0, True])
    frames = control[last_rows]
    result = read_json(path / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json")
    trial = read_json(path / "trial_result.json")
    return {
        "path": path,
        "event": event,
        "last_rows": last_rows,
        "frames": frames,
        "q": event["MEASURED_Q"][last_rows].astype(np.float64),
        "position": event["object_position_world_m"][last_rows].astype(np.float64),
        "quaternion": event["object_quaternion_xyzw"][last_rows].astype(np.float64),
        "intent": event["DIRECT_COMMON_TASK_INTENT"][last_rows].astype(str),
        "result": result,
        "trial": trial,
    }


def dex3_violation_details(trace: dict[str, Any]) -> dict[str, Any] | None:
    event = trace["event"]
    _, _, names = authoritative_joint_limits(read_json(JOINT_CONTRACT))
    specs = read_json(JOINT_CONTRACT)["joint_specs"]
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    measured = event["MEASURED_Q"].astype(np.float64)
    mask = (measured[:, 14:] < lower[14:] - 1.0e-6) | (measured[:, 14:] > upper[14:] + 1.0e-6)
    rows, columns = np.where(mask)
    if not len(rows):
        return None
    excursions = np.maximum(lower[columns + 14] - measured[rows, columns + 14], measured[rows, columns + 14] - upper[columns + 14])
    maximum = int(np.argmax(excursions))
    row = int(rows[maximum])
    joint = int(columns[maximum] + 14)
    first_row = int(rows[0])
    return {
        "first_invalid_row": first_row,
        "last_valid_row": first_row - 1,
        "last_valid_control_frame": int(event["control_frame"][first_row - 1]),
        "maximum_row": row,
        "control_frame": int(event["control_frame"][row]),
        "joint_index": joint,
        "joint_name": str(names[joint]),
        "commanded_q_rad": float(event["EXECUTED_COMMAND"][row, joint]),
        "measured_q_rad": float(measured[row, joint]),
        "authoritative_min_rad": float(lower[joint]),
        "authoritative_max_rad": float(upper[joint]),
        "violating_samples": int(len(rows)),
        "left_digit_contact_n": {
            digit: float(event[f"left_{digit}_force_n"][row]) for digit in DIGITS
        },
    }


def window(trace: dict[str, Any], invalid: dict[str, Any] | None = None) -> np.ndarray:
    frames = trace["frames"]
    intent = trace["intent"]
    close_rows = np.flatnonzero(intent == "LEFT_CLOSE_INTENT")
    if not len(close_rows):
        raise RuntimeError(f"LEFT_CLOSE_INTENT missing: {trace['path']}")
    start_frame = max(int(frames[0]), int(frames[close_rows[0]]) - FPS)
    handoff_rows = np.flatnonzero(intent == "HANDOFF_INTENT")
    end_frame = int(frames[-1]) if not len(handoff_rows) else int(frames[handoff_rows[0]]) + FPS // 2
    if invalid is not None:
        end_frame = min(end_frame, int(invalid["last_valid_control_frame"]))
    selected = np.flatnonzero((frames >= start_frame) & (frames <= end_frame))
    if not len(selected):
        raise RuntimeError(f"empty grasp-review window: {trace['path']}")
    return selected


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temp = path.with_name(path.stem + ".incomplete" + path.suffix)
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(FPS),
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.temp),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, image: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin unavailable")
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def finish(self) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin unavailable")
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {self.path}")
        os.replace(self.temp, self.path)


def put(image: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.6, color: tuple[int, int, int] = (245, 245, 245), thickness: int = 1) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def update_close_camera(renderer: PhysicalRenderer) -> None:
    # Center the aperture itself (three distal digit links plus the doll), not
    # the wrist.  Looking from +X avoids the torso/forearm occlusion seen from
    # the approved overview-camera side while retaining the wrist in frame.
    points = [np.asarray(renderer.doll_position, dtype=np.float64)]
    for body_name in (
        "left_hand_thumb_2_link",
        "left_hand_index_1_link",
        "left_hand_middle_1_link",
    ):
        body = mujoco.mj_name2id(renderer.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body >= 0:
            points.append(np.asarray(renderer.data.xpos[body], dtype=np.float64))
    target = np.mean(points, axis=0)
    eye = target + np.asarray([-0.30, -0.28, 0.14])
    renderer.cameras["left_close_dynamic"] = camera(eye, target)


def panel(renderer: PhysicalRenderer, trace: dict[str, Any], index: int, method: str, number: int, diagnostic: str = "") -> np.ndarray:
    renderer.set_state(trace["q"][index], trace["position"][index], trace["quaternion"][index])
    update_close_camera(renderer)
    views = (
        renderer.view("top"),
        renderer.view("overview"),
        renderer.view("left_close_dynamic"),
    )
    canvas = np.zeros((TRIPTYCH_H, TRIPTYCH_W, 3), dtype=np.uint8)
    frame = int(trace["frames"][index])
    intent = str(trace["intent"][index])
    validity = "INVALID PHYSICS" if trace["result"]["status"] == "INVALID" else "VALID"
    header = f"ACT-{method}{number:02d} | control {frame:03d} | {intent} | {validity} | grasp confirmed: {'YES' if trace['result']['event_frames']['grasp_confirm'] is not None else 'NO'}"
    cv2.rectangle(canvas, (0, 0), (TRIPTYCH_W, HEADER_H), (19, 21, 27), -1)
    put(canvas, header, (16, 27), 0.62, (255, 255, 255), 1)
    if diagnostic:
        put(canvas, diagnostic, (16, 53), 0.50, (80, 190, 255), 1)
    for column, (name, image) in enumerate(zip(VIEWS, views, strict=True)):
        resized = cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
        x = column * PANEL_W
        canvas[HEADER_H:, x:x + PANEL_W] = resized
        cv2.rectangle(canvas, (x, HEADER_H), (x + PANEL_W - 1, TRIPTYCH_H - 1), (70, 70, 70), 1)
        cv2.rectangle(canvas, (x + 8, HEADER_H + 8), (x + 245, HEADER_H + 39), (16, 18, 22), -1)
        put(canvas, name, (x + 18, HEADER_H + 31), 0.58, (255, 255, 255), 1)
    return canvas


def hold(writer: VideoWriter, image: np.ndarray, frames: int) -> None:
    for _ in range(frames):
        writer.write(image)


def render_method_video(renderer: PhysicalRenderer, method: str, traces: list[dict[str, Any]], invalids: list[dict[str, Any] | None], path: Path) -> None:
    writer = VideoWriter(path, TRIPTYCH_W, TRIPTYCH_H)
    try:
        for number, (trace, invalid) in enumerate(zip(traces, invalids, strict=True), start=1):
            indices = window(trace, invalid)
            first = panel(renderer, trace, int(indices[0]), method, number, "ACTUAL SAVED PHYSX STATE")
            hold(writer, first, FPS)
            for index in indices:
                note = ""
                if invalid is not None:
                    note = f"TRUNCATES AT LAST VALID CONTROL {invalid['last_valid_control_frame']} BEFORE ARTICULATION INVALIDITY"
                writer.write(panel(renderer, trace, int(index), method, number, note))
            hold(writer, panel(renderer, trace, int(indices[-1]), method, number, "END OF AVAILABLE VALID PHYSICAL PREFIX"), FPS // 2)
            if invalid is not None:
                row = int(invalid["maximum_row"])
                event = trace["event"]
                diagnostic_trace = {
                    **trace,
                    "q": event["MEASURED_Q"][row:row + 1],
                    "position": event["object_position_world_m"][row:row + 1],
                    "quaternion": event["object_quaternion_xyzw"][row:row + 1],
                    "frames": event["control_frame"][row:row + 1],
                    "intent": event["DIRECT_COMMON_TASK_INTENT"][row:row + 1].astype(str),
                }
                contact = invalid["left_digit_contact_n"]
                note = (
                    f"INVALID SAMPLE: {invalid['joint_name']} cmd {invalid['commanded_q_rad']:+.6f}, "
                    f"measured {invalid['measured_q_rad']:+.6f} rad; doll T/I/M "
                    f"{contact['thumb']:.3f}/{contact['index']:.3f}/{contact['middle']:.3f} N"
                )
                image = panel(renderer, diagnostic_trace, 0, method, number, note)
                hold(writer, image, 2 * FPS)
        writer.finish()
    except Exception:
        if writer.process.poll() is None:
            writer.process.kill()
        raise


def render_matched(renderer: PhysicalRenderer, a: list[dict[str, Any]], b: list[dict[str, Any]], b_invalid: list[dict[str, Any] | None], path: Path) -> None:
    writer = VideoWriter(path, TRIPTYCH_W, MATCHED_H)
    try:
        for number in range(1, 4):
            ta, tb = a[number - 1], b[number - 1]
            invalid = b_invalid[number - 1]
            ia, ib = window(ta, invalid), window(tb, invalid)
            frame_start = max(int(ta["frames"][ia[0]]), int(tb["frames"][ib[0]]))
            frame_end = min(int(ta["frames"][ia[-1]]), int(tb["frames"][ib[-1]]))
            frames = range(frame_start, frame_end + 1)
            first_a = int(np.argmin(np.abs(ta["frames"] - frame_start)))
            first_b = int(np.argmin(np.abs(tb["frames"] - frame_start)))
            first_top = panel(renderer, ta, first_a, "A", number)[HEADER_H:]
            first_bottom = panel(renderer, tb, first_b, "B", number)[HEADER_H:]
            title = np.zeros((MATCHED_H, TRIPTYCH_W, 3), dtype=np.uint8)
            title[HEADER_H:HEADER_H + PANEL_H] = first_top
            title[HEADER_H + PANEL_H:] = first_bottom
            put(title, f"MATCHED EPISODE {number:02d} | ACT-A (top row) vs ACT-B (bottom row) | actual saved PhysX states", (16, 39), 0.72)
            hold(writer, title, FPS)
            for frame in frames:
                a_index = int(np.argmin(np.abs(ta["frames"] - frame)))
                b_index = int(np.argmin(np.abs(tb["frames"] - frame)))
                top = panel(renderer, ta, a_index, "A", number)[HEADER_H:]
                bottom = panel(renderer, tb, b_index, "B", number)[HEADER_H:]
                canvas = np.zeros((MATCHED_H, TRIPTYCH_W, 3), dtype=np.uint8)
                canvas[HEADER_H:HEADER_H + PANEL_H] = top
                canvas[HEADER_H + PANEL_H:] = bottom
                suffix = " | B prefix ends before invalid control 192" if invalid is not None else ""
                put(canvas, f"MATCHED EPISODE {number:02d} | synchronized control {frame:03d}{suffix}", (16, 39), 0.68)
                put(canvas, "ACT-A", (12, HEADER_H + 30), 0.72, (80, 220, 255), 2)
                put(canvas, "ACT-B", (12, HEADER_H + PANEL_H + 30), 0.72, (100, 255, 130), 2)
                writer.write(canvas)
            hold(writer, canvas, FPS // 2)
        writer.finish()
    except Exception:
        if writer.process.poll() is None:
            writer.process.kill()
        raise


def metrics(traces: list[tuple[str, int, dict[str, Any], dict[str, Any] | None]]) -> list[dict[str, Any]]:
    contact_model = read_json(CONTACT_MODEL)
    config = read_json(CONFIG)
    common = load_common_config(COMMON)
    g1 = G1Kinematics(common, load_scene(common))
    _, _, joint_names = authoritative_joint_limits(read_json(JOINT_CONTRACT))
    collision = np.asarray(contact_model["collision_dimensions_m"], dtype=np.float64)
    visual = np.asarray(contact_model["visual_dimensions_m"], dtype=np.float64)
    z_offset = float((collision[2] - visual[2]) / 2.0)
    model = DistanceModel(g1, collision, float(config["object"]["table_surface_world_z_m"]), joint_names)
    rows: list[dict[str, Any]] = []
    for method, number, trace, invalid in traces:
        event = trace["event"]
        valid_end = len(event["control_frame"]) - 1 if invalid is None else int(invalid["last_valid_row"])
        physics_rows = np.arange(len(event["control_frame"])) <= valid_end
        close_physics = physics_rows & np.isin(event["DIRECT_COMMON_TASK_INTENT"].astype(str), ["LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT"])
        selected_control = window(trace, invalid)
        close_control = selected_control[np.isin(trace["intent"][selected_control], ["LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT"])]
        minimum = {name: math.inf for name in (*DIGITS, "palm")}
        closest_index = int(close_control[0])
        closest_value = math.inf
        for index in close_control:
            model.assign(trace["q"][index])
            model.set_object_pose(pose_matrix(trace["position"][index], trace["quaternion"][index]), z_offset)
            mujoco.mj_forward(model.model, model.data)
            values = model.distances()
            for name, value in values.items():
                minimum[name] = min(minimum[name], float(value))
            current = min(values.values())
            if current < closest_value:
                closest_value = current
                closest_index = int(index)
        initial_z = float(event["object_position_world_m"][0, 2])
        result = trace["result"]
        row = {
            "episode": f"{number:02d}",
            "method": f"ACT-{method}",
            "physical_validity": "INVALID" if result["status"] == "INVALID" else "VALID",
            "first_failure_stage": result["first_failure_stage"],
            "palm_to_doll_min_mm": 1000.0 * minimum["palm"],
            "thumb_to_doll_min_mm": 1000.0 * minimum["thumb"],
            "index_to_doll_min_mm": 1000.0 * minimum["index"],
            "middle_to_doll_min_mm": 1000.0 * minimum["middle"],
            "thumb_contact_max": float(np.max(event["left_thumb_force_n"][close_physics], initial=0.0)),
            "index_contact_max": float(np.max(event["left_index_force_n"][close_physics], initial=0.0)),
            "middle_contact_max": float(np.max(event["left_middle_force_n"][close_physics], initial=0.0)),
            "mechanical_grasp_confirmed": result["event_frames"]["grasp_confirm"] is not None,
            "max_doll_com_lift_mm": 1000.0 * float(np.max(event["object_position_world_m"][:valid_end + 1, 2], initial=initial_z) - initial_z),
            "_closest_control_index": closest_index,
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "episode", "method", "physical_validity", "first_failure_stage",
        "palm_to_doll_min_mm", "thumb_to_doll_min_mm", "index_to_doll_min_mm",
        "middle_to_doll_min_mm", "thumb_contact_max", "index_contact_max",
        "middle_contact_max", "mechanical_grasp_confirmed", "max_doll_com_lift_mm",
    ]
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    os.replace(temporary, path)


def contact_sheet(renderer: PhysicalRenderer, all_traces: list[tuple[str, int, dict[str, Any], dict[str, Any] | None]], rows: list[dict[str, Any]], path: Path) -> None:
    tile_w, tile_h, text_h = 480, 360, 96
    canvas = np.full((2 * (tile_h + text_h), 4 * tile_w, 3), 238, dtype=np.uint8)
    for cell, ((method, number, trace, invalid), row) in enumerate(zip(all_traces, rows, strict=True)):
        index = int(row["_closest_control_index"])
        renderer.set_state(trace["q"][index], trace["position"][index], trace["quaternion"][index])
        update_close_camera(renderer)
        image = cv2.resize(renderer.view("left_close_dynamic"), (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        r, c = divmod(cell, 4)
        x, y = c * tile_w, r * (tile_h + text_h)
        canvas[y:y + tile_h, x:x + tile_w] = image
        validity = row["physical_validity"]
        color = (20, 20, 210) if validity == "INVALID" else (30, 90, 30)
        put(canvas, f"ACT-{method}{number:02d} | {validity} | {row['first_failure_stage']}", (x + 8, y + tile_h + 25), 0.53, color, 1)
        put(canvas, f"surface mm P/T/I/M: {row['palm_to_doll_min_mm']:.1f} / {row['thumb_to_doll_min_mm']:.1f} / {row['index_to_doll_min_mm']:.1f} / {row['middle_to_doll_min_mm']:.1f}", (x + 8, y + tile_h + 50), 0.40, (25, 25, 25), 1)
        put(canvas, f"contact N T/I/M: {row['thumb_contact_max']:.3f} / {row['index_contact_max']:.3f} / {row['middle_contact_max']:.3f}; COM lift {row['max_doll_com_lift_mm']:.1f} mm", (x + 8, y + tile_h + 75), 0.37, (25, 25, 25), 1)
    temporary = path.with_name(path.stem + ".incomplete" + path.suffix)
    if not cv2.imwrite(str(temporary), canvas):
        raise RuntimeError(f"cannot write {temporary}")
    os.replace(temporary, path)


def b03_image(renderer: PhysicalRenderer, trace: dict[str, Any], invalid: dict[str, Any], path: Path) -> None:
    row = int(invalid["maximum_row"])
    event = trace["event"]
    diagnostic = {
        **trace,
        "q": event["MEASURED_Q"][row:row + 1],
        "position": event["object_position_world_m"][row:row + 1],
        "quaternion": event["object_quaternion_xyzw"][row:row + 1],
        "frames": event["control_frame"][row:row + 1],
        "intent": event["DIRECT_COMMON_TASK_INTENT"][row:row + 1].astype(str),
    }
    contacts = invalid["left_digit_contact_n"]
    note = (
        f"{invalid['joint_name']}: command {invalid['commanded_q_rad']:+.9f} rad, measured {invalid['measured_q_rad']:+.9f} rad; "
        f"doll contact T/I/M {contacts['thumb']:.3f}/{contacts['index']:.3f}/{contacts['middle']:.3f} N"
    )
    image = panel(renderer, diagnostic, 0, "B", 3, note)
    temporary = path.with_name(path.stem + ".incomplete" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"cannot write {temporary}")
    os.replace(temporary, path)


def probe(path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)],
        text=True,
    )
    return json.loads(output)["streams"][0]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    # Exact stopped scope: A01-A05 and available B01-B03. B04/B05 must not exist.
    if list(B_ROOT.glob("eval_0[34]_*")):
        raise RuntimeError("B04/B05 unexpectedly exist; refusing to render an ambiguous scope")
    a = [load_trace(unique_run(A_ROOT, index)) for index in range(5)]
    b = [load_trace(unique_run(B_ROOT, index)) for index in range(3)]
    a_invalid = [dex3_violation_details(trace) for trace in a]
    b_invalid = [dex3_violation_details(trace) for trace in b]
    if any(value is not None for value in a_invalid) or b_invalid[:2] != [None, None] or b_invalid[2] is None:
        raise RuntimeError("saved trace validity pattern does not match the stopped diagnostic")
    invalid = b_invalid[2]
    assert invalid is not None
    all_traces = [
        *(("A", number, trace, None) for number, trace in enumerate(a, start=1)),
        *(("B", number, trace, b_invalid[number - 1]) for number, trace in enumerate(b, start=1)),
    ]
    rows = metrics(all_traces)
    csv_path = DEST / "ACT_AB_AVAILABLE_GRASP_DIAGNOSTICS.csv"
    write_csv(rows, csv_path)
    paths = {
        "A01_A05": DEST / "A01_A05_PHYSICAL_GRASP_REVIEW.mp4",
        "B01_B03": DEST / "B01_B03_AVAILABLE_PHYSICAL_GRASP_REVIEW.mp4",
        "matched": DEST / "ACT_AB_AVAILABLE_MATCHED_GRASP_REVIEW.mp4",
        "contact_sheet": DEST / "ACT_AB_AVAILABLE_GRASP_CONTACT_SHEET.png",
        "B03_frame192": DEST / "B03_FRAME192_ARTICULATION_DIAGNOSTIC.png",
        "diagnostics_csv": csv_path,
    }
    renderer = PhysicalRenderer()
    try:
        render_method_video(renderer, "A", a, a_invalid, paths["A01_A05"])
        render_method_video(renderer, "B", b, b_invalid, paths["B01_B03"])
        render_matched(renderer, a, b, b_invalid, paths["matched"])
        contact_sheet(renderer, all_traces, rows, paths["contact_sheet"])
        b03_image(renderer, b[2], invalid, paths["B03_frame192"])
    finally:
        renderer.close()
    video_probes = {key: probe(path) for key, path in paths.items() if path.suffix == ".mp4"}
    report = {
        "status": "PASS",
        "scientific_configuration_modified": False,
        "physics_rerun": False,
        "command_only_trajectory_used": False,
        "saved_trace_fields": ["MEASURED_Q", "object_position_world_m", "object_quaternion_xyzw"],
        "freeze_sha256": subprocess.check_output(["sha256sum", str(FREEZE)], text=True).split()[0],
        "scope": {"ACT_A": "A01-A05", "ACT_B": "B01-B02 valid plus B03 valid prefix"},
        "B03": invalid,
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "video_probes": video_probes,
    }
    report_path = DEST / "ACT_AB_AVAILABLE_VISUAL_DIAGNOSIS_MANIFEST.json"
    temporary = report_path.with_suffix(report_path.suffix + ".incomplete")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
