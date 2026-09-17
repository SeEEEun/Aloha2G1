#!/usr/bin/env python3
"""Build key-event Source|A|B contact sheets from already rendered videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY / "outputs/doll_handoff_retargeting/ab_smoke_final_candidate"
EPISODES = (0, 24, 49)
CAMERAS = ("overview", "top", "side")


def read_frame(path: Path, index: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode frame {index} from {path}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    events = json.loads((root / "event_audit/events.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "source_audit/source_manifest.json").read_text(encoding="utf-8")
    )
    records = {int(value["episode_index"]): value for value in manifest["records"]}
    output = root / "comparison/contact_sheets"
    output.mkdir(parents=True, exist_ok=True)
    for episode in EPISODES:
        stable = records[episode]["stable_episode_id"]
        event = events[str(episode)]["frames"]
        source_frames = [
            0,
            event.get("LEFT_GRASP"),
            event.get("RIGHT_GRASP"),
            event.get("LEFT_RELEASE"),
            event.get("RIGHT_FINAL_RELEASE"),
            int(records[episode]["frame_count"]) - 1,
        ]
        source_frames = [int(value) for value in source_frames if value is not None]
        source_frames = list(dict.fromkeys(source_frames))
        for camera in CAMERAS:
            video = root / "comparison/videos" / f"{stable}_comparison_{camera}.mp4"
            images = [read_frame(video, round(frame / 3)) for frame in source_frames]
            sheet = cv2.vconcat(images)
            path = output / f"ep{episode:03d}_{camera}.png"
            if not cv2.imwrite(str(path), sheet):
                raise RuntimeError(f"failed to write {path}")
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
