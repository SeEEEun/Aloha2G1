#!/usr/bin/env python3
"""Compose prior frozen SmolVLA and new ACT Isaac diagnostic videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DEFAULT_SMOL = ROOT / "outputs/policy_b_isaac_validation/full_policy_b_diagnostic_rollout"
DEFAULT_OUTPUT = ROOT / "outputs/policy_b_act/comparison"
VIEWS = ("overview", "top", "side")
FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-dir", type=Path, required=True)
    parser.add_argument("--smolvla-dir", type=Path, default=DEFAULT_SMOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def annotate(frame, title: str, subtitle: str):
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 52), (8, 8, 12), -1)
    cv2.putText(result, title, (9, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, subtitle, (9, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 220, 255), 1, cv2.LINE_AA)
    return result


def main() -> None:
    args = parse_args()
    act_dir = args.act_dir.resolve()
    smol_dir = args.smolvla_dir.resolve()
    output = args.output.resolve()
    act_report = json.loads((act_dir / "isaac_diagnostic_report.json").read_text(encoding="utf-8"))
    smol_report = json.loads((smol_dir / "stage_report.json").read_text(encoding="utf-8"))
    if act_report["stage"] != "kinematic-doll":
        raise RuntimeError("ACT input is not the kinematic-doll diagnostic")
    records = {}
    for view in VIEWS:
        smol_path = smol_dir / f"{view}.mp4"
        act_path = act_dir / f"{view}.mp4"
        if not smol_path.is_file() or not act_path.is_file():
            raise FileNotFoundError(smol_path if not smol_path.is_file() else act_path)
        smol_capture = cv2.VideoCapture(str(smol_path))
        act_capture = cv2.VideoCapture(str(act_path))
        if not smol_capture.isOpened() or not act_capture.isOpened():
            raise RuntimeError(f"cannot decode {view} inputs")
        smol_frames = int(smol_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        act_frames = int(act_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = min(smol_frames, act_frames)
        if count <= 0:
            raise RuntimeError(f"empty {view} input")
        destination = output / f"smolvla_vs_act_closed_loop_{view}.mp4"
        output.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 480))
        if not writer.isOpened():
            raise RuntimeError(f"cannot write {destination}")
        written = 0
        try:
            while written < count:
                smol_ok, smol_frame = smol_capture.read()
                act_ok, act_frame = act_capture.read()
                if not smol_ok or not act_ok:
                    break
                smol_frame = cv2.resize(smol_frame, (640, 480))
                act_frame = cv2.resize(act_frame, (640, 480))
                left = annotate(
                    smol_frame,
                    "SMOLVLA PRIOR CLOSED LOOP",
                    "frozen established diagnostic | H=4 | no new tuning",
                )
                right = annotate(
                    act_frame,
                    f"ACT-{act_report['execution'].upper()} CLOSED LOOP",
                    "official select_action | current RGB + measured 28D state",
                )
                writer.write(cv2.hconcat((left, right)))
                written += 1
        finally:
            writer.release()
            smol_capture.release()
            act_capture.release()
        records[view] = {
            "output": str(destination),
            "output_sha256": sha256_file(destination),
            "frames": written,
            "fps": FPS,
            "smolvla_input": str(smol_path),
            "smolvla_input_sha256": sha256_file(smol_path),
            "act_input": str(act_path),
            "act_input_sha256": sha256_file(act_path),
        }
    manifest = {
        "schema_version": "smolvla_vs_act_closed_loop_comparison_v1",
        "status": "PASS",
        "smolvla_checkpoint": smol_report["checkpoint"],
        "smolvla_checkpoint_sha256": smol_report["checkpoint_model_sha256"],
        "smolvla_existing_artifact_reused_without_new_tuning": True,
        "act_checkpoint": act_report["checkpoint"],
        "act_checkpoint_sha256": act_report["checkpoint_model_sha256"],
        "act_execution": act_report["execution"],
        "views": records,
    }
    atomic_json(output / "smolvla_vs_act_closed_loop_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
