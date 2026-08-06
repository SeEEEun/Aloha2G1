#!/usr/bin/env python3
"""Reusable, simulation-only ALOHA -> G1 MagSafe task orchestrator.

This file is deliberately an orchestrator. Isaac rendering/physics and the
validated retargeter remain in their authoritative modules; this CLI records
every invoked command and refuses unavailable or unapproved backends.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
TASK_ROOT = ROOT / "outputs/aloha_g1_tasks"
ISAAC = Path("/home/jbnu/IsaacLab-3-beta/isaaclab.sh")
SCENE = ROOT / "isaaclab_magsafe_fixed_scene"
AUTHORITATIVE_SCENE = SCENE / "generated/magsafe_magnetic_scene_v2.usda"
CONFIGS = {
    "root": ROOT / "configs/magsafe_g1_root.approved.json",
    "dex3": ROOT / "configs/magsafe_dex3_primitives.approved.json",
    "grasp": ROOT / "configs/magsafe_g1_grasp_calibration.approved.json",
}
CONVERTER = ROOT / "tools/retarget_episode49_consensus_relative_bimanual_to_g1.py"
PHASE_EXTRACTOR = ROOT / "tools/extract_magsafe_gripper_phases.py"
COMPOSER = ROOT / "tools/compose_g1_arm_dex3_trajectory.py"
ALOHA_REPLAY = SCENE / "replay_smolvla_aloha_prediction.py"
G1_REPLAY = SCENE / "replay_magsafe_g1_trajectory.py"
G1_ASSET = Path("/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd")

BANNER = "SIMULATION-ONLY ALOHA–G1 TASK PIPELINE\nNO REAL ROBOT COMMANDS\nNO DDS OR PUBLISHER"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle(episode_id: str) -> Path:
    if not episode_id or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for x in episode_id):
        raise ValueError("episode-id must contain only letters, digits, '_' or '-'")
    return TASK_ROOT / episode_id


def make_dirs(base: Path) -> None:
    for name in ("source", "converted", "videos", "metrics", "logs"):
        (base / name).mkdir(parents=True, exist_ok=True)


def load_manifest(base: Path) -> dict:
    path = base / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"episode is not prepared: {path}")
    return json.loads(path.read_text())


def write_dashboard(base: Path, manifest: dict) -> None:
    videos = "".join(f'<li><a href="videos/{p.name}">{p.name}</a></li>' for p in sorted((base / "videos").glob("*.mp4"))) or "<li>None generated</li>"
    failures = "<br>".join(manifest.get("failure_reasons", [])) or "None"
    html = f"""<!doctype html><meta charset=utf-8><title>{manifest['episode_id']}</title>
<h1>ALOHA–G1 simulation task: {manifest['episode_id']}</h1>
<p><b>SIMULATION ONLY — NO REAL ROBOT COMMANDS — NO DDS OR PUBLISHER</b></p>
<h2>Status</h2><pre>{json.dumps(manifest.get('validation_status', {}), indent=2)}</pre>
<h2>Failures / gates</h2><p>{failures}</p><h2>Videos</h2><ul>{videos}</ul>
<h2>Manifest</h2><pre>{json.dumps(manifest, indent=2)}</pre>"""
    (base / "index.html").write_text(html)


def inspect_action(path: Path) -> tuple[np.ndarray, float, str]:
    with np.load(path, allow_pickle=False) as z:
        key = "optimized_action" if "optimized_action" in z.files else "action" if "action" in z.files else None
        if key is None:
            raise ValueError("ALOHA NPZ must contain optimized_action or action")
        action = np.asarray(z[key], dtype=np.float64)
        fps = float(z["fps"]) if "fps" in z.files else 30.0
    if action.ndim != 2 or action.shape[1] != 14 or len(action) < 2:
        raise ValueError(f"expected [T,14] ALOHA action, got {action.shape}")
    if not np.isfinite(action).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("non-finite action or invalid FPS")
    return action, fps, key


def run_logged(cmd: list[str], log: Path, *, execute: bool = True) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(map(str, cmd)) + "\n")
        if not execute:
            stream.write("DRY RUN\n")
            return 0
        result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        return result.returncode


def calibration_status() -> tuple[bool, dict]:
    detail = {}
    valid = True
    for name, path in CONFIGS.items():
        item = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            try:
                data = json.loads(path.read_text())
                item["approved"] = bool(data.get("approved", data.get("status", "").startswith("APPROVED")))
                item["sha256"] = sha256(path)
            except Exception as exc:
                item.update(approved=False, error=str(exc))
        else:
            item["approved"] = False
        valid &= item["approved"]
        detail[name] = item
    return valid, detail


def prepare(args: argparse.Namespace) -> int:
    source = args.aloha_action.resolve()
    action, fps, key = inspect_action(source)
    base = bundle(args.episode_id); make_dirs(base)
    dst = base / "source/aloha_action.npz"
    if dst.exists() and sha256(dst) != sha256(source) and not args.force:
        raise RuntimeError("source already exists with a different hash; use --force explicitly")
    if not dst.exists() or sha256(dst) != sha256(source):
        shutil.copy2(source, dst)
    approved, calibrations = calibration_status()
    source_type = args.source_type or ("smolvla" if key == "optimized_action" else "raw")
    manifest = {
        "schema_version": 1, "episode_id": args.episode_id, "source_type": source_type,
        "source_action_path": str(source), "bundle_action_path": str(dst), "action_key": key,
        "source_sha256": sha256(source), "frame_count": len(action), "fps": fps,
        "task_timeline": str(args.timeline.resolve()) if args.timeline else None,
        "converter_version": sha256(CONVERTER), "g1_root_config_version": calibrations["root"].get("sha256"),
        "dex3_primitive_config_version": calibrations["dex3"].get("sha256"),
        "grasp_calibration_version": calibrations["grasp"].get("sha256"),
        "authoritative_scene": str(AUTHORITATIVE_SCENE), "authoritative_scene_sha256": sha256(AUTHORITATIVE_SCENE),
        "g1_asset": str(G1_ASSET), "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_files": [str(dst.relative_to(base))],
        "validation_status": {"input": "PASS", "calibration": "PASS" if approved else "CALIBRATION_REQUIRED", "conversion": "NOT_RUN"},
        "calibrations": calibrations, "failure_reasons": [], "simulation_only": True,
    }
    atomic_json(base / "source/metadata.json", {k: manifest[k] for k in ("source_type", "source_action_path", "action_key", "source_sha256", "frame_count", "fps")})
    if not approved:
        manifest["failure_reasons"].append("ONE_TIME_APPROVED_CALIBRATION_CONFIGS_MISSING")
        atomic_json(base / "manifest.json", manifest); write_dashboard(base, manifest)
        print("CALIBRATION_REQUIRED: no trajectory was generated; approve the three one-time configs or use a separately reviewed calibration workflow")
        return 4
    arm = base / "converted/g1_arm_trajectory.npz"
    phases = base / "metrics/gripper_phases.csv"
    full = base / "converted/g1_arm_dex3_trajectory.npz"
    log = base / "logs/prepare.log"
    commands = [
        [sys.executable, str(CONVERTER), "--source", str(dst), "--output", str(arm), "--execute"],
        [sys.executable, str(PHASE_EXTRACTOR), "--input", str(dst), "--output", str(phases)],
        [sys.executable, str(COMPOSER), "--arm-trajectory", str(arm), "--phases", str(phases), "--primitives", str(CONFIGS["dex3"]), "--output", str(full)],
    ]
    for command in commands:
        rc = run_logged(command, log, execute=not args.dry_run)
        if rc:
            manifest["validation_status"]["conversion"] = "FAILED"
            manifest["failure_reasons"].append(f"command failed rc={rc}: {command[1]}")
            atomic_json(base / "manifest.json", manifest); write_dashboard(base, manifest)
            return rc
    manifest["generated_files"] += [str(x.relative_to(base)) for x in (arm, phases, full) if x.exists()]
    manifest["validation_status"]["conversion"] = "DRY_RUN" if args.dry_run else "PASS"
    atomic_json(base / "manifest.json", manifest); write_dashboard(base, manifest)
    return 0


def isaac_command(script: Path, extra: list[str], view: str) -> list[str]:
    cmd = [str(ISAAC), "-p", str(script), *extra, "--device", "cpu"]
    if view == "video":
        cmd += ["--headless", "--enable_cameras", "--viz", "none"]
    else:
        cmd += ["--viz", "kit"]
    return cmd


def replay(args: argparse.Namespace) -> int:
    base = bundle(args.episode_id); m = load_manifest(base); make_dirs(base)
    if args.robot == "aloha":
        extra = ["--prediction-npz", str(base / "source/aloha_action.npz"), "--action-key", m["action_key"], "--speed", str(args.speed), "--max-frames", str(args.end_frame or m["frame_count"])]
        if args.view == "video":
            extra += ["--record-video", "--video-output", str(base / "videos/aloha_replay.mp4")]
        command = isaac_command(ALOHA_REPLAY, extra, args.view)
    else:
        if args.mode == "physics":
            return unavailable(base, "physics", "VERIFIED_G1_PHYSICS_TRAJECTORY_CONTROLLER_NOT_AVAILABLE")
        extra = ["--arm-input", str(base / "converted/g1_arm_trajectory.npz"), "--full-input", str(base / "converted/g1_arm_dex3_trajectory.npz"), "--mode", "kinematic-full", "--speed", str(args.speed), "--camera", args.camera]
        if args.loop: extra.append("--loop")
        if args.end_frame: extra += ["--max-frames", str(args.end_frame)]
        if args.view == "video":
            return unavailable(base, "g1_video", "CURRENT_AUTHORITATIVE_G1_REPLAY_HAS_NO_HEADLESS_CAMERA_ENCODER")
        command = isaac_command(G1_REPLAY, extra, args.view)
    return run_logged(command, base / "logs/replay.log", execute=not args.dry_run)


def import_existing(args: argparse.Namespace) -> int:
    """Register, without converting, an already validated arm/full pair."""
    base = bundle(args.episode_id); make_dirs(base); manifest = load_manifest(base)
    arm_source, full_source = args.arm_input.resolve(), args.full_input.resolve()
    with np.load(arm_source, allow_pickle=False) as arm_z, np.load(full_source, allow_pickle=False) as full_z:
        required_arm = ("g1_arm_joint_trajectory", "arm_joint_names", "fps")
        required_full = ("arm_qpos", "left_dex3_qpos", "right_dex3_qpos", "arm_joint_names", "left_dex3_joint_names", "right_dex3_joint_names", "fps")
        missing = [x for x in required_arm if x not in arm_z.files] + [x for x in required_full if x not in full_z.files]
        if missing: raise RuntimeError(f"existing trajectory metadata missing: {missing}")
        arm = np.asarray(arm_z["g1_arm_joint_trajectory"]); full_arm = np.asarray(full_z["arm_qpos"])
        left, right = np.asarray(full_z["left_dex3_qpos"]), np.asarray(full_z["right_dex3_qpos"])
        if arm.shape != (990, 14) or full_arm.shape != arm.shape or left.shape != (990, 7) or right.shape != (990, 7):
            raise RuntimeError(f"unexpected existing shapes: {arm.shape}/{full_arm.shape}/{left.shape}/{right.shape}")
        if not np.array_equal(arm.astype(np.float32), full_arm.astype(np.float32)):
            raise RuntimeError("existing full trajectory arm does not exactly match authoritative arm")
        arrays = (arm, full_arm, left, right)
        if not all(np.isfinite(x).all() for x in arrays): raise RuntimeError("existing trajectory contains NaN/Inf")
        if float(arm_z["fps"]) != 30.0 or float(full_z["fps"]) != 30.0: raise RuntimeError("existing trajectory FPS is not 30")
        arm_names = arm_z["arm_joint_names"].astype(str).tolist(); full_arm_names = full_z["arm_joint_names"].astype(str).tolist()
        left_names = full_z["left_dex3_joint_names"].astype(str).tolist(); right_names = full_z["right_dex3_joint_names"].astype(str).tolist()
        if arm_names != full_arm_names or len(set(arm_names + left_names + right_names)) != 28:
            raise RuntimeError("existing trajectory joint names/order are inconsistent")
        primitive_source = str(full_z["primitive_source"]) if "primitive_source" in full_z.files else "UNKNOWN"
    arm_dst, full_dst = base / "converted/g1_arm_trajectory.npz", base / "converted/g1_arm_dex3_trajectory.npz"
    for src, dst in ((arm_source, arm_dst), (full_source, full_dst)):
        if dst.exists() and sha256(dst) != sha256(src) and not args.force:
            raise RuntimeError(f"bundle file differs: {dst}; use --force")
        if not dst.exists() or sha256(dst) != sha256(src): shutil.copy2(src, dst)
    registration = {
        "status": "EXISTING_VALIDATED_TRAJECTORY_IMPORTED", "new_conversion_performed": False,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "arm_source": str(arm_source), "arm_source_sha256": sha256(arm_source), "arm_bundle": str(arm_dst),
        "full_source": str(full_source), "full_source_sha256": sha256(full_source), "full_bundle": str(full_dst),
        "shapes": {"arm": [990,14], "full_qpos": [990,50], "left_dex3": [990,7], "right_dex3": [990,7]},
        "fps": 30.0, "arm_joint_names": arm_names, "left_dex3_joint_names": left_names,
        "right_dex3_joint_names": right_names, "primitive_source": primitive_source,
    }
    atomic_json(base / "converted/conversion_metadata.json", registration)
    manifest["existing_trajectory_registration"] = registration
    manifest["generated_files"] = sorted(set(manifest.get("generated_files", []) + [str(arm_dst.relative_to(base)), str(full_dst.relative_to(base)), "converted/conversion_metadata.json"]))
    manifest["validation_status"]["conversion"] = "EXISTING_VALIDATED_TRAJECTORY_IMPORTED"
    manifest["validation_status"]["kinematic_replay"] = "READY_WITH_PREVIEW_POSE; CALIBRATION_JSON_NOT_REQUIRED"
    manifest["validation_status"]["physics"] = "BLOCKED"
    atomic_json(base / "manifest.json", manifest); write_dashboard(base, manifest)
    print(json.dumps(registration, indent=2)); return 0


def unavailable(base: Path, stage: str, reason: str) -> int:
    m = load_manifest(base); m["validation_status"][stage] = "NOT_AVAILABLE"; m.setdefault("failure_reasons", []).append(reason)
    atomic_json(base / "manifest.json", m); write_dashboard(base, m)
    print(f"{stage.upper()} NOT AVAILABLE: {reason}")
    return 5


def compare(args: argparse.Namespace) -> int:
    base = bundle(args.episode_id); m = load_manifest(base)
    a, g = base / "videos/aloha_replay.mp4", base / "videos/g1_kinematic.mp4"
    if not a.exists() or not g.exists():
        return unavailable(base, "comparison", "ALOHA_AND_G1_SYNCHRONIZED_MP4_INPUTS_REQUIRED")
    out = base / "videos/aloha_vs_g1.mp4"
    cmd = ["ffmpeg", "-y", "-i", str(a), "-i", str(g), "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-r", str(m["fps"]), str(out)]
    return run_logged(cmd, base / "logs/compare.log", execute=not args.dry_run)


class ClosedLoopAdapter(Protocol):
    def observe(self) -> dict: ...
    def infer(self, observation: dict, horizon: int) -> np.ndarray: ...
    def convert(self, aloha_chunk: np.ndarray) -> np.ndarray: ...
    def execute(self, g1_chunk: np.ndarray, horizon: int) -> dict: ...


def minimum_closed_loop(adapter: ClosedLoopAdapter, inference_horizon: int, execution_horizon: int, max_steps: int) -> list[dict]:
    """Injectable receding-horizon loop; never reads a prerecorded full trajectory."""
    records = []
    for step in range(max_steps):
        observation = adapter.observe()
        if "rgb" not in observation or "robot_state" not in observation or "instruction" not in observation:
            raise RuntimeError("policy observation requires rgb, robot_state and instruction")
        aloha_chunk = adapter.infer(observation, inference_horizon)
        if len(aloha_chunk) < execution_horizon:
            raise RuntimeError("inference chunk shorter than execution horizon")
        g1_chunk = adapter.convert(aloha_chunk)
        result = adapter.execute(g1_chunk, execution_horizon)
        records.append({"step": step, "replanned": True, **result})
        if result.get("terminated"):
            break
    return records


def physics(args: argparse.Namespace) -> int:
    return unavailable(bundle(args.episode_id), "physics", "CONTACT_DETECTOR_EXISTS_BUT_VERIFIED_G1_PHYSICS_CONTROLLER_AND_APPROVED_DEX3_CALIBRATION_ARE_REQUIRED")


def closed_loop(args: argparse.Namespace) -> int:
    return unavailable(bundle(args.episode_id), "closed_loop", "ISAAC_RGB_STATE_TO_SMOLVLA_ONLINE_ADAPTER_NOT_IMPLEMENTED; RECEDING_HORIZON_CORE_IS_AVAILABLE_FOR_INJECTION")


def status(args: argparse.Namespace) -> int:
    base = bundle(args.episode_id); m = load_manifest(base); print(json.dumps(m, indent=2, ensure_ascii=False)); return 0


def list_episodes(_: argparse.Namespace) -> int:
    TASK_ROOT.mkdir(parents=True, exist_ok=True)
    for p in sorted(TASK_ROOT.iterdir()):
        if (p / "manifest.json").exists():
            m = json.loads((p / "manifest.json").read_text()); print(f"{p.name}\t{m.get('validation_status')}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--episode-id", required=True); q.add_argument("--aloha-action", type=Path, required=True); q.add_argument("--source-type", choices=("raw", "smolvla")); q.add_argument("--timeline", type=Path); q.add_argument("--force", action="store_true"); q.add_argument("--dry-run", action="store_true"); q.set_defaults(func=prepare)
    q = sub.add_parser("replay"); q.add_argument("--episode-id", required=True); q.add_argument("--robot", choices=("aloha", "g1"), required=True); q.add_argument("--mode", choices=("kinematic", "physics"), default="kinematic"); q.add_argument("--view", choices=("gui", "video"), default="gui"); q.add_argument("--camera", choices=("overview", "front", "side", "top"), default="overview"); q.add_argument("--speed", type=float, choices=(.25, .5, 1.0), default=1.0); q.add_argument("--loop", action="store_true"); q.add_argument("--start-frame", type=int, default=0); q.add_argument("--end-frame", type=int); q.add_argument("--dry-run", action="store_true"); q.set_defaults(func=replay)
    q = sub.add_parser("import-existing"); q.add_argument("--episode-id", required=True); q.add_argument("--arm-input", type=Path, required=True); q.add_argument("--full-input", type=Path, required=True); q.add_argument("--force", action="store_true"); q.set_defaults(func=import_existing)
    q = sub.add_parser("compare"); q.add_argument("--episode-id", required=True); q.add_argument("--dry-run", action="store_true"); q.set_defaults(func=compare)
    q = sub.add_parser("physics"); q.add_argument("--episode-id", required=True); q.set_defaults(func=physics)
    q = sub.add_parser("closed-loop"); q.add_argument("--episode-id", required=True); q.add_argument("--inference-horizon", type=int, default=50); q.add_argument("--execution-horizon", type=int, default=10); q.add_argument("--maximum-control-steps", type=int, default=100); q.add_argument("--camera", default="overview"); q.add_argument("--checkpoint", type=Path); q.add_argument("--task-instruction", default="Perform the MagSafe task"); q.add_argument("--seed", type=int, default=0); q.add_argument("--record-video", action="store_true"); q.set_defaults(func=closed_loop)
    q = sub.add_parser("status"); q.add_argument("--episode-id", required=True); q.set_defaults(func=status)
    q = sub.add_parser("list"); q.set_defaults(func=list_episodes)
    return p


def main() -> int:
    print(BANNER)
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
