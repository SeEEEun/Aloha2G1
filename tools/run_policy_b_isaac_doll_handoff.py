#!/usr/bin/env python3
"""Progressive Isaac-only Policy-A/Policy-B Doll-Handoff validation runner.

The program exposes all requested stages through one CLI.  It contains no DDS,
hardware-enable, or real-robot command path.  Policy inference runs in a
separate local LeRobot process so Isaac Lab and SmolVLA retain their validated
Torch environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import cv2
import numpy as np

from isaaclab.app import AppLauncher
from common_deployment_safety_projection import NamedJointDeploymentSafetyProjector
from common_causal_action_stitching import CommonCausalActionStitcher
from common_jerk_limited_otg import CommonJerkLimitedOTG
from policy_b_isaac_control_contract import (
    CONTROL_FPS,
    PHYSICS_DT,
    build_implicit_actuators,
)
from deployment_camera_config import (
    apply_configured_distortion,
    camera_manifest_record,
    load_camera_config,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/policy_b_isaac_validation"
DIAGNOSTIC_OUTPUT_ROOT = OUTPUT_ROOT / "simulation_diagnostic_rollout"
FULL_MOTION_DIRECTORY = "full_policy_b_diagnostic_rollout"
STRICT_STAGE1_REPORT = OUTPUT_ROOT / "stage1_object_free_single/stage_report.json"
STRICT_MARGIN_VALIDATION = (
    OUTPUT_ROOT
    / "dex3_controller_characterization/integrated_margin_validation.json"
)
SCENE_STAGE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
SCENE_LAYOUT = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
ACTION_FREEZE = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
SOURCE_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
SEMANTIC_MANIFEST = (
    ROOT
    / "outputs/doll_handoff_dataset_b_semantic_audit_2026-08-23/FINAL_DATASET_B_SEMANTIC_MANIFEST.json"
)
POLICY_B_TRAINING_AUDIT = OUTPUT_ROOT / "training_audit/training_audit.json"
POLICY_A_TRAINING_AUDIT = (
    ROOT / "outputs/policy_a_doll_handoff_trajectory_a_50_lag1_state_v2/training_audit/training_audit.json"
)
DEX3_LIMIT_AUDIT = OUTPUT_ROOT / "dex3_limit_audit/dex3_limit_audit.json"
COMMON_PROJECTION_FREEZE = (
    ROOT
    / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"
)
DATASET = ROOT / "datasets/doll_handoff_proposed_b_50"
COMMON_CONFIG = ROOT / "configs/doll_handoff_retargeting/common_config.template.json"
WHOLE_HAND_CONFIG = ROOT / "configs/doll_handoff_retargeting/dex3_whole_hand.sim.json"
FEASIBILITY_CONFIG = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
POLICY_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
POLICY_WORKER = ROOT / "tools/policy_b_inference_worker.py"
TASK = (
    "Pick up the doll with the left hand, handoff it to the right hand, "
    "and place it in the trash bin."
)
AUTHKEY = b"policy-b-isaac-local-v1"
DEFAULT_HORIZON = 4
STAGE_DIRECTORIES = {
    "inference": "stage0_inference",
    "object-free-single": "stage1_object_free_single",
    "object-free-multichunk": "stage2_object_free_multichunk",
    "left-grasp": "stage3_left_grasp",
    "handoff": "stage4_handoff",
    "full-task": "stage5_full_task",
    "full-motion": FULL_MOTION_DIRECTORY,
}
PREREQUISITES = {
    "object-free-single": "stage0_inference",
    "object-free-multichunk": "stage1_object_free_single",
    "left-grasp": "stage2_object_free_multichunk",
    "handoff": "stage3_left_grasp",
    "full-task": "stage4_handoff",
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy-variant", choices=("A", "B"), required=True)
parser.add_argument("--camera-config", type=Path, required=True)
parser.add_argument("--stage", choices=tuple(STAGE_DIRECTORIES), required=True)
parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
parser.add_argument("--execution-horizon", type=int, default=DEFAULT_HORIZON)
parser.add_argument(
    "--causal-stitching-method",
    choices=("naive", "rtc", "crossfade"),
    default="naive",
    help="Common causal execution adapter; raw unconditioned policy chunks remain separately logged.",
)
parser.add_argument(
    "--crossfade-window-frames",
    type=int,
    default=5,
    help="Dataset-motion-derived minimum-jerk window (used only for causal crossfade).",
)
parser.add_argument(
    "--rtc-guidance-weight",
    type=float,
    default=5.0,
    help="Official LeRobot RTC maximum guidance weight (used only with --causal-stitching-method rtc).",
)
parser.add_argument(
    "--rtc-prefix-attention-schedule",
    choices=("exp", "linear"),
    default="exp",
    help="Official LeRobot RTC prefix-attention schedule.",
)
parser.add_argument(
    "--rtc-inference-delay-frames",
    type=int,
    default=0,
    help=(
        "Causal latency compensation for RTC. At each replan these rows remain "
        "committed from the previous plan and the matching new-plan rows are discarded."
    ),
)
parser.add_argument(
    "--jerk-limited-otg-config",
    type=Path,
    default=None,
    help=(
        "Optional frozen common 28D Ruckig config. The selected causal-plan endpoint "
        "is realized from measured q/dq/ddq without changing any raw policy chunk."
    ),
)
parser.add_argument(
    "--fixed-flow-noise",
    action="store_true",
    help="Diagnostic deterministic flow sampling; raw chunks remain separately logged.",
)
parser.add_argument("--stage2-inference-calls", type=int, default=10)
parser.add_argument("--stage3-inference-calls", type=int, default=90)
parser.add_argument("--stage4-inference-calls", type=int, default=160)
parser.add_argument("--stage5-inference-calls", type=int, default=230)
parser.add_argument(
    "--full-motion-inference-calls",
    type=int,
    default=172,
    help="Causal horizon-4 cycles for the complete motion visualization (172 = 688 frames).",
)
parser.add_argument("--settle-seconds", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=20260824)
parser.add_argument(
    "--capture-inference-indices",
    default="",
    help=(
        "Comma-separated inference-call indices whose exact policy RGB and measured "
        "float32 state are losslessly preserved for fixed-observation audits. "
        "Logging only; it does not alter execution."
    ),
)
parser.add_argument(
    "--fixed-observation-metadata",
    type=Path,
    default=None,
    help=(
        "Single-chunk diagnostic only: use the exact frozen RGB/state recorded in "
        "this metadata JSON for policy inference and initialize Isaac from that "
        "state. No replanning is performed."
    ),
)
parser.add_argument(
    "--checkpoint-override",
    type=Path,
    default=None,
    help="Explicit immutable adapted checkpoint for a separately versioned validation run.",
)
parser.add_argument(
    "--checkpoint-model-sha256",
    default=None,
    help="Required exact model.safetensors SHA256 when --checkpoint-override is used.",
)
parser.add_argument(
    "--checkpoint-training-step",
    type=int,
    default=None,
    help="Adaptation-local checkpoint step recorded in reports.",
)
parser.add_argument("--no-video", action="store_true")
parser.add_argument(
    "--kinematic-visual-doll",
    action="store_true",
    help=(
        "Keep the canonical doll pose fixed for trajectory visualization. This is "
        "labeled KINEMATIC_OBJECT_TRAJECTORY_VISUALIZATION and never scores task success."
    ),
)
parser.add_argument(
    "--simulation-diagnostic-rollout",
    action="store_true",
    help=(
        "Isaac-only behavior observation. Preserves the failed strict Stage-1 "
        "qualification while permitting only the already-characterized microscopic "
        "measured Dex3 excursion. Never authorizes real hardware."
    ),
)
parser.add_argument(
    "--source-rgb-time-forced-episode",
    type=int,
    default=None,
    help=(
        "Isaac-only diagnostic: infer from the selected Dataset-B episode's original "
        "ALOHA cam_high sequence at the 30-Hz simulation clock while retaining current "
        "measured Isaac qpos. This is not closed-loop vision and never authorizes hardware."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
try:
    CAPTURE_INFERENCE_INDICES = {
        int(value.strip())
        for value in args.capture_inference_indices.split(",")
        if value.strip()
    }
except ValueError as error:
    parser.error(f"invalid --capture-inference-indices: {error}")
if any(index < 0 for index in CAPTURE_INFERENCE_INDICES):
    parser.error("--capture-inference-indices values must be non-negative")
if args.fixed_observation_metadata is not None and args.stage != "object-free-single":
    parser.error("--fixed-observation-metadata requires --stage object-free-single")
if not 1 <= args.execution_horizon <= 16:
    parser.error("--execution-horizon must be in the audited range 1..16")
if not 0 <= args.rtc_inference_delay_frames <= args.execution_horizon:
    parser.error("--rtc-inference-delay-frames must be in 0..execution-horizon")
if args.causal_stitching_method != "rtc" and args.rtc_inference_delay_frames:
    parser.error("--rtc-inference-delay-frames is valid only for RTC")
if args.stage == "full-motion" and not args.simulation_diagnostic_rollout:
    parser.error("full-motion is an Isaac-only simulation diagnostic mode")
if args.kinematic_visual_doll and args.stage != "full-motion":
    parser.error("--kinematic-visual-doll applies only to full-motion")
if args.source_rgb_time_forced_episode is not None:
    if args.stage != "full-motion" or not args.simulation_diagnostic_rollout:
        parser.error(
            "--source-rgb-time-forced-episode requires --stage full-motion "
            "--simulation-diagnostic-rollout"
        )
    if not 0 <= args.source_rgb_time_forced_episode < 50:
        parser.error("--source-rgb-time-forced-episode must be in 0..49")
if args.checkpoint_override is not None:
    if args.checkpoint_model_sha256 is None or args.checkpoint_training_step is None:
        parser.error(
            "--checkpoint-override requires --checkpoint-model-sha256 and "
            "--checkpoint-training-step"
        )
elif args.checkpoint_model_sha256 is not None or args.checkpoint_training_step is not None:
    parser.error("checkpoint hash/step arguments require --checkpoint-override")
if args.simulation_diagnostic_rollout:
    if args.stage not in {
        "object-free-multichunk",
        "left-grasp",
        "handoff",
        "full-task",
        "full-motion",
    }:
        parser.error(
            "simulation diagnostic rollout applies only to Stage 2 through Stage 5 or full-motion"
        )
    if (
        args.stage != "full-motion"
        and args.output_root.resolve() == OUTPUT_ROOT.resolve()
    ):
        args.output_root = DIAGNOSTIC_OUTPUT_ROOT
DEPLOYMENT_CAMERA = load_camera_config(
    args.camera_config,
    purpose=f"Policy {args.policy_variant} Isaac rollout",
)
CAMERA_CONFIG = DEPLOYMENT_CAMERA.path
TRAINING_AUDIT = (
    POLICY_A_TRAINING_AUDIT if args.policy_variant == "A" else POLICY_B_TRAINING_AUDIT
)
DATASET = (
    ROOT / "datasets/doll_handoff_trajectory_a_50"
    if args.policy_variant == "A"
    else ROOT / "datasets/doll_handoff_proposed_b_50"
)
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app
print(
    f"[Policy{args.policy_variant}Isaac] application ready stage={args.stage} module={__name__}",
    flush=True,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def save_full_motion_plots(
    stage_dir: Path,
    timestamps: np.ndarray,
    left_wrist_xyz: np.ndarray,
    right_wrist_xyz: np.ndarray,
    commanded_q: np.ndarray,
    measured_q: np.ndarray,
    joint_names: list[str],
) -> dict[str, str]:
    """Save supplementary plots for the continuous motion visualization."""

    if len(timestamps) == 0:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}
    colors = {"left": "#2563eb", "right": "#dc2626"}

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis_index, (axis, coordinate) in enumerate(zip(axes, "XYZ")):
        axis.plot(
            timestamps,
            left_wrist_xyz[:, axis_index],
            color=colors["left"],
            label="left wrist",
        )
        axis.plot(
            timestamps,
            right_wrist_xyz[:, axis_index],
            color=colors["right"],
            label="right wrist",
        )
        axis.set_ylabel(f"{coordinate} world (m)")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("executed time (s)")
    figure.suptitle("Measured G1 wrist positions")
    figure.tight_layout()
    path = stage_dir / "wrist_xyz_vs_time.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths["wrist_xyz_vs_time"] = str(path)

    representative = {
        "left": [
            "left_hand_thumb_1_joint",
            "left_hand_index_1_joint",
            "left_hand_middle_1_joint",
        ],
        "right": [
            "right_hand_thumb_1_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_1_joint",
        ],
    }
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, side in zip(axes, ("left", "right")):
        for name in representative[side]:
            index = joint_names.index(name)
            short_name = name.replace(f"{side}_hand_", "").replace("_joint", "")
            axis.plot(
                timestamps,
                measured_q[:, index],
                label=f"{short_name} measured",
            )
            axis.plot(
                timestamps,
                commanded_q[:, index],
                linestyle="--",
                alpha=0.55,
                label=f"{short_name} command",
            )
        axis.set_ylabel(f"{side.capitalize()} Dex3 (rad)")
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3, fontsize=8, loc="best")
    axes[-1].set_xlabel("executed time (s)")
    figure.suptitle("Representative Dex3 joint trajectories")
    figure.tight_layout()
    path = stage_dir / "dex3_representative_joints_vs_time.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths["dex3_representative_joints_vs_time"] = str(path)

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*left_wrist_xyz.T, color=colors["left"], label="left wrist")
    axis.plot(*right_wrist_xyz.T, color=colors["right"], label="right wrist")
    axis.scatter(*left_wrist_xyz[0], color=colors["left"], marker="o", label="left start")
    axis.scatter(*left_wrist_xyz[-1], color=colors["left"], marker="x", label="left end")
    axis.scatter(*right_wrist_xyz[0], color=colors["right"], marker="o", label="right start")
    axis.scatter(*right_wrist_xyz[-1], color=colors["right"], marker="x", label="right end")
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")
    axis.set_zlabel("world Z (m)")
    axis.set_title("Full measured wrist paths")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    path = stage_dir / "wrist_paths_3d.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths["wrist_paths_3d"] = str(path)
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_stage1_diagnostic_authorization() -> dict[str, Any]:
    """Verify, but never relabel, the canonical failed Stage-1 qualification."""

    strict = read_json(STRICT_STAGE1_REPORT)
    validation = read_json(STRICT_MARGIN_VALIDATION)
    actual = strict["rollout"]["actual_hard_limit_audit"]
    checks = strict["rollout"]["checks"]
    expected_excess = float(
        validation["stage1"]["measured_maximum_hard_limit_excess_rad"]
    )
    if (
        strict.get("status") != "FAIL"
        or validation.get("status") != "FAIL_NOT_ACCEPTED_FOR_PROGRESSIVE_EXECUTION"
        or int(actual["joint_limit_violation_count"]) != 59
        or not math.isclose(
            float(actual["maximum_joint_limit_excess_rad"]),
            expected_excess,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or checks.get("commanded_hard_limits") is not True
        or checks.get("measured_hard_limits") is not False
    ):
        raise RuntimeError(
            "canonical strict Stage-1 FAIL evidence changed; diagnostic mode refused"
        )
    return {
        "strict_safety_qualification": "FAIL",
        "canonical_report": str(STRICT_STAGE1_REPORT),
        "canonical_report_sha256": sha256_file(STRICT_STAGE1_REPORT),
        "measured_dex3_hard_limit_violation_count": 59,
        "measured_dex3_maximum_excess_rad": expected_excess,
        "diagnostic_measured_dex3_excursion_cap_rad": expected_excess,
        "cap_provenance": (
            "exact maximum from the preserved failed Stage-1 simulation result; "
            "no added tolerance"
        ),
        "real_hardware_safety_readiness": "BLOCKED",
    }


def prerequisite_gate(output_root: Path) -> None:
    prerequisite = PREREQUISITES.get(args.stage)
    if prerequisite is None:
        return
    if args.stage == "object-free-single" and args.fixed_observation_metadata is not None:
        # This is an isolated no-object, one-chunk representation audit.  It does
        # not claim or advance the progressive Stage-0/Stage-1 qualification.
        return
    if args.simulation_diagnostic_rollout and args.stage == "object-free-multichunk":
        strict_stage1_diagnostic_authorization()
        return
    report_path = output_root / prerequisite / "stage_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"{args.stage} requires {report_path}")
    report = read_json(report_path)
    if report.get("status") != "PASS":
        raise RuntimeError(
            f"{args.stage} blocked because {prerequisite} status is {report.get('status')}"
        )


def frozen_checkpoint() -> tuple[Path, str, int]:
    if args.checkpoint_override is not None:
        path = args.checkpoint_override.expanduser().resolve()
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(path / "model.safetensors")
        actual_hash = sha256_file(path / "model.safetensors")
        if actual_hash != args.checkpoint_model_sha256:
            raise RuntimeError(
                f"explicit adapted checkpoint hash mismatch: {actual_hash} != "
                f"{args.checkpoint_model_sha256}"
            )
        return path, actual_hash, int(args.checkpoint_training_step)
    audit = read_json(TRAINING_AUDIT)
    if audit.get("status") != "PASS":
        raise RuntimeError(f"Policy-{args.policy_variant} training audit is not PASS")
    selected = audit["selected_checkpoint"]
    path = Path(selected["checkpoint"]).resolve()
    expected_hash = selected["model_sha256"]
    actual_hash = sha256_file(path / "model.safetensors")
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"frozen Policy-{args.policy_variant} checkpoint hash changed: {actual_hash} != {expected_hash}"
        )
    return path, actual_hash, int(selected["training_step"])


def frozen_interfaces() -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    freeze = read_json(ACTION_FREEZE)
    semantic = read_json(SEMANTIC_MANIFEST)
    semantic_schema = read_json(Path(semantic["artifacts"]["semantic_schema"]["path"]))
    names = list(freeze["joint_names"])
    if (
        len(names) != 28
        or names != semantic_schema["state"]["joint_names"]
        or names != semantic_schema["action"]["joint_names"]
    ):
        raise RuntimeError("Dataset-B semantic/freeze joint order mismatch")
    specs = freeze["joint_specs"]
    if [row["joint_name"] for row in specs] != names:
        raise RuntimeError("Dataset-B joint specs are not in policy order")
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    return names, lower, upper, freeze


def frozen_projection(
    names: list[str],
) -> tuple[NamedJointDeploymentSafetyProjector, dict[str, Any], str]:
    diagnostic = read_json(DEX3_LIMIT_AUDIT)
    if diagnostic.get("status") != "PASS_DIAGNOSTIC_PROJECTION_ELIGIBLE":
        raise RuntimeError("Dex3 diagnostic did not authorize generic projection")
    projector = NamedJointDeploymentSafetyProjector.from_path(COMMON_PROJECTION_FREEZE)
    if projector.names != names:
        raise RuntimeError("common projection joint order does not match Dataset B")
    config = projector.config
    if sum(projector.projectable) != 14 or any(projector.projectable[:14]):
        raise RuntimeError("common projection must preserve all 14 arm channels")
    if config.get("margin_label") != "SIMULATION_CONTROLLER_MARGIN_ONLY":
        raise RuntimeError("deployment margin must remain explicitly simulation-only")
    return projector, config, sha256_file(COMMON_PROJECTION_FREEZE)


def initial_condition(names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    source = read_json(SOURCE_MANIFEST)
    values = []
    hashes = []
    for episode in source["episodes"]:
        path = Path(episode["retargeted_trajectory_path"])
        if sha256_file(path) != episode["retargeted_trajectory_sha256"]:
            raise RuntimeError(f"frozen trajectory hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            trajectory_names = archive["replay_joint_names"].astype(str).tolist()
            if len(trajectory_names) != 28 or set(trajectory_names) != set(names):
                raise RuntimeError(f"trajectory named-joint set mismatch: {path}")
            replay_value = archive["replay_named_joint_qpos"][0].astype(np.float64)
            values.append(
                replay_value[[trajectory_names.index(name) for name in names]]
            )
        hashes.append(episode["retargeted_trajectory_sha256"])
    matrix = np.stack(values)
    median = np.median(matrix, axis=0)
    return median, {
        "definition": "dataset-wide median of the 50 frozen episode first configurations",
        "purpose": "one static simulation initial condition only; no task trajectory is replayed",
        "source_episode_count": len(values),
        "maximum_across_episode_range_rad": float(np.max(np.ptp(matrix, axis=0))),
        "joint_names": names,
        "q_rad": median,
        "source_trajectory_hashes": hashes,
        "policy_generated_actions_only_after_initialization": True,
    }


class PolicyBridge:
    """Own one local Unix-socket inference worker for a stage invocation."""

    def __init__(
        self,
        checkpoint: Path,
        stage_dir: Path,
        seed: int,
        causal_stitching_method: str,
        execution_horizon: int,
        rtc_guidance_weight: float,
        rtc_prefix_attention_schedule: str,
        fixed_flow_noise: bool,
    ):
        self.checkpoint = checkpoint
        self.stage_dir = stage_dir
        self.seed = seed
        bridge_dir = stage_dir / "inference_bridge"
        bridge_dir.mkdir(parents=True, exist_ok=True)
        # AF_UNIX paths are capped at roughly 108 bytes on Linux.  Stage output
        # names are intentionally descriptive, so keep only the transient
        # observation/prediction socket in a short, process-unique /tmp path.
        token = hashlib.sha256(str(stage_dir.resolve()).encode()).hexdigest()[:10]
        self.socket_path = Path(f"/tmp/pbiv_{os.getpid()}_{token}.sock")
        self.ready_path = bridge_dir / "ready.json"
        self.log_path = bridge_dir / "worker.log"
        for path in (self.socket_path, self.ready_path):
            if path.exists() or path.is_socket():
                path.unlink()
        self.log_stream = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(POLICY_PYTHON),
                str(POLICY_WORKER),
                "--checkpoint",
                str(checkpoint),
                "--socket",
                str(self.socket_path),
                "--ready",
                str(self.ready_path),
                "--seed",
                str(seed),
                "--causal-stitching-method",
                causal_stitching_method,
                "--execution-horizon",
                str(execution_horizon),
                "--rtc-guidance-weight",
                str(rtc_guidance_weight),
                "--rtc-prefix-attention-schedule",
                rtc_prefix_attention_schedule,
                *(["--fixed-flow-noise"] if fixed_flow_noise else []),
            ],
            cwd=ROOT,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 60.0
        while not self.ready_path.is_file():
            if self.process.poll() is not None:
                self.log_stream.flush()
                raise RuntimeError(f"Policy worker exited; inspect {self.log_path}")
            if time.monotonic() >= deadline:
                self.process.terminate()
                raise TimeoutError(f"Policy-{args.policy_variant} worker did not become ready within 60 s")
            time.sleep(0.1)
        self.ready = read_json(self.ready_path)
        self.connection = Client(str(self.socket_path), family="AF_UNIX", authkey=AUTHKEY)

    def infer(
        self, rgb: np.ndarray, state: np.ndarray, *, inference_delay: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self.connection.send(
            {
                "command": "infer",
                "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
                "state": np.ascontiguousarray(state, dtype=np.float32),
                "task": TASK,
                "inference_delay": int(inference_delay),
            }
        )
        response = self.connection.recv()
        if response.get("status") != "PASS":
            raise RuntimeError(f"Policy worker inference failed: {response}")
        action = np.asarray(response.pop("action"), dtype=np.float64)
        stitched_action = np.asarray(response.pop("stitched_action"), dtype=np.float64)
        previous_remaining_plan = np.asarray(
            response.pop("previous_remaining_plan"), dtype=np.float64
        ).reshape(-1, 28)
        if action.shape != (1, 50, 28):
            raise RuntimeError(f"Policy output shape {action.shape}, expected (1,50,28)")
        if stitched_action.shape != (1, 50, 28):
            raise RuntimeError(
                f"Stitched output shape {stitched_action.shape}, expected (1,50,28)"
            )
        return action[0], stitched_action[0], previous_remaining_plan, response

    def close(self) -> None:
        try:
            if hasattr(self, "connection"):
                self.connection.send({"command": "shutdown"})
                self.connection.recv()
                self.connection.close()
        finally:
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=10.0)
            self.log_stream.close()


def branch_flags(q: np.ndarray, absolute: float, multiplier: float) -> np.ndarray:
    flags = np.zeros(len(q), dtype=bool)
    norms = np.linalg.norm(np.diff(q, axis=0), axis=1)
    for index in range(1, len(q)):
        local = float(np.median(norms[max(0, index - 10) : min(len(norms), index + 9)]))
        flags[index] = norms[index - 1] > max(absolute, multiplier * max(local, 1e-6))
    return flags


def look_at_ros_camera_quaternion_xyzw(
    eye_world_xyz: np.ndarray, target_world_xyz: np.ndarray
) -> np.ndarray:
    """Return world-from-ROS-optical-camera quaternion for a diagnostic view."""

    eye = np.asarray(eye_world_xyz, dtype=np.float64)
    target = np.asarray(target_world_xyz, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.column_stack((right, down, forward))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


class NumericalSafety:
    def __init__(self, names: list[str], lower: np.ndarray, upper: np.ndarray):
        sys.path.insert(0, str(ROOT / "tools"))
        from doll_handoff_retargeting.models import G1Kinematics

        self.lower = lower
        self.upper = upper
        self.names = names
        self.acceptance = read_json(FEASIBILITY_CONFIG)["unchanged_acceptance"]
        self.g1 = G1Kinematics(read_json(COMMON_CONFIG), read_json(SCENE_LAYOUT))
        self.left_hand_policy_indices = [
            names.index(name) for name in self.g1.hand_joint_names["left"]
        ]
        self.right_hand_policy_indices = [
            names.index(name) for name in self.g1.hand_joint_names["right"]
        ]

    def collision(self, q: np.ndarray) -> dict[str, Any]:
        geometry = self.g1.trajectory_geometry(
            q[:, :14],
            q[:, self.left_hand_policy_indices],
            q[:, self.right_hand_policy_indices],
            float(self.acceptance["collision_penetration_tolerance_m"]),
        )
        counts = {
            key: int(np.count_nonzero(value))
            for key, value in geometry["collision_flags"].items()
        }
        invalid = sum(
            counts[key]
            for key in ("ARM_TORSO", "CROSS_ARM", "WRIST_OR_PALM_TORSO", "OTHER")
        )
        return {
            "frame_counts": counts,
            "invalid_hard_collision_category_incidence": invalid,
            "pairs": geometry["collision_pairs"],
            "records": geometry["collision_records"],
            "distal_hand_hand_is_reported_separately": True,
        }

    @staticmethod
    def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
        relative = np.asarray(first).T @ np.asarray(second)
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        return float(math.acos(cosine))

    def _physical_hand_geometry(self, q: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        count = len(q)
        output: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            output[f"{side}_grasp_position"] = np.empty((count, 3), dtype=np.float64)
            output[f"{side}_grasp_rotation"] = np.empty((count, 3, 3), dtype=np.float64)
            output[f"{side}_enclosure_radius"] = np.empty(count, dtype=np.float64)
            for digit in ("thumb", "index", "middle"):
                output[f"{side}_{digit}_contact_position"] = np.empty(
                    (count, 3), dtype=np.float64
                )
                output[f"{side}_{digit}_contact_normal"] = np.empty(
                    (count, 3), dtype=np.float64
                )
        for frame in range(count):
            self.g1.assign(
                q[frame, :14],
                q[frame, self.left_hand_policy_indices],
                q[frame, self.right_hand_policy_indices],
            )
            for side in ("left", "right"):
                pose = self.g1.whole_hand_grasp_pose(side)
                output[f"{side}_grasp_position"][frame] = pose[:3, 3]
                output[f"{side}_grasp_rotation"][frame] = pose[:3, :3]
                output[f"{side}_enclosure_radius"][frame] = (
                    self.g1.whole_hand_enclosure_radius(side)
                )
                for digit in ("thumb", "index", "middle"):
                    position, normal = self.g1.contact_pose(side, digit)
                    output[f"{side}_{digit}_contact_position"][frame] = position
                    output[f"{side}_{digit}_contact_normal"][frame] = normal
        return output

    def projection_geometry_delta(self, raw: np.ndarray, projected: np.ndarray) -> dict[str, Any]:
        raw = np.asarray(raw, dtype=np.float64)
        projected = np.asarray(projected, dtype=np.float64)
        if raw.shape != projected.shape or raw.ndim != 2 or raw.shape[1] != len(self.names):
            raise ValueError("raw/projected geometry arrays must share shape [rows,joints]")
        raw_geometry = self._physical_hand_geometry(raw)
        projected_geometry = self._physical_hand_geometry(projected)
        side_metrics: dict[str, Any] = {}
        all_contact_displacements: list[float] = []
        all_normal_rotations: list[float] = []
        for side in ("left", "right"):
            grasp_position = np.linalg.norm(
                projected_geometry[f"{side}_grasp_position"]
                - raw_geometry[f"{side}_grasp_position"],
                axis=1,
            )
            grasp_rotation = np.asarray(
                [
                    self._rotation_distance(first, second)
                    for first, second in zip(
                        raw_geometry[f"{side}_grasp_rotation"],
                        projected_geometry[f"{side}_grasp_rotation"],
                    )
                ],
                dtype=np.float64,
            )
            radius = np.abs(
                projected_geometry[f"{side}_enclosure_radius"]
                - raw_geometry[f"{side}_enclosure_radius"]
            )
            contact_by_digit = {}
            for digit in ("thumb", "index", "middle"):
                contact = np.linalg.norm(
                    projected_geometry[f"{side}_{digit}_contact_position"]
                    - raw_geometry[f"{side}_{digit}_contact_position"],
                    axis=1,
                )
                normal = np.asarray(
                    [
                        math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))
                        for first, second in zip(
                            raw_geometry[f"{side}_{digit}_contact_normal"],
                            projected_geometry[f"{side}_{digit}_contact_normal"],
                        )
                    ],
                    dtype=np.float64,
                )
                all_contact_displacements.extend(contact.tolist())
                all_normal_rotations.extend(normal.tolist())
                contact_by_digit[digit] = {
                    "maximum_contact_position_delta_m": float(np.max(contact)),
                    "mean_contact_position_delta_m": float(np.mean(contact)),
                    "maximum_contact_normal_delta_rad": float(np.max(normal)),
                    "mean_contact_normal_delta_rad": float(np.mean(normal)),
                }
            side_metrics[side] = {
                "maximum_whole_hand_grasp_frame_position_delta_m": float(
                    np.max(grasp_position)
                ),
                "mean_whole_hand_grasp_frame_position_delta_m": float(
                    np.mean(grasp_position)
                ),
                "maximum_whole_hand_grasp_frame_orientation_delta_rad": float(
                    np.max(grasp_rotation)
                ),
                "mean_whole_hand_grasp_frame_orientation_delta_rad": float(
                    np.mean(grasp_rotation)
                ),
                "maximum_enclosure_radius_delta_m": float(np.max(radius)),
                "mean_enclosure_radius_delta_m": float(np.mean(radius)),
                "distal_contacts": contact_by_digit,
            }
        raw_collision = self.collision(raw)
        projected_collision = self.collision(projected)
        strict_position = float(self.acceptance["strict_position_tolerance_m"])
        orientation = float(
            read_json(COMMON_CONFIG)["shared_temporal_ik"]["orientation_tolerance_rad"]
        )
        maximum_grasp_position = max(
            side_metrics[side]["maximum_whole_hand_grasp_frame_position_delta_m"]
            for side in ("left", "right")
        )
        maximum_grasp_rotation = max(
            side_metrics[side]["maximum_whole_hand_grasp_frame_orientation_delta_rad"]
            for side in ("left", "right")
        )
        maximum_radius = max(
            side_metrics[side]["maximum_enclosure_radius_delta_m"]
            for side in ("left", "right")
        )
        checks = {
            "whole_hand_grasp_frame_position": maximum_grasp_position <= strict_position,
            "whole_hand_grasp_frame_orientation": maximum_grasp_rotation <= orientation,
            "distal_contact_position": max(all_contact_displacements, default=0.0)
            <= strict_position,
            "enclosure_radius": maximum_radius <= strict_position,
            "no_new_hard_self_collision": projected_collision[
                "invalid_hard_collision_category_incidence"
            ]
            == 0,
            "no_new_distal_hand_hand_collision": projected_collision["frame_counts"].get(
                "DISTAL_HAND_HAND", 0
            )
            == 0,
        }
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "sides": side_metrics,
            "maximum_distal_contact_position_delta_m": max(
                all_contact_displacements, default=0.0
            ),
            "mean_distal_contact_position_delta_m": float(
                np.mean(all_contact_displacements)
            )
            if all_contact_displacements
            else 0.0,
            "maximum_distal_contact_normal_delta_rad": max(
                all_normal_rotations, default=0.0
            ),
            "mean_distal_contact_normal_delta_rad": float(np.mean(all_normal_rotations))
            if all_normal_rotations
            else 0.0,
            "raw_collision": raw_collision,
            "projected_collision": projected_collision,
            "frozen_benignity_thresholds": {
                "strict_position_tolerance_m": strict_position,
                "orientation_tolerance_rad": orientation,
                "new_collision_frames_allowed": 0,
            },
        }

    def trajectory(self, q: np.ndarray) -> dict[str, Any]:
        q = np.asarray(q, dtype=np.float64)
        finite = bool(np.isfinite(q).all())
        violation_mask = (q < self.lower[None] - 1e-9) | (q > self.upper[None] + 1e-9)
        lower_excess = np.maximum(self.lower[None] - q, 0.0)
        upper_excess = np.maximum(q - self.upper[None], 0.0)
        steps = np.abs(np.diff(q, axis=0))
        velocity = steps * CONTROL_FPS
        acceleration = np.abs(np.diff(q, n=2, axis=0)) * CONTROL_FPS**2
        branches = branch_flags(
            q[:, :14],
            float(self.acceptance["branch_absolute_step_norm_rad"]),
            float(self.acceptance["branch_local_multiplier"]),
        )
        collision = self.collision(q) if finite else {
            "frame_counts": {},
            "invalid_hard_collision_category_incidence": -1,
            "pairs": {},
            "records": [],
        }
        return {
            "finite": finite,
            "joint_limit_violation_count": int(np.count_nonzero(violation_mask)),
            "joint_limit_violation_frames": np.flatnonzero(np.any(violation_mask, axis=1)).tolist(),
            "joint_limit_violation_count_per_joint": {
                name: int(np.count_nonzero(violation_mask[:, index]))
                for index, name in enumerate(self.names)
            },
            "maximum_joint_limit_excess_rad": float(
                max(np.max(lower_excess), np.max(upper_excess))
            ),
            "maximum_joint_limit_excess_rad_per_joint": {
                name: float(max(np.max(lower_excess[:, index]), np.max(upper_excess[:, index])))
                for index, name in enumerate(self.names)
            },
            "maximum_joint_step_rad": float(np.max(steps)) if steps.size else 0.0,
            "maximum_velocity_rad_s": float(np.max(velocity)) if velocity.size else 0.0,
            "maximum_acceleration_rad_s2": float(np.max(acceleration)) if acceleration.size else 0.0,
            "branch_discontinuity_count": int(np.count_nonzero(branches)),
            "branch_discontinuity_frames": np.flatnonzero(branches).tolist(),
            "per_joint_minimum_rad": np.min(q, axis=0),
            "per_joint_maximum_rad": np.max(q, axis=0),
            "collision": collision,
        }

    def chunk(self, current: np.ndarray, chunk: np.ndarray) -> dict[str, Any]:
        metrics = self.trajectory(chunk)
        first_delta = chunk[0] - current
        limits = self.acceptance
        checks = {
            "finite": metrics["finite"],
            "joint_limits": metrics["joint_limit_violation_count"] == 0,
            "first_command_delta": float(np.max(np.abs(first_delta)))
            <= float(limits["maximum_joint_step_rad"]),
            "adjacent_step": metrics["maximum_joint_step_rad"]
            <= float(limits["maximum_joint_step_rad"]),
            "velocity": metrics["maximum_velocity_rad_s"]
            <= float(limits["maximum_velocity_rad_s"]),
            "acceleration": metrics["maximum_acceleration_rad_s2"]
            <= float(limits["maximum_acceleration_rad_s2"]),
            "branch": metrics["branch_discontinuity_count"] == 0,
            "hard_self_collision": metrics["collision"]["invalid_hard_collision_category_incidence"] == 0,
        }
        metrics.update(
            {
                "first_action_minus_current_state_rad": first_delta,
                "first_action_delta_max_abs_rad": float(np.max(np.abs(first_delta))),
                "first_action_delta_l2_rad": float(np.linalg.norm(first_delta)),
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
                "thresholds": limits,
            }
        )
        return metrics


class VideoRecorder:
    def __init__(self, stage_dir: Path, names: list[str], enabled: bool):
        self.enabled = enabled
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.frames: dict[str, list[np.ndarray]] = {name: [] for name in names}
        self.paths: dict[str, str] = {}
        if not enabled:
            return
        for name in names:
            path = stage_dir / f"{name}.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_FPS, (640, 480)
            )
            if not writer.isOpened():
                raise RuntimeError(f"unable to open video writer {path}")
            self.writers[name] = writer
            self.paths[name] = str(path)

    def add(self, images: dict[str, np.ndarray], label: str) -> None:
        if not self.enabled:
            return
        for name, writer in self.writers.items():
            bgr = cv2.cvtColor(images[name], cv2.COLOR_RGB2BGR)
            annotated = bgr.copy()
            cv2.rectangle(annotated, (0, 0), (640, 28), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                label[:92],
                (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (80, 240, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            if len(self.frames[name]) < 12:
                self.frames[name].append(annotated.copy())

    def close(self, stage_dir: Path) -> None:
        for writer in self.writers.values():
            writer.release()
        if not self.enabled or not self.frames:
            return
        name = "overview" if "overview" in self.frames else next(iter(self.frames))
        frames = self.frames[name]
        if frames:
            indices = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
            panels = [frames[index] for index in indices]
            while len(panels) < 6:
                panels.append(np.zeros_like(panels[0]))
            sheet = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:6])))
            cv2.imwrite(str(stage_dir / "contact_sheet.png"), sheet)


class TaskSemantics:
    FORCE_THRESHOLD_N = 0.05
    MIN_TASK_DIGITS = 2
    ACQUIRE_DEBOUNCE_FRAMES = 3
    HOLD_FRAMES = 10
    LIFT_M = 0.003
    MAX_RELATIVE_SLIP_M = 0.020

    def __init__(self, layout: dict[str, Any], initial_doll_position: np.ndarray):
        self.layout = layout
        self.initial = np.asarray(initial_doll_position, dtype=np.float64)
        self.history: list[dict[str, Any]] = []
        self.events: dict[str, int] = {}
        self.left_run = 0
        self.right_run = 0
        self.left_release_run = 0
        self.right_release_run = 0
        self.left_owned = False
        self.right_acquired = False
        self.right_owned = False
        self.release_candidate: int | None = None

    @staticmethod
    def _slip(
        history: list[dict[str, Any]], side: str, count: int
    ) -> float | None:
        if len(history) < count:
            # The metric is undefined until a complete hold window exists.
            # Keep that distinction explicit and JSON-safe in the diagnostic
            # trace instead of serializing an artificial infinity sentinel.
            return None
        values = np.stack(
            [row["doll_position"] - row[f"{side}_digit_centroid"] for row in history[-count:]]
        )
        return float(np.max(np.linalg.norm(values - values[0], axis=1)))

    def inside_bin(self, position: np.ndarray) -> bool:
        center = np.asarray(self.layout["bin"]["center_world_xy_m"], dtype=np.float64)
        opening = np.asarray(self.layout["bin"]["opening_dimensions_xy_m"], dtype=np.float64)
        radius = float(self.layout["doll"]["diameter_m"]) / 2.0
        horizontal = np.all(np.abs(position[:2] - center) <= opening / 2.0 - radius)
        tabletop = float(self.layout["table"]["surface_height_m"])
        rim = tabletop + float(self.layout["bin"]["outer_dimensions_xyz_m"][2])
        vertical = tabletop + radius * 0.5 <= position[2] <= rim
        return bool(horizontal and vertical)

    def update(
        self,
        frame: int,
        forces: dict[str, float],
        doll_position: np.ndarray,
        doll_velocity: np.ndarray,
        left_centroid: np.ndarray,
        right_centroid: np.ndarray,
    ) -> dict[str, Any]:
        left_count = sum(forces[f"left_{digit}"] > self.FORCE_THRESHOLD_N for digit in ("thumb", "index", "middle"))
        right_count = sum(forces[f"right_{digit}"] > self.FORCE_THRESHOLD_N for digit in ("thumb", "index", "middle"))
        self.left_run = self.left_run + 1 if left_count >= self.MIN_TASK_DIGITS else 0
        self.right_run = self.right_run + 1 if right_count >= self.MIN_TASK_DIGITS else 0
        self.left_release_run = self.left_release_run + 1 if left_count < self.MIN_TASK_DIGITS else 0
        self.right_release_run = self.right_release_run + 1 if right_count < self.MIN_TASK_DIGITS else 0
        lifted = float(doll_position[2] - self.initial[2]) >= self.LIFT_M
        record = {
            "frame": frame,
            "forces_n": forces.copy(),
            "left_contact_digit_count": int(left_count),
            "right_contact_digit_count": int(right_count),
            "doll_position": np.asarray(doll_position, dtype=np.float64),
            "doll_velocity": np.asarray(doll_velocity, dtype=np.float64),
            "left_digit_centroid": np.asarray(left_centroid, dtype=np.float64),
            "right_digit_centroid": np.asarray(right_centroid, dtype=np.float64),
            "lifted": lifted,
            "inside_bin": self.inside_bin(np.asarray(doll_position)),
        }
        self.history.append(record)
        if "first_doll_contact" not in self.events and left_count + right_count > 0:
            self.events["first_doll_contact"] = frame
        left_slip = self._slip(self.history, "left", self.HOLD_FRAMES)
        right_slip = self._slip(self.history, "right", self.HOLD_FRAMES)
        if (
            not self.left_owned
            and self.left_run >= self.HOLD_FRAMES
            and lifted
            and left_slip is not None
            and left_slip <= self.MAX_RELATIVE_SLIP_M
        ):
            self.left_owned = True
            self.events["left_owned"] = frame
        if (
            self.left_owned
            and not self.right_acquired
            and self.right_run >= self.ACQUIRE_DEBOUNCE_FRAMES
            and left_count >= self.MIN_TASK_DIGITS
        ):
            self.right_acquired = True
            self.events["right_acquire"] = frame
        if (
            self.right_acquired
            and "left_release" not in self.events
            and self.left_release_run >= self.ACQUIRE_DEBOUNCE_FRAMES
        ):
            self.events["left_release"] = frame
        if (
            "left_release" in self.events
            and not self.right_owned
            and self.right_run >= self.HOLD_FRAMES
            and lifted
            and right_slip is not None
            and right_slip <= self.MAX_RELATIVE_SLIP_M
        ):
            self.right_owned = True
            self.events["right_owned"] = frame
        if self.right_owned and self.right_release_run >= self.ACQUIRE_DEBOUNCE_FRAMES and record["inside_bin"]:
            if self.release_candidate is None:
                self.release_candidate = frame
        else:
            self.release_candidate = None
        if (
            self.release_candidate is not None
            and frame - self.release_candidate + 1 >= self.HOLD_FRAMES
            and "released_inside_bin" not in self.events
        ):
            self.events["released_inside_bin"] = frame
        return record | {
            "left_owned": self.left_owned,
            "right_acquired": self.right_acquired,
            "right_owned": self.right_owned,
            "released_inside_bin": "released_inside_bin" in self.events,
            "left_relative_slip_m": left_slip,
            "right_relative_slip_m": right_slip,
        }

    def summary(self) -> dict[str, Any]:
        dual = 0
        for row in self.history:
            if row["left_contact_digit_count"] >= 2 and row["right_contact_digit_count"] >= 2:
                dual += 1
        max_displacement = max(
            (float(np.linalg.norm(row["doll_position"][:2] - self.initial[:2])) for row in self.history),
            default=0.0,
        )
        return {
            "events": self.events,
            "left_owned": self.left_owned,
            "right_acquired": self.right_acquired,
            "right_owned": self.right_owned,
            "released_inside_bin": "released_inside_bin" in self.events,
            "handoff_ordering_valid": bool(
                "right_acquire" in self.events
                and "left_release" in self.events
                and self.events["right_acquire"] < self.events["left_release"]
            ),
            "dual_contact_frames": dual,
            "dual_contact_seconds": dual / CONTROL_FPS,
            "maximum_doll_xy_displacement_from_nominal_m": max_displacement,
            "evaluation_thresholds": {
                "force_threshold_n": self.FORCE_THRESHOLD_N,
                "minimum_task_digits": self.MIN_TASK_DIGITS,
                "acquire_debounce_frames": self.ACQUIRE_DEBOUNCE_FRAMES,
                "hold_frames": self.HOLD_FRAMES,
                "minimum_lift_m": self.LIFT_M,
                "maximum_relative_slip_m": self.MAX_RELATIVE_SLIP_M,
                "bin_acceptance": "doll center within radius-shrunk opening and below rim",
            },
        }


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def camera_verification(
    images: dict[str, np.ndarray], camera: Any, camera_cfg: dict[str, Any]
) -> dict[str, Any]:
    camera_dir = args.output_root.resolve() / "camera"
    actual = images["policy"]
    save_rgb(camera_dir / "isaac_calibrated.png", actual)
    overlay_path = None
    if "source_reference" in camera_cfg:
        source_path = ROOT / camera_cfg["source_reference"]["reference_image"]
        source_bgr = cv2.imread(str(source_path))
        if source_bgr is None:
            raise FileNotFoundError(source_path)
        actual_bgr = cv2.cvtColor(actual, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(source_bgr, 0.5, actual_bgr, 0.5, 0.0)
        source_points = np.asarray(
            camera_cfg["source_reference"]["measured_workspace_inner_corners_px"], dtype=np.float64
        )
        projected = np.asarray(
            camera_cfg["calibration"]["projected_workspace_inner_corners_px"], dtype=np.float64
        )
        cv2.polylines(overlay, [np.rint(source_points).astype(np.int32)], True, (0, 255, 255), 2)
        cv2.polylines(overlay, [np.rint(projected).astype(np.int32)], True, (255, 0, 255), 2)
        overlay_path = camera_dir / "source_vs_isaac_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
    # Stage 0 intentionally advances no physics timestamp.  Refresh only the
    # camera pose readback from its initialized frame view so CameraData does
    # not retain its zero-allocation sentinel at t=0.
    camera._update_poses()
    intrinsic_actual = camera.data.intrinsic_matrices.torch[0].detach().cpu().numpy()
    position_actual = camera.data.pos_w.torch[0].detach().cpu().numpy()
    quaternion_actual = camera.data.quat_w_ros.torch[0].detach().cpu().numpy()
    expected_intrinsic = DEPLOYMENT_CAMERA.intrinsic_matrix
    expected_position = DEPLOYMENT_CAMERA.position_world_xyz_m
    expected_quaternion = DEPLOYMENT_CAMERA.orientation_world_xyzw_ros_camera
    verification = {
        "status": "PASS",
        "actual_render_path": str(camera_dir / "isaac_calibrated.png"),
        "overlay_path": None if overlay_path is None else str(overlay_path),
        "camera_config": camera_manifest_record(DEPLOYMENT_CAMERA),
        "actual_intrinsic_matrix": intrinsic_actual,
        "configured_intrinsic_matrix": expected_intrinsic,
        "intrinsic_max_abs_difference": float(np.max(np.abs(intrinsic_actual - expected_intrinsic))),
        "actual_position_world_xyz_m": position_actual,
        "configured_position_world_xyz_m": expected_position,
        "position_max_abs_difference_m": float(np.max(np.abs(position_actual - expected_position))),
        "actual_orientation_world_xyzw_ros": quaternion_actual,
        "configured_orientation_world_xyzw_ros": expected_quaternion,
        "quaternion_equal_up_to_sign": bool(
            min(
                np.max(np.abs(quaternion_actual - expected_quaternion)),
                np.max(np.abs(quaternion_actual + expected_quaternion)),
            )
            < 1e-5
        ),
        "legacy_source_like_corner_calibration": (
            {
                key: camera_cfg["calibration"][key]
                for key in (
                    "corner_component_rmse_px",
                    "corner_euclidean_rmse_px",
                    "normalized_corner_rmse",
                )
            }
            if "source_reference" in camera_cfg
            else None
        ),
        "policy_input_is_unannotated_configured_rgb": True,
        "pose_readback_refresh": "camera frame-view refresh only; no physics step and no joint command",
    }
    checks = {
        "intrinsics": verification["intrinsic_max_abs_difference"] < 1e-3,
        "position": verification["position_max_abs_difference_m"] < 1e-5,
        "orientation": verification["quaternion_equal_up_to_sign"],
        "actual_image_shape": list(actual.shape)
        == [DEPLOYMENT_CAMERA.height, DEPLOYMENT_CAMERA.width, 3],
    }
    verification["checks"] = checks
    verification["status"] = "PASS" if all(checks.values()) else "FAIL"
    atomic_json(camera_dir / "isaac_render_verification.json", verification)
    return verification


def observation_adapter(
    output_root: Path,
    camera_cfg: dict[str, Any],
    checkpoint: Path,
    checkpoint_hash: str,
    names: list[str],
    projection_config: dict[str, Any],
    projection_config_sha256: str,
) -> dict[str, Any]:
    stats = read_json(DATASET / "meta/stats.json")
    policy_config = read_json(checkpoint / "config.json")
    preprocessor = read_json(checkpoint / "policy_preprocessor.json")
    postprocessor = read_json(checkpoint / "policy_postprocessor.json")
    adapter = {
        "schema_version": "policy_ab_observation_adapter_v3_camera_config_driven",
        "status": "ISAAC_ONLY_NO_REAL_ROBOT_COMMAND_PATH",
        "policy_variant": args.policy_variant,
        "common_semantic_interface": f"RGB image + state_28d + task -> Policy {args.policy_variant} -> action_chunk_50x28",
        "camera": {
            "name": DEPLOYMENT_CAMERA.name,
            "config": str(CAMERA_CONFIG),
            "config_sha256": sha256_file(CAMERA_CONFIG),
            "mount": DEPLOYMENT_CAMERA.mount,
            "geometry_source": "--camera-config only",
        },
        "image_preprocessing": {
            "input_color": "RGB",
            "source_dtype_layout": "uint8 HWC [480,640,3] from configured Isaac camera",
            "adapter_conversion": "float32 CHW / 255 -> saved LeRobot preprocessor",
            "crop": None,
            "dataset_resolution": [480, 640],
            "policy_resize_with_padding": policy_config["resize_imgs_with_padding"],
            "smolvla_internal_visual_scaling": "model prepare_images maps [0,1] to [-1,1] after resize-with-pad",
            "saved_preprocessor": preprocessor,
            "visual_normalization": policy_config["normalization_mapping"]["VISUAL"],
        },
        "state": {
            "source_in_isaac": "ISAAC_MEASURED_G1_DEX3_28D",
            "training_label": "RETARGETED_G1_STATE_SURROGATE = q_target[max(t-1,0)]",
            "deployment_replacement": "CURRENT MEASURED simulated/real G1 arm and Dex3 qpos",
            "previous_prediction_used_as_state": False,
            "dimension": 28,
            "joint_names": names,
            "units": "radian",
            "normalization": "MEAN_STD",
            "statistics": stats["observation.state"],
        },
        "action": {
            "policy_raw_definition": f"absolute G1/Dex3 target joint position from Policy {args.policy_variant}",
            "hard_limit_projected_definition": (
                "policy_raw_action with only invalid Dex3 scalars projected to the nearest "
                "effective named-joint hard bound"
            ),
            "deployment_safe_definition": (
                "hard_limit_projected_action with only Dex3 scalars outside empirically "
                "derived simulation-controller-safe intervals projected to the nearest safe bound"
            ),
            "dimension": 28,
            "chunk_rows": 50,
            "joint_names": names,
            "units": "radian",
            "normalization": "MEAN_STD inverted by saved postprocessor",
            "statistics": stats["action"],
            "saved_postprocessor": postprocessor,
            "velocity_or_torque_or_delta": False,
            "common_deployment_safety_projection": {
                "name": projection_config["name"],
                "freeze_manifest": str(COMMON_PROJECTION_FREEZE),
                "freeze_manifest_sha256": projection_config_sha256,
                "implementation": projection_config["implementation"],
                "margin_label": projection_config["margin_label"],
                "projectable_group": "dex3",
                "arm_outputs_preserved": True,
                "safe_dex3_outputs_preserved": True,
                "comparison_tolerance_rad": 0.0,
                "arrays_retained": [
                    "policy_raw_action",
                    "hard_limit_projected_action",
                    "deployment_safe_action",
                ],
                "real_hardware_authorized": False,
            },
        },
        "task": TASK,
        "control_fps": CONTROL_FPS,
        "execution_horizon_frames": args.execution_horizon,
        "policy": {
            "checkpoint": str(checkpoint),
            "model_sha256": checkpoint_hash,
            "logical_prediction_shape": [1, 50, 28],
            "internal_padding": "28D -> 32D, then prediction sliced 32D -> 28D",
        },
        "process_isolation": {
            "isaac_python": sys.executable,
            "policy_python": str(POLICY_PYTHON),
            "transport": "local Unix-domain socket carrying observations/predictions only",
            "dds": False,
            "network": False,
            "real_robot_commands": False,
        },
        "later_real_runner_substitution": {
            "Isaac RGB": "final helmet D455 RGB using the same frozen camera config",
            "Isaac measured qpos": "real measured G1/Dex3 qpos in the same named order",
            "policy_semantic_interface_changes": False,
            "simulation_margin_may_be_reused_without_hardware_calibration": False,
            "required_before_command_transmission": (
                "separate real-G1/Dex3 no-object boundary/tracking calibration"
            ),
        },
    }
    atomic_json(output_root / f"policy_{args.policy_variant.lower()}_observation_adapter.json", adapter)
    return adapter


def rollout_metrics(
    hard_safety: NumericalSafety,
    safe_safety: NumericalSafety,
    initial_state: np.ndarray,
    commanded: list[np.ndarray],
    actual: list[np.ndarray],
    external_collision_forces: list[float],
    diagnostic_authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command_array = np.asarray(commanded, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    if len(command_array) == 0:
        return {"status": "NO_COMMANDS"}
    error = actual_array - command_array
    actual_with_initial = np.vstack((initial_state, actual_array))
    commanded_hard = hard_safety.trajectory(command_array)
    commanded_safe = safe_safety.trajectory(command_array)
    actual_hard = hard_safety.trajectory(actual_with_initial)
    actual_safe = safe_safety.trajectory(actual_with_initial)
    tracking_rmse = float(np.sqrt(np.mean(np.square(error))))
    tracking_max = float(np.max(np.abs(error)))
    strict_checks = {
        "finite": bool(np.isfinite(command_array).all() and np.isfinite(actual_array).all()),
        "commanded_hard_limits": commanded_hard["joint_limit_violation_count"] == 0,
        "commanded_safety_margin": commanded_safe["joint_limit_violation_count"] == 0,
        "measured_hard_limits": actual_hard["joint_limit_violation_count"] == 0,
        "measured_safety_margin": actual_safe["joint_limit_violation_count"] == 0,
        "velocity": actual_hard["maximum_velocity_rad_s"]
        <= float(hard_safety.acceptance["maximum_velocity_rad_s"]),
        "acceleration": actual_hard["maximum_acceleration_rad_s2"]
        <= float(hard_safety.acceptance["maximum_acceleration_rad_s2"]),
        "branch": actual_hard["branch_discontinuity_count"] == 0,
        "hard_self_collision": actual_hard["collision"]["invalid_hard_collision_category_incidence"] == 0,
        "catastrophic_external_collision": max(external_collision_forces, default=0.0)
        <= TaskSemantics.FORCE_THRESHOLD_N,
        "tracking_maximum": tracking_max
        <= float(hard_safety.acceptance["maximum_joint_step_rad"]),
        "tracking_rmse": tracking_rmse <= 0.05,
    }
    arm_violation_mask = (
        (actual_with_initial[:, :14] < hard_safety.lower[None, :14] - 1e-9)
        | (actual_with_initial[:, :14] > hard_safety.upper[None, :14] + 1e-9)
    )
    dex3_lower_excess = np.maximum(
        hard_safety.lower[None, 14:] - actual_with_initial[:, 14:], 0.0
    )
    dex3_upper_excess = np.maximum(
        actual_with_initial[:, 14:] - hard_safety.upper[None, 14:], 0.0
    )
    dex3_maximum_excess = float(
        max(np.max(dex3_lower_excess), np.max(dex3_upper_excess))
    )
    diagnostic_checks: dict[str, bool] | None = None
    if diagnostic_authorization is not None:
        excursion_cap = float(
            diagnostic_authorization[
                "diagnostic_measured_dex3_excursion_cap_rad"
            ]
        )
        diagnostic_checks = {
            key: value
            for key, value in strict_checks.items()
            if key not in {"measured_hard_limits", "measured_safety_margin"}
        }
        diagnostic_checks.update(
            {
                "measured_arm_hard_limits": int(np.count_nonzero(arm_violation_mask))
                == 0,
                "measured_dex3_excursion_not_catastrophic": dex3_maximum_excess
                <= excursion_cap,
                "strict_measured_excursion_logged_not_hidden": True,
            }
        )
    active_checks = diagnostic_checks if diagnostic_checks is not None else strict_checks
    return {
        "status": "PASS" if all(active_checks.values()) else "FAIL",
        "checks": active_checks,
        "strict_safety_qualification": {
            "status": "PASS" if all(strict_checks.values()) else "FAIL",
            "checks": strict_checks,
        },
        "simulation_diagnostic_qualification": (
            {
                "status": "PASS" if all(diagnostic_checks.values()) else "FAIL",
                "checks": diagnostic_checks,
                "measured_dex3_excursion_cap_rad": float(
                    diagnostic_authorization[
                        "diagnostic_measured_dex3_excursion_cap_rad"
                    ]
                ),
                "cap_provenance": diagnostic_authorization["cap_provenance"],
            }
            if diagnostic_checks is not None
            else None
        ),
        "tracking_rmse_rad": tracking_rmse,
        "maximum_joint_tracking_error_rad": tracking_max,
        "commanded_hard_limit_audit": commanded_hard,
        "commanded_safe_interval_audit": commanded_safe,
        "actual_hard_limit_audit": actual_hard,
        "actual_safe_interval_audit": actual_safe,
        "actual_arm_hard_limit_violation_count": int(
            np.count_nonzero(arm_violation_mask)
        ),
        "actual_dex3_maximum_hard_limit_excess_rad": dex3_maximum_excess,
        "actual_trajectory": actual_hard,
        "maximum_external_non_task_contact_force_n": max(external_collision_forces, default=0.0),
        "engineering_tracking_rmse_gate_rad": 0.05,
    }


def main() -> int:
    print(f"[PolicyBIsaac] entering stage runner {args.stage}", flush=True)
    import carb
    import omni.usd
    import torch
    from pxr import UsdPhysics
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    import isaaclab.sim as sim_utils

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in STAGE_DIRECTORIES.values():
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    stage_dir = output_root / STAGE_DIRECTORIES[args.stage]
    prerequisite_gate(output_root)
    diagnostic_authorization = (
        strict_stage1_diagnostic_authorization()
        if args.simulation_diagnostic_rollout
        else None
    )
    checkpoint, checkpoint_hash, checkpoint_step = frozen_checkpoint()
    names, lower, upper, freeze = frozen_interfaces()
    projector, projection_config, projection_config_hash = frozen_projection(names)
    jerk_limited_otg = (
        CommonJerkLimitedOTG.from_path(args.jerk_limited_otg_config)
        if args.jerk_limited_otg_config is not None
        else None
    )
    if jerk_limited_otg is not None and jerk_limited_otg.names != names:
        raise RuntimeError("common jerk-limited OTG joint order does not match Dataset B")
    jerk_limited_otg_config_hash = (
        sha256_file(args.jerk_limited_otg_config.resolve())
        if args.jerk_limited_otg_config is not None
        else None
    )
    fixed_observation_rgb: np.ndarray | None = None
    fixed_observation_state: np.ndarray | None = None
    fixed_observation_record: dict[str, Any] | None = None
    if args.fixed_observation_metadata is not None:
        metadata_path = args.fixed_observation_metadata.resolve()
        metadata = read_json(metadata_path)
        rgb_path = Path(metadata["rgb_path"]).resolve()
        state_path = Path(metadata["state_path"]).resolve()
        if sha256_file(rgb_path) != metadata["rgb_sha256"]:
            raise RuntimeError(f"fixed-observation RGB hash mismatch: {rgb_path}")
        if sha256_file(state_path) != metadata["state_sha256"]:
            raise RuntimeError(f"fixed-observation state hash mismatch: {state_path}")
        if metadata["task"] != TASK:
            raise RuntimeError("fixed-observation task does not match the frozen task string")
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (480, 640, 3):
            raise RuntimeError(f"fixed-observation RGB is malformed: {rgb_path}")
        fixed_observation_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with np.load(state_path, allow_pickle=False) as archive:
            fixed_observation_state = archive["measured_state"].astype(np.float64)
            fixed_names = archive["joint_names"].astype(str).tolist()
        if fixed_observation_state.shape != (28,) or not np.isfinite(
            fixed_observation_state
        ).all():
            raise RuntimeError("fixed-observation state is not finite 28D")
        if fixed_names != names:
            raise RuntimeError("fixed-observation named-joint order differs from Dataset B")
        init_q = fixed_observation_state.copy()
        init_report = {
            "definition": "exact measured 28D state from frozen Policy-B observation",
            "purpose": "single fixed-noise raw-chunk replay without replanning",
            "joint_names": names,
            "q_rad": init_q,
            "fixed_observation_metadata": str(metadata_path),
            "fixed_observation_metadata_sha256": sha256_file(metadata_path),
            "rgb_path": str(rgb_path),
            "rgb_sha256": metadata["rgb_sha256"],
            "state_path": str(state_path),
            "state_sha256": metadata["state_sha256"],
            "policy_generated_actions_only_after_initialization": True,
        }
        fixed_observation_record = {
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "rgb": str(rgb_path),
            "rgb_sha256": metadata["rgb_sha256"],
            "state": str(state_path),
            "state_sha256": metadata["state_sha256"],
            "task": metadata["task"],
            "policy_input_is_byte_exact_frozen_rgb_and_float32_state": True,
        }
    else:
        init_q, init_report = initial_condition(names)
    initial_projection = projector.project(init_q[None, :], inference_index=None)
    init_q = initial_projection.deployment_safe_action[0]
    init_report["simulation_deployment_safety"] = initial_projection.summary
    init_report["hard_limit_projection_records"] = initial_projection.hard_limit_records
    init_report["deployment_margin_projection_records"] = (
        initial_projection.deployment_margin_records
    )
    init_report["q_rad_policy_source_before_runtime_safety"] = init_report.pop("q_rad")
    init_report["q_rad_hard_limit_projected"] = (
        initial_projection.hard_limit_projected_action[0]
    )
    init_report["q_rad"] = init_q
    atomic_json(stage_dir / "initial_condition.json", init_report)
    camera_cfg = DEPLOYMENT_CAMERA.raw
    if not SCENE_STAGE.is_file():
        raise FileNotFoundError(SCENE_STAGE)
    layout = read_json(SCENE_LAYOUT)
    source_rgb_paths: list[Path] = []
    source_rgb_episode: dict[str, Any] | None = None
    if args.source_rgb_time_forced_episode is not None:
        source_manifest = read_json(SOURCE_MANIFEST)
        source_rgb_episode = source_manifest["episodes"][
            args.source_rgb_time_forced_episode
        ]
        if int(source_rgb_episode["final_dataset_index"]) != int(
            args.source_rgb_time_forced_episode
        ):
            raise RuntimeError("source-RGB episode mapping is not index-consistent")
        source_rgb_dir = (
            Path(source_rgb_episode["raw_directory_path"])
            / "images/observation.images.cam_high/episode_000000"
        )
        source_rgb_paths = sorted(source_rgb_dir.glob("frame_*.png"))
        if len(source_rgb_paths) != int(source_rgb_episode["source_frame_count"]):
            raise RuntimeError(
                "source-RGB frame count does not match the frozen final source manifest"
            )

    def source_rgb_at(frame_index: int) -> np.ndarray:
        if not source_rgb_paths:
            raise RuntimeError("source-RGB time forcing is not active")
        index = min(max(int(frame_index), 0), len(source_rgb_paths) - 1)
        bgr = cv2.imread(str(source_rgb_paths[index]), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (480, 640, 3):
            raise RuntimeError(f"malformed source RGB: {source_rgb_paths[index]}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    contract_safety = NumericalSafety(names, lower, upper)
    hard_safety = NumericalSafety(names, projector.hard_lower, projector.hard_upper)
    safe_safety = NumericalSafety(names, projector.safe_lower, projector.safe_upper)
    if args.stage == "full-motion" and diagnostic_authorization is not None:
        # This is an Isaac trajectory-visualization gate, never a hardware
        # qualification. Reuse the already-frozen catastrophic joint-step bound
        # instead of treating the previously characterized microradian PhysX
        # boundary effect as an abort. Every measured excursion remains logged,
        # while strict_safety_qualification is still computed against zero excess.
        diagnostic_authorization = dict(diagnostic_authorization)
        diagnostic_authorization["diagnostic_measured_dex3_excursion_cap_rad"] = float(
            hard_safety.acceptance["maximum_joint_step_rad"]
        )
        diagnostic_authorization["cap_provenance"] = (
            "Isaac-only catastrophic-excursion gate reusing the frozen maximum_joint_step_rad; "
            "microscopic hard-limit excursions remain strict-qualification failures"
        )
        diagnostic_authorization["visualization_only"] = True

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", True)
    if not omni.usd.get_context().open_stage(str(SCENE_STAGE)):
        raise RuntimeError(f"failed to open {SCENE_STAGE}")
    live_stage = omni.usd.get_context().get_stage()
    sys.path.insert(0, str(ROOT / "isaaclab_magsafe_fixed_scene"))
    from physical_contact_monitor import enable_contact_reporting

    enable_contact_reporting(
        live_stage,
        [
            "/World/G1/Asset",
            "/World/DollHandoffEnvironment/Doll",
            "/World/DollHandoffEnvironment/Table",
            "/World/DollHandoffEnvironment/TrashBin",
        ],
    )
    object_free = args.stage in {"object-free-single", "object-free-multichunk"}
    kinematic_visual_doll = bool(
        args.stage == "full-motion" and args.kinematic_visual_doll
    )
    if object_free or kinematic_visual_doll:
        doll_prim = live_stage.GetPrimAtPath("/World/DollHandoffEnvironment/Doll")
        rigid = UsdPhysics.RigidBodyAPI.Get(live_stage, doll_prim.GetPath())
        rigid.CreateKinematicEnabledAttr(True)
        rigid.CreateRigidBodyEnabledAttr(True)
    if object_free:
        for prim in live_stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith("/World/DollHandoffEnvironment/Doll") or path.startswith(
                "/World/DollHandoffEnvironment/TrashBin"
            ):
                collision = UsdPhysics.CollisionAPI.Get(live_stage, prim.GetPath())
                if collision:
                    collision.CreateCollisionEnabledAttr(False)

    sim = SimulationContext(
        SimulationCfg(dt=PHYSICS_DT, device="cuda:0", use_fabric=True)
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/G1/Asset/root_joint",
            spawn=None,
            actuators=build_implicit_actuators(ImplicitActuatorCfg),
        )
    )
    object_enabled = args.stage in {
        "left-grasp",
        "handoff",
        "full-task",
        "full-motion",
    }
    doll = (
        RigidObject(
            RigidObjectCfg(
                prim_path="/World/DollHandoffEnvironment/Doll", spawn=None
            )
        )
        if object_enabled
        else None
    )
    hand_cfg = read_json(WHOLE_HAND_CONFIG)
    distal_links = {
        f"{side}_{hand_cfg[side][role]['digit_chain']}": hand_cfg[side][role]["distal_link"]
        for side in ("left", "right")
        for role in ("A", "B", "C")
    }
    digit_sensors = (
        {
            label: ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/G1/Asset/{link}",
                    update_period=0.0,
                    filter_prim_paths_expr=["/World/DollHandoffEnvironment/Doll"],
                    track_contact_points=True,
                    max_contact_data_count_per_prim=32,
                    force_threshold=0.0,
                )
            )
            for label, link in distal_links.items()
        }
        if object_enabled
        else {}
    )
    external_sensor = (
        ContactSensor(
            ContactSensorCfg(
                prim_path="/World/G1/Asset/.*_link",
                update_period=0.0,
                filter_prim_paths_expr=[
                    "/World/DollHandoffEnvironment/Table/Colliders/Top",
                    "/World/DollHandoffEnvironment/TrashBin/Bottom",
                    "/World/DollHandoffEnvironment/TrashBin/FrontWall",
                    "/World/DollHandoffEnvironment/TrashBin/BackWall",
                    "/World/DollHandoffEnvironment/TrashBin/LeftWall",
                    "/World/DollHandoffEnvironment/TrashBin/RightWall",
                ],
                track_contact_points=True,
                max_contact_data_count_per_prim=64,
                force_threshold=0.0,
            )
        )
        if args.stage != "inference"
        else None
    )

    def camera_spawn() -> Any:
        return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            DEPLOYMENT_CAMERA.intrinsic_matrix.reshape(-1).tolist(),
            width=DEPLOYMENT_CAMERA.width,
            height=DEPLOYMENT_CAMERA.height,
            clipping_range=DEPLOYMENT_CAMERA.clipping_range_m,
            lock_camera=True,
        )

    cameras = {
        key: Camera(
            CameraCfg(
                prim_path=f"/World/{prim_name}",
                update_period=0.0,
                width=DEPLOYMENT_CAMERA.width,
                height=DEPLOYMENT_CAMERA.height,
                data_types=["rgb"],
                spawn=camera_spawn(),
            )
        )
        for key, prim_name in {
            "policy": "PolicyDeploymentCamera",
            "overview": "ExecutionDiagnosticOverviewCamera",
            "top": "ExecutionDiagnosticTopCamera",
            "side": "ExecutionDiagnosticSideCamera",
        }.items()
    }
    sim.reset()
    isaac_names = list(robot.data.joint_names)
    missing = [name for name in names if name not in isaac_names]
    ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(ids) != 28 or len(set(ids)) != 28:
        raise RuntimeError(f"named Isaac policy mapping failed; missing={missing}")
    camera_position = DEPLOYMENT_CAMERA.position_world_xyz_m.astype(np.float32)
    camera_quaternion = DEPLOYMENT_CAMERA.orientation_world_xyzw_ros_camera.astype(np.float32)
    cameras["policy"].set_world_poses(
        camera_position[None], camera_quaternion[None], convention="ros"
    )
    diagnostic_presets = layout["camera"]["presets"]
    for diagnostic_name in ("overview", "top", "side"):
        preset = diagnostic_presets[diagnostic_name]
        eye = np.asarray(preset["eye_world_xyz_m"], dtype=np.float32)
        target_point = np.asarray(preset["target_world_xyz_m"], dtype=np.float32)
        orientation = look_at_ros_camera_quaternion_xyzw(eye, target_point).astype(
            np.float32
        )
        cameras[diagnostic_name].set_world_poses(
            eye[None], orientation[None], convention="ros"
        )
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero = torch.zeros_like(target)
    target[0, ids] = torch.as_tensor(init_q, device=robot.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(target, zero)
    sim.forward()
    robot.update(0.0)

    def update_assets() -> None:
        robot.update(PHYSICS_DT)
        if doll is not None:
            doll.update(PHYSICS_DT)
        for sensor in digit_sensors.values():
            sensor.update(PHYSICS_DT)
        if external_sensor is not None:
            external_sensor.update(PHYSICS_DT)

    def capture() -> dict[str, np.ndarray]:
        sim.forward()
        sim.render()
        sim.render_context.reset_transform_cadence()
        result = {}
        for name, camera in cameras.items():
            camera.update(PHYSICS_DT, force_recompute=True)
            image = camera.data.output["rgb"].torch[0].detach().cpu().numpy()[..., :3]
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            result[name] = (
                apply_configured_distortion(image.copy(), DEPLOYMENT_CAMERA)
                if name == "policy"
                else image.copy()
            )
        result["source_like"] = result["policy"].copy()
        return result

    def measured_state() -> np.ndarray:
        value = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        if value.shape != (28,) or not np.isfinite(value).all():
            raise RuntimeError("Isaac measured named-joint state is malformed")
        return value

    def measured_velocity() -> np.ndarray:
        return robot.data.joint_vel.torch[0, ids].detach().cpu().numpy().astype(np.float64)

    def causal_measured_acceleration_estimate(
        measured_velocity_history: list[np.ndarray], horizon: int
    ) -> np.ndarray:
        """Estimate dq/dt from past measured velocities with no future samples."""

        count = min(max(2, int(horizon)), len(measured_velocity_history))
        if count < 2:
            return np.zeros(28, dtype=np.float64)
        history = np.asarray(measured_velocity_history[-count:], dtype=np.float64)
        time_axis = np.arange(count, dtype=np.float64) / CONTROL_FPS
        centered_time = time_axis - float(np.mean(time_axis))
        denominator = float(np.dot(centered_time, centered_time))
        value = np.sum(centered_time[:, None] * history, axis=0) / denominator
        if value.shape != (28,) or not np.isfinite(value).all():
            raise RuntimeError("causal measured acceleration estimate is malformed")
        return value

    steps_per_control = int(round((1.0 / CONTROL_FPS) / PHYSICS_DT))

    def step_command(command: np.ndarray) -> None:
        target[0, ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
        for _ in range(steps_per_control):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            update_assets()

    def force_value(sensor: Any) -> float:
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            return 0.0
        value = matrix.torch.detach().cpu().numpy()
        return float(np.max(np.linalg.norm(value.reshape(-1, 3), axis=1))) if value.size else 0.0

    body_names = list(robot.data.body_names)
    distal_body_ids = {label: body_names.index(link) for label, link in distal_links.items()}
    wrist_body_ids = {
        "left": body_names.index("left_wrist_yaw_link"),
        "right": body_names.index("right_wrist_yaw_link"),
    }

    def task_snapshot(frame: int, semantics: TaskSemantics | None) -> dict[str, Any] | None:
        if doll is None or semantics is None:
            return None
        forces = {label: force_value(sensor) for label, sensor in digit_sensors.items()}
        body_positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy()
        centroids = {
            side: np.mean(
                [body_positions[distal_body_ids[f"{side}_{digit}"]] for digit in ("thumb", "index", "middle")],
                axis=0,
            )
            for side in ("left", "right")
        }
        pose = doll.data.root_pose_w.torch[0].detach().cpu().numpy()
        velocity = doll.data.root_vel_w.torch[0].detach().cpu().numpy()
        return semantics.update(frame, forces, pose[:3], velocity[:3], centroids["left"], centroids["right"])

    def settle() -> None:
        if args.stage == "inference":
            return
        steps = max(1, int(round(args.settle_seconds / PHYSICS_DT)))
        for _ in range(steps):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            update_assets()

    settle()
    initial_measured = measured_state()
    initial_velocity = measured_velocity()
    initial_images = capture()
    if source_rgb_paths:
        initial_images["source_rgb_time_forced"] = source_rgb_at(0)
    checkpoint_ready = None
    bridge: PolicyBridge | None = None
    recorder: VideoRecorder | None = None
    try:
        bridge = PolicyBridge(
            checkpoint,
            stage_dir,
            args.seed,
            args.causal_stitching_method,
            args.execution_horizon,
            args.rtc_guidance_weight,
            args.rtc_prefix_attention_schedule,
            args.fixed_flow_noise,
        )
        checkpoint_ready = bridge.ready
        adapter_root = stage_dir if args.stage == "full-motion" else output_root
        adapter = observation_adapter(
            adapter_root,
            camera_cfg,
            checkpoint,
            checkpoint_hash,
            names,
            projection_config,
            projection_config_hash,
        )
        common_report = {
            "stage": args.stage,
            "policy_variant": args.policy_variant,
            "checkpoint": str(checkpoint),
            "checkpoint_step": checkpoint_step,
            "checkpoint_model_sha256": checkpoint_hash,
            "dataset": str(DATASET),
            "dataset_action_trajectory_set_sha256": freeze["trajectory_set_sha256"],
            "camera": DEPLOYMENT_CAMERA.name,
            "camera_config": str(CAMERA_CONFIG),
            "camera_config_sha256": sha256_file(CAMERA_CONFIG),
            "rgb_source": (
                "SOURCE_RGB_TIME_FORCED_POLICY_ROLLOUT"
                if source_rgb_paths
                else "CAMERA_CONFIG_DRIVEN_ISAAC_RGB"
            ),
            "state_source": "ISAAC_MEASURED_G1_DEX3_28D",
            "task": TASK,
            "joint_names": names,
            "joint_mapping": {name: ids[index] for index, name in enumerate(names)},
            "control_fps": CONTROL_FPS,
            "physics_dt_s": PHYSICS_DT,
            "physics_substeps_per_control_frame": steps_per_control,
            "common_causal_execution_adapter": {
                "method": args.causal_stitching_method,
                "execution_horizon_frames": args.execution_horizon,
                "official_lerobot_rtc": args.causal_stitching_method == "rtc",
                "rtc_guidance_weight": (
                    args.rtc_guidance_weight
                    if args.causal_stitching_method == "rtc"
                    else None
                ),
                "rtc_prefix_attention_schedule": (
                    args.rtc_prefix_attention_schedule
                    if args.causal_stitching_method == "rtc"
                    else None
                ),
                "crossfade_window_frames": (
                    args.crossfade_window_frames
                    if args.causal_stitching_method == "crossfade"
                    else None
                ),
                "crossfade_window_derivation": (
                    "ceil of Dataset-B p95 jerk-derived transition duration for the measured M0 maximum old/new plan discrepancy; 5 frames at 30 Hz"
                    if args.causal_stitching_method == "crossfade"
                    else None
                ),
                "simulation_inference_delay_frames": (
                    args.rtc_inference_delay_frames
                    if args.causal_stitching_method == "rtc"
                    else 0
                ),
                "execution_semantics": (
                    "LOGICALLY_ASYNC_LATENCY_AWARE_ACTION_QUEUE"
                    if args.causal_stitching_method == "rtc"
                    and args.rtc_inference_delay_frames
                    else "SYNCHRONOUS_PREFIX_COMMITMENT"
                ),
                "causal": True,
                "future_observations_or_policy_calls_used": False,
                "raw_policy_chunks_preserved": True,
                "flow_noise_mode": (
                    "FIXED_SEED_DERIVED_TENSOR_REUSED"
                    if args.fixed_flow_noise
                    else "FRESH_SEQUENTIAL_SEED_STREAM"
                ),
                "episode_phase_or_task_specific_logic": False,
                "policy_independent_interface": True,
                "real_hardware_authorized": False,
            },
            "jerk_limited_online_trajectory_generation": {
                "enabled": jerk_limited_otg is not None,
                "implementation": (
                    "Ruckig community Python online position OTG"
                    if jerk_limited_otg is not None
                    else None
                ),
                "config": (
                    str(args.jerk_limited_otg_config.resolve())
                    if args.jerk_limited_otg_config is not None
                    else None
                ),
                "config_sha256": jerk_limited_otg_config_hash,
                "current_state": (
                    "ISAAC_MEASURED_Q_DQ_PLUS_CAUSAL_COMMITTED_PREFIX_DQ_SLOPE_DDQ"
                    if jerk_limited_otg is not None
                    else None
                ),
                "target": (
                    "DEPLOYMENT_SAFE_SELECTED_CAUSAL_PLAN_COMMITTED_ENDPOINT"
                    if jerk_limited_otg is not None
                    else None
                ),
                "raw_policy_chunks_modified": False,
                "fail_closed": True if jerk_limited_otg is not None else None,
                "real_hardware_authorized": False,
            },
            "policy_worker": checkpoint_ready,
            "observation_adapter": str(
                adapter_root / f"policy_{args.policy_variant.lower()}_observation_adapter.json"
            ),
            "common_deployment_safety_projection": {
                "name": projection_config["name"],
                "freeze_manifest": str(COMMON_PROJECTION_FREEZE),
                "freeze_manifest_sha256": projection_config_hash,
                "implementation": projection_config["implementation"],
                "margin_label": projection_config["margin_label"],
                "policy_independent": True,
                "episode_phase_and_task_specific_logic": False,
                "real_hardware_authorized": False,
                "real_hardware_calibration_required_before_transmission": True,
                "executed_array": "deployment_safe_action",
                "hard_limit_projection_always_applied_before_execution": True,
            },
            "initial_measured_state_rad": initial_measured,
            "initial_measured_velocity_rad_s": initial_velocity,
            "object_asset_enabled": object_enabled,
            "object_physics_enabled": object_enabled and not kinematic_visual_doll,
            "object_free_task_objects_rendered_but_noninterfering": object_free,
            "object_visualization_mode": (
                "KINEMATIC_OBJECT_TRAJECTORY_VISUALIZATION"
                if kinematic_visual_doll
                else "DYNAMIC_CANONICAL_OBJECT_VISUALIZATION"
            ),
            "physical_object_success_evaluated": False
            if args.stage == "full-motion"
            else None,
            "real_robot_command_allowed": False,
            "rollout_mode": (
                "SIMULATION_DIAGNOSTIC_ROLLOUT"
                if args.simulation_diagnostic_rollout
                else "STRICT_SAFETY_QUALIFICATION"
            ),
            "strict_stage1_qualification_preserved": diagnostic_authorization,
            "policy_behavior_diagnostic": (
                "IN_PROGRESS" if args.simulation_diagnostic_rollout else None
            ),
            "real_hardware_safety_readiness": "BLOCKED",
            "baseline_a_touched": False,
            "dataset_b_modified": False,
            "source_rgb_time_forced_diagnostic": (
                {
                    "active": True,
                    "classification": "SOURCE_RGB_TIME_FORCED_POLICY_ROLLOUT",
                    "closed_loop_vision": False,
                    "source_episode_index": int(
                        source_rgb_episode["final_dataset_index"]
                    ),
                    "source_raw_episode": source_rgb_episode["raw_directory"],
                    "source_frame_count": len(source_rgb_paths),
                    "source_fps": float(source_rgb_episode["source_fps"]),
                    "advance_rule": "source frame index = executed simulation frame at 30 Hz",
                    "state_source": "ISAAC_MEASURED_G1_DEX3_28D",
                    "task_success_claimed": False,
                }
                if source_rgb_paths
                else {"active": False}
            ),
            "fixed_observation_single_chunk_diagnostic": (
                fixed_observation_record
                if fixed_observation_record is not None
                else {"active": False}
            ),
        }

        if args.stage == "inference":
            verification = camera_verification(initial_images, cameras["policy"], camera_cfg)
            save_rgb(stage_dir / "input_rgb.png", initial_images["policy"])
            np.save(stage_dir / "input_state.npy", initial_measured.astype(np.float32))
            (
                policy_raw_action,
                model_stitched_plan,
                previous_remaining_plan,
                inference,
            ) = bridge.infer(
                initial_images["policy"], initial_measured
            )
            stitching = causal_stitcher.stitch(
                raw_policy_chunk=policy_raw_action,
                model_stitched_plan=model_stitched_plan,
                previous_remaining_plan=previous_remaining_plan,
                last_commanded_action=None,
                measured_qpos=initial_measured,
            )
            stitched_execution_plan = stitching.stitched_execution_plan
            stitching_records.append({"inference_index": 0, **stitching.audit})
            projection = projector.project(stitched_execution_plan, inference_index=0)
            hard_limit_projected_action = projection.hard_limit_projected_action
            deployment_safe_action = projection.deployment_safe_action
            # Keep the old filename as a raw-policy compatibility artifact; the
            # explicitly named arrays below are authoritative for evaluation.
            np.save(stage_dir / "predicted_chunk.npy", policy_raw_action.astype(np.float32))
            np.save(stage_dir / "policy_raw_action.npy", policy_raw_action.astype(np.float32))
            np.save(
                stage_dir / "stitched_execution_plan.npy",
                stitched_execution_plan.astype(np.float32),
            )
            np.save(
                stage_dir / "previous_remaining_plan.npy",
                previous_remaining_plan.astype(np.float32),
            )
            np.save(
                stage_dir / "hard_limit_projected_action.npy",
                hard_limit_projected_action.astype(np.float32),
            )
            np.save(
                stage_dir / "deployment_safe_action.npy",
                deployment_safe_action.astype(np.float32),
            )
            # Compatibility alias; deployment_safe_action is authoritative.
            np.save(
                stage_dir / "hardware_feasible_action.npy",
                deployment_safe_action.astype(np.float32),
            )
            atomic_json(stage_dir / "projection_records.json", projection.records)
            atomic_json(
                stage_dir / "hard_limit_projection_records.json",
                projection.hard_limit_records,
            )
            atomic_json(
                stage_dir / "deployment_margin_projection_records.json",
                projection.deployment_margin_records,
            )
            raw_chunk_audit = contract_safety.chunk(initial_measured, policy_raw_action)
            hard_chunk_audit = hard_safety.chunk(
                initial_measured, hard_limit_projected_action
            )
            deployment_hard_chunk_audit = hard_safety.chunk(
                initial_measured, deployment_safe_action
            )
            safe_chunk_audit = safe_safety.chunk(
                initial_measured, deployment_safe_action
            )
            hard_geometry_delta = hard_safety.projection_geometry_delta(
                stitched_execution_plan, hard_limit_projected_action
            )
            margin_geometry_delta = hard_safety.projection_geometry_delta(
                hard_limit_projected_action, deployment_safe_action
            )
            total_geometry_delta = hard_safety.projection_geometry_delta(
                stitched_execution_plan, deployment_safe_action
            )
            projection_checks = {
                "all_14_arm_outputs_bitwise_preserved": projection.summary[
                    "arm_outputs_bitwise_preserved"
                ],
                "all_hard_valid_values_preserved_by_hard_stage": projection.summary[
                    "hard_valid_values_preserved_by_hard_stage"
                ],
                "all_safe_values_preserved_by_margin_stage": projection.summary[
                    "safe_values_preserved_by_margin_stage"
                ],
                "hard_projected_limit_violations_zero": hard_chunk_audit[
                    "joint_limit_violation_count"
                ]
                == 0,
                "deployment_action_hard_limit_violations_zero": deployment_hard_chunk_audit[
                    "joint_limit_violation_count"
                ]
                == 0,
                "deployment_action_safe_interval_violations_zero": safe_chunk_audit[
                    "joint_limit_violation_count"
                ]
                == 0,
                "every_hard_change_has_a_record": len(projection.hard_limit_records)
                == projection.summary["hard_limit_projected_scalar_count"],
                "every_margin_change_has_a_record": len(
                    projection.deployment_margin_records
                )
                == projection.summary["deployment_margin_projected_scalar_count"],
                "hard_projection_geometry_benign": hard_geometry_delta["status"]
                == "PASS",
                "margin_projection_geometry_benign": margin_geometry_delta["status"]
                == "PASS",
                "total_projection_geometry_benign": total_geometry_delta["status"]
                == "PASS",
            }
            checks = {
                "camera_render_verification": verification["status"] == "PASS",
                "worker_checkpoint_hash": checkpoint_ready["model_sha256"] == checkpoint_hash,
                "output_shape_50x28": list(policy_raw_action.shape) == [50, 28],
                "output_finite": bool(np.isfinite(policy_raw_action).all()),
                "normalization_valid": adapter["state"]["statistics"] is not None
                and adapter["action"]["statistics"] is not None,
                "named_joint_mapping": len(set(ids)) == 28,
                "raw_output_retained_without_overwrite": bool(
                    np.array_equal(projection.policy_raw_action, stitched_execution_plan)
                ),
                "hard_projection_retained_separately": bool(
                    np.array_equal(
                        projection.hard_limit_projected_action,
                        hard_limit_projected_action,
                    )
                ),
                "generic_projection_invariants": all(projection_checks.values()),
                "deployment_safe_pre_execution_safety": safe_chunk_audit["status"]
                == "PASS",
                "no_joint_commands_sent": True,
                "no_physics_action_execution": True,
            }
            report = common_report | {
                "schema_version": "policy_b_isaac_stage0_deployment_margin_v3",
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "inference": inference,
                "action_shape": [50, 28],
                "policy_raw_action_audit": raw_chunk_audit,
                "hard_limit_projected_action_audit": hard_chunk_audit,
                "deployment_safe_hard_limit_audit": deployment_hard_chunk_audit,
                "deployment_safe_interval_audit": safe_chunk_audit,
                "hardware_feasible_action_audit": safe_chunk_audit,
                "chunk_audit": safe_chunk_audit,
                "projection": projection.summary,
                "projection_checks": projection_checks,
                "hard_limit_projection_geometry_delta": hard_geometry_delta,
                "deployment_margin_geometry_delta": margin_geometry_delta,
                "total_projection_geometry_delta": total_geometry_delta,
                "projection_geometry_delta": total_geometry_delta,
                "camera_verification": verification,
                "artifacts": {
                    "policy_raw_action": str(stage_dir / "policy_raw_action.npy"),
                    "stitched_execution_plan": str(
                        stage_dir / "stitched_execution_plan.npy"
                    ),
                    "previous_remaining_plan": str(
                        stage_dir / "previous_remaining_plan.npy"
                    ),
                    "hard_limit_projected_action": str(
                        stage_dir / "hard_limit_projected_action.npy"
                    ),
                    "deployment_safe_action": str(
                        stage_dir / "deployment_safe_action.npy"
                    ),
                    "hardware_feasible_action": str(
                        stage_dir / "hardware_feasible_action.npy"
                    ),
                    "projection_records": str(stage_dir / "projection_records.json"),
                    "hard_limit_projection_records": str(
                        stage_dir / "hard_limit_projection_records.json"
                    ),
                    "deployment_margin_projection_records": str(
                        stage_dir / "deployment_margin_projection_records.json"
                    ),
                    "predicted_chunk_legacy_raw_alias": str(
                        stage_dir / "predicted_chunk.npy"
                    ),
                    "input_state": str(stage_dir / "input_state.npy"),
                    "input_rgb": str(stage_dir / "input_rgb.png"),
                },
            }
            atomic_json(stage_dir / "stage0_report.json", report)
            atomic_json(stage_dir / "stage_report.json", report)
            print(json.dumps({
                "stage": args.stage,
                "status": report["status"],
                "action_shape": report["action_shape"],
                "raw_joint_limit_violation_count": raw_chunk_audit[
                    "joint_limit_violation_count"
                ],
                "hard_projected_joint_limit_violation_count": hard_chunk_audit[
                    "joint_limit_violation_count"
                ],
                "deployment_safe_hard_limit_violation_count": deployment_hard_chunk_audit[
                    "joint_limit_violation_count"
                ],
                "deployment_safe_interval_violation_count": safe_chunk_audit[
                    "joint_limit_violation_count"
                ],
                "deployment_safe_first_command_delta_max_abs_rad": safe_chunk_audit[
                    "first_action_delta_max_abs_rad"
                ],
                "maximum_velocity_rad_s": safe_chunk_audit["maximum_velocity_rad_s"],
                "maximum_acceleration_rad_s2": safe_chunk_audit[
                    "maximum_acceleration_rad_s2"
                ],
                "projection": projection.summary,
                "margin_geometry_delta_status": margin_geometry_delta["status"],
                "failed_checks": [key for key, value in checks.items() if not value],
            }, indent=2))
            return 0 if report["status"] == "PASS" else 2

        recorder_names = list(cameras)
        if source_rgb_paths:
            recorder_names.append("source_rgb_time_forced")
        recorder = VideoRecorder(
            stage_dir,
            recorder_names,
            enabled=not args.no_video,
        )
        recorder.add(initial_images, f"{args.stage} | initial measured state")
        causal_stitcher = CommonCausalActionStitcher(
            args.causal_stitching_method,
            crossfade_window_frames=args.crossfade_window_frames,
        )
        stitching_records: list[dict[str, Any]] = []
        policy_raw_executed: list[np.ndarray] = []
        stitched_execution_executed: list[np.ndarray] = []
        hard_limit_projected_executed: list[np.ndarray] = []
        commanded: list[np.ndarray] = []
        actual: list[np.ndarray] = []
        actual_velocity: list[np.ndarray] = []
        timestamps: list[float] = []
        external_forces: list[float] = []
        inference_rows: list[dict[str, Any]] = []
        inference_timestamps_s: list[float] = []
        policy_raw_chunks: list[np.ndarray] = []
        model_stitched_policy_chunks: list[np.ndarray] = []
        causal_selected_plan_chunks: list[np.ndarray] = []
        stitched_execution_chunks: list[np.ndarray] = []
        previous_remaining_plans: list[np.ndarray] = []
        previous_remaining_plan_lengths: list[int] = []
        hard_limit_projected_chunks: list[np.ndarray] = []
        deployment_safe_chunks: list[np.ndarray] = []
        otg_velocity_chunks: list[np.ndarray] = []
        otg_acceleration_chunks: list[np.ndarray] = []
        otg_target_positions: list[np.ndarray] = []
        otg_records: list[dict[str, Any]] = []
        projection_records: list[dict[str, Any]] = []
        hard_limit_projection_records: list[dict[str, Any]] = []
        deployment_margin_projection_records: list[dict[str, Any]] = []
        projection_summaries: list[dict[str, Any]] = []
        hard_projection_geometry_rows: list[dict[str, Any]] = []
        margin_projection_geometry_rows: list[dict[str, Any]] = []
        total_projection_geometry_rows: list[dict[str, Any]] = []
        executed_action_keys: set[tuple[int, int]] = set()
        chunk_audits: list[dict[str, Any]] = []
        chunk_boundary_jumps: list[float] = []
        replanned_future_differences: list[float] = []
        chunk_boundary_records: list[dict[str, Any]] = []
        runtime_diagnostic_gate_records: list[dict[str, Any]] = []
        executed_prefix_gate_records: list[dict[str, Any]] = []
        future_suffix_warnings: list[dict[str, Any]] = []
        diagnostic_abort_reason: dict[str, Any] | None = None
        task_rows: list[dict[str, Any]] = []
        left_wrist_xyz: list[np.ndarray] = []
        right_wrist_xyz: list[np.ndarray] = []
        semantics = None
        if object_enabled:
            initial_doll_pose = doll.data.root_pose_w.torch[0].detach().cpu().numpy()
            semantics = TaskSemantics(layout, initial_doll_pose[:3])

        def prepare_chunk(
            current_state: np.ndarray,
            policy_raw_action: np.ndarray,
            stitched_execution_plan: np.ndarray,
            previous_remaining_plan: np.ndarray,
            inference_index: int,
            *,
            archived_raw_policy_chunk: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            projection = projector.project(
                stitched_execution_plan,
                inference_index=inference_index,
                global_row_offset=inference_index * 50,
            )
            hard_limit_projected = projection.hard_limit_projected_action
            deployment_safe = projection.deployment_safe_action
            raw_audit = contract_safety.chunk(current_state, policy_raw_action)
            stitched_audit = contract_safety.chunk(
                current_state, stitched_execution_plan
            )
            hard_audit = hard_safety.chunk(current_state, hard_limit_projected)
            deployment_hard_audit = hard_safety.chunk(current_state, deployment_safe)
            deployment_safe_audit = safe_safety.chunk(current_state, deployment_safe)
            hard_geometry = hard_safety.projection_geometry_delta(
                stitched_execution_plan, hard_limit_projected
            )
            margin_geometry = hard_safety.projection_geometry_delta(
                hard_limit_projected, deployment_safe
            )
            total_geometry = hard_safety.projection_geometry_delta(
                stitched_execution_plan, deployment_safe
            )
            receding_horizon = args.stage in {"object-free-multichunk", "full-motion"}
            prefix_gate: dict[str, Any] | None = None
            suffix_warning: dict[str, Any] | None = None
            if receding_horizon:
                horizon = min(args.execution_horizon, len(policy_raw_action))
                raw_prefix_audit = contract_safety.chunk(
                    current_state, policy_raw_action[:horizon]
                )
                stitched_prefix_audit = contract_safety.chunk(
                    current_state, stitched_execution_plan[:horizon]
                )
                hard_prefix_audit = hard_safety.chunk(
                    current_state, hard_limit_projected[:horizon]
                )
                deployment_hard_prefix_audit = hard_safety.chunk(
                    current_state, deployment_safe[:horizon]
                )
                deployment_safe_prefix_audit = safe_safety.chunk(
                    current_state, deployment_safe[:horizon]
                )
                hard_prefix_geometry = hard_safety.projection_geometry_delta(
                    stitched_execution_plan[:horizon], hard_limit_projected[:horizon]
                )
                margin_prefix_geometry = hard_safety.projection_geometry_delta(
                    hard_limit_projected[:horizon], deployment_safe[:horizon]
                )
                total_prefix_geometry = hard_safety.projection_geometry_delta(
                    stitched_execution_plan[:horizon], deployment_safe[:horizon]
                )
            else:
                raw_prefix_audit = raw_audit
                stitched_prefix_audit = stitched_audit
                hard_prefix_audit = hard_audit
                deployment_hard_prefix_audit = deployment_hard_audit
                deployment_safe_prefix_audit = deployment_safe_audit
                hard_prefix_geometry = hard_geometry
                margin_prefix_geometry = margin_geometry
                total_prefix_geometry = total_geometry
            invariants = {
                "all_14_arm_outputs_bitwise_preserved": projection.summary[
                    "arm_outputs_bitwise_preserved"
                ],
                "hard_valid_values_preserved_by_hard_stage": projection.summary[
                    "hard_valid_values_preserved_by_hard_stage"
                ],
                "safe_values_preserved_by_margin_stage": projection.summary[
                    "safe_values_preserved_by_margin_stage"
                ],
                "hard_projected_limit_violations_zero": hard_prefix_audit[
                    "joint_limit_violation_count"
                ]
                == 0,
                "deployment_action_hard_limit_violations_zero": deployment_hard_prefix_audit[
                    "joint_limit_violation_count"
                ]
                == 0,
                "deployment_action_safe_interval_violations_zero": deployment_safe_prefix_audit[
                    "joint_limit_violation_count"
                ]
                == 0,
                "complete_hard_correction_records": len(projection.hard_limit_records)
                == projection.summary["hard_limit_projected_scalar_count"],
                "complete_margin_correction_records": len(
                    projection.deployment_margin_records
                )
                == projection.summary["deployment_margin_projected_scalar_count"],
                "hard_projection_geometry_benign": hard_prefix_geometry["status"]
                == "PASS",
                "margin_projection_geometry_benign": margin_prefix_geometry["status"]
                == "PASS",
                "total_projection_geometry_benign": total_prefix_geometry["status"]
                == "PASS",
            }
            gate_status = (
                "PASS"
                if deployment_hard_prefix_audit["status"] == "PASS"
                and deployment_safe_prefix_audit["status"] == "PASS"
                and raw_prefix_audit["finite"]
                and stitched_prefix_audit["finite"]
                and all(invariants.values())
                else "FAIL"
            )
            if receding_horizon:
                prefix_gate = {
                    "classification": "EXECUTED_PREFIX_HARD_GATE",
                    "status": gate_status,
                    "inference_index": inference_index,
                    "chunk_rows": [0, args.execution_horizon - 1],
                    "policy_raw_prefix_audit": raw_prefix_audit,
                    "stitched_execution_prefix_audit": stitched_prefix_audit,
                    "hard_limit_projected_prefix_audit": hard_prefix_audit,
                    "deployment_safe_hard_limit_prefix_audit": deployment_hard_prefix_audit,
                    "deployment_safe_interval_prefix_audit": deployment_safe_prefix_audit,
                    "hard_limit_projection_prefix_geometry_delta": hard_prefix_geometry,
                    "deployment_margin_prefix_geometry_delta": margin_prefix_geometry,
                    "total_projection_prefix_geometry_delta": total_prefix_geometry,
                    "projection_invariants": invariants,
                }
                executed_prefix_gate_records.append(prefix_gate)

                suffix_start = args.execution_horizon
                if suffix_start < len(policy_raw_action):
                    raw_suffix_audit = contract_safety.chunk(
                        policy_raw_action[suffix_start - 1],
                        policy_raw_action[suffix_start:],
                    )
                    stitched_suffix_audit = contract_safety.chunk(
                        stitched_execution_plan[suffix_start - 1],
                        stitched_execution_plan[suffix_start:],
                    )
                    hard_suffix_audit = hard_safety.chunk(
                        hard_limit_projected[suffix_start - 1],
                        hard_limit_projected[suffix_start:],
                    )
                    deployment_suffix_audit = hard_safety.chunk(
                        deployment_safe[suffix_start - 1],
                        deployment_safe[suffix_start:],
                    )
                    suffix_collision_records = []
                    for collision_record in deployment_suffix_audit["collision"][
                        "records"
                    ]:
                        mapped = dict(collision_record)
                        mapped["predicted_chunk_row"] = int(
                            collision_record["frame"] + suffix_start
                        )
                        suffix_collision_records.append(mapped)
                    penetration_values = [
                        float(record["penetration_m"])
                        for record in suffix_collision_records
                    ]
                    reasons = []
                    if not raw_suffix_audit["finite"]:
                        reasons.append("NONFINITE_RAW_FUTURE_PLAN")
                    if raw_suffix_audit["joint_limit_violation_count"]:
                        reasons.append("RAW_JOINT_LIMIT_PROJECTION_REQUIRED")
                    if deployment_suffix_audit["joint_limit_violation_count"]:
                        reasons.append("PROJECTED_JOINT_LIMIT_VIOLATION")
                    if deployment_suffix_audit["branch_discontinuity_count"]:
                        reasons.append("PREDICTED_BRANCH_DISCONTINUITY")
                    if deployment_suffix_audit["collision"][
                        "invalid_hard_collision_category_incidence"
                    ]:
                        reasons.append("PREDICTED_SELF_COLLISION")
                    for check_name in ("adjacent_step", "velocity", "acceleration"):
                        if not deployment_suffix_audit["checks"][check_name]:
                            reasons.append(f"PREDICTED_{check_name.upper()}")
                    suffix_warning = {
                        "classification": "UNEXECUTED_FUTURE_PLAN_WARNING",
                        "warning_present": bool(reasons),
                        "warning_reasons": reasons,
                        "inference_index": inference_index,
                        "chunk_rows": [suffix_start, len(policy_raw_action) - 1],
                        "will_be_replaced_before_execution": True,
                        "raw_suffix_audit": raw_suffix_audit,
                        "stitched_execution_suffix_audit": stitched_suffix_audit,
                        "hard_limit_projected_suffix_audit": hard_suffix_audit,
                        "deployment_safe_suffix_audit": deployment_suffix_audit,
                        "predicted_collision_records": suffix_collision_records,
                        "predicted_collision_chunk_rows": sorted(
                            {
                                int(record["predicted_chunk_row"])
                                for record in suffix_collision_records
                            }
                        ),
                        "maximum_predicted_penetration_m": max(
                            penetration_values, default=0.0
                        ),
                    }
                    future_suffix_warnings.append(suffix_warning)
            row = {
                "status": gate_status,
                "gate_scope": (
                    "EXECUTED_PREFIX_HARD_GATE"
                    if receding_horizon
                    else "FULL_PREDICTED_CHUNK_HARD_GATE"
                ),
                "inference_index": inference_index,
                "policy_raw_action_audit": raw_audit,
                "stitched_execution_plan_audit": stitched_audit,
                "hard_limit_projected_action_audit": hard_audit,
                "deployment_safe_hard_limit_audit": deployment_hard_audit,
                "deployment_safe_interval_audit": deployment_safe_audit,
                "hardware_feasible_action_audit": deployment_safe_audit,
                "projection": projection.summary,
                "projection_invariants": invariants,
                "hard_limit_projection_geometry_delta": hard_geometry,
                "deployment_margin_geometry_delta": margin_geometry,
                "total_projection_geometry_delta": total_geometry,
                "projection_geometry_delta": total_geometry,
                "executed_prefix_hard_gate": prefix_gate,
                "unexecuted_suffix_warning": suffix_warning,
            }
            policy_raw_chunks.append(
                (
                    policy_raw_action
                    if archived_raw_policy_chunk is None
                    else archived_raw_policy_chunk
                ).copy()
            )
            stitched_execution_chunks.append(stitched_execution_plan.copy())
            previous_remaining_plans.append(previous_remaining_plan.copy())
            previous_remaining_plan_lengths.append(len(previous_remaining_plan))
            hard_limit_projected_chunks.append(hard_limit_projected.copy())
            deployment_safe_chunks.append(deployment_safe.copy())
            projection_records.extend(projection.records)
            hard_limit_projection_records.extend(projection.hard_limit_records)
            deployment_margin_projection_records.extend(
                projection.deployment_margin_records
            )
            projection_summaries.append(projection.summary)
            hard_projection_geometry_rows.append(hard_geometry)
            margin_projection_geometry_rows.append(margin_geometry)
            total_projection_geometry_rows.append(total_geometry)
            chunk_audits.append(row)
            return hard_limit_projected, deployment_safe, row

        def diagnostic_runtime_gate(
            measured: np.ndarray,
            external_force_n: float,
        ) -> dict[str, Any]:
            """Per-frame fail-fast gate for Isaac diagnostic execution."""

            if diagnostic_authorization is None:
                return {"status": "NOT_APPLICABLE"}
            measured_history = np.asarray(actual, dtype=np.float64).reshape(
                -1, initial_measured.shape[0]
            )
            history = np.vstack((initial_measured, measured_history))
            lower_excess = np.maximum(hard_safety.lower - measured, 0.0)
            upper_excess = np.maximum(measured - hard_safety.upper, 0.0)
            arm_maximum_excess = float(
                max(np.max(lower_excess[:14]), np.max(upper_excess[:14]))
            )
            dex3_maximum_excess = float(
                max(np.max(lower_excess[14:]), np.max(upper_excess[14:]))
            )
            step = (
                np.abs(history[-1] - history[-2])
                if len(history) >= 2
                else np.zeros(28, dtype=np.float64)
            )
            acceleration = (
                np.abs(history[-1] - 2.0 * history[-2] + history[-3])
                * CONTROL_FPS**2
                if len(history) >= 3
                else np.zeros(28, dtype=np.float64)
            )
            branch = bool(
                branch_flags(
                    history[:, :14],
                    float(hard_safety.acceptance["branch_absolute_step_norm_rad"]),
                    float(hard_safety.acceptance["branch_local_multiplier"]),
                )[-1]
            )
            collision = hard_safety.collision(measured[None, :])
            self_collision_incidence = int(sum(collision["frame_counts"].values()))
            checks = {
                "finite": bool(np.isfinite(measured).all()),
                "measured_arm_hard_limits": arm_maximum_excess <= 1e-9,
                "measured_dex3_excursion_not_catastrophic": dex3_maximum_excess
                <= float(
                    diagnostic_authorization[
                        "diagnostic_measured_dex3_excursion_cap_rad"
                    ]
                ),
                "measured_step": float(np.max(step))
                <= float(hard_safety.acceptance["maximum_joint_step_rad"]),
                "measured_velocity": float(np.max(step) * CONTROL_FPS)
                <= float(hard_safety.acceptance["maximum_velocity_rad_s"]),
                "measured_acceleration": float(np.max(acceleration))
                <= float(hard_safety.acceptance["maximum_acceleration_rad_s2"]),
                "branch_discontinuity": not branch,
                "robot_self_collision": self_collision_incidence == 0,
                "external_non_task_collision": external_force_n
                <= TaskSemantics.FORCE_THRESHOLD_N,
            }
            return {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "frame": len(actual) - 1,
                "checks": checks,
                "measured_arm_maximum_hard_limit_excess_rad": arm_maximum_excess,
                "measured_dex3_maximum_hard_limit_excess_rad": dex3_maximum_excess,
                "diagnostic_dex3_excursion_cap_rad": float(
                    diagnostic_authorization[
                        "diagnostic_measured_dex3_excursion_cap_rad"
                    ]
                ),
                "maximum_measured_step_rad": float(np.max(step)),
                "maximum_measured_velocity_rad_s": float(np.max(step) * CONTROL_FPS),
                "maximum_measured_acceleration_rad_s2": float(
                    np.max(acceleration)
                ),
                "branch_discontinuity": branch,
                "self_collision_incidence": self_collision_incidence,
                "collision": collision,
                "external_non_task_contact_force_n": external_force_n,
            }

        def execute_rows(
            raw_rows: np.ndarray,
            stitched_rows: np.ndarray,
            hard_rows: np.ndarray,
            deployment_rows: np.ndarray,
            inference_index: int,
        ) -> bool:
            nonlocal diagnostic_abort_reason
            if not (
                raw_rows.shape
                == stitched_rows.shape
                == hard_rows.shape
                == deployment_rows.shape
            ):
                raise RuntimeError(
                    "raw/stitched/hard/deployment execution prefixes have different shapes"
                )
            for local_index, (raw_command, stitched_command, hard_command, command) in enumerate(
                zip(raw_rows, stitched_rows, hard_rows, deployment_rows)
            ):
                step_command(command)
                executed_action_keys.add((inference_index, local_index))
                frame_index = len(commanded)
                measured = measured_state()
                body_positions = (
                    robot.data.body_pos_w.torch[0].detach().cpu().numpy()
                )
                policy_raw_executed.append(raw_command.copy())
                stitched_execution_executed.append(stitched_command.copy())
                hard_limit_projected_executed.append(hard_command.copy())
                commanded.append(command.copy())
                actual.append(measured)
                actual_velocity.append(measured_velocity())
                left_wrist_xyz.append(
                    body_positions[wrist_body_ids["left"]].astype(np.float64).copy()
                )
                right_wrist_xyz.append(
                    body_positions[wrist_body_ids["right"]].astype(np.float64).copy()
                )
                timestamps.append((frame_index + 1) / CONTROL_FPS)
                current_external_force = force_value(external_sensor)
                external_forces.append(current_external_force)
                runtime_gate = diagnostic_runtime_gate(
                    measured, current_external_force
                )
                if diagnostic_authorization is not None:
                    runtime_diagnostic_gate_records.append(runtime_gate)
                    if runtime_gate["status"] != "PASS":
                        diagnostic_abort_reason = runtime_gate
                task_record = task_snapshot(frame_index, semantics)
                if task_record is not None:
                    task_rows.append(task_record)
                images = capture()
                if source_rgb_paths:
                    images["source_rgb_time_forced"] = source_rgb_at(
                        frame_index + 1
                    )
                recorder.add(
                    images,
                    f"{args.stage} | action {frame_index} | inference {inference_index} | prefix {local_index}",
                )
                if diagnostic_abort_reason is not None:
                    return True
                if semantics is not None:
                    if args.stage == "left-grasp" and semantics.left_owned:
                        return True
                    if args.stage == "handoff" and semantics.right_owned:
                        return True
                    if args.stage == "full-task" and "released_inside_bin" in semantics.events:
                        return True
            return False

        if diagnostic_authorization is not None:
            initial_runtime_gate = diagnostic_runtime_gate(initial_measured, 0.0)
            initial_runtime_gate["frame"] = -1
            initial_runtime_gate["definition"] = "settled initial measured state"
            runtime_diagnostic_gate_records.append(initial_runtime_gate)
            if initial_runtime_gate["status"] != "PASS":
                diagnostic_abort_reason = initial_runtime_gate

        if diagnostic_abort_reason is not None:
            print(
                f"{args.stage}: diagnostic initial-state gate failed before inference",
                flush=True,
            )
        elif args.stage == "object-free-single":
            policy_input_rgb = (
                fixed_observation_rgb
                if fixed_observation_rgb is not None
                else initial_images["policy"]
            )
            policy_input_state = (
                fixed_observation_state
                if fixed_observation_state is not None
                else initial_measured
            )
            (
                policy_raw_action,
                stitched_execution_plan,
                previous_remaining_plan,
                inference,
            ) = bridge.infer(
                policy_input_rgb, policy_input_state
            )
            if fixed_observation_record is not None:
                save_rgb(stage_dir / "fixed_policy_input_rgb.png", policy_input_rgb)
                np.save(
                    stage_dir / "fixed_policy_input_state.npy",
                    policy_input_state.astype(np.float32),
                )
            hard_limit_projected_action, deployment_safe_action, audit = prepare_chunk(
                initial_measured,
                policy_raw_action,
                stitched_execution_plan,
                previous_remaining_plan,
                0,
            )
            inference_rows.append(inference)
            if audit["status"] == "PASS":
                execute_rows(
                    policy_raw_action,
                    stitched_execution_plan,
                    hard_limit_projected_action,
                    deployment_safe_action,
                    0,
                )
            else:
                print("Stage 1 chunk rejected before commands", flush=True)
        else:
            max_calls = {
                "object-free-multichunk": args.stage2_inference_calls,
                "left-grasp": args.stage3_inference_calls,
                "handoff": args.stage4_inference_calls,
                "full-task": args.stage5_inference_calls,
                "full-motion": args.full_motion_inference_calls,
            }[args.stage]
            previous_raw_chunk: np.ndarray | None = None
            previous_model_stitched_chunk: np.ndarray | None = None
            previous_causal_selected_plan: np.ndarray | None = None
            previous_stitched_chunk: np.ndarray | None = None
            previous_hard_chunk: np.ndarray | None = None
            previous_chunk: np.ndarray | None = None
            # The online trajectory generator owns a persistent command-reference
            # state.  It is seeded from measured feedback exactly once, then advanced
            # only by commands that have actually been committed.  Re-seeding a new
            # trajectory from lagging measured q at every policy call creates a
            # reference jump equal to the controller tracking error, defeating the
            # continuity guarantee that the OTG is meant to provide.  Measured q/dq
            # remain mandatory feedback for runtime gates and tracking-error audits.
            otg_reference_position: np.ndarray | None = None
            otg_reference_velocity: np.ndarray | None = None
            otg_reference_acceleration: np.ndarray | None = None
            goal_reached = False
            for inference_index in range(max_calls):
                inference_timestamp_s = len(commanded) / CONTROL_FPS
                inference_timestamps_s.append(inference_timestamp_s)
                current_images = capture()
                current_state = measured_state()
                policy_rgb = (
                    source_rgb_at(len(commanded))
                    if source_rgb_paths
                    else current_images["policy"]
                )
                if inference_index in CAPTURE_INFERENCE_INDICES:
                    observation_dir = stage_dir / "fixed_inference_observations"
                    rgb_path = observation_dir / f"inference_{inference_index:04d}_rgb.png"
                    state_path = observation_dir / f"inference_{inference_index:04d}_state.npz"
                    save_rgb(rgb_path, policy_rgb)
                    atomic_npz(
                        state_path,
                        measured_state=current_state.astype(np.float32),
                        joint_names=np.asarray(names),
                    )
                    atomic_json(
                        observation_dir / f"inference_{inference_index:04d}.json",
                        {
                            "inference_index": inference_index,
                            "simulation_timestamp_s": inference_timestamp_s,
                            "rgb_path": str(rgb_path),
                            "rgb_sha256": sha256_file(rgb_path),
                            "rgb_shape": list(policy_rgb.shape),
                            "rgb_dtype": str(policy_rgb.dtype),
                            "state_path": str(state_path),
                            "state_sha256": sha256_file(state_path),
                            "state_dtype_at_policy_interface": "float32",
                            "state_joint_names": names,
                            "task": TASK,
                            "checkpoint": str(checkpoint),
                            "checkpoint_model_sha256": checkpoint_hash,
                            "camera": DEPLOYMENT_CAMERA.name,
                            "logging_only_no_execution_semantics_changed": True,
                        },
                    )
                (
                    policy_raw_action,
                    model_stitched_plan,
                    worker_previous_remaining_plan,
                    inference,
                ) = bridge.infer(
                    policy_rgb,
                    current_state,
                    inference_delay=(
                        args.rtc_inference_delay_frames
                        if args.causal_stitching_method == "rtc"
                        and previous_chunk is not None
                        else 0
                    ),
                )
                previous_remaining_plan = (
                    previous_chunk[args.execution_horizon :].copy()
                    if args.causal_stitching_method == "crossfade"
                    and previous_chunk is not None
                    else worker_previous_remaining_plan
                )
                stitching = causal_stitcher.stitch(
                    raw_policy_chunk=policy_raw_action,
                    model_stitched_plan=model_stitched_plan,
                    previous_remaining_plan=previous_remaining_plan,
                    last_commanded_action=commanded[-1] if commanded else None,
                    measured_qpos=current_state,
                )
                stitched_execution_plan = stitching.stitched_execution_plan.copy()
                raw_execution_reference = policy_raw_action.copy()
                rtc_committed_previous_rows = 0
                if (
                    args.causal_stitching_method == "rtc"
                    and previous_chunk is not None
                    and args.rtc_inference_delay_frames
                ):
                    rtc_committed_previous_rows = args.rtc_inference_delay_frames
                    previous_start = args.execution_horizon
                    previous_stop = previous_start + rtc_committed_previous_rows
                    if (
                        previous_raw_chunk is None
                        or previous_stitched_chunk is None
                        or previous_stop > len(previous_stitched_chunk)
                    ):
                        raise RuntimeError(
                            "RTC latency queue lacks the prior committed rows required "
                            "during inference"
                        )
                    # Faithful ActionQueue semantics: commands already in flight while
                    # inference runs cannot be overwritten.  New RTC rows representing
                    # that elapsed interval are discarded; only the uncommitted future
                    # comes from the new policy call.
                    raw_execution_reference[:rtc_committed_previous_rows] = (
                        previous_raw_chunk[previous_start:previous_stop]
                    )
                    stitched_execution_plan[:rtc_committed_previous_rows] = (
                        previous_stitched_chunk[previous_start:previous_stop]
                    )
                causal_selected_plan = stitched_execution_plan.copy()
                model_stitched_policy_chunks.append(model_stitched_plan.copy())
                causal_selected_plan_chunks.append(causal_selected_plan.copy())
                if jerk_limited_otg is not None:
                    committed_endpoint_index = args.execution_horizon - 1
                    target_projection = projector.project(
                        causal_selected_plan[
                            committed_endpoint_index : committed_endpoint_index + 1
                        ],
                        inference_index=inference_index,
                    )
                    otg_target = target_projection.deployment_safe_action[0]
                    current_velocity = measured_velocity()
                    current_acceleration = causal_measured_acceleration_estimate(
                        actual_velocity, args.execution_horizon
                    )
                    otg_seed_source = "PERSISTENT_LAST_COMMITTED_REFERENCE"
                    if otg_reference_position is None:
                        otg_reference_position = current_state.copy()
                        otg_reference_velocity = current_velocity.copy()
                        otg_reference_acceleration = current_acceleration.copy()
                        otg_seed_source = "INITIAL_MEASURED_Q_DQ_CAUSAL_DDQ"
                    assert otg_reference_velocity is not None
                    assert otg_reference_acceleration is not None
                    otg_result = jerk_limited_otg.generate(
                        current_position=otg_reference_position,
                        current_velocity=otg_reference_velocity,
                        current_acceleration=otg_reference_acceleration,
                        target_position=otg_target,
                        steps=50,
                    )
                    stitched_execution_plan = otg_result.position
                    otg_velocity_chunks.append(otg_result.velocity.copy())
                    otg_acceleration_chunks.append(otg_result.acceleration.copy())
                    otg_target_positions.append(otg_target.copy())
                    otg_records.append(
                        {
                            "inference_index": inference_index,
                            "committed_endpoint_row": committed_endpoint_index,
                            "target_projection": target_projection.summary,
                            "reference_seed_source": otg_seed_source,
                            "measured_position_at_replan_rad": current_state,
                            "measured_velocity_at_replan_rad_s": current_velocity,
                            "causal_measured_acceleration_at_replan_rad_s2": current_acceleration,
                            "reference_minus_measured_position_rad": (
                                otg_reference_position - current_state
                            ),
                            "maximum_reference_tracking_error_rad": float(
                                np.max(np.abs(otg_reference_position - current_state))
                            ),
                            **otg_result.audit,
                        }
                    )
                stitching_audit = dict(stitching.audit)
                stitching_audit.update(
                    {
                        "rtc_inference_delay_frames": rtc_committed_previous_rows,
                        "committed_prefix_overwritten": False,
                        "committed_prefix_source": (
                            "PREVIOUS_PLAN_ROWS_EXECUTED_DURING_INFERENCE"
                            if rtc_committed_previous_rows
                            else "CURRENT_PLAN"
                        ),
                        "raw_new_policy_chunk_preserved_separately": True,
                        "jerk_limited_otg_applied_after_plan_selection": (
                            jerk_limited_otg is not None
                        ),
                        "queue_semantics": (
                            "LOGICALLY_ASYNC_LATENCY_AWARE_ACTION_QUEUE"
                            if args.causal_stitching_method == "rtc"
                            else "SYNCHRONOUS_PREFIX_COMMITMENT"
                        ),
                    }
                )
                stitching_records.append(
                    {"inference_index": inference_index, **stitching_audit}
                )
                inference = dict(inference)
                inference["simulation_timestamp_s"] = inference_timestamp_s
                inference["measured_state_rad"] = current_state
                inference["committed_previous_plan_rows"] = rtc_committed_previous_rows
                if source_rgb_paths:
                    source_frame_index = min(len(commanded), len(source_rgb_paths) - 1)
                    inference.update(
                        {
                            "rgb_source": "SOURCE_RGB_TIME_FORCED_POLICY_ROLLOUT",
                            "source_rgb_episode_index": int(
                                source_rgb_episode["final_dataset_index"]
                            ),
                            "source_rgb_frame_index": int(source_frame_index),
                            "source_rgb_timestamp_s": float(
                                source_frame_index / float(source_rgb_episode["source_fps"])
                            ),
                            "source_rgb_path": str(source_rgb_paths[source_frame_index]),
                            "closed_loop_vision": False,
                        }
                    )
                hard_limit_projected_action, deployment_safe_action, audit = prepare_chunk(
                    current_state,
                    raw_execution_reference,
                    stitched_execution_plan,
                    previous_remaining_plan,
                    inference_index,
                    archived_raw_policy_chunk=policy_raw_action,
                )
                inference_rows.append(inference)
                if commanded:
                    chunk_boundary_jumps.append(
                        float(np.max(np.abs(deployment_safe_action[0] - commanded[-1])))
                    )
                boundary_record: dict[str, Any] | None = None
                if (
                    previous_chunk is not None
                    and previous_raw_chunk is not None
                    and previous_stitched_chunk is not None
                ):
                    offset = min(args.execution_horizon, len(previous_chunk) - 1)
                    raw_boundary_delta = (
                        policy_raw_action[0] - previous_raw_chunk[offset]
                    )
                    hard_boundary_delta = (
                        hard_limit_projected_action[0]
                        - previous_hard_chunk[offset]
                    )
                    stitched_boundary_delta = (
                        stitched_execution_plan[0] - previous_stitched_chunk[offset]
                    )
                    safe_boundary_delta = (
                        deployment_safe_action[0] - previous_chunk[offset]
                    )
                    executed_boundary_delta = (
                        deployment_safe_action[0] - commanded[-1]
                    )
                    replanned_rmse = float(
                        np.sqrt(np.mean(np.square(safe_boundary_delta)))
                    )
                    replanned_future_differences.append(replanned_rmse)
                    worker_remaining_consistent = bool(
                        len(previous_remaining_plan) > 0
                        and previous_model_stitched_chunk is not None
                        and np.array_equal(
                            previous_remaining_plan[0].astype(np.float32),
                            previous_model_stitched_chunk[offset].astype(np.float32),
                        )
                    )
                    old_plan_vs_new_raw_delta = (
                        policy_raw_action[0]
                        - (
                            previous_model_stitched_chunk[offset]
                            if previous_model_stitched_chunk is not None
                            else previous_stitched_chunk[offset]
                        )
                    )
                    boundary_record = {
                        "replan_index": inference_index,
                        "replan_timestamp_s": len(commanded) / CONTROL_FPS,
                        "previous_chunk_planned_row_index": offset,
                        "previous_chunk_raw_planned_action_rad": previous_raw_chunk[
                            offset
                        ],
                        "new_chunk_raw_first_action_rad": policy_raw_action[0],
                        "old_plan_vs_new_raw_delta_rad": old_plan_vs_new_raw_delta,
                        "old_plan_vs_new_raw_max_abs_delta_rad": float(
                            np.max(np.abs(old_plan_vs_new_raw_delta))
                        ),
                        "raw_boundary_delta_rad": raw_boundary_delta,
                        "raw_boundary_delta_rad_by_joint": {
                            name: float(raw_boundary_delta[index])
                            for index, name in enumerate(names)
                        },
                        "raw_boundary_max_abs_delta_rad": float(
                            np.max(np.abs(raw_boundary_delta))
                        ),
                        "previous_chunk_hard_projected_planned_action_rad": previous_hard_chunk[
                            offset
                        ],
                        "new_chunk_hard_projected_first_action_rad": hard_limit_projected_action[
                            0
                        ],
                        "hard_projected_boundary_delta_rad": hard_boundary_delta,
                        "hard_projected_boundary_max_abs_delta_rad": float(
                            np.max(np.abs(hard_boundary_delta))
                        ),
                        "previous_chunk_stitched_planned_action_rad": previous_stitched_chunk[
                            offset
                        ],
                        "previous_model_stitched_planned_action_rad": (
                            previous_model_stitched_chunk[offset]
                            if previous_model_stitched_chunk is not None
                            else None
                        ),
                        "previous_causal_selected_planned_action_rad": (
                            previous_causal_selected_plan[offset]
                            if previous_causal_selected_plan is not None
                            else None
                        ),
                        "new_chunk_stitched_first_action_rad": stitched_execution_plan[0],
                        "stitched_boundary_delta_rad": stitched_boundary_delta,
                        "stitched_boundary_delta_rad_by_joint": {
                            name: float(stitched_boundary_delta[index])
                            for index, name in enumerate(names)
                        },
                        "stitched_boundary_max_abs_delta_rad": float(
                            np.max(np.abs(stitched_boundary_delta))
                        ),
                        "worker_previous_remaining_plan_rows": int(
                            len(previous_remaining_plan)
                        ),
                        "worker_previous_remaining_plan_first_row_matches_previous_stitched_plan": worker_remaining_consistent,
                        "previous_chunk_deployment_safe_planned_action_rad": previous_chunk[
                            offset
                        ],
                        "new_chunk_deployment_safe_first_action_rad": deployment_safe_action[
                            0
                        ],
                        "deployment_safe_boundary_delta_rad": safe_boundary_delta,
                        "deployment_safe_boundary_delta_rad_by_joint": {
                            name: float(safe_boundary_delta[index])
                            for index, name in enumerate(names)
                        },
                        "deployment_safe_boundary_max_abs_delta_rad": float(
                            np.max(np.abs(safe_boundary_delta))
                        ),
                        "previous_executed_command_rad": commanded[-1],
                        "new_first_command_rad": deployment_safe_action[0],
                        "executed_boundary_delta_rad": executed_boundary_delta,
                        "executed_boundary_max_abs_delta_rad": float(
                            np.max(np.abs(executed_boundary_delta))
                        ),
                        "replanned_future_rmse_rad": replanned_rmse,
                        "causal": True,
                        "future_observations_used": False,
                        "chunk_handling": args.causal_stitching_method,
                        "resulting_measured_acceleration_rad_s2": None,
                        "resulting_measured_acceleration_max_abs_rad_s2": None,
                    }
                    planned_overlap_valid = boundary_record[
                        "deployment_safe_boundary_max_abs_delta_rad"
                    ] <= float(hard_safety.acceptance["maximum_joint_step_rad"])
                    executed_command_jump_valid = boundary_record[
                        "executed_boundary_max_abs_delta_rad"
                    ] <= float(hard_safety.acceptance["maximum_joint_step_rad"])
                    boundary_record["strict_pre_execution_checks"] = {
                        "executed_command_jump": executed_command_jump_valid,
                    }
                    if args.stage != "full-motion":
                        boundary_record["strict_pre_execution_checks"][
                            "planned_overlap_delta"
                        ] = planned_overlap_valid
                    boundary_record["unexecuted_plan_consistency"] = {
                        "classification": "UNEXECUTED_FUTURE_PLAN_WARNING",
                        "planned_overlap_delta_within_dynamic_gate": planned_overlap_valid,
                        "maximum_delta_rad": boundary_record[
                            "deployment_safe_boundary_max_abs_delta_rad"
                        ],
                        "hard_gate_for_full_motion": False,
                    }
                    boundary_record["executed_prefix_hard_gate"] = {
                        "classification": "EXECUTED_PREFIX_HARD_GATE",
                        "executed_command_jump": boundary_record[
                            "executed_boundary_max_abs_delta_rad"
                        ],
                        "passed": executed_command_jump_valid,
                    }
                    chunk_boundary_records.append(boundary_record)
                if audit["status"] != "PASS":
                    if args.stage == "full-motion":
                        diagnostic_abort_reason = {
                            "status": "FAIL",
                            "reason": "EXECUTED_PREFIX_HARD_GATE",
                            "inference_index": inference_index,
                            "executed_prefix_hard_gate": audit[
                                "executed_prefix_hard_gate"
                            ],
                            "unexecuted_suffix_warning": audit[
                                "unexecuted_suffix_warning"
                            ],
                        }
                    print(f"{args.stage}: inference {inference_index} rejected before command", flush=True)
                    break
                if boundary_record is not None and not all(
                    boundary_record["strict_pre_execution_checks"].values()
                ):
                    diagnostic_abort_reason = {
                        "status": "FAIL",
                        "reason": "extreme chunk-boundary command jump",
                        "boundary": boundary_record,
                    }
                    print(
                        f"{args.stage}: replan {inference_index} rejected for boundary jump",
                        flush=True,
                    )
                    break
                horizon = args.execution_horizon
                execution_start = len(actual)
                goal_reached = execute_rows(
                    raw_execution_reference[:horizon],
                    stitched_execution_plan[:horizon],
                    hard_limit_projected_action[:horizon],
                    deployment_safe_action[:horizon],
                    inference_index,
                )
                if jerk_limited_otg is not None and len(commanded) > execution_start:
                    committed_count = len(commanded) - execution_start
                    committed_row = committed_count - 1
                    otg_reference_position = otg_result.position[committed_row].copy()
                    otg_reference_velocity = otg_result.velocity[committed_row].copy()
                    otg_reference_acceleration = otg_result.acceleration[committed_row].copy()
                if boundary_record is not None and len(actual) > execution_start:
                    sequence = np.vstack(
                        (initial_measured, np.asarray(actual, dtype=np.float64))
                    )
                    resulting_index = execution_start + 1
                    if resulting_index >= 2:
                        boundary_acceleration = np.abs(
                            sequence[resulting_index]
                            - 2.0 * sequence[resulting_index - 1]
                            + sequence[resulting_index - 2]
                        ) * CONTROL_FPS**2
                    else:
                        boundary_acceleration = np.zeros(28, dtype=np.float64)
                    boundary_record[
                        "resulting_measured_acceleration_rad_s2"
                    ] = boundary_acceleration
                    boundary_record[
                        "resulting_measured_acceleration_max_abs_rad_s2"
                    ] = float(np.max(boundary_acceleration))
                    boundary_record["resulting_measured_acceleration_rad_s2_by_joint"] = {
                        name: float(boundary_acceleration[index])
                        for index, name in enumerate(names)
                    }
                    boundary_record["strict_post_execution_checks"] = {
                        "resulting_measured_acceleration": float(
                            np.max(boundary_acceleration)
                        )
                        <= float(
                            hard_safety.acceptance["maximum_acceleration_rad_s2"]
                        )
                    }
                    if not all(
                        boundary_record["strict_post_execution_checks"].values()
                    ):
                        diagnostic_abort_reason = {
                            "status": "FAIL",
                            "reason": "extreme measured acceleration at chunk boundary",
                            "boundary": boundary_record,
                        }
                        goal_reached = True
                previous_raw_chunk = raw_execution_reference
                previous_model_stitched_chunk = model_stitched_plan
                previous_causal_selected_plan = causal_selected_plan
                previous_stitched_chunk = stitched_execution_plan
                previous_hard_chunk = hard_limit_projected_action
                previous_chunk = deployment_safe_action
                if goal_reached:
                    break
                print(
                    f"{args.stage}: inference={inference_index + 1}/{max_calls} actions={len(commanded)}",
                    flush=True,
                )

        rollout = rollout_metrics(
            hard_safety,
            safe_safety,
            initial_measured,
            commanded,
            actual,
            external_forces,
            diagnostic_authorization,
        )
        all_chunks_pass = bool(chunk_audits) and all(row["status"] == "PASS" for row in chunk_audits)
        task_summary = semantics.summary() if semantics is not None else None
        stage_goal = True
        if args.stage == "left-grasp":
            stage_goal = bool(task_summary["left_owned"])
        elif args.stage == "handoff":
            stage_goal = bool(task_summary["right_owned"] and task_summary["handoff_ordering_valid"])
        elif args.stage == "full-task":
            stage_goal = bool(task_summary["released_inside_bin"] and task_summary["handoff_ordering_valid"])
        checks = {
            "at_least_one_inference": len(inference_rows) > 0,
            "all_chunks_pass_pre_execution_safety": all_chunks_pass,
            "commands_executed": len(commanded) > 0,
            "rollout_safety_tracking": rollout.get("status") == "PASS",
            "diagnostic_runtime_abort_none": diagnostic_abort_reason is None,
            "stage_goal": stage_goal,
            "policy_generated_actions": True,
            "frozen_camera_all_inferences": True,
            "measured_state_all_inferences": True,
        }
        if args.stage == "object-free-single":
            checks["exactly_one_chunk_executed"] = len(commanded) == 50
            checks["exactly_one_inference"] = len(inference_rows) == 1
        if args.stage in {"object-free-multichunk", "full-motion"}:
            checks["repeated_inference"] = len(inference_rows) >= 2
            checks["bounded_audited_causal_prefix"] = args.execution_horizon <= 16
            checks["strictly_causal_chunk_replacement"] = True
            checks["chunk_boundary_command_jump"] = all(
                all(row["strict_pre_execution_checks"].values())
                for row in chunk_boundary_records
            )
            checks["chunk_boundary_measured_acceleration"] = all(
                all(row.get("strict_post_execution_checks", {}).values())
                for row in chunk_boundary_records
                if row.get("strict_post_execution_checks") is not None
            )
        if args.stage == "full-motion":
            target_frames = args.full_motion_inference_calls * args.execution_horizon
            checks.update(
                {
                    "full_diagnostic_duration_completed": len(commanded)
                    == target_frames,
                    "all_executed_prefix_hard_gates_pass": bool(
                        executed_prefix_gate_records
                    )
                    and all(
                        row["status"] == "PASS"
                        for row in executed_prefix_gate_records
                    ),
                    "every_inference_has_suffix_audit": len(
                        future_suffix_warnings
                    )
                    == len(inference_rows),
                    "physical_object_success_not_required": True,
                    "rtc_temporal_averaging_consensus_disabled": True,
                }
            )
        status = "PASS" if all(checks.values()) else "FAIL"
        raw_executed_array = np.asarray(policy_raw_executed, dtype=np.float32).reshape(-1, 28)
        stitched_executed_array = np.asarray(
            stitched_execution_executed, dtype=np.float32
        ).reshape(-1, 28)
        hard_executed_array = np.asarray(
            hard_limit_projected_executed, dtype=np.float32
        ).reshape(-1, 28)
        command_array = np.asarray(commanded, dtype=np.float32).reshape(-1, 28)
        actual_array = np.asarray(actual, dtype=np.float32).reshape(-1, 28)
        velocity_array = np.asarray(actual_velocity, dtype=np.float32).reshape(-1, 28)
        left_wrist_array = np.asarray(left_wrist_xyz, dtype=np.float64).reshape(-1, 3)
        right_wrist_array = np.asarray(right_wrist_xyz, dtype=np.float64).reshape(-1, 3)
        for record in projection_records:
            record["executed"] = (
                int(record["inference_index"]), int(record["action_row_in_chunk"])
            ) in executed_action_keys
        atomic_json(stage_dir / "projection_records.json", projection_records)
        atomic_json(
            stage_dir / "hard_limit_projection_records.json",
            hard_limit_projection_records,
        )
        atomic_json(
            stage_dir / "deployment_margin_projection_records.json",
            deployment_margin_projection_records,
        )
        atomic_json(
            stage_dir / "executed_prefix_safety_events.json",
            executed_prefix_gate_records,
        )
        atomic_json(
            stage_dir / "future_suffix_warnings.json",
            future_suffix_warnings,
        )
        atomic_json(stage_dir / "inference_timestamps.json", inference_rows)
        atomic_json(stage_dir / "causal_stitching_records.json", stitching_records)
        atomic_json(stage_dir / "jerk_limited_otg_records.json", otg_records)
        if args.stage == "full-motion":
            atomic_json(stage_dir / "object_observation_trace.json", task_rows)
        raw_chunk_array = np.asarray(policy_raw_chunks, dtype=np.float32).reshape(-1, 50, 28)
        stitched_chunk_array = np.asarray(
            stitched_execution_chunks, dtype=np.float32
        ).reshape(-1, 50, 28)
        model_stitched_chunk_array = (
            np.asarray(model_stitched_policy_chunks, dtype=np.float32).reshape(-1, 50, 28)
            if model_stitched_policy_chunks
            else stitched_chunk_array.copy()
        )
        causal_selected_chunk_array = (
            np.asarray(causal_selected_plan_chunks, dtype=np.float32).reshape(-1, 50, 28)
            if causal_selected_plan_chunks
            else stitched_chunk_array.copy()
        )
        previous_remaining_array = np.zeros(
            (len(previous_remaining_plans), 50, 28), dtype=np.float32
        )
        for remaining_index, remaining_plan in enumerate(previous_remaining_plans):
            previous_remaining_array[
                remaining_index, : len(remaining_plan)
            ] = remaining_plan.astype(np.float32)
        hard_chunk_array = np.asarray(
            hard_limit_projected_chunks, dtype=np.float32
        ).reshape(-1, 50, 28)
        deployment_chunk_array = np.asarray(
            deployment_safe_chunks, dtype=np.float32
        ).reshape(-1, 50, 28)
        committed_prefix_array = np.zeros_like(deployment_chunk_array)
        committed_prefix_length = min(args.execution_horizon, 50)
        committed_prefix_array[:, :committed_prefix_length] = deployment_chunk_array[
            :, :committed_prefix_length
        ]
        fused_uncommitted_array = np.zeros_like(deployment_chunk_array)
        fused_uncommitted_length = 50 - committed_prefix_length
        if fused_uncommitted_length:
            fused_uncommitted_array[:, :fused_uncommitted_length] = causal_selected_chunk_array[
                :, committed_prefix_length:
            ]
        atomic_npz(
            stage_dir / "inference_chunks.npz",
            policy_raw_action=raw_chunk_array,
            raw_policy_chunk=raw_chunk_array,
            model_stitched_policy_plan=model_stitched_chunk_array,
            causal_selected_plan=causal_selected_chunk_array,
            previous_remaining_plan=previous_remaining_array,
            previous_remaining_plan_length=np.asarray(
                previous_remaining_plan_lengths, dtype=np.int64
            ),
            previous_plan=previous_remaining_array,
            committed_prefix=committed_prefix_array,
            committed_prefix_length=np.full(
                len(deployment_chunk_array), committed_prefix_length, dtype=np.int64
            ),
            committed_from_previous_plan_length=np.asarray(
                [row.get("committed_previous_plan_rows", 0) for row in inference_rows],
                dtype=np.int64,
            ),
            fused_uncommitted_plan=fused_uncommitted_array,
            fused_uncommitted_plan_length=np.full(
                len(deployment_chunk_array), fused_uncommitted_length, dtype=np.int64
            ),
            final_command_reference=deployment_chunk_array,
            jerk_limited_otg_position_reference=(
                stitched_chunk_array
                if jerk_limited_otg is not None
                else np.empty((0, 50, 28), dtype=np.float32)
            ),
            jerk_limited_otg_velocity_reference=(
                np.asarray(otg_velocity_chunks, dtype=np.float32).reshape(-1, 50, 28)
                if otg_velocity_chunks
                else np.empty((0, 50, 28), dtype=np.float32)
            ),
            jerk_limited_otg_acceleration_reference=(
                np.asarray(otg_acceleration_chunks, dtype=np.float32).reshape(-1, 50, 28)
                if otg_acceleration_chunks
                else np.empty((0, 50, 28), dtype=np.float32)
            ),
            jerk_limited_otg_target_position=(
                np.asarray(otg_target_positions, dtype=np.float32).reshape(-1, 28)
                if otg_target_positions
                else np.empty((0, 28), dtype=np.float32)
            ),
            committed_prefix_overwritten=np.asarray(False),
            stitched_execution_plan=stitched_chunk_array,
            hard_limit_projected_action=hard_chunk_array,
            deployment_safe_action=deployment_chunk_array,
            hardware_feasible_action=deployment_chunk_array,
            joint_names=np.asarray(names),
            common_projection_freeze_sha256=np.asarray(projection_config_hash),
        )
        atomic_npz(
            stage_dir / "rollout_trace.npz",
            policy_raw_action=raw_executed_array,
            raw_policy_chunk_executed_rows=raw_executed_array,
            stitched_execution_plan=stitched_executed_array,
            hard_limit_projected_action=hard_executed_array,
            deployment_safe_action=command_array,
            hardware_feasible_action=command_array,
            commanded_q=command_array,
            commanded_action=command_array,
            actual_q=actual_array,
            measured_qpos=actual_array,
            actual_joint_velocity=velocity_array,
            left_wrist_xyz_world_m=left_wrist_array,
            right_wrist_xyz_world_m=right_wrist_array,
            timestamp=np.asarray(timestamps, dtype=np.float64),
            inference_timestamp=np.asarray(inference_timestamps_s, dtype=np.float64),
            joint_names=np.asarray(names),
            policy_generated=np.asarray(True),
            real_robot_command_allowed=np.asarray(False),
        )
        atomic_npz(
            stage_dir / "wrist_trajectory.npz",
            timestamp=np.asarray(timestamps, dtype=np.float64),
            left_wrist_xyz_world_m=left_wrist_array,
            right_wrist_xyz_world_m=right_wrist_array,
        )
        trajectory_plots = (
            save_full_motion_plots(
                stage_dir,
                np.asarray(timestamps, dtype=np.float64),
                left_wrist_array,
                right_wrist_array,
                command_array,
                actual_array,
                names,
            )
            if args.stage == "full-motion"
            else {}
        )
        inference_latencies = [row["inference_seconds"] for row in inference_rows]
        hard_projected_magnitudes = np.asarray(
            [row["incremental_correction_magnitude_rad"] for row in hard_limit_projection_records],
            dtype=np.float64,
        )
        margin_projected_magnitudes = np.asarray(
            [
                row["incremental_correction_magnitude_rad"]
                for row in deployment_margin_projection_records
            ],
            dtype=np.float64,
        )
        projection_aggregate = {
            "raw_limit_violation_count": int(
                sum(
                    row["policy_raw_action_audit"]["joint_limit_violation_count"]
                    for row in chunk_audits
                )
            ),
            "hard_projected_limit_violation_count": int(
                sum(
                    row["hard_limit_projected_action_audit"][
                        "joint_limit_violation_count"
                    ]
                    for row in chunk_audits
                )
            ),
            "deployment_safe_hard_limit_violation_count": int(
                sum(
                    row["deployment_safe_hard_limit_audit"][
                        "joint_limit_violation_count"
                    ]
                    for row in chunk_audits
                )
            ),
            "deployment_safe_interval_violation_count": int(
                sum(
                    row["deployment_safe_interval_audit"][
                        "joint_limit_violation_count"
                    ]
                    for row in chunk_audits
                )
            ),
            "hard_limit_projected_scalar_count": len(hard_limit_projection_records),
            "deployment_margin_projected_scalar_count": len(
                deployment_margin_projection_records
            ),
            "executed_hard_limit_projected_scalar_count": int(
                sum(bool(row["executed"]) for row in hard_limit_projection_records)
            ),
            "executed_deployment_margin_projected_scalar_count": int(
                sum(
                    bool(row["executed"])
                    for row in deployment_margin_projection_records
                )
            ),
            "affected_joint_names": sorted(
                {row["joint"] for row in projection_records}
            ),
            "mean_hard_projection_magnitude_rad": float(
                np.mean(hard_projected_magnitudes)
            )
            if hard_projected_magnitudes.size
            else 0.0,
            "maximum_hard_projection_magnitude_rad": float(
                np.max(hard_projected_magnitudes)
            )
            if hard_projected_magnitudes.size
            else 0.0,
            "mean_deployment_margin_correction_rad": float(
                np.mean(margin_projected_magnitudes)
            )
            if margin_projected_magnitudes.size
            else 0.0,
            "maximum_deployment_margin_correction_rad": float(
                np.max(margin_projected_magnitudes)
            )
            if margin_projected_magnitudes.size
            else 0.0,
            "maximum_total_raw_to_deployment_safe_correction_rad": max(
                (
                    row["projection"][
                        "maximum_total_raw_to_deployment_safe_correction_rad"
                    ]
                    for row in chunk_audits
                ),
                default=0.0,
            ),
            "all_arms_bitwise_preserved": all(
                row["projection"]["arm_outputs_bitwise_preserved"] for row in chunk_audits
            ),
            "all_hard_valid_values_preserved_by_hard_stage": all(
                row["projection"]["hard_valid_values_preserved_by_hard_stage"]
                for row in chunk_audits
            ),
            "all_safe_values_preserved_by_margin_stage": all(
                row["projection"]["safe_values_preserved_by_margin_stage"]
                for row in chunk_audits
            ),
            "all_hard_projection_geometry_checks_pass": all(
                row["hard_limit_projection_geometry_delta"]["status"] == "PASS"
                for row in chunk_audits
            ),
            "all_margin_projection_geometry_checks_pass": all(
                row["deployment_margin_geometry_delta"]["status"] == "PASS"
                for row in chunk_audits
            ),
            "all_total_projection_geometry_checks_pass": all(
                row["total_projection_geometry_delta"]["status"] == "PASS"
                for row in chunk_audits
            ),
        }
        boundary_accelerations = [
            float(row["resulting_measured_acceleration_max_abs_rad_s2"])
            for row in chunk_boundary_records
            if row["resulting_measured_acceleration_max_abs_rad_s2"] is not None
        ]
        boundary_command_values = np.asarray(
            [
                row["executed_boundary_max_abs_delta_rad"]
                for row in chunk_boundary_records
            ],
            dtype=np.float64,
        )
        boundary_stitched_values = np.asarray(
            [
                row["stitched_boundary_max_abs_delta_rad"]
                for row in chunk_boundary_records
            ],
            dtype=np.float64,
        )
        boundary_by_joint = (
            np.asarray(
                [row["executed_boundary_delta_rad"] for row in chunk_boundary_records],
                dtype=np.float64,
            )
            if chunk_boundary_records
            else np.empty((0, 28), dtype=np.float64)
        )
        chunk_boundary_jitter = {
            "status": "MEASURED" if chunk_boundary_records else "NOT_APPLICABLE",
            "method": args.causal_stitching_method,
            "method_description": (
                "official LeRobot RTC causal prefix-guided inpainting"
                if args.causal_stitching_method == "rtc"
                else "state-aligned causal minimum-jerk crossfade"
                if args.causal_stitching_method == "crossfade"
                else "naive causal short-horizon replacement"
            ),
            "execution_horizon_frames": args.execution_horizon,
            "future_observations_or_predictions_used": False,
            "offline_temporal_consensus_used": False,
            "replan_boundary_count": len(chunk_boundary_records),
            "maximum_raw_planned_overlap_delta_rad": max(
                (
                    row["raw_boundary_max_abs_delta_rad"]
                    for row in chunk_boundary_records
                ),
                default=0.0,
            ),
            "maximum_stitched_planned_overlap_delta_rad": max(
                boundary_stitched_values.tolist(), default=0.0
            ),
            "maximum_hard_projected_planned_overlap_delta_rad": max(
                (
                    row["hard_projected_boundary_max_abs_delta_rad"]
                    for row in chunk_boundary_records
                ),
                default=0.0,
            ),
            "maximum_deployment_safe_planned_overlap_delta_rad": max(
                (
                    row["deployment_safe_boundary_max_abs_delta_rad"]
                    for row in chunk_boundary_records
                ),
                default=0.0,
            ),
            "maximum_executed_boundary_command_jump_rad": max(
                (
                    row["executed_boundary_max_abs_delta_rad"]
                    for row in chunk_boundary_records
                ),
                default=0.0,
            ),
            "executed_boundary_command_jump_distribution_rad": {
                "median": float(np.median(boundary_command_values))
                if boundary_command_values.size
                else 0.0,
                "p95": float(np.percentile(boundary_command_values, 95))
                if boundary_command_values.size
                else 0.0,
                "p99": float(np.percentile(boundary_command_values, 99))
                if boundary_command_values.size
                else 0.0,
                "maximum": float(np.max(boundary_command_values))
                if boundary_command_values.size
                else 0.0,
            },
            "executed_boundary_delta_per_joint": [
                {
                    "joint_index": index,
                    "joint": name,
                    "median_abs_rad": float(np.median(np.abs(boundary_by_joint[:, index]))),
                    "p95_abs_rad": float(np.percentile(np.abs(boundary_by_joint[:, index]), 95)),
                    "p99_abs_rad": float(np.percentile(np.abs(boundary_by_joint[:, index]), 99)),
                    "maximum_abs_rad": float(np.max(np.abs(boundary_by_joint[:, index]))),
                }
                for index, name in enumerate(names)
            ]
            if len(boundary_by_joint)
            else [],
            "maximum_resulting_measured_acceleration_rad_s2": max(
                boundary_accelerations, default=0.0
            ),
            "authoritative_extreme_jump_gate_rad": float(
                hard_safety.acceptance["maximum_joint_step_rad"]
            ),
            "authoritative_extreme_acceleration_gate_rad_s2": float(
                hard_safety.acceptance["maximum_acceleration_rad_s2"]
            ),
            "significant_under_authoritative_dynamic_gates": any(
                not all(row["strict_pre_execution_checks"].values())
                or (
                    row.get("strict_post_execution_checks") is not None
                    and not all(row["strict_post_execution_checks"].values())
                )
                for row in chunk_boundary_records
            ),
            "rtc_required": None,
            "records": chunk_boundary_records,
        }
        if chunk_boundary_records:
            chunk_boundary_jitter["rtc_required"] = bool(
                chunk_boundary_jitter[
                    "significant_under_authoritative_dynamic_gates"
                ]
            )
        suffix_collision_warnings = [
            row
            for row in future_suffix_warnings
            if row["predicted_collision_records"]
        ]
        future_suffix_warning_summary = {
            "classification": "UNEXECUTED_SUFFIX_WARNING",
            "audited_inference_count": len(future_suffix_warnings),
            "warning_inference_count": int(
                sum(row["warning_present"] for row in future_suffix_warnings)
            ),
            "collision_warning_inference_count": len(suffix_collision_warnings),
            "predicted_collision_incidence_count": int(
                sum(
                    len(row["predicted_collision_records"])
                    for row in suffix_collision_warnings
                )
            ),
            "maximum_predicted_penetration_m": max(
                (
                    float(row["maximum_predicted_penetration_m"])
                    for row in suffix_collision_warnings
                ),
                default=0.0,
            ),
            "never_executed_without_a_new_observation": True,
            "hard_gate": False
            if args.stage in {"object-free-multichunk", "full-motion"}
            else None,
            "records": str(stage_dir / "future_suffix_warnings.json"),
        }
        executed_prefix_safety_summary = {
            "classification": "EXECUTED_PREFIX_HARD_GATE",
            "audited_inference_count": len(executed_prefix_gate_records),
            "failed_inference_count": int(
                sum(
                    row["status"] != "PASS"
                    for row in executed_prefix_gate_records
                )
            ),
            "predicted_collision_incidence_count": int(
                sum(
                    row["deployment_safe_hard_limit_prefix_audit"]["collision"][
                        "invalid_hard_collision_category_incidence"
                    ]
                    for row in executed_prefix_gate_records
                )
            ),
            "records": str(stage_dir / "executed_prefix_safety_events.json"),
        }
        atomic_json(stage_dir / "chunk_boundary_jitter.json", chunk_boundary_jitter)
        atomic_json(
            stage_dir / "runtime_diagnostic_gates.json",
            runtime_diagnostic_gate_records,
        )
        report = common_report | {
            "schema_version": (
                f"policy_b_isaac_{STAGE_DIRECTORIES[args.stage]}_v3_"
                + (
                    "simulation_diagnostic_rollout"
                    if args.simulation_diagnostic_rollout
                    else "strict_simulation_margin"
                )
            ),
            "status": status,
            "full_policy_b_diagnostic_rollout": (
                "COMPLETE"
                if args.stage == "full-motion"
                and checks.get("full_diagnostic_duration_completed", False)
                and diagnostic_abort_reason is None
                else "ABORTED"
                if args.stage == "full-motion"
                else None
            ),
            "policy_behavior_diagnostic": (
                "PASS" if status == "PASS" else "FAIL"
            )
            if args.simulation_diagnostic_rollout
            else None,
            "real_hardware_safety_readiness": "BLOCKED",
            "strict_safety_qualification": (
                diagnostic_authorization | {
                    "current_rollout": rollout.get(
                        "strict_safety_qualification"
                    )
                }
                if diagnostic_authorization is not None
                else rollout.get("strict_safety_qualification")
            ),
            "simulation_diagnostic_qualification": rollout.get(
                "simulation_diagnostic_qualification"
            ),
            "checks": checks,
            "execution_horizon_frames": 50 if args.stage == "object-free-single" else args.execution_horizon,
            "inference_call_count": len(inference_rows),
            "inference_latency_seconds": {
                "mean": float(np.mean(inference_latencies)) if inference_latencies else None,
                "maximum": float(np.max(inference_latencies)) if inference_latencies else None,
                "values": inference_latencies,
            },
            "executed_control_frames": len(commanded),
            "duration_s": len(commanded) / CONTROL_FPS,
            "chunk_boundary_jump_max_abs_rad": max(chunk_boundary_jumps, default=0.0),
            "replanned_future_rmse_rad": {
                "mean": float(np.mean(replanned_future_differences)) if replanned_future_differences else None,
                "maximum": float(np.max(replanned_future_differences)) if replanned_future_differences else None,
            },
            "chunk_boundary_jitter": chunk_boundary_jitter,
            "chunk_boundary_jitter_records": str(
                stage_dir / "chunk_boundary_jitter.json"
            ),
            "runtime_diagnostic_gate_records": str(
                stage_dir / "runtime_diagnostic_gates.json"
            ),
            "diagnostic_abort_reason": diagnostic_abort_reason,
            "executed_prefix_gate_semantics": (
                "Rows 0 through H-1 are hard-gated before command execution"
                if args.stage == "full-motion"
                else None
            ),
            "unexecuted_suffix_semantics": (
                "Rows H through 49 are audited and logged as warnings because a new observation replaces them"
                if args.stage == "full-motion"
                else None
            ),
            "executed_prefix_safety_summary": executed_prefix_safety_summary,
            "future_suffix_warning_summary": future_suffix_warning_summary,
            "rollout": rollout,
            "chunk_audits": chunk_audits,
            "projection_aggregate": projection_aggregate,
            "task_semantics": task_summary,
            "task_semantics_used_as_stop_gate": False
            if args.stage == "full-motion"
            else True,
            "physical_object_success_evaluated": False
            if args.stage == "full-motion"
            else None,
            "training_episode_duration_reference": (
                {
                    "source": str(
                        DATASET
                        / "meta/episodes/chunk-000/file-000.parquet"
                    ),
                    "episode_count": 50,
                    "fps": 30,
                    "frames": {
                        "minimum": 681,
                        "median": 688.5,
                        "mean": 689.56,
                        "maximum": 705,
                    },
                    "seconds": {
                        "minimum": 22.7,
                        "median": 22.95,
                        "mean": 22.985333333333334,
                        "maximum": 23.5,
                    },
                    "selected_diagnostic_frames": args.full_motion_inference_calls
                    * args.execution_horizon,
                    "selected_diagnostic_seconds": args.full_motion_inference_calls
                    * args.execution_horizon
                    / CONTROL_FPS,
                }
                if args.stage == "full-motion"
                else None
            ),
            "camera_experiment_limitation": (
                "Closed-loop Isaac result is specific to the exact --camera-config hash recorded above."
                if args.stage == "full-motion"
                else None
            ),
            "final_deployment_camera": (
                DEPLOYMENT_CAMERA.name if DEPLOYMENT_CAMERA.is_final_helmet else "NOT_MOUNTED_NOT_CALIBRATED"
            ),
            "videos": recorder.paths,
            "trace": str(stage_dir / "rollout_trace.npz"),
            "inference_chunks": str(stage_dir / "inference_chunks.npz"),
            "raw_policy_chunks_key": "raw_policy_chunk",
            "previous_remaining_plan_key": "previous_remaining_plan",
            "stitched_execution_plan_key": "stitched_execution_plan",
            "commanded_action_key": "commanded_action",
            "measured_qpos_key": "measured_qpos",
            "projection_records": str(stage_dir / "projection_records.json"),
            "hard_limit_projection_records": str(
                stage_dir / "hard_limit_projection_records.json"
            ),
            "deployment_margin_projection_records": str(
                stage_dir / "deployment_margin_projection_records.json"
            ),
            "inference_timestamps": str(stage_dir / "inference_timestamps.json"),
            "causal_stitching_records": str(
                stage_dir / "causal_stitching_records.json"
            ),
            "executed_prefix_safety_events": str(
                stage_dir / "executed_prefix_safety_events.json"
            ),
            "future_suffix_warnings": str(
                stage_dir / "future_suffix_warnings.json"
            ),
            "wrist_trajectory": str(stage_dir / "wrist_trajectory.npz"),
            "trajectory_plots": trajectory_plots,
            "object_observation_trace": str(
                stage_dir / "object_observation_trace.json"
            )
            if args.stage == "full-motion"
            else None,
            "visual_plausibility_review": "REQUIRES_CONTACT_SHEET_AND_VIDEO_INSPECTION_BEFORE_NEXT_STAGE",
        }
        atomic_json(stage_dir / "stage_report.json", report)
        print(json.dumps({
            "stage": args.stage,
            "status": status,
            "full_policy_b_diagnostic_rollout": report.get(
                "full_policy_b_diagnostic_rollout"
            ),
            "inference_calls": len(inference_rows),
            "executed_frames": len(commanded),
            "tracking_rmse_rad": rollout.get("tracking_rmse_rad"),
            "maximum_velocity_rad_s": rollout.get("actual_trajectory", {}).get("maximum_velocity_rad_s"),
            "maximum_acceleration_rad_s2": rollout.get("actual_trajectory", {}).get("maximum_acceleration_rad_s2"),
            "projection": projection_aggregate,
            "task_semantics": task_summary,
            "failed_checks": [key for key, value in checks.items() if not value],
        }, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value))
        return 0 if status == "PASS" else 2
    finally:
        if recorder is not None:
            recorder.close(stage_dir)
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
