#!/usr/bin/env python3
"""Build a bounded handoff backward from the validated RIGHT P14 grasp.

The doll, controller, bilateral single-hand P14 endpoints, arm solver, and
safety contract are immutable.  The validated standalone RIGHT grasp defines
an object-relative terminal tool transform.  Each declared gate candidate is
one early point on a single minimum-jerk path from a collision-free preshape
to that transform.  At that point, one shared handoff-only 7D acquisition
posture is solved kinematically so the three distal pads enclose the doll while
the LEFT thumb remains in its frozen P14 support state.

Gate commands stop before LEFT-thumb release.  Full commands may be built only
after an external physics gate has validated the candidate; their metadata
requires the Isaac runner to verify sustained RIGHT three-digit support before
executing LEFT-thumb release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import find_g1_dex3_static_phone_grasp as old  # noqa: E402
import refine_g1_dex3_static_phone_contact as contact  # noqa: E402
from tools.build_doll_handoff_proxy_v2_handoff_gate import (  # noqa: E402
    canonical_row,
    hand_model,
    minimum_jerk,
    preliminary_candidate_audit,
    rotations_between,
    solve_bounded_pose,
    solve_path,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import (  # noqa: E402
    AUTHORITATIVE_REFERENCES,
    authoritative_joint_ranges,
)


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
FREEZE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "frozen_p14_bilateral/FREEZE_MANIFEST.json"
)
BASE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "scripted_full_task/p14_bilateral"
)
DEFAULT_OUTPUT = BASE / "backward_constructed_handoff"
SEQUENCE_REVISION = "gated_left_radial_exit_with_backward_endpoint_v15"
LEFT_INDEX_MIDDLE_RELAXATION_FRACTION = 0.15
LEFT_OBJECT_RADIAL_CLEARANCE_M = 0.045
LEFT_RELEASE_COMPLETION_FRACTIONS = (0.60, 0.40, 0.25)

# One bounded path family.  The scalar is path progress before ownership
# transfer, not an independent static wrist pose.  B2 is evaluated first by a
# predeclared rule: it maximizes the static three-pad enclosure margin while
# remaining clear of the LEFT support hand in the initial audit.
CANDIDATES: dict[str, float] = {
    "B1_PATH_F35": 0.35,
    "B2_PATH_F40": 0.40,
    "B3_PATH_F45": 0.45,
}
EVALUATION_ORDER = ("B2_PATH_F40", "B1_PATH_F35", "B3_PATH_F45")
PREPOSE_TRANSLATION_FROM_ENDPOINT_M = np.asarray([0.015, 0.0, 0.005])
PREPOSE_YAW_FROM_ENDPOINT_DEG = 50.0
STATIC_CONTACT_TARGET_M = -0.0002


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray)
            else value.item()
            if isinstance(value, np.generic)
            else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_freeze(freeze: dict[str, Any], config: dict[str, Any]) -> None:
    if freeze.get("status") != "FROZEN_BILATERAL_FULL_GRASP_PASS":
        raise RuntimeError("P14 bilateral freeze is not active")
    for key, row in freeze["hashes"].items():
        path = (FREEZE.parent / row["path"]).resolve()
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"frozen dependency changed: {key}: {path}")
    for side in ("left", "right"):
        expected = freeze["p14"][f"{side}_7d_rad"]
        actual = config["hand_states"][side]["POWER_GRASP_P14"]
        if actual != expected:
            raise RuntimeError(f"frozen {side} P14 endpoint changed")


def scalar_minimum_jerk(value: float) -> float:
    return float(10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5)


class StaticProxyModel:
    """Cached MuJoCo narrow-phase model for one fixed handoff doll pose."""

    def __init__(
        self,
        g1: G1Kinematics,
        object_center_world: np.ndarray,
        dimensions_m: np.ndarray,
    ) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="p14_backward_handoff_"
        )
        base = mujoco.MjModel.from_xml_path(str(g1.path))
        xml_path = Path(self._temporary.name) / "model.xml"
        mujoco.mj_saveLastXML(str(xml_path), base)
        text = xml_path.read_text(encoding="utf-8").replace(
            'meshdir="assets/"', f'meshdir="{g1.path.parent / "assets"}/"'
        )
        center = g1.world_to_model_position(object_center_world)
        radii = np.asarray(dimensions_m, dtype=np.float64) / 2.0
        body = (
            f'<body name="backward_handoff_proxy" '
            f'pos="{center[0]} {center[1]} {center[2]}">'
            f'<geom name="backward_handoff_proxy_geom" type="ellipsoid" '
            f'size="{radii[0]} {radii[1]} {radii[2]}" '
            'contype="1" conaffinity="1"/>'
            "</body>"
        )
        xml_path.write_text(text.replace("<worldbody>", "<worldbody>" + body, 1))
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.arm_qpos = np.asarray(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                ]
                for name in g1.arm_joint_names
            ],
            dtype=np.int64,
        )
        self.hand_qpos = {
            side: np.asarray(
                [
                    self.model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                        )
                    ]
                    for name in g1.hand_joint_names[side]
                ],
                dtype=np.int64,
            )
            for side in ("left", "right")
        }
        self.proxy = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "backward_handoff_proxy_geom",
        )
        self.right_distal = {
            digit: contact.collision_geoms(
                self.model,
                f"right_hand_{digit}_{'2_link' if digit == 'thumb' else '1_link'}",
            )[-1]
            for digit in ("thumb", "index", "middle")
        }
        hand_geoms: dict[str, list[tuple[int, str]]] = {"left": [], "right": []}
        for geom_id in range(self.model.ngeom):
            body_name = old.body_name(self.model, geom_id)
            for side in hand_geoms:
                if (
                    body_name.startswith(f"{side}_hand")
                    or body_name.startswith(f"{side}_wrist")
                ) and (
                    self.model.geom_contype[geom_id]
                    or self.model.geom_conaffinity[geom_id]
                ):
                    hand_geoms[side].append((geom_id, body_name))
        self.cross_pairs = [
            (left_geom, right_geom, left_body, right_body)
            for left_geom, left_body in hand_geoms["left"]
            for right_geom, right_body in hand_geoms["right"]
        ]

    def evaluate(
        self, arm: np.ndarray, left: np.ndarray, right: np.ndarray
    ) -> dict[str, Any]:
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qpos[self.arm_qpos] = arm
        self.data.qpos[self.hand_qpos["left"]] = left
        self.data.qpos[self.hand_qpos["right"]] = right
        mujoco.mj_forward(self.model, self.data)
        distances = {
            digit: float(
                contact.distance(self.model, self.data, geom, self.proxy)[0]
            )
            for digit, geom in self.right_distal.items()
        }
        cross = [
            (
                float(contact.distance(self.model, self.data, left, right)[0]),
                left_name,
                right_name,
            )
            for left, right, left_name, right_name in self.cross_pairs
        ]
        closest = min(cross, key=lambda row: row[0])
        return {
            "right_distal_proxy_signed_distance_m": distances,
            "minimum_left_right_clearance_m": closest[0],
            "closest_left_right_body_pair": [closest[1], closest[2]],
        }


def solve_acquisition_hand(
    evaluator: StaticProxyModel,
    arm: np.ndarray,
    left_support: np.ndarray,
    right_preshape: np.ndarray,
    right_p14: np.ndarray,
    right_limits: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    lower = np.maximum(
        right_limits[:, 0] + 1.0e-7,
        np.minimum(right_preshape, right_p14) - 0.25,
    )
    upper = np.minimum(
        right_limits[:, 1] - 1.0e-7,
        np.maximum(right_preshape, right_p14) + 0.25,
    )

    def residual(active: np.ndarray) -> np.ndarray:
        audit = evaluator.evaluate(arm, left_support, active)
        distances = audit["right_distal_proxy_signed_distance_m"]
        clearance = float(audit["minimum_left_right_clearance_m"])
        return np.r_[
            400.0
            * np.asarray(
                [
                    distances["thumb"] - STATIC_CONTACT_TARGET_M,
                    distances["index"] - STATIC_CONTACT_TARGET_M,
                    distances["middle"] - STATIC_CONTACT_TARGET_M,
                ]
            ),
            20.0 * max(0.0, 0.001 - clearance),
            0.03 * (active - right_p14),
        ]

    solution = least_squares(
        residual,
        np.clip(right_p14, lower, upper),
        bounds=(lower, upper),
        max_nfev=900,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
    )
    result = np.asarray(solution.x, dtype=np.float64)
    audit = evaluator.evaluate(arm, left_support, result)
    report = {
        "success": bool(solution.success),
        "status": int(solution.status),
        "cost": float(solution.cost),
        "nfev": int(solution.nfev),
        "acquisition_7d_rad": result,
        "delta_from_frozen_p14_rad": result - right_p14,
        "bounds_lower_rad": lower,
        "bounds_upper_rad": upper,
        **audit,
    }
    return result, report


def save_command(
    path: Path,
    commands: np.ndarray,
    stages: np.ndarray,
    names: list[str],
    fps: float,
    candidate: str,
    handoff_object: np.ndarray,
    bin_object: np.ndarray,
    full: bool,
) -> None:
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            commanded_q_rad=commands.astype(np.float32),
            stage=stages,
            joint_names=np.asarray(names),
            control_fps_hz=np.asarray(fps),
            candidate_id=np.asarray(candidate),
            handoff_object_center_world_m=handoff_object.astype(np.float32),
            bin_object_target_world_m=bin_object.astype(np.float32),
            policy_independent=np.asarray(True),
            scripted_full_task=np.asarray(full),
            backward_constructed_handoff=np.asarray(True),
            runtime_right_three_digit_gate_required=np.asarray(full),
            right_three_digit_gate_minimum_s=np.asarray(0.5),
            left_release_permitted=np.asarray(full),
            attachment_used=np.asarray(False),
            object_follow_used=np.asarray(False),
        )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--mode", choices=("gate", "full"), default="gate")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--physics-gate-evidence",
        type=Path,
        help="Required in full mode; JSON must explicitly validate the candidate.",
    )
    parser.add_argument(
        "--left-release-completion-fraction",
        type=float,
        choices=LEFT_RELEASE_COMPLETION_FRACTIONS,
        default=LEFT_RELEASE_COMPLETION_FRACTIONS[0],
        help="Bounded timing profile on the fixed endpoint path.",
    )
    args = parser.parse_args()
    full = args.mode == "full"
    if full:
        if args.physics_gate_evidence is None:
            raise RuntimeError("full mode requires --physics-gate-evidence")
        gate_evidence = read_json(args.physics_gate_evidence.resolve())
        if (
            gate_evidence.get("status") != "PASS"
            or gate_evidence.get("candidate_id") != args.candidate
        ):
            raise RuntimeError("physics gate evidence is not a matching PASS")

    freeze = read_json(FREEZE)
    config = read_json(CONFIG)
    verify_freeze(freeze, config)
    # Keep the initial simultaneous-relaxation diagnostic immutable.  This
    # revision changes only support sequencing: establish RIGHT contact/load
    # while LEFT remains fully closed, then relax LEFT index+middle.
    timing_revision = (
        f"{SEQUENCE_REVISION}_rf"
        f"{int(round(100 * args.left_release_completion_fraction)):02d}"
    )
    candidate_dir = (
        args.output_root.resolve()
        / args.candidate
        / timing_revision
        / args.mode
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)

    with np.load(BASE / "scripted_full_task_command.npz", allow_pickle=False) as archive:
        base_commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        base_stages = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
        handoff_object = np.asarray(
            archive["handoff_object_center_world_m"], dtype=np.float64
        )
        bin_object = np.asarray(
            archive["bin_object_target_world_m"], dtype=np.float64
        )
    authoritative_names, _ = authoritative_joint_ranges()
    if (
        names != authoritative_names
        or base_commands.shape != (792, 28)
        or not np.isclose(fps, 30.0)
    ):
        raise RuntimeError("frozen 792-frame base command interface changed")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    left_open = hand_model(g1, names, config, "left", "OPEN")
    right_open = hand_model(g1, names, config, "right", "OPEN")
    right_preshape = hand_model(g1, names, config, "right", "PRESHAPE")
    left_p14 = hand_model(g1, names, config, "left", "POWER_GRASP_P14")
    right_p14 = hand_model(g1, names, config, "right", "POWER_GRASP_P14")
    p1 = {
        side: hand_model(g1, names, config, side, "POWER_GRASP_P1")
        for side in ("left", "right")
    }
    left_thumb_support = left_p14.copy()
    left_thumb_support[3:] = left_open[3:]

    primitives: dict[str, dict[str, np.ndarray]] = {}
    static_tool: dict[str, np.ndarray] = {}
    reference_rotation_world: dict[str, np.ndarray] = {}
    reference_arm: dict[str, np.ndarray] = {}
    origin = g1.model_to_world_position(np.zeros(3))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3)[axis]) - origin
            for axis in range(3)
        ]
    )
    for side in ("left", "right"):
        primitive_path = Path(config["source_arm_primitives"][side])
        with np.load(primitive_path, allow_pickle=False) as archive:
            primitives[side] = {
                key: np.asarray(archive[key]) for key in archive.files
            }
        reference_arm[side] = np.asarray(
            primitives[side]["lift_arm_q_rad"][-1], dtype=np.float64
        )
        g1.assign(reference_arm[side], p1["left"], p1["right"])
        static_tool[side] = (
            np.linalg.inv(g1.wrist_pose(side)) @ g1.whole_hand_grasp_pose(side)
        )
        g1.assign(reference_arm[side], left_p14, right_p14)
        _, rotation_model, _, _ = g1.static_tool_pose_state(
            side, static_tool[side]
        )
        reference_rotation_world[side] = world_from_model @ rotation_model

    # Reconstruct the frozen standalone object-to-tool transform from the
    # successful elevated hold; no grasp or retargeting quantity is recomputed.
    trial_root = (
        ROOT
        / "outputs/dex3_simple_graspable_doll_proxy_v1"
        / "hand_calibration_v2/p14_three_digit_preload/trials"
    )
    tool_offsets: dict[str, np.ndarray] = {}
    object_to_tool_translations: dict[str, np.ndarray] = {}
    object_to_tool_rotations: dict[str, np.ndarray] = {}
    standalone_evidence: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        result_path = trial_root / side / "trial_result.json"
        result = read_json(result_path)
        if result["status"] != "PASS" or not result["three_meaningful_digit_contacts"]:
            raise RuntimeError(f"frozen standalone {side} P14 evidence is not PASS")
        with np.load(result["event_log"], allow_pickle=False) as archive:
            mask = archive["stage"].astype(str) == "HOLD_ELEVATED"
            centers = np.asarray(
                archive["object_position_world_m"], dtype=np.float64
            )[mask]
            quaternions = np.asarray(
                archive["object_quaternion_xyzw"], dtype=np.float64
            )[mask]
        object_center = np.median(centers[-240:], axis=0)
        object_rotation_world = Rotation.from_quat(
            quaternions[-240:]
        ).mean().as_matrix()
        declared_tool = np.asarray(
            primitives[side]["target_whole_hand_position_world_m"][-1],
            dtype=np.float64,
        )
        tool_offsets[side] = declared_tool - object_center
        object_to_tool_translations[side] = (
            object_rotation_world.T @ tool_offsets[side]
        )
        object_to_tool_rotations[side] = (
            object_rotation_world.T @ reference_rotation_world[side]
        )
        standalone_evidence[side] = {
            "result": str(result_path),
            "result_sha256": sha256_file(result_path),
            "event_log": result["event_log"],
            "event_log_sha256": sha256_file(Path(result["event_log"])),
            "object_center_hold_world_m": object_center,
            "object_rotation_hold_world": object_rotation_world,
            "object_to_tool_translation_world_m": tool_offsets[side],
            "object_to_tool_translation_object_m": object_to_tool_translations[side],
            "object_to_tool_rotation_object": object_to_tool_rotations[side],
        }

    left_hold_rows = np.flatnonzero(base_stages == "LEFT_HANDOFF_HOLD")
    if len(left_hold_rows) != 15:
        raise RuntimeError("frozen LEFT_HANDOFF_HOLD prefix changed")
    prefix_end = int(left_hold_rows[-1] + 1)
    left_handoff_arm = np.asarray(
        base_commands[left_hold_rows[-1], arm_indices], dtype=np.float64
    )

    validated_position_world = handoff_object + tool_offsets["right"]
    validated_rotation_world = reference_rotation_world["right"]
    prepose_position_world = (
        validated_position_world + PREPOSE_TRANSLATION_FROM_ENDPOINT_M
    )
    prepose_rotation_world = (
        Rotation.from_euler(
            "z", PREPOSE_YAW_FROM_ENDPOINT_DEG, degrees=True
        ).as_matrix()
        @ validated_rotation_world
    )
    progress = CANDIDATES[args.candidate]
    blend = scalar_minimum_jerk(progress)
    candidate_position_world = prepose_position_world + blend * (
        validated_position_world - prepose_position_world
    )
    candidate_rotation_world = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(
            np.stack((prepose_rotation_world, validated_rotation_world))
        ),
    )([blend]).as_matrix()[0]

    # Collision-free approach from the frozen LEFT-held pose to the shared
    # preshape, followed by the single declared path toward the endpoint.
    g1.assign(left_handoff_arm, left_p14, right_open)
    start_position_model, start_rotation_model, _, _ = g1.static_tool_pose_state(
        "right", static_tool["right"]
    )
    start_position_world = g1.model_to_world_position(start_position_model)
    start_rotation_world = world_from_model @ start_rotation_model
    approach_count = 60
    approach_arm, approach_reports = solve_path(
        g1,
        "right",
        static_tool["right"],
        g1.world_to_model_position(
            minimum_jerk(
                start_position_world, prepose_position_world, approach_count
            )
        ),
        np.einsum(
            "ij,tjk->tik",
            world_from_model.T,
            rotations_between(
                start_rotation_world, prepose_rotation_world, approach_count
            ),
        ),
        left_handoff_arm,
        left_p14,
        right_open,
        clearance_target_m=0.001,
    )
    acquisition_count = 45
    acquisition_arm, acquisition_reports = solve_path(
        g1,
        "right",
        static_tool["right"],
        g1.world_to_model_position(
            minimum_jerk(
                prepose_position_world,
                candidate_position_world,
                acquisition_count,
            )
        ),
        np.einsum(
            "ij,tjk->tik",
            world_from_model.T,
            rotations_between(
                prepose_rotation_world,
                candidate_rotation_world,
                acquisition_count,
            ),
        ),
        approach_arm[-1],
        left_thumb_support,
        right_p14,
        clearance_target_m=0.001,
    )

    dimensions = np.asarray(
        config["geometry_candidates"][0]["dimensions_m"], dtype=np.float64
    )
    evaluator = StaticProxyModel(g1, handoff_object, dimensions)
    acquisition_hand, hand_report = solve_acquisition_hand(
        evaluator,
        acquisition_arm[-1],
        left_thumb_support,
        right_preshape,
        right_p14,
        np.asarray(g1.hand_limits["right"], dtype=np.float64),
    )

    rows: list[np.ndarray] = [row.copy() for row in base_commands[:prefix_end]]
    stages: list[str] = base_stages[:prefix_end].tolist()

    def append(arm: np.ndarray, left: np.ndarray, right: np.ndarray, stage: str) -> None:
        rows.append(canonical_row(names, g1, arm, left, right))
        stages.append(stage)

    for arm in approach_arm[1:]:
        append(arm, left_p14, right_open, "RIGHT_COLLISION_FREE_APPROACH")
    for right in minimum_jerk(right_open, right_preshape, 30)[1:]:
        append(approach_arm[-1], left_p14, right, "RIGHT_PRESHAPE")
    for _ in range(6):
        append(approach_arm[-1], left_p14, right_preshape, "RIGHT_PRESHAPE_HOLD")
    right_close = minimum_jerk(right_preshape, acquisition_hand, acquisition_count)
    for arm, right in zip(acquisition_arm[1:], right_close[1:], strict=True):
        append(arm, left_p14, right, "BACKWARD_ACQUISITION_PATH")
    # RIGHT is physically preloaded before any LEFT support is removed.  The
    # subsequent relaxation keeps the frozen LEFT P14 thumb active.
    for _ in range(15):
        append(
            acquisition_arm[-1],
            left_p14,
            acquisition_hand,
            "RIGHT_THREE_DIGIT_PRELOAD",
        )
    partial_left_relax = left_p14 + LEFT_INDEX_MIDDLE_RELAXATION_FRACTION * (
        left_thumb_support - left_p14
    )
    for left in minimum_jerk(left_p14, partial_left_relax, 30)[1:]:
        append(
            acquisition_arm[-1],
            left,
            acquisition_hand,
            "LEFT_INDEX_MIDDLE_PARTIAL_RELAX_AFTER_RIGHT_PRELOAD",
        )
    verification_frames = 18 if full else 30
    for _ in range(verification_frames):
        append(
            acquisition_arm[-1],
            partial_left_relax,
            acquisition_hand,
            "RIGHT_THREE_DIGIT_VERIFICATION",
        )

    tail_reports: list[dict[str, Any]] = []
    if full:
        # RIGHT has passed the physical three-digit gate at the acquisition
        # point.  Withdraw the residual LEFT support and follow the already-
        # validated 45 mm LEFT radial clearance while continuing the one
        # predeclared RIGHT path to the frozen standalone wrist endpoint.
        # Neither endpoint nor grasp changes; the released hand simply exits
        # the shared volume during ownership transfer.
        endpoint_count = 60
        endpoint_arm, endpoint_reports = solve_path(
            g1,
            "right",
            static_tool["right"],
            g1.world_to_model_position(
                minimum_jerk(
                    candidate_position_world,
                    validated_position_world,
                    endpoint_count,
                )
            ),
            np.einsum(
                "ij,tjk->tik",
                world_from_model.T,
                rotations_between(
                    candidate_rotation_world,
                    validated_rotation_world,
                    endpoint_count,
                ),
            ),
            acquisition_arm[-1],
            left_open,
            acquisition_hand,
            clearance_target_m=0.001,
        )
        left_handoff_tool = handoff_object + tool_offsets["left"]
        left_radial = left_handoff_tool - handoff_object
        left_radial /= np.linalg.norm(left_radial)
        left_local_clearance_tool = (
            left_handoff_tool + LEFT_OBJECT_RADIAL_CLEARANCE_M * left_radial
        )
        left_rotation_model = world_from_model.T @ reference_rotation_world["left"]
        left_local_clearance, left_local_clearance_reports = solve_path(
            g1,
            "left",
            static_tool["left"],
            g1.world_to_model_position(
                minimum_jerk(
                    left_handoff_tool,
                    left_local_clearance_tool,
                    endpoint_count,
                )
            ),
            np.repeat(left_rotation_model[None], endpoint_count, axis=0),
            acquisition_arm[-1],
            left_open,
            acquisition_hand,
        )
        combined_endpoint_arm = np.asarray(
            [
                np.r_[left_arm[:7], right_arm[7:14]]
                for left_arm, right_arm in zip(
                    left_local_clearance, endpoint_arm, strict=True
                )
            ],
            dtype=np.float64,
        )
        endpoint_progress = np.linspace(0.0, 1.0, endpoint_count)
        release_progress = np.clip(
            endpoint_progress / args.left_release_completion_fraction,
            0.0,
            1.0,
        )
        release_blend = np.asarray(
            [scalar_minimum_jerk(value) for value in release_progress]
        )
        endpoint_left_release = (
            partial_left_relax[None]
            + release_blend[:, None]
            * (left_open - partial_left_relax)[None]
        )
        for arm, left in zip(
            combined_endpoint_arm[1:], endpoint_left_release[1:], strict=True
        ):
            append(
                arm,
                left,
                acquisition_hand,
                "LEFT_THUMB_RELEASE",
            )
        for _ in range(30):
            append(
                combined_endpoint_arm[-1],
                left_open,
                acquisition_hand,
                "RIGHT_POST_RELEASE_RETENTION",
            )
        for _ in range(15):
            append(
                combined_endpoint_arm[-1],
                left_open,
                acquisition_hand,
                "RIGHT_ACQUISITION_HOLD_AFTER_LEFT_CLEARANCE",
            )

        # Ownership has transferred at the already-validated standalone wrist
        # endpoint.  Settle only the Dex3 joints into the immutable P14 vector;
        # no post-release wrist correction or object-relative feedback occurs.
        right_hold_arm = combined_endpoint_arm[-1].copy()
        right_settle_hand = minimum_jerk(acquisition_hand, right_p14, 45)
        for right in right_settle_hand[1:]:
            append(
                right_hold_arm,
                left_open,
                right,
                "RIGHT_SETTLE_TO_VALIDATED_P14",
            )
        for _ in range(30):
            append(right_hold_arm, left_open, right_p14, "RIGHT_OWNED_HOLD")

        left_retreat_world = left_handoff_tool + np.asarray([-0.10, -0.06, 0.04])
        left_retreat, left_reports = solve_path(
            g1,
            "left",
            static_tool["left"],
            g1.world_to_model_position(
                minimum_jerk(left_local_clearance_tool, left_retreat_world, 45)
            ),
            np.repeat(left_rotation_model[None], 45, axis=0),
            right_hold_arm,
            left_open,
            right_p14,
        )
        for arm in left_retreat[1:]:
            append(arm, left_open, right_p14, "LEFT_RETREAT_AFTER_RIGHT_P14")

        bin_tool = bin_object + tool_offsets["right"]
        right_rotation_model = world_from_model.T @ validated_rotation_world
        right_transport, right_transport_reports = solve_path(
            g1,
            "right",
            static_tool["right"],
            g1.world_to_model_position(
                minimum_jerk(validated_position_world, bin_tool, 75)
            ),
            np.repeat(right_rotation_model[None], 75, axis=0),
            left_retreat[-1],
            left_open,
            right_p14,
        )
        for arm in right_transport[1:]:
            append(arm, left_open, right_p14, "RIGHT_TRANSPORT_TO_BIN")
        for _ in range(15):
            append(right_transport[-1], left_open, right_p14, "RIGHT_HOLD_OVER_BIN")
        for right in minimum_jerk(right_p14, right_open, 30)[1:]:
            append(right_transport[-1], left_open, right, "RIGHT_RELEASE")
        for _ in range(60):
            append(right_transport[-1], left_open, right_open, "POST_RELEASE")
        tail_reports = (
            endpoint_reports
            + left_local_clearance_reports
            + left_reports
            + right_transport_reports
        )

    commands = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(stages)
    expected_frames = 881 if full else 493
    if commands.shape != (expected_frames, 28):
        raise RuntimeError(
            f"unexpected command shape {commands.shape}; expected {(expected_frames, 28)}"
        )

    contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray([float(row["minimum"]) for row in contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in contract["joint_specs"]])
    violations = (commands < lower[None] - 1.0e-9) | (
        commands > upper[None] + 1.0e-9
    )
    arm = commands[:, arm_indices]
    left = commands[:, left_indices]
    right = commands[:, right_indices]
    trajectory_geometry = g1.trajectory_geometry(arm, left, right, 1.0e-5)
    collision_counts = {
        key: int(np.count_nonzero(value))
        for key, value in trajectory_geometry["collision_flags"].items()
    }
    endpoint_audit = preliminary_candidate_audit(
        g1,
        acquisition_arm[-1],
        partial_left_relax,
        acquisition_hand,
        handoff_object,
        dimensions,
    )
    reports = approach_reports + acquisition_reports + tail_reports
    max_position_error = max(row["position_error_m"] for row in reports)
    max_orientation_error = max(row["orientation_error_rad"] for row in reports)
    distal = endpoint_audit["right_distal_proxy_signed_distance_m"]
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and endpoint_audit["no_left_right_hand_overlap"]
        and all(float(distal[digit]) <= 0.001 for digit in ("thumb", "index", "middle"))
        and max_position_error <= 0.001
        and max_orientation_error <= 0.02
    )

    command_path = candidate_dir / (
        "scripted_full_task_command.npz"
        if full
        else "backward_acquisition_gate_command.npz"
    )
    save_command(
        command_path,
        commands,
        labels,
        names,
        fps,
        args.candidate,
        handoff_object,
        bin_object,
        full,
    )
    report = {
        "schema_version": "p14_backward_constructed_handoff_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "mode": args.mode,
        "candidate_id": args.candidate,
        "declared_candidate_count": len(CANDIDATES),
        "declared_candidates_path_fraction": CANDIDATES,
        "evaluation_order": EVALUATION_ORDER,
        "path_fraction": progress,
        "minimum_jerk_blend": blend,
        "construction": "backward from frozen standalone RIGHT object-to-tool transform",
        "support_sequence_revision": timing_revision,
        "bounded_left_release_completion_fractions": LEFT_RELEASE_COMPLETION_FRACTIONS,
        "left_release_completion_fraction_selected": args.left_release_completion_fraction,
        "support_sequence": [
            "LEFT frozen P14 full hold",
            "RIGHT path and handoff-only three-digit closure",
            "RIGHT 0.5 s contact preload",
            "LEFT index+middle gradual 15% relaxation with LEFT P14 thumb held",
            "RIGHT three-digit physics verification",
            "LEFT thumb release only after runtime gate in full mode",
        ],
        "left_index_middle_relaxation_fraction_before_gate": LEFT_INDEX_MIDDLE_RELAXATION_FRACTION,
        "validated_endpoint_world": {
            "position_m": validated_position_world,
            "rotation_matrix": validated_rotation_world,
        },
        "pre_release_backward_path_endpoint_world": {
            "position_m": validated_position_world,
            "rotation_matrix": validated_rotation_world,
        }
        if full
        else None,
        "post_release_wrist_correction_used": False,
        "runtime_object_feedback_or_follow_used": False,
        "collision_free_preshape_world": {
            "translation_from_endpoint_m": PREPOSE_TRANSLATION_FROM_ENDPOINT_M,
            "yaw_from_endpoint_deg": PREPOSE_YAW_FROM_ENDPOINT_DEG,
            "position_m": prepose_position_world,
            "rotation_matrix": prepose_rotation_world,
        },
        "acquisition_world": {
            "position_m": candidate_position_world,
            "rotation_matrix": candidate_rotation_world,
            "translation_from_validated_endpoint_m": candidate_position_world
            - validated_position_world,
        },
        "handoff_only_acquisition_hand": hand_report,
        "final_right_p14_7d_rad": right_p14,
        "final_right_p14_unchanged": True,
        "left_p14_unchanged": True,
        "doll_changed": False,
        "controller_changed": False,
        "weld_attachment_magnet_teleport_object_follow_used": False,
        "standalone_endpoint_evidence": standalone_evidence,
        "endpoint_static_audit": endpoint_audit,
        "robot_collision_frame_counts": collision_counts,
        "robot_collision_pairs": trajectory_geometry["collision_pairs"],
        "joint_limit_violations": int(np.count_nonzero(violations)),
        "ik_max_position_error_m": max_position_error,
        "ik_max_orientation_error_rad": max_orientation_error,
        "left_release_in_command": full,
        "runtime_three_digit_gate_required": full,
        "runtime_three_digit_gate_minimum_s": 0.5,
        "physics_gate_evidence": str(args.physics_gate_evidence.resolve())
        if args.physics_gate_evidence
        else None,
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "freeze_manifest": str(FREEZE),
        "freeze_manifest_sha256": sha256_file(FREEZE),
        "config_sha256": sha256_file(CONFIG),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "policy_used": False,
        "real_robot": False,
    }
    atomic_json(candidate_dir / "offline_report.json", report)
    print((candidate_dir / "offline_report.json").read_text(encoding="utf-8"), end="")
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
