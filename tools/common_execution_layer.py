#!/usr/bin/env python3
"""Representation-neutral Dex3-only execution for final ACT EVAL35.

This module deliberately contains no arm or wrist target generator.  It consumes
an already-frozen graspability envelope, observes the current measured whole-hand
pose, and may replace only named Dex3 joint targets.  The implementation is kept
independent of Isaac Lab so its invariants can be tested without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DIGITS = ("thumb", "index", "middle")
SIDES = ("left", "right")
ARM_INDICES = np.asarray([0, 1, 2, 3, 7, 8, 9, 10], dtype=np.int64)
WRIST_INDICES = np.asarray([4, 5, 6, 11, 12, 13], dtype=np.int64)
LEFT_DEX3 = np.arange(14, 21, dtype=np.int64)
RIGHT_DEX3 = np.arange(21, 28, dtype=np.int64)
DEX3_INDICES = np.arange(14, 28, dtype=np.int64)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str | Path, relative_to: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _rotation_matrix_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must be one finite XYZW row")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(cosine))


def pose_matrix(position_m: Sequence[float], quaternion_xyzw: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation_matrix_xyzw(quaternion_xyzw)
    result[:3, 3] = np.asarray(position_m, dtype=np.float64)
    return result


def inverse_pose(value: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def _normalize(value: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1.0e-12 else np.asarray(fallback, dtype=np.float64)


def _triangle_circumcenter(points: np.ndarray) -> np.ndarray:
    a, b, c = np.asarray(points, dtype=np.float64)
    ab, ac = b - a, c - a
    cross = np.cross(ab, ac)
    denominator = 2.0 * float(cross @ cross)
    if denominator <= 1.0e-16:
        return np.mean(points, axis=0)
    offset = (
        float(ac @ ac) * np.cross(cross, ab)
        + float(ab @ ab) * np.cross(ac, cross)
    ) / denominator
    return a + offset


def whole_hand_pose_from_pad_centers(
    pad_centers_world_m: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Use the frozen three-pad geometric whole-hand frame definition."""

    points = np.stack(
        [np.asarray(pad_centers_world_m[digit], dtype=np.float64) for digit in DIGITS]
    )
    center = _triangle_circumcenter(points)
    closing = _normalize(0.5 * (points[1] + points[2]) - points[0], (1.0, 0.0, 0.0))
    spread = points[1] - points[2]
    spread = _normalize(spread - closing * float(spread @ closing), (0.0, 0.0, 1.0))
    transverse = _normalize(np.cross(spread, closing), (0.0, 1.0, 0.0))
    spread = _normalize(np.cross(closing, transverse), (0.0, 0.0, 1.0))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.column_stack((closing, transverse, spread))
    result[:3, 3] = center
    if not np.isclose(np.linalg.det(result[:3, :3]), 1.0, atol=1.0e-7):
        raise RuntimeError("invalid measured whole-hand frame")
    return result


class GraspabilityEnvelope(Protocol):
    """Read-only interface implemented by a frozen physical evaluator."""

    evaluator_sha256: str

    def margin(self, object_from_whole_hand: np.ndarray) -> float:
        """Return the frozen signed margin; non-negative is eligible."""


@dataclass(frozen=True)
class _EnvelopeCenter:
    translation_m: np.ndarray
    rotation: np.ndarray
    effective_radius_normalized: float


class FrozenUnionBallEnvelope:
    """Exact runtime form of the predeclared union-of-SE(3)-balls classifier.

    No radius is estimated here.  Every positive center must carry its already-
    frozen effective radius, including the frozen negative guard.  This avoids
    silently reconstructing or retuning classifier decisions in the execution
    layer.
    """

    def __init__(
        self,
        centers: Sequence[_EnvelopeCenter],
        translation_normalization_m: float,
        rotation_normalization_deg: float,
        evaluator_sha256: str,
    ) -> None:
        if not centers:
            raise ValueError("frozen envelope contains no positive centers")
        if translation_normalization_m <= 0 or rotation_normalization_deg <= 0:
            raise ValueError("invalid frozen envelope normalization")
        if not HEX_SHA256.fullmatch(evaluator_sha256):
            raise ValueError("missing valid frozen evaluator SHA256")
        self.centers = tuple(centers)
        self.translation_normalization_m = float(translation_normalization_m)
        self.rotation_normalization_rad = math.radians(float(rotation_normalization_deg))
        self.evaluator_sha256 = evaluator_sha256

    def margin(self, object_from_whole_hand: np.ndarray) -> float:
        pose = np.asarray(object_from_whole_hand, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return -math.inf
        margins = []
        for center in self.centers:
            translation_distance = float(
                np.linalg.norm(pose[:3, 3] - center.translation_m)
                / self.translation_normalization_m
            )
            rotation_distance = _rotation_angle(center.rotation.T @ pose[:3, :3])
            rotation_distance /= self.rotation_normalization_rad
            distance = math.hypot(translation_distance, rotation_distance)
            margins.append(center.effective_radius_normalized - distance)
        return float(max(margins))

    @classmethod
    def from_artifact(
        cls, artifact: Mapping[str, Any], evaluator_sha256: str
    ) -> "FrozenUnionBallEnvelope":
        schema = str(artifact.get("schema_version", ""))
        if "graspability" not in schema or "envelope" not in schema:
            raise RuntimeError(f"unexpected envelope schema: {schema!r}")
        if artifact.get("status") != "FROZEN":
            raise RuntimeError("graspability envelope is not FROZEN")
        if artifact.get("feature_space") not in {
            "OBJECT_FROM_WHOLE_HAND_SE3",
            "object_from_whole_hand_se3",
        }:
            raise RuntimeError("envelope is not expressed in object-from-whole-hand SE(3)")
        normalization = artifact["normalization"]
        rows = artifact.get("positive_centers", artifact.get("centers", []))
        centers: list[_EnvelopeCenter] = []
        for row in rows:
            radius = row.get(
                "effective_radius_normalized",
                row.get("frozen_effective_radius_normalized"),
            )
            if radius is None:
                raise RuntimeError(
                    "envelope center lacks a frozen negative-guarded effective radius"
                )
            translation = row.get(
                "object_from_whole_hand_translation_m", row.get("translation_m")
            )
            quaternion = row.get(
                "object_from_whole_hand_quaternion_xyzw", row.get("quaternion_xyzw")
            )
            if translation is None or quaternion is None:
                raise RuntimeError("envelope center lacks the frozen SE(3) pose")
            centers.append(
                _EnvelopeCenter(
                    translation_m=np.asarray(translation, dtype=np.float64),
                    rotation=_rotation_matrix_xyzw(quaternion),
                    effective_radius_normalized=float(radius),
                )
            )
        return cls(
            centers,
            float(normalization["translation_m"]),
            float(normalization["rotation_deg"]),
            evaluator_sha256,
        )


@dataclass(frozen=True)
class FrozenEvaluatorBundle:
    manifest_path: Path
    manifest_sha256: str
    evaluator_sha256: str
    envelope: FrozenUnionBallEnvelope
    common_intent_artifact: Path


def _declared_evaluator_sha(manifest: Mapping[str, Any]) -> str:
    for key in (
        "evaluator_sha256",
        "frozen_evaluator_sha256",
        "evaluator_bundle_sha256",
        "freeze_sha256",
    ):
        value = str(manifest.get(key, ""))
        if HEX_SHA256.fullmatch(value):
            return value
    raise RuntimeError("EVALUATOR_FREEZE_MANIFEST has no frozen evaluator SHA256")


def load_frozen_evaluator(manifest_path: Path) -> FrozenEvaluatorBundle:
    """Verify and load the authoritative evaluator without modifying it."""

    path = manifest_path.resolve()
    manifest = read_json(path)
    if manifest.get("status") != "FROZEN":
        raise RuntimeError("physical evaluator is not FROZEN")
    evaluator_sha = _declared_evaluator_sha(manifest)
    files = manifest.get("files", manifest.get("artifacts"))
    if not isinstance(files, list) or not files:
        raise RuntimeError("evaluator freeze manifest has no hashed artifact list")
    verified: list[tuple[dict[str, Any], Path]] = []
    for row in files:
        if not isinstance(row, dict) or "path" not in row or "sha256" not in row:
            raise RuntimeError("invalid evaluator freeze artifact record")
        dependency = _resolve_path(row["path"], path.parent)
        expected = str(row["sha256"])
        if not dependency.is_file():
            raise FileNotFoundError(f"missing frozen evaluator artifact: {dependency}")
        actual = sha256_file(dependency)
        if actual != expected:
            raise RuntimeError(
                f"frozen evaluator artifact hash drift: {dependency}: {actual} != {expected}"
            )
        verified.append((row, dependency))

    def role_matches(row: Mapping[str, Any], dependency: Path, terms: Sequence[str]) -> bool:
        value = " ".join(
            [str(row.get("role", "")), str(row.get("name", "")), dependency.name]
        ).lower()
        return all(term in value for term in terms)

    envelope_paths = [
        dependency
        for row, dependency in verified
        if role_matches(row, dependency, ("graspability", "envelope"))
    ]
    intent_paths = [
        dependency
        for row, dependency in verified
        if role_matches(row, dependency, ("common", "grasp", "intent"))
        and "spec" not in dependency.name.lower()
    ]
    if len(envelope_paths) != 1:
        raise RuntimeError(
            "freeze manifest must identify exactly one DEX3 physical graspability envelope"
        )
    if len(intent_paths) != 1:
        raise RuntimeError(
            "freeze manifest must identify exactly one materialized common grasp-intent artifact"
        )
    envelope = FrozenUnionBallEnvelope.from_artifact(
        read_json(envelope_paths[0]), evaluator_sha
    )
    return FrozenEvaluatorBundle(
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        evaluator_sha256=evaluator_sha,
        envelope=envelope,
        common_intent_artifact=intent_paths[0],
    )


def load_common_grasp_intent(
    artifact_path: Path, stable_episode_id: str, expected_frames: int
) -> np.ndarray:
    """Load an already-materialized frozen timeline; never infer one here."""

    path = artifact_path.resolve()
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            candidates = (
                stable_episode_id,
                f"{stable_episode_id}_intent",
                "common_grasp_intent",
            )
            key = next((name for name in candidates if name in archive.files), None)
            if key is None:
                raise RuntimeError(f"common intent missing for {stable_episode_id}")
            values = np.asarray(archive[key]).astype(str)
    else:
        artifact = read_json(path)
        episodes = artifact.get("episodes", artifact.get("timelines", {}))
        if isinstance(episodes, list):
            matches = [
                row for row in episodes if row.get("stable_episode_id") == stable_episode_id
            ]
            if len(matches) != 1:
                raise RuntimeError(f"common intent identity mismatch: {stable_episode_id}")
            values = np.asarray(
                matches[0].get("intent", matches[0].get("states", []))
            ).astype(str)
        elif stable_episode_id in episodes:
            row = episodes[stable_episode_id]
            values = np.asarray(
                row.get("intent", row.get("states", [])) if isinstance(row, dict) else row
            ).astype(str)
        else:
            raise RuntimeError(f"common intent missing for {stable_episode_id}")
    if values.shape != (expected_frames,):
        raise RuntimeError(
            f"common intent length mismatch for {stable_episode_id}: "
            f"{values.shape} != {(expected_frames,)}"
        )
    allowed = {"OPEN_INTENT", "CLOSE_INTENT", "HOLD_INTENT", "RELEASE_INTENT"}
    if not set(np.unique(values)).issubset(allowed):
        raise RuntimeError("frozen common intent contains an unknown state")
    return values


@dataclass(frozen=True)
class Dex3Primitive:
    left_open: np.ndarray
    left_preshape: np.ndarray
    left_full_close: np.ndarray
    right_open: np.ndarray
    right_preshape: np.ndarray
    right_full_close: np.ndarray
    preshape_frames: int
    close_frames: int
    release_frames: int
    force_threshold_n: float
    left_retention_frames: int
    right_verification_frames: int
    right_retention_frames: int
    maximum_table_force_n: float
    bin_center_xy_m: np.ndarray
    bin_opening_xy_m: np.ndarray
    bin_bottom_z_m: float
    bin_rim_z_m: float

    @classmethod
    def from_frozen_dependencies(
        cls,
        physics_config: Mapping[str, Any],
        physical_environment: Mapping[str, Any],
        common_controller: Mapping[str, Any],
    ) -> "Dex3Primitive":
        if physical_environment.get("status") != "FROZEN":
            raise RuntimeError("physical environment is not frozen")
        if common_controller.get("status") != "FROZEN":
            raise RuntimeError("physical common controller is not frozen")
        fps = float(physics_config["timing"]["control_fps_hz"])
        if not np.isclose(fps, float(physical_environment["physics"]["control_fps_hz"])):
            raise RuntimeError("frozen control-rate mismatch")
        states = physics_config["hand_states"]
        timing = physics_config["timing"]
        gates = physics_config["gates"]
        verification_s = float(
            common_controller["right_three_digit_gate"]["minimum_simultaneous_support_s"]
        )
        # The final opening uses the same morphology transition duration as
        # PRESHAPE.  It is fixed from the independent P14 physical primitive,
        # not estimated from A/B trajectories.
        transition_frames = max(1, int(round(float(timing["preshape_transition_s"]) * fps)))
        return cls(
            left_open=np.asarray(states["left"]["OPEN"], dtype=np.float64),
            left_preshape=np.asarray(states["left"]["PRESHAPE"], dtype=np.float64),
            left_full_close=np.asarray(
                states["left"]["POWER_GRASP_P14"], dtype=np.float64
            ),
            right_open=np.asarray(states["right"]["OPEN"], dtype=np.float64),
            right_preshape=np.asarray(states["right"]["PRESHAPE"], dtype=np.float64),
            right_full_close=np.asarray(
                states["right"]["POWER_GRASP_P14"], dtype=np.float64
            ),
            preshape_frames=transition_frames,
            close_frames=max(1, int(round(float(timing["power_close_s"]) * fps))),
            release_frames=transition_frames,
            force_threshold_n=float(gates["meaningful_digit_force_n"]),
            left_retention_frames=max(
                1, int(round(float(gates["minimum_retention_contact_s"]) * fps))
            ),
            right_verification_frames=max(1, int(round(verification_s * fps))),
            right_retention_frames=max(
                1, int(round(float(gates["minimum_retention_contact_s"]) * fps))
            ),
            maximum_table_force_n=float(gates["maximum_table_force_for_elevated_n"]),
            bin_center_xy_m=np.asarray(
                physical_environment["bin"]["opening_center_world_xy_m"], dtype=np.float64
            ),
            bin_opening_xy_m=np.asarray(
                physical_environment["bin"]["opening_dimensions_xy_m"], dtype=np.float64
            ),
            bin_bottom_z_m=float(physical_environment["bin"]["bottom_world_z_m"]),
            bin_rim_z_m=float(physical_environment["bin"]["rim_world_z_m"]),
        )


@dataclass(frozen=True)
class ExecutionSnapshot:
    measured_q_rad: np.ndarray
    object_world: np.ndarray
    whole_hand_world: Mapping[str, np.ndarray]
    digit_force_n: Mapping[str, Mapping[str, float]]
    table_force_n: float
    previous_control_frame_support: Mapping[str, bool] | None = None


@dataclass(frozen=True)
class ExecutionDecision:
    executed_command: np.ndarray
    override_mask: np.ndarray
    arm_override_mask: np.ndarray
    wrist_override_mask: np.ndarray
    dex3_override_mask: np.ndarray
    phase: str
    graspability_margin: float
    left_envelope_eligible: bool
    events: tuple[str, ...]


def _minimum_jerk(alpha: float) -> float:
    value = float(np.clip(alpha, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _transition(start: np.ndarray, stop: np.ndarray, elapsed: int, frames: int) -> np.ndarray:
    if frames <= 1:
        return stop.copy()
    weight = _minimum_jerk(elapsed / float(frames - 1))
    return (1.0 - weight) * start + weight * stop


class CommonDex3ExecutionLayer:
    """Causal, method-blind PRESHAPE -> FULL CLOSE -> HOLD controller."""

    def __init__(
        self,
        envelope: GraspabilityEnvelope,
        primitive: Dex3Primitive,
        common_grasp_intent: Sequence[str],
        raw_policy_command: np.ndarray,
        policy_safe_command: np.ndarray,
        method: str,
    ) -> None:
        if method not in {"ACT-A40", "ACT-B40"}:
            raise ValueError("execution layer accepts only frozen ACT-A40/ACT-B40 identities")
        raw = np.asarray(raw_policy_command, dtype=np.float64)
        safe = np.asarray(policy_safe_command, dtype=np.float64)
        intent = np.asarray(common_grasp_intent).astype(str)
        if raw.shape != safe.shape or raw.ndim != 2 or raw.shape[1] != 28:
            raise ValueError("ACT command must have shape (T, 28)")
        if intent.shape != (len(raw),):
            raise ValueError("common intent and command lengths differ")
        if not np.isfinite(raw).all() or not np.isfinite(safe).all():
            raise ValueError("ACT command contains non-finite values")
        self.envelope = envelope
        self.primitive = primitive
        self.intent = intent
        self.raw = raw
        self.safe = safe
        self.method = method
        self.left_trigger: int | None = None
        self.left_start_q: np.ndarray | None = None
        self.left_owned_frame: int | None = None
        self.right_trigger: int | None = None
        self.right_start_q: np.ndarray | None = None
        self.right_support_frame: int | None = None
        self.left_release_frame: int | None = None
        self.left_release_start_q: np.ndarray | None = None
        self.right_owned_frame: int | None = None
        self.right_release_frame: int | None = None
        self.right_release_start_q: np.ndarray | None = None
        self.left_support_counter = 0
        self.right_three_counter = 0
        self.right_retention_counter = 0
        self.last_frame = -1
        self.trace: list[ExecutionDecision] = []

    @staticmethod
    def _three_digit(
        force: Mapping[str, float], threshold: float
    ) -> bool:
        return all(float(force.get(digit, 0.0)) >= threshold for digit in DIGITS)

    @staticmethod
    def _two_digit(force: Mapping[str, float], threshold: float) -> bool:
        return sum(float(force.get(digit, 0.0)) >= threshold for digit in DIGITS) >= 2

    def _left_target(self, frame: int, base: np.ndarray) -> np.ndarray:
        if self.left_trigger is None or self.left_start_q is None:
            return base
        p = self.primitive
        if self.left_release_frame is not None:
            if self.left_release_start_q is None:
                raise RuntimeError("missing left release start state")
            return _transition(
                self.left_release_start_q,
                p.left_open,
                frame - self.left_release_frame,
                p.release_frames,
            )
        elapsed = frame - self.left_trigger
        if elapsed < p.preshape_frames:
            return _transition(self.left_start_q, p.left_preshape, elapsed, p.preshape_frames)
        elapsed -= p.preshape_frames
        if elapsed < p.close_frames:
            return _transition(p.left_preshape, p.left_full_close, elapsed, p.close_frames)
        return p.left_full_close.copy()

    def _right_target(self, frame: int, base: np.ndarray) -> np.ndarray:
        if self.right_trigger is None or self.right_start_q is None:
            return base
        p = self.primitive
        if self.right_release_frame is not None:
            if self.right_release_start_q is None:
                raise RuntimeError("missing right release start state")
            return _transition(
                self.right_release_start_q,
                p.right_open,
                frame - self.right_release_frame,
                p.release_frames,
            )
        elapsed = frame - self.right_trigger
        if elapsed < p.preshape_frames:
            return _transition(self.right_start_q, p.right_preshape, elapsed, p.preshape_frames)
        elapsed -= p.preshape_frames
        if elapsed < p.close_frames:
            return _transition(p.right_preshape, p.right_full_close, elapsed, p.close_frames)
        return p.right_full_close.copy()

    def _inside_frozen_bin_region(self, object_position: np.ndarray) -> bool:
        p = self.primitive
        return bool(
            np.all(
                np.abs(object_position[:2] - p.bin_center_xy_m)
                <= p.bin_opening_xy_m / 2.0
            )
            and p.bin_bottom_z_m < float(object_position[2]) < p.bin_rim_z_m
        )

    def _phase(self, frame: int) -> str:
        if self.left_trigger is None:
            return "APPROACH"
        if self.left_owned_frame is None:
            return "GRASP"
        if self.right_trigger is not None and self.right_owned_frame is None:
            return "HANDOFF"
        if self.right_owned_frame is not None and self.right_release_frame is None:
            if frame - self.right_owned_frame < self.primitive.right_retention_frames:
                return "RIGHT_OWNERSHIP"
            return "TRANSPORT"
        if self.right_release_frame is not None:
            return "RELEASE"
        return "TRANSPORT"

    def step(self, frame: int, snapshot: ExecutionSnapshot) -> ExecutionDecision:
        if frame != self.last_frame + 1 or frame >= len(self.safe):
            raise RuntimeError("execution controller requires one ordered call per control frame")
        self.last_frame = frame
        p = self.primitive
        base = self.safe[frame].copy()
        events: list[str] = []
        object_from_left = inverse_pose(snapshot.object_world) @ snapshot.whole_hand_world["left"]
        margin = float(self.envelope.margin(object_from_left))
        eligible = bool(np.isfinite(margin) and margin >= 0.0)
        intent = str(self.intent[frame])

        if (
            self.left_trigger is None
            and intent in {"CLOSE_INTENT", "HOLD_INTENT"}
            and eligible
        ):
            self.left_trigger = frame
            self.left_start_q = np.asarray(snapshot.measured_q_rad[LEFT_DEX3], dtype=np.float64).copy()
            events.append("LEFT_ENVELOPE_ENTRY")

        left_force = snapshot.digit_force_n["left"]
        right_force = snapshot.digit_force_n["right"]
        table_free = float(snapshot.table_force_n) <= p.maximum_table_force_n
        previous = snapshot.previous_control_frame_support
        left_three_complete = (
            bool(previous["left_three_table_free"])
            if previous is not None
            else self._three_digit(left_force, p.force_threshold_n) and table_free
        )
        if self.left_trigger is not None and left_three_complete:
            self.left_support_counter += 1
        else:
            self.left_support_counter = 0
        if self.left_owned_frame is None and self.left_support_counter >= p.left_retention_frames:
            self.left_owned_frame = frame
            events.append("LEFT_PHYSICAL_OWNERSHIP")

        # Handoff entry has no distance tolerance.  A real receiving-digit
        # contact plus stable LEFT ownership proves the policy brought the
        # receiving hand into a physically reachable local configuration.
        right_any_contact = any(
            float(right_force.get(digit, 0.0)) >= p.force_threshold_n for digit in DIGITS
        )
        left_current_support = any(
            float(left_force.get(digit, 0.0)) >= p.force_threshold_n for digit in DIGITS
        )
        if (
            self.right_trigger is None
            and self.left_owned_frame is not None
            and left_current_support
            and table_free
            and right_any_contact
        ):
            self.right_trigger = frame
            self.right_start_q = np.asarray(snapshot.measured_q_rad[RIGHT_DEX3], dtype=np.float64).copy()
            events.append("HANDOFF_REAL_CONTACT_ENTRY")

        right_three_complete = (
            bool(previous["right_three_table_free"])
            if previous is not None
            else self._three_digit(right_force, p.force_threshold_n) and table_free
        )
        if self.right_trigger is not None and right_three_complete:
            self.right_three_counter += 1
        else:
            self.right_three_counter = 0
        if self.right_support_frame is None and self.right_three_counter >= p.right_verification_frames:
            self.right_support_frame = frame
            self.left_release_frame = frame
            self.left_release_start_q = np.asarray(
                snapshot.measured_q_rad[LEFT_DEX3], dtype=np.float64
            ).copy()
            events.extend(("RIGHT_THREE_DIGIT_SUPPORT_CONFIRMED", "GIVING_HAND_RELEASE_START"))

        left_release_complete = bool(
            self.left_release_frame is not None
            and frame - self.left_release_frame >= p.release_frames - 1
        )
        right_two_complete = (
            bool(previous["right_two_table_free"])
            if previous is not None
            else self._two_digit(right_force, p.force_threshold_n) and table_free
        )
        if left_release_complete and right_two_complete:
            self.right_retention_counter += 1
        else:
            self.right_retention_counter = 0
        if self.right_owned_frame is None and self.right_retention_counter >= p.right_retention_frames:
            self.right_owned_frame = frame
            events.append("RIGHT_PHYSICAL_OWNERSHIP")

        if (
            self.right_release_frame is None
            and self.right_owned_frame is not None
            and self._inside_frozen_bin_region(snapshot.object_world[:3, 3])
        ):
            self.right_release_frame = frame
            self.right_release_start_q = np.asarray(
                snapshot.measured_q_rad[RIGHT_DEX3], dtype=np.float64
            ).copy()
            events.append("FROZEN_BIN_REGION_RELEASE_START")

        executed = base.copy()
        executed[LEFT_DEX3] = self._left_target(frame, base[LEFT_DEX3])
        executed[RIGHT_DEX3] = self._right_target(frame, base[RIGHT_DEX3])
        # These exact equality guards make arm rescue, wrist snap, and hidden
        # safety projection inside this layer impossible.
        if not np.array_equal(executed[ARM_INDICES], base[ARM_INDICES]):
            raise RuntimeError("COMMON EXECUTION LAYER INVALID: arm target changed")
        if not np.array_equal(executed[WRIST_INDICES], base[WRIST_INDICES]):
            raise RuntimeError("COMMON EXECUTION LAYER INVALID: wrist target changed")
        override = np.abs(executed - base) > 1.0e-12
        arm_mask = np.zeros(28, dtype=bool)
        wrist_mask = np.zeros(28, dtype=bool)
        dex3_mask = np.zeros(28, dtype=bool)
        arm_mask[ARM_INDICES] = override[ARM_INDICES]
        wrist_mask[WRIST_INDICES] = override[WRIST_INDICES]
        dex3_mask[DEX3_INDICES] = override[DEX3_INDICES]
        decision = ExecutionDecision(
            executed_command=executed,
            override_mask=override,
            arm_override_mask=arm_mask,
            wrist_override_mask=wrist_mask,
            dex3_override_mask=dex3_mask,
            phase=self._phase(frame),
            graspability_margin=margin,
            left_envelope_eligible=eligible,
            events=tuple(events),
        )
        self.trace.append(decision)
        return decision

    def summary(self) -> dict[str, Any]:
        frames = max(1, len(self.trace))
        masks = (
            np.stack([row.override_mask for row in self.trace])
            if self.trace
            else np.zeros((0, 28), dtype=bool)
        )
        safety = self.safe[: len(self.trace)] - self.raw[: len(self.trace)]
        return {
            "schema_version": "common_dex3_execution_runtime_summary_v1",
            "method": self.method,
            "frozen_evaluator_sha256": self.envelope.evaluator_sha256,
            "frames": len(self.trace),
            "events": {
                "left_envelope_entry_frame": self.left_trigger,
                "left_physical_ownership_frame": self.left_owned_frame,
                "handoff_real_contact_entry_frame": self.right_trigger,
                "right_three_digit_support_frame": self.right_support_frame,
                "giving_hand_release_frame": self.left_release_frame,
                "right_physical_ownership_frame": self.right_owned_frame,
                "bin_region_release_frame": self.right_release_frame,
            },
            "common_override_fraction": float(np.count_nonzero(masks) / max(1, masks.size)),
            "arm_common_override_scalar_count": int(
                np.count_nonzero(masks[:, ARM_INDICES]) if len(masks) else 0
            ),
            "wrist_common_override_scalar_count": int(
                np.count_nonzero(masks[:, WRIST_INDICES]) if len(masks) else 0
            ),
            "dex3_common_override_scalar_count": int(
                np.count_nonzero(masks[:, DEX3_INDICES]) if len(masks) else 0
            ),
            "maximum_raw_to_policy_safe_arm_delta_rad": float(
                np.max(np.abs(safety[:, ARM_INDICES]), initial=0.0)
            ),
            "maximum_raw_to_policy_safe_wrist_delta_rad": float(
                np.max(np.abs(safety[:, WRIST_INDICES]), initial=0.0)
            ),
            "wrist_rescue_used": False,
            "scripted_arm_rescue_used": False,
            "object_motion_commanded": False,
            "direct_state_write_during_execution": False,
            "phase_counts": {
                phase: sum(row.phase == phase for row in self.trace)
                for phase in (
                    "APPROACH",
                    "GRASP",
                    "HANDOFF",
                    "RIGHT_OWNERSHIP",
                    "TRANSPORT",
                    "RELEASE",
                )
            },
        }


__all__ = [
    "ARM_INDICES",
    "WRIST_INDICES",
    "DEX3_INDICES",
    "LEFT_DEX3",
    "RIGHT_DEX3",
    "CommonDex3ExecutionLayer",
    "Dex3Primitive",
    "ExecutionDecision",
    "ExecutionSnapshot",
    "FrozenEvaluatorBundle",
    "FrozenUnionBallEnvelope",
    "GraspabilityEnvelope",
    "inverse_pose",
    "load_common_grasp_intent",
    "load_frozen_evaluator",
    "pose_matrix",
    "read_json",
    "sha256_file",
    "whole_hand_pose_from_pad_centers",
]
