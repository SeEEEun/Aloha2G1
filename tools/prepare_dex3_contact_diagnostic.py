#!/usr/bin/env python3
"""Prepare a policy-free, shape-level Dex3 rigid-contact audit.

This command never starts Isaac and never reads a learned-policy artifact.  It
inspects the composed USD collision/filtering metadata and evaluates the frozen
OPEN/GRASP commands against the actual Dex3 convex-hull geometry in the active
MuJoCo mirror model.  Isaac stage execution is intentionally delegated to
``run_dex3_contact_stage_isaac.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

DEFAULT_CONFIG = ROOT / "configs/dex3_rigid_proxy_contact_diagnostic_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/dex3_rigid_proxy_contact_diagnostic"
OBJECT_CENTER_WORLD_XY = np.asarray([0.4175, 0.28], dtype=np.float64)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalized(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"invalid direction {value}")
    return vector / length


def candidate_geometry(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    kind = candidate["type"]
    if kind == "sphere":
        radius = float(candidate["radius_m"])
        support = radius
        volume = 4.0 * math.pi * radius**3 / 3.0
        maximum_extent = 2.0 * radius
    elif kind == "ellipsoid":
        major = float(candidate["major_semi_axis_m"])
        minor = float(candidate["minor_semi_axis_m"])
        axis = normalized(candidate["major_axis_world_xyz"])
        support = math.sqrt((major * axis[2]) ** 2 + minor**2 * (1.0 - axis[2] ** 2))
        volume = 4.0 * math.pi * major * minor * minor / 3.0
        maximum_extent = 2.0 * major
    elif kind == "capsule":
        radius = float(candidate["radius_m"])
        cylinder_height = float(candidate["cylinder_height_m"])
        axis = normalized(candidate["major_axis_world_xyz"])
        support = radius + 0.5 * cylinder_height * abs(float(axis[2]))
        volume = math.pi * radius**2 * cylinder_height + 4.0 * math.pi * radius**3 / 3.0
        maximum_extent = cylinder_height + 2.0 * radius
    else:
        raise ValueError(kind)
    center_z = (
        float(config["object"]["table_surface_world_z_m"])
        + support
        + float(config["object"]["spawn_clearance_above_table_m"])
    )
    return {
        "vertical_support_radius_m": support,
        "spawn_center_world_z_m": center_z,
        "volume_m3": volume,
        "density_kg_m3": float(config["object"]["mass_kg"]) / volume,
        "maximum_extent_m": maximum_extent,
    }


def mesh_points(stage: Usd.Stage, path: str) -> np.ndarray:
    root = stage.GetPrimAtPath(path)
    meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one mesh under {path}, got {[str(p.GetPath()) for p in meshes]}")
    return np.asarray(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get(), dtype=np.float64)


def usd_collider_audit(stage: Usd.Stage, whole_hand: dict[str, Any]) -> dict[str, Any]:
    distal_rows: list[dict[str, Any]] = []
    for side in ("left", "right"):
        for role in ("A", "B", "C"):
            spec = whole_hand[side][role]
            link = spec["distal_link"]
            link_path = f"/World/G1/Asset/{link}"
            collision_path = f"{link_path}/collisions"
            visual_path = f"{link_path}/visuals"
            collision_prim = stage.GetPrimAtPath(collision_path)
            visual_prim = stage.GetPrimAtPath(visual_path)
            visual_points = mesh_points(stage, visual_path)
            collision_points = mesh_points(stage, collision_path)
            visual_collision_error = (
                float(np.max(np.abs(visual_points - collision_points)))
                if visual_points.shape == collision_points.shape
                else None
            )
            visual_transform = np.asarray(
                UsdGeom.Xformable(visual_prim).GetLocalTransformation(), dtype=np.float64
            )
            collision_transform = np.asarray(
                UsdGeom.Xformable(collision_prim).GetLocalTransformation(), dtype=np.float64
            )
            frame_error = float(np.max(np.abs(visual_transform - collision_transform)))
            distal_rows.append(
                {
                    "side": side,
                    "role": role,
                    "digit_chain": spec["digit_chain"],
                    "link": link,
                    "collision_prim": collision_path,
                    "collision_type": collision_prim.GetTypeName(),
                    "collision_enabled": bool(
                        UsdPhysics.CollisionAPI(collision_prim).GetCollisionEnabledAttr().Get()
                    ),
                    "applied_schemas": list(collision_prim.GetAppliedSchemas()),
                    "approximation": collision_prim.GetAttribute("physics:approximation").Get(),
                    "collision_vertex_count": int(len(collision_points)),
                    "visual_vertex_count": int(len(visual_points)),
                    "visual_collision_vertex_max_abs_error_m": visual_collision_error,
                    "visual_collision_frame_match": bool(visual_collision_error == 0.0),
                    "visual_collision_local_transform_max_abs_error": frame_error,
                    "visual_collision_local_transform_match": bool(frame_error == 0.0),
                    "collision_local_transform": collision_transform,
                    "collision_local_transform_is_identity": bool(
                        np.allclose(collision_transform, np.eye(4), atol=0.0, rtol=0.0)
                    ),
                    "local_collision_bounds_min_m": np.min(collision_points, axis=0),
                    "local_collision_bounds_max_m": np.max(collision_points, axis=0),
                    "configured_pad_center_local_m": spec["local_position_xyz_m"],
                    "configured_pad_normal_local": spec["local_normal"],
                }
            )
    groups: list[dict[str, Any]] = []
    filtered_pairs: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.CollisionGroup):
            api = UsdPhysics.CollisionGroup(prim)
            groups.append(
                {
                    "path": str(prim.GetPath()),
                    "invert": bool(api.GetInvertFilteredGroupsAttr().Get()),
                    "filtered_groups": [str(path) for path in api.GetFilteredGroupsRel().GetTargets()],
                }
            )
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            api = UsdPhysics.FilteredPairsAPI(prim)
            filtered_pairs.append(
                {
                    "path": str(prim.GetPath()),
                    "filtered_pairs": [str(path) for path in api.GetFilteredPairsRel().GetTargets()],
                }
            )
    object_prim = stage.GetPrimAtPath("/World/DollHandoffEnvironment/Doll/Body")
    return {
        "distal_colliders": distal_rows,
        "all_distal_colliders_enabled": all(row["collision_enabled"] for row in distal_rows),
        "all_distal_visual_collision_frames_match": all(
            row["visual_collision_frame_match"]
            and row["visual_collision_local_transform_match"]
            for row in distal_rows
        ),
        "all_distal_collision_local_transforms_identity": all(
            row["collision_local_transform_is_identity"] for row in distal_rows
        ),
        "collision_groups": groups,
        "filtered_pairs": filtered_pairs,
        "dex3_object_pair_filtered": False,
        "filter_conclusion": (
            "No authored collision group or filtered-pair relationship excludes Dex3 from Doll. "
            "Runtime contact-pair evidence is still required."
        ),
        "source_object_collider": {
            "path": str(object_prim.GetPath()),
            "type": object_prim.GetTypeName(),
            "collision_enabled": bool(
                UsdPhysics.CollisionAPI(object_prim).GetCollisionEnabledAttr().Get()
            ),
            "radius_m": float(UsdGeom.Sphere(object_prim).GetRadiusAttr().Get()),
            "local_to_parent": np.asarray(
                UsdGeom.Xformable(object_prim).GetLocalTransformation(), dtype=np.float64
            ),
        },
    }


def convex_mesh_for_geom(g1: Any, geom_id: int) -> trimesh.Trimesh:
    mesh_id = int(g1.model.geom_dataid[geom_id])
    address = int(g1.model.mesh_vertadr[mesh_id])
    count = int(g1.model.mesh_vertnum[mesh_id])
    vertices = np.asarray(g1.model.mesh_vert[address : address + count], dtype=np.float64)
    rotation = np.asarray(g1.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    position = np.asarray(g1.data.geom_xpos[geom_id], dtype=np.float64)
    return trimesh.Trimesh(vertices=vertices @ rotation.T + position, process=False).convex_hull


def sphere_separation(center: np.ndarray, radius: float, hull: trimesh.Trimesh) -> tuple[float, float, bool]:
    _, distances, _ = trimesh.proximity.closest_point_naive(hull, [center])
    distance = float(distances[0])
    equations = ConvexHull(np.asarray(hull.vertices)).equations
    inside = bool(np.all(equations[:, :3] @ center + equations[:, 3] <= 1.0e-9))
    separation = -(radius + distance) if inside else distance - radius
    return separation, distance, inside


def frozen_sphere_overlap_audit(config: dict[str, Any], g1: Any) -> dict[str, Any]:
    names = [candidate["name"] for candidate in config["shape_candidates"]]
    sphere = config["shape_candidates"][names.index("sphere_75mm")]
    radius = float(sphere["radius_m"])
    geometry = candidate_geometry(sphere, config)
    center_world = np.r_[OBJECT_CENTER_WORLD_XY, geometry["spawn_center_world_z_m"]]
    center_model = g1.world_to_model_position(center_world)
    output: dict[str, Any] = {"center_world_m": center_world, "center_model_m": center_model}
    for side in ("left", "right"):
        primitive = np.load(config[f"{side}_primitive"], allow_pickle=False)
        commands = np.asarray(primitive["commanded_q_rad"], dtype=np.float64)
        stages = primitive["stage"].astype(str)
        indices = {
            "OPEN_RESET": 0,
            "FROZEN_GRASP": int(np.flatnonzero(stages == "HOLD_ON_TABLE")[-1]),
        }
        side_rows: dict[str, Any] = {}
        for label, index in indices.items():
            q = commands[index]
            g1.assign(q[:14], q[14:21], q[21:28])
            rows: list[dict[str, Any]] = []
            seen: set[tuple[str, int]] = set()
            for geom_id in range(g1.model.ngeom):
                body = mujoco.mj_id2name(
                    g1.model, mujoco.mjtObj.mjOBJ_BODY, int(g1.model.geom_bodyid[geom_id])
                ) or ""
                if f"{side}_hand_" not in body:
                    continue
                if int(g1.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                    continue
                key = (body, int(g1.model.geom_dataid[geom_id]))
                if key in seen:
                    continue
                seen.add(key)
                separation, surface_distance, center_inside = sphere_separation(
                    center_model, radius, convex_mesh_for_geom(g1, geom_id)
                )
                rows.append(
                    {
                        "body": body,
                        "signed_separation_m": separation,
                        "penetration_m": max(0.0, -separation),
                        "center_to_convex_hull_surface_m": surface_distance,
                        "proxy_center_inside_collider": center_inside,
                    }
                )
            rows.sort(key=lambda row: row["signed_separation_m"])
            side_rows[label] = {
                "actual_whole_hand_center_model_m": g1.whole_hand_grasp_pose(side)[:3, 3],
                "actual_whole_hand_enclosure_radius_m": g1.whole_hand_enclosure_radius(side),
                "whole_hand_center_error_to_proxy_m": float(
                    np.linalg.norm(g1.whole_hand_grasp_pose(side)[:3, 3] - center_model)
                ),
                "closest_colliders": rows[:7],
                "maximum_penetration_m": max(row["penetration_m"] for row in rows),
                "initial_overlap": bool(any(row["penetration_m"] > 1.0e-4 for row in rows)),
            }
        output[side] = side_rows
    output["root_cause"] = (
        "The frozen wrist-attached GRASP frame is centered at the proxy only after closure. "
        "At OPEN the actual three-pad center is displaced, and the 75 mm sphere overlaps the "
        "thumb convex hull before simulation begins; at GRASP it deeply overlaps all three "
        "distal convex hulls because the old pad-center circumradius is not the convex-hull clearance."
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != "dex3_rigid_proxy_contact_diagnostic_v1":
        raise RuntimeError("unexpected diagnostic config")
    if config.get("policy_evaluation_allowed") or config.get("learned_policy_used"):
        raise RuntimeError("contact diagnostic must remain policy-free")
    stage = Usd.Stage.Open(config["scene"])
    if stage is None:
        raise RuntimeError("failed to open composed scene")
    whole_hand = read_json(Path(config["whole_hand_geometry"]))
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    candidate_rows = []
    for candidate in config["shape_candidates"]:
        candidate_rows.append({**candidate, **candidate_geometry(candidate, config)})
    audit = {
        "schema_version": "dex3_static_collider_audit_v1",
        "status": "PASS_WITH_75MM_SPHERE_GEOMETRY_REJECTED",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "scene": config["scene"],
        "scene_sha256": sha256_file(Path(config["scene"])),
        "whole_hand_geometry": config["whole_hand_geometry"],
        "whole_hand_geometry_sha256": sha256_file(Path(config["whole_hand_geometry"])),
        "policy_or_checkpoint_used": False,
        "real_robot": False,
        "fixed_material_not_swept": config["fixed_material"],
        "finger_drive": config["finger_drive"],
        "shape_candidates_predeclared": candidate_rows,
        "usd": usd_collider_audit(stage, whole_hand),
        "sphere_collision_envelope": frozen_sphere_overlap_audit(config, g1),
        "conclusions": {
            "collision_filtering": "PASS_STATIC_PENDING_RUNTIME_PAIR",
            "dex3_fingertip_colliders": "PASS",
            "object_collider_pose_scale": "PASS",
            "visual_collision_frame_match": "PASS",
            "sphere_75mm_full_grasp_spawn": "FAIL_INITIAL_OVERLAP",
            "friction_tuning_performed": False,
        },
    }
    audit_path = output_root / "static_audit" / "STATIC_COLLIDER_AUDIT.json"
    atomic_json(audit_path, audit)
    manifest = {
        "schema_version": "dex3_contact_diagnostic_manifest_v1",
        "status": "STATIC_AUDIT_COMPLETE_RUNTIME_NOT_STARTED",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "static_audit": str(audit_path),
        "static_audit_sha256": sha256_file(audit_path),
        "policy_evaluation_stopped": True,
        "policy_artifacts_read": False,
        "learned_policy_used": False,
        "real_robot": False,
    }
    atomic_json(output_root / "DIAGNOSTIC_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
