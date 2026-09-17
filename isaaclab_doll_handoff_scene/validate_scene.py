#!/usr/bin/env python3
"""Static structural and geometric acceptance checks for Doll-Handoff."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pxr import Usd, UsdGeom, UsdPhysics

from preview_common import compose_robot_preview, g1_geometry_report, load_layout


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "generated" / "doll_handoff_scene.usda"
ALOHA_PREVIEW = ROOT / "generated" / "doll_handoff_aloha_model_preview.usda"
G1_PREVIEW = ROOT / "generated" / "doll_handoff_g1_model_preview.usda"
REPORT_PATH = ROOT / "outputs" / "doll_handoff_scene" / "static_validation.json"
FORBIDDEN_TASK_TERMS = ("phone", "accessory", "charger", "magnetic", "magsafe")


def _require(stage: Usd.Stage, path: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsActive():
        raise RuntimeError(f"Required active prim is missing: {path}")
    return prim


def _active_forbidden_prim_paths(stage: Usd.Stage) -> list[str]:
    return sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if any(term in str(prim.GetPath()).lower() for term in FORBIDDEN_TASK_TERMS)
    )


def _collision_descendants(stage: Usd.Stage, root_path: str) -> list[str]:
    root = _require(stage, root_path)
    return sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )


def _inside_workspace(layout: dict[str, Any], center: list[float], half_size: list[float]) -> bool:
    lower, upper = layout["black_frame"]["inner_opening_extents_world_xy_m"]
    return all(
        float(lower[index]) < float(center[index]) - float(half_size[index])
        and float(center[index]) + float(half_size[index]) < float(upper[index])
        for index in range(2)
    )


def main() -> int:
    layout = load_layout()
    stage = Usd.Stage.Open(str(SCENE))
    if stage is None:
        raise RuntimeError(f"Could not open built scene: {SCENE}")

    required = [
        "/DollHandoffScene/Table",
        "/DollHandoffScene/Table/Visuals/FrameFront",
        "/DollHandoffScene/Table/Visuals/FrameBack",
        "/DollHandoffScene/Table/Visuals/FrameLeft",
        "/DollHandoffScene/Table/Visuals/FrameRight",
        "/DollHandoffScene/Doll",
        "/DollHandoffScene/TrashBin",
        "/DollHandoffScene/Cameras/overview",
        "/DollHandoffScene/Cameras/top",
        "/DollHandoffScene/Lights/Dome",
        "/DollHandoffScene/Lights/Key",
        "/DollHandoffScene/Lights/Fill",
    ]
    for path in required:
        _require(stage, path)

    doll = _require(stage, "/DollHandoffScene/Doll")
    if not doll.HasAPI(UsdPhysics.RigidBodyAPI) or not doll.HasAPI(UsdPhysics.MassAPI):
        raise RuntimeError("Doll must have rigid-body and explicit-mass APIs")
    mass = float(UsdPhysics.MassAPI(doll).GetMassAttr().Get())
    if not math.isclose(mass, float(layout["doll"]["mass_kg"]), abs_tol=1e-9):
        raise RuntimeError(f"Doll mass mismatch: {mass}")
    doll_colliders = _collision_descendants(stage, "/DollHandoffScene/Doll")
    if doll_colliders != ["/DollHandoffScene/Doll/Body"]:
        raise RuntimeError(f"Unexpected Doll collision geometry: {doll_colliders}")

    expected_bin_colliders = sorted(
        f"/DollHandoffScene/TrashBin/{name}" for name in layout["bin"]["collision_parts"]
    )
    bin_colliders = _collision_descendants(stage, "/DollHandoffScene/TrashBin")
    if bin_colliders != expected_bin_colliders:
        raise RuntimeError(f"Bin must have exactly five open-container colliders: {bin_colliders}")
    if _require(stage, "/DollHandoffScene/TrashBin").HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("Trash bin must remain static")

    outer = list(map(float, layout["bin"]["outer_dimensions_xyz_m"]))
    wall = float(layout["bin"]["wall_thickness_m"])
    expected_opening = [outer[0] - 2.0 * wall, outer[1] - 2.0 * wall]
    if any(abs(a - b) > 1e-12 for a, b in zip(expected_opening, layout["bin"]["opening_dimensions_xy_m"])):
        raise RuntimeError(f"Bin opening dimensions are inconsistent: expected={expected_opening}")

    doll_center = list(map(float, layout["doll"]["center_world_xy_m"]))
    doll_radius = float(layout["doll"]["diameter_m"]) / 2.0
    bin_center = list(map(float, layout["bin"]["center_world_xy_m"]))
    if not _inside_workspace(layout, doll_center, [doll_radius, doll_radius]):
        raise RuntimeError("Doll is not fully inside the black workspace frame")
    if not _inside_workspace(layout, bin_center, [outer[0] / 2.0, outer[1] / 2.0]):
        raise RuntimeError("Trash bin is not fully inside the black workspace frame")
    if not doll_center[0] < bin_center[0]:
        raise RuntimeError("Doll/bin left-right placement is reversed")

    scene_forbidden = _active_forbidden_prim_paths(stage)
    if scene_forbidden:
        raise RuntimeError(f"Old task prims remain active in the clean scene: {scene_forbidden}")

    aloha_stage = compose_robot_preview(ALOHA_PREVIEW, "aloha", "StationaryALOHA")
    g1_stage = compose_robot_preview(G1_PREVIEW, "g1", "G1")
    g1_report = g1_geometry_report(g1_stage)
    preview_forbidden = {
        "aloha": _active_forbidden_prim_paths(aloha_stage),
        "g1": _active_forbidden_prim_paths(g1_stage),
    }
    if any(preview_forbidden.values()):
        raise RuntimeError(f"Old task prims remain active in a preview: {preview_forbidden}")

    report = {
        "status": "PASS",
        "required_prim_count": len(required),
        "doll": {
            "rigid_body": True,
            "mass_kg": mass,
            "colliders": doll_colliders,
            "inside_black_frame": True,
        },
        "bin": {
            "static": True,
            "colliders": bin_colliders,
            "opening_dimensions_xy_m": expected_opening,
            "top_collider": None,
            "interior_filler_collider": None,
            "inside_black_frame": True,
        },
        "g1_placement": g1_report,
        "old_task_active_prim_count": 0,
        "old_task_active_prim_paths": [],
        "old_episode_hard_dependency": False,
        "preview_usd": {"aloha": str(ALOHA_PREVIEW), "g1": str(G1_PREVIEW)},
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: DOLL_HANDOFF_STATIC_VALIDATION")
    print(f"G1_PELVIS_TO_TABLE_FRONT_GAP_M = {g1_report['pelvis_to_table_front_gap_m']:.6f}")
    print(f"TARGET = {g1_report['target_m']:.3f}")
    print(f"ERROR = {g1_report['error_m']:.6f}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
