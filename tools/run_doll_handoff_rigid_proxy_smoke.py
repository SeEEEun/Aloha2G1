#!/usr/bin/env python3
"""Deferred Isaac smoke test for the policy-independent rigid doll proxy.

This checks one material candidate, gravity, table settling, finite motion, and
the configured object parameters.  It does not import or execute any policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCENE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
DOLL = "/World/DollHandoffEnvironment/Doll"
DOLL_BODY = f"{DOLL}/Body"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--config", type=Path, default=ROOT / "configs/doll_handoff_rigid_proxy_v1.json"
)
parser.add_argument("--material", choices=("LOW", "MEDIUM", "HIGH"))
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seconds", type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import omni.usd
import torch
from pxr import PhysxSchema, Sdf, UsdPhysics, UsdShade
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from tools.evaluation.contracts import sha256_file
from tools.evaluation.io import atomic_json


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_proxy(stage: Any, config: dict[str, Any], candidate: dict[str, Any]) -> None:
    doll = stage.GetPrimAtPath(DOLL)
    body = stage.GetPrimAtPath(DOLL_BODY)
    if not doll.IsValid() or not body.IsValid():
        raise RuntimeError("rigid doll prims are missing")
    material = UsdShade.Material.Define(stage, "/World/RigidDollProxyPhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(candidate["static_friction"]))
    api.CreateDynamicFrictionAttr(float(candidate["dynamic_friction"]))
    api.CreateRestitutionAttr(float(config["object"]["restitution"]))
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set(
        str(config["material_combine_modes"]["friction"])
    )
    physx_material.CreateRestitutionCombineModeAttr().Set(
        str(config["material_combine_modes"]["restitution"])
    )
    UsdShade.MaterialBindingAPI.Apply(body).Bind(material, materialPurpose="physics")
    UsdPhysics.MassAPI.Apply(doll).CreateMassAttr(float(config["object"]["mass_kg"]))
    doll.AddAppliedSchema("PhysxRigidBodyAPI")
    doll.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(
        float(config["object"]["linear_damping"])
    )
    doll.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(
        float(config["object"]["angular_damping"])
    )
    doll.CreateAttribute(
        "physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float
    ).Set(float(config["simulation"]["contact_settings"]["max_depenetration_velocity_m_s"]))
    body.AddAppliedSchema("PhysxCollisionAPI")
    body.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(
        float(config["simulation"]["contact_settings"]["contact_offset_m"])
    )
    body.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
        float(config["simulation"]["contact_settings"]["rest_offset_m"])
    )


def run() -> int:
    config_path = args.config.resolve()
    config = _read(config_path)
    if config.get("schema_version") != "doll_handoff_rigid_proxy_v1":
        raise RuntimeError("unexpected rigid proxy config")
    if Path(config["source_scene"]["g1_preview_usd"]).resolve() != SCENE.resolve():
        raise RuntimeError("smoke runner scene differs from rigid-proxy config")
    selected = config.get("selected_material_candidate")
    material_name = args.material or selected
    if material_name is None:
        raise RuntimeError("precalibration smoke requires an explicit --material candidate")
    if bool(config.get("freeze", {}).get("frozen")) and material_name != selected:
        raise RuntimeError("a frozen config may only use its selected material")
    candidates = {row["name"]: row for row in config["material_candidates"]}
    candidate = candidates[material_name]
    duration_s = float(args.seconds if args.seconds is not None else config["smoke_test"]["duration_s"])
    if duration_s <= 0.0:
        raise ValueError("--seconds must be positive")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output}")
    output.mkdir(parents=True)
    if not omni.usd.get_context().open_stage(str(SCENE)):
        raise RuntimeError(f"failed to open {SCENE}")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    _apply_proxy(stage, config, candidate)
    dt = float(config["simulation"]["physics_dt_s"])
    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            device=args.device,
            gravity=(0.0, 0.0, -float(config["simulation"]["gravity_m_s2"])),
            use_fabric=True,
        )
    )
    doll = RigidObject(RigidObjectCfg(prim_path=DOLL, spawn=None))
    sim.reset()
    center = np.asarray(config["object"]["initial_center_world_xyz_m"], dtype=np.float32)
    # IsaacLab root-state pose order is XYZ + quaternion WXYZ.
    pose = torch.as_tensor(
        np.r_[center, [1.0, 0.0, 0.0, 0.0]][None],
        dtype=torch.float32,
        device=doll.device,
    )
    doll.write_root_pose_to_sim_index(root_pose=pose)
    doll.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=doll.device)
    )
    positions = []
    velocities = []
    steps = max(1, int(round(duration_s / dt)))
    for _ in range(steps):
        sim.step(render=False)
        doll.update(dt)
        positions.append(doll.data.root_pos_w.torch[0].detach().cpu().numpy().copy())
        velocities.append(doll.data.root_vel_w.torch[0].detach().cpu().numpy().copy())
    positions_array = np.asarray(positions, dtype=np.float64)
    velocity_array = np.asarray(velocities, dtype=np.float64)
    trajectory = output / "smoke_trace.npz"
    temporary = trajectory.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            object_com_m=positions_array,
            object_velocity_m_s=velocity_array,
            timestamp_s=np.arange(steps, dtype=np.float64) * dt,
        )
    os.replace(temporary, trajectory)
    radius = float(config["object"]["collision_dimensions_m"]["radius"])
    expected_z = float(config["scene_fairness"]["table_surface_world_z_m"]) + radius
    smoke = config["smoke_test"]
    tail = positions_array[
        -max(2, int(round(float(smoke["settle_window_duration_s"]) / dt))) :, 2
    ]
    displacement = np.linalg.norm(np.diff(positions_array, axis=0), axis=1)
    tolerance = float(smoke["table_supported_center_height_tolerance_m"])
    checks = {
        "finite_state": bool(
            np.isfinite(positions_array).all() and np.isfinite(velocity_array).all()
        ),
        "settled_on_table": bool(abs(float(positions_array[-1, 2]) - expected_z) <= tolerance),
        "settled_tail_span_m": bool(
            float(np.ptp(tail))
            <= float(smoke["settle_window_center_z_span_max_m"])
        ),
        "no_teleportation": bool(
            not len(displacement)
            or float(np.max(displacement))
            <= float(config["success_thresholds"]["maximum_allowed_object_com_step_m"])
        ),
        "source_scene_unchanged": True,
        "no_policy_or_checkpoint_used": True,
    }
    report = {
        "schema_version": "doll_handoff_rigid_proxy_smoke_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "material_candidate": material_name,
        "material": candidate,
        "scene": str(SCENE),
        "scene_sha256": sha256_file(SCENE),
        "trace": str(trajectory),
        "trace_sha256": sha256_file(trajectory),
        "initial_object_com_m": positions_array[0].tolist(),
        "final_object_com_m": positions_array[-1].tolist(),
        "expected_table_supported_center_z_m": expected_z,
        "maximum_object_com_step_m": float(np.max(displacement)) if len(displacement) else 0.0,
        "policy_specific_logic": False,
        "isaac_executed": True,
    }
    atomic_json(output / "smoke_result.json", report)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    code = 1
    try:
        code = run()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(code)
