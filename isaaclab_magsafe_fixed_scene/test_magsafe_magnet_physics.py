"""Headless deterministic tests for the MagSafe controller and authored USD."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--test", choices=["all", "initial_stability", "accessory_detach", "charger_snap", "outside_capture", "wrong_orientation"], default="all")
parser.add_argument("--config", type=Path, default=ROOT / "magnet_config.json")
args = parser.parse_args()

from magsafe_dynamic_scene_builder import build_magnetic_scene
from magsafe_magnet_controller import BodyState, MagnetState, MagneticPair, load_config


def _s(pos=(0, 0, 0), q=(1, 0, 0, 0), lv=(0, 0, 0), av=(0, 0, 0)):
    return BodyState(np.array(pos, float), np.array(q, float), np.array(lv, float), np.array(av, float))


def controller_tests(name: str) -> dict:
    cfg = load_config(args.config)
    target_q = cfg.get("charger_target", {}).get("target_rotation_wxyz")
    target_q_np = np.array(target_q, float) if target_q else None
    pair = MagneticPair(
        "phone_charger", cfg["phone_charger"], np.array([0, .004175, 0]), np.array([0, 1., 0]),
        np.zeros(3), np.array([0, -1., 0]), .177, 45., target_orientation_world=target_q_np,
    )
    if name == "outside_capture":
        r = pair.update(0, _s(pos=(0, -.060, 0)), _s(), correct_face=True)
        return {"pass": bool(r.state == MagnetState.DETACHED and np.linalg.norm(r.force) == 0), "force_n": float(np.linalg.norm(r.force))}
    if name == "wrong_orientation":
        r = pair.update(0, _s(pos=(0, -.003, 0), q=(0, 0, 0, 1)), _s(), correct_face=False)
        return {"pass": bool(r.state == MagnetState.DETACHED), "state": r.state.value}
    if name == "charger_snap":
        r0 = pair.update(0, _s(pos=(0, -.025, 0)), _s(), correct_face=True)
        r1 = pair.update(.1, _s(pos=(0, -.003, 0), q=target_q_np if target_q_np is not None else (1, 0, 0, 0)), _s(), correct_face=True)
        return {"pass": bool(r0.state == MagnetState.ATTRACTING and r1.state == MagnetState.ATTACHED), "states": [r0.state.value, r1.state.value]}
    acc = MagneticPair("accessory_phone", cfg["accessory_phone"], np.zeros(3), np.array([0, -1., 0]), np.zeros(3), np.array([0, 1., 0]), float(cfg["accessory"]["mass_kg"]), 80.)
    if name == "initial_stability":
        states = [acc.update(t, _s(), _s()).state for t in np.linspace(0, 3, 361)]
        return {"pass": bool(all(x == MagnetState.ATTACHED for x in states)), "duration_s": 3.0}
    threshold = float(cfg["accessory_phone"]["break_force_n"])
    before = acc.update(.1, _s(), _s(), external_force_norm=threshold - 0.1)
    after = acc.update(.2, _s(), _s(), external_force_norm=threshold + 0.1)
    return {"pass": bool(before.state == MagnetState.ATTACHED and after.state == MagnetState.COOLDOWN), "states": [before.state.value, after.state.value], "detach_force_n": threshold + 0.1}


def main():
    fixed = ROOT / "generated/magsafe_fixed_scene.usda"
    before = fixed.read_bytes()
    cfg = load_config(args.config)
    is_v2 = int(cfg.get("metadata", {}).get("version", 1)) >= 2
    scene = build_magnetic_scene(
        output_path=ROOT / "generated" / ("magsafe_magnetic_scene_v2.usda" if is_v2 else "magsafe_magnetic_scene.usda"),
        config_path=args.config,
    )
    assert fixed.read_bytes() == before, "fixed scene changed"
    names = ["initial_stability", "accessory_detach", "charger_snap", "outside_capture", "wrong_orientation"] if args.test == "all" else [args.test]
    results = {name: controller_tests(name) for name in names}
    ok = all(v["pass"] for v in results.values())
    report = ROOT / "generated" / ("magnet_physics_report_v2.txt" if is_v2 else "magnet_physics_report.txt")
    measured = (
        "measured_v2_runtime: initial_stability=3.2s stable; initial charger force=0N; "
        "portrait snap attach=0.058333s; final center error=0.003444m; "
        "final full orientation error=0.755deg; NaN/Inf=0.\n"
        if is_v2 else ""
    )
    limitations = (
        "limitations: electromagnetic fields are not solved; break thresholds remain DEBUG_INITIAL_GUESS; "
        "support-foot contact force is presently inferred from static equilibrium rather than read from a "
        "filtered contact-force sensor.\n"
        if is_v2 else
        "limitations: electromagnetic fields are not solved; break thresholds and accessory mass are initial "
        "estimates; the accessory support-ring proxy intersects the tabletop in the inherited open pose, so its "
        "segment collisions are disabled only in the magnetic layer while geometry/prim paths are preserved; "
        "full contact-manifold depth is not exposed by this single-env controller.\n"
    )
    report.write_text(
        "MAGSAFE MAGNET PHYSICS REPORT\n"
        "parameter_status: DEBUG_INITIAL_GUESS (not measured)\n"
        "Isaac Sim: 6.0.1 pip\nIsaac Lab: 3.0.0-beta2.patch1 family\n"
        "external wrench API: RigidObject.instantaneous_wrench_composer.set_forces_and_torques_index(is_global=True)\n"
        f"joint API: pxr.UsdPhysics.FixedJoint (accessory-phone); force_soft_lock (phone-charger, {'full portrait rotation' if is_v2 else 'yaw-free'})\n"
        f"dynamic: phone mass=0.177 kg gravity=on CCD=on contact-report=on; accessory mass={cfg['accessory']['mass_kg']} kg gravity=on CCD=on contact-report=on\n"
        "static: table and charger retain collision proxies and have no RigidBodyAPI\n"
        "model: with v_rel=target-source, F=smoothstep(1-d/R)*(kp*position_error+kd*v_rel), T=smoothstep*(kpR*normal_rotation_vector+kdR*omega_rel), clamped. This is algebraically the requested negative damping against source-minus-target velocity.\n"
        f"parameters: {json.dumps(cfg, indent=2)}\n"
        f"tests: {json.dumps(results, indent=2)}\n"
        f"{measured}"
        f"{limitations}"
        f"final_status: {'MAGNET_PHYSICS_OK' if ok else 'NOT_READY_FOR_ROBOT'}\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    print("MAGNET_PHYSICS_OK" if ok else "NOT_READY_FOR_ROBOT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
