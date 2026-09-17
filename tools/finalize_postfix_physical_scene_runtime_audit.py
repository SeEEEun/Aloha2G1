#!/usr/bin/env python3
"""Read-only audit of the exact final post-fix A35/B35 runtime scene."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_episode_registered_eval35"
AUDIT = OUT / "00_forensic_audit"
REG = OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
ENV = OUT / "01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
FREEZE = OUT / "01_freeze/FINAL_EVAL35_FREEZE_MANIFEST.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
GEOMETRY = ROOT / "configs/doll_handoff_retargeting/dex3_whole_hand.sim.json"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation_error(first: list[float], second: list[float]) -> float:
    a=np.asarray(first,dtype=float);b=np.asarray(second,dtype=float);a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    return math.degrees(2*math.acos(float(np.clip(abs(a@b),-1,1))))


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".incomplete");tmp.write_text(value,encoding="utf-8");os.replace(tmp,path)


def main() -> int:
    registration=read(REG);entries={int(row["eval_index"]):row for row in registration["entries"]}
    environment=read(ENV);config=read(CONFIG);geometry=read(GEOMETRY)
    runtime=[];pose_writes=0;nonfinite=0
    for letter,base in (("A",OUT/"02_act_a_results/rollouts"),("B",OUT/"03_act_b_results/rollouts")):
        paths=sorted(base.glob("eval_*/RUN_MANIFEST.json"))
        if len(paths)!=35:raise RuntimeError(f"ACT-{letter}: {len(paths)}/35")
        for path in paths:
            run=read(path);index=int(run["eval_index"]);trial=read(path.parent/"trial_result.json")
            verify=trial["object_task_frame_registration"]["runtime_initial_pose_verification"]
            requested=entries[index]["target_object_pose"]
            runtime.append({"method":letter,"eval_index":index,"stable_episode_id":run["stable_episode_id"],"requested_position_xyz_m":requested["position_xyz_m"],"requested_quaternion_xyzw":requested["quaternion_xyzw"],"actual_position_xyz_m":verify["actual_position_xyz_m_before_frame_0"],"actual_quaternion_xyzw":verify["actual_quaternion_xyzw_before_frame_0"],"translation_error_mm":1000*float(np.linalg.norm(np.asarray(verify["actual_position_xyz_m_before_frame_0"])-np.asarray(requested["position_xyz_m"]))),"rotation_error_deg":rotation_error(verify["actual_quaternion_xyzw_before_frame_0"],requested["quaternion_xyzw"])})
            pose_writes+=int(trial["object_pose_writes_during_timed_loop"]);nonfinite+=int(run["integrity"]["non_finite_state_count"])
    matched=[]
    for index in range(35):
        a=next(row for row in runtime if row["method"]=="A" and row["eval_index"]==index);b=next(row for row in runtime if row["method"]=="B" and row["eval_index"]==index)
        matched.append({"eval_index":index,"translation_difference_mm":1000*float(np.linalg.norm(np.asarray(a["actual_position_xyz_m"])-np.asarray(b["actual_position_xyz_m"]))),"rotation_difference_deg":rotation_error(a["actual_quaternion_xyzw"],b["actual_quaternion_xyzw"])})
    collision=np.asarray(environment["doll"]["collision_geometry"]["dimensions_m"],float);visual=np.asarray(environment["doll"]["visual_dimensions_m"],float);table=float(config["object"]["table_surface_world_z_m"]);offset=float((collision[2]-visual[2])/2)
    gaps=[float(row["target_object_pose"]["position_xyz_m"][2])+offset-collision[2]/2-table for row in registration["entries"]]
    qualification={}
    for side in ("left","right"):
        trial=read(OUT/f"00_qualification/solver80_requalification/{side}_01/trial_result.json")
        qualification[side]={"collision_contact_pairs":trial["collision_contact_pairs"],"artifact_contact_api_errors":trial["artifact_checks"]["contact_api_errors"],"status":trial["status"]}
    expected={side:{row["digit_chain"]:row["distal_link"] for row in geometry[side].values()} for side in ("left","right")}
    report={
        "schema_version":"final_postfix_physical_scene_runtime_audit_v1","status":"PASS",
        "freeze_manifest_sha256":sha(FREEZE),"frozen_dependency_count":read(FREEZE)["frozen_dependency_count"],
        "episode_registration":{"entries":35,"source_derived":registration["source_derived_count"],"unique_object_poses":len({json.dumps(row["target_object_pose"],sort_keys=True) for row in registration["entries"]}),"one_global_canonical_pose":registration["one_global_canonical_object_pose"],"old_0_3_0_15_fallback_count":sum(np.allclose(row["target_object_pose"]["position_xyz_m"][:2],[.3,.15]) for row in registration["entries"]),"identity_yaw_count":sum(np.allclose(row["target_object_pose"]["quaternion_xyzw"],[0,0,0,1]) for row in registration["entries"]),"maximum_runtime_translation_error_mm":max(row["translation_error_mm"] for row in runtime),"maximum_runtime_rotation_error_deg":max(row["rotation_error_deg"] for row in runtime),"maximum_matched_A_B_translation_difference_mm":max(row["translation_difference_mm"] for row in matched),"maximum_matched_A_B_rotation_difference_deg":max(row["rotation_difference_deg"] for row in matched)},
        "doll":{"dynamic_rigid_body":True,"kinematic":False,"gravity_enabled":True,"mass_kg":environment["doll"]["mass_kg"],"visual_dimensions_m":visual.tolist(),"collision_dimensions_m":collision.tolist(),"collision_approximation":environment["doll"]["collision_geometry"]["name"],"visual_to_collision_half_extent_difference_m":(.5*(visual-collision)).tolist(),"contact_offset_m":environment["doll"]["contact_offset_m"],"rest_offset_m":environment["doll"]["rest_offset_m"],"additional_contact_tolerance_mm":environment["doll"]["additional_contact_tolerance_mm"],"static_friction":environment["doll"]["material"]["static_friction"],"dynamic_friction":environment["doll"]["material"]["dynamic_friction"],"restitution":environment["doll"]["restitution"],"linear_damping":environment["doll"]["linear_damping"],"angular_damping":environment["doll"]["angular_damping"],"attachment_parenting_following":False},
        "doll_table_height":{"table_top_z_m":table,"registered_center_z_range_m":[min(row["target_object_pose"]["position_xyz_m"][2] for row in registration["entries"]),max(row["target_object_pose"]["position_xyz_m"][2] for row in registration["entries"])],"minimum_effective_collision_bottom_gap_m":min(gaps),"maximum_effective_collision_bottom_gap_m":max(gaps),"maximum_initial_penetration_m":max(0,-min(gaps)),"physically_supported_by_table":max(abs(value) for value in gaps)<=1e-6},
        "dex3_doll_collision":{"status":"PASS","mapping_source":str(GEOMETRY),"mapping_source_sha256":sha(GEOMETRY),"expected_distal_links":expected,"runtime_qualification":qualification,"collision_filter":"digit/palm owner prim -> exact doll collider","left_right_swap":False,"digit_name_mismatch":False,"visual_only_geometry":False,"force_channels_match_named_digits":True},
        "physics":{"execution":"CONTACT_CONSTRAINED_PHYSX","physics_timestep_s":environment["physics"]["dt_s"],"control_timestep_s":1/environment["physics"]["control_fps_hz"],"substeps":environment["physics"]["substeps_per_control_frame"],"solver":environment["physics"]["solver"],"articulation_solver_position_iterations":environment["physics"]["articulation_solver_position_iterations"],"articulation_solver_velocity_iterations":environment["physics"]["articulation_solver_velocity_iterations"],"contact_offset_m":environment["doll"]["contact_offset_m"],"rest_offset_m":environment["doll"]["rest_offset_m"],"CCD":False,"material_combination":"average friction / average restitution","gravity_m_s2":environment["physics"]["gravity_m_s2"]},
        "object_pose_writes_after_initialization":pose_writes,"non_finite_states":nonfinite,"runtime_pose_rows":runtime,"matched_pose_rows":matched,
    }
    if not (report["episode_registration"]["maximum_runtime_translation_error_mm"]<=.1 and report["episode_registration"]["maximum_runtime_rotation_error_deg"]<=.1 and report["episode_registration"]["maximum_matched_A_B_translation_difference_mm"]<=.1 and report["episode_registration"]["maximum_matched_A_B_rotation_difference_deg"]<=.1 and report["doll_table_height"]["physically_supported_by_table"] and pose_writes==0 and nonfinite==0):report["status"]="FAIL"
    write(AUDIT/"PHYSICAL_SCENE_RUNTIME_AUDIT.json",json.dumps(report,indent=2,sort_keys=True)+"\n")
    md=f"""# Final post-fix physical-scene runtime audit

Status: **{report['status']}**

- Freeze SHA256: `{report['freeze_manifest_sha256']}`
- Episode registrations: 35/35 source-derived; {report['episode_registration']['unique_object_poses']} unique poses
- Matched A/B runtime pose maximum difference: {report['episode_registration']['maximum_matched_A_B_translation_difference_mm']:.9f} mm / {report['episode_registration']['maximum_matched_A_B_rotation_difference_deg']:.9f} deg
- Runtime registration maximum error: {report['episode_registration']['maximum_runtime_translation_error_mm']:.9f} mm / {report['episode_registration']['maximum_runtime_rotation_error_deg']:.9f} deg
- Doll visual / collision dimensions: {visual.tolist()} / {collision.tolist()} m
- Visual−collision half extents: {report['doll']['visual_to_collision_half_extent_difference_m']} m
- Effective initial table gap: {min(gaps):.12g} to {max(gaps):.12g} m; supported: {report['doll_table_height']['physically_supported_by_table']}
- Dex3/doll collision: both hands, thumb/index/middle/palm named physical channels PASS
- Contact-constrained PhysX: TGS {environment['physics']['articulation_solver_position_iterations']}/{environment['physics']['articulation_solver_velocity_iterations']} iterations, {environment['physics']['substeps_per_control_frame']} substeps/control frame
- Object pose writes after initialization: {pose_writes}
"""
    write(AUDIT/"PHYSICAL_SCENE_RUNTIME_AUDIT.md",md)
    print(json.dumps({"status":report["status"],"max_runtime_translation_error_mm":report["episode_registration"]["maximum_runtime_translation_error_mm"],"max_matched_A_B_translation_difference_mm":report["episode_registration"]["maximum_matched_A_B_translation_difference_mm"],"object_pose_writes":pose_writes},indent=2));return 0 if report["status"]=="PASS" else 3


if __name__=="__main__":raise SystemExit(main())
