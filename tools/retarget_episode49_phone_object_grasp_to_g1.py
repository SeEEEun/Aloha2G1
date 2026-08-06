#!/usr/bin/env python3
"""Phone-object grasp feasibility gate for episode 49.

Stops with PHONE_GRASP_POSITION_CONFLICT when no requested position tolerance
can satisfy the phone-fixed grasp orientation before temporal retargeting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT/"tools")]
try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pandas"] = types.ModuleType("pandas")

import retarget_episode49_optimized_action_to_g1 as latest  # noqa: E402

SOURCE = ROOT / (
    "converted_runs/smolvla_20k_episode49_consensus_relative_g1/"
    "g1_episode49_consensus_relative_trajectory.npz"
)
BLOCKED = ROOT / (
    "converted_runs/smolvla_20k_episode49_relative_neutral_pinch_g1/"
    "g1_episode49_relative_neutral_pinch_trajectory.npz"
)
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
OUT_ROOT = ROOT / "converted_runs/smolvla_20k_episode49_phone_object_grasp_g1"
REPORT = OUT_ROOT / "g1_episode49_phone_object_grasp_report.json"
TRAJECTORY = OUT_ROOT / "g1_episode49_phone_object_grasp_trajectory.npz"
TOLERANCES = (.005, .010, .020, .030)


def args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=SOURCE)
    p.add_argument("--blocked-neutral",type=Path,default=BLOCKED)
    p.add_argument("--layout",type=Path,default=LAYOUT)
    p.add_argument("--report",type=Path,default=REPORT)
    p.add_argument("--execute",action="store_true")
    return p.parse_args()


def hysteretic_phase(width: np.ndarray) -> np.ndarray:
    closure=np.clip((.044-width)/.044,0,1)
    closed=bool(closure[0]>=.65);phase=np.empty(len(width));previous=float(closed)
    for t,value in enumerate(closure):
        if closed and value<=.45:closed=False
        elif not closed and value>=.65:closed=True
        target=.5+.5*np.clip((value-.45)/.55,0,1) if closed else .5*np.clip(value/.45,0,1)
        previous+=np.clip(target-previous,-.04,.04);phase[t]=previous
    return phase


def phone_and_torso_frames(info: dict, layout: dict) -> dict:
    model=info["model"];data=mujoco.MjData(model)
    latest.ik.assign_arm_qpos(data,info["stand_qpos"],info["arm_qpos_ids"],info["stand_arm_q"])
    mujoco.mj_forward(model,data)
    torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"torso_link")
    torso_rotation=data.xmat[torso].reshape(3,3).copy()
    forward, lateral, up = torso_rotation[:,0],torso_rotation[:,1],torso_rotation[:,2]
    # Phone local +X long -> torso up; local +Z short -> torso lateral;
    # screen local -Y -> torso forward. This is proper right-handed.
    phone_rotation=np.column_stack((up,-forward,lateral))
    left_rotation=np.column_stack((-lateral,np.cross(forward,-lateral),forward))
    right_rotation=np.column_stack((lateral,np.cross(forward,lateral),forward))
    return {
        "torso_rotation":torso_rotation,"forward":forward,"lateral":lateral,"up":up,
        "phone_rotation":phone_rotation,"screen_normal":forward,"long_axis":up,
        "short_axis":lateral,"left_grasp_rotation":left_rotation,
        "right_grasp_rotation":right_rotation,
        "phone_dimensions":np.asarray(layout["phone"]["size_landscape_xyz"],float),
        "phone_initial_center":np.array([
            .5*(layout["phone"]["bottom_left_xy"][0]+layout["phone"]["bottom_right_xy"][0]),
            .5*(layout["phone"]["bottom_left_xy"][1]+layout["phone"]["bottom_right_xy"][1]),
            layout["table"]["surface_height"]+.5*layout["phone"]["size_landscape_xyz"][2],
        ]),
    }


def solve_frame(info: dict, lp: np.ndarray, rp: np.ndarray, q0: np.ndarray,
                left_rotation: np.ndarray, right_rotation: np.ndarray,
                tolerance: float) -> dict:
    data=mujoco.MjData(info["model"])
    def residual(q: np.ndarray) -> np.ndarray:
        state=latest.frame_state(info,data,q)
        lrot=Rotation.from_quat(state["left_quat"][[1,2,3,0]]).as_matrix()
        rrot=Rotation.from_quat(state["right_quat"][[1,2,3,0]]).as_matrix()
        le=state["left_pos"]-lp;re=state["right_pos"]-rp
        # Hinge activates only outside the candidate position tolerance.
        pl=max(0.,np.linalg.norm(le)-tolerance)*le/(np.linalg.norm(le)+1e-12)
        pr=max(0.,np.linalg.norm(re)-tolerance)*re/(np.linalg.norm(re)+1e-12)
        return np.r_[300*pl,300*pr,
                     Rotation.from_matrix(lrot.T@left_rotation).as_rotvec(),
                     Rotation.from_matrix(rrot.T@right_rotation).as_rotvec(),
                     .001*(q-q0)]
    result=least_squares(
        residual,q0,bounds=(info["joint_limits"][:,0],info["joint_limits"][:,1]),
        max_nfev=1500,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    state=latest.frame_state(info,data,result.x)
    lrot=Rotation.from_quat(state["left_quat"][[1,2,3,0]]).as_matrix()
    rrot=Rotation.from_quat(state["right_quat"][[1,2,3,0]]).as_matrix()
    return {
        "left_position_error_m":float(np.linalg.norm(state["left_pos"]-lp)),
        "right_position_error_m":float(np.linalg.norm(state["right_pos"]-rp)),
        "left_orientation_error_deg":float(np.degrees(np.linalg.norm(
            Rotation.from_matrix(lrot.T@left_rotation).as_rotvec()))),
        "right_orientation_error_deg":float(np.degrees(np.linalg.norm(
            Rotation.from_matrix(rrot.T@right_rotation).as_rotvec()))),
        "q":result.x,"nfev":int(result.nfev),"cost":float(result.cost),
        "minimum_joint_limit_margin_rad":float(np.min(np.minimum(
            result.x-info["joint_limits"][:,0],info["joint_limits"][:,1]-result.x))),
    }


def main() -> int:
    a=args()
    for path in (a.source,a.blocked_neutral,a.layout,latest.G1_XML):
        if not path.exists():raise FileNotFoundError(path)
    layout=json.loads(a.layout.read_text())
    with np.load(a.source,allow_pickle=False) as z:
        raw=z["optimized_action"].astype(float)
        lp=z["g1_target_left_position"].astype(float)
        rp=z["g1_target_right_position"].astype(float)
        q=z["g1_arm_joint_trajectory"].astype(float)
    left_phase=hysteretic_phase(raw[:,6]);right_phase=hysteretic_phase(raw[:,13])
    grasp=np.flatnonzero((left_phase>=.95)&(right_phase>=.95))
    if not len(grasp):raise RuntimeError("No automatically detected bilateral phone-grasp phase")
    # First, midpoint, and last are phase-derived, never hard-coded.
    samples=np.unique(np.array([grasp[0],grasp[len(grasp)//2],grasp[-1]],int))
    info=latest.ik.validate_model(latest.G1_XML)
    frames=phone_and_torso_frames(info,layout)
    rows=[]
    for tolerance in TOLERANCES:
        per=[]
        for frame in samples:
            solved=solve_frame(
                info,lp[frame],rp[frame],q[frame],
                frames["left_grasp_rotation"],frames["right_grasp_rotation"],tolerance)
            solved["frame"]=int(frame);per.append(solved)
        passed=all(
            max(x["left_position_error_m"],x["right_position_error_m"])<=tolerance+5e-5
            and max(x["left_orientation_error_deg"],x["right_orientation_error_deg"])<5.
            and x["minimum_joint_limit_margin_rad"]>=-1e-9 for x in per)
        rows.append({"position_tolerance_m":tolerance,"passed":passed,"frames":per})
        if passed:break
    selected=next((x for x in rows if x["passed"]),None)
    verdict="G1_PHONE_OBJECT_GRASP_READY_FOR_VISUAL_REVIEW" if selected else "PHONE_GRASP_POSITION_CONFLICT"
    report={
        "verdict":verdict,"trajectory_generated":False,
        "source":str(a.source.resolve()),"blocked_neutral_source":str(a.blocked_neutral.resolve()),
        "scene_layout_source":str(a.layout.resolve()),
        "phone_prim":"/MagSafeScene/Phone","phone_dimensions_xyz_m":frames["phone_dimensions"].tolist(),
        "phone_local_axes":{"long":"+X","thickness":"+Y","short":"+Z","screen_normal":"-Y"},
        "phone_initial_center_world_m":frames["phone_initial_center"].tolist(),
        "table_surface_z_m":float(layout["table"]["surface_height"]),
        "torso_forward_axis":frames["forward"].tolist(),"torso_lateral_axis":frames["lateral"].tolist(),
        "torso_up_axis":frames["up"].tolist(),"phone_target_rotation":frames["phone_rotation"].tolist(),
        "phone_screen_normal":frames["screen_normal"].tolist(),
        "left_object_grasp_rotation":frames["left_grasp_rotation"].tolist(),
        "right_object_grasp_rotation":frames["right_grasp_rotation"].tolist(),
        "grasp_phase_detection":{
            "method":"ALOHA scalar OPEN/PREGRASP/PINCH Schmitt hysteresis",
            "first_frame":int(grasp[0]),"last_frame":int(grasp[-1]),
            "representative_frames":samples.tolist()},
        "candidate_results":rows,
        "selected_position_tolerance_m":None if selected is None else selected["position_tolerance_m"],
        "conflicting_constraints":[] if selected else [
            "phone-fixed grasp orientation <5 deg",
            "left/right palm position deviation <=30 mm",
            "G1 wrist joint limits",
        ],
        "stop_reason":None if selected else (
            "At 30 mm the requested phone-fixed wrist orientation remains above 5 deg "
            "on automatically selected grasp-phase frames. Temporal trajectory, phone proxy, "
            "and review videos were intentionally not fabricated."
        ),
        "isaac_lab_executed":False,"hardware_executed":False,
    }
    # Remove large q vectors from JSON while retaining all diagnostics.
    for candidate in report["candidate_results"]:
        for frame in candidate["frames"]:frame["q"]=frame["q"].tolist()
    a.report.parent.mkdir(parents=True,exist_ok=True)
    if a.execute:
        if TRAJECTORY.exists():
            raise FileExistsError(TRAJECTORY)
        tmp=a.report.with_suffix(".json.incomplete")
        tmp.write_text(json.dumps(report,indent=2));os.replace(tmp,a.report)
    print(json.dumps(report,indent=2))
    return 0 if selected else 2


if __name__=="__main__":
    raise SystemExit(main())
