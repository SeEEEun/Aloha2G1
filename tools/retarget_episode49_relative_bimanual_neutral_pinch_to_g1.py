#!/usr/bin/env python3
"""Add neutral wrist calibration, torso clearance, and Dex3 two-finger pinch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

import retarget_episode49_consensus_relative_bimanual_to_g1 as relative

latest = relative.latest
ROOT = Path("/home/jbnu/aloha_g1_dataset")
SOURCE = ROOT / (
    "converted_runs/smolvla_20k_episode49_consensus_relative_g1/"
    "g1_episode49_consensus_relative_trajectory.npz"
)
OUT = ROOT / (
    "converted_runs/smolvla_20k_episode49_relative_neutral_pinch_g1/"
    "g1_episode49_relative_neutral_pinch_trajectory.npz"
)
CLEARANCE_M = .005
CLEARANCE_ACTIVATION_M = .010
ORIENTATION_WEIGHT = .02
TOOL_AXIS_WEIGHT = .10
PALM_AXIS_WEIGHT = .05
OPEN_THRESHOLD = .45
PINCH_THRESHOLD = .65
PHASE_SLEW = .04


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--iterations", type=int, default=12)
    return p.parse_args()


def hand_joint_schema(info: dict) -> tuple[dict, dict]:
    model, hands, arm_addr, actuator_names, actuator_joints = latest.model_layout(
        latest.G1_XML, info["joint_names"]
    )
    schema = {}
    for side in ("left", "right"):
        h = hands[side]
        schema[side] = {
            name: {
                "index": i, "range": h["ranges"][i].copy(),
                "axis": model.jnt_axis[int(h["jids"][i])].copy(),
            }
            for i, name in enumerate(h["names"])
        }
    return {
        "model": model, "hands": hands, "arm_addr": arm_addr,
        "actuator_names": actuator_names, "actuator_joints": actuator_joints,
    }, schema


def set_named(pose: np.ndarray, schema: dict, name: str, value: float) -> None:
    rec = schema[name]
    lo, hi = rec["range"]
    if not lo <= value <= hi:
        raise ValueError(f"{name}={value} outside [{lo},{hi}]")
    pose[rec["index"]] = value


def pinch_poses(layout: dict, schema: dict) -> dict:
    """Fixed mirror-symmetric poses, defined by XML joint names and signs."""
    old = latest.poses(layout["model"], layout["hands"])
    result = {}
    for side in ("left", "right"):
        sign = -1.0 if side == "left" else 1.0
        open_pose = old[side][0].copy()
        pre = open_pose.copy()
        pinch = open_pose.copy()
        prefix = f"{side}_hand_"
        # thumb_0 opposition is common-sign; thumb_1/2 and fingers are mirrored.
        for pose, values in (
            (pre, {
                "thumb_0_joint": -.12, "thumb_1_joint": -sign*.62,
                "thumb_2_joint": -sign*.62, "index_0_joint": sign*.32,
                "index_1_joint": sign*.38, "middle_0_joint": sign*.08,
                "middle_1_joint": sign*.10,
            }),
            (pinch, {
                "thumb_0_joint": -.40, "thumb_1_joint": -sign*.80,
                "thumb_2_joint": -sign*1.10, "index_0_joint": sign*1.30,
                "index_1_joint": sign*.90, "middle_0_joint": sign*.12,
                "middle_1_joint": sign*.16,
            }),
        ):
            for suffix, value in values.items():
                set_named(pose, schema[side], prefix+suffix, value)
        result[side] = {"open": open_pose, "pregrasp": pre, "pinch": pinch}
    return result


def hysteretic_phase(width: np.ndarray) -> tuple[np.ndarray, dict]:
    closure = np.clip((.044-width)/.044, 0, 1)
    closed = bool(closure[0] >= PINCH_THRESHOLD)
    phase = np.empty(len(closure))
    state_transitions = []
    previous = 1.0 if closed else min(.5, .5*closure[0]/OPEN_THRESHOLD)
    for t, value in enumerate(closure):
        new_closed = closed
        if closed and value <= OPEN_THRESHOLD:
            new_closed = False
        elif not closed and value >= PINCH_THRESHOLD:
            new_closed = True
        if new_closed != closed:
            state_transitions.append(t)
        closed = new_closed
        if closed:
            target = .5+.5*np.clip((value-OPEN_THRESHOLD)/(1-OPEN_THRESHOLD), 0, 1)
        else:
            target = .5*np.clip(value/OPEN_THRESHOLD, 0, 1)
        previous += np.clip(target-previous, -PHASE_SLEW, PHASE_SLEW)
        phase[t] = previous
    transitions = {
        "hysteresis_state": state_transitions,
        "enter_pregrasp": np.flatnonzero((phase[:-1] < .5) & (phase[1:] >= .5)).astype(int).tolist(),
        "enter_pinch": np.flatnonzero((phase[:-1] < .95) & (phase[1:] >= .95)).astype(int).tolist(),
        "leave_pinch": np.flatnonzero((phase[:-1] >= .95) & (phase[1:] < .95)).astype(int).tolist(),
    }
    return phase, {"closure": closure, "transitions": transitions}


def interpolate_pinch(phase: np.ndarray, poses: dict) -> np.ndarray:
    out = np.empty((len(phase), len(poses["open"])))
    low = phase <= .5
    u = np.clip(phase[low]/.5, 0, 1)[:, None]
    out[low] = poses["open"]+u*(poses["pregrasp"]-poses["open"])
    u = np.clip((phase[~low]-.5)/.5, 0, 1)[:, None]
    out[~low] = poses["pregrasp"]+u*(poses["pinch"]-poses["pregrasp"])
    return out


def neutral_reference(info: dict, start_arm: np.ndarray) -> dict:
    data = mujoco.MjData(info["model"])
    natural_state = latest.frame_state(info, data, start_arm)
    neutral_q=start_arm.copy()
    neutral_q[[4,5,6,11,12,13]]=0.0
    state=latest.frame_state(info,data,neutral_q)
    refs = {}
    for side in ("left", "right"):
        natural = Rotation.from_quat(natural_state[f"{side}_quat"][[1,2,3,0]]).as_matrix()
        rot = Rotation.from_quat(state[f"{side}_quat"][[1,2,3,0]]).as_matrix()
        refs[side] = {
            "rotation": rot, "tool_forward": rot[:, 0],
            "palm_axis": rot[:, 1], "palm_normal": rot[:, 2],
            "natural_rotation": natural,
        }
    return refs


def geom_groups(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    torso, protected = [], []
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])
        ) or ""
        if body == "torso_link":
            torso.append(gid)
        if body.startswith(("left_hand_thumb", "left_hand_index", "left_hand_middle",
                            "right_hand_thumb", "right_hand_index", "right_hand_middle")):
            protected.append(gid)
    return np.asarray(torso, int), np.asarray(protected, int)


def minimum_torso_clearance(model: mujoco.MjModel, data: mujoco.MjData,
                            torso: np.ndarray, fingers: np.ndarray) -> float:
    return min(
        float(mujoco.mj_geomDistance(model, data, int(a), int(b), .25, None))
        for a in torso for b in fingers
    )


def assign_full(data: mujoco.MjData, info: dict, q: np.ndarray, left: np.ndarray,
                right: np.ndarray, layout: dict) -> None:
    data.qpos[:] = info["stand_qpos"]
    data.qpos[info["arm_qpos_ids"]] = q
    data.qpos[layout["hands"]["left"]["qadr"]] = left
    data.qpos[layout["hands"]["right"]["qadr"]] = right
    data.qvel[:] = 0
    mujoco.mj_forward(info["model"], data)


def add_block(rows, cols, vals, rhs, row: int, col: int,
              matrix: np.ndarray, residual: np.ndarray) -> int:
    rr, cc = np.nonzero(matrix)
    rows.extend((row+rr).tolist()); cols.extend((col+cc).tolist())
    vals.extend(matrix[rr,cc].tolist()); rhs.extend(np.asarray(residual).tolist())
    return row+len(residual)


def solve(info: dict, targets: dict, initial: np.ndarray, nominal: np.ndarray,
          refs: dict, left_hand: np.ndarray, right_hand: np.ndarray,
          layout: dict, iterations: int) -> np.ndarray:
    """Latest position-temporal objective plus calibrated axes and geom clearance."""
    q = initial.copy(); n, d = q.shape
    limits = info["joint_limits"]; model = info["model"]
    torso, fingers = geom_groups(model)
    data = mujoco.MjData(model)
    wp, wr, wv, wa, wn = 3.0, .80, .018, .030, .001
    for iteration in range(iterations):
        rows=[]; cols=[]; vals=[]; rhs=[]; row=0
        active_clearance = 0
        for t in range(n):
            assign_full(data, info, q[t], left_hand[t], right_hand[t], layout)
            state = latest.ik.current_bimanual_state(info, data)
            jl=np.hstack((state["left_jac"][:3],np.zeros((3,7))))
            jr=np.hstack((np.zeros((3,7)),state["right_jac"][:3]))
            row=add_block(rows,cols,vals,rhs,row,t*d,wp*jl,wp*(targets["lp"][t]-state["left_pos"]))
            row=add_block(rows,cols,vals,rhs,row,t*d,wp*jr,wp*(targets["rp"][t]-state["right_pos"]))
            row=add_block(rows,cols,vals,rhs,row,t*d,wr*(jr-jl),
                          wr*((targets["rp"][t]-targets["lp"][t])-(state["right_pos"]-state["left_pos"])))
            # Full neutral is weakest; calibrated tool and palm axes receive the
            # higher partial-orientation priorities.
            for side, offset in (("left",0),("right",7)):
                current = Rotation.from_quat(state[f"{side}_quat"][[1,2,3,0]]).as_matrix()
                jacr = state[f"{side}_jac"][3:]
                block=np.zeros((3,14)); block[:,offset:offset+7]=jacr
                rot_error=latest.ik.rotation_log_world(current,refs[side]["rotation"])
                row=add_block(rows,cols,vals,rhs,row,t*d,ORIENTATION_WEIGHT*block,
                              ORIENTATION_WEIGHT*rot_error)
                for axis,weight in ((0,TOOL_AXIS_WEIGHT),(1,PALM_AXIS_WEIGHT)):
                    axis_error=np.cross(current[:,axis],refs[side]["rotation"][:,axis])
                    row=add_block(rows,cols,vals,rhs,row,t*d,weight*block,weight*axis_error)
            clearance = minimum_torso_clearance(model,data,torso,fingers)
            if clearance < CLEARANCE_ACTIVATION_M:
                grad=np.zeros(d); eps=2e-5
                for j in range(d):
                    qp=q[t].copy(); qp[j]+=eps
                    assign_full(data,info,qp,left_hand[t],right_hand[t],layout)
                    grad[j]=(minimum_torso_clearance(model,data,torso,fingers)-clearance)/eps
                weight=1.5
                row=add_block(rows,cols,vals,rhs,row,t*d,weight*grad[None],
                              np.array([weight*(CLEARANCE_M-clearance)]))
                active_clearance += 1
            row=add_block(rows,cols,vals,rhs,row,t*d,wn*np.eye(d),wn*(nominal-q[t]))
        eye=np.eye(d)
        for t in range(1,n):
            row=add_block(rows,cols,vals,rhs,row,(t-1)*d,np.hstack((-wv*eye,wv*eye)),-wv*(q[t]-q[t-1]))
        for t in range(1,n-1):
            row=add_block(rows,cols,vals,rhs,row,(t-1)*d,np.hstack((wa*eye,-2*wa*eye,wa*eye)),
                          -wa*(q[t+1]-2*q[t]+q[t-1]))
        A=coo_matrix((vals,(rows,cols)),shape=(row,n*d)).tocsr()
        dq=lsqr(A,np.asarray(rhs),damp=2e-3,atol=2e-5,btol=2e-5,iter_lim=180)[0].reshape(n,d)
        alpha=.45 if iteration<2 else .7
        q=np.minimum(np.maximum(q+alpha*np.clip(dq,-.035,.035),limits[:,0]),limits[:,1])
        print(f"neutral-clearance temporal {iteration+1}/{iterations}; "
              f"active_clearance={active_clearance}",flush=True)
    return q


def orientation_seed(info: dict, targets: dict, position_q: np.ndarray,
                     nominal: np.ndarray, refs: dict) -> np.ndarray:
    """Continuation seed only; the final result is still the temporal solve."""
    data=mujoco.MjData(info["model"]);out=np.empty_like(position_q);prev=position_q[0]
    lquat=latest.ik.mat_to_quat_wxyz(refs["left"]["rotation"])
    rquat=latest.ik.mat_to_quat_wxyz(refs["right"]["rotation"])
    for t in range(len(out)):
        result=latest.work.solve_frame(
            info,data,targets["lp"][t],targets["rp"][t],lquat,rquat,
            position_q[t],prev,nominal,.08,140)
        out[t]=result["q"];prev=out[t]
    return out


def kinematics(info: dict, q: np.ndarray, refs: dict) -> dict:
    data=mujoco.MjData(info["model"]); n=len(q)
    rotations={s:np.empty((n,3,3)) for s in ("left","right")}
    positions={s:np.empty((n,3)) for s in ("left","right")}
    for t in range(n):
        state=latest.frame_state(info,data,q[t])
        for side in ("left","right"):
            positions[side][t]=state[f"{side}_pos"]
            rotations[side][t]=Rotation.from_quat(state[f"{side}_quat"][[1,2,3,0]]).as_matrix()
    result={}
    for side in ("left","right"):
        rot=rotations[side]
        result[f"{side}_rotation"]=rot
        result[f"{side}_tool"]=rot[:,:,0];result[f"{side}_palm_axis"]=rot[:,:,1]
        result[f"{side}_normal"]=rot[:,:,2]
        result[f"{side}_orientation_error"]=latest.rotation_error(
            rot,np.repeat(refs[side]["rotation"][None],n,axis=0))
        result[f"{side}_position"]=positions[side]
    return result


def fingertip_metrics(model: mujoco.MjModel, qpos: np.ndarray) -> dict:
    data=mujoco.MjData(model); out={}
    for side in ("left","right"):
        ids={name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f"{side}_hand_{name}_link")
             for name in ("thumb_2","index_1","middle_1")}
        ti=np.empty(len(qpos));tm=np.empty(len(qpos))
        for t,q in enumerate(qpos):
            data.qpos[:]=q;data.qvel[:]=0;mujoco.mj_forward(model,data)
            ti[t]=np.linalg.norm(data.xpos[ids["thumb_2"]]-data.xpos[ids["index_1"]])
            tm[t]=np.linalg.norm(data.xpos[ids["thumb_2"]]-data.xpos[ids["middle_1"]])
        out[f"{side}_thumb_index_aperture"]=ti
        out[f"{side}_third_finger_aperture"]=tm
    return out


def main() -> int:
    a=args()
    if a.output.exists(): raise FileExistsError(a.output)
    with np.load(a.source,allow_pickle=False) as z:
        required=("optimized_action","g1_target_left_position","g1_target_right_position",
                  "g1_start_arm_q","g1_start_qpos")
        missing=[k for k in required if k not in z.files]
        if missing: raise RuntimeError(f"missing source keys {missing}")
        raw=z["optimized_action"].astype(float)
        targets={"lp":z["g1_target_left_position"].astype(float),
                 "rp":z["g1_target_right_position"].astype(float)}
        start_arm=z["g1_start_arm_q"].astype(float); start_qpos=z["g1_start_qpos"].astype(float)
    if a.max_frames: raw=raw[:a.max_frames]; targets={k:v[:a.max_frames] for k,v in targets.items()}
    n=len(raw); targets["lr"]=np.repeat(np.eye(3)[None],n,axis=0);targets["rr"]=targets["lr"].copy()
    info=latest.ik.validate_model(latest.G1_XML)
    layout,schema=hand_joint_schema(info); pose=pinch_poses(layout,schema)
    phases={}; phase_info={}; task_hand={}
    for side,j in (("left",6),("right",13)):
        phases[side],phase_info[side]=hysteretic_phase(raw[:,j])
        task_hand[side]=interpolate_pinch(phases[side],pose[side])
    refs=neutral_reference(info,start_arm)
    position_seed=latest.position_seed(info,targets,start_arm)
    seed=position_seed
    task_arm=solve(info,targets,seed,start_arm,refs,task_hand["left"],task_hand["right"],layout,a.iterations)
    kin=kinematics(info,task_arm,refs)
    le=np.linalg.norm(kin["left_position"]-targets["lp"],axis=1)
    re=np.linalg.norm(kin["right_position"]-targets["rp"],axis=1)
    relerr=np.linalg.norm((kin["right_position"]-kin["left_position"])-(targets["rp"]-targets["lp"]),axis=1)
    base=layout["model"].key_qpos[0].copy()
    app_arm=latest.approach_mod.minimum_jerk(info["stand_arm_q"],task_arm[0],latest.APPROACH)
    app_l=latest.approach_mod.minimum_jerk(pose["left"]["open"],task_hand["left"][0],latest.APPROACH)
    app_r=latest.approach_mod.minimum_jerk(pose["right"]["open"],task_hand["right"][0],latest.APPROACH)
    hold_arm=np.repeat(task_arm[:1],latest.HOLD,axis=0)
    hold_l=np.repeat(task_hand["left"][:1],latest.HOLD,axis=0);hold_r=np.repeat(task_hand["right"][:1],latest.HOLD,axis=0)
    full_arm=np.vstack((app_arm,hold_arm,task_arm));full_l=np.vstack((app_l,hold_l,task_hand["left"]))
    full_r=np.vstack((app_r,hold_r,task_hand["right"]))
    full_q=latest.full_qpos(layout["model"],base,full_arm,full_l,full_r,layout["arm_addr"],layout["hands"])
    collision,cross,pairs=relative.actual_full_qpos_contacts(layout["model"],full_q)
    torso,fingers=geom_groups(layout["model"]);data=mujoco.MjData(layout["model"])
    clearance=np.empty(len(full_q))
    for t,q in enumerate(full_q):
        data.qpos[:]=q;data.qvel[:]=0;mujoco.mj_forward(layout["model"],data)
        clearance[t]=minimum_torso_clearance(layout["model"],data,torso,fingers)
    apertures=fingertip_metrics(layout["model"],full_q)
    lim=info["joint_limits"];viol=(full_arm<lim[:,0]-1e-9)|(full_arm>lim[:,1]+1e-9)
    failed=(le>.005)|(re>.005);finite=all(np.isfinite(x).all() for x in (full_q,task_arm,clearance))
    step=np.abs(np.diff(full_arm,axis=0));vel=step*latest.FPS;acc=np.abs(np.diff(full_arm,n=2,axis=0))*latest.FPS**2
    norms=np.linalg.norm(np.diff(task_arm,axis=0),axis=1);branch=np.zeros(n,bool)
    for t in range(1,n):
        local=np.median(norms[max(0,t-10):min(len(norms),t+9)])
        branch[t]=norms[t-1]>max(.15,8*max(local,1e-5))
    boundary=float(np.max(np.abs(hold_arm[-1]-task_arm[0])))
    safety=bool(not failed.any() and not branch.any() and not viol.any() and not collision.any()
                and not cross.any() and finite and step.max(initial=0)<=latest.LIMITS["step"]
                and vel.max(initial=0)<=latest.LIMITS["velocity"]
                and acc.max(initial=0)<=latest.LIMITS["acceleration"] and boundary==0)
    def deg_stats(x): return {k:float(np.degrees(v)) for k,v in latest.stats(x).items()}
    report={
        "verdict":"G1_NEUTRAL_PINCH_READY_FOR_VISUAL_REVIEW" if safety else "G1_NEUTRAL_PINCH_SAFETY_BLOCKED",
        "safety_pass":safety,"source":str(a.source),"position_targets_reused_without_modification":True,
        "wrist_links":{"left":"left_wrist_yaw_link","right":"right_wrist_yaw_link"},
        "tool_forward_axis_local":[1,0,0],"palm_axis_local":[0,1,0],"palm_normal_axis_local":[0,0,1],
        "dex3_joint_schema":{s:{k:{"range":v["range"].tolist(),"axis":v["axis"].tolist()}
                                  for k,v in schema[s].items()} for s in schema},
        "dex3_poses":{s:{k:v.tolist() for k,v in pose[s].items()} for s in pose},
        "gripper_mapping":{"open_threshold":OPEN_THRESHOLD,"pinch_threshold":PINCH_THRESHOLD,
                           "phase_slew_per_frame":PHASE_SLEW,
                           "transitions":{s:phase_info[s]["transitions"] for s in phase_info}},
        "orientation_weights":{"neutral":ORIENTATION_WEIGHT,"tool_forward":TOOL_AXIS_WEIGHT,
                               "palm_axis":PALM_AXIS_WEIGHT},
        "torso_clearance_constraint_m":CLEARANCE_M,"clearance_activation_m":CLEARANCE_ACTIVATION_M,
        "task250_290_min_torso_clearance_m":float(clearance[105+250:105+291].min()),
        "full_min_torso_clearance_m":float(clearance.min()),
        "left_position_error_mm":{k:v*1000 for k,v in latest.stats(le).items()},
        "right_position_error_mm":{k:v*1000 for k,v in latest.stats(re).items()},
        "bimanual_relative_error_mm":{k:v*1000 for k,v in latest.stats(relerr).items()},
        "wrist_neutral_orientation_error_deg":{"left":deg_stats(kin["left_orientation_error"]),
                                                "right":deg_stats(kin["right_orientation_error"])},
        "palm_normal_frame_change_deg":{
            s:deg_stats(np.arccos(np.clip(np.sum(kin[f"{s}_normal"][1:]*kin[f"{s}_normal"][:-1],axis=1),-1,1)))
            for s in ("left","right")},
        "pinch_axis_frame_change_deg":{
            s:deg_stats(np.arccos(np.clip(np.sum(kin[f"{s}_tool"][1:]*kin[f"{s}_tool"][:-1],axis=1),-1,1)))
            for s in ("left","right")},
        "thumb_index_aperture_m":{s:latest.stats(apertures[f"{s}_thumb_index_aperture"]) for s in ("left","right")},
        "third_finger_aperture_m":{s:latest.stats(apertures[f"{s}_third_finger_aperture"]) for s in ("left","right")},
        "ik_failed_frames":np.flatnonzero(failed).tolist(),"branch_discontinuity_count":int(branch.sum()),
        "joint_limit_violation_count":int(viol.sum()),"self_collision_frames":int(collision.sum()),
        "cross_arm_collision_frames":int(cross.sum()),"finger_torso_collision_frames":int(np.sum(clearance<0)),
        "collision_pairs":pairs.tolist(),"nan_inf_count":0 if finite else 1,
        "joint_step_rad":latest.stats(step),"joint_velocity_rad_s":latest.stats(vel),
        "joint_acceleration_rad_s2":latest.stats(acc),"approach_task_boundary_jump_rad":boundary,
    }
    payload=dict(optimized_action=raw.astype(np.float32),g1_target_left_position=targets["lp"],
        g1_target_right_position=targets["rp"],wrist_reference_left_orientation=refs["left"]["rotation"],
        wrist_reference_right_orientation=refs["right"]["rotation"],actual_left_wrist_orientation=kin["left_rotation"],
        actual_right_wrist_orientation=kin["right_rotation"],left_palm_normal=kin["left_normal"],
        right_palm_normal=kin["right_normal"],left_tool_pinch_forward=kin["left_tool"],
        right_tool_pinch_forward=kin["right_tool"],left_wrist_orientation_error=kin["left_orientation_error"],
        right_wrist_orientation_error=kin["right_orientation_error"],dex3_left_open=pose["left"]["open"],
        dex3_left_pregrasp=pose["left"]["pregrasp"],dex3_left_pinch=pose["left"]["pinch"],
        dex3_right_open=pose["right"]["open"],dex3_right_pregrasp=pose["right"]["pregrasp"],
        dex3_right_pinch=pose["right"]["pinch"],dex3_left_phase=phases["left"],dex3_right_phase=phases["right"],
        dex3_left_task_command=task_hand["left"],dex3_right_task_command=task_hand["right"],
        g1_arm_joint_trajectory=task_arm,full_arm=full_arm,full_left_dex3=full_l,full_right_dex3=full_r,
        full_g1_joint_trajectory=full_q,torso_clearance_m=clearance,self_collision_flag=collision,
        cross_arm_collision_flag=cross,ik_success=~failed,ik_branch_discontinuity=branch,
        thumb_index_aperture_left=apertures["left_thumb_index_aperture"],
        thumb_index_aperture_right=apertures["right_thumb_index_aperture"],
        third_finger_aperture_left=apertures["left_third_finger_aperture"],
        third_finger_aperture_right=apertures["right_third_finger_aperture"],
        task_start_frame=np.asarray(105),fps=np.asarray(latest.FPS),arm_joint_names=info["joint_names"])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report_path=a.output.with_name("g1_episode49_relative_neutral_pinch_report.json")
    if a.execute:
        tmp=a.output.with_suffix(".npz.incomplete")
        with tmp.open("wb") as f:np.savez_compressed(f,**payload)
        os.replace(tmp,a.output)
    else:report["dry_run"]=True
    tmpj=report_path.with_suffix(".json.incomplete");tmpj.write_text(json.dumps(report,indent=2));os.replace(tmpj,report_path)
    print(json.dumps(report,indent=2))
    return 0 if safety else 2


if __name__=="__main__":
    raise SystemExit(main())
