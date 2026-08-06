#!/usr/bin/env python3
"""Offline FK/workspace audit for fixed-root MagSafe G1 candidates (no IK)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
ARM = ROOT / "converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz"
FULL = ROOT / "outputs/g1_magsafe_arm_dex3_full_trajectory.npz"
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
POSE = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
OUT = ROOT / "outputs/g1_magsafe_workspace_calibration/candidate_analysis.json"
OFFSETS = (0.0, 0.05, 0.10, 0.15, 0.20)
PALM = {"left": np.array([.0415, .003, 0.]), "right": np.array([.0415, -.003, 0.])}


def body_id(model, name):
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise RuntimeError(f"missing G1 body {name}")
    return value


def point_box_distance(p, lo, hi):
    return float(np.linalg.norm(np.maximum(np.maximum(lo-p, 0), p-hi)))


def main() -> None:
    layout = json.loads(LAYOUT.read_text())
    pose = json.loads(POSE.read_text())["g1"]
    root = np.asarray(pose["position_xyz_m"], float)
    phone = np.array([.5*(layout["phone"]["bottom_left_xy"][0]+layout["phone"]["bottom_right_xy"][0]),
                      .5*(layout["phone"]["bottom_left_xy"][1]+layout["phone"]["bottom_right_xy"][1]),
                      layout["table"]["surface_height"]+.5*layout["phone"]["size_landscape_xyz"][2]])
    accessory = phone.copy(); accessory[1] += (.5*layout["phone"]["size_landscape_xyz"][1]
        + layout["accessory"]["phone_back_clearance"] + .5*layout["accessory"]["main_depth"])
    charger = np.array([*layout["charger"]["center_xy"], layout["table"]["surface_height"]+layout["charger"]["mount_plate"]["size_xyz"][2]])
    task = .5*(phone+accessory)
    forward = task-root; forward[2] = 0; forward /= np.linalg.norm(forward)
    lateral = np.array([-forward[1], forward[0], 0.])

    with np.load(ARM, allow_pickle=False) as z:
        q = z["g1_arm_joint_trajectory"].astype(float); names = z["arm_joint_names"].astype(str).tolist()
    with np.load(FULL, allow_pickle=False) as z:
        full = z["full_qpos"].astype(float)
    model = mujoco.MjModel.from_xml_path(str(MODEL)); data = mujoco.MjData(model)
    qids = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]
    limits = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    ids = {s: body_id(model, f"{s}_wrist_yaw_link") for s in ("left", "right")}
    elbow_ids = [names.index("left_elbow_joint"), names.index("right_elbow_joint")]
    shoulder_ids = [i for i,n in enumerate(names) if "shoulder" in n]
    wrist_ids = [i for i,n in enumerate(names) if "wrist" in n]
    margins = np.minimum(q-limits[:,0], limits[:,1]-q)

    local_wrist = {s: np.empty((len(q),3)) for s in ids}
    body_geoms = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid]); name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid) or "world"
        category = "hand" if "hand" in name else "arm" if any(x in name for x in ("shoulder","elbow","wrist")) else "body"
        body_geoms.append((gid, category, name))
    local_geom = np.empty((len(q), model.ngeom, 3))
    for f in range(len(q)):
        data.qpos[:] = full[f]; data.qpos[qids] = q[f]; mujoco.mj_forward(model,data)
        for side,bid in ids.items():
            local_wrist[side][f] = data.xpos[bid] + data.xmat[bid].reshape(3,3) @ PALM[side]
        local_geom[f] = data.geom_xpos

    # +90 degree authored yaw: local +X -> world +Y.
    rot = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    z_shift = root[2] - .79
    table_lo = np.array([0., 0., layout["table"]["surface_height"]-layout["table"]["top_thickness"]])
    table_hi = np.array([layout["table"]["size_x"],layout["table"]["size_y"],layout["table"]["surface_height"]])
    events = {e["event"]:int(e["frame"]) for e in json.loads(TIMELINE.read_text())["events"]}
    keys = [("initial",0),("left_phone_approach",max(0,events["left_phone_grasp_start"]-30)),
      ("left_phone_grasp",events["left_phone_grasp_start"]),("phone_rotation",events["phone_rotation_to_portrait_start"]),
      ("right_accessory_approach",max(0,events["right_accessory_grasp_start"]-30)),
      ("right_accessory_grasp",events["right_accessory_grasp_start"]),("accessory_removal",events["accessory_removed"]),
      ("phone_charger_approach",events["phone_move_to_charger_start"]),("phone_placement",events["phone_charger_attachment_complete"]),
      ("left_release",events["left_phone_release_complete"]),("right_accessory_placement",max(0,events["right_accessory_release_complete"]-30)),
      ("right_release",events["right_accessory_release_complete"])]
    rows=[]
    for off in OFFSETS:
        rp=root+off*forward
        wrists={s: local_wrist[s]@rot.T + np.array([rp[0],rp[1],z_shift]) for s in ids}
        geom_world=local_geom@rot.T + np.array([rp[0],rp[1],z_shift])
        collisions={"body":[],"arm":[],"hand":[]}
        torso_min=1e9
        for f in range(len(q)):
            for gid,cat,name in body_geoms:
                radius=float(model.geom_rbound[gid]); dist=point_box_distance(geom_world[f,gid],table_lo,table_hi)-radius
                if "torso" in name: torso_min=min(torso_min,dist)
                if dist < 0 and f not in collisions[cat]: collisions[cat].append(f)
        # A frame is workspace-reachable when either palm is within 20 cm of a task object.
        all_targets=np.stack((phone,accessory,charger))
        near=np.minimum(np.linalg.norm(wrists["left"][:,None]-all_targets,axis=2).min(1),
                        np.linalg.norm(wrists["right"][:,None]-all_targets,axis=2).min(1))
        krows=[]
        for label,f in keys:
            target = accessory if "accessory" in label or label=="right_release" else charger if label in ("phone_charger_approach","phone_placement") else phone
            side="right" if "right" in label or "accessory" in label else "left"
            krows.append({"label":label,"frame":f,"left_wrist_world":wrists["left"][f].tolist(),
              "right_wrist_world":wrists["right"][f].tolist(),"target":target.tolist(),
              "assigned_hand":side,"assigned_hand_target_distance_m":float(np.linalg.norm(wrists[side][f]-target)),
              "left_elbow_straight_margin_rad":float(abs(q[f,elbow_ids[0]])),
              "right_elbow_straight_margin_rad":float(abs(q[f,elbow_ids[1]])),
              "collision":any(f in collisions[x] for x in collisions)})
        rows.append({"offset_m":off,"root_position":rp.tolist(),"table_front_horizontal_distance_m":max(0.,-rp[1]),
          "root_phone_distance_m":float(np.linalg.norm(rp-phone)),"root_accessory_distance_m":float(np.linalg.norm(rp-accessory)),
          "reachable_frame_ratio_20cm":float(np.mean(near<=.20)),"minimum_elbow_straight_margin_rad":float(np.min(np.abs(q[:,elbow_ids]))),
          "minimum_shoulder_limit_margin_rad":float(margins[:,shoulder_ids].min()),"minimum_wrist_limit_margin_rad":float(margins[:,wrist_ids].min()),
          "joint_limit_violation_count":int((margins<0).sum()),"body_table_collision_frames":collisions["body"],
          "arm_table_collision_frames":collisions["arm"],"hand_table_collision_frames":collisions["hand"],
          "torso_table_minimum_clearance_m_conservative":torso_min,"near_full_extension_frames_lt_0p05rad":np.flatnonzero(np.min(np.abs(q[:,elbow_ids]),axis=1)<.05).tolist(),
          "key_frames":krows})
    report={"method":"unchanged trajectory; MuJoCo FK transformed by authoritative root yaw/position; conservative geom bounding-sphere vs table-top AABB collision screen",
      "thresholds":{"reachable_object_distance_m":.20,"near_full_extension_elbow_abs_rad":.05},
      "current_root":root.tolist(),"phone_center_world":phone.tolist(),"accessory_center_world":accessory.tolist(),"charger_center_world":charger.tolist(),
      "task_center_world":task.tolist(),"table_forward_direction":forward.tolist(),"table_lateral_direction":lateral.tolist(),
      "root_to_task_center_horizontal_distance_m":float(np.linalg.norm((task-root)[:2])),"candidates":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
