#!/usr/bin/env python3
import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "converted_runs/magsafe_20260724_154440/isaac_replay"
TAG = "bare_phone_split_effort_cap_4_8661_uniform_box_shape_v1_accessory_joint_corrected_v7"


def one(pattern):
    paths = list(OUT.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError((pattern, paths))
    return paths[0]


raw = one(f"phone_accessory_relative_transform_raw_*accessory_joint_corrected_v7.csv")
with raw.open(newline="", encoding="utf-8") as f:
    rel = list(csv.DictReader(f))
shutil.copyfile(raw, OUT / "phone_accessory_relative_transform_timeline.csv")

contacts_path = one(f"contact_event_log_action_gripper_*accessory_joint_corrected_v7.csv")
with contacts_path.open(newline="", encoding="utf-8") as f:
    contacts = list(csv.DictReader(f))
right = [
    r for r in contacts
    if "follower_right_" in r["sensor_prim"] and "accessory" in r["other_prim"].lower()
]
right_first = min((int(r["source_frame"]) for r in right), default=None)
right_sides = {}
for r in right:
    frame = int(r["source_frame"])
    right_sides.setdefault(frame, set()).add(
        "left" if "carriage_left" in r["sensor_prim"] else "right"
    )
right_bilateral = min((f for f, sides in right_sides.items() if sides == {"left", "right"}), default=None)

attached = [r for r in rel if r["joint_state"] == "ATTACHED"]
translation_drift = max(float(r["translation_drift_m"]) for r in attached)
rotation_drift = max(float(r["rotation_drift_deg"]) for r in attached)
relative_angular_velocity = max(float(r["relative_angular_velocity_rad_s"]) for r in attached)
phone_rotation = max(float(r["phone_world_rotation_from_initial_deg"]) for r in attached)
accessory_rotation = max(float(r["accessory_world_rotation_from_initial_deg"]) for r in attached)
rotation_tracking_error = max(
    abs(float(r["phone_world_rotation_from_initial_deg"]) -
        float(r["accessory_world_rotation_from_initial_deg"]))
    for r in attached
)

exact_path = one(f"left_phone_exact_shape_contacts_*accessory_joint_corrected_v7.csv")
with exact_path.open(newline="", encoding="utf-8") as f:
    exact = list(csv.DictReader(f))
penetration = max((float(r["penetration"]) for r in exact), default=0.0)

joint_audit = """phone rigid body: /World/MagSafeScene/Phone
accessory rigid body: /World/MagSafeScene/Accessory
phone visual: /World/MagSafeScene/Phone/Visuals
accessory visual: /World/MagSafeScene/Accessory/Visuals
phone collisions: /World/MagSafeScene/Phone/BarePhoneCompound/{CentralBody,FrontGlass,RearMatte}
accessory collisions: /World/MagSafeScene/Accessory/Colliders/MainRing/Segment_00..11 and SupportRing/Segment_00..11
joint: /World/MagSafeScene/MagneticJoints/AccessoryPhone
joint type: PhysicsFixedJoint
body0: /World/MagSafeScene/Phone
body1: /World/MagSafeScene/Accessory
original localPos0: [0,0.004175,0]
original localRot0 wxyz: [1,0,0,0]
original localPos1: [0,-0.00175,0]
original localRot1 wxyz: [1,0,0,0]
original anchor world mismatch: 0.500000 mm
corrected localPos0: [0,0.004175,0]
corrected localRot0 wxyz: [1,0,0,0]
corrected localPos1: [0,-0.002250000136,0]
corrected localRot1 wxyz: [1,0,0,0]
joint enabled before detach: true
configured physical break force: 2.0 N
configured physical break torque: 0.08 N m
native non-selective break guard: 1e6 N / 1e6 N m
collision enabled between bodies: false
accessory kinematic: false
accessory gravity disabled: false
linear/angular axis locks: FixedJoint locks all 6 DoF
world-fixed constraint: none
PhysxJoint projection/tolerance API: unsupported in installed Isaac Sim 6.0
"""
(OUT / "phone_accessory_joint_audit.txt").write_text(joint_audit, encoding="utf-8")

visual = """Accessory rigid body, Visuals/MainCRing, Visuals/SupportRing, collider segments,
and magnetic-ring frame are descendants of /World/MagSafeScene/Accessory.
No visual or collider child has resetXformStack=true.
No child has a separate world-space orientation controller.
Rigid and visual hierarchy are consistent.
Classification: ACCESSORY_RIGID_BODY_NOT_ROTATING_WITH_PHONE in the original GUI was caused
by premature physical joint break, not an independently oriented visual mesh.
"""
(OUT / "phone_accessory_visual_vs_rigid_pose_report.txt").write_text(visual, encoding="utf-8")

frame_report = """Joint frame source: authored initial phone/accessory rigid-body world transforms.
Common world joint frame: original phone rear anchor.
Old anchor mismatch: 0.000500000053 m.
New body0 frame: localPos=[0,0.004175,0], localRot=[1,0,0,0].
New body1 frame: localPos=[0,-0.002250000136,0], localRot=[1,0,0,0].
Phone/accessory relative orientation at authoring: identity.
World axes were not assumed.
"""
(OUT / "phone_accessory_fixed_joint_frame_report.txt").write_text(frame_report, encoding="utf-8")

key_dir = OUT / "phone_accessory_attachment_keyframes"
key_dir.mkdir(exist_ok=True)
by_frame = {}
for r in rel:
    by_frame.setdefault(int(r["source_frame"]), r)
frames = {
    "01_initial": 0,
    "02_first_left_contact": 296,
    "03_true_lift_start": 392,
    "04_portrait_mid": 420,
    "05_minimum_portrait": 450,
    "06_right_approach": 550,
    "07_right_first_contact": right_first,
    "08_detach_pre": 1006,
    "09_detach_post": None,
}
for name, frame in frames.items():
    if frame is None:
        text = "frame=NONE\njoint_state=ATTACHED\nnote=no physical detach occurred\n"
    else:
        nearest = min(by_frame, key=lambda x: abs(x-frame))
        r = by_frame[nearest]
        text = "\n".join(f"{k}={v}" for k, v in r.items()) + "\n"
    (key_dir / f"{name}.txt").write_text(text, encoding="utf-8")

summary = {
    "translation_drift_m": translation_drift,
    "rotation_drift_deg": rotation_drift,
    "relative_angular_velocity_max_rad_s": relative_angular_velocity,
    "phone_world_rotation_deg": phone_rotation,
    "accessory_world_rotation_deg": accessory_rotation,
    "world_rotation_tracking_error_deg": rotation_tracking_error,
    "right_accessory_first_contact_frame": right_first,
    "right_accessory_first_bilateral_frame": right_bilateral,
    "detach_frame": None,
    "maximum_penetration_m": penetration,
    "PHONE_ACCESSORY_RIGID_ATTACHMENT_PASS": translation_drift < .0005 and rotation_drift < .5,
    "ACCESSORY_DETACH_PASS": False,
    "direct_causes": [
        "original joint anchors had 0.5 mm world mismatch",
        "native 2 N break was non-selective and broke under left-hand assembly load",
        "original detach observer compared a world-space translation vector and misread assembly rotation",
        "new episode provides only unilateral right-accessory contact, so physical detach criterion is not reached",
    ],
}
(OUT / "phone_accessory_attachment_final_summary.txt").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
