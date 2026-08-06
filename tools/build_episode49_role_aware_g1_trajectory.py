#!/usr/bin/env python3
"""Fail-closed role-aware trajectory gate; never substitutes a bimanual phone grasp."""
from pathlib import Path
import json
ROOT=Path("/home/jbnu/aloha_g1_dataset")
OUT=ROOT/"converted_runs/smolvla_20k_episode49_role_aware_g1"
def main():
    report=json.loads((OUT/"role_aware_grasp_report.json").read_text())
    if report.get("verdict")!="ROLE_AWARE_STATIC_POSE_READY":
        blocked={
            "verdict":"LEFT_HAND_PHONE_GRASP_NOT_FEASIBLE",
            "trajectory_generated":False,
            "stop_gate":"mandatory_single_left_hand_phone_grasp",
            "left_role":"phone_grasp_move_place",
            "right_role":"accessory_grasp_remove",
            "bimanual_side_support_used":False,
            "right_accessory_search_run":False,
            "combined_static_pose_generated":False,
            "mujoco_videos_generated":False,
            "workspace_calibration":False,
            "isaac_lab_executed":False,
            "hardware_executed":False,
        }
        (OUT/"g1_episode49_role_aware_report.json").write_text(
            json.dumps(blocked,indent=2)+"\n")
        print(report.get("verdict","LEFT_HAND_PHONE_GRASP_NOT_FEASIBLE"))
        print("No temporal trajectory generated: static role gates are not complete.")
        return 2
    raise RuntimeError("Static gates passed; temporal solver requires a separate reviewed run.")
if __name__=="__main__": raise SystemExit(main())
