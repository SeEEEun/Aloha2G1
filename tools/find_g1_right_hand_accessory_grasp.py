#!/usr/bin/env python3
"""Fail-closed right-accessory search gate after left phone-grasp validation."""
from pathlib import Path
import json
ROOT=Path("/home/jbnu/aloha_g1_dataset")
OUT=ROOT/"converted_runs/smolvla_20k_episode49_role_aware_g1"
def main():
    report=json.loads((OUT/"role_aware_grasp_report.json").read_text())
    if report.get("verdict")!="LEFT_HAND_PHONE_GRASP_FEASIBLE":
        print("LEFT_HAND_PHONE_GRASP_NOT_FEASIBLE")
        print("Right accessory search not run: mandatory left-phone gate failed.")
        return 2
    raise RuntimeError("Left gate passed; right accessory solver requires a separate reviewed run.")
if __name__=="__main__": raise SystemExit(main())
