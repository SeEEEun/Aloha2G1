#!/usr/bin/env python3
"""Fail-closed viewer for a hard-valid mirrored-posture phone grasp."""
from __future__ import annotations
import argparse
from pathlib import Path
ROOT=Path("/home/jbnu/aloha_g1_dataset")
DEFAULT=ROOT/"converted_runs/g1_left_phone_grasp_mirrored_posture/selected_left_phone_mirrored_grasp.npz"
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--grasp",type=Path,default=DEFAULT);a=p.parse_args()
 if not a.grasp.exists():
  print("G1_LEFT_PHONE_MIRRORED_POSTURE_BLOCKED")
  print("No hard-valid mirrored grasp exists; refusing to render the failed diagnostic as success.")
  return 2
 print("Validated mirrored grasp found:",a.grasp)
 return 0
if __name__=="__main__":raise SystemExit(main())
