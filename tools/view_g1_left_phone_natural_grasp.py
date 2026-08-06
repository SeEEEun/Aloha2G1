#!/usr/bin/env python3
"""Fail-closed viewer for a validated natural-posture C-gap grasp."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path("/home/jbnu/aloha_g1_dataset")
DEFAULT=ROOT/"converted_runs/g1_left_phone_cgap_natural_posture/selected_left_phone_natural_grasp.npz"
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--grasp",type=Path,default=DEFAULT);a=p.parse_args()
 if not a.grasp.exists():
  print("G1_LEFT_PHONE_NATURAL_POSTURE_BLOCKED")
  print("No hard-valid natural posture exists; refusing to display a failed candidate as success.")
  return 2
 print("Validated NPZ exists. Use the generated viewer implementation after a successful search.")
 return 0
if __name__=="__main__":raise SystemExit(main())
