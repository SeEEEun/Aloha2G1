#!/usr/bin/env python3
"""Fail-closed viewer for a validated horizontal edge pinch."""
from pathlib import Path
import argparse
ROOT=Path("/home/jbnu/aloha_g1_dataset")
DEFAULT=ROOT/"converted_runs/g1_left_phone_horizontal_edge_pinch/selected_horizontal_edge_pinch.npz"
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--grasp",type=Path,default=DEFAULT);a=p.parse_args()
 if not a.grasp.exists():
  print("G1_LEFT_HORIZONTAL_EDGE_PINCH_BLOCKED")
  print("No hard-valid horizontal edge pinch exists; physics and trajectory gates remain closed.")
  return 2
 print("Validated horizontal edge pinch:",a.grasp);return 0
if __name__=="__main__":raise SystemExit(main())
