#!/usr/bin/env bash
set -euo pipefail

scene_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaac_python="/home/jbnu/miniconda3/envs/isaaclab6/bin/python"

if [[ ! -x "$isaac_python" ]]; then
  echo "FAIL: required Isaac Python is not executable: $isaac_python" >&2
  exit 2
fi

"$isaac_python" -c 'import sys, tomllib; from pxr import Usd; print("ISAAC_PYTHON=" + sys.executable); print("TOMLLIB=PASS PXR=PASS")'
"$isaac_python" "$scene_dir/build_doll_handoff_scene.py"
"$isaac_python" "$scene_dir/validate_scene.py"

echo "PASS: DOLL_HANDOFF_INSTALL_COMPLETE"
