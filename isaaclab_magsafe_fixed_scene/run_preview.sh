#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"

if [[ ! -x "$ISAACLAB_ROOT/isaaclab.sh" ]]; then
  echo "Isaac Lab launcher not found: $ISAACLAB_ROOT/isaaclab.sh" >&2
  echo "Set ISAACLAB_ROOT, for example:" >&2
  echo "  ISAACLAB_ROOT=/path/to/IsaacLab $0 --rebuild" >&2
  exit 2
fi

exec "$ISAACLAB_ROOT/isaaclab.sh" -p "$ROOT_DIR/preview_magsafe_scene.py" "$@"
