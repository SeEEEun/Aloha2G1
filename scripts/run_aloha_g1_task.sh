#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jbnu/aloha_g1_dataset
PYTHON_BIN=/home/jbnu/miniconda3/envs/isaaclab6/bin/python
CLI="$ROOT/tools/aloha_g1_task.py"
verb="${1:-}"
episode="${2:-}"
case "$verb" in
  prepare)
    test -n "${3:-}" || { echo "usage: $0 prepare EPISODE_ID ACTION_PATH"; exit 2; }
    exec "$PYTHON_BIN" "$CLI" prepare --episode-id "$episode" --aloha-action "$3"
    ;;
  aloha)
    exec "$PYTHON_BIN" "$CLI" replay --episode-id "$episode" --robot aloha --view gui
    ;;
  g1)
    exec "$PYTHON_BIN" "$CLI" replay --episode-id "$episode" --robot g1 --mode kinematic --view gui
    ;;
  compare)
    exec "$PYTHON_BIN" "$CLI" compare --episode-id "$episode"
    ;;
  physics)
    exec "$PYTHON_BIN" "$CLI" physics --episode-id "$episode"
    ;;
  closed-loop)
    exec "$PYTHON_BIN" "$CLI" closed-loop --episode-id "$episode"
    ;;
  open)
    exec xdg-open "$ROOT/outputs/aloha_g1_tasks/$episode/index.html"
    ;;
  list)
    exec "$PYTHON_BIN" "$CLI" list
    ;;
  status)
    exec "$PYTHON_BIN" "$CLI" status --episode-id "$episode"
    ;;
  *)
    echo "usage: $0 {prepare|aloha|g1|compare|physics|closed-loop|open|list|status} [EPISODE_ID] [ACTION_PATH]"
    exit 2
    ;;
esac
