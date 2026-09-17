#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$HOME/aloha_g1_dataset}"
SRC="$ROOT/isaaclab_magsafe_fixed_scene"
DST="$ROOT/isaaclab_doll_handoff_scene"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab-3-beta/isaaclab.sh}"

if [[ ! -d "$SRC" ]]; then echo "ERROR: source scene missing: $SRC" >&2; exit 2; fi
if [[ -e "$DST" ]]; then
  echo "ERROR: destination already exists: $DST" >&2
  echo "Rename/delete it first so the old MagSafe scene is never overwritten." >&2
  exit 3
fi

echo "[1/5] Copying authoritative scene -> doll handoff scene"
cp -a "$SRC" "$DST"

echo "[2/5] Installing task config/assets/build tools"
cp "$PKG_DIR/scene_layout_doll_handoff.json" "$DST/"
cp "$PKG_DIR/build_doll_handoff_overlay.py" "$DST/"
cp "$PKG_DIR/patch_g1_pelvis_gap.py" "$DST/"
mkdir -p "$DST/assets/doll_handoff"
cp -a "$PKG_DIR/assets/." "$DST/assets/doll_handoff/"

# New friendly preview names. Existing scripts remain too.
cp "$DST/preview_magsafe_aloha_model.py" "$DST/preview_doll_handoff_aloha.py"
cp "$DST/preview_magsafe_g1_model.py" "$DST/preview_doll_handoff_g1.py"

echo "[3/5] Applying requested G1 pelvis/table gap = 0.15 m in the COPIED scene only"
set +e
python3 "$DST/patch_g1_pelvis_gap.py" --scene-dir "$DST" --gap 0.15
PATCH_RC=$?
set -e
if [[ $PATCH_RC -ne 0 ]]; then
  echo "WARNING: automatic G1 root patch was ambiguous. Scene/object modeling will still be built."
  echo "See: $DST/g1_pelvis_gap_patch_report.json"
fi

echo "[4/5] Building doll/bin USD overlay"
if [[ ! -x "$ISAACLAB" ]]; then echo "ERROR: IsaacLab launcher missing: $ISAACLAB" >&2; exit 4; fi
"$ISAACLAB" -p "$DST/build_doll_handoff_overlay.py" --scene-dir "$DST" --apply

echo "[5/5] Done"
cat <<EOF

NEW SCENE:
  $DST

ALOHA PREVIEW:
  cd $DST
  $ISAACLAB -p preview_doll_handoff_aloha.py --pose episode_frame0 --viz kit --camera overview

G1 PREVIEW:
  cd $DST
  $ISAACLAB -p preview_doll_handoff_g1.py --viz kit --camera overview

G1 GAP PATCH REPORT:
  $DST/g1_pelvis_gap_patch_report.json

BUILD REPORT:
  $DST/doll_handoff_build_report.json
EOF
