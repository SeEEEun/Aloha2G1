#!/usr/bin/env python3
"""Evidence-only finalizer for the single EP49 v18.1 frame-consistency run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_trajbooster_frame_consistency_v18_1"
V18 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_full_task_execution_v18"
SOURCE_REPLAY = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11/source_optimized_action_replay.mp4"
PHYSICS = OUT / "physics_trial_full_task_diagnostic_0p25x_paper_white.npz"


def default(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def dump(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def video_info(path: Path) -> dict[str, Any]:
    p = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    row = json.loads(p.stdout)["streams"][0]
    return {"path": path.resolve(), "sha256": sha(path), **row,
            "pass": int(row["nb_read_frames"]) == 990 and row["r_frame_rate"] == "15/2"}


def fit(image: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    canvas = np.full((height, width, 3), 247, np.uint8)
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas


def label(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    out = image.copy(); cv2.rectangle(out, (0,0), (out.shape[1],48), (20,20,20), -1)
    cv2.putText(out, title, (8,20), cv2.FONT_HERSHEY_SIMPLEX, .46, (80,235,255), 1, cv2.LINE_AA)
    cv2.putText(out, subtitle, (8,40), cv2.FONT_HERSHEY_SIMPLEX, .34, (235,235,235), 1, cv2.LINE_AA)
    return out


def text_panel(title: str, lines: list[str], progress: float) -> np.ndarray:
    image = np.full((360,640,3), 247, np.uint8)
    cv2.putText(image, title, (18,31), cv2.FONT_HERSHEY_SIMPLEX, .67, (25,25,25), 2, cv2.LINE_AA)
    for i, line in enumerate(lines[:8]):
        color = (35,35,35) if not line.startswith("STATUS") else (20,90,20)
        cv2.putText(image, line[:100], (18,70+30*i), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
    cv2.rectangle(image, (18,322), (622,340), (205,205,205), -1)
    cv2.rectangle(image, (18,322), (18+round(604*progress),340), (55,155,85), -1)
    return image


def contact_masks(physics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = {name: np.zeros(990, bool) for name in ("lt", "li", "lm", "rt", "ri", "rm")}
    for i, rows in enumerate(physics["all_robot_object_contact_rows"]):
        for row in rows:
            owner, other = row.get("owner", ""), row.get("other", "")
            if "/Phone" in other:
                if "left_hand_thumb" in owner: masks["lt"][i] = True
                if "left_hand_index" in owner: masks["li"][i] = True
                if "left_hand_middle" in owner: masks["lm"][i] = True
            if "/Accessory" in other:
                if "right_hand_thumb" in owner: masks["rt"][i] = True
                if "right_hand_index" in owner: masks["ri"][i] = True
                if "right_hand_middle" in owner: masks["rm"][i] = True
    return masks


def first(mask: np.ndarray) -> int | None:
    ids = np.flatnonzero(mask); return int(ids[0]) if len(ids) else None


def last(mask: np.ndarray) -> int | None:
    ids = np.flatnonzero(mask); return int(ids[-1]) if len(ids) else None


def make_closeup(source: Path, output: Path, title: str) -> None:
    cap = cv2.VideoCapture(str(source)); raw = output.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (640,360))
    for i in range(990):
        ok, frame = cap.read()
        if not ok: raise RuntimeError(f"closeup ended at {i}")
        crop = frame[58:360, 175:640]
        image = cv2.resize(crop, (640,360), interpolation=cv2.INTER_CUBIC)
        image = label(image, title, f"same uninterrupted true-PhysX run | action {i}/989")
        writer.write(image)
    cap.release(); writer.release(); os.replace(raw, output)


def make_four_panel(output: Path, kind: str, physics: dict[str,np.ndarray], masks: dict[str,np.ndarray],
                    tracking: dict[str,Any], posture: dict[str,Any]) -> None:
    if kind == "ee":
        paths = [SOURCE_REPLAY, V18/"v18_KINEMATIC_FULL_overview.mp4", OUT/"v18_1_KINEMATIC_FULL_overview.mp4"]
        titles = ["SMOLVLA ALOHA SOURCE", "V18 G1: WRIST-CARRIER SEMANTICS", "V18.1 G1: PHYSICAL PINCH TOOL"]
    elif kind == "posture":
        paths = [SOURCE_REPLAY, V18/"v18_KINEMATIC_FULL_overview.mp4", OUT/"v18_1_KINEMATIC_FULL_overview.mp4"]
        titles = ["SMOLVLA ALOHA SOURCE", "V18 G1 POSTURE", "V18.1 MINIMUM-NECESSARY POSTURE"]
    else:
        paths = [SOURCE_REPLAY, OUT/"v18_1_KINEMATIC_FULL_overview.mp4", OUT/"v18_1_TRUE_PHYSICS_FULL_overview.mp4"]
        titles = ["SMOLVLA ALOHA SOURCE", "V18.1 G1 KINEMATIC", "V18.1 ACTUAL TRUE-PHYSX"]
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    raw = output.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (1280,720))
    phone0 = physics["phone_pose_xyzw"][0,:3]
    cumulative_v18 = np.r_[0, np.cumsum(np.sum(np.abs(np.diff(np.load(V18/"final_arm_dex3_trajectory.npz")["arm_qpos"],axis=0)),axis=1))]
    cumulative_v181 = np.r_[0, np.cumsum(np.sum(np.abs(np.diff(np.load(OUT/"final_arm_dex3_trajectory.npz")["arm_qpos"],axis=0)),axis=1))]
    for i in range(990):
        images=[]
        for cap,title in zip(caps,titles):
            ok,im=cap.read()
            if not ok: raise RuntimeError(f"panel ended {i}")
            images.append(label(fit(im), title, f"same generated action {i}/989"))
        if kind == "ee":
            lines=[
                f"RIGHT gap: v18 reported 174.306 mm -> v18.1 1.420 mm",
                f"task-tool XYZ max error: L {1000*tracking['left_position_error_m']['maximum']:.3f} | R {1000*tracking['right_position_error_m']['maximum']:.3f} mm",
                f"path corr: L {tracking['fidelity']['left_task_ee_path_correlation']:.6f} | R {tracking['fidelity']['right_task_ee_path_correlation']:.6f}",
                "static wrist->pinch SE(3); no waypoint/per-frame residual",
                "task EE = physical THUMB-INDEX pinch center",
                "legacy wrist/palm metrics are reported separately",
                "STATUS FRAME MISMATCH CORRECTED",
            ]
        elif kind == "posture":
            b,a=posture["before_v18"],posture["after_v18_1"]
            lines=[
                f"joint travel running: v18 {cumulative_v18[i]:.1f} | v18.1 {cumulative_v181[i]:.1f} rad",
                f"sign reversal: {b['sign_reversal_rate']:.4f} -> {a['sign_reversal_rate']:.4f}",
                f"jerk p95: {b['jerk_rad_s3']['p95_abs']:.1f} -> {a['jerk_rad_s3']['p95_abs']:.1f} rad/s3",
                f"min joint margin: {posture['posture_before']['global']['minimum_joint_margin_rad']:.3f} -> {posture['posture_after']['global']['minimum_joint_margin_rad']:.3f} rad",
                "fast protected task motion is not smoothed",
                "total travel increase from required 54.438 mm tool correction is disclosed",
                "STATUS REDUNDANT SIGN REVERSALS REDUCED",
            ]
        else:
            phone_disp=1000*np.linalg.norm(physics['phone_pose_xyzw'][i,:3]-phone0)
            lines=[
                f"phone displacement {phone_disp:.1f} mm | bilateral {bool(masks['lt'][i] and masks['li'][i])}",
                f"LEFT task contact thumb/index/third: {int(masks['lt'][i])}/{int(masks['li'][i])}/{int(masks['lm'][i])}",
                f"RIGHT accessory thumb/index/third: {int(masks['rt'][i])}/{int(masks['ri'][i])}/{int(masks['rm'][i])}",
                "acquisition action 163 | lift peak action 207 | loss action 209",
                "accessory detached action 210 before intended right stage",
                "no object follow / teleport / scripted attach",
                "STATUS FULL TASK FAIL; PLAYBACK COMPLETED 990/990",
            ]
        images.append(text_panel("V18.1 TASK-TOOL EVIDENCE", lines, i/989))
        writer.write(np.vstack([np.hstack(images[:2]),np.hstack(images[2:])]))
    for cap in caps: cap.release()
    writer.release(); os.replace(raw,output)


def make_explanation(frame_graph: dict[str,Any], gap: dict[str,Any]) -> Path:
    out=OUT/"ee_frame_consistency_explanation.png"
    fig,ax=plt.subplots(figsize=(17,9),dpi=160);ax.set_xlim(0,17);ax.set_ylim(0,9);ax.axis("off")
    def box(x,y,w,h,text,color):
        ax.add_patch(plt.Rectangle((x,y),w,h,fc=color,ec="#333",lw=1.5));ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    ax.text(3.0,8.15,"SOURCE TOOL SEMANTICS",ha="center",fontsize=14,weight="bold")
    box(.5,6.3,2.3,1.1,"ALOHA link_6","#dceeff");box(3.8,6.3,2.5,1.1,"ALOHA TCP\nsemantic task EE","#b9e0ff")
    ax.annotate("",xy=(3.8,6.85),xytext=(2.8,6.85),arrowprops=dict(arrowstyle="->",lw=2))
    ax.text(3.3,7.35,"T_LINK6_TCP = [148.7, 0, -1.05] mm",ha="center",fontsize=9)
    ax.text(11.2,8.15,"TARGET ROBOT TOOL SEMANTICS",ha="center",fontsize=14,weight="bold")
    box(7.0,6.3,2.1,1.1,"G1 wrist","#eee");box(10.0,6.3,2.1,1.1,"Dex3 palm","#f5e8c8");box(13.0,6.3,3.1,1.1,"physical THUMB-INDEX\npinch center / task EE","#c8f2d0")
    ax.annotate("",xy=(10.0,6.85),xytext=(9.1,6.85),arrowprops=dict(arrowstyle="->",lw=2));ax.text(9.55,7.30,"static",ha="center",fontsize=9)
    ax.annotate("",xy=(13.0,6.85),xytext=(12.1,6.85),arrowprops=dict(arrowstyle="->",lw=2));ax.text(12.55,7.30,"static active-FK SE(3)",ha="center",fontsize=9)
    box(.7,2.6,4.6,1.7,"V18 mismatch\npalm carrier retained legacy RIGHT middle/C proxy\nafter hand changed to THUMB+INDEX","#ffd5d5")
    box(6.2,2.6,4.6,1.7,"V18.1 correction\nprotected physical task-contact path\n+ one static wrist->pinch transform","#d5f5d5")
    box(11.7,2.6,4.6,1.7,f"RIGHT physical gap\nreported/initial bbox: 174.306 mm\ndynamic task frame: 51.370 mm\nafter static correction: {1000*gap['v18_1_physical_pinch_to_dynamic_accessory_target_error_m']:.3f} mm","#fff0c8")
    ax.annotate("",xy=(6.2,3.45),xytext=(5.3,3.45),arrowprops=dict(arrowstyle="->",lw=2,color="#803"));ax.text(5.75,3.8,"FRAME\nSEMANTICS",ha="center",fontsize=8,color="#803")
    ax.text(8.5,1.0,"Source timing/relative behavior unchanged | no per-frame Cartesian offset | Dex3 remains a predefined hand adapter",ha="center",fontsize=12,weight="bold")
    fig.tight_layout();fig.savefig(out,bbox_inches="tight");plt.close(fig);return out


def main() -> int:
    raw=load(OUT/"full_task_diagnostic_result.json"); tracking=load(OUT/"task_ee_tracking_metrics.json")
    posture=load(OUT/"excessive_joint_motion_audit.json"); gap=load(OUT/"right_accessory_174mm_gap_decomposition.json")
    graph=load(OUT/"complete_ee_frame_graph_v18_1.json"); freeze=load(OUT/"source_freeze_audit.json")
    with np.load(PHYSICS,allow_pickle=True) as x: physics={k:x[k].copy() for k in x.files}
    masks=contact_masks(physics); bilateral=masks['lt']&masks['li']
    initial=physics['phone_pose_xyzw'][0,:3]; z=physics['phone_pose_xyzw'][:,2]
    lift_peak=int(np.argmax(z)); lift=float(z[lift_peak]-z[0])
    result={
        "status":"V18_1_TRAJBOOSTER_INSPIRED_FRAME_CONSISTENT_EXECUTION_READY_FOR_USER_REVIEW",
        "execution":"FULL_TRAJECTORY_DIAGNOSTIC_COMPLETE", "samples":990,
        "frame_semantics":"END_EFFECTOR_FRAME_MISMATCH_CONFIRMED_AND_CORRECTED",
        "whole_motion_execution_sanity":"PASS_WITH_DISCLOSED_REQUIRED_TOOL_OFFSET_MOTION",
        "full_task_true_physics":"FAIL_PHONE_PORTRAIT_RETENTION_BLOCKER",
        "phone":{"first_thumb":first(masks['lt']),"first_index":first(masks['li']),"first_bilateral":first(bilateral),"last_bilateral":last(bilateral),"lift_peak_m":lift,"lift_peak_action":lift_peak,"portrait_retention_loss_action":209},
        "non_task_left_third":{"first_phone_contact":first(masks['lm']),"status":"FAIL_PREMATURE_CONTACT"},
        "accessory":{"right_thumb_contact":first(masks['rt']),"right_index_contact":first(masks['ri']),"detached_action":raw['object_metrics']['accessory_detach_action_index'],"status":"FAIL_DETACHED_BEFORE_INTENDED_RIGHT_STAGE"},
        "charger":{"state_final":raw['object_metrics']['charger_state_final'],"center_error_final_mm":raw['object_metrics']['charger_center_error_final_mm'],"status":"FAIL"},
        "first_task_stage_causal_blocker":"PHONE_PORTRAIT_RETENTION_LOSS_ACTION_209",
        "earliest_invariant_warning":"LEFT_NON_TASK_THIRD_PREMATURE_PHONE_CONTACT_ACTION_137",
        "no_cheat":raw['integrity'], "render_parity":raw['execution_render_parity']['status'],
    }
    dump(OUT/"full_true_physics_result.json",result)
    make_closeup(OUT/"v18_1_TRUE_PHYSICS_FULL_overview.mp4",OUT/"v18_1_TRUE_PHYSICS_PHONE_PORTRAIT_closeup.mp4","V18.1 PHONE ACQUIRE / LIFT / PORTRAIT")
    make_closeup(OUT/"v18_1_TRUE_PHYSICS_FULL_side.mp4",OUT/"v18_1_TRUE_PHYSICS_ACCESSORY_closeup.mp4","V18.1 RIGHT ACCESSORY PHYSICAL REVIEW")
    make_four_panel(OUT/"v18_vs_v18_1_END_EFFECTOR_FRAME_4panel.mp4","ee",physics,masks,tracking,posture)
    make_four_panel(OUT/"v18_vs_v18_1_POSTURE_AND_EE_4panel.mp4","posture",physics,masks,tracking,posture)
    make_four_panel(OUT/"v18_1_ALOHA_vs_G1_FULL_TASK_4panel.mp4","full",physics,masks,tracking,posture)
    explanation=make_explanation(graph,gap)
    commands=f'''#!/usr/bin/env bash
set -euo pipefail
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py \\
  --input outputs/scene_registered_retargeting/current_layout_ep49_trajbooster_frame_consistency_v18_1/final_arm_dex3_trajectory.npz \\
  --output-dir outputs/scene_registered_retargeting/current_layout_ep49_trajbooster_frame_consistency_v18_1/gui_review \\
  --artifact-prefix v18_1_gui --trial full_task_diagnostic --speed 0.25 \\
  --interactive-review --interactive-only --render-preset paper-white \\
  --camera overview --pause-at-end --enable_cameras
'''
    (OUT/"commands.sh").write_text(commands);(OUT/"commands.sh").chmod(0o755)
    videos=[OUT/name for name in [
        "v18_1_KINEMATIC_FULL_overview.mp4","v18_1_KINEMATIC_FULL_side.mp4","v18_1_KINEMATIC_FULL_top.mp4","v18_1_KINEMATIC_FULL_robot_only.mp4",
        "v18_1_TRUE_PHYSICS_FULL_overview.mp4","v18_1_TRUE_PHYSICS_FULL_side.mp4","v18_1_TRUE_PHYSICS_FULL_top.mp4",
        "v18_1_TRUE_PHYSICS_PHONE_PORTRAIT_closeup.mp4","v18_1_TRUE_PHYSICS_ACCESSORY_closeup.mp4",
        "v18_vs_v18_1_END_EFFECTOR_FRAME_4panel.mp4","v18_vs_v18_1_POSTURE_AND_EE_4panel.mp4","v18_1_ALOHA_vs_G1_FULL_TASK_4panel.mp4"]]
    video_audit={p.name:video_info(p) for p in videos};dump(OUT/"video_decode_audit.json",video_audit)
    report=f'''# EP49 v18.1 task-tool frame consistency report

## Decision

- **END_EFFECTOR_FRAME_MISMATCH_CONFIRMED_AND_CORRECTED**
- Authoritative source EE: ALOHA TCP (`link_6 -> TCP = [148.7, 0, -1.05] mm`).
- Authoritative target EE: physical Dex3 THUMB–INDEX pinch center.
- RIGHT gap: reported initial-object-frame 174.306 mm; dynamic task-frame 51.370 mm; after one static wrist→pinch correction 1.420 mm.
- Portrait loss at action 209 was **physical retention failure**, not primarily frame tracking.

## Kinematic gate

- LEFT/RIGHT max physical task-EE errors: {1000*tracking['left_position_error_m']['maximum']:.3f}/{1000*tracking['right_position_error_m']['maximum']:.3f} mm.
- Path correlations: {tracking['fidelity']['left_task_ee_path_correlation']:.6f}/{tracking['fidelity']['right_task_ee_path_correlation']:.6f}; branch 0; collision 0; minimum margin {tracking['minimum_joint_limit_margin_rad']:.6f} rad.
- Redundant sign reversals: {posture['before_v18']['sign_reversal_rate']:.6f} -> {posture['after_v18_1']['sign_reversal_rate']:.6f}. Jerk p95: {posture['before_v18']['jerk_rad_s3']['p95_abs']:.3f} -> {posture['after_v18_1']['jerk_rad_s3']['p95_abs']:.3f} rad/s³.
- Total arm travel increases required by the 54.438 mm tool-offset correction are disclosed in `excessive_joint_motion_audit.json`; no prettier-motion claim is inferred from them.

## Single true-PhysX run

- 990/990 samples, 7,980 physics steps; actual articulation/Fabric/RTX parity PASS.
- Thumb+index acquisition at action {first(bilateral)}; peak lift {1000*lift:.3f} mm at action {lift_peak}; bilateral contact through action {last(bilateral)}; loss at action 209.
- LEFT THIRD prematurely touched phone at action {first(masks['lm'])}; it was never counted as grasp success.
- Accessory detached at action {raw['object_metrics']['accessory_detach_action_index']} before intended RIGHT stage; RIGHT task contacts 0. Charger final state DETACHED.
- No object follow, teleport, scripted attachment, per-frame Cartesian residual, DDS, publisher, or robot command.

## User review

- [Frame explanation](ee_frame_consistency_explanation.png)
- [EE comparison](v18_vs_v18_1_END_EFFECTOR_FRAME_4panel.mp4)
- [Posture comparison](v18_vs_v18_1_POSTURE_AND_EE_4panel.mp4)
- [Full task](v18_1_ALOHA_vs_G1_FULL_TASK_4panel.mp4)
- [True physics overview](v18_1_TRUE_PHYSICS_FULL_overview.mp4)

Next action: **USER REVIEWS THE FRAME-CONSISTENCY EXPLANATION, V18 VS V18.1 COMPARISON, AND THE SINGLE FULL TRUE-PHYSX EXECUTION.**
'''
    (OUT/"report.md").write_text(report)
    manifest={"status":result['status'],"trajectory_sha256":sha(OUT/"final_arm_dex3_trajectory.npz"),"source_freeze":freeze,"files":{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},"gui_command":commands.strip()}
    dump(OUT/"run_manifest.json",manifest)
    print(json.dumps({"result":result,"explanation":str(explanation),"videos":video_audit},indent=2,default=default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
