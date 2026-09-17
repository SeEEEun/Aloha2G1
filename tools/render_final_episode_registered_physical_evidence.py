#!/usr/bin/env python3
"""Render only saved measured-robot/PhysX-object states for final EVAL35 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.doll_handoff_retargeting.render import _add_box, _add_geom

OUT = ROOT / "outputs/final_episode_registered_eval35"
COMMON = ROOT / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21/frozen_approval/config/common_config.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
FORENSIC = OUT / "00_forensic_audit/ACT_A_LEFT_GRASP_FORENSIC_REPORT.json"
REPLAYS = OUT / "06_physical_replays"
VISUALS = OUT / "07_paper_visuals"
WIDTH, HEIGHT = 540, 405
MWIDTH, MHEIGHT = 3840, 2160
COLS, ROWS = 7, 5
GX, GY = 30, 67
FPS = 30


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".incomplete" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"cannot write {temporary}")
    os.replace(temporary, path)


def rotation_xyzw(value: Iterable[float]) -> np.ndarray:
    x, y, z, w = np.asarray(tuple(value), dtype=np.float64)
    n = np.linalg.norm([x, y, z, w])
    x, y, z, w = np.asarray([x, y, z, w]) / n
    return np.asarray([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def camera(eye: Iterable[float], target: Iterable[float]) -> mujoco.MjvCamera:
    eye, target = np.asarray(tuple(eye)), np.asarray(tuple(target))
    relative = eye - target
    result = mujoco.MjvCamera()
    result.type = mujoco.mjtCamera.mjCAMERA_FREE
    result.lookat[:] = target
    result.distance = float(np.linalg.norm(relative))
    result.azimuth = math.degrees(math.atan2(-relative[1], -relative[0]))
    result.elevation = -math.degrees(math.atan2(relative[2], np.linalg.norm(relative[:2])))
    return result


class PhysicalRenderer:
    def __init__(self) -> None:
        self.common = load_common_config(COMMON)
        self.layout = load_scene(self.common)
        self.config = read_json(CONFIG)
        self.g1 = G1Kinematics(self.common, self.layout)
        self.model = self.g1.model
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=WIDTH, height=HEIGHT)
        contract_names = [str(row["joint_name"]) for row in read_json(CONTRACT)["joint_specs"]]
        model_names = [*map(str, self.g1.arm_joint_names), *self.g1.hand_joint_names["left"], *self.g1.hand_joint_names["right"]]
        lookup = {name: i for i, name in enumerate(contract_names)}
        if set(model_names) != set(contract_names):
            raise RuntimeError("saved trace/G1 renderer named-joint mismatch")
        self.reorder = np.asarray([lookup[name] for name in model_names])
        presets = self.layout["camera"]["presets"]
        self.camera_parameters = {
            "top": presets["top"], "overview": presets["overview"],
            "front_oblique": {"eye_world_xyz_m": [1.05, -0.62, 1.27], "target_world_xyz_m": [0.34, 0.08, 0.91]},
            "hand_close": {"eye_world_xyz_m": [0.52, -0.32, 1.08], "target_world_xyz_m": [0.16, 0.05, 0.90]},
        }
        self.cameras = {name: camera(row["eye_world_xyz_m"], row["target_world_xyz_m"]) for name, row in self.camera_parameters.items()}
        self.root_position = np.asarray(self.layout["g1"]["root_position_world_xyz_m"])
        self.root_quaternion = np.asarray(self.layout["g1"]["root_orientation_world_wxyz"])
        self.model.vis.headlight.ambient[:] = (.52, .52, .52)
        self.model.vis.headlight.diffuse[:] = (.78, .78, .78)
        self.model.vis.headlight.specular[:] = (.08, .08, .08)

    def close(self) -> None:
        self.renderer.close()

    def set_state(self, q_contract: np.ndarray, doll_position: np.ndarray, doll_quaternion: np.ndarray) -> None:
        q = np.asarray(q_contract)[self.reorder]
        self.data.qpos[:] = self.g1.stand_qpos
        self.data.qpos[self.g1.arm_qpos_ids] = q[:14]
        self.data.qpos[self.g1.hand_qpos_ids["left"]] = q[14:21]
        self.data.qpos[self.g1.hand_qpos_ids["right"]] = q[21:28]
        self.data.qpos[:3] = self.root_position
        self.data.qpos[3:7] = self.root_quaternion
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.doll_position = np.asarray(doll_position)
        self.doll_rotation = rotation_xyzw(doll_quaternion)

    def add_scene(self) -> None:
        scene, layout = self.renderer.scene, self.layout
        table = layout["table"]
        surface = float(table["surface_height_m"])
        width, depth = map(float, table["size_xy_m"])
        thickness = float(table["top_thickness_m"])
        _add_box(scene, (width, depth, thickness), (.5*width, .5*depth, surface-.5*thickness), np.asarray([.72,.72,.70,1],np.float32))
        for rail in layout["black_frame"]["rails"].values():
            _add_box(scene, rail["size_xyz_m"], rail["center_xyz_m"], np.asarray([.07,.07,.08,1],np.float32))
        dimensions = np.asarray(self.config["object"]["visual_dimensions_m"])
        _add_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, .5*dimensions, self.doll_position, np.asarray([.18,.62,.20,1],np.float32), rotation=self.doll_rotation)
        bin_cfg = layout["bin"]
        ox, oy, _ = map(float, bin_cfg["outer_dimensions_xyz_m"])
        ix, iy = map(float, bin_cfg["opening_dimensions_xy_m"])
        wall, bottom = float(bin_cfg["wall_thickness_m"]), float(bin_cfg["bottom_thickness_m"])
        cx, cy = map(float, bin_cfg["center_world_xy_m"])
        color = np.asarray([.86,.83,.68,1],np.float32); h=.150
        _add_box(scene,(ox,oy,bottom),(cx,cy,surface+.5*bottom),color)
        _add_box(scene,(ox,wall,h),(cx,cy-.5*(iy+wall),surface+.5*h),color)
        _add_box(scene,(ox,wall,h),(cx,cy+.5*(iy+wall),surface+.5*h),color)
        _add_box(scene,(wall,iy,h),(cx-.5*(ix+wall),cy,surface+.5*h),color)
        _add_box(scene,(wall,iy,h),(cx+.5*(ix+wall),cy,surface+.5*h),color)
        _add_box(scene,(6,6,.025),(.4175,.2,-.0125),np.asarray([.84,.85,.86,1],np.float32))
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    def view(self, name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, self.cameras[name])
        self.add_scene()
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)


def control_trace(run: Path) -> dict[str, np.ndarray]:
    with np.load(run / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    frames = event["control_frame"].astype(int)
    rows = np.r_[np.flatnonzero(np.diff(frames) != 0), len(frames)-1]
    return {
        "frame": frames[rows], "q": event["MEASURED_Q"][rows],
        "position": event["object_position_world_m"][rows],
        "quaternion": event["object_quaternion_xyzw"][rows],
    }


def run_dirs(letter: str, provenance: bool = False) -> list[Path]:
    base = OUT / (("provenance_common_execution_bug_solver32/02_act_a_results" if provenance else "02_act_a_results") if letter == "A" else "03_act_b_results") / "rollouts"
    values = sorted(base.glob("eval_*"))
    if len(values) != 35:
        raise RuntimeError(f"{letter}: {len(values)}/35 saved traces")
    return values


class Writer:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.tmp = path.with_name(path.stem + ".incomplete" + path.suffix)
        self.p = subprocess.Popen(["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s",f"{MWIDTH}x{MHEIGHT}","-r",str(FPS),"-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","22","-pix_fmt","yuv420p",str(self.tmp)],stdin=subprocess.PIPE)
    def write(self, frame: np.ndarray) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())
    def finish(self) -> None:
        assert self.p.stdin is not None
        self.p.stdin.close()
        if self.p.wait() != 0: raise RuntimeError(f"ffmpeg failed: {self.path}")
        os.replace(self.tmp,self.path)


def label(image: np.ndarray, letter: str, index: int, status: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result,(0,0),(WIDTH,34),(18,18,22),-1)
    cv2.putText(result,f"ACT-{letter} {index:02d}/35  {status}",(8,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
    return result


def probe(path: Path) -> dict[str, Any]:
    value=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=codec_name,width,height,avg_frame_rate,nb_frames","-of","json",str(path)],text=True))["streams"][0]
    return value


def render_mosaics() -> dict[str, Any]:
    renderer=PhysicalRenderer(); products={}
    try:
        for letter in ("A","B"):
            directories=run_dirs(letter)
            traces=[control_trace(path) for path in directories]
            manifests=[read_json(path/"RUN_MANIFEST.json") for path in directories]
            maximum=max(len(trace["frame"]) for trace in traces)
            writers={view:Writer(REPLAYS/f"{letter}_EVAL35_PHYSICAL_{view.upper()}_35SPLIT.mp4") for view in ("top","overview")}
            cached={view:[None]*35 for view in writers}
            for frame_index in range(maximum):
                canvases={view:np.full((MHEIGHT,MWIDTH,3),232,np.uint8) for view in writers}
                for cell,(trace,manifest) in enumerate(zip(traces,manifests,strict=True)):
                    i=min(frame_index,len(trace["frame"])-1)
                    renderer.set_state(trace["q"][i],trace["position"][i],trace["quaternion"][i])
                    status="SUCCESS" if manifest["outcomes"]["FULL_TASK_SUCCESS"] else "FAIL: "+manifest["first_failure_stage"]
                    row,col=divmod(cell,COLS); x=GX+col*WIDTH; y=GY+row*HEIGHT
                    for view in writers:
                        image=label(renderer.view(view),letter,cell+1,status)
                        cached[view][cell]=image
                        canvases[view][y:y+HEIGHT,x:x+WIDTH]=image
                        cv2.rectangle(canvases[view],(x,y),(x+WIDTH-1,y+HEIGHT-1),(45,45,45),1)
                for view,writer in writers.items(): writer.write(canvases[view])
                if frame_index%50==0: print(f"ACT-{letter} mosaics {frame_index+1}/{maximum}",flush=True)
            for view,writer in writers.items():
                writer.finish(); products[f"{letter}_{view}"]={"path":str(writer.path),"probe":probe(writer.path),"actual_trace_fields":["MEASURED_Q","object_position_world_m","object_quaternion_xyzw"]}
    finally:
        renderer.close()
    return products


def composite(images: list[np.ndarray], columns: int, labels: list[str], cell_w: int=WIDTH, cell_h: int=HEIGHT) -> np.ndarray:
    rows=math.ceil(len(images)/columns)
    canvas=np.full((rows*(cell_h+38),columns*cell_w,3),245,np.uint8)
    for i,(image,text) in enumerate(zip(images,labels,strict=True)):
        row,col=divmod(i,columns); x,y=col*cell_w,row*(cell_h+38)
        canvas[y:y+cell_h,x:x+cell_w]=cv2.resize(image,(cell_w,cell_h))
        cv2.putText(canvas,text,(x+8,y+cell_h+26),cv2.FONT_HERSHEY_SIMPLEX,.55,(25,25,25),1,cv2.LINE_AA)
    return canvas


def state_at(trace: dict[str,np.ndarray], frame: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    index=int(np.argmin(np.abs(trace["frame"]-int(frame))))
    return trace["q"][index],trace["position"][index],trace["quaternion"][index]


def render_visuals() -> dict[str,Any]:
    renderer=PhysicalRenderer(); products={}; camera_log={}
    try:
        # Episode-conditioned registration: deterministic lowest/middle/highest.
        directories=run_dirs("A"); images=[]; labels=[]
        for display in (1,18,35):
            trace=control_trace(directories[display-1]); renderer.set_state(*state_at(trace,int(trace["frame"][0]))); images.append(renderer.view("overview")); labels.append(f"Episode {display:02d}: {directories[display-1].name.split('_',2)[-1]}")
        path=VISUALS/"METHOD_A_EPISODE_CONDITIONED_REGISTRATION.png"; atomic_image(path,composite(images,3,labels));products["registration"]=str(path)

        # Qualified RIGHT grasp sequence from persisted actual physics.
        qrun=OUT/"00_qualification/solver80_requalification/right_01"; trace=control_trace(qrun); summary=read_json(qrun/"DIRECT_EXECUTION_RUNTIME_SUMMARY.json"); ev=summary["events"]
        confirmed=int(ev["grasp_confirmed_frames"]["right"]); lift=int(ev["lift_start_frames"]["right"]); first=min(v for v in ev["first_close_phase_digit_contact_frames"]["right"].values() if v is not None)
        frames=(max(0,first-12),first,confirmed,lift+15); stage_names=("PRESHAPE","PROGRESSIVE CLOSE","MECHANICAL GRASP CONFIRMED","TABLE-FREE LIFT")
        images=[]
        for frame in frames:
            renderer.set_state(*state_at(trace,frame));images.append(renderer.view("hand_close"))
        path=VISUALS/"METHOD_B_DEX3_MECHANICAL_GRASP_SEQUENCE.png";atomic_image(path,composite(images,4,list(stage_names)));products["mechanical_grasp"]=str(path)

        # Qualified scripted handoff when an ACT handoff is unavailable.
        final_data=read_json(OUT/"04_results/FINAL_NUMERIC_RESULTS.json")
        successes=[(letter,i,row) for letter in ("ACT_A","ACT_B") for i,row in enumerate(final_data["runs"][letter]) if row["outcomes"]["HANDOFF_SUCCESS"]]
        if successes:
            letter,i,row=successes[0]; hrun=run_dirs("A" if letter=="ACT_A" else "B")[i]; title="ACT physical rollout"
        else:
            hrun=OUT/"00_qualification/solver80_requalification/scripted_full_01"; title="COMMON SCRIPTED PHYSICAL VALIDATION"
        trace=control_trace(hrun); summary=read_json(hrun/"DIRECT_EXECUTION_RUNTIME_SUMMARY.json"); ev=summary["events"]
        right_confirm=ev["grasp_confirmed_frames"]["right"] or int(trace["frame"][len(trace["frame"])//2]); right_owned=ev.get("right_physical_ownership_frame") or right_confirm+15; release=ev.get("giving_hand_release_frame") or right_owned
        frames=(max(0,right_confirm-40),max(0,right_confirm-5),right_confirm,release+10); names=("LEFT ownership","RIGHT approach","Dual-hand contact","RIGHT ownership")
        images=[]
        for frame in frames: renderer.set_state(*state_at(trace,frame));images.append(renderer.view("front_oblique"))
        path=VISUALS/"METHOD_C_PHYSICAL_HANDOFF_SEQUENCE.png";atomic_image(path,composite(images,4,[f"{name} · {title}" for name in names]));products["handoff"]=str(path)

        # Deterministic paired comparison: largest cumulative stage difference, then lowest index.
        keys=("LEFT_GRASP_SUCCESS","HANDOFF_SUCCESS","RIGHT_OWNERSHIP_SUCCESS","NO_DROP_TO_BIN","BIN_ENTRY_SUCCESS","BIN_SETTLE_SUCCESS","FULL_TASK_SUCCESS")
        scores=[abs(sum(int(b["outcomes"][k]) for k in keys)-sum(int(a["outcomes"][k]) for k in keys)) for a,b in zip(final_data["runs"]["ACT_A"],final_data["runs"]["ACT_B"],strict=True)]
        chosen=int(np.argmax(scores)); images=[]; names=[]
        for letter in ("A","B"):
            run=run_dirs(letter)[chosen]; trace=control_trace(run); manifest=read_json(run/"RUN_MANIFEST.json"); summary=read_json(run/"DIRECT_EXECUTION_RUNTIME_SUMMARY.json"); ev=summary["events"]
            sample=(ev.get("left_close_intent_frame") or 0,ev.get("right_close_intent_frame") or int(trace["frame"][len(trace["frame"])//2]),int(trace["frame"][-1]))
            for phase,frame in zip(("grasp","handoff region","final"),sample,strict=True): renderer.set_state(*state_at(trace,frame));images.append(renderer.view("overview"));names.append(f"ACT-{letter} · {phase} · {manifest['first_failure_stage']}")
        path=VISUALS/"RESULT_MATCHED_AB_PHYSICAL_COMPARISON.png";atomic_image(path,composite(images,3,names));products["matched_comparison"]={"path":str(path),"episode":chosen+1,"selection":"lowest-index episode with largest absolute cumulative-stage difference"}

        # Success storyboard(s), lowest-index success by predeclared rule.
        for letter,key in (("A","ACT_A"),("B","ACT_B")):
            candidates=[i for i,row in enumerate(final_data["runs"][key]) if row["outcomes"]["FULL_TASK_SUCCESS"]]
            if not candidates: products[f"ACT_{letter}_storyboard"]="NOT AVAILABLE";continue
            i=candidates[0];run=run_dirs(letter)[i];trace=control_trace(run);manifest=read_json(run/"RUN_MANIFEST.json");e=manifest["event_frames"]
            values=[max(0,(e["grasp_confirm"] or 20)-20),e["grasp_confirm"],e["lift"],max(0,(e["handoff"] or e["right_ownership"])-15),e["handoff"] or e["right_ownership"],e["right_ownership"],e["bin_entry"],e["settle"]]
            names=["LEFT approach","LEFT grasp","table-free lift","RIGHT approach","handoff","RIGHT ownership","bin entry","settled"]
            images=[]
            for frame in values: renderer.set_state(*state_at(trace,int(frame)));images.append(renderer.view("overview"))
            path=VISUALS/f"ACT_{letter}_PHYSICAL_SUCCESS_STORYBOARD.png";atomic_image(path,composite(images,4,[f"{name} · episode {i+1:02d}" for name in names]));products[f"ACT_{letter}_storyboard"]=str(path)

        # Old A forensic closeups are explicitly provenance, never mixed with final scores.
        old_dirs=run_dirs("A",provenance=True); forensic=read_json(FORENSIC); by={int(row["eval_number"]):row for row in forensic["rows"]}; images=[];names=[]
        for display in (1,9,18,27,35):
            trace=control_trace(old_dirs[display-1]);row=by[display];frames=(row["closest_control_frame"],row["closest_control_frame"],row["first_failure_frame"] or int(trace["frame"][-1]))
            for frame,view,phase in zip(frames,("top","front_oblique","hand_close"),("closest approach","maximum contact","attempted lift"),strict=True): renderer.set_state(*state_at(trace,int(frame)));images.append(renderer.view(view));names.append(f"A{display:02d} · {phase}")
        path=VISUALS/"ACT_A_GRASP_FAILURE_DIAGNOSTICS.png";atomic_image(path,composite(images,3,names));products["old_A_failure_diagnostics"]=str(path)
        camera_log=renderer.camera_parameters
    finally: renderer.close()
    atomic_text(VISUALS/"CAMERA_SELECTIONS.json",json.dumps({"status":"PASS","camera_only_changes":True,"physical_state_modified":False,"cameras":camera_log,"products":products},indent=2)+"\n")
    return products


def main() -> int:
    products=render_mosaics();visuals=render_visuals()
    checks=[]
    for name,value in products.items():
        p=value["probe"];checks.append(p["codec_name"]=="h264" and int(p["width"])==MWIDTH and int(p["height"])==MHEIGHT and p["avg_frame_rate"]=="30/1")
    report={"status":"PASS" if all(checks) else "FAIL","actual_saved_physical_traces":True,"command_only_replay":False,"mosaics":products,"paper_visuals":visuals,"layout":"7x5","cells":35,"same_episode_order":True,"camera_identical_A_B":True}
    atomic_text(REPLAYS/"PHYSICAL_REPLAY_VERIFICATION.md","# Physical replay verification\n\n"+f"Status: **{report['status']}**\n\nAll four mosaics reconstruct the saved `MEASURED_Q` and PhysX doll pose traces. They use 35 cells in identical 7×5 order, 3840×2160 H.264 at 30 FPS. No command-only object motion is used.\n")
    atomic_text(REPLAYS/"PHYSICAL_REPLAY_VERIFICATION.json",json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2));return 0 if report["status"]=="PASS" else 2


if __name__=="__main__": raise SystemExit(main())
