#!/usr/bin/env python3
"""Generate and validate an official-LeRobot RTC trajectory for episode 49."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path("/home/jbnu/aloha_g1_dataset")
LEROBOT = Path("/home/jbnu/lerobot-smolvla")
sys.path.insert(0, str(LEROBOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from lerobot.__version__ import __version__ as lerobot_version  # noqa: E402
from lerobot.configs import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.rtc import ActionQueue, RTCConfig  # noqa: E402

from evaluate_smolvla_checkpoints import (  # noqa: E402
    ACTION_DIM, TASK, LeRobotDataset, SmolVLAPolicy, episode_bounds,
    finite_action, make_pre_post_processors, seed_all,
)
from validate_smolvla_in_stationary_aloha_mujoco import (  # noqa: E402
    dynamic_replay, fk, load_validated_model, mapped_limit_counts, mapped_qpos,
)

CHECKPOINT = ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407/checkpoints/020000/pretrained_model"
DATASET_ROOT = ROOT / "lerobot_magsafe_50_cam_high_v3"
REPO_ID = "local/magsafe_aloha_50_cam_high_v3"
H10_FILE = ROOT / "evaluation/smolvla_20k_chunk_stitched_preflight/episode_000049_chunk_stitched.npz"
FULL_PREDICTION_FILE = ROOT / "evaluation/smolvla_20k_full_predictions/episode_000049_prediction.npz"
XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
OUTPUT = ROOT / "evaluation/smolvla_episode49_rtc"
FPS = 30.0
SEED = 1000
HORIZONS = (8, 10, 12)
GUIDANCE = (5.0, 10.0, 15.0)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--screen-frames", type=int, default=150)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def atomic_json(path: Path, x: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    tmp = path.with_suffix(path.suffix + ".incomplete")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def reset_once(policy: Any, pre: Any, post: Any) -> None:
    policy.reset(); pre.reset(); post.reset()


def raw_batch(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation.images.cam_high": item["observation.images.cam_high"],
        "observation.state": item["observation.state"],
        "task": TASK,
    }


def latency_measure(policy: Any, pre: Any, dataset: Any, start: int, device: str) -> dict[str, Any]:
    cfg = RTCConfig(enabled=False)
    policy.config.rtc_config = cfg; policy.init_rtc_processor()
    policy.reset(); pre.reset()
    times = []
    for i in range(6):
        seed_all(SEED + i)
        batch = pre(raw_batch(dataset[start + i]))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i:
            times.append(dt)
        if out.shape != (1, 50, ACTION_DIM):
            raise RuntimeError(f"Latency probe chunk shape mismatch: {out.shape}")
    arr = np.asarray(times)
    delay = int(math.ceil(float(arr.max()) * FPS))
    return {"samples_s": arr.tolist(), "mean_ms": float(arr.mean()*1000),
            "p95_ms": float(np.percentile(arr, 95)*1000), "max_ms": float(arr.max()*1000),
            "inference_delay_frames": delay, "official_rule": "ceil(max measured latency / (1/fps))",
            "device": device}


def trajectory(policy: Any, pre: Any, post: Any, dataset: Any, start: int, n: int,
               horizon: int, guidance: float, delay: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = RTCConfig(enabled=True, execution_horizon=horizon, max_guidance_weight=guidance,
                    prefix_attention_schedule=RTCAttentionSchedule.EXP, debug=False)
    policy.config.rtc_config = cfg; policy.init_rtc_processor()
    queue = ActionQueue(cfg)
    reset_once(policy, pre, post)
    result = np.empty((n, ACTION_DIM), np.float32)
    chunk_starts: list[int] = []
    latencies: list[float] = []
    for t in range(n):
        if t % horizon == 0 or queue.empty():
            chunk_starts.append(t)
            item = dataset[start + t]
            batch = pre(raw_batch(item))
            previous = queue.get_left_over()
            seed_all(SEED + t)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                actions = policy.predict_action_chunk(
                    batch, inference_delay=delay, prev_chunk_left_over=previous,
                    execution_horizon=horizon)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            if actions.shape != (1, 50, ACTION_DIM) or not torch.isfinite(actions).all():
                raise RuntimeError(f"RTC output invalid at frame {t}: {actions.shape}")
            original = actions.squeeze(0).clone()
            processed = post(actions).squeeze(0)
            if processed.shape != (50, ACTION_DIM) or not torch.isfinite(processed).all():
                raise RuntimeError(f"RTC postprocessed output invalid at frame {t}: {processed.shape}")
            queue.merge(original, processed, delay)
        action = queue.get()
        if action is None:
            raise RuntimeError(f"RTC ActionQueue empty at frame {t}")
        result[t] = finite_action(action, f"rtc_action[{t}]", (ACTION_DIM,))
    return result, np.asarray(chunk_starts, np.int64), np.asarray(latencies)


def transitions(expert: np.ndarray, x: np.ndarray, j: int) -> dict[str, Any]:
    threshold = float((expert[:, j].min()+expert[:, j].max())/2)
    e = np.flatnonzero((expert[1:, j] >= threshold) != (expert[:-1, j] >= threshold))+1
    p = np.flatnonzero((x[1:, j] >= threshold) != (x[:-1, j] >= threshold))+1
    return {"threshold": threshold, "expert": e.tolist(), "predicted": p.tolist()}


def high_frequency_energy(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0))**2 / max(len(x), 1)
    freq = np.fft.rfftfreq(len(x), d=1/FPS)
    mask = (freq >= 3) & (freq <= 15)
    return float(power[mask].mean()) if mask.any() else 0.0


def values(x: np.ndarray, expert: np.ndarray, chunks: np.ndarray, model: Any,
           label: str, horizon: int, guidance: float, stage: str) -> dict[str, Any]:
    d = np.abs(np.diff(x, axis=0)); vel = d*FPS
    acc = np.abs(np.diff(x, n=2, axis=0))*FPS**2
    qpos, gripper_map = mapped_qpos(x)
    limit_count, per_qpos = mapped_limit_counts(qpos, model)
    f = fk(model, qpos)
    lj = np.linalg.norm(np.diff(f["left_position_m"], axis=0), axis=1)*1000
    rj = np.linalg.norm(np.diff(f["right_position_m"], axis=0), axis=1)*1000
    boundary_ids = chunks[1:]
    boundary_jump = np.max(np.abs(x[boundary_ids]-x[boundary_ids-1]), axis=1) if len(boundary_ids) else np.array([0])
    error = x-expert
    return {
        "stage": stage, "label": label, "execution_horizon": horizon,
        "guidance_weight": guidance, "frames": len(x),
        "action_mae": float(np.mean(np.abs(error))), "action_rmse": float(np.sqrt(np.mean(error**2))),
        "max_jump": float(d.max(initial=0)), "p95_jump": float(np.percentile(d,95)),
        "p99_jump": float(np.percentile(d,99)), "max_velocity": float(vel.max(initial=0)),
        "p95_velocity": float(np.percentile(vel,95)), "p99_velocity": float(np.percentile(vel,99)),
        "max_acceleration": float(acc.max(initial=0)), "p95_acceleration": float(np.percentile(acc,95)),
        "p99_acceleration": float(np.percentile(acc,99)), "high_frequency_energy_3_15hz": high_frequency_energy(x),
        "replanning_boundary_max_jump": float(boundary_jump.max(initial=0)),
        "replanning_boundary_p99_jump": float(np.percentile(boundary_jump,99)),
        "joint_limit_violations": int(limit_count), "per_qpos_limit_violations": per_qpos,
        "nan_inf_count": int((~np.isfinite(x)).sum()), "existing_gripper_mapped_frames": gripper_map,
        "left_fk_max_jump_mm": float(lj.max(initial=0)), "left_fk_p99_jump_mm": float(np.percentile(lj,99)),
        "right_fk_max_jump_mm": float(rj.max(initial=0)), "right_fk_p99_jump_mm": float(np.percentile(rj,99)),
        "left_gripper_transitions": transitions(expert,x,6),
        "right_gripper_transitions": transitions(expert,x,13),
    }


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["joint_limit_violations"], row["nan_inf_count"], row["action_rmse"],
            row["p99_jump"], row["p99_acceleration"], row["high_frequency_energy_3_15hz"])


def plot_metrics(path: Path, expert: np.ndarray, h10: np.ndarray, rtc: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(7,2,figsize=(16,19),sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(expert[:,j],label="expert",lw=.8); ax.plot(h10[:,j],label="H10",lw=.6)
        ax.plot(rtc[:,j],label="RTC",lw=.7); ax.set_title(f"joint {j}")
    axes.flat[0].legend(); fig.tight_layout(); fig.savefig(path/"joint_trajectories.png",dpi=130); plt.close(fig)
    for name, fn in (("jump",lambda x:np.max(np.abs(np.diff(x,axis=0)),axis=1)),
                     ("acceleration",lambda x:np.max(np.abs(np.diff(x,n=2,axis=0)),axis=1)*FPS**2)):
        fig,ax=plt.subplots(figsize=(13,4))
        ax.plot(fn(h10),label="H10");ax.plot(fn(rtc),label="RTC");ax.legend();ax.set_title(name)
        fig.tight_layout();fig.savefig(path/f"{name}.png",dpi=140);plt.close(fig)


def overlay(image: np.ndarray, text: str) -> np.ndarray:
    im=Image.fromarray(image);d=ImageDraw.Draw(im);d.rectangle((8,8,460,38),fill=(0,0,0));d.text((16,15),text,fill=(255,255,255))
    return np.asarray(im)


def videos(model: Any, rtc: np.ndarray, h10: np.ndarray, output: Path) -> None:
    cam=mujoco.MjvCamera();cam.azimuth=135;cam.elevation=-24;cam.distance=1.9;cam.lookat[:]=[.25,0,.18]
    def writer(path:Path,width:int):
        return subprocess.Popen(["ffmpeg","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x480",
            "-r","30","-i","pipe:0","-an","-vcodec","mpeg4","-pix_fmt","yuv420p","-q:v","3","-y",str(path)],stdin=subprocess.PIPE)
    rq,_=mapped_qpos(rtc);hq,_=mapped_qpos(h10)
    render=mujoco.Renderer(model,height=480,width=640);data=mujoco.MjData(model)
    p=writer(output/"aloha_rtc_kinematic.mp4",640)
    p2=writer(output/"aloha_h10_vs_rtc_side_by_side.mp4",1280)
    try:
        for t in range(len(rtc)):
            imgs=[]
            for q,label in ((hq[t],"H10"),(rq[t],"RTC")):
                data.qpos[:]=q;data.qvel[:]=0;mujoco.mj_forward(model,data);render.update_scene(data,camera=cam)
                imgs.append(overlay(render.render(),f"{label} frame {t}/{len(rtc)-1}"))
            assert p.stdin and p2.stdin
            p.stdin.write(imgs[1].tobytes());p2.stdin.write(np.concatenate(imgs,axis=1).tobytes())
        p.stdin.close();p2.stdin.close()
        if p.wait() or p2.wait(): raise RuntimeError("ffmpeg video encoding failed")
    finally:
        render.close()
        for proc in (p,p2):
            if proc.poll() is None: proc.terminate()


def main() -> int:
    a=args()
    if a.dry_run == a.execute: raise ValueError("Specify exactly one of --dry-run or --execute")
    if lerobot_version != "0.6.1": raise RuntimeError(f"Expected LeRobot 0.6.1, got {lerobot_version}")
    if a.device!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA device required")
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise RuntimeError(f"Refusing overwrite: {a.output_dir}")
    a.output_dir.mkdir(parents=True);(a.output_dir/"plots").mkdir()
    dataset=LeRobotDataset(a.repo_id,root=a.dataset_root,download_videos=False)
    start,end=episode_bounds(dataset,49); total=end-start
    with np.load(H10_FILE,allow_pickle=False) as z:
        expert=z["expert_action"].astype(np.float32)
        h10=z["chunk_stitched_h10"].astype(np.float32);frame=z["frame_index"].astype(np.int64)
        timestamp=z["timestamp"].astype(np.float32)
    with np.load(FULL_PREDICTION_FILE,allow_pickle=False) as z:
        state=z["observation_state"].astype(np.float32)
        if not np.array_equal(z["frame_index"],frame) or not np.allclose(z["expert_action"],expert):
            raise RuntimeError("Full-prediction state source does not align with H10 comparison file")
    if total!=990 or expert.shape!=(total,14): raise RuntimeError(f"Episode/input mismatch {total} {expert.shape}")
    policy=SmolVLAPolicy.from_pretrained(a.checkpoint,local_files_only=True).to(a.device);policy.eval()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(a.checkpoint))
    latency=latency_measure(policy,pre,dataset,start,a.device);delay=latency["inference_delay_frames"]
    model,_=load_validated_model(XML)
    screen_n=min(a.screen_frames,total)
    if a.dry_run: screen_n=min(20,screen_n)
    rows=[]; cache={}
    for h in HORIZONS:
        for g in GUIDANCE:
            x,ch,lats=trajectory(policy,pre,post,dataset,start,screen_n,h,g,delay)
            row=values(x,expert[:screen_n],ch,model,f"rtc_h{h}_g{g:g}",h,g,"screen")
            row["inference_latency_mean_ms"]=float(lats.mean()*1000);cache[(h,g)]=(x,ch,lats);rows.append(row)
            print(f"screen h={h} g={g}: limit={row['joint_limit_violations']} rmse={row['action_rmse']:.5f} p99={row['p99_jump']:.5f}",flush=True)
    top=sorted(rows,key=rank_key)[:3]
    full_results={}
    if not a.dry_run:
        for selected in top:
            h,g=selected["execution_horizon"],selected["guidance_weight"]
            x,ch,lats=trajectory(policy,pre,post,dataset,start,total,h,g,delay)
            row=values(x,expert,ch,model,f"rtc_h{h}_g{g:g}",h,g,"full")
            row["inference_latency_mean_ms"]=float(lats.mean()*1000);rows.append(row);full_results[(h,g)]=(x,ch,lats,row)
            print(f"full h={h} g={g}: limit={row['joint_limit_violations']} rmse={row['action_rmse']:.5f} p99={row['p99_jump']:.5f}",flush=True)
        selected=min((v for v in full_results.values()),key=lambda v:rank_key(v[3]))
        rtc,ch,lats,best=selected; h,g=best["execution_horizon"],best["guidance_weight"]
        h10row=values(h10,expert,np.arange(0,total,10),model,"chunk_stitched_h10",10,0,"baseline")
        expertrow=values(expert,expert,np.arange(0,total,10),model,"expert_action",0,0,"baseline")
        rows.extend([h10row,expertrow])
        q,_=mapped_qpos(rtc)
        dynamic={"status":"DYNAMIC_REPLAY_NOT_AVAILABLE"}
        if best["joint_limit_violations"]==0 and best["nan_inf_count"]==0:
            ctrl=rtc.copy();ctrl[:,6]=q[:,6];ctrl[:,13]=q[:,14]
            dyn,_=dynamic_replay(model,q[0],ctrl);dynamic={"status":"SUCCESS",**dyn}
        payload=dict(rtc_action=rtc,expert_action=expert,observation_state=state,frame_index=frame,timestamp=timestamp,
            execution_horizon=np.asarray(h),guidance_weight=np.asarray(g),inference_delay_frames=np.asarray(delay),
            chunk_start_frames=ch)
        tmp=a.output_dir/"episode_000049_rtc_trajectory.npz.incomplete"
        with tmp.open("wb") as f:np.savez_compressed(f,**payload)
        os.replace(tmp,a.output_dir/"episode_000049_rtc_trajectory.npz")
        plot_metrics(a.output_dir/"plots",expert,h10,rtc)
        videos(model,rtc,h10,a.output_dir)
        reduction=lambda key:float((h10row[key]-best[key])/h10row[key]*100) if h10row[key] else 0.
        residual_jitter_reasons=[]
        if best["p99_acceleration"] > 2.0 * expertrow["p99_acceleration"]:
            residual_jitter_reasons.append("p99_acceleration_exceeds_2x_expert")
        if best["high_frequency_energy_3_15hz"] > 2.0 * expertrow["high_frequency_energy_3_15hz"]:
            residual_jitter_reasons.append("high_frequency_energy_exceeds_2x_expert")
        if (best["left_fk_max_jump_mm"] > h10row["left_fk_max_jump_mm"] or
                best["right_fk_max_jump_mm"] > h10row["right_fk_max_jump_mm"]):
            residual_jitter_reasons.append("fk_max_jump_worse_than_h10")
        visual_block=bool(
            best["joint_limit_violations"] or best["nan_inf_count"] or residual_jitter_reasons
        )
        report={"evaluation_type":"training-set teacher-forced offline RTC sanity evaluation",
          "warning":"Not closed-loop, validation, or generalization performance.",
          "official_sources":[str(LEROBOT/"examples/rtc/eval_dataset.py"),str(LEROBOT/"src/lerobot/rollout/inference/rtc.py")],
          "latency":latency,"screening_top3":[{k:r[k] for k in ("label","execution_horizon","guidance_weight","joint_limit_violations","action_rmse","p99_jump")} for r in top],
          "selected":best,"h10_baseline":h10row,"expert_baseline":expertrow,"dynamic_replay":dynamic,
          "h10_reduction_percent":{"p99_jump":reduction("p99_jump"),"p99_acceleration":reduction("p99_acceleration"),
             "high_frequency_energy_3_15hz":reduction("high_frequency_energy_3_15hz")},
          "residual_jitter_assessment":{"blocked":bool(residual_jitter_reasons),
              "reasons":residual_jitter_reasons,
              "criteria":{"p99_acceleration_vs_expert":2.0,
                  "high_frequency_energy_vs_expert":2.0,
                  "fk_max_jump_must_not_worsen_vs_h10":True}},
          "g1_conversion_status":"RTC_ALOHA_SAFETY_BLOCKED" if visual_block else "ELIGIBLE_AFTER_HUMAN_VIDEO_REVIEW",
          "hardware_commands_sent":False,"isaac_lab_executed":False,"g1_conversion_executed":False}
    else:
        report={"evaluation_type":"RTC dry-run","latency":latency,"screening_rows":rows,
                "hardware_commands_sent":False,"isaac_lab_executed":False,"g1_conversion_executed":False}
    atomic_csv(a.output_dir/"rtc_candidate_comparison.csv",rows)
    atomic_json(a.output_dir/"rtc_report.json",report)
    return 0


if __name__=="__main__": raise SystemExit(main())
