"""Shared offline I/O and signal helpers for the MagSafe gripper pipeline."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DEFAULT_ACTION = Path("/home/jbnu/aloha_g1_dataset/evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz")
DEFAULT_ACTION_KEY = "optimized_action"
LEFT_INDEX, RIGHT_INDEX = 6, 13

def load_action(path: Path, left_index: int, right_index: int, fps_override=None):
    path = path.expanduser().resolve()
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if DEFAULT_ACTION_KEY not in z.files:
                raise ValueError(f"{path}: missing {DEFAULT_ACTION_KEY}; keys={z.files}")
            action = z[DEFAULT_ACTION_KEY].astype(float)
            fps = float(fps_override if fps_override is not None else z["fps"] if "fps" in z.files else 0)
            timestamps = z["timestamp"].astype(float) if "timestamp" in z.files else None
    elif path.suffix == ".npy":
        action=np.load(path,allow_pickle=False).astype(float); fps=float(fps_override or 0); timestamps=None
    else: raise ValueError("input must be .npz or .npy")
    if action.ndim != 2 or max(left_index,right_index) >= action.shape[1]:
        raise ValueError(f"invalid action shape/index: {action.shape}, {left_index}/{right_index}")
    if not np.isfinite(action).all(): raise ValueError("action contains NaN/Inf")
    if fps <= 0: raise ValueError("fps is missing; provide --fps")
    if timestamps is None: timestamps=np.arange(len(action))/fps
    if timestamps.shape != (len(action),) or not np.all(np.diff(timestamps)>0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return action,timestamps,fps

def smooth(x, window=7):
    if window<=1:return np.asarray(x,float).copy()
    if window%2==0:window+=1
    pad=window//2
    return np.convolve(np.pad(x,pad,mode="edge"),np.ones(window)/window,mode="valid")

def two_cluster_threshold(x):
    x=np.asarray(x,float); c=np.percentile(x,[20,80])
    for _ in range(50):
        labels=np.abs(x[:,None]-c[None,:]).argmin(1)
        new=np.array([x[labels==i].mean() if np.any(labels==i) else c[i] for i in range(2)])
        if np.allclose(new,c,rtol=0,atol=1e-12):break
        c=new
    c.sort(); return float(c[0]),float(c[1]),float(c.mean())

def atomic_json(path:Path,data):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n"); tmp.replace(path)

def runs(values):
    out=[]; start=0
    for i in range(1,len(values)+1):
        if i==len(values) or values[i]!=values[start]:
            out.append((start,i-1,str(values[start]))); start=i
    return out
