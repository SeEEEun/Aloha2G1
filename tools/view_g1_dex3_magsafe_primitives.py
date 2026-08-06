#!/usr/bin/env python3
"""Offline, kinematic-only MuJoCo viewer for recorded Dex3 primitives."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/dex3_magsafe_grasp_primitives.json"
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")


def args_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--primitive")
    p.add_argument("--side", choices=("left", "right"))
    p.add_argument("--loop-all", action="store_true")
    return p.parse_args()


def main() -> int:
    a = args_parse()
    import mujoco
    import mujoco.viewer
    with a.config.expanduser().open() as f: cfg = json.load(f)
    model = mujoco.MjModel.from_xml_path(str(MODEL)); data = mujoco.MjData(model)
    base = model.key_qpos[0].copy() if model.nkey else np.zeros(model.nq)
    # A symmetric, natural task-ready display pose; never read as trajectory data.
    ready = {
        "left_shoulder_pitch_joint": 0.25, "left_shoulder_roll_joint": 0.35,
        "left_elbow_joint": 1.15, "right_shoulder_pitch_joint": 0.25,
        "right_shoulder_roll_joint": -0.35, "right_elbow_joint": 1.15,
    }
    for name, value in ready.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0: base[model.jnt_qposadr[jid]] = value
    names = list(cfg.get("primitives", {}))
    if a.primitive:
        if a.primitive not in names: raise ValueError(f"unknown primitive {a.primitive}")
        names = [a.primitive]
    elif a.side:
        names = [n for n in names if n.startswith(a.side.upper() + "_")]
    elif not a.loop_all:
        raise ValueError("select --primitive, --side, or --loop-all")
    names = [n for n in names if cfg["primitives"].get(n)]
    if not names: raise ValueError("no recorded primitives selected")

    def apply(name: str) -> None:
        item = cfg["primitives"][name]; side = item["authoritative_side"]
        expected = cfg["joint_names"][f"{side}_dex3"]
        if item["joint_names"] != expected: raise ValueError(f"{name}: joint order mismatch")
        q = np.asarray(item["qpos"], float)
        if q.shape != (7,) or not np.isfinite(q).all(): raise ValueError(f"{name}: invalid qpos")
        data.qpos[:] = base
        for joint, value in zip(expected, q):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jid < 0: raise ValueError(f"model missing joint {joint}")
            data.qpos[model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(model, data)
        print(f"Showing {name} ({side} fingers only; qpos + mj_forward)")

    index, changed = 0, time.monotonic()
    apply(names[index])
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if a.loop_all and time.monotonic() - changed >= 2.0:
                index = (index + 1) % len(names); apply(names[index]); changed = time.monotonic()
            viewer.sync(); time.sleep(1 / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
