#!/usr/bin/env python3
"""GUI/offscreen viewer for the selected static Dex3 phone primitive."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old  # noqa: E402


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def model_data(payload):
    pose = payload["phone_proxy_pose"].astype(float)
    rpy = Rotation.from_quat(pose[3:][[1,2,3,0]]).as_euler("xyz")
    model, _ = old.expanded_phone_model(pose[:3], rpy)
    data = mujoco.MjData(model)
    data.qpos[:] = payload["full_g1_qpos"]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return model, data


def render_one(model, data, path, az, el, title, payload):
    renderer = mujoco.Renderer(model, width=640, height=480)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.22, 0, 1.0]
    cam.distance, cam.azimuth, cam.elevation = 1.25, az, el
    renderer.update_scene(data, camera=cam)
    scene = renderer.scene
    def sphere(point, color, radius=.006):
        i = scene.ngeom
        mujoco.mjv_initGeom(scene.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, 0, 0]), np.asarray(point),
                            np.eye(3).ravel(), np.asarray(color, np.float32))
        scene.ngeom += 1
    def arrow(start, vector, color, scale=.045):
        i = scene.ngeom
        end = np.asarray(start)+scale*np.asarray(vector)
        mujoco.mjv_initGeom(scene.geoms[i], mujoco.mjtGeom.mjGEOM_ARROW,
                            np.array([.003, .003, .003]), np.zeros(3),
                            np.eye(3).ravel(), np.asarray(color, np.float32))
        mujoco.mjv_connector(scene.geoms[i], mujoco.mjtGeom.mjGEOM_ARROW,
                             .004, np.asarray(start), end)
        scene.ngeom += 1
    for point, normal in zip(payload["intended_contact_points"],
                             payload["contact_normals"]):
        sphere(point, [1, .15, .05, 1])
        arrow(point, -normal, [1, .8, .05, 1])
    phone = payload["phone_proxy_pose"]
    prot = Rotation.from_quat(phone[3:][[1,2,3,0]]).as_matrix()
    for axis, color in zip(prot.T, ([1,0,0,1], [0,1,0,1], [0,0,1,1])):
        arrow(phone[:3], axis, color, .09)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    for axis, color in zip(data.xmat[torso].reshape(3,3).T,
                           ([1,0,0,1], [0,1,0,1], [0,0,1,1])):
        arrow(data.xpos[torso], axis, color, .08)
    im = Image.fromarray(renderer.render())
    renderer.close()
    draw = ImageDraw.Draw(im)
    draw.rectangle((10, 10, 630, 86), fill=(0, 0, 0))
    draw.text((20, 18), title, fill="white")
    draw.text((20, 42), "Red/green/blue convention: torso X forward, Y lateral, Z up", fill="white")
    draw.text((20, 62), f"contacts={len(payload['intended_contact_points'])}; "
              f"min forbidden={payload['robot_collision_clearances'].min(initial=1):.5f} m",
              fill="white")
    im.save(path)


def render_all(grasp: Path, output: Path):
    payload = load(grasp)
    model, data = model_data(payload)
    kind = str(payload["selected_grasp_type"])
    for name, az, el in (("front", 180, -5), ("top", 180, -88),
                         ("side", 90, -5), ("contacts", 165, -15)):
        render_one(model, data, output/f"selected_grasp_{name}.png",
                   az, el, f"{kind} | {name} | qpos + mj_forward", payload)
    compare = {
        "THREE_POINT_FACE_CLAMP": "three_point_face_clamp.png",
        "THUMB_INDEX_WITH_BOTTOM_SUPPORT": "thumb_index_bottom_support.png",
        "BIMANUAL_SIDE_SUPPORT": "bimanual_side_support.png",
    }
    shutil.copy2(output/"selected_grasp_contacts.png", output/compare[kind])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grasp", type=Path, default=ROOT/(
        "converted_runs/g1_dex3_phone_grasp_primitives/selected_static_phone_grasp.npz"))
    p.add_argument("--render", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    payload = load(a.grasp)
    if a.render:
        render_all(a.grasp, a.output or a.grasp.parent)
        return 0
    model, data = model_data(payload)
    from mujoco import viewer
    with viewer.launch_passive(model, data) as window:
        window.cam.lookat[:] = [.22, 0, 1.]
        window.cam.distance, window.cam.azimuth, window.cam.elevation = 1.25, 180, -8
        print("Static qpos + mj_forward only. No mj_step or actuator control.")
        print("Contact points/normals are stored in the NPZ for exact inspection.")
        while window.is_running():
            mujoco.mj_forward(model, data)
            window.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
