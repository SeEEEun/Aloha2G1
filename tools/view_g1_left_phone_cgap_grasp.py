#!/usr/bin/env python3
"""View/render the hard-validated static left-hand C-gap phone grasp."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old
DEFAULT=ROOT/"converted_runs/g1_left_phone_cgap_grasp/selected_left_phone_cgap_grasp.npz"
EVAL=ROOT/"evaluation/g1_left_phone_cgap_grasp"

def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def state(payload):
 pose=payload["phone_proxy_pose"].astype(float)
 rpy=Rotation.from_quat(pose[3:][[1,2,3,0]]).as_euler("xyz")
 model,_=old.expanded_phone_model(pose[:3],rpy);data=mujoco.MjData(model)
 data.qpos[:]=payload["full_g1_qpos"];data.qvel[:]=0;mujoco.mj_forward(model,data)
 return model,data

def sphere(scene,p,color,r=.006):
 if scene.ngeom>=scene.maxgeom:return
 g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,
  np.array([r,0,0]),np.asarray(p),np.eye(3).ravel(),np.asarray(color,np.float32));scene.ngeom+=1

def arrow(scene,p,v,color,length=.05):
 if scene.ngeom>=scene.maxgeom:return
 g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,
  np.array([.003,.003,.003]),np.zeros(3),np.eye(3).ravel(),np.asarray(color,np.float32))
 mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.004,np.asarray(p),np.asarray(p)+length*np.asarray(v))
 scene.ngeom+=1

def debug(scene,model,data,p):
 for point,normal,color in zip(p["actual_contact_points"],p["actual_contact_normals"],
                               ([1,.15,.05,1],[1,.8,.05,1])):
  sphere(scene,point,color);arrow(scene,point,normal,color)
 center=p["phone_proxy_pose"][:3]
 for axis,color in zip(np.eye(3).T,([1,0,0,1],[0,1,0,1],[0,0,1,1])):arrow(scene,center,axis,color,.09)
 torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"torso_link")
 for axis,color in zip(data.xmat[torso].reshape(3,3).T,
                       ([1,0,0,1],[0,1,0,1],[0,0,1,1])):
  arrow(scene,data.xpos[torso],axis,color,.08)

def render(model,data,p,path,az,el,lookat,distance,title):
 renderer=mujoco.Renderer(model,width=640,height=480);cam=mujoco.MjvCamera()
 cam.lookat[:]=lookat;cam.distance=distance;cam.azimuth=az;cam.elevation=el
 renderer.update_scene(data,camera=cam);debug(renderer.scene,model,data,p)
 im=Image.fromarray(renderer.render());renderer.close();draw=ImageDraw.Draw(im)
 draw.rectangle((8,8,632,84),fill=(0,0,0))
 draw.text((18,16),f"G1_LEFT_PHONE_CGAP_GRASP_READY | {title} | qpos + mj_forward",fill="white")
 draw.text((18,39),f"aperture {float(p['aperture'])*1000:.6f} mm | "
  f"min forbidden {float(np.min(p['forbidden_clearances']))*1000:.3f} mm | "
  f"min arm/wrist margin {float(np.min(p['joint_limit_margins'])):.5f} rad",fill="white")
 draw.text((18,61),"contact points: orange/yellow; arrows: actual normals; axes: X red, Y green, Z blue",fill="white")
 path.parent.mkdir(parents=True,exist_ok=True);im.save(path)

def render_all(model,data,p,out):
 phone=p["phone_proxy_pose"][:3]
 views=(("left_phone_cgap_front.png",180,-5,[.20,0,1.0],1.25,"front"),
        ("left_phone_cgap_top.png",180,-88,[.20,0,1.0],1.25,"top"),
        ("left_phone_cgap_side.png",90,-5,[.20,0,1.0],1.25,"side"),
        ("left_phone_cgap_hand_closeup.png",145,-12,phone,.30,"hand closeup"),
        ("left_phone_cgap_contacts.png",165,-8,phone,.22,"contacts"))
 for name,az,el,look,dist,title in views:render(model,data,p,out/name,az,el,look,dist,title)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--grasp",type=Path,default=DEFAULT)
 ap.add_argument("--render",action="store_true");ap.add_argument("--output",type=Path,default=EVAL);a=ap.parse_args()
 if not a.grasp.exists():
  print("G1_LEFT_PHONE_CGAP_GRASP_BLOCKED");print("No validated NPZ; refusing to display a failed pose.");return 2
 p=load(a.grasp);model,data=state(p)
 if a.render:render_all(model,data,p,a.output);print(a.output);return 0
 from mujoco import viewer
 with viewer.launch_passive(model,data) as window:
  window.cam.lookat[:]=[.20,0,1.0];window.cam.distance=1.25;window.cam.azimuth=180;window.cam.elevation=-8
  debug(window.user_scn,model,data,p)
  print("G1_LEFT_PHONE_CGAP_GRASP_READY")
  print("Static qpos + mj_forward only; no mj_step, actuator control, Isaac Lab, or hardware.")
  while window.is_running():
   mujoco.mj_forward(model,data);window.sync();time.sleep(.02)
 return 0
if __name__=="__main__":raise SystemExit(main())
