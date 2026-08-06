#!/usr/bin/env python3
"""GUI/offscreen viewer for FAILED mirrored C-gap collision diagnostics."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old
TITLE="FAILED COLLISION DIAGNOSTIC — NOT A VALID GRASP"

def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def state(p):
 pose=p["phone_proxy_pose"].astype(float);rpy=Rotation.from_quat(pose[3:][[1,2,3,0]]).as_euler("xyz")
 model,_=old.expanded_phone_model(pose[:3],rpy);data=mujoco.MjData(model)
 data.qpos[:]=p["full_g1_qpos"];data.qvel[:]=0;mujoco.mj_forward(model,data);return model,data
def sphere(scene,p,color,r=.008):
 if scene.ngeom>=scene.maxgeom:return
 g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,
  np.array([r,0,0]),np.asarray(p),np.eye(3).ravel(),np.asarray(color,np.float32));scene.ngeom+=1
def arrow(scene,p,v,color,L=.055):
 if scene.ngeom>=scene.maxgeom:return
 g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,
  np.array([.003,.003,.003]),np.zeros(3),np.eye(3).ravel(),np.asarray(color,np.float32))
 mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.004,np.asarray(p),np.asarray(p)+L*np.asarray(v));scene.ngeom+=1
def debug(scene,p):
 for pt,n in zip(p["intended_contact_points"],p["actual_contact_normals"]):
  sphere(scene,pt,[1,.65,0,1],.006);arrow(scene,pt,n,[1,.8,0,1])
 for pt in p["forbidden_penetration_points"]:sphere(scene,pt,[1,0,1,1],.010)
 for pt in p["arm_torso_collision_points"]:sphere(scene,pt,[0,1,1,1],.013)
 center=p["phone_proxy_pose"][:3]
 for ax,col in zip(np.eye(3),([1,0,0,1],[0,1,0,1],[0,0,1,1])):arrow(scene,center,ax,col,.08)
def metrics(p):
 return (f"Stage {int(p['stage'])} | elbow {float(p['elbow_flexion_deg']):.2f} deg | "
  f"wrist {float(p['wrist_bend_deg']):.2f} deg | forbidden "
  f"{max(0,-float(p['minimum_forbidden_clearance_m']))*1000:.2f} mm\n"
  f"normal errors {np.asarray(p['normal_errors_deg']).round(2).tolist()} deg | "
  f"dot {float(p['normal_dot']):.5f}\n"
  f"magenta=forbidden phone/robot penetration; cyan=arm-torso; orange=intended contacts")
def render(model,data,p,path):
 ren=mujoco.Renderer(model,width=640,height=480);cam=mujoco.MjvCamera()
 cam.lookat[:]=[.20,.05,.85];cam.distance=.85;cam.azimuth=160;cam.elevation=-10
 ren.update_scene(data,camera=cam);debug(ren.scene,p);im=Image.fromarray(ren.render());ren.close()
 draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,96),fill=(120,0,0))
 draw.text((12,10),TITLE,fill="white");draw.multiline_text((12,34),metrics(p),fill="white",spacing=3)
 path.parent.mkdir(parents=True,exist_ok=True);im.save(path);print(path)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--diagnostic",type=Path,required=True)
 ap.add_argument("--render",type=Path,help="optional output PNG");a=ap.parse_args()
 p=load(a.diagnostic)
 if str(p["diagnostic_label"])!=TITLE:raise RuntimeError("NPZ is not a failed collision diagnostic")
 print("\n"+"!"*78);print(TITLE);print("!"*78);print(a.diagnostic);print(metrics(p))
 for x in p["collision_pairs"]:print("COLLISION:",x)
 model,data=state(p)
 if a.render:render(model,data,p,a.render);return 0
 from mujoco import viewer
 with viewer.launch_passive(model,data) as win:
  win.cam.lookat[:]=[.20,.05,.85];win.cam.distance=.85;win.cam.azimuth=160;win.cam.elevation=-10
  debug(win.user_scn,p)
  if hasattr(win,"add_overlay"):
   try:win.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT,TITLE,metrics(p))
   except Exception:pass
  while win.is_running():
   mujoco.mj_forward(model,data);win.sync();time.sleep(.02)
 return 0
if __name__=="__main__":raise SystemExit(main())
