#!/usr/bin/env python3
"""FAILED DIAGNOSTIC viewer — this is explicitly not a valid grasp."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
ROOT=Path("/home/jbnu/aloha_g1_dataset");sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
import find_g1_dex3_static_phone_grasp as old
DEFAULT=ROOT/"converted_runs/g1_left_phone_grasp_diagnostic/best_failed_candidate_diagnostic.npz"
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def scene(path):
 p=load(path);pose=p["phone_proxy_pose"];rpy=Rotation.from_quat(pose[3:][[1,2,3,0]]).as_euler("xyz")
 m,_=old.expanded_phone_model(pose[:3],rpy);d=mujoco.MjData(m);d.qpos[:]=p["full_g1_qpos"];mujoco.mj_forward(m,d);return p,m,d
def markers(renderer,p):
 s=renderer.scene
 def arrow(a,v,color,L=.06):
  i=s.ngeom;mujoco.mjv_initGeom(s.geoms[i],mujoco.mjtGeom.mjGEOM_ARROW,np.array([.003]*3),np.zeros(3),np.eye(3).ravel(),np.array(color,np.float32))
  mujoco.mjv_connector(s.geoms[i],mujoco.mjtGeom.mjGEOM_ARROW,.004,a,a+L*v);s.ngeom+=1
 def sphere(a,color):
  i=s.ngeom;mujoco.mjv_initGeom(s.geoms[i],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.007,0,0]),a,np.eye(3).ravel(),np.array(color,np.float32));s.ngeom+=1
 for a,n,c in zip(p["contact_pad_points"],p["raw_surface_normals"],p["calculated_contact_normals"]):
  sphere(a,[1,0,0,1]);arrow(a,n,[0,1,0,1]);arrow(a,c,[1,1,0,1])
 for n in p["phone_face_normals"]:arrow(p["phone_proxy_pose"][:3],n,[0,0.5,1,1],.09)
 arrow(p["contact_pad_points"][0],p["contact_pad_points"][1]-p["contact_pad_points"][0],[1,0,1,1],1)
def render(path,out):
 p,m,d=scene(path)
 for name,az,el in (("best_failed_front.png",180,-5),("best_failed_top.png",180,-88),("best_failed_side.png",90,-5),("contact_normals_closeup.png",165,-15)):
  r=mujoco.Renderer(m,width=640,height=480);cam=mujoco.MjvCamera();cam.lookat[:]=[.27,.08,1.0];cam.distance=.75 if "closeup" in name else 1.25;cam.azimuth=az;cam.elevation=el
  r.update_scene(d,camera=cam);markers(r,p);im=Image.fromarray(r.render());r.close();dr=ImageDraw.Draw(im);dr.rectangle((8,8,632,54),fill="black");dr.text((16,16),"FAILED DIAGNOSTIC — NOT A VALID GRASP",fill=(255,60,60));dr.text((16,34),"green=raw surface normal, yellow=used normal, blue=phone faces",fill="white");im.save(out/name)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--diagnostic",type=Path,default=DEFAULT);ap.add_argument("--render",action="store_true");ap.add_argument("--output",type=Path,default=ROOT/"evaluation/g1_left_phone_grasp_diagnostic");a=ap.parse_args()
 print("\\n*** FAILED DIAGNOSTIC — NOT A VALID GRASP ***\\n")
 if a.render:a.output.mkdir(parents=True,exist_ok=True);render(a.diagnostic,a.output);return 0
 p,m,d=scene(a.diagnostic);from mujoco import viewer
 with viewer.launch_passive(m,d) as w:
  w.cam.lookat[:]=[.27,.08,1.0];w.cam.distance=1.2;w.cam.azimuth=180;w.cam.elevation=-8
  while w.is_running():mujoco.mj_forward(m,d);w.sync()
 return 0
if __name__=="__main__":raise SystemExit(main())
