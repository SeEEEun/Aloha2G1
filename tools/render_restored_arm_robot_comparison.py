#!/usr/bin/env python3
"""Render real active ALOHA/G1 models for the restored arm-only comparison.

Kinematic FK only: no stepping, contacts, Dex3 IK, DDS, publisher or hardware client.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess
import cv2, mujoco, numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import retarget_episode49_optimized_action_to_g1 as core
import validate_smolvla_in_stationary_aloha_mujoco as aloha
import validate_g1_targets_and_sparse_ik as ik

ROOT=Path('/home/jbnu/aloha_g1_dataset'); OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_restore_original_v8'
EXACT=OUT/'restored_exact_arm_trajectory.npz'; NULL=OUT/'restored_nullspace_arm_trajectory.npz'; TIMELINE=ROOT/'configs/episode49_task_timeline.approved.json'
VIEWS={'overview':(135,-18,1.75,[.43,.02,1.02]),'front':(90,-6,1.55,[.43,.02,1.02]),'side':(0,-7,1.55,[.43,.02,1.02])}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def embed_metadata(raw, output, camera_name, exact_hash, null_hash):
 """Finish an MP4 atomically and put the provenance in the MP4 container itself."""
 provenance=json.dumps({
  'status':'ARM_ROBOT_RENDER_READY_FOR_VISUAL_APPROVAL',
  'camera':camera_name,
  'exact_npz':str(EXACT.resolve()),'exact_sha256':exact_hash,
  'nullspace_npz':str(NULL.resolve()),'nullspace_sha256':null_hash,
  'simulation_only':True,'physics':False,'dex3_contact_ik':False,
 },separators=(',',':'))
 finished=output.with_name(output.stem+'.metadata.mp4')
 subprocess.run([
  'ffmpeg','-y','-loglevel','error','-i',str(raw),'-map','0','-c','copy',
  '-metadata',f'title=ALOHA source | RESTORED_EXACT G1 | RESTORED_NULLSPACE G1 ({camera_name})',
  '-metadata',f'comment={provenance}','-movflags','+faststart',str(finished)
 ],check=True)
 os.replace(finished,output)
 raw.unlink()
def decoded_frames(path):
 cap=cv2.VideoCapture(str(path)); count=0
 if not cap.isOpened(): raise RuntimeError(f'cannot decode video: {path}')
 while True:
  ok,_=cap.read()
  if not ok: break
  count+=1
 cap.release();return count
def camera(spec):
 c=mujoco.MjvCamera();c.type=mujoco.mjtCamera.mjCAMERA_FREE;c.azimuth,c.elevation,c.distance=spec[:3];c.lookat[:]=spec[3];return c
def add_geom(scene,typ,size,pos,rgba,mat=None):
 if scene.ngeom>=scene.maxgeom:return
 g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,typ,np.asarray(size,float),np.asarray(pos,float),np.eye(3).ravel() if mat is None else np.asarray(mat,float).ravel(),np.asarray(rgba,float));scene.ngeom+=1
def add_scene(renderer):
 s=renderer.scene
 add_geom(s,mujoco.mjtGeom.mjGEOM_BOX,[.55,.43,.015],[.4175,.20,.780],[.32,.34,.36,1])
 add_geom(s,mujoco.mjtGeom.mjGEOM_BOX,[.0748,.003975,.03575],[.525,.07,.83075],[.08,.10,.13,1])
 add_geom(s,mujoco.mjtGeom.mjGEOM_CYLINDER,[.0525,.012,0],[.42,.21,.819],[.9,.9,.92,1])
 add_geom(s,mujoco.mjtGeom.mjGEOM_CYLINDER,[.0295,.004,0],[.42,.22,.938],[.95,.95,.97,1])
 # MagSafe accessory: annulus visualized as capsule segments centered on the phone back.
 cen=np.array([.525,.076425,.83075]);rad=.025
 for k in range(16):
  a=2*np.pi*k/16;b=2*np.pi*(k+1)/16;p1=cen+np.array([rad*np.cos(a),0,rad*np.sin(a)]);p2=cen+np.array([rad*np.cos(b),0,rad*np.sin(b)])
  if s.ngeom>=s.maxgeom:break
  g=s.geoms[s.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.array([.0018,0,0]),np.zeros(3),np.eye(3).ravel(),np.array([.03,.03,.03,1.]));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.0018,p1,p2);s.ngeom+=1
def overlay(im,title,frame,event,hash8):
 cv2.rectangle(im,(0,0),(im.shape[1],58),(15,15,15),-1);cv2.putText(im,title,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA);cv2.putText(im,f'frame {frame:03d} | {event} | {hash8}',(10,43),cv2.FONT_HERSHEY_SIMPLEX,.43,(80,220,255),1,cv2.LINE_AA);return im
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--camera',choices=VIEWS,default='overview');ap.add_argument('--fps',type=int,default=10);ap.add_argument('--stride',type=int,default=3);a=ap.parse_args()
 ez=np.load(EXACT,allow_pickle=False);nz=np.load(NULL,allow_pickle=False);action=ez['optimized_action'];eq=ez['g1_arm_joint_trajectory'];nq=nz['g1_arm_joint_trajectory']
 am,_=aloha.load_validated_model(core.ALOHA_XML);aq,_=aloha.mapped_qpos(action);ad=mujoco.MjData(am)
 info=ik.validate_model(core.G1_XML);gm=info['model'];ed=mujoco.MjData(gm);nd=mujoco.MjData(gm)
 root=np.asarray(ez['g1_root_position']);R=np.asarray(json.load(open(ROOT/'configs/magsafe_task_frame_registration.sim.json'))['T_scene_from_g1_base'])[:3,:3];quat=ik.mat_to_quat_wxyz(R)
 ar=mujoco.Renderer(am,360,480);er=mujoco.Renderer(gm,360,480);nr=mujoco.Renderer(gm,360,480);cam=camera(VIEWS[a.camera]);acam=camera((135,-18,1.65,[0,0,.85]))
 events=sorted((int(x['frame']),x['event']) for x in json.load(open(TIMELINE))['events']); current='pre_task';eh=sha(EXACT);nh=sha(NULL)
 out=OUT/f'aloha_exact_nullspace_robot_{a.camera}.mp4';raw=OUT/f'.{out.stem}.raw.mp4'
 exact_out=OUT/f'restored_exact_robot_{a.camera}.mp4';exact_raw=OUT/f'.{exact_out.stem}.raw.mp4'
 null_out=OUT/f'restored_nullspace_robot_{a.camera}.mp4';null_raw=OUT/f'.{null_out.stem}.raw.mp4'
 writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(1440,360))
 exact_writer=cv2.VideoWriter(str(exact_raw),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(480,360))
 null_writer=cv2.VideoWriter(str(null_raw),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(480,360))
 if not all(w.isOpened() for w in (writer,exact_writer,null_writer)): raise RuntimeError('failed to open MP4 writers')
 for f in range(0,len(eq),a.stride):
  current=next((name for frame,name in reversed(events) if frame<=f),'pre_task')
  ad.qpos[:]=aq[f];ad.qvel[:]=0;mujoco.mj_forward(am,ad);ar.update_scene(ad,acam);ai=np.clip(ar.render().astype(np.float32)*1.8+18,0,255).astype(np.uint8);ai=overlay(ai,'ALOHA SOURCE ROBOT',f,current,'source')
  for data,q,ren,title,h in [(ed,eq[f],er,'RESTORED_EXACT G1',eh),(nd,nq[f],nr,'RESTORED_NULLSPACE G1',nh)]:
   ik.assign_arm_qpos(data,info['stand_qpos'],info['arm_qpos_ids'],q);data.qpos[:3]=root;data.qpos[3:7]=quat;data.qvel[:]=0;mujoco.mj_forward(gm,data);ren.update_scene(data,cam);add_scene(ren);im=overlay(ren.render(),title,f,current,h[:12]);
   if title.startswith('RESTORED_EXACT'):ei=im
   else:ni=im
  writer.write(cv2.cvtColor(np.hstack((ai,ei,ni)),cv2.COLOR_RGB2BGR));exact_writer.write(cv2.cvtColor(ei,cv2.COLOR_RGB2BGR));null_writer.write(cv2.cvtColor(ni,cv2.COLOR_RGB2BGR))
 writer.release();exact_writer.release();null_writer.release();ar.close();er.close();nr.close()
 embed_metadata(raw,out,a.camera,eh,nh);embed_metadata(exact_raw,exact_out,a.camera,eh,nh);embed_metadata(null_raw,null_out,a.camera,eh,nh)
 expected=len(range(0,len(eq),a.stride));counts={str(p):decoded_frames(p) for p in (out,exact_out,null_out)}
 if any(v!=expected for v in counts.values()): raise RuntimeError(f'decode-frame validation failed: {counts}, expected {expected}')
 exact_video_hash=sha(exact_out);null_video_hash=sha(null_out)
 if exact_video_hash==null_video_hash: raise RuntimeError('FAIL: Exact and Nullspace rendered-video SHA-256 are identical')
 meta={'status':'ARM_ROBOT_RENDER_READY_FOR_VISUAL_APPROVAL','camera':a.camera,'output':str(out.resolve()),'exact_video':str(exact_out.resolve()),'nullspace_video':str(null_out.resolve()),'aloha_source_action_container':str(EXACT.resolve()),'exact_npz':str(EXACT.resolve()),'nullspace_npz':str(NULL.resolve()),'exact_sha256':eh,'nullspace_sha256':nh,'video_sha256':sha(out),'exact_video_sha256':exact_video_hash,'nullspace_video_sha256':null_video_hash,'video_hashes_differ':True,'decoded_frames':counts,'active_aloha_model':str(core.ALOHA_XML),'active_g1_model':str(core.G1_XML),'frames_source':len(eq),'rendered_frames':expected,'dex3_contact_ik':False,'physics':False,'hardware':False,'simulation_only':True};(OUT/f'robot_render_{a.camera}_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
