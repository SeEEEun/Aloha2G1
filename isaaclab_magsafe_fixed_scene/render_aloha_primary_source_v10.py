#!/usr/bin/env python3
"""Render all 990 SmolVLA ALOHA source frames in the user-approved source scene."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,traceback
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

ROOT=Path('/home/jbnu/aloha_g1_dataset')
SC=ROOT/'isaaclab_magsafe_fixed_scene'
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_aloha_primary_object_anchored_v10'
SOURCE_SCENE=ROOT/'outputs/episode49_source_scene/generated/source_magsafe_fixed_scene.usda'
ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
ALOHA=Path('/home/jbnu/robot_assets/stationary_aloha/usd_imported/stationary_aloha_imported.usd')
JOINT_NAMES=[*(f'follower_left_joint_{i}' for i in range(6)),'follower_left_left_carriage_joint','follower_left_right_carriage_joint',*(f'follower_right_joint_{i}' for i in range(6)),'follower_right_left_carriage_joint','follower_right_right_carriage_joint']
p=argparse.ArgumentParser();p.add_argument('--max-frames',type=int);p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=540);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
launcher=AppLauncher(args);app=launcher.app

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def fixed_root(stage):
 from pxr import Gf,PhysxSchema,Sdf,UsdGeom,UsdPhysics
 root=Sdf.Path('/World/StationaryALOHA/Asset/Geometry/tabletop_link');jp=root.AppendChild('V10WorldFixedJoint');world=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(root));tr=Gf.Transform(world);j=UsdPhysics.FixedJoint.Define(stage,jp);j.CreateBody1Rel().SetTargets([root]);j.CreateLocalPos0Attr().Set(tr.GetTranslation());q=tr.GetRotation().GetQuat();j.CreateLocalRot0Attr().Set(Gf.Quatf(float(q.GetReal()),Gf.Vec3f(q.GetImaginary())));j.CreateLocalPos1Attr().Set(Gf.Vec3f(0));j.CreateLocalRot1Attr().Set(Gf.Quatf(1));j.CreateCollisionEnabledAttr().Set(False);rootp=stage.GetPrimAtPath(root);rootp.RemoveAPI(UsdPhysics.ArticulationRootAPI);UsdPhysics.ArticulationRootAPI.Apply(j.GetPrim());PhysxSchema.PhysxArticulationAPI.Apply(j.GetPrim());return str(jp)
def main():
 import cv2,torch,omni.usd
 from pxr import Gf,Usd,UsdGeom,UsdLux,UsdPhysics
 from isaaclab.assets import Articulation,ArticulationCfg
 from isaaclab.actuators import ImplicitActuatorCfg
 from isaaclab.sim import SimulationCfg,SimulationContext
 from isaaclab.sensors import Camera,CameraCfg
 import isaaclab.sim as sim_utils
 from robot_model_preview_common import suppress_stationary_aloha_fixture
 timeline=sorted(json.loads((ROOT/'configs/episode49_task_timeline.approved.json').read_text())['events'],key=lambda x:(x['frame'],x['event']))
 zz=np.load(ACTION,allow_pickle=False);q=zz['optimized_action'].astype(np.float32);timestamps=zz['timestamp'].copy();zz.close()
 stage_path=OUT/'isaaclab_v10_source_aloha.usda';stage=Usd.Stage.CreateNew(str(stage_path));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);world=UsdGeom.Xform.Define(stage,'/World').GetPrim();stage.SetDefaultPrim(world)
 scene=UsdGeom.Xform.Define(stage,'/World/SourceScene').GetPrim();scene.GetReferences().AddReference(str(SOURCE_SCENE))
 robot=UsdGeom.Xform.Define(stage,'/World/StationaryALOHA');asset=UsdGeom.Xform.Define(stage,'/World/StationaryALOHA/Asset').GetPrim();asset.GetReferences().AddReference(str(ALOHA));robot.AddTranslateOp().Set(Gf.Vec3d(.4175,.36,.756));robot.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(.7071067812,Gf.Vec3d(0,0,.7071067812)))
 suppress_stationary_aloha_fixture(stage);dome=UsdLux.DomeLight.Define(stage,'/World/V10SourceLights/Dome');dome.CreateIntensityAttr(850.0);key=UsdLux.DistantLight.Define(stage,'/World/V10SourceLights/Key');key.CreateIntensityAttr(2500.0);key.AddRotateXYZOp().Set(Gf.Vec3f(-45,25,15))
 for prim in stage.Traverse():
  if any(x in prim.GetName().lower() for x in ('phone','accessory','charger')):
   api=UsdPhysics.RigidBodyAPI.Get(stage,prim.GetPath())
   if api:api.GetKinematicEnabledAttr().Set(True);api.GetRigidBodyEnabledAttr().Set(False)
 joint=fixed_root(stage);stage.GetRootLayer().Save()
 if not omni.usd.get_context().open_stage(str(stage_path)):raise RuntimeError(stage_path)
 sim=SimulationContext(SimulationCfg(device='cuda:0'));art=Articulation(ArticulationCfg(prim_path=joint,spawn=None,actuators={'all':ImplicitActuatorCfg(joint_names_expr=['follower_.*'],effort_limit_sim=60.,velocity_limit_sim=8.,stiffness=100.,damping=10.)}));cam=Camera(CameraCfg(prim_path='/World/V10SourceCamera',update_period=0,height=args.height,width=args.width,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,clipping_range=(.05,20.))))
 sim.reset();runtime=list(art.data.joint_names);missing=[n for n in JOINT_NAMES if n not in runtime]
 if missing:raise RuntimeError(f'missing={missing}')
 ids={n:runtime.index(n) for n in JOINT_NAMES};pos=art.data.default_joint_pos.torch.clone().to(art.device,dtype=torch.float32);vel=torch.zeros_like(pos);cam.set_world_poses_from_view(np.asarray([[1.50,-1.45,1.45]],np.float32),np.asarray([[.43,.31,.80]],np.float32))
 total=len(q) if args.max_frames is None else min(args.max_frames,len(q));raw=OUT/'.isaaclab_source_aloha_overview.raw.mp4';out=OUT/'isaaclab_source_aloha_overview.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),7.5,(args.width,args.height));maxerr=0.
 for f in range(total):
  x=q[f]
  for side,off in [('left',0),('right',7)]:
   for j in range(6):pos[0,ids[f'follower_{side}_joint_{j}']]=float(x[off+j])
   g=float(np.clip(x[off+6],0,.044));pos[0,ids[f'follower_{side}_left_carriage_joint']]=g;pos[0,ids[f'follower_{side}_right_carriage_joint']]=g
  art.write_joint_state_to_sim(pos,vel);sim.forward();sim.render();cam.update(sim.get_physics_dt());actual=art.data.joint_pos.torch[0].detach().cpu().numpy();maxerr=max(maxerr,max(abs(actual[ids[n]]-float(pos[0,ids[n]])) for n in JOINT_NAMES));im=cam.data.output['rgb'][0].detach().cpu().numpy()[...,:3];im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR);ev='pre_task'
  for e in timeline:
   if e['frame']<=f:ev=e['event']
  cv2.rectangle(im,(0,0),(args.width,82),(10,10,10),-1);lines=[f'ALOHA SOURCE MODEL | frame {f:03d}/989 | {ev}','SMOLVLA optimized_action | USER-APPROVED SOURCE SCENE | RELATION RECOVERY ONLY','SOURCE WORLD COORDINATES NOT USED AS G1 TARGET | KINEMATIC ONLY']
  for i,s in enumerate(lines):cv2.putText(im,s,(14,23+25*i),cv2.FONT_HERSHEY_SIMPLEX,.52,(40,220,255),1,cv2.LINE_AA)
  writer.write(im)
 writer.release();meta=json.dumps({'source_action_npz':str(ACTION),'source_action_sha256':sha(ACTION),'source_scene_usd':str(SOURCE_SCENE),'source_scene_usd_sha256':sha(SOURCE_SCENE),'source_frame_count':990,'encoded_frame_count':total,'fps':7.5,'physics':False,'camera_registration_used':False},separators=(',',':'));tmp=OUT/'.isaaclab_source_aloha_overview.metadata.mp4';subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-map','0','-c','copy','-metadata','title=Isaac Lab ALOHA primary source','-metadata',f'comment={meta}','-movflags','+faststart',str(tmp)],check=True);os.replace(tmp,out);raw.unlink();report={'status':'SOURCE_ALOHA_RENDERED','frames':total,'timestamps_shape':list(timestamps.shape),'runtime_joint_mapping':'NAME_BASED','missing_joints':missing,'max_mapped_joint_error_rad':maxerr,'physics_steps':0,'video':str(out.resolve())};(OUT/'isaaclab_source_aloha_headless.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  failure={'status':'BLOCKED_ISAACLAB_SOURCE_RENDER','exception':str(exc),'traceback':traceback.format_exc()};(OUT/'isaaclab_source_aloha_failure.json').write_text(json.dumps(failure,indent=2)+'\n');print(json.dumps(failure));raise
 finally:app.close()
