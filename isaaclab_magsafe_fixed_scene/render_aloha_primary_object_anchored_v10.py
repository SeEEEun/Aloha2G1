#!/usr/bin/env python3
"""Actual Isaac Lab G1 render/GUI for v10 best or best-failure arm candidates."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher
ROOT=Path('/home/jbnu/aloha_g1_dataset');SC=ROOT/'isaaclab_magsafe_fixed_scene';OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_aloha_primary_object_anchored_v10';G1=Path('/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd')
p=argparse.ArgumentParser();p.add_argument('--trajectory',choices=('exact','nullspace'),default='nullspace');p.add_argument('--mode',choices=('fixed','object-follow'),default='fixed');p.add_argument('--cameras',nargs='+',choices=('overview','front','side','top'),default=['overview']);p.add_argument('--max-frames',type=int);p.add_argument('--gui',action='store_true');p.add_argument('--speed',type=float,default=.25);p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=540);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
NPZ=OUT/f'aloha_anchored_{args.trajectory}_arm_trajectory.npz';z=np.load(NPZ,allow_pickle=False);q=z['g1_arm_joint_trajectory'].astype(np.float32);names=z['arm_joint_names'].astype(str).tolist();targetL=z['corrected_left_position_scene'];targetR=z['corrected_right_position_scene'];achL=z['achieved_left_position_scene'];achR=z['achieved_right_position_scene'];candidate=str(z['selected_phasewarp_candidate']);z.close();base=np.load(OUT/'restored_base_aloha_targets.npz',allow_pickle=False);baseL=base['base_left_position_scene'];baseR=base['base_right_position_scene'];base.close();timeline=sorted(json.loads((ROOT/'configs/episode49_task_timeline.approved.json').read_text())['events'],key=lambda x:(x['frame'],x['event']));ik=json.loads((OUT/'ik_metrics.json').read_text());scene=SC/'generated/magsafe_fixed_scene.usda';object_frames=json.loads((OUT/'target_object_frames.json').read_text())
launcher=AppLauncher(args);app=launcher.app
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def event(frame):
 cur='pre_task'
 for e in timeline:
  if e['frame']<=frame:cur=e['event']
 return cur
def embed(raw,out,camera):
 meta=json.dumps({'trajectory_npz':str(NPZ.resolve()),'trajectory_sha256':sha(NPZ),'scene_usd':str(scene.resolve()),'scene_usd_sha256':sha(scene),'source_action_sha256':str(np.load(NPZ,allow_pickle=False)['optimized_action_sha256']),'root_forward_offset_m':.15,'source_frame_count':990,'encoded_frame_count':len(q) if args.max_frames is None else min(args.max_frames,len(q)),'fps':7.5,'mode':args.mode,'camera':camera,'dex3_applied':False,'physics':False},separators=(',',':'))
 tmp=out.with_name(out.stem+'.metadata.mp4');subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-map','0','-c','copy','-metadata',f'title=Isaac Lab v10 {args.trajectory} {args.mode} {camera}','-metadata',f'comment={meta}','-movflags','+faststart',str(tmp)],check=True);os.replace(tmp,out);raw.unlink()
def main():
 import cv2,torch,omni.usd
 from pxr import Gf,UsdGeom,UsdPhysics,UsdLux
 from isaaclab.assets import Articulation,ArticulationCfg
 from isaaclab.actuators import ImplicitActuatorCfg
 from isaaclab.sim import SimulationCfg,SimulationContext
 from isaaclab.sensors import Camera,CameraCfg
 import isaaclab.sim as sim_utils
 from robot_model_preview_common import CAMERAS,compose_stage
 print('[V10_RENDER] composing active scene',flush=True)
 stage_path=OUT/f'isaaclab_v10_{args.trajectory}_{args.mode}.usda';stage=compose_stage(stage_path,'G1',G1,'g1',forward_offset_m=.15)
 dome=UsdLux.DomeLight.Define(stage,'/World/V10RenderLights/Dome');dome.CreateIntensityAttr(850.0);dome.CreateColorAttr(Gf.Vec3f(1.0,.97,.93))
 key=UsdLux.DistantLight.Define(stage,'/World/V10RenderLights/Key');key.CreateIntensityAttr(2500.0);key.CreateAngleAttr(2.0);key.AddRotateXYZOp().Set(Gf.Vec3f(-45.0,25.0,15.0))
 for prim in stage.Traverse():
  if any(x in prim.GetName().lower() for x in ('phone','accessory','charger')):
   api=UsdPhysics.RigidBodyAPI.Get(stage,prim.GetPath())
   if api:api.GetKinematicEnabledAttr().Set(True);api.GetRigidBodyEnabledAttr().Set(False)
 markers={}
 for name,color in [('BaseL',(1.,.1,1.)),('BaseR',(.1,1.,1.)),('TargetL',(1.,.1,.1)),('TargetR',(.1,.3,1.)),('AchievedL',(1.,.8,.1)),('AchievedR',(.1,1.,.5))]:
  s=UsdGeom.Sphere.Define(stage,f'/World/Diagnostics/{name}');s.CreateRadiusAttr(.012 if name.startswith('Target') else .008);s.CreateDisplayColorAttr([color]);markers[name]=s.AddTranslateOp()
 error_lines={}
 for name,color in [('ErrorL',(1.,.15,.05)),('ErrorR',(.05,.25,1.))]:
  c=UsdGeom.BasisCurves.Define(stage,f'/World/Diagnostics/{name}');c.CreateTypeAttr('linear');c.CreateCurveVertexCountsAttr([2]);c.CreateWidthsAttr([.004]);c.CreateDisplayColorAttr([color]);error_lines[name]=c
 for object_name,key_name in [('Phone','T_target_scene_from_phone'),('Accessory','T_target_scene_from_accessory'),('ChargerPad','T_target_scene_from_charger_pad_face')]:
  T=np.asarray(object_frames[key_name],float);origin=T[:3,3]
  for axis,color in [(0,(1.,0.,0.)),(1,(0.,1.,0.)),(2,(0.,.3,1.))]:
   c=UsdGeom.BasisCurves.Define(stage,f'/World/Diagnostics/Frames/{object_name}_{axis}');c.CreateTypeAttr('linear');c.CreateCurveVertexCountsAttr([2]);c.CreateWidthsAttr([.003]);c.CreateDisplayColorAttr([color]);end=origin+.055*T[:3,axis];c.CreatePointsAttr([Gf.Vec3f(*map(float,origin)),Gf.Vec3f(*map(float,end))])
 stage.GetRootLayer().Save()
 if not omni.usd.get_context().open_stage(str(stage_path)):raise RuntimeError(stage_path)
 print('[V10_RENDER] stage opened',flush=True)
 sim=SimulationContext(SimulationCfg(device='cuda:0'))
 print('[V10_RENDER] simulation context ready',flush=True)
 robot=Articulation(ArticulationCfg(prim_path='/World/G1/Asset/root_joint',spawn=None,actuators={'arms':ImplicitActuatorCfg(joint_names_expr=[r'(left|right)_(shoulder|wrist)_.*_joint',r'(left|right)_elbow_joint'],effort_limit_sim=25.,velocity_limit_sim=12.,stiffness=100.,damping=5.)}))
 print('[V10_RENDER] articulation configured',flush=True)
 cameras={}
 if not args.gui:
  for n in args.cameras:cameras[n]=Camera(CameraCfg(prim_path=f'/World/V10Camera_{n}',update_period=0,height=args.height,width=args.width,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,clipping_range=(.05,20.))))
 print('[V10_RENDER] camera sensors configured',flush=True)
 sim.reset();print('[V10_RENDER] simulation reset',flush=True);runtime=list(robot.data.joint_names);missing=[n for n in names if n not in runtime]
 if missing:raise RuntimeError(f'missing joints {missing}')
 ids=[runtime.index(n) for n in names];pos=robot.data.default_joint_pos.torch.clone().to(robot.device,dtype=torch.float32);vel=torch.zeros_like(pos);dt=sim.get_physics_dt()
 for n,c in cameras.items():eye,target=CAMERAS[n];c.set_world_poses_from_view(np.asarray([eye],np.float32),np.asarray([target],np.float32))
 total=len(q) if args.max_frames is None else min(args.max_frames,len(q));writers={};paths={}
 for n in cameras:
  if args.mode=='fixed':fn=f'isaaclab_fixed_objects_{args.trajectory}_{n}.mp4'
  else:fn=f'isaaclab_object_follow_{args.trajectory}_{n}.mp4'
  out=OUT/fn;raw=OUT/f'.{out.stem}.raw.mp4';writers[n]=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),7.5,(args.width,args.height));paths[n]=(raw,out)
  if not writers[n].isOpened():raise RuntimeError(raw)
 maxerr=0.;seen=[]
 if args.gui:
  eye,target=CAMERAS[args.cameras[0]];sim.set_camera_view(eye,target);start=time.monotonic()
  while app.is_running():
   f=min(int((time.monotonic()-start)*30*args.speed),len(q)-1);pos[0,ids]=torch.as_tensor(q[f],device=robot.device);robot.write_joint_state_to_sim(pos,vel);sim.forward();sim.render()
   if f==len(q)-1:start=time.monotonic()
  return
 for f in range(total):
  pos[0,ids]=torch.as_tensor(q[f],device=robot.device);robot.write_joint_state_to_sim(pos,vel)
  for k,v in [('BaseL',baseL[f]),('BaseR',baseR[f]),('TargetL',targetL[f]),('TargetR',targetR[f]),('AchievedL',achL[f]),('AchievedR',achR[f])]:markers[k].Set(Gf.Vec3d(*map(float,v)))
  error_lines['ErrorL'].GetPointsAttr().Set([Gf.Vec3f(*map(float,targetL[f])),Gf.Vec3f(*map(float,achL[f]))]);error_lines['ErrorR'].GetPointsAttr().Set([Gf.Vec3f(*map(float,targetR[f])),Gf.Vec3f(*map(float,achR[f]))])
  # Object following is fail-closed because this best-failure candidate misses the <=5 mm attach gate.
  sim.forward();sim.render();[c.update(dt) for c in cameras.values()];actual=robot.data.joint_pos.torch[0,ids].detach().cpu().numpy();maxerr=max(maxerr,float(np.max(np.abs(actual-q[f]))));seen.append(f)
  ae=max(np.linalg.norm(achL[f]-targetL[f]),np.linalg.norm(achR[f]-targetR[f]))*1000
  for n,c in cameras.items():
   im=c.data.output['rgb'][0].detach().cpu().numpy()[...,:3];im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR);cv2.rectangle(im,(0,0),(args.width,130),(10,10,10),-1)
   lines=[f'FAILED DIAGNOSTIC CANDIDATE | NOT APPROVED | {args.trajectory.upper()}',f'frame {f:03d}/989 | {event(f)} | max target error {ae:.1f} mm',f'{candidate} | DEX3 NOT YET APPLIED | KINEMATIC ONLY','markers: base magenta/cyan | corrected red/blue | achieved yellow/green | error lines']
   if args.mode=='object-follow':lines.append('OBJECT FOLLOW BLOCKED: PALM ANCHOR GATE > 5 mm | NOT PHYSICS GRASP')
   for i,s in enumerate(lines):cv2.putText(im,s,(14,24+25*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(40,220,255) if i else (60,80,255),1,cv2.LINE_AA)
   writers[n].write(im)
 [w.release() for w in writers.values()]
 for n,(raw,out) in paths.items():embed(raw,out,n)
 report={'status':'BLOCKED_IK_BEST_FAILURE_RENDERED','trajectory':args.trajectory,'mode':args.mode,'frames':total,'runtime_joint_mapping':'NAME_BASED','missing_joints':missing,'max_mapped_joint_error_rad':maxerr,'physics_steps':0,'physics':False,'objects_followed':False,'object_follow_blocker':'anchor gate >5 mm' if args.mode=='object-follow' else None,'hardware_commands_sent':False,'dds_initialized':False,'videos':{n:str(x[1].resolve()) for n,x in paths.items()}}
 dump_path=OUT/f'isaaclab_{args.trajectory}_{args.mode}_headless.json';dump_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 try:
  main()
 except BaseException as exc:
  failure={'status':'BLOCKED_ISAACLAB_REPLAY','exception_type':type(exc).__name__,'exception':str(exc),'traceback':traceback.format_exc()}
  (OUT/f'isaaclab_{args.trajectory}_{args.mode}_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
  print('[V10_RENDER_FAILURE] '+json.dumps(failure),flush=True)
  raise
 finally:app.close()
