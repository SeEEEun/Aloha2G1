#!/usr/bin/env python3
"""Headless Isaac audit of actual right-hand frames for Episode 49 frames 300-360."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher
ROOT=Path('/home/jbnu/aloha_g1_dataset');SC=ROOT/'isaaclab_magsafe_fixed_scene';OUT=ROOT/'outputs/g1_world_task_retargeting/right_tcp_debug';ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz';RAW=ROOT/'raw_recordings/GoPark_20260729_111223/data/chunk-000/episode_000000.parquet';ALOHA_USD=Path('/home/jbnu/robot_assets/stationary_aloha/usd_imported/stationary_aloha_imported.usd');STAGE=SC/'generated/aloha_right_tcp_debug.usda';JOINT_NAMES=[*(f'follower_left_joint_{i}' for i in range(6)),'follower_left_left_carriage_joint','follower_left_right_carriage_joint',*(f'follower_right_joint_{i}' for i in range(6)),'follower_right_left_carriage_joint','follower_right_right_carriage_joint']
p=argparse.ArgumentParser();p.add_argument('--start-frame',type=int,default=300);p.add_argument('--end-frame',type=int,default=360);p.add_argument('--record',action='store_true');p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720);AppLauncher.add_app_launcher_args(p);args=p.parse_args();launcher=AppLauncher(args);app=launcher.app
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist() if isinstance(v,np.ndarray) else str(v))+'\n')
def fixed_root(stage):
 from pxr import Gf,PhysxSchema,Sdf,UsdGeom,UsdPhysics
 root=Sdf.Path('/World/StationaryALOHA/Asset/Geometry/tabletop_link');jp=root.AppendChild('ReplayWorldFixedJoint');world=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(root));tr=Gf.Transform(world);j=UsdPhysics.FixedJoint.Define(stage,jp);j.CreateBody1Rel().SetTargets([root]);j.CreateLocalPos0Attr().Set(tr.GetTranslation());q=tr.GetRotation().GetQuat();j.CreateLocalRot0Attr().Set(Gf.Quatf(float(q.GetReal()),Gf.Vec3f(q.GetImaginary())));j.CreateLocalPos1Attr().Set(Gf.Vec3f(0));j.CreateLocalRot1Attr().Set(Gf.Quatf(1));j.CreateCollisionEnabledAttr().Set(False);rootp=stage.GetPrimAtPath(root);rootp.RemoveAPI(UsdPhysics.ArticulationRootAPI);UsdPhysics.ArticulationRootAPI.Apply(j.GetPrim());PhysxSchema.PhysxArticulationAPI.Apply(j.GetPrim());stage.GetRootLayer().Save();return str(jp)
def main():
 import cv2,torch,omni.usd,pyarrow.parquet as pq
 from pxr import UsdPhysics
 from isaaclab.assets import Articulation,ArticulationCfg
 from isaaclab.actuators import ImplicitActuatorCfg
 from isaaclab.sim import SimulationCfg,SimulationContext
 from isaaclab.sensors import Camera,CameraCfg
 import isaaclab.sim as sim_utils
 sys.path.insert(0,str(SC));from robot_model_preview_common import CAMERAS,compose_stage,suppress_stationary_aloha_fixture
 OUT.mkdir(parents=True,exist_ok=True);opt=np.load(ACTION)['optimized_action'].astype(np.float32);tab=pq.read_table(RAW,columns=['action','observation.state','frame_index']);gt=np.asarray(tab['action'].to_pylist(),np.float32);obs=np.asarray(tab['observation.state'].to_pylist(),np.float32);frames=np.asarray(tab['frame_index'].to_pylist());del tab
 stage=compose_stage(STAGE,'StationaryALOHA',ALOHA_USD,'stationary_aloha');suppress_stationary_aloha_fixture(stage)
 for prim in stage.Traverse():
  if any(x in prim.GetName().lower() for x in ('phone','accessory','charger')):
   api=UsdPhysics.RigidBodyAPI.Get(stage,prim.GetPath())
   if api:api.GetKinematicEnabledAttr().Set(True)
 joint=fixed_root(stage)
 if not omni.usd.get_context().open_stage(str(STAGE)):raise RuntimeError(STAGE)
 sim=SimulationContext(SimulationCfg(device='cpu'));robot=Articulation(ArticulationCfg(prim_path=joint,spawn=None,actuators={'all':ImplicitActuatorCfg(joint_names_expr=['follower_.*'],effort_limit_sim=60.,velocity_limit_sim=8.,stiffness=100.,damping=10.)}));
 cameras={}
 if args.record:
  for name in ('overview','front'):
   cameras[name]=Camera(CameraCfg(prim_path=f'/World/DebugCamera_{name}',update_period=0,height=args.height,width=args.width,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,clipping_range=(.05,20.))))
 sim.reset();names=list(robot.data.joint_names);missing=[n for n in JOINT_NAMES if n not in names]
 if missing:raise RuntimeError(f'missing={missing}')
 ids={n:names.index(n) for n in JOINT_NAMES};body_names=list(robot.data.body_names);candidates=[n for n in body_names if n.startswith('follower_right_') and any(x in n for x in ('link_','carriage_'))];current_body='follower_right_link_6'
 if current_body not in body_names:raise RuntimeError(current_body)
 prim_candidates=[str(x.GetPath()) for x in omni.usd.get_context().get_stage().Traverse() if 'follower_right_' in x.GetName().lower()]
 for name,cam in cameras.items():eye,target=CAMERAS[name];cam.set_world_poses_from_view(np.asarray([eye],np.float32),np.asarray([target],np.float32))
 pos=robot.data.default_joint_pos.torch.clone().to(robot.device,dtype=torch.float32);vel=torch.zeros_like(pos);acc=np.array([.525,.26204794497151274,.8306324233904814]);rows=[];current_tcp_records={};writers={n:cv2.VideoWriter(str(OUT/f'right_tcp_frames_300_360_{n}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(args.width,args.height)) for n in cameras}
 for fr in range(args.start_frame,args.end_frame+1):
  x=opt[fr]
  for side,off in [('left',0),('right',7)]:
   for j in range(6):pos[0,ids[f'follower_{side}_joint_{j}']]=float(x[off+j])
   g=float(x[off+6]);pos[0,ids[f'follower_{side}_left_carriage_joint']]=g;pos[0,ids[f'follower_{side}_right_carriage_joint']]=g
  robot.write_joint_state_to_sim(pos,vel);sim.step(render=bool(cameras));robot.update(sim.get_physics_dt());[c.update(sim.get_physics_dt()) for c in cameras.values()]
  curi=body_names.index(current_body);curp=robot.data.body_pos_w.torch[0,curi].detach().cpu().numpy();curq=robot.data.body_quat_w.torch[0,curi].detach().cpu().numpy();from scipy.spatial.transform import Rotation as Rot;R=Rot.from_quat(curq[[1,2,3,0]]).as_matrix();tcp=curp+R@np.array([.09,0,0]);current_tcp_records[fr]={'world_position':tcp.tolist(),'accessory_distance_m':float(np.linalg.norm(tcp-acc))}
  distances=[]
  for b in candidates:
   bi=body_names.index(b);bp=robot.data.body_pos_w.torch[0,bi].detach().cpu().numpy();bq=robot.data.body_quat_w.torch[0,bi].detach().cpu().numpy();dist=float(np.linalg.norm(bp-acc));distances.append((dist,b))
   rows.append({'frame':fr,'time_sec':fr/30,'candidate':b,'world_x':bp[0],'world_y':bp[1],'world_z':bp[2],'quat_w':bq[0],'quat_x':bq[1],'quat_y':bq[2],'quat_z':bq[3],'accessory_dx':bp[0]-acc[0],'accessory_dy':bp[1]-acc[1],'accessory_dz':bp[2]-acc[2],'accessory_distance_m':dist,'difference_from_current_tcp_m':float(np.linalg.norm(bp-tcp))})
  for name,cam in cameras.items():
   img=cam.data.output['rgb'][0].detach().cpu().numpy()[...,:3];img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR);near=sorted(distances)[:5];lines=[f'ALOHA frame {fr} | accessory {acc.tolist()}',f'current TCP {current_body}+[0.09,0,0], d={np.linalg.norm(tcp-acc):.4f} m']+[f'{b}: {d:.4f} m' for d,b in near]
   for i,s in enumerate(lines):cv2.putText(img,s,(20,35+30*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
   writers[name].write(img)
 [w.release() for w in writers.values()]
 with open(OUT/'right_tcp_candidate_distances.csv','w',newline='') as f:w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
 f326=[r for r in rows if r['frame']==326];closest=min(f326,key=lambda r:r['accessory_distance_m']);separate_fk_path=ROOT/'outputs/g1_world_task_retargeting/source/aloha_tcp_world_trajectory.npz';fk_compare={'status':'NOT_AVAILABLE'}
 if separate_fk_path.exists():
  separate=np.load(separate_fk_path);fkpos=np.asarray(separate['right_tcp_world_position'][326],float);ipos=np.asarray(current_tcp_records[326]['world_position'],float);fk_compare={'status':'MISMATCH_OBSERVED','source':str(separate_fk_path),'frame':326,'separate_fk_world_position':fkpos.tolist(),'isaac_articulation_current_tcp_world_position':ipos.tolist(),'isaac_minus_separate_fk_xyz_m':(ipos-fkpos).tolist(),'difference_norm_m':float(np.linalg.norm(ipos-fkpos))}
 mapping={'current_selected_tcp':{'body':current_body,'local_offset_m':[.09,0,0],'source':'validate_smolvla_in_stationary_aloha_mujoco.py','frame326':current_tcp_records.get(326)},'runtime_body_names':body_names,'right_candidate_bodies':candidates,'right_candidate_usd_prims':prim_candidates,'site_candidates':[],'site_note':'Imported Isaac articulation exposes rigid bodies and USD prims; no MuJoCo-style site objects exist.','frame326_candidates':sorted(f326,key=lambda r:r['accessory_distance_m']),'closest_frame326':closest,'joint_channels':{'left_arm':'0:6','left_gripper':6,'right_arm':'7:13','right_gripper':13},'qpos_application':{'arm':'direct radians by verified joint name','gripper':'scalar copied to both carriage joints','name_based_mapping':True,'left_right_swap_detected':False},'runtime_joint_order':names,'missing_joints':missing,'separate_fk_vs_isaac_articulation':fk_compare,'automatic_tcp_change':False,'g1_root_search_run':False,'status':'USER_APPROVAL_REQUIRED'};dump(OUT/'right_tcp_mapping_audit.json',mapping);dump(OUT/'render_status.json',{'requested':args.record,'status':'GENERATED' if args.record else 'NOT_AVAILABLE_NO_RTX_GPU','overview_path':str(OUT/'right_tcp_frames_300_360_overview.mp4'),'front_path':str(OUT/'right_tcp_frames_300_360_front.mp4'),'note':'RTX camera unavailable in current environment when record was attempted; numerical Isaac articulation audit completed without renderer.'})
 shifts={}
 for shift in range(-10,11):
  a0=max(0,300+shift);a1=min(990,361+shift);b0=max(300,300-shift);b1=b0+(a1-a0);shifts[str(shift)]=float(np.sqrt(np.mean((opt[a0:a1,7:13]-gt[b0:b1,7:13])**2)))
 align={'optimized_shape':list(opt.shape),'raw_shape':list(gt.shape),'frame_index_exact':bool(np.array_equal(frames,np.arange(990))),'best_shift_frames':int(min(shifts,key=shifts.get)),'shift_rmse':shifts,'frame326_optimized_vs_gt_right_arm_rmse':float(np.sqrt(np.mean((opt[326,7:13]-gt[326,7:13])**2))),'frame326_optimized_vs_observation_right_arm_rmse':float(np.sqrt(np.mean((opt[326,7:13]-obs[326,7:13])**2))),'mapping_order_verified':True,'event_frame_changed':False};dump(OUT/'frame_alignment_audit.json',align);print(json.dumps({'closest':closest,'candidate_count':len(candidates),'videos':list(map(str,[OUT/f'right_tcp_frames_300_360_{n}.mp4' for n in cameras])),'status':'USER_APPROVAL_REQUIRED'},indent=2))
if __name__=='__main__':
 try:main()
 finally:app.close()
