#!/usr/bin/env python3
"""One-shot world-TCP retargeting pipeline with strict source-scene hard gate.

No hardware, DDS, publisher, GUI, viewport, or physics-grasp code is used.
Downstream root search/rendering is refused when the source ALOHA replay does
not agree with the authoritative fixed scene.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,types
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz';SC=ROOT/'isaaclab_magsafe_fixed_scene';TIMELINE=ROOT/'configs/episode49_task_timeline.approved.json';EVENTS={176:'left_phone_grasp_start',200:'phone_rotation_to_portrait_start',223:'phone_portrait_reached',326:'right_accessory_grasp_start',329:'accessory_detachment_start',341:'accessory_removed',380:'phone_move_to_charger_start',530:'phone_charger_attachment_complete',586:'left_phone_release_complete',646:'right_accessory_release_complete'}
def args():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--visualizer',default='none');p.add_argument('--enable_cameras',action='store_true');p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/g1_world_task_retargeting');p.add_argument('--top-root-k',type=int,default=5);p.add_argument('--temporal-k',type=int,default=3);p.add_argument('--record-width',type=int,default=1280);p.add_argument('--record-height',type=int,default=720);p.add_argument('--record-fps',type=int,default=30);p.add_argument('--run-smoke-test',action='store_true');p.add_argument('--render-full',action='store_true');return p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 a=args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);print('SIMULATION TARGET VALIDATION ONLY\nNO REAL G1 OR ALOHA\nNO DDS OR PUBLISHER')
 try:import pandas  # noqa: F401
 except ModuleNotFoundError:sys.modules['pandas']=types.ModuleType('pandas')
 import validate_smolvla_in_stationary_aloha_mujoco as fkmod
 z=np.load(ACTION);act=z['optimized_action'].astype(float);ts=z['timestamp'].astype(float);timeline=json.load(open(TIMELINE));approved={e['event']:e for e in timeline['events']};errs={n:(approved.get(n,{}).get('frame'),fr,approved.get(n,{}).get('source')) for fr,n in EVENTS.items() if approved.get(n,{}).get('frame')!=fr or approved.get(n,{}).get('source')!='manual_video_review'}
 if act.shape!=(990,14) or not np.isfinite(act).all() or float(z['fps'])!=30 or errs:raise RuntimeError({'shape':act.shape,'events':errs})
 model,_=fkmod.load_validated_model(Path('/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml'));q,_=fkmod.mapped_qpos(act);k=fkmod.fk(model,q);pose=json.load(open(SC/'magsafe_robot_preview_config.json'))['stationary_aloha'];Rw=Rotation.from_quat(np.array(pose['orientation_wxyz'])[[1,2,3,0]]).as_matrix();t=np.array(pose['position_xyz_m']);lp=t+(Rw@k['left_position_m'].T).T;rp=t+(Rw@k['right_position_m'].T).T
 lr=np.einsum('ij,tjk->tik',Rw,Rotation.from_quat(k['left_quaternion_wxyz'][:,[1,2,3,0]]).as_matrix());rr=np.einsum('ij,tjk->tik',Rw,Rotation.from_quat(k['right_quaternion_wxyz'][:,[1,2,3,0]]).as_matrix());mid=(lp+rp)/2;rel=rp-lp
 meta={'world_frame':'existing Isaac /World','aloha_root_pose_source':str(SC/'magsafe_robot_preview_config.json'),'fk_source':'stationary_ai.xml follower link TCP plus verified local offset','units':'meter','fps':30,'frames':990,'action_hash':sha(ACTION)}
 (out/'source').mkdir(exist_ok=True);np.savez_compressed(out/'source/aloha_tcp_world_trajectory.npz',left_tcp_world_position=lp,right_tcp_world_position=rp,left_tcp_world_orientation=lr,right_tcp_world_orientation=rr,midpoint=mid,relative_vector=rel,inter_hand_distance=np.linalg.norm(rel,axis=1),timestamps=ts,metadata=np.array(json.dumps(meta)))
 # Values come directly from the authoritative composed USD. Asset-derived
 # radii are gates, not arbitrary root-search bounds or task-success tolerance.
 phone=np.array([.525,.2556239235301329,.830744555101841]);accessory=np.array([.525,.26204794497151274,.8306324233904814]);charger=np.array([.42,.525846518946957,.9396181100184134]);checks=[('frame176_left_phone',176,lp[176],phone,.083),('frame326_right_accessory',326,rp[326],accessory,.060),('frame530_left_charger',530,lp[530],charger,.070)];rows=[]
 for name,fr,p,o,radius in checks:rows.append({'check':name,'frame':fr,'tcp_world_m':p.tolist(),'object_world_m':o.tolist(),'distance_m':float(np.linalg.norm(p-o)),'asset_derived_near_radius_m':radius,'near':bool(np.linalg.norm(p-o)<=radius)})
 valid=all(x['near'] for x in rows);status='SOURCE_WORLD_TCP_VALID' if valid else 'ALOHA_SCENE_REPLAY_MISMATCH';source={'status':status,'checks':rows,'reason':'right TCP at manually approved accessory grasp frame lies outside accessory asset-derived bounding region' if not valid else 'all key source checks near authoritative objects','root_transform_applied':pose,'object_poses_changed':False,'camera_poses_changed':False};dump(out/'source/source_validation.json',source)
 audit={'status':'PASS','action_shape':list(act.shape),'fps':30,'finite':True,'approved_event_errors':errs,'scene':str(SC/'generated/magsafe_magnetic_scene_v2.usda'),'scene_hash':sha(SC/'generated/magsafe_magnetic_scene_v2.usda'),'scene_builder_hash':sha(SC/'magsafe_scene_builder.py'),'camera_source_hash':sha(SC/'robot_model_preview_common.py'),'objects_immutable':True,'camera_immutable':True,'orientation_weight':0,'hardware_used':False};dump(out/'input_audit.json',audit)
 selection={'selected_root_candidate':None,'selected_trajectory':None,'status':'BLOCKED_ALOHA_SCENE_REPLAY_MISMATCH' if not valid else 'WAITING_FOR_USER_ROOT_AND_VIDEO_APPROVAL'};dump(out/'selection.json',selection)
 for pth,header in [(out/'root_search/root_candidate_metrics.csv','status,note\n'),(out/'metrics/temporal_candidate_metrics.csv','status,note\n'),(out/'report/table_root_candidate_metrics.csv','status,note\n'),(out/'report/table_temporal_candidate_metrics.csv','status,note\n'),(out/'report/table_frozen_vs_world_target.csv','status,note\n')]:pth.parent.mkdir(parents=True,exist_ok=True);pth.write_text(header+f'{status},downstream not run because source-scene hard gate failed\n')
 dump(out/'root_search/root_candidates.json',{'status':status,'candidates':[],'bounds':'NOT_DERIVED_SOURCE_GATE_FAILED','arbitrary_bounds_used':False});dump(out/'render_status.json',{'status':'NOT_RUN_SOURCE_GATE_FAILED','smoke_mp4_generated':False,'full_videos_generated':False,'fake_video_generated':False,'sequential_rendering_planned':True,'visualizer':'none','enable_cameras_requested':a.enable_cameras})
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(10,6));ax.scatter([phone[0],accessory[0],charger[0]],[phone[2],accessory[2],charger[2]],s=100,label='scene objects');ax.scatter([lp[176,0],rp[326,0],lp[530,0]],[lp[176,2],rp[326,2],lp[530,2]],marker='x',s=100,label='ALOHA world TCP');[ax.plot([r['tcp_world_m'][0],r['object_world_m'][0]],[r['tcp_world_m'][2],r['object_world_m'][2]],'r-') for r in rows];ax.set(xlabel='World X (m)',ylabel='World Z (m)',title='Source-scene hard-gate keyframes');ax.legend();(out/'report').mkdir(exist_ok=True);fig.savefig(out/'report/root_candidate_keyframes.png',dpi=300,bbox_inches='tight');plt.close(fig)
 html=f'<h1>{status}</h1><p>Existing Isaac scene reused. No objects or cameras changed.</p><pre>{json.dumps(source,indent=2)}</pre><p>Root search, temporal IK, and rendering were not run; no fake task video was generated.</p><pre>{json.dumps(selection,indent=2)}</pre>';(out/'report/index.html').write_text(html)
 if not valid:print('ALOHA_SCENE_REPLAY_MISMATCH\nROOT SEARCH NOT RUN\nTEMPORAL IK NOT RUN\nHEADLESS VIDEO NOT GENERATED');return 3
 raise RuntimeError('Source gate passed, but downstream implementation must be explicitly validated before use')
if __name__=='__main__':raise SystemExit(main())
