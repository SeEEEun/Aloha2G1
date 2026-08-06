#!/usr/bin/env python3
"""Offline quick scene-registered G1 target preview using the fixed 10 events."""
from __future__ import annotations
import csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
OUT=ROOT/'outputs/quick_scene_registered_preview';ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz';FROZEN=ROOT/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz';FULL=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz';TIMELINE=ROOT/'configs/episode49_task_timeline.approved.json';SEM=ROOT/'outputs/task_frame_registration/magsafe_task_semantic_definition.json';XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml');EVENTS={'left_phone_grasp_start':176,'phone_rotation_to_portrait_start':200,'phone_portrait_reached':223,'right_accessory_grasp_start':326,'accessory_detachment_start':329,'accessory_removed':341,'phone_move_to_charger_start':380,'phone_charger_attachment_complete':530,'left_phone_release_complete':586,'right_accessory_release_complete':646};FPS=30.;SCALE=.42
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def stats(x):x=np.asarray(x,float);return {'mean':float(np.mean(x)),'max':float(np.max(x,initial=0)),'rmse':float(np.sqrt(np.mean(x*x)))}
def T(R=np.eye(3),p=(0,0,0)):x=np.eye(4);x[:3,:3]=R;x[:3,3]=p;return x
def audit():
 a=np.load(ACTION,allow_pickle=False);f=np.load(FROZEN,allow_pickle=False);tl=json.load(open(TIMELINE));m={x['event']:x for x in tl['events']};bad={k:(m.get(k,{}).get('frame'),v) for k,v in EVENTS.items() if m.get(k,{}).get('frame')!=v or m.get(k,{}).get('source')!='manual_video_review'}
 out={'status':'PASS' if not bad else 'FAIL','action_shape':list(a['optimized_action'].shape),'frozen_shape':list(f['g1_arm_joint_trajectory'].shape),'fps':float(a['fps']),'frames':len(a['optimized_action']),'timestamps_monotonic':bool(np.all(np.diff(a['timestamp'])>0)),'nan_inf':int(np.size(a['optimized_action'])-np.isfinite(a['optimized_action']).sum()),'source_hash':sha(ACTION),'frozen_hash':sha(FROZEN),'approved_events':EVENTS,'event_errors':bad,'scale':SCALE,'axis_alignment_rpy_deg':[0,-7,0],'scene_registration':'PROVISIONAL_SIMULATION_ONLY','object_poses_immutable':True,'coarse_phase':{'frames':[586,646],'name':'FINAL_ACCESSORY_HOLD_TO_RELEASE_UNANNOTATED'}};dump(OUT/'audit.json',out)
 if bad or out['action_shape']!=[990,14] or out['frozen_shape']!=[990,14] or out['fps']!=30:raise RuntimeError(out)
 return a,f
def fk_source(a):
 import validate_smolvla_in_stationary_aloha_mujoco as av
 m,_=av.load_validated_model(Path('/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml'));q,_=av.mapped_qpos(a['optimized_action']);k=av.fk(m,q);lp=k['left_position_m'];rp=k['right_position_m'];lr=Rotation.from_quat(k['left_quaternion_wxyz'][:,[1,2,3,0]]).as_matrix();rr=Rotation.from_quat(k['right_quaternion_wxyz'][:,[1,2,3,0]]).as_matrix();mid=(lp+rp)/2;rel=rp-lp;relrot=np.einsum('tji,tjk->tik',lr,rr)
 np.savez_compressed(OUT/'source_aloha_task_space.npz',timestamps=a['timestamp'],left_tcp_position=lp,right_tcp_position=rp,left_tcp_rotation=lr,right_tcp_rotation=rr,bimanual_midpoint=mid,relative_vector=rel,relative_orientation=relrot,inter_hand_distance=np.linalg.norm(rel,axis=1),delta_p_left=lp-lp[0],delta_p_right=rp-rp[0],left_gripper=a['optimized_action'][:,6],right_gripper=a['optimized_action'][:,13])
 dump(OUT/'source_aloha_summary.json',{'frames':990,'fps':30,'left_path_length_m':float(np.linalg.norm(np.diff(lp,axis=0),axis=1).sum()),'right_path_length_m':float(np.linalg.norm(np.diff(rp,axis=0),axis=1).sum()),'source_grasp_relations':{'left_frame':176,'status':['SOURCE_ALOHA_GRASP_RELATION','NOT_DEX3_CALIBRATION','PALM_PROXY_ONLY'],'right_frame':326,'right_status':['SOURCE_ACCESSORY_GRASP_RELATION','NOT_REAL_GRASP_CALIBRATION']}});return lp,rp,lr,rr
def eval_fk(core,info,q):
 d=core.mujoco.MjData(info['model']);lp=[];rp=[];lr=[];rr=[]
 for x in q:
  s=core.frame_state(info,d,x);lp.append(s['left_pos']);rp.append(s['right_pos']);lr.append(Rotation.from_quat(s['left_quat'][[1,2,3,0]]).as_matrix());rr.append(Rotation.from_quat(s['right_quat'][[1,2,3,0]]).as_matrix())
 return map(np.asarray,(lp,rp,lr,rr))
def metrics(name,q,targets,core,info,runtime):
 lp,rp,lr,rr=eval_fk(core,info,q);le=np.linalg.norm(lp-targets['lp'],axis=1);re=np.linalg.norm(rp-targets['rp'],axis=1);lim=info['joint_limits'];step=np.abs(np.diff(q,axis=0));vel=step*30;acc=np.abs(np.diff(q,n=2,axis=0))*900;jerk=np.abs(np.diff(q,n=3,axis=0))*27000;norm=np.linalg.norm(np.diff(q,axis=0),axis=1);branch=int(np.sum(norm>.15));rel=(rp-lp)-(targets['rp']-targets['lp'])
 out={'method':name,'status':'ELIGIBLE_FOR_VISUAL_REVIEW' if np.mean((le<=.005)&(re<=.005))>=.99 and not branch else 'STRUCTURAL_REVIEW_REQUIRED','ik_success':float(np.mean((le<=.005)&(re<=.005))),'left_wrist_rmse_m':float(np.sqrt(np.mean(le**2))),'right_wrist_rmse_m':float(np.sqrt(np.mean(re**2))),'max_wrist_error_m':float(max(le.max(),re.max())),'orientation_error_rad':'NOT_APPLICABLE_POSITION_ONLY','relative_hand_error_m':float(np.sqrt(np.mean(np.sum(rel*rel,axis=1)))),'joint_limit_violations':int(np.sum((q<lim[:,0])|(q>lim[:,1]))),'branch_discontinuity':branch,'max_joint_step_rad':float(step.max()),'max_velocity_rad_s':float(vel.max()),'max_acceleration_rad_s2':float(acc.max()),'max_jerk_rad_s3':float(jerk.max()),'solver_runtime_sec':runtime,'arm_collision_frames':'NOT_COMPUTED_QUICK_PREVIEW','placeholder_hand':'PLACEHOLDER_HAND_DIAGNOSTIC_ONLY'}
 return out,{'lp':lp,'rp':rp,'lr':lr,'rr':rr,'le':le,'re':re}
def compose_full(arm):
 z=np.load(FULL,allow_pickle=False);full=z['full_qpos'].copy();names=z['full_joint_names'].astype(str).tolist();an=z['arm_joint_names'].astype(str);ids=[names.index(x) for x in an];full[:,ids]=arm;return full
def collision_metrics(name,full):
 import mujoco
 m=mujoco.MjModel.from_xml_path(str(XML));d=mujoco.MjData(m);struct=set();placeholder=set();records=0
 for i,q in enumerate(full):
  d.qpos[:]=q;mujoco.mj_forward(m,d)
  for j in range(d.ncon):
   c=d.contact[j];b=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or '' for g in (c.geom1,c.geom2)];records+=1
   finger=any(any(x in z for x in ('thumb','index','middle')) for z in b)
   if finger:placeholder.add(i)
   elif ('torso_link' in b and any(any(x in z for x in ('shoulder','elbow','wrist','palm')) for z in b)) or (any(z.startswith('left_') for z in b) and any(z.startswith('right_') for z in b)):struct.add(i)
 return {'method':name,'arm_palm_structural_collision_frames':len(struct),'placeholder_dex3_contact_frames':len(placeholder),'contact_pair_records':records,'placeholder_status':'PLACEHOLDER_HAND_DIAGNOSTIC_ONLY','note':'MuJoCo geometry diagnostic; not real-robot safety.'}
def render_candidate(name,full,kin,phone0,acc0):
 import cv2,mujoco
 from render_registration_approval_views import model_data
 os.environ.setdefault('MUJOCO_GL','egl');m,d,_,_,_=model_data();palm={s:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'{s}_wrist_yaw_link') for s in ('left','right')};pg=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'approval_phone');ags=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'approval_accessory_{i}') for i in range(24)];acc_local=np.array([m.geom_pos[g]-acc0 for g in ags]);views={'front':(180,-5,2.2),'side':(90,-5,2.),'top':(180,-89,2.4),'isometric':(135,-25,2.3)};writers={}
 od=OUT/'videos'/name;od.mkdir(parents=True,exist_ok=True)
 renderers={}
 for v in views:
  writers[v]=cv2.VideoWriter(str(od/f'{v}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
  renderers[v]=mujoco.Renderer(m,360,640)
 Tp176=None;Ta326=None;phone_pose=[];acc_pose=[]
 for i,q in enumerate(full):
  d.qpos[:]=q;mujoco.mj_forward(m,d)
  def palmT(side):R=d.xmat[palm[side]].reshape(3,3);return T(R,d.xpos[palm[side]]+R@np.array([.0415,.003 if side=='left' else -.003,0]))
  if i==176:Tp176=np.linalg.inv(palmT('left'))@T(np.eye(3),phone0)
  if i<176:ph=T(np.eye(3),phone0)
  elif i<=585:ph=palmT('left')@Tp176
  else:ph=phone_pose[-1]
  ac_initial=T(ph[:3,:3],ph[:3,3]+ph[:3,:3]@np.array([0,.006425,0]))
  if i==326:Ta326=np.linalg.inv(palmT('right'))@ac_initial
  if i<326:ac=ac_initial
  elif i<=645:ac=palmT('right')@Ta326
  else:ac=acc_pose[-1]
  phone_pose.append(ph.copy());acc_pose.append(ac.copy());m.geom_pos[pg]=ph[:3,3];m.geom_quat[pg]=Rotation.from_matrix(ph[:3,:3]).as_quat()[[3,0,1,2]]
  for g,local in zip(ags,acc_local):m.geom_pos[g]=ac[:3,3]+ac[:3,:3]@local
  for v,(azi,ele,dist) in views.items():
   cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[.4175,.18,.9];cam.azimuth=azi;cam.elevation=ele;cam.distance=dist
   rr=renderers[v];rr.update_scene(d,cam);img=rr.render()
   img=cv2.resize(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),(1280,720));phase='FINAL_ACCESSORY_HOLD_TO_RELEASE_UNANNOTATED' if 586<=i<=646 else 'APPROVED_EVENT_SEGMENT';lines=[name,f'frame {i} t={i/30:.3f}s',phase,'PROVISIONAL SIMULATION REGISTRATION','KINEMATIC OBJECT REPLAY / OBJECT POSE FOLLOWS PALM PROXY','NOT PHYSICS GRASP / NOT REAL ROBOT / NO OBJECT SUCCESS SNAP']
   for k,s in enumerate(lines):cv2.putText(img,s,(20,38+k*34),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,0,255) if k>=3 else (255,255,255),2)
   writers[v].write(img)
 for w in writers.values():w.release()
 for rr in renderers.values():rr.close()
 return np.asarray(phone_pose),np.asarray(acc_pose)

def comparison_outputs(statuses):
 import cv2
 names=['frozen_reference','scene_position_only','scene_partial_orientation_w00025','scene_partial_orientation_w0005']
 def failure_card(name,size=(640,360)):
  im=np.zeros((size[1],size[0],3),np.uint8);cv2.putText(im,name,(22,80),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2);cv2.putText(im,'CANDIDATE NOT GENERATED',(22,155),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,255),2);cv2.putText(im,'verified partial-axis calibration unavailable',(22,210),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,165,255),1);return im
 for view in ('front','isometric'):
  caps=[cv2.VideoCapture(str(OUT/'videos'/n/f'{view}.mp4')) if (OUT/'videos'/n/f'{view}.mp4').exists() else None for n in names]
  wr=cv2.VideoWriter(str(OUT/f'candidate_grid_{view}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
  for i in range(990):
   tiles=[]
   for n,c in zip(names,caps):
    ok,img=c.read() if c else (False,None);tiles.append(cv2.resize(img,(640,360)) if ok else failure_card(n))
   wr.write(np.vstack((np.hstack(tiles[:2]),np.hstack(tiles[2:]))))
  wr.release();[c.release() for c in caps if c]
 # Event-focused grid: hold each manually approved keyframe for one second.
 cap=cv2.VideoCapture(str(OUT/'candidate_grid_front.mp4'));wr=cv2.VideoWriter(str(OUT/'event_montage.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
 for fr in (176,223,326,341,530,586,646):
  cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,img=cap.read()
  if ok:
   cv2.putText(img,f'MANUALLY APPROVED EVENT FRAME {fr}',(340,690),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,255,255),2)
   for _ in range(30):wr.write(img)
 cap.release();wr.release()
 # Source | FK diagnostic | generated G1 comparison (no expert panel).
 raw=ROOT/'raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000';g=cv2.VideoCapture(str(OUT/'videos/scene_position_only/front.mp4'));wr=cv2.VideoWriter(str(OUT/'source_to_generated_g1.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
 imgs=sorted(raw.glob('*.png'))
 for i in range(990):
  src=cv2.imread(str(imgs[i])) if i<len(imgs) else None;src=cv2.resize(src,(426,720)) if src is not None else np.zeros((720,426,3),np.uint8)
  mid=np.zeros((720,426,3),np.uint8);cv2.putText(mid,'ALOHA TASK-SPACE FK',(35,90),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);cv2.putText(mid,'diagnostic trajectory',(70,140),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);cv2.putText(mid,f'frame {i}',(145,360),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
  ok,gi=g.read();gi=cv2.resize(gi,(428,720)) if ok else np.zeros((720,428,3),np.uint8);wr.write(np.hstack((src,mid,gi)))
 g.release();wr.release()
def task_metrics(name,obj,kin,phone0,acc0,sem):
 phone,acc=obj;pad=np.array(sem['charger_pad_verified_from_asset']['pad_face_center_scene_m']);normal=np.array(sem['charger_pad_verified_from_asset']['pad_outward_normal_scene']);long=phone[530,:3,0];back=phone[530,:3,1];disp=kin['rp'][341]-kin['rp'][329];proj=float(disp@np.array([0,1,0]));orth=float(np.linalg.norm(disp-proj*np.array([0,1,0])));ang=lambda a,b:float(np.degrees(np.arccos(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
 return {'method':name,'phone_grasp_distance_m':float(np.linalg.norm(kin['lp'][176]-(phone0+np.array([-.0748,0,0])))),'portrait_orientation_error_deg':ang(phone[223,:3,0],[0,0,1]),'accessory_approach_distance_m':float(np.linalg.norm(kin['rp'][326]-acc[326,:3,3])),'removal_projected_m':proj,'removal_orthogonal_m':orth,'removal_direction_error_deg':ang(disp,[0,1,0]),'charger_center_distance_m':float(np.linalg.norm(phone[530,:3,3]-pad)),'charger_normal_angle_deg':ang(back,-normal),'release_pose_distance_m':float(np.linalg.norm(phone[586,:3,3]-pad)),'frame646_accessory_z_m':float(acc[646,2,3]),'note':'MEASURED_RAW_VALUE; NO APPROVED TOLERANCE — NOT TASK SUCCESS; MANUAL VIDEO REVIEW REQUIRED'}
def paper(metrics,task,statuses,collisions):
 rd=OUT/'report';rd.mkdir(parents=True,exist_ok=True)
 def table(path,rows):
  keys=sorted(set().union(*(r.keys() for r in rows)));f=open(path,'w',newline='');w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows);f.close()
 table(rd/'table_retargeting_ablation.csv',metrics+statuses);table(rd/'table_task_semantic_raw_metrics.csv',task)
 table(rd/'collision_metrics.csv',collisions)
 import cv2,matplotlib.pyplot as plt
 frames=[0,176,223,326,341,530,586,646];methods=['ALOHA source','frozen_reference','scene_position_only','scene_partial_orientation_w00025','scene_partial_orientation_w0005'];fig,axs=plt.subplots(5,8,figsize=(24,15));raw=ROOT/'raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000';raws=sorted(raw.glob('*.png'))
 for r,m in enumerate(methods):
  if m=='ALOHA source':
   for c,fr in enumerate(frames):
    ax=axs[r,c];ax.axis('off');ax.set_title(str(fr));img=cv2.imread(str(raws[fr])) if fr<len(raws) else None
    if img is not None:ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    if c==0:ax.set_ylabel(m)
   continue
  cap=cv2.VideoCapture(str(OUT/'videos'/m/'front.mp4')) if (OUT/'videos'/m/'front.mp4').exists() else None
  for c,fr in enumerate(frames):
   ax=axs[r,c];ax.axis('off');ax.set_title(str(fr));
   if cap:cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,img=cap.read();ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) if ok else ax.text(.5,.5,'READ FAIL')
   else:ax.text(.5,.5,'CANDIDATE FAILED',ha='center',va='center',color='red')
   if c==0:ax.set_ylabel(m)
  if cap:cap.release()
 fig.savefig(rd/'figure_candidate_keyframes.png',dpi=300,bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(2,8,figsize=(24,6))
 cap=cv2.VideoCapture(str(OUT/'videos/scene_position_only/front.mp4'))
 for c,fr in enumerate(frames):
  for r in range(2):axs[r,c].axis('off');axs[r,c].set_title(str(fr))
  src=cv2.imread(str(raws[fr])) if fr<len(raws) else None
  if src is not None:axs[0,c].imshow(cv2.cvtColor(src,cv2.COLOR_BGR2RGB))
  cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,img=cap.read()
  if ok:axs[1,c].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
 axs[0,0].set_ylabel('ALOHA source');axs[1,0].set_ylabel('Generated G1 target');cap.release();fig.savefig(rd/'figure_source_to_g1_keyframes.png',dpi=300,bbox_inches='tight');plt.close(fig)
 sel={'selected_candidate':None,'status':'WAITING_FOR_USER_VIDEO_REVIEW','eligible':[x['method'] for x in metrics if x['status']=='ELIGIBLE_FOR_VISUAL_REVIEW']};dump(OUT/'selection.json',sel)
 html='<h1>Quick scene-registered G1 preview</h1><h2>SIMULATION TARGET VALIDATION ONLY — NO REAL ROBOT</h2><p>10 manually approved events. Frames 586–646 are coarse and unannotated. No object success snap.</p><p><a href="../candidate_grid_front.mp4">Front grid</a> | <a href="../candidate_grid_isometric.mp4">Isometric grid</a> | <a href="../event_montage.mp4">Event montage</a> | <a href="../source_to_generated_g1.mp4">Source to G1</a></p><p><a href="table_retargeting_ablation.csv">Structural</a> | <a href="table_task_semantic_raw_metrics.csv">Raw task metrics</a> | <a href="collision_metrics.csv">Collisions</a></p><img src="figure_candidate_keyframes.png" width="1200"><img src="figure_source_to_g1_keyframes.png" width="1200"><pre>'+json.dumps(sel,indent=2)+'</pre>';(rd/'index.html').write_text(html)
def main():
 print('SIMULATION TARGET VALIDATION ONLY\nNO REAL G1 WAS USED\nNO REAL ALOHA WAS USED\nNO DDS OR PUBLISHER WAS USED');a,f=audit();slp,srp,slr,srr=fk_source(a);import retarget_episode49_optimized_action_to_g1 as core;info=core.ik.validate_model(XML)
 with np.load(FROZEN,allow_pickle=False) as z:fq=z['g1_arm_joint_trajectory'].astype(float);base_lp=z['g1_target_left_position'];base_rp=z['g1_target_right_position'];start=z['g1_start_arm_q']
 sem=json.load(open(SEM));phone0=np.array(sem['poses']['initial_phone_pose']['position_scene_m']);acc0=phone0+np.array([0,.006425,0]);shift=phone0+np.array([-.0748,0,0])-base_lp[176];targets={'lp':base_lp+shift,'rp':base_rp+shift,'lr':np.repeat(np.eye(3)[None],990,0),'rr':np.repeat(np.eye(3)[None],990,0)};anchor={'translation_m':shift.tolist(),'rotation':core.align_mod.make_align_rotation(core.ALIGN_RPY).tolist(),'scale':SCALE,'anchor_event':'left_phone_grasp_start','anchor_frame':176,'single_constant_transform':True,'both_hands_same_transform':True,'all_candidates_same_transform':True,'stage_offsets':False,'right_hand_offset':False,'object_poses_unchanged':True,'status':'PROVISIONAL_SIMULATION_ONLY'};dump(OUT/'global_scene_anchor.json',anchor)
 # A translated target must first receive the converter's verified frame-wise
 # position seed.  Starting temporal Gauss-Newton directly from the unshifted
 # frozen qpos was a diagnosed integration bug.  Also refuse a known
 # statically-unreachable registration instead of emitting another candidate.
 static_report=ROOT/'outputs/scene_registration_failure_diagnosis/static_reachability_summary.json'
 if static_report.exists() and not json.load(open(static_report))['all_targets_reachable']:
  raise RuntimeError('CURRENT G1 SCENE PLACEMENT IS UNREACHABLE; USER APPROVAL REQUIRED')
 candidates={'frozen_reference':fq};t=time.perf_counter();seed=core.position_seed(info,targets,start);scene=core.temporal_solve(info,targets,seed,start,0.,8);runtime=time.perf_counter()-t;candidates['scene_position_only']=scene
 failed=[{'method':'scene_partial_orientation_w00025','status':'FAILED_USER_DECISION_REQUIRED','reason':'Verified partial-axis closing objective unavailable; ALOHA-to-G1/Dex3 closing-axis calibration remains pending. No arbitrary full-orientation substitute used.'},{'method':'scene_partial_orientation_w0005','status':'FAILED_USER_DECISION_REQUIRED','reason':'Same verified partial-orientation objective unavailable.'}]
 mets=[];tasks=[];collisions=[]
 for name,q in candidates.items():
  tar={'lp':base_lp,'rp':base_rp} if name=='frozen_reference' else targets;m,k=metrics(name,q,tar,core,info,0 if name=='frozen_reference' else runtime);mets.append(m);full=compose_full(q);collisions.append(collision_metrics(name,full));obj=render_candidate(name,full,k,phone0,acc0);tasks.append(task_metrics(name,obj,k,phone0,acc0,sem));np.savez_compressed(OUT/f'{name}.npz',arm_qpos=q,full_qpos=full,target_left_position=tar['lp'],target_right_position=tar['rp'],phone_pose=obj[0],accessory_pose=obj[1],fps=np.array(30.))
 comparison_outputs(failed);paper(mets,tasks,failed,collisions);dump(OUT/'candidate_status.json',{'generated':list(candidates),'failed':failed,'coarse_final_phase':[586,646],'no_additional_events_required':True});print('QUICK G1 CANDIDATE PREVIEW GENERATED\nEXISTING 10 MANUAL EVENTS USED\nCOARSE FINAL PHASE USED\nPAPER FIGURES AND TABLES GENERATED\nNO REAL ROBOT WAS USED\nNO FINAL CANDIDATE SELECTED');return 0
if __name__=='__main__':raise SystemExit(main())
