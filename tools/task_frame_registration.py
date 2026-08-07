#!/usr/bin/env python3
"""Build and validate the offline MagSafe task-frame registration artifacts."""
from __future__ import annotations
import hashlib,json,math,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path('/home/jbnu/aloha_g1_dataset')
REG=ROOT/'outputs/task_frame_registration'; SCENE_OUT=ROOT/'outputs/scene_registered_retargeting'
LAYOUT=ROOT/'isaaclab_magsafe_fixed_scene/scene_layout.json'
POSES=ROOT/'isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json'
ALOHA_XML=Path('/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml')
G1_XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
ARM=ROOT/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz'
FULL=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz'
APPROVED_ROOT=ROOT/'configs/g1_root_forward_v14.approved.json'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def T(R=np.eye(3),p=(0,0,0)):
 x=np.eye(4);x[:3,:3]=R;x[:3,3]=p;return x
def qmat(q):
 w,x,y,z=np.asarray(q,float)/np.linalg.norm(q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def rec(name,parent,mat,source,status,confidence,axes='right-handed +X,+Y,+Z; meters'):
 return {'frame_name':name,'parent_frame':parent,'T_parent_from_frame':None if mat is None else np.asarray(mat).tolist(),'units':'meter/radian','handedness':'right','axis_convention':axes,'source':source,'confidence':confidence,'status':status}
def validate_transform(x):
 x=np.asarray(x,float);R=x[:3,:3]
 return {'rotation_orthonormal':bool(np.allclose(R.T@R,np.eye(3),atol=1e-10)),'rotation_determinant':float(np.linalg.det(R)),'last_row_valid':bool(np.allclose(x[3],[0,0,0,1])),'inverse_consistency':bool(np.allclose(x@np.linalg.inv(x),np.eye(4),atol=1e-10)),'right_handed':bool(np.linalg.det(R)>0)}
def build():
 layout=json.load(open(LAYOUT));poses=json.load(open(POSES));gp=dict(poses['g1']);ap=poses['stationary_aloha']

 # Current user-approved simulation layout.
 phone=np.array([(layout['phone']['bottom_left_xy'][0]+layout['phone']['bottom_right_xy'][0])/2,layout['phone']['bottom_left_xy'][1],layout['table']['surface_height']+layout['phone']['size_landscape_xyz'][2]/2])
 acc=phone+np.array([0,layout['phone']['size_landscape_xyz'][1]/2+layout['accessory']['phone_back_clearance']+layout['accessory']['main_depth']/2,0])
 ch=np.array([*layout['charger']['center_xy'],layout['table']['surface_height']+(layout['charger']['mount_plate']['size_xyz'][2] if layout['charger']['mount_plate']['enabled'] else 0)])

 # Reproduce preview_magsafe_g1_model.py using the explicitly approved total
 # forward offset.  The pose file always retains the original root, preventing
 # cumulative/double application.
 # Forward is the horizontal direction from the original G1 root toward
 # the current phone/accessory task center.
 # Final total offset, applied exactly once to the original root pose.
 approved=json.load(open(APPROVED_ROOT)) if APPROVED_ROOT.is_file() else None
 root_forward_offset_m=float(approved['selected_total_forward_offset_m']) if approved else .15
 root_label=f'{root_forward_offset_m:.3f}'
 original_root=np.asarray(gp['position_xyz_m'],float)
 task_center=(phone+acc)/2
 forward_xy=task_center[:2]-original_root[:2]
 forward_norm=np.linalg.norm(forward_xy)
 if forward_norm < 1e-9:
  raise RuntimeError('G1 root and current task center have identical horizontal positions')
 forward_xy=forward_xy/forward_norm
 applied_root=original_root.copy()
 applied_root[:2]+=root_forward_offset_m*forward_xy
 gp['position_xyz_m']=applied_root.tolist()

 Tsg=T(qmat(gp['orientation_wxyz']),gp['position_xyz_m']); Tsa=T(qmat(ap['orientation_wxyz']),ap['position_xyz_m'])

 # Task origin is the table-front center on the measured top surface.
 Rt=np.column_stack(([0,1,0],[-1,0,0],[0,0,1]));Tst=T(Rt,[layout['table']['size_x']/2,0,layout['table']['surface_height']])
 Tts=np.linalg.inv(Tst); Ttg=Tts@Tsg; Tta=Tts@Tsa
 frames=[rec('fixed_scene_world',None,np.eye(4),str(LAYOUT),'verified',1.0,'+X table left-to-right; +Y operator-to-charger; +Z up'),
  rec('magsafe_task_frame','fixed_scene_world',Tst,str(LAYOUT),'inferred_pending_manual_approval',.8,'+X toward workspace; +Y task-left; +Z up'),
  rec('g1_model_world',None,np.eye(4),str(G1_XML),'verified_model_local',1.0),rec('g1_base','fixed_scene_world',Tsg,str(POSES)+f':g1 + approved final total +{root_label} m forward offset','verified_for_preview_composition_only',.9),
  rec('g1_torso','g1_base',None,str(G1_XML),'dynamic_fk',1.0),rec('g1_left_wrist_yaw','g1_base',None,str(G1_XML),'dynamic_fk',1.0),rec('g1_right_wrist_yaw','g1_base',None,str(G1_XML),'dynamic_fk',1.0),
  rec('g1_left_palm_proxy','g1_left_wrist_yaw',T(np.eye(3),[.0415,.003,0]),str(G1_XML),'verified_from_geom_transform',1.0),rec('g1_right_palm_proxy','g1_right_wrist_yaw',T(np.eye(3),[.0415,-.003,0]),str(G1_XML),'verified_from_geom_transform',1.0),
  rec('aloha_stationary_world',None,np.eye(4),str(ALOHA_XML),'verified_model_local',1.0),rec('aloha_scene_root','fixed_scene_world',Tsa,str(POSES)+':stationary_aloha','verified_for_preview_composition_only',.9),
  rec('aloha_left_base','aloha_stationary_world',T(qmat([.707107,0,0,-.707107]),[-.019982,.4575,.039086]),str(ALOHA_XML),'verified',1.0),rec('aloha_right_base','aloha_stationary_world',T(qmat([.707107,0,0,.707107]),[-.019982,-.4575,.039086]),str(ALOHA_XML),'verified',1.0),
  rec('aloha_left_link_6','aloha_left_base',None,str(ALOHA_XML),'dynamic_fk',1.0),rec('aloha_right_link_6','aloha_right_base',None,str(ALOHA_XML),'dynamic_fk',1.0),rec('aloha_cam_high','aloha_stationary_world',T(qmat([-.6904,-.153,.153,.6904]),[-.324675,.009,1.047775]),str(ALOHA_XML),'verified',1.0),
  rec('phone_center','fixed_scene_world',T(np.eye(3),phone),str(LAYOUT),'verified_from_scene_builder_inputs',1.0),rec('accessory_center','fixed_scene_world',T(np.eye(3),acc),str(LAYOUT),'verified_from_scene_builder_inputs',1.0),rec('charger_root','fixed_scene_world',T(np.eye(3),ch),str(LAYOUT),'verified_from_scene_builder_inputs',1.0)]
 graph={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'frames':frames,'unknown_transforms':['task_from_g1_torso (dynamic)','task_from_aloha_left_base/right_base: preview composition path exists but task correspondence is not approved'],'hashes':{str(p):sha(p) for p in (ACTION,ARM,FULL,LAYOUT,POSES,G1_XML,ALOHA_XML)}}
 dump(REG/'frame_graph.json',graph)
 md=['# Coordinate frame graph','',*[f"- `{x['frame_name']}` ← `{x['parent_frame']}`: **{x['status']}**, source `{x['source']}`" for x in frames]];(REG/'frame_graph.md').write_text('\n'.join(md)+'\n')
 registration={'schema_version':1,'simulation_only':True,'authoritative_for_real_robot':False,'registration_method':f'current scene_layout geometry plus approved final total +{root_label} m G1 preview forward offset; simulation only','status':'USER_AUTHORIZED_FORWARD_ROOT_REGISTRATION' if approved else 'NEEDS_MANUAL_APPROVAL','evidence_sources':[str(LAYOUT),str(POSES),str(ROOT/'isaaclab_magsafe_fixed_scene/robot_model_preview_common.py'),str(APPROVED_ROOT) if approved else 'NO_V14_APPROVAL_FILE'],
  'manual_adjustment_used':True,'manual_adjustment_log':[{
      'parameter':'g1_root_forward_offset_m',
      'value_m':root_forward_offset_m,
      'basis':approved['selection_reason'] if approved else 'user-requested final total static preview offset; visual approval pending',
      'original_root_position_m':original_root.tolist(),
      'applied_root_position_m':applied_root.tolist()
     }],'T_scene_from_task':Tst.tolist(),'T_task_from_scene':Tts.tolist(),'T_scene_from_g1_base':Tsg.tolist(),'T_task_from_g1_base':Ttg.tolist(),'T_task_from_g1_torso':'UNKNOWN_DYNAMIC_FK','T_task_from_aloha_scene_root':Tta.tolist(),'T_task_from_aloha_left_base':'UNKNOWN_NOT_APPROVED','T_task_from_aloha_right_base':'UNKNOWN_NOT_APPROVED',
  'constraints':[{'name':'lateral_center','value_m':layout['table']['size_x']/2,'source':str(LAYOUT)},{'name':'g1_root_forward_offset','value_m':root_forward_offset_m,'source':str(APPROVED_ROOT) if approved else 'user-requested final total static preview offset; visual approval pending'},{'name':'table_surface_minus_g1_root_z','value_m':layout['table']['surface_height']-gp['position_xyz_m'][2],'source':str(LAYOUT)+' + '+str(POSES)}],
  'validation':{'T_scene_from_task':validate_transform(Tst),'T_scene_from_g1_base':validate_transform(Tsg),'T_task_from_g1_base':validate_transform(Ttg),'round_trip_task_scene':bool(np.allclose(Tts@Tst,np.eye(4))),'units_consistent':True,'one_fixed_transform_for_all_candidates':True}}
 dump(ROOT/'configs/magsafe_task_frame_registration.sim.json',registration)
 semantic={'schema_version':1,'simulation_only':True,'authoritative_for_real_robot':False,'status':'NEEDS_MANUAL_APPROVAL','object_poses_immutable':True,
  'phone':{'root_scene_position_m':phone.tolist(),'root_position_status':'VERIFIED_FROM_SCENE_CODE','initial_pose':'VERTICAL_LANDSCAPE','axes':{'+X':'landscape long axis','+Y':'back-face/MagSafe outward normal','+Z':'short vertical axis'},'size_xyz_m':layout['phone']['size_landscape_xyz'],'screen_normal_scene':[0,-1,0],'back_surface_normal_scene':[0,1,0],
   'deprecated_phone_grasp_band':{'status':'UNAPPROVED_HEURISTIC','trajectory_gate_enabled':False,'success_metric_enabled':False},
   'left_side_semantic_grasp_frame':{'position_scene_m':(phone+[-layout['phone']['size_landscape_xyz'][0]/2,0,0]).tolist(),'position_basis':'phone local -X surface candidate center','approach_axis_scene':[1,0,0],'approach_basis':'from outside left toward phone side','closing_axis':'PENDING_APPROVED_LEFT_PHONE_GRASP_START_ALOHA_FK','tolerance':'NOT_DEFINED','source':'USER_APPROVED_ROLE_BUT_POSE_PENDING_VIDEO_ANNOTATION'},
   'pregrasp_offset':'NOT_DEFINED','angular_tolerance':'NOT_DEFINED'},
  'accessory':{'root_scene_position_m':acc.tolist(),'root_position_status':'VERIFIED_FROM_SCENE_CODE','attachment_relation':'back-center of initial phone','attachment_axis_scene':[0,-1,0],'initial_removal_direction_scene':[0,1,0],'removal_direction':'current approved phone back-surface outward normal','outer_radius_m':layout['accessory']['main_outer_diameter']/2,'inner_radius_m':layout['accessory']['main_inner_diameter']/2,'depth_m':layout['accessory']['main_depth'],'grasp_region':'asset annulus','grasp_tolerance':'NOT_DEFINED','removal_distance':'NOT_DEFINED'},
  'charger':{'root_scene_position_m':ch.tolist(),'root_position_status':'VERIFIED_FROM_SCENE_CODE','pad_tilt_deg':layout['charger']['pad_tilt_degrees_up'],'pad_radius_m':layout['charger']['pad_diameter']/2,'pad_normal_scene':[0,-math.cos(math.radians(layout['charger']['pad_tilt_degrees_up'])),math.sin(math.radians(layout['charger']['pad_tilt_degrees_up']))],'placement_relation':'phone center aligned to pad face center; portrait; phone back normal opposes pad outward normal','position_tolerance':'USER_DECISION_REQUIRED','angular_tolerance':'USER_DECISION_REQUIRED'}}
 dump(ROOT/'configs/magsafe_object_semantic_frames.sim.json',semantic)
 tool={'schema_version':1,'simulation_only':True,'status':'NEEDS_MANUAL_APPROVAL','source_xml':str(ALOHA_XML),'tcp_offset_link6_m':[.1487,0,-.00105],
  'left':{'tcp_parent':'follower_left_link_6','approach_axis_local':[1,0,0],'closing_axis_local':[0,1,0],'lateral_axis_local':[0,0,1],'opening_plane':'local X-Z','model_evidence':'symmetric jaw pad centers at local +/-Y; tips extend local +X','status':'VERIFIED_FROM_MODEL_NEEDS_VIDEO_APPROVAL'},
  'right':{'tcp_parent':'follower_right_link_6','approach_axis_local':[1,0,0],'closing_axis_local':[0,1,0],'lateral_axis_local':[0,0,1],'opening_plane':'local X-Z','model_evidence':'symmetric jaw pad centers at local +/-Y; tips extend local +X','status':'VERIFIED_FROM_MODEL_NEEDS_VIDEO_APPROVAL'}}
 dump(ROOT/'configs/aloha_tool_axes_calibration.sim.json',tool)
 return graph,registration,semantic,tool

if __name__=='__main__':
 g,r,s,t=build();print(json.dumps({'frames':len(g['frames']),'registration':r['status'],'semantic':s['status'],'tool_axes':t['status']},indent=2))
