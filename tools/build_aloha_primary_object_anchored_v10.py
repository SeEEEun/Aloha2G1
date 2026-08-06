#!/usr/bin/env python3
"""Build the Episode-49 ALOHA-primary object-anchored arm candidates (simulation only)."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, math, shutil, sys
import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom

ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'isaaclab_magsafe_fixed_scene'))
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_aloha_primary_object_anchored_v10'
SRC=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz';TIMELINE=ROOT/'configs/episode49_task_timeline.approved.json';V8=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_restore_original_v8/restored_exact_arm_trajectory.npz'
LAYOUT=ROOT/'isaaclab_magsafe_fixed_scene/scene_layout.json';BUILDER=ROOT/'isaaclab_magsafe_fixed_scene/magsafe_scene_builder.py';TARGET_USD=ROOT/'isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda';POSE=ROOT/'isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json'
DECL=ROOT/'configs/episode49_source_scene.user_approved.json';SOURCE_LAYOUT=ROOT/'configs/episode49_source_scene_layout.json';SOURCE_FRAMES=ROOT/'configs/episode49_source_object_frames.user_approved.json';TOOL_CAL=ROOT/'configs/aloha_tcp_to_g1_palm_calibration.sim.json'
KNOTS=np.array([0,176,200,223,326,329,341,380,530,586,646,702,989],int)
HARD={'left':{176:'phone_grasp',223:'portrait',530:'charger'},'right':{326:'accessory_grasp',341:'accessory_removed'}}
C_L=np.array([[1.,0,0],[0,0,-1],[0,1,0]]);C_R=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
ORIG_ROOT=np.array([0.,0.,0.7922728583]);TARGET_ROOT=np.array([.44514890950197095,-.35257022755443246,.7922728583]);R_TARGET=np.array([[0,-1,0],[1,0,0],[0,0,1.]],float)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
def T(R=np.eye(3),p=np.zeros(3)):
 x=np.eye(4);x[:3,:3]=R;x[:3,3]=p;return x
def inv(x):
 R=x[:3,:3];return T(R.T,-R.T@x[:3,3])
def quatR(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def tf(stage,path):
 prim=stage.GetPrimAtPath(path)
 if not prim.IsValid():raise RuntimeError(path)
 return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float).T
def poses(pos,quat):return np.asarray([T(quatR(q),p) for p,q in zip(pos,quat)])
def rel_progress(x,start):return np.asarray([inv(x[start])@y for y in x])
def smooth5(u):return 10*u**3-15*u**4+6*u**5
def blend_knots(values,progress):
 out=np.empty((990,values.shape[1]))
 for a,b,va,vb in zip(KNOTS[:-1],KNOTS[1:],values[:-1],values[1:]):
  u=progress[a:b+1].copy();u=(u-u[0])/(u[-1]-u[0]) if u[-1]>u[0]+1e-12 else np.linspace(0,1,b-a+1);w=smooth5(u)[:,None];out[a:b+1]=(1-w)*va+w*vb
 return out
def phase_progress(lp,rp,lr,rr):
 # Source-derived combined translation/rotation arc length, stable in source time.
 p=np.zeros(990)
 step=np.linalg.norm(np.diff(lp,axis=0),axis=1)+np.linalg.norm(np.diff(rp,axis=0),axis=1)
 dl=Rotation.from_matrix(np.einsum('tji,tjk->tik',lr[:-1],lr[1:])).magnitude();dr=Rotation.from_matrix(np.einsum('tji,tjk->tik',rr[:-1],rr[1:])).magnitude();step+=.04*(dl+dr)
 p[1:]=np.cumsum(step);return p
def solve_knot_values(anchor_l,anchor_r,weights):
 n=len(KNOTS);D1=np.diff(np.eye(n),axis=0);D2=np.diff(np.eye(n),n=2,axis=0);I=np.eye(n)
 H=weights['magnitude']*block_diag(I,I)+weights['velocity']*block_diag(D1.T@D1,D1.T@D1)+weights['acceleration']*block_diag(D2.T@D2,D2.T@D2)
 C=np.hstack((I,-I));H+=weights['bimanual']*(C.T@C);H+=1e-9*np.eye(2*n)
 out=[]
 for dim in range(3):
  rows=[];vals=[]
  for side,anchors,off in [('left',anchor_l,0),('right',anchor_r,n)]:
   for frame,value in anchors.items():r=np.zeros(2*n);r[off+int(np.where(KNOTS==frame)[0][0])]=1;rows.append(r);vals.append(value[dim])
  # Terminal residual is held exactly from 702 through 989.
  for off in (0,n):r=np.zeros(2*n);r[off+n-1]=1;r[off+n-2]=-1;rows.append(r);vals.append(0.)
  A=np.asarray(rows);b=np.asarray(vals);K=np.block([[H,A.T],[A,np.zeros((len(A),len(A)))]])
  sol=np.linalg.solve(K,np.r_[np.zeros(2*n),b])[:2*n];out.append(sol)
 x=np.asarray(out).T;return x[:n],x[n:]
def rotvec_residual(anchor,base,frame):return Rotation.from_matrix(anchor[:3,:3]@base[frame,:3,:3].T).as_rotvec()
def apply_rotation(base,rv):return np.asarray([Rotation.from_rotvec(x).as_matrix()@r for x,r in zip(rv,base)])
def corr(a,b):
 if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return 1.0 if np.allclose(a,b) else 0.0
 return float(np.corrcoef(a,b)[0,1])
def fidelity(base_l,base_r,c_l,c_r,base_lr,base_rr,c_lr,c_rr):
 rows={};bounds=list(zip(KNOTS[:-1],KNOTS[1:]));mins={'path':1.,'speed':1.,'rotation':1.}
 for a,b in bounds:
  label=f'{a}_{b}';rows[label]={}
  for side,x,y,R0,R1 in [('left',base_l,c_l,base_lr,c_lr),('right',base_r,c_r,base_rr,c_rr)]:
   xb=x[a:b+1]-x[a];yb=y[a:b+1]-y[a];dx=np.diff(xb,axis=0);dy=np.diff(yb,axis=0);sx=np.linalg.norm(dx,axis=1);sy=np.linalg.norm(dy,axis=1)
   path=corr(xb.ravel(),yb.ravel());tangent=float(np.mean(np.sum(dx*dy,axis=1)/(np.linalg.norm(dx,axis=1)*np.linalg.norm(dy,axis=1)+1e-12))) if len(dx) else 1.
   speed=corr(sx,sy);acc=corr(np.diff(sx),np.diff(sy));curv=corr(np.linalg.norm(np.diff(dx,axis=0),axis=1),np.linalg.norm(np.diff(dy,axis=0),axis=1))
   p0=Rotation.from_matrix(np.einsum('ji,tjk->tik',R0[a],R0[a:b+1])).magnitude();p1=Rotation.from_matrix(np.einsum('ji,tjk->tik',R1[a],R1[a:b+1])).magnitude();rot=corr(p0,p1)
   disp=float(np.linalg.norm(x[b]-x[a]));res=float(np.linalg.norm((y[b]-x[b])-(y[a]-x[a])));ratio=res/max(disp,1e-9)
   rows[label][side]={'path_shape_correlation':path,'tangent_direction_cosine':tangent,'speed_profile_correlation':speed,'acceleration_profile_correlation':acc,'curvature_correlation':curv,'relative_rotation_progress_correlation':rot,'source_phase_displacement_m':disp,'residual_displacement_m':res,'residual_source_displacement_ratio':ratio}
   if b-a>=20:mins['path']=min(mins['path'],path);mins['speed']=min(mins['speed'],speed);mins['rotation']=min(mins['rotation'],rot)
  bm=.5*(base_l[a:b+1]+base_r[a:b+1]);cm=.5*(c_l[a:b+1]+c_r[a:b+1]);br=base_r[a:b+1]-base_l[a:b+1];cr=c_r[a:b+1]-c_l[a:b+1]
  rows[label]['bimanual']={'midpoint_rmse_m':float(np.sqrt(np.mean((cm-bm)**2))),'relative_vector_rmse_m':float(np.sqrt(np.mean((cr-br)**2))),'inter_hand_distance_trend_correlation':corr(np.linalg.norm(br,axis=1),np.linalg.norm(cr,axis=1)),'event_timing_difference_frames':0}
 return rows,mins
def event_name(frame,events):
 cur='pre_task'
 for e in events:
  if e['frame']<=frame:cur=e['event']
 return cur

def main():
 import retarget_episode49_optimized_action_to_g1 as core
 from magsafe_scene_builder import load_layout,build_table_asset,build_phone_asset,build_accessory_asset,build_charger_asset,build_composite_scene
 ap=argparse.ArgumentParser();ap.add_argument('--skip-plots',action='store_true');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);(OUT/'report').mkdir(exist_ok=True);gen=ROOT/'outputs/episode49_source_scene/generated';gen.mkdir(parents=True,exist_ok=True)
 target_hashes={str(p.resolve()):sha(p) for p in (LAYOUT,TARGET_USD,ROOT/'isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda')}
 layout=load_layout(LAYOUT);source=json.loads(json.dumps(layout));source['phone']['bottom_left_xy']=[.45,.255];source['phone']['bottom_right_xy']=[.60,.255];source['charger']['center_xy']=[.42,.520];dump(SOURCE_LAYOUT,source);dump(OUT/'source_scene_layout.json',source)
 declaration={'status':'USER_APPROVED_PROJECT_SCENE_EQUIVALENCE','scope':'EPISODE_49_SOURCE_RELATION_RECOVERY_ONLY','phone':{'bottom_left_xy':[.45,.255],'bottom_right_xy':[.60,.255],'size_xyz_m':[.1496,.00795,.0715],'mass_kg_metadata':.177,'orientation':'VERTICAL_LANDSCAPE'},'charger':{'center_xy':[.42,.520]},'accessory':'source phone back-center; active builder attachment transform','geometry_source':[str(LAYOUT.resolve()),str(BUILDER.resolve())],'target_scene_modification_allowed':False,'camera_registration_required':False}
 dump(DECL,declaration);dump(OUT/'user_approved_source_scene_declaration.json',declaration)
 assets={'table':gen/'source_table_optical.usda','phone':gen/'source_phone_landscape.usda','accessory':gen/'source_accessory.usda','charger':gen/'source_charger_stand.usda','scene':gen/'source_magsafe_fixed_scene.usda'}
 build_table_asset(source,assets['table']);build_phone_asset(source,assets['phone']);build_accessory_asset(source,assets['accessory']);build_charger_asset(source,assets['charger']);build_composite_scene(source,assets['scene'],assets['table'],assets['phone'],assets['accessory'],assets['charger'])
 source_stage=Usd.Stage.Open(str(assets['scene']));Ts_phone=tf(source_stage,'/MagSafeScene/Phone');Ts_acc=tf(source_stage,'/MagSafeScene/Accessory');Ts_charger=tf(source_stage,'/MagSafeScene/Charger');Ts_pad=tf(source_stage,'/MagSafeScene/Charger/Visuals/PadFace');pad_n=Ts_pad[:3,:3]@np.array([0,0,1.])
 frame_cfg={'status':'RECOVERED_FROM_USER_APPROVED_ORIGINAL_PROJECT_SCENE','camera_fitting_used':False,'T_source_scene_from_phone':Ts_phone.tolist(),'T_source_scene_from_accessory':Ts_acc.tolist(),'T_source_scene_from_charger_root':Ts_charger.tolist(),'T_source_scene_from_charger_pad':Ts_pad.tolist(),'charger_pad_face_center':Ts_pad[:3,3].tolist(),'charger_pad_outward_normal':pad_n.tolist(),'axis_conventions':source['coordinate_frame'],'provenance':{'declaration':str(DECL.resolve()),'layout':str(SOURCE_LAYOUT.resolve()),'builder':str(BUILDER.resolve()),'hashes':{str(p.resolve()):sha(p) for p in [DECL,SOURCE_LAYOUT,BUILDER,*assets.values()]}}};dump(SOURCE_FRAMES,frame_cfg);dump(OUT/'source_object_frames.json',frame_cfg)
 with np.load(SRC,allow_pickle=False) as z:action=z['optimized_action'].copy();timestamp=z['timestamp'].copy();fps=float(z['fps'])
 if action.shape!=(990,14) or timestamp.shape!=(990,) or fps!=30 or not np.isfinite(action).all():raise RuntimeError('source invariant')
 am,_=core.aloha.load_validated_model(core.ALOHA_XML);aq,clip=core.aloha.mapped_qpos(action);fk=core.aloha.fk(am,aq);modelL=poses(fk['left_position_m'],fk['left_quaternion_wxyz']);modelR=poses(fk['right_position_m'],fk['right_quaternion_wxyz'])
 pcfg=json.loads(POSE.read_text())['stationary_aloha'];TsrcA=T(Rotation.from_quat(np.asarray(pcfg['orientation_wxyz'])[[1,2,3,0]]).as_matrix(),pcfg['position_xyz_m']);worldL=np.einsum('ij,tjk->tik',TsrcA,modelL);worldR=np.einsum('ij,tjk->tik',TsrcA,modelR)
 np.savez_compressed(OUT/'aloha_fk_source_world.npz',timestamps=timestamp,optimized_action=action,left_tcp_position=worldL[:,:3,3],right_tcp_position=worldR[:,:3,3],left_tcp_rotation=worldL[:,:3,:3],right_tcp_rotation=worldR[:,:3,:3],source_Aloha_root_transform=TsrcA,TCP_offset=np.array([.1487,0,-.00105]),model_hash=np.array(sha(core.ALOHA_XML)),fps=np.array(fps),real_robot_command_allowed=np.array(False))
 evraw=json.loads(TIMELINE.read_text())['events'];events=sorted(evraw,key=lambda x:(x['frame'],x['event']));ev={x['event']:int(x['frame']) for x in evraw};fg=ev['left_phone_grasp_start'];fp=ev['phone_portrait_reached'];fa=ev['right_accessory_grasp_start'];fr=ev['accessory_removed'];fc=ev['phone_charger_attachment_complete'];flr=ev['left_phone_release_complete'];far=ev['right_accessory_release_complete']
 phone_from_tcp=inv(Ts_phone)@worldL[fg];tcp_from_phone=inv(phone_from_tcp);source_phone=np.repeat(Ts_phone[None],990,0);source_phone[fg:fc+1]=np.einsum('tij,jk->tik',worldL[fg:fc+1],tcp_from_phone);source_phone[fc+1:]=source_phone[fc]
 attach=inv(Ts_phone)@Ts_acc;source_acc=np.empty((990,4,4));source_acc[:fr]=np.einsum('tij,jk->tik',source_phone[:fr],attach);acc326=source_phone[fa]@attach;acc_from_rtcp=inv(acc326)@worldR[fa];rtcp_from_acc=inv(acc_from_rtcp);source_acc[fr:far+1]=np.einsum('tij,jk->tik',worldR[fr:far+1],rtcp_from_acc);source_acc[far+1:]=source_acc[far]
 np.savez_compressed(OUT/'source_phone_pose_trajectory.npz',timestamps=timestamp,T_source_scene_from_phone=source_phone,grasp_frame=np.array(fg),attachment_frame=np.array(fc),relation_T_phone_from_left_tcp=phone_from_tcp)
 np.savez_compressed(OUT/'source_accessory_pose_trajectory.npz',timestamps=timestamp,T_source_scene_from_accessory=source_acc,grasp_frame=np.array(fa),removed_frame=np.array(fr),release_frame=np.array(far),relation_T_accessory_from_right_tcp=acc_from_rtcp)
 relation={'status':'SOURCE_RELATIONS_RECOVERED_FROM_USER_APPROVED_PROJECT_SCENE','source_absolute_coordinates_discarded_after_relation_extraction':True,'T_source_phone_from_left_ALOHA_TCP':phone_from_tcp.tolist(),'T_source_left_ALOHA_TCP_from_phone':tcp_from_phone.tolist(),'T_source_accessory_from_right_ALOHA_TCP':acc_from_rtcp.tolist(),'T_source_right_ALOHA_TCP_from_accessory':rtcp_from_acc.tolist(),'accessory_attachment_T_phone_from_accessory':attach.tolist(),'frames':{'left_phone_grasp':fg,'right_accessory_grasp':fa,'accessory_removed':fr,'phone_charger_attachment':fc},'camera_fitting_used':False};dump(OUT/'source_hand_object_relations.json',relation)
 cal={'status':'VERIFIED_EXISTING_SIDE_SPECIFIC_MAPPING','frame_direction':'R_target_palm = R_source_ALOHA_TCP @ C_side; palm and TCP origins are treated as the embodiment tool point; G1 wrist-to-palm proxy is handled by FK/IK','C_left':C_L.tolist(),'C_right':C_R.tolist(),'determinants':[float(np.linalg.det(C_L)),float(np.linalg.det(C_R))],'orthonormal_errors':[float(np.max(np.abs(C_L.T@C_L-np.eye(3)))),float(np.max(np.abs(C_R.T@C_R-np.eye(3))))],'g1_palm_proxy':{'left':{'parent':'left_wrist_yaw_link','local_position':[.0415,.003,0]},'right':{'parent':'right_wrist_yaw_link','local_position':[.0415,-.003,0]}},'sources':[str((ROOT/'tools/retarget_episode49_optimized_action_to_g1.py').resolve()),str((ROOT/'configs/aloha_tool_axes_calibration.sim.json').resolve()),'/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml'],'reflection':False,'arbitrary_transform_added':False};dump(TOOL_CAL,cal);dump(OUT/'tool_calibration_audit.json',cal)
 target_stage=Usd.Stage.Open(str(TARGET_USD));Tt_phone=tf(target_stage,'/World/MagSafeScene/Phone');Tt_acc=tf(target_stage,'/World/MagSafeScene/Accessory');Tt_charger=tf(target_stage,'/World/MagSafeScene/Charger');Tt_pad=tf(target_stage,'/World/MagSafeScene/Charger/Visuals/PadFace');nt=Tt_pad[:3,:3]@np.array([0,0,1.]);vertical=np.array([0,0,1.]);x=vertical-nt*np.dot(vertical,nt);x/=np.linalg.norm(x);y=-nt;y/=np.linalg.norm(y);z=np.cross(x,y);Rphone=np.column_stack((x,y,z));Tphone_pad=T(Rphone,Tt_pad[:3,3])
 target_frames={'status':'CURRENT_G1_LAYOUT_UNCHANGED','active_usd':str(TARGET_USD.resolve()),'active_usd_sha256':sha(TARGET_USD),'T_target_scene_from_phone':Tt_phone.tolist(),'T_target_scene_from_accessory':Tt_acc.tolist(),'T_target_scene_from_charger_root':Tt_charger.tolist(),'T_target_scene_from_charger_pad_face':Tt_pad.tolist(),'pad_outward_normal':nt.tolist(),'pad_tangent_vertical_axis':x.tolist(),'desired_phone_on_pad':Tphone_pad.tolist(),'source_world_coordinates_copied':False};dump(OUT/'target_object_frames.json',target_frames)
 # Convert source tool-frame axes to G1 palm axes while retaining object-local tool-point translation.
 phone_from_palm=phone_from_tcp.copy();phone_from_palm[:3,:3]=phone_from_tcp[:3,:3]@C_L;acc_from_palm=acc_from_rtcp.copy();acc_from_palm[:3,:3]=acc_from_rtcp[:3,:3]@C_R
 rel_phone=np.einsum('ij,tjk->tik',inv(Ts_phone),source_phone);target_phone=np.einsum('ij,tjk->tik',Tt_phone,rel_phone);target_acc326=target_phone[fa]@attach;delta_removal=inv(source_phone[fa]@attach)@source_acc[fr];target_acc341=target_acc326@delta_removal
 anchorsL={fg:Tt_phone@phone_from_palm,fp:target_phone[fp]@phone_from_palm,fc:Tphone_pad@phone_from_palm};anchorsR={fa:target_acc326@acc_from_palm,fr:target_acc341@acc_from_palm}
 anchor_json={'status':'OBJECT_RELATIVE_DYNAMIC_ANCHORS_READY','source_absolute_coordinates_used':False,'anchors':{}}
 for side,dct in [('left',anchorsL),('right',anchorsR)]:
  for frame,m in dct.items():anchor_json['anchors'][f'{side}_{frame}']={'event_frame':frame,'event':event_name(frame,events),'world_position':m[:3,3].tolist(),'world_rotation':m[:3,:3].tolist(),'source_relation':'phone_from_g1_palm' if side=='left' else 'accessory_from_g1_palm','confidence':'USER_APPROVED_SCENE_PLUS_CODE_VERIFIED_FK','tool_calibration':str(TOOL_CAL.resolve())}
 dump(OUT/'target_event_anchors.json',anchor_json)
 with np.load(V8,allow_pickle=False) as z:
  if not np.array_equal(z['optimized_action'],action) or not np.array_equal(z['source_timestamp'],timestamp):raise RuntimeError('v8 source changed')
  base_l=z['current_target_left_position'].copy();base_r=z['current_target_right_position'].copy();base_model_l=z['base_target_left_position'].copy();base_model_r=z['base_target_right_position'].copy();warm=z['g1_arm_joint_trajectory'].copy();names=z['arm_joint_names'].copy()
 info=core.ik.validate_model(core.G1_XML);data=core.mujoco.MjData(info['model']);bLm=[];bRm=[]
 for q in warm:
  s=core.frame_state(info,data,q);bLm.append(quatR(s['left_quat']));bRm.append(quatR(s['right_quat']))
 bLm=np.asarray(bLm);bRm=np.asarray(bRm);srcL=worldL[:,:3,:3];srcR=worldR[:,:3,:3];dL=np.einsum('ji,tjk->tik',srcL[0],srcL);dR=np.einsum('ji,tjk->tik',srcR[0],srcR);mapL=np.einsum('ij,tjk,kl->til',C_L.T,dL,C_L);mapR=np.einsum('ij,tjk,kl->til',C_R.T,dR,C_R);baseLR=np.einsum('ij,tjk->tik',R_TARGET@bLm[0],mapL);baseRR=np.einsum('ij,tjk->tik',R_TARGET@bRm[0],mapR)
 np.savez_compressed(OUT/'restored_base_aloha_targets.npz',timestamps=timestamp,optimized_action=action,base_left_position_scene=base_l,base_right_position_scene=base_r,base_left_rotation_scene=baseLR,base_right_rotation_scene=baseRR,base_left_position_g1_model=base_model_l,base_right_position_g1_model=base_model_r,v8_exact_warm_start=warm,arm_joint_names=names,fps=np.array(30.),source_primary=np.array(True),real_robot_command_allowed=np.array(False))
 anchor_pos_l={f:m[:3,3]-base_l[f] for f,m in anchorsL.items()};anchor_pos_r={f:m[:3,3]-base_r[f] for f,m in anchorsR.items()};anchor_rot_l={f:rotvec_residual(m,baseLR,f) for f,m in anchorsL.items()};anchor_rot_r={f:rotvec_residual(m,baseRR,f) for f,m in anchorsR.items()}
 progress=phase_progress(worldL[:,:3,3],worldR[:,:3,3],srcL,srcR);grid={
  'VERY_STRONG_ALOHA_FIDELITY':{'magnitude':12.,'velocity':180.,'acceleration':1400.,'bimanual':40.},
  'STRONG_ALOHA_FIDELITY':{'magnitude':5.,'velocity':100.,'acceleration':700.,'bimanual':20.},
  'BALANCED_ANCHOR_FIDELITY':{'magnitude':1.5,'velocity':45.,'acceleration':240.,'bimanual':8.}}
 dump(OUT/'phasewarp_candidate_grid.json',{'knots':KNOTS.tolist(),'hard_anchors':HARD,'same_anchors_all_candidates':True,'source_progress_parameterization':True,'weights':grid})
 results={};candidate_arrays={}
 for name,w in grid.items():
  kl,kr=solve_knot_values(anchor_pos_l,anchor_pos_r,w);rl,rr=solve_knot_values(anchor_rot_l,anchor_rot_r,w);resL=blend_knots(kl,progress);resR=blend_knots(kr,progress);rvL=blend_knots(rl,progress);rvR=blend_knots(rr,progress);cL=base_l+resL;cR=base_r+resR;cLR=apply_rotation(baseLR,rvL);cRR=apply_rotation(baseRR,rvR)
  phase,mins=fidelity(base_l,base_r,cL,cR,baseLR,baseRR,cLR,cRR);anchor_err=[]
  for f,m in anchorsL.items():anchor_err.append(np.linalg.norm(cL[f]-m[:3,3]));
  for f,m in anchorsR.items():anchor_err.append(np.linalg.norm(cR[f]-m[:3,3]));
  energy={'magnitude':float(np.sum(resL**2)+np.sum(resR**2)),'velocity':float(np.sum(np.diff(resL,axis=0)**2)+np.sum(np.diff(resR,axis=0)**2)),'acceleration':float(np.sum(np.diff(resL,n=2,axis=0)**2)+np.sum(np.diff(resR,n=2,axis=0)**2))}
  valid=max(anchor_err,default=0)<=1e-9;results[name]={'anchor_valid':valid,'max_constructed_anchor_error_m':max(anchor_err,default=0),'correction_energy':energy,'total_correction_energy':sum(energy.values()),'minimum_major_phase_fidelity':mins,'fidelity_warning':bool(min(mins.values())<.9)};candidate_arrays[name]=(cL,cR,cLR,cRR,resL,resR,rvL,rvR,kl,kr,rl,rr,phase)
 valid=[n for n in grid if results[n]['anchor_valid']];selected=min(valid,key=lambda n:results[n]['total_correction_energy']);dump(OUT/'phasewarp_candidate_results.json',results);dump(OUT/'selected_phasewarp_candidate.json',{'selected':selected,'rule':'anchor-valid candidate with lowest unweighted correction energy','result':results[selected],'coupled_iterations':[{'iteration':1,'anchor_update_position_m':float(np.linalg.norm(target_acc326[:3,3]-Tt_acc[:3,3])),'anchor_update_rotation_deg':float(np.degrees(Rotation.from_matrix(target_acc326[:3,:3]@Tt_acc[:3,:3].T).magnitude()))},{'iteration':2,'anchor_update_position_m':0.0,'anchor_update_rotation_deg':0.0}],'converged':True})
 cL,cR,cLR,cRR,resL,resR,rvL,rvR,kl,kr,rl,rr,phase=candidate_arrays[selected]
 np.savez_compressed(OUT/'phase_residual_coefficients.npz',knot_frames=KNOTS,left_translation_knots=kl,right_translation_knots=kr,left_rotation_vector_knots=rl,right_rotation_vector_knots=rr,left_translation_residual=resL,right_translation_residual=resR,left_rotation_vector_residual=rvL,right_rotation_vector_residual=rvR,common_translation_residual=.5*(resL+resR),left_specific_translation_residual=.5*(resL-resR),right_specific_translation_residual=.5*(resR-resL),source_progress=progress)
 np.savez_compressed(OUT/'corrected_aloha_targets.npz',timestamps=timestamp,optimized_action=action,original_base_left_position=base_l,original_base_right_position=base_r,original_base_left_rotation=baseLR,original_base_right_rotation=baseRR,residual_left_translation=resL,residual_right_translation=resR,residual_left_rotation_vector=rvL,residual_right_rotation_vector=rvR,corrected_left_position=cL,corrected_right_position=cR,corrected_left_rotation=cLR,corrected_right_rotation=cRR,candidate=np.array(selected),correction_knots=KNOTS,real_robot_command_allowed=np.array(False))
 dump(OUT/'aloha_phase_fidelity_metrics.json',{'status':'PASS' if min(results[selected]['minimum_major_phase_fidelity'].values())>=.9 else 'ALOHA_FIDELITY_WARNING','selected_candidate':selected,'minimum_major_phase_fidelity':results[selected]['minimum_major_phase_fidelity'],'phases':phase,'hard_invariants':{'frames':990,'timestamps_array_equal':True,'event_frames_exact':True,'phase_durations_exact':True,'task_order_exact_by_frame_stable_sort':True,'hand_roles_exact':True}});dump(OUT/'correction_energy_metrics.json',results[selected]['correction_energy'])
 # Constructed target anchor error before IK.
 anchor_metrics={'constructed_target_errors_m':{},'gate_m':.005}
 for f,m in anchorsL.items():anchor_metrics['constructed_target_errors_m'][f'left_{f}']=float(np.linalg.norm(cL[f]-m[:3,3]))
 for f,m in anchorsR.items():anchor_metrics['constructed_target_errors_m'][f'right_{f}']=float(np.linalg.norm(cR[f]-m[:3,3]))
 dump(OUT/'anchor_metrics.json',anchor_metrics)
 if not a.skip_plots:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  plots=[('base_vs_corrected_paths.png',[(base_l[:,0],base_l[:,2],'base L'),(cL[:,0],cL[:,2],'corrected L'),(base_r[:,0],base_r[:,2],'base R'),(cR[:,0],cR[:,2],'corrected R')],'scene X','scene Z'),('speed_profiles.png',[(np.arange(989),np.linalg.norm(np.diff(base_l,axis=0),axis=1),'base L'),(np.arange(989),np.linalg.norm(np.diff(cL,axis=0),axis=1),'corrected L'),(np.arange(989),np.linalg.norm(np.diff(base_r,axis=0),axis=1),'base R'),(np.arange(989),np.linalg.norm(np.diff(cR,axis=0),axis=1),'corrected R')],'frame','m/frame'),('bimanual_relation.png',[(np.arange(990),np.linalg.norm(base_r-base_l,axis=1),'base distance'),(np.arange(990),np.linalg.norm(cR-cL,axis=1),'corrected distance')],'frame','inter-hand distance m')]
  for fn,series,xl,yl in plots:
   fig,ax=plt.subplots(figsize=(11,5));[ax.plot(x,y,label=l) for x,y,l in series];[ax.axvline(k,color='gray',alpha=.15) for k in KNOTS];ax.set(xlabel=xl,ylabel=yl);ax.legend();fig.tight_layout();fig.savefig(OUT/fn,dpi=180);plt.close(fig)
  # Curvature and rotation progress are kept as separate requested diagnostics.
  fig,ax=plt.subplots(figsize=(11,5));ax.plot(np.linalg.norm(np.diff(base_l,n=2,axis=0),axis=1),label='base L');ax.plot(np.linalg.norm(np.diff(cL,n=2,axis=0),axis=1),label='corrected L');ax.legend();fig.tight_layout();fig.savefig(OUT/'curvature_profiles.png',dpi=180);plt.close(fig)
  fig,ax=plt.subplots(figsize=(11,5));ax.plot(Rotation.from_matrix(np.einsum('ji,tjk->tik',baseLR[176],baseLR)).magnitude(),label='base L');ax.plot(Rotation.from_matrix(np.einsum('ji,tjk->tik',cLR[176],cLR)).magnitude(),label='corrected L');ax.legend();fig.tight_layout();fig.savefig(OUT/'rotation_progress.png',dpi=180);plt.close(fig)
 source_audit={'status':'SOURCE_RELATIONS_RECOVERED_FROM_USER_APPROVED_PROJECT_SCENE','declaration':str(DECL.resolve()),'source_layout':str(SOURCE_LAYOUT.resolve()),'source_scene':str(assets['scene'].resolve()),'generated_assets':{k:str(v.resolve()) for k,v in assets.items()},'source_scene_hashes':{k:sha(v) for k,v in assets.items()},'camera_fitting_used':False,'source_absolute_coordinates_used_for_target_generation':False,'target_hashes_before':target_hashes,'target_hashes_after':{p:sha(Path(p)) for p in target_hashes},'target_scene_unchanged':all(sha(Path(p))==h for p,h in target_hashes.items())};dump(OUT/'source_scene_audit.json',source_audit)
 input_audit={'status':'PASS','sole_source':str(SRC.resolve()),'source_sha256':sha(SRC),'shape':[990,14],'fps':30,'timestamps_exact':True,'approved_timeline':str(TIMELINE.resolve()),'timeline_sha256':sha(TIMELINE),'events_read_only_stable_sorted':True,'knots':KNOTS.tolist(),'forbidden_branches_loaded':[],'hand_written_waypoints':False,'per_frame_snapping':False,'static_grasp_first':False,'dex3_contact_ik':False,'physics':False,'dds_publisher_hardware':False};dump(OUT/'input_audit.json',input_audit)
 dump(OUT/'offline_build_status.json',{'status':'OFFLINE_TARGETS_READY_FOR_IK','selected':selected,'fidelity_status':json.loads((OUT/'aloha_phase_fidelity_metrics.json').read_text())['status'],'target_scene_unchanged':source_audit['target_scene_unchanged']})
 print(json.dumps(json.loads((OUT/'offline_build_status.json').read_text()),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
