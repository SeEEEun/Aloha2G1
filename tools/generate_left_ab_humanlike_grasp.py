#!/usr/bin/env python3
"""Static human-like left A+B grasp search; no trajectory and no physics."""
from __future__ import annotations
import json,math,os,sys,tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/humanlike_v6_mpl')
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_left_ab_humanlike_v6';OLD=ROOT/'converted_runs/g1_left_phone_cgap_grasp/selected_left_phone_cgap_grasp.npz'
sys.path[:0]=[str(ROOT/'tools'),str(ROOT)];import retarget_episode49_optimized_action_to_g1 as epi
PHONE=np.array([.525,.07,.83075]);SIZE=np.array([.1496,.00795,.0715]);ROOTP=np.array([.44514890950197095,-.35257022755443246,.7922728583]);ROOTQ=np.array([.7071067812,0,0,.7071067812]);XLEFT=PHONE[0]-SIZE[0]/2
NAMES=['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint']
def ser(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:ser(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [ser(v) for v in x]
 return x
def dump(n,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(ser(x),indent=2)+'\n')
def body(m,d,n):i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n);return d.xpos[i].copy(),d.xmat[i].reshape(3,3).copy()
def make_scene():
 t=epi.G1_XML.read_text();a=epi.G1_XML.parent/'assets';t=t.replace('meshdir="assets"',f'meshdir="{a}"').replace('meshdir="assets/"',f'meshdir="{a}/"');p=f'<geom name="phone_collision" type="box" pos="{PHONE[0]} {PHONE[1]} {PHONE[2]}" size="{SIZE[0]/2} {SIZE[1]/2} {SIZE[2]/2}" rgba=".2 .5 .9 .5" contype="1" conaffinity="1"/>';table='<geom name="table_collision" type="box" pos=".4175 .36 .7725" size=".4175 .36 .0225" rgba=".5 .5 .5 .3" contype="1" conaffinity="1"/>';t=t.replace('<worldbody>','<worldbody>\n'+p+'\n'+table,1);td=tempfile.TemporaryDirectory();x=Path(td.name)/'m.xml';x.write_text(t);return mujoco.MjModel.from_xml_path(str(x)),td
def contacts(m,d):
 o=[]
 for c in d.contact:
  bs=[]
  for g in (c.geom1,c.geom2):bs.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or 'world')
  o.append({'pair':bs,'distance_m':float(c.dist),'position':np.asarray(c.pos)})
 return o
def main():
 OUT.mkdir(parents=True,exist_ok=True);model,tmp=make_scene();data=mujoco.MjData(model);info=epi.ik.validate_model(epi.G1_XML);ids=np.asarray(info['arm_qpos_ids'][:7]);wid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left_wrist_yaw_link');eid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left_elbow_link');sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left_shoulder_roll_link')
 with np.load(OLD) as z:qold=z['full_g1_qpos'].copy();hand=z['left_dex3_qpos'].copy();oldphone=z['phone_proxy_pose'][:3];cp=z['actual_contact_points'].copy();qnom=z['left_arm_qpos'].copy()
 base=mujoco.MjModel.from_xml_path(str(epi.G1_XML));bd=mujoco.MjData(base);bd.qpos[:]=qold;mujoco.mj_forward(base,bd);ow,owR=body(base,bd,'left_wrist_yaw_link');rel=np.array([owR.T@(x-ow) for x in cp]);midrel=rel.mean(0)
 # Old proxy axes (thickness,short,long) -> scene (Y,Z,X), proper det +1.
 Rmap=np.array([[0,0,1],[1,0,0],[0,1,0]],float);assert np.linalg.det(Rmap)>0
 qbase=qold.copy();qbase[:3]=ROOTP;qbase[3:7]=ROOTQ;qbase[ids[7:] if len(ids)>7 else []]=[]
 lo=info['joint_limits'][:7,0]+1e-4;hi=info['joint_limits'][:7,1]-1e-4
 swivels=list(range(-70,71,10));patches=[(XLEFT+x,PHONE[2]+z) for x in (.008,.012,.016) for z in (-.012,0,.012)];handrows=[];armrows=[]
 for assignment,sgn in [('A_SCREEN_B_BACK',1),('A_BACK_B_SCREEN',-1)]:
  targetaxis=np.array([0,sgn,0.]);Aoff=-.5*SIZE[1]*targetaxis;Boff=.5*SIZE[1]*targetaxis
  for px,pz in patches:
   mt=np.array([px,PHONE[1],pz]);At=mt+Aoff;Bt=mt+Boff
   handrows.append({'assignment':assignment,'patch_xz':[px,pz],'A_target':At,'B_target':Bt,'target_axis':targetaxis,'axis_error_deg':0.,'separation_m':np.linalg.norm(Bt-At),'inside_surface':bool(XLEFT<px<PHONE[0]+SIZE[0]/2 and abs(pz-PHONE[2])<SIZE[2]/2-.005),'hand_qpos':hand,'status':'PASS_ANALYTIC_DIRECT_PHONE_FRAME_TARGET'})
   # central patch only proceeds to carrier search; other patches establish the finite patch grid.
   if abs(px-(XLEFT+.012))>1e-9 or abs(pz-PHONE[2])>1e-9:continue
   for sw in swivels:
    roll=Rotation.from_rotvec(np.radians(sw)*targetaxis).as_matrix();Rw=roll@Rmap@owR;tp=mt-Rw@midrel
    x0=qnom.copy();x0[1]+=np.radians(sw)*.20;x0[2]-=np.radians(sw)*.15;x0=np.clip(x0,lo,hi)
    def fun(q):
     data.qpos[:]=qbase;data.qpos[ids]=q;hand_ids=[]
     for n,v in zip(['left_hand_thumb_0_joint','left_hand_thumb_1_joint','left_hand_thumb_2_joint','left_hand_middle_0_joint','left_hand_middle_1_joint','left_hand_index_0_joint','left_hand_index_1_joint'],hand):
      j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n);data.qpos[model.jnt_qposadr[j]]=v
     mujoco.mj_forward(model,data);wp=data.xpos[wid];WR=data.xmat[wid].reshape(3,3);ep=data.xpos[eid];rv=Rotation.from_matrix(WR.T@Rw).as_rotvec();fore=wp-ep
     return np.r_[55*(wp-tp),3.2*rv,2.0*(wp[2]-ep[2]),.35*q[4:7],.025*(q-qnom)]
    sol=least_squares(fun,x0,bounds=(lo,hi),max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9);data.qpos[:]=qbase;data.qpos[ids]=sol.x
    for n,v in zip(['left_hand_thumb_0_joint','left_hand_thumb_1_joint','left_hand_thumb_2_joint','left_hand_middle_0_joint','left_hand_middle_1_joint','left_hand_index_0_joint','left_hand_index_1_joint'],hand):j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n);data.qpos[model.jnt_qposadr[j]]=v
    mujoco.mj_forward(model,data);wp,WR=body(model,data,'left_wrist_yaw_link');ep,_=body(model,data,'left_elbow_link');sp,_=body(model,data,'left_shoulder_roll_link');fore=wp-ep;elev=abs(np.degrees(np.arctan2(fore[2],np.linalg.norm(fore[:2]))));rv=Rotation.from_matrix(WR.T@Rw).as_rotvec();marg=np.minimum(sol.x-info['joint_limits'][:7,0],info['joint_limits'][:7,1]-sol.x);cs=contacts(model,data);forbid=[c for c in cs if ('torso_link' in c['pair'] and any('shoulder' in x or 'elbow' in x or 'wrist' in x for x in c['pair'])) or ('table_collision' in c['pair'])]
    wristneutral=np.degrees(np.linalg.norm(sol.x[4:7]));armrows.append({'assignment':assignment,'elbow_swivel_deg':sw,'initial_qpos':x0,'solved_qpos':sol.x,'hand_qpos':hand,'full_qpos':data.qpos.copy(),'wrist_target_position':tp,'wrist_target_rotation':Rw,'wrist_achieved_position':wp,'wrist_achieved_rotation':WR,'wrist_position_error_m':np.linalg.norm(wp-tp),'wrist_orientation_error_deg':np.degrees(np.linalg.norm(rv)),'shoulder_position':sp,'elbow_position':ep,'wrist_position':wp,'forearm_elevation_deg':elev,'wrist_neutral_deviation_deg':wristneutral,'joint_margins':marg,'forbidden_contacts':forbid,'all_contacts':cs,'target_A':At,'target_B':Bt,'target_midpoint':mt,'target_axis':targetaxis,'solver_success':bool(sol.success)})
 dump('human_reference_pose_definition.json',{'source_image':'/tmp/codex-clipboard-wMbehn.png','constraints':{'forearm_elevation_max_deg':15,'wrist_neutral_max_deg':20,'elbow_outside_torso':True},'simulation_only':True});dump('contact_target_definition.json',{'phone_center':PHONE,'phone_size':SIZE,'left_edge_x':XLEFT,'assignments':['A_SCREEN_B_BACK','A_BACK_B_SCREEN'],'patch_grid_x_inset_m':[.008,.012,.016],'patch_grid_z_offset_m':[-.012,0,.012],'pinch_axis':'scene +/-Y','old_local_rigid_target_reused':False});dump('hand_only_results.json',{'candidate_count':len(handrows),'candidates':handrows});dump('arm_carrier_results.json',{'candidate_count':len(armrows),'candidates':[{k:ser(v) for k,v in r.items() if k not in ('full_qpos','wrist_target_rotation','wrist_achieved_rotation','all_contacts')} for r in armrows]});dump('elbow_swivel_candidates.json',{'range_deg':[-70,70],'step_deg':10,'candidate_count_per_assignment':15})
 # Carrier gate before coupled contact: exact same 3 mm/10 deg plus human posture.
 valid=[r for r in armrows if r['wrist_position_error_m']<=.003 and r['wrist_orientation_error_deg']<=10 and r['forearm_elevation_deg']<=15 and r['wrist_neutral_deviation_deg']<=20 and not r['forbidden_contacts'] and np.min(r['joint_margins'])>=0]
 best=min(armrows,key=lambda r:(r['wrist_position_error_m']>.003,r['wrist_orientation_error_deg']>10,bool(r['forbidden_contacts']),r['forearm_elevation_deg']>15,r['wrist_neutral_deviation_deg']>20,r['wrist_position_error_m']+np.radians(r['wrist_orientation_error_deg'])))
 if not valid:
  if best['wrist_position_error_m']>.003 or best['wrist_orientation_error_deg']>10:status='BLOCKED_HUMANLIKE_ARM_CARRIER'
  elif best['forbidden_contacts']:status='BLOCKED_COLLISION'
  elif best['forearm_elevation_deg']>15:status='BLOCKED_FOREARM_POSTURE'
  else:status='BLOCKED_WRIST_NEUTRALITY'
  selected=None
 else:status='LEFT_AB_HUMANLIKE_GRASP_READY_FOR_VISUAL_APPROVAL';selected=min(valid,key=lambda r:(r['wrist_position_error_m'],r['forearm_elevation_deg'],r['wrist_neutral_deviation_deg']))
 dump('coupled_results.json',{'status':'NOT_RUN_ARM_CARRIER_GATE_FAILED' if selected is None else 'STATIC_CARRIER_READY_CONTACT_REVALIDATION_REQUIRED','new_trajectory':False});dump('selected_humanlike_left_grasp.json',{'status':status,'selected':None if selected is None else {k:ser(v) for k,v in selected.items() if k not in ('full_qpos','wrist_target_rotation','wrist_achieved_rotation','all_contacts')},'best_failed':{k:ser(v) for k,v in best.items() if k not in ('full_qpos','wrist_target_rotation','wrist_achieved_rotation','all_contacts')}});dump('collision_breakdown.json',{'status':status,'best_forbidden_contacts':best['forbidden_contacts'],'all_contact_count':len(best['all_contacts'])})
 chosen=selected if selected is not None else best
 np.savez_compressed(OUT/('selected_humanlike_left_grasp.npz' if selected is not None else 'best_failed_humanlike_left_grasp.npz'),full_qpos=chosen['full_qpos'],left_arm_qpos=chosen['solved_qpos'],left_dex3_qpos=hand,wrist_target_position=chosen['wrist_target_position'],wrist_target_rotation=chosen['wrist_target_rotation'],wrist_achieved_position=chosen['wrist_achieved_position'],wrist_achieved_rotation=chosen['wrist_achieved_rotation'],shoulder_position=chosen['shoulder_position'],elbow_position=chosen['elbow_position'],wrist_position=chosen['wrist_position'],target_A=chosen['target_A'],target_B=chosen['target_B'],forearm_elevation_deg=chosen['forearm_elevation_deg'],wrist_neutral_deviation_deg=chosen['wrist_neutral_deviation_deg'],elbow_swivel_deg=chosen['elbow_swivel_deg'],diagnostic_only=True,accepted=selected is not None)
 report=f'''# Human-like static left A+B grasp\n\nStatus: `{status}`\n\nDirect phone-frame targets use scene +/-Y; no prior incorrect rigid target was reused. Hand-only grid: {len(handrows)}. Arm carrier candidates: {len(armrows)}. Best wrist error {best['wrist_position_error_m']*1000:.3f} mm / {best['wrist_orientation_error_deg']:.3f} deg; forearm {best['forearm_elevation_deg']:.3f} deg; wrist-neutral {best['wrist_neutral_deviation_deg']:.3f} deg; forbidden contacts {len(best['forbidden_contacts'])}. No trajectory or physics.\n''';(OUT/'report.md').write_text(report);(OUT/'commands.sh').write_text('#!/usr/bin/env bash\n# No viewer command unless selected_humanlike_left_grasp.npz exists.\n')
 print(status);print('BEST',report);return 0 if selected is not None else 2
if __name__=='__main__':raise SystemExit(main())
