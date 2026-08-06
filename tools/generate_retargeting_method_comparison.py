#!/usr/bin/env python3
"""Generate three controlled offline ALOHA->G1 retargeting methods."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
import retarget_episode49_consensus_relative_bimanual_to_g1 as rel
import retarget_episode49_optimized_action_to_g1 as core
OUT=ROOT/'outputs/retargeting_method_comparison';SOURCE=core.SOURCE;PROPOSED=rel.OUT;FULL=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz';XML=core.G1_XML
METHODS=('relative_temporal_proposed','relative_framewise','relative_no_workspace_scale')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def cfg_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def targets(fk,start,scale):
 R=core.align_mod.make_align_rotation(core.ALIGN_RPY);al=fk['left_position_m'];ar=fk['right_position_m'];am=.5*(al+ar);av=ar-al;tm=start['midpoint']+scale*((am-am[0])@R.T);tv=start['relative']+scale*((av-av[0])@R.T)
 return {'lp':tm-.5*tv,'rp':tm+.5*tv,'lr':np.repeat(np.eye(3)[None],len(tm),axis=0),'rr':np.repeat(np.eye(3)[None],len(tm),axis=0),'target_midpoint':tm,'target_relative':tv}
def compose(method,arm,tar,ev,runtime,config,base):
 with np.load(FULL,allow_pickle=False) as z:f={k:z[k] for k in z.files}
 full=f['full_qpos'].copy();ids=[list(f['full_joint_names']).index(n) for n in f['arm_joint_names']];full[:,ids]=arm;f['full_qpos']=full;f['arm_qpos']=arm
 d=OUT/'trajectories'/method;d.mkdir(parents=True,exist_ok=True);ch=cfg_hash(config)
 np.savez_compressed(d/'arm_trajectory.npz',timestamps=f['timestamps'],fps=f['fps'],arm_qpos=arm,arm_joint_names=f['arm_joint_names'],target_left_wrist_position=tar['lp'],target_right_wrist_position=tar['rp'],achieved_left_wrist_position=ev['lp'],achieved_right_wrist_position=ev['rp'],ik_success=(ev['le']<=.005)&(ev['re']<=.005),solver_runtime_sec=np.array(runtime),input_action_hash=np.array(sha(SOURCE)),method_config_hash=np.array(ch))
 np.savez_compressed(d/'full_trajectory_placeholder_dex3.npz',**f)
 dump(d/'method_config.json',config|{'method_config_hash':ch,'input_action_hash':sha(SOURCE),'proposed_arm_hash':sha(PROPOSED),'placeholder_warning':'PLACEHOLDER DEX3; FULL-HAND COLLISION RESULT IS DIAGNOSTIC ONLY'})
 dump(d/'solver_log.json',{'runtime_sec':runtime,'warm_start':config['warm_start'],'status':'existing_frozen' if method==METHODS[0] else 'computed_offline'})
 return d
def main():
 global OUT
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--temporal-iterations',type=int,default=8);p.add_argument('--force',action='store_true');a=p.parse_args();OUT=a.output_dir
 raw,timestamp=core.load_source(SOURCE,None);amodel,_=core.aloha.load_validated_model(core.ALOHA_XML);aq,_=core.aloha.mapped_qpos(raw);fk=core.aloha.fk(amodel,aq);info=core.ik.validate_model(XML);start=rel.load_natural_start(rel.NATURAL_START,info)
 with np.load(PROPOSED,allow_pickle=False) as z:proposed=z['g1_arm_joint_trajectory'].astype(float);pt={'lp':z['g1_target_left_position'],'rp':z['g1_target_right_position'],'lr':np.repeat(np.eye(3)[None],len(proposed),0),'rr':np.repeat(np.eye(3)[None],len(proposed),0)};pev={'lp':z['g1_achieved_left_position'],'rp':z['g1_achieved_right_position'],'le':np.linalg.norm(z['g1_achieved_left_position']-z['g1_target_left_position'],axis=1),'re':np.linalg.norm(z['g1_achieved_right_position']-z['g1_target_right_position'],axis=1)}
 common={'source':str(SOURCE),'frames':len(raw),'fps':core.FPS,'axis_alignment_rpy_deg':core.ALIGN_RPY.tolist(),'initial_anchor_source':str(rel.NATURAL_START),'joint_limits_source':str(XML),'orientation_weight':0.0,'collision_objective':False,'manual_tuning':False}
 compose(METHODS[0],proposed,pt,pev,None,common|{'method':METHODS[0],'workspace_scale':core.SCALE,'solver':'whole-trajectory sparse Gauss-Newton','temporal_velocity_weight':.018,'temporal_acceleration_weight':.030,'joint_regularization_weight':.001,'bimanual_relative_weight':.80,'warm_start':'framewise continuation seed'},None)
 tar=targets(fk,start,core.SCALE);t=time.perf_counter();frame=core.position_seed(info,tar,start['arm_q']);rt=time.perf_counter()-t;fev=core.evaluate(info,tar,frame);compose(METHODS[1],frame,tar,fev,rt,common|{'method':METHODS[1],'workspace_scale':core.SCALE,'solver':'per-frame solve_frame','temporal_velocity_weight':0.0,'temporal_acceleration_weight':0.0,'joint_regularization_weight':'solve_frame existing','bimanual_relative_weight':'solve_frame existing','warm_start':'previous frame solution; objective has no cross-frame residual'},None)
 tar1=targets(fk,start,1.0);t=time.perf_counter();seed=core.position_seed(info,tar1,start['arm_q']);noscale=core.temporal_solve(info,tar1,seed,start['arm_q'],0.0,a.temporal_iterations);rt=time.perf_counter()-t;nev=core.evaluate(info,tar1,noscale);compose(METHODS[2],noscale,tar1,nev,rt,common|{'method':METHODS[2],'workspace_scale':1.0,'solver':'whole-trajectory sparse Gauss-Newton','temporal_velocity_weight':.018,'temporal_acceleration_weight':.030,'joint_regularization_weight':.001,'bimanual_relative_weight':.80,'warm_start':'framewise continuation seed'},None)
 unavailable=OUT/'trajectories/absolute_temporal';unavailable.mkdir(parents=True,exist_ok=True);dump(unavailable/'method_config.json',{'method':'absolute_temporal','status':'NOT_AVAILABLE','reason':'No validated absolute ALOHA-to-G1 coordinate-frame mapping exists; not fabricated.'})
 freeze={'status':'proposed_current_frozen','source_action':str(SOURCE),'source_action_sha256':sha(SOURCE),'proposed_arm':str(PROPOSED),'proposed_arm_sha256':sha(PROPOSED),'exact_array_reproduction':bool(np.array_equal(np.load(OUT/'trajectories'/METHODS[0]/'arm_trajectory.npz')['arm_qpos'],proposed))};dump(OUT/'proposed_freeze.json',freeze);print(json.dumps(freeze,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
