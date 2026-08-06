#!/usr/bin/env python3
"""Read-only canonical G1 behavior recorder. Default backend is offline dry-run."""
from __future__ import annotations
import argparse,copy,json,socket,sys,threading,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));from g1_behavior_schema import *
ROOT=Path('/home/jbnu/aloha_g1_dataset');XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
ARM_NAMES=['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint'];ARM_IDS=[15,16,17,18,19,20,21,22,23,24,25,26,27,28]
HAND_NAMES={s:[f'{s}_hand_{x}_joint' for x in ('thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1')] for s in ('left','right')}
class LiveReader:
 def __init__(self,interface):
  from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_,LowState_
  ChannelFactoryInitialize(0,interface);self.lock=threading.Lock();self.s={k:None for k in ('g1','left','right')};self.seq=0
  self.sub=[ChannelSubscriber('rt/lowstate',LowState_),ChannelSubscriber('rt/lf/dex3/left/state',HandState_),ChannelSubscriber('rt/lf/dex3/right/state',HandState_)]
  for sub,key in zip(self.sub,('g1','left','right')):sub.Init(lambda msg,k=key:self.put(k,msg),10)
 def put(self,k,msg):
  with self.lock:self.s[k]=(time.monotonic(),copy.deepcopy(msg));self.seq+=1
 def read(self,timeout):
  end=time.monotonic()+timeout
  while time.monotonic()<end:
   with self.lock:s=dict(self.s);seq=self.seq
   if all(v and time.monotonic()-v[0]<timeout for v in s.values()):return s,seq
   time.sleep(.002)
  raise TimeoutError('state timeout')
def cli():
 p=argparse.ArgumentParser();p.add_argument('--role',choices=('generated_executed','expert_actual'),required=True);p.add_argument('--backend',choices=('dry-run','replay','real'),default='dry-run');p.add_argument('--network-interface',default='UNKNOWN');p.add_argument('--output',type=Path);p.add_argument('--task-name',default='magsafe_phone_accessory');p.add_argument('--trial-id',default='001');p.add_argument('--fps',type=float,default=30);p.add_argument('--duration',type=float,default=2);p.add_argument('--source-target',type=Path);p.add_argument('--operator-note',default='');p.add_argument('--success-label',choices=('full_success','partial_success','failure','unlabeled'),default='unlabeled');p.add_argument('--inspect',type=Path);p.add_argument('--dry-run',action='store_true');p.add_argument('--replay-input',type=Path);return p.parse_args()
def main():
 a=cli()
 if a.inspect:
  ar,me=load_behavior(a.inspect);print(json.dumps({'metadata':me,'summary':summarize_behavior(ar,me),'fields':{k:list(v.shape) for k,v in ar.items()}},indent=2));return 0
 if a.dry_run:a.backend='dry-run'
 print('\nREAD-ONLY G1 BEHAVIOR RECORDER\nTHIS PROGRAM DOES NOT COMMAND THE ROBOT\n')
 out=a.output or ROOT/f"outputs/behavior_comparison/{a.role}/{a.role}_trial_{a.trial_id}.npz"
 source=None
 if a.source_target:source,_=load_behavior(a.source_target)
 if a.backend=='replay':
  if not a.replay_input:raise ValueError('--replay-input required')
  r,_=load_behavior(a.replay_input);source=r
 import mujoco;m=mujoco.MjModel.from_xml_path(str(XML));base=m.key_qpos[0].copy();n=max(2,int(round(a.duration*a.fps)));ts=np.arange(n)/a.fps
 if source:
  sk='actual_full_qpos' if 'actual_full_qpos' in source else 'target_full_qpos';x=np.linspace(0,len(source[sk])-1,n);lo=np.floor(x).astype(int);hi=np.minimum(lo+1,len(source[sk])-1);w=x-lo;full=source[sk][lo]*(1-w[:,None])+source[sk][hi]*w[:,None]
 else:full=np.repeat(base[None],n,axis=0)
 timeout=np.zeros(n,bool);seq=np.arange(n);mode=np.zeros((n,2),int);wall=np.asarray([now() for _ in range(n)])
 if a.backend=='real':
  reader=LiveReader(a.network_interface);full=[];timeout=[];seq=[];mode=[];wall=[];start=time.monotonic()
  arm_addr=[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,x)]) for x in ARM_NAMES];hand_addr={s:[int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,x)]) for x in HAND_NAMES[s]] for s in HAND_NAMES}
  while time.monotonic()-start<a.duration:
   tick=time.monotonic()
   try:s,sq=reader.read(1/a.fps*2);q=base.copy();q[arm_addr]=[s['g1'][1].motor_state[i].q for i in ARM_IDS];q[hand_addr['left']]=[s['left'][1].motor_state[i].q for i in range(7)];q[hand_addr['right']]=[s['right'][1].motor_state[i].q for i in range(7)];full.append(q);timeout.append(False);seq.append(sq);mode.append([getattr(s['g1'][1],'mode_machine',0),getattr(s['g1'][1],'mode_pr',0)]);wall.append(now())
   except TimeoutError:timeout.append(True)
   time.sleep(max(0,1/a.fps-(time.monotonic()-tick)))
  full=np.asarray(full);n=len(full);ts=np.arange(n)/a.fps;timeout=np.asarray(timeout[:n]);seq=np.asarray(seq);mode=np.asarray(mode);wall=np.asarray(wall)
 names=np.array(['']*m.nq,dtype='U64')
 for j in range(m.njnt):
  nm=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j);ad=m.jnt_qposadr[j]
  if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
   for k,s in enumerate(('x','y','z','qw','qx','qy','qz')):names[ad+k]=f'{nm}:{s}'
  else:names[ad]=nm
 def take(ns):return full[:,[list(names).index(x) for x in ns]]
 phase=lambda s: source[f'{s}_phase'][np.minimum(np.rint(np.linspace(0,len(source[f"{s}_phase"])-1,n)).astype(int),len(source[f'{s}_phase'])-1)] if source else np.repeat('UNKNOWN',n)
 arr={'timestamps':ts,'fps':np.array(a.fps),'full_joint_names':names,'arm_joint_names':np.asarray(ARM_NAMES),'left_dex3_joint_names':np.asarray(HAND_NAMES['left']),'right_dex3_joint_names':np.asarray(HAND_NAMES['right']),'left_phase':phase('left'),'right_phase':phase('right'),'actual_full_qpos':full,'actual_arm_qpos':take(ARM_NAMES),'actual_left_dex3_qpos':take(HAND_NAMES['left']),'actual_right_dex3_qpos':take(HAND_NAMES['right']),'valid_frame_mask':~timeout,'state_timeout_mask':timeout,'interpolation_mask':np.zeros(n,bool),'monotonic_timestamp':ts,'wall_clock_timestamp':wall,'packet_sequence':seq,'robot_mode_state':mode};arr.update(compute_behavior_fk(full,XML))
 if a.role=='generated_executed' and source is not None:arr['target_full_qpos']=full.copy() if a.backend!='real' else source['target_full_qpos'][:n];arr['target_tracking_error']=arr['actual_full_qpos']-arr['target_full_qpos'];arr['source_target_frame']=np.arange(n)
 synthetic=a.backend!='real';meta={'schema_version':SCHEMA,'behavior_id':f'{a.role}_trial_{a.trial_id}','behavior_role':a.role,'task_name':a.task_name,'robot':'Unitree G1','hand':'Dex3','source_type':f'readonly_recorder_{a.backend}','source_path':str(a.replay_input or a.source_target or ''),'created_at':now(),'fps':a.fps,'frame_count':n,'duration_sec':float(ts[-1]),'coordinate_frame':'g1_base_fixed_mujoco_world_frame','joint_units':'radian','orientation_representation':'quaternion_wxyz','execution_status':'synthetic' if synthetic else 'executed','is_synthetic':synthetic,'valid_for_paper_result':False if synthetic else True,'primitive_source':'simulation_placeholder' if synthetic else 'real_recorded','notes':a.operator_note,'success_label':a.success_label,'task_trial_id':a.trial_id,'source_machine':socket.gethostname(),'network_interface':a.network_interface,'state_topics':{'g1':'rt/lowstate','left':'rt/lf/dex3/left/state','right':'rt/lf/dex3/right/state'},'topic_verification':'code_verified_but_not_runtime_verified','sample_rate_hz':a.fps,'state_timeout_count':int(timeout.sum()),'packet_loss_estimate':'UNKNOWN_WITHOUT_TRANSPORT_SEQUENCE'};save_behavior(out,arr,meta);print(json.dumps({'output':str(out),'backend':a.backend,'frames':n,'real_backend_executed':a.backend=='real'},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
