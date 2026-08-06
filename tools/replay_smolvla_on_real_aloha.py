#!/usr/bin/env python3
"""Fail-closed standalone runner for offline Episode-49 ALOHA source validation.

Inspect and dry-run never import or construct a hardware robot. Hardware mode is
present for later operator review; this file's tests never exercise that path.
"""
from __future__ import annotations
import argparse,json,os,select,signal,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from aloha_source_validation_common import *

UNREVIEWED_SAFETY=ROOT/'configs/aloha_source_validation_safety.unreviewed.json'
REVIEWED_SAFETY=ROOT/'configs/aloha_source_validation_safety.reviewed.json'
LOCK=Path('/tmp/aloha_smolvla_hardware_control.lock')
PROCESS_MARKERS=('trossen_ai_data_collection_ui','control_robot.py','teleop','replay_aloha','replay_smolvla_on_real_aloha.py --mode hardware')
HARDWARE_PHRASE='ALOHA EP49 HARDWARE TEST';START_PHRASE='EXECUTE START TRANSITION';PREFLIGHT_PHRASE='ALOHA CONNECT-ONLY PREFLIGHT';HOME_OBSERVED_PHRASE='CONNECT HOME MOTION OBSERVED';DISCONNECT_PHRASE='EXECUTE VENDOR DISCONNECT'

class Blocked(RuntimeError):pass
def process_conflicts(text=None):
 if text is None:text=subprocess.run(['ps','-eo','pid=,args='],capture_output=True,text=True,check=True).stdout
 me=os.getpid();found=[]
 for line in text.splitlines():
  parts=line.strip().split(None,1)
  if len(parts)==2 and parts[0].isdigit() and int(parts[0])!=me and any(x in parts[1].lower() for x in PROCESS_MARKERS):found.append(line.strip())
 return found
class ControlLock:
 def __init__(self,path=LOCK):self.path=Path(path);self.fd=None
 def __enter__(self):
  try:self.fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(self.fd,f'{os.getpid()}\n'.encode())
  except FileExistsError:raise Blocked(f'hardware control lock exists: {self.path}')
  return self
 def __exit__(self,*_):
  if self.fd is not None:os.close(self.fd);self.path.unlink(missing_ok=True)
def parser():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('inspect','dry-run','hardware-preflight','hardware'),default='inspect');p.add_argument('--trajectory',type=Path,default=TRAJECTORY);p.add_argument('--start-frame',type=int,default=0);p.add_argument('--end-frame',type=int,default=29);p.add_argument('--playback-speed',type=float,default=.1);p.add_argument('--command-hz',type=float,default=30);p.add_argument('--gripper-policy',choices=('hold-current','trajectory'),default='hold-current');p.add_argument('--start-transition-seconds',type=float,default=5);p.add_argument('--preflight-duration',type=float,default=5);p.add_argument('--output',type=Path);p.add_argument('--lock-file',type=Path,default=LOCK)
 for f in ('enable-hardware','confirmed-ui-closed','confirmed-empty-workspace','confirmed-power-switch-ready','confirmed-operator-present','confirmed-thresholds-reviewed','confirmed-short-test-passed','confirmed-segments-passed','confirmed-no-object-full-test','confirmed-gripper-motion','confirmed-vendor-disconnect-motion'):p.add_argument('--'+f,action='store_true')
 p.add_argument('--hardware-confirmation');p.add_argument('--preflight-confirmation');p.add_argument('--start-transition-confirmation');p.add_argument('--normal-exit-policy',choices=('vendor-disconnect',),default=None);return p
def reviewed_safety(path=REVIEWED_SAFETY):
 path=Path(path)
 if not path.exists():raise Blocked(f'reviewed safety config missing: {path}')
 cfg=json.loads(path.read_text())
 if cfg.get('status')!='REVIEWED' or cfg.get('hardware_replay_allowed') is not True:raise Blocked('only status=REVIEWED with hardware_replay_allowed=true may authorize replay')
 return cfg
def threshold(cfg,name):
 value=cfg.get(name)
 if isinstance(value,dict):value=value.get('candidate_value')
 if value is None:raise Blocked(f'reviewed threshold unresolved: {name}')
 return np.asarray(value,float)
def hardware_preflight(a,report,cfg,process_text=None):
 required=('enable_hardware','confirmed_ui_closed','confirmed_empty_workspace','confirmed_power_switch_ready','confirmed_operator_present','confirmed_thresholds_reviewed')
 missing=[x for x in required if not getattr(a,x)]
 if missing:raise Blocked('missing hardware gates: '+', '.join(missing))
 if a.hardware_confirmation!=HARDWARE_PHRASE:raise Blocked('hardware confirmation phrase mismatch')
 if a.start_transition_confirmation!=START_PHRASE:raise Blocked('start transition confirmation phrase mismatch')
 if cfg.get('status')!='REVIEWED' or cfg.get('hardware_replay_allowed') is not True:raise Blocked('hardware replay requires approved reviewed config; candidate/unreviewed configs are rejected')
 if not report['sha256_matches_frozen_episode49']:raise Blocked('trajectory SHA256 mismatch')
 n=a.end_frame-a.start_frame+1
 if n>30 and not a.confirmed_short_test_passed:raise Blocked('short test PASS not confirmed')
 if a.start_frame==0 and a.end_frame==989 and not (a.confirmed_segments_passed and a.confirmed_no_object_full_test):raise Blocked('full-run prerequisites missing')
 if a.gripper_policy=='trajectory' and not a.confirmed_gripper_motion:raise Blocked('gripper motion not confirmed')
 if a.normal_exit_policy!='vendor-disconnect' or not a.confirmed_vendor_disconnect_motion:raise Blocked('vendor disconnect motion not explicitly accepted')
 conflicts=process_conflicts(process_text)
 if conflicts:raise Blocked('conflicting ALOHA process: '+' | '.join(conflicts))
def connect_only_preflight_gate(a,report,process_text=None):
 required=('enable_hardware','confirmed_ui_closed','confirmed_empty_workspace','confirmed_power_switch_ready','confirmed_operator_present')
 missing=[x for x in required if not getattr(a,x)]
 if missing:raise Blocked('missing connect-only gates: '+', '.join(missing))
 if a.preflight_confirmation!=PREFLIGHT_PHRASE:raise Blocked('connect-only preflight confirmation phrase mismatch')
 if a.normal_exit_policy!='vendor-disconnect' or not a.confirmed_vendor_disconnect_motion:raise Blocked('vendor disconnect motion not explicitly accepted')
 if not report['sha256_matches_frozen_episode49']:raise Blocked('trajectory SHA256 mismatch')
 conflicts=process_conflicts(process_text)
 if conflicts:raise Blocked('conflicting ALOHA process: '+' | '.join(conflicts))
def create_stationary_robot():
 """The only hardware import/construction site; called after every gate and lock."""
 from lerobot.common.robot_devices.robots.configs import StationaryAlohaRobotConfig
 from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
 return ManipulatorRobot(StationaryAlohaRobotConfig())
def state14(robot):
 obs=robot.capture_observation();q=np.asarray(obs['observation.state'],float)
 if q.shape!=(14,) or not np.isfinite(q).all():raise Blocked(f'invalid actual state {q.shape}')
 return q
def q_pressed(stream=sys.stdin):
 try:
  return bool(stream.isatty() and select.select([stream],[],[],0)[0] and stream.readline().strip().lower()=='q')
 except (OSError,ValueError):return False
class RuntimeSafetyMonitor:
 def __init__(self,cfg):
  self.tracking=np.asarray(threshold(cfg,'max_tracking_error'));self.tracking_duration=float(threshold(cfg,'tracking_error_duration'));self.state_timeout=float(threshold(cfg,'state_timeout'));self.overrun=float(threshold(cfg,'control_loop_overrun_threshold'));self.overrun_limit=int(threshold(cfg,'consecutive_overrun_limit'));self.tracking_elapsed=0.;self.overruns=0
 def check(self,error,state_age,period):
  if state_age>self.state_timeout:raise Blocked('STATE_TIMEOUT')
  bad=bool(np.any(np.abs(error)>self.tracking));self.tracking_elapsed=self.tracking_elapsed+period if bad else 0.
  if self.tracking_elapsed>=self.tracking_duration:raise Blocked('PERSISTENT_TRACKING_ERROR')
  self.overruns=self.overruns+1 if period-1/30>self.overrun else 0
  if self.overruns>=self.overrun_limit:raise Blocked('CONSECUTIVE_CONTROL_LOOP_OVERRUN')
def run_mock(a,action,report):
 source=action[a.start_frame:a.end_frame+1];current=source[0].copy();current[:6]+=.02;current[7:13]-=.02
 transition=minimum_jerk_transition(current,source[0],a.start_transition_seconds,a.command_hz,True)
 commands,frames,indices,sub=resample_minimum_jerk(source,30,a.playback_speed,a.command_hz,a.gripper_policy,current[[6,13]])
 commands=np.vstack((transition,commands));frames=np.r_[np.full(len(transition),-1),frames+a.start_frame];indices=np.r_[np.arange(len(transition)),indices]
 log=CommandActualLog();last=time.monotonic();actual=current.copy()
 for q,fr,ix in zip(commands,frames,indices):
  now=time.monotonic();actual=actual+.6*(q-actual);wall=time.time();log.append(timestamp_monotonic=now,timestamp_wall=wall,source_frame=fr,interpolation_index=ix,command_action_14d=q,actual_state_14d=actual.copy(),command_actual_error_14d=q-actual,left_gripper_command=q[6],right_gripper_command=q[13],left_gripper_actual=actual[6],right_gripper_actual=actual[13],control_period=now-last,state_read_latency=0.,command_send_latency=0.);last=now
 out=a.output or BASE/'hardware_trials'/('dry_run_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
 meta={'trial_id':out.name,'mode':'dry-run','offline_temporal_consensus':True,'trajectory_path':report['path'],'trajectory_sha256':report['sha256'],'source_frame_range':[a.start_frame,a.end_frame],'playback_speed':a.playback_speed,'command_hz':a.command_hz,'gripper_policy':a.gripper_policy,'interpolation':'minimum_jerk_resampling','subdivisions_per_source_interval':sub,'stop_reason':'COMPLETED_DRY_RUN','hardware_connection':'NOT PERFORMED','hardware_command':'NOT PERFORMED','mode_change':'NOT PERFORMED','created_at':datetime.now(timezone.utc).isoformat(),'source_metrics':motion_metrics(source,30),'interpolated_metrics':motion_metrics(commands,a.command_hz),'schema_compatibility':'ALOHA command_actual_v1; provenance/timestamp conventions aligned with g1_behavior_v1'}
 log.save(out,meta,source);return meta
def run_connect_only_preflight(a,action,report,robot_factory=create_stationary_robot,sleep_fn=time.sleep,input_fn=input):
 connect_only_preflight_gate(a,report)
 out=a.output or BASE/'hardware_preflight'/('preflight_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True,exist_ok=True)
 console=['[HARDWARE MOTION WARNING]','Vendor connect moves both follower arms to home.','NO REPLAY OR START-TRANSITION COMMAND IS SENT.','Vendor disconnect may move home/sleep.']
 with ControlLock(a.lock_file):
  robot=robot_factory();robot.connect();states=[];mono=[];wall=[];lat=[]
  try:
   if input_fn(f'Type {HOME_OBSERVED_PHRASE}: ').strip()!=HOME_OBSERVED_PHRASE:raise Blocked('connect home motion was not confirmed after connection')
   end=time.monotonic()+a.preflight_duration
   while time.monotonic()<end:
    tick=time.monotonic();q=state14(robot);done=time.monotonic();states.append(q);mono.append(done);wall.append(time.time());lat.append(done-tick);sleep_fn(max(0,1/a.command_hz-(time.monotonic()-tick)))
  finally:
   if input_fn(f'Type {DISCONNECT_PHRASE}: ').strip()!=DISCONNECT_PHRASE:raise Blocked('vendor disconnect not confirmed; operator must manage connected process')
   robot.disconnect()
 states=np.asarray(states);mono=np.asarray(mono);wall=np.asarray(wall);lat=np.asarray(lat);period=np.diff(mono);current=states[-1];delta=action[0]-current;transition=minimum_jerk_transition(current,action[0],a.start_transition_seconds,a.command_hz,True);metrics=motion_metrics(transition,a.command_hz)
 np.savez_compressed(out/'actual_state.npz',timestamp_monotonic=mono,timestamp_wall=wall,observation_state=states,state_read_latency=lat,joint_names=np.asarray(NAMES))
 with (out/'state_timing.csv').open('w') as f:
  f.write('sample,timestamp_monotonic,timestamp_wall,state_read_latency,period\n')
  for i in range(len(states)):f.write(f'{i},{mono[i]},{wall[i]},{lat[i]},{"" if i==0 else period[i-1]}\n')
 dump(out/'current_to_action0.json',{'current_state':current.tolist(),'action0':action[0].tolist(),'signed_difference':delta.tolist(),'absolute_difference':np.abs(delta).tolist(),'l2_norm':float(np.linalg.norm(delta)),'threshold_evaluation':'NOT_RUN_WITHOUT_REVIEWED_START_ERROR'})
 dump(out/'planned_start_transition_metrics.json',{'executed':False,'gripper_policy':'hold-current','duration_seconds':a.start_transition_seconds,**metrics})
 dump(out/'trial_metadata.json',{'mode':'hardware-preflight','trajectory_sha256':report['sha256'],'samples':len(states),'duration_requested_seconds':a.preflight_duration,'state_period_mean':float(period.mean()) if len(period) else None,'state_period_max':float(period.max()) if len(period) else None,'packet_gap_definition':'synchronous capture interval; no transport sequence exposed','hardware_trajectory_command':'NOT PERFORMED','start_transition_command':'NOT PERFORMED','vendor_connect_disconnect_motion':'PERFORMED_BY_OPERATOR'})
 (out/'console.log').write_text('\n'.join(console)+'\n');return out
def run_hardware(a,action,report,cfg):
 hardware_preflight(a,report,cfg)
 print('\n[HARDWARE MOTION WARNING]\nConnecting will move both follower arms to the configured home pose.\nDisconnecting may also move the arms to home/sleep poses.\n긴급 상황: 물리 전원 스위치 사용 (전원 차단 시 팔이 아래로 처질 수 있음)\n')
 with ControlLock(a.lock_file):
  robot=create_stationary_robot();robot.connect() # reviewed vendor path; moves to home
  try:
   samples=[state14(robot) for _ in range(5)];current=samples[-1]
   if np.max(np.ptp(np.asarray(samples),axis=0))>.1:raise Blocked('actual state unstable')
   source=action[a.start_frame:a.end_frame+1];transition=minimum_jerk_transition(current,source[0],a.start_transition_seconds,a.command_hz,True);commands,frames,indices,sub=resample_minimum_jerk(source,30,a.playback_speed,a.command_hz,a.gripper_policy,current[[6,13]])
   commands=np.vstack((transition,commands));frames=np.r_[np.full(len(transition),-1),frames+a.start_frame];indices=np.r_[np.arange(len(transition)),indices]
   limits={'max_step_per_channel':threshold(cfg,'max_command_step'),'max_velocity_per_channel':threshold(cfg,'max_velocity'),'max_acceleration_per_channel':threshold(cfg,'max_acceleration')};metrics=motion_metrics(commands,a.command_hz)
   for key,limit in limits.items():
    measured=np.asarray(metrics[key]);
    if np.any(measured>limit):raise Blocked(f'{key} exceeds reviewed threshold')
   start_limit=threshold(cfg,'max_start_position_error')
   if np.any(np.abs(source[0]-current)>start_limit):raise Blocked('CURRENT_TO_START_ERROR')
   monitor=RuntimeSafetyMonitor(cfg)
   paused=False
   def pause(*_):
    nonlocal paused;paused=True
   old_int=signal.signal(signal.SIGINT,pause);old_term=signal.signal(signal.SIGTERM,pause);log=CommandActualLog();last=time.monotonic();period=1/a.command_hz
   try:
    for q,fr,ix in zip(commands,frames,indices):
     if q_pressed():paused=True;log.event('PAUSED','OPERATOR_Q')
     if paused:
      if not log.events or log.events[-1]['detail']!='OPERATOR_Q':log.event('PAUSED','SIGINT_OR_SIGTERM')
      break
     tick=time.monotonic();sr=time.monotonic();actual=state14(robot);state_latency=time.monotonic()-sr
     monitor.check(q-actual,state_latency,time.monotonic()-last)
     cs=time.monotonic();robot.send_action(q);send_latency=time.monotonic()-cs;now=time.monotonic()
     log.append(timestamp_monotonic=now,timestamp_wall=time.time(),source_frame=fr,interpolation_index=ix,command_action_14d=q.copy(),actual_state_14d=actual.copy(),command_actual_error_14d=q-actual,left_gripper_command=q[6],right_gripper_command=q[13],left_gripper_actual=actual[6],right_gripper_actual=actual[13],control_period=now-last,state_read_latency=state_latency,command_send_latency=send_latency);last=now
     remaining=period-(time.monotonic()-tick)
     if remaining>0:time.sleep(remaining)
     else:log.event('CONTROL_LOOP_OVERRUN',str(-remaining))
   finally:signal.signal(signal.SIGINT,old_int);signal.signal(signal.SIGTERM,old_term)
   out=a.output or BASE/'hardware_trials'/('hardware_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
   meta={'trial_id':out.name,'mode':'hardware','trajectory_path':report['path'],'trajectory_sha256':report['sha256'],'source_frame_range':[a.start_frame,a.end_frame],'playback_speed':a.playback_speed,'command_hz':a.command_hz,'gripper_policy':a.gripper_policy,'interpolation':'minimum_jerk_resampling','subdivisions_per_source_interval':sub,'stop_reason':'PAUSED' if paused else 'COMPLETED','hardware_connection':'PERFORMED_BY_OPERATOR','hardware_command':'PERFORMED_BY_OPERATOR','created_at':datetime.now(timezone.utc).isoformat()};log.save(out,meta,source)
  finally:
   if a.normal_exit_policy=='vendor-disconnect':robot.disconnect() # explicitly accepted; may move
def main(argv=None):
 a=parser().parse_args(argv);action,fps,_=load_action(a.trajectory);report=trajectory_report(a.trajectory)
 if not (0<=a.start_frame<=a.end_frame<len(action)):raise Blocked('invalid frame range')
 print(json.dumps(report,indent=2))
 if a.mode=='inspect':return 0
 if a.mode=='dry-run':print(json.dumps(run_mock(a,action,report),indent=2));return 0
 if a.mode=='hardware-preflight':print(run_connect_only_preflight(a,action,report));return 0
 cfg=reviewed_safety()
 hardware_preflight(a,report,cfg);return run_hardware(a,action,report,cfg) or 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except Blocked as e:print(f'BLOCKED: {e}',file=sys.stderr);raise SystemExit(2)
