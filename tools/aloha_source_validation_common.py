"""Shared, hardware-inert primitives for Episode-49 ALOHA source validation."""
from __future__ import annotations
import csv, hashlib, json, os, time
from pathlib import Path
import numpy as np

ROOT=Path('/home/jbnu/aloha_g1_dataset')
TRAJECTORY=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
BASE=ROOT/'outputs/aloha_source_validation/episode49'
DATASET_INFO=ROOT/'lerobot_magsafe_50_cam_high_v3/meta/info.json'
NAMES=[*(f'left_joint_{i}' for i in range(7)),*(f'right_joint_{i}' for i in range(7))]
SIDES=['left']*7+['right']*7
SEMANTICS=[*(f'arm_vendor_ordinal_{i}' for i in range(6)),'gripper',*(f'arm_vendor_ordinal_{i}' for i in range(6)),'gripper']
EXPECTED_SHA256='a7f5543e07e315d59f52004dab48423a4ee52dfcbafb9b6d5d1a731fcbd3694c'

def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');return path
def sha(path):
 h=hashlib.sha256();
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_action(path=TRAJECTORY):
 with np.load(path,allow_pickle=False) as z:
  keys=list(z.files)
  if 'optimized_action' not in z:raise ValueError('optimized_action missing')
  a=np.asarray(z['optimized_action']);fps=float(z['fps']) if 'fps' in z else 30.
 if a.shape!=(990,14):raise ValueError(f'optimized_action shape must be (990,14), got {a.shape}')
 if not np.isfinite(a).all():raise ValueError('optimized_action contains NaN/Inf')
 if fps<=0 or not np.isfinite(fps):raise ValueError(f'invalid fps {fps}')
 return a.astype(float),fps,keys
def trajectory_report(path=TRAJECTORY):
 a,fps,keys=load_action(path); d=np.diff(a,axis=0); v=d*fps; acc=np.diff(a,n=2,axis=0)*fps*fps
 i,j=np.unravel_index(np.argmax(np.abs(d)),d.shape)
 return {'path':str(Path(path).resolve()),'sha256':sha(path),'sha256_matches_frozen_episode49':sha(path)==EXPECTED_SHA256,
  'keys':keys,'shape':list(a.shape),'dtype':str(a.dtype),'finite':True,'fps':fps,'duration_seconds':len(a)/fps,
  'channel_order':NAMES,'units':['rad']*6+['m']+['rad']*6+['m'],
  'max_step':float(np.max(np.abs(d))),'max_step_frame_pair':[int(i),int(i+1)],'max_step_channel':int(j),
  'max_velocity':float(np.max(np.abs(v))),'max_acceleration':float(np.max(np.abs(acc))),
  'gripper_ranges':{'left':[float(a[:,6].min()),float(a[:,6].max())],'right':[float(a[:,13].min()),float(a[:,13].max())]}}
def minimum_jerk_transition(start,target,duration,command_hz,hold_gripper=True):
 start=np.asarray(start,float);target=np.asarray(target,float).copy()
 if start.shape!=(14,) or target.shape!=(14,):raise ValueError('start/target must be 14D')
 if hold_gripper:target[[6,13]]=start[[6,13]]
 n=max(2,int(round(duration*command_hz))+1);u=np.linspace(0,1,n);s=10*u**3-15*u**4+6*u**5
 return start+(target-start)*s[:,None]
def resample_minimum_jerk(source,source_fps,playback_speed,command_hz,gripper_policy='hold-current',current_grippers=None):
 source=np.asarray(source,float)
 if source.ndim!=2 or source.shape[1]!=14 or len(source)<1:raise ValueError('source must be Nx14')
 if playback_speed<=0 or command_hz<=0:raise ValueError('rates must be positive')
 subdivisions=max(1,int(round(command_hz/(source_fps*playback_speed))))
 rows=[];frames=[];indices=[]
 for frame in range(max(1,len(source)-1)):
  q0=source[frame];q1=source[min(frame+1,len(source)-1)]
  for k in range(subdivisions if len(source)>1 else 1):
   u=k/subdivisions;s=10*u**3-15*u**4+6*u**5;rows.append(q0+(q1-q0)*s);frames.append(frame);indices.append(k)
 if len(source)>1:rows.append(source[-1].copy());frames.append(len(source)-1);indices.append(0)
 out=np.asarray(rows,float)
 if gripper_policy=='hold-current':
  if current_grippers is None:current_grippers=source[0,[6,13]]
  out[:,[6,13]]=np.asarray(current_grippers,float)
 elif gripper_policy!='trajectory':raise ValueError('unknown gripper policy')
 return out,np.asarray(frames),np.asarray(indices),subdivisions
def motion_metrics(q,command_hz):
 q=np.asarray(q,float);d=np.diff(q,axis=0);v=d*command_hz;acc=np.diff(q,n=2,axis=0)*command_hz**2
 z=np.zeros(14)
 return {'max_step_per_channel':np.max(np.abs(d),axis=0).tolist() if len(d) else z.tolist(),
  'max_velocity_per_channel':np.max(np.abs(v),axis=0).tolist() if len(v) else z.tolist(),
  'max_acceleration_per_channel':np.max(np.abs(acc),axis=0).tolist() if len(acc) else z.tolist(),
  'max_step':float(np.max(np.abs(d))) if len(d) else 0.0}
class CommandActualLog:
 FIELDS=('timestamp_monotonic','timestamp_wall','source_frame','interpolation_index','command_action_14d','actual_state_14d','command_actual_error_14d','left_gripper_command','right_gripper_command','left_gripper_actual','right_gripper_actual','control_period','state_read_latency','command_send_latency')
 def __init__(self):self.rows=[];self.events=[]
 def append(self,**row):
  missing=set(self.FIELDS)-set(row)
  if missing:raise ValueError(f'missing log fields: {sorted(missing)}')
  self.rows.append(row)
 def event(self,kind,detail=''):self.events.append({'timestamp_monotonic':time.monotonic(),'event':kind,'detail':detail})
 def save(self,out,metadata,trajectory):
  out=Path(out);out.mkdir(parents=True,exist_ok=True);arrays={k:np.asarray([r[k] for r in self.rows]) for k in self.FIELDS}
  np.savez_compressed(out/'command_actual_log.npz',**arrays)
  with (out/'command_actual_log.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(self.FIELDS)
   for r in self.rows:w.writerow([json.dumps(np.asarray(r[k]).tolist()) if np.asarray(r[k]).ndim else r[k] for k in self.FIELDS])
  dump(out/'trial_metadata.json',metadata)
  with (out/'safety_events.jsonl').open('w') as f:
   for e in self.events:f.write(json.dumps(e)+'\n')
  np.savez_compressed(out/'trajectory_used.npz',optimized_action=np.asarray(trajectory),source_sha256=np.asarray(metadata['trajectory_sha256']))
  (out/'run_report.md').write_text('# ALOHA source validation run\n\n```json\n'+json.dumps(metadata,indent=2)+'\n```\n')
  return out
def dataset_names():
 d=json.loads(DATASET_INFO.read_text());return d['features']['action']['names'],d['features']['observation.state']['names']
def validate_mapping(raw_names):
 action_names,state_names=dataset_names(); raw=list(map(str,raw_names))
 reasons=[]
 if len(raw)!=14:reasons.append('runtime mapping is not 14D')
 if len(set(raw))!=len(raw):reasons.append('duplicate runtime joint name')
 if raw!=action_names:reasons.append('runtime order differs from dataset action order')
 if state_names!=action_names:reasons.append('dataset state/action order mismatch')
 if raw[:7]!=[f'left_joint_{i}' for i in range(7)] or raw[7:]!=[f'right_joint_{i}' for i in range(7)]:reasons.append('left/right order ambiguous or swapped')
 if len(raw)<=13 or raw[6]!='left_joint_6' or raw[13]!='right_joint_6':reasons.append('gripper positions unresolved')
 return {'status':'PASS' if not reasons else 'BLOCK','hardware_execution_allowed':False,'reasons':reasons,'runtime_joint_names':raw,'dataset_action_names':action_names,'dataset_state_names':state_names,
  'rows':[{'action_index':i,'side':SIDES[i],'semantic_joint':SEMANTICS[i],'runtime_joint_name':raw[i] if i<len(raw) else None,'verified_source':str(DATASET_INFO)} for i in range(14)]}
def load_record(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def tracking_metrics(command,actual,fps=30.):
 c=np.asarray(command,float);a=np.asarray(actual,float);n=min(len(c),len(a));c=c[:n];a=a[:n]
 if c.shape!=a.shape or c.ndim!=2 or c.shape[1]!=14:raise ValueError('command/actual must match Nx14')
 e=a-c;lags=range(-min(30,n//3),min(30,n//3)+1);scores=[]
 for lag in lags:
  if lag>=0:x,y=c[:n-lag,:6],a[lag:n,:6]
  else:x,y=c[-lag:n,:6],a[:n+lag,:6]
  scores.append(float(np.mean((x-y)**2)) if len(x) else np.inf)
 lag=int(list(lags)[int(np.argmin(scores))])
 if lag>=0:x,y=c[:n-lag],a[lag:n]
 else:x,y=c[-lag:n],a[:n+lag]
 ec=y-x
 def event_delay(ch):
  dc=np.abs(np.diff(c[:,ch]));da=np.abs(np.diff(a[:,ch]));return (int(np.argmax(da))-int(np.argmax(dc)))/fps if len(dc) else None
 return {'samples':n,'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),'per_joint_rmse':np.sqrt(np.mean(e*e,axis=0)).tolist(),
  'max_abs_error':float(np.max(np.abs(e))),'max_error_frame_channel':list(map(int,np.unravel_index(np.argmax(np.abs(e)),e.shape))),
  'estimated_lag_frames':lag,'estimated_lag_seconds':lag/fps,'lag_corrected_rmse':float(np.sqrt(np.mean(ec*ec))),
  'lag_method':'bounded discrete MSE/cross-alignment on arm channels; ambiguous for low-motion or nonlinear response','left_gripper_event_delay_s':event_delay(6),'right_gripper_event_delay_s':event_delay(13)}
