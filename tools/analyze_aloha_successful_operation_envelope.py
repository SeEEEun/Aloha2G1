#!/usr/bin/env python3
"""Compute an empirical envelope from the 50 successful ALOHA demonstrations."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).parent))
from aloha_source_validation_common import ROOT,BASE,NAMES,dump,load_action,motion_metrics,resample_minimum_jerk,sha
DATA=ROOT/'lerobot_magsafe_50_cam_high_v3/data/chunk-000/file-000.parquet'
INFO=ROOT/'lerobot_magsafe_50_cam_high_v3/meta/info.json'
OUT=BASE/'preflight/aloha_50episode_operational_envelope.json'
CANDIDATE=ROOT/'configs/aloha_source_validation_safety.candidate.json'
MANIFEST=ROOT/'reports/magsafe_lerobot_v3_manifest.csv'
Q=(50,95,99,99.9,100)
def vec(col):return np.asarray(col.to_pylist(),float)
def summary(values,episode,frame,channel=None,channel_map=None):
 x=np.asarray(values,float); flat=np.abs(x).reshape(-1);k=int(np.argmax(flat));idx=np.unravel_index(k,x.shape)
 loc={'episode':int(np.asarray(episode)[idx[0]]),'frame':int(np.asarray(frame)[idx[0]])}
 if channel is not None:loc['channel']=int(channel);loc['channel_name']=NAMES[int(channel)]
 elif channel_map is not None and x.ndim>1:
  actual=int(channel_map[idx[1]]);loc['channel']=actual;loc['channel_name']=NAMES[actual]
 return {'median':float(np.percentile(flat,50)),'p95':float(np.percentile(flat,95)),'p99':float(np.percentile(flat,99)),'p99_9':float(np.percentile(flat,99.9)),'maximum':float(flat[k]),'maximum_location':loc}
def consecutive(mask):
 best=run=0
 for x in mask:run=run+1 if x else 0;best=max(best,run)
 return best
def validate_raw_sources(state,action,ts,ep,manifest=MANIFEST):
 rows=list(csv.DictReader(Path(manifest).open()));issues=[];checked=0
 for row in rows:
  e=int(row['output_episode_index']);raw=Path(row['source_parquet']);z=np.flatnonzero(ep==e)
  if not raw.exists():issues.append(f'missing {raw}');continue
  table=pq.read_table(raw,columns=['observation.state','action','timestamp']);rs=vec(table['observation.state']);ra=vec(table['action']);rt=np.asarray(table['timestamp'])
  if len(z)!=len(rs) or not np.array_equal(state[z],rs) or not np.array_equal(action[z],ra) or not np.array_equal(ts[z],rt):issues.append(f'episode {e} differs from {raw}')
  else:checked+=1
 return {'manifest':str(Path(manifest).resolve()),'manifest_rows':len(rows),'raw_episodes_exactly_matched':checked,'status':'PASS' if checked==50 and not issues else 'FAIL','issues':issues}
def build(data=DATA):
 t=pq.read_table(data);state=vec(t['observation.state']);action=vec(t['action']);ts=np.asarray(t['timestamp']);ep=np.asarray(t['episode_index']);fr=np.asarray(t['frame_index'])
 if state.shape[1:]!=(14,) or action.shape!=state.shape:raise ValueError('expected aligned Nx14 state/action')
 if len(np.unique(ep))!=50:raise ValueError(f'expected 50 episodes, got {len(np.unique(ep))}')
 same=np.r_[False,ep[1:]==ep[:-1]];pair=np.flatnonzero(same);triple=np.flatnonzero(same & np.r_[False,same[:-1]])
 ds=state[1:]-state[:-1];da=action[1:]-action[:-1];ds=ds[same[1:]];da=da[same[1:]]
 dv_s=ds*30;dv_a=da*30
 acc_s=[];acc_a=[];acc_ep=[];acc_fr=[]
 for e in np.unique(ep):
  z=np.flatnonzero(ep==e);acc_s.append(np.diff(state[z],n=2,axis=0)*900);acc_a.append(np.diff(action[z],n=2,axis=0)*900);acc_ep.extend([e]*max(0,len(z)-2));acc_fr.extend(fr[z][2:])
 acc_s=np.vstack(acc_s);acc_a=np.vstack(acc_a);acc_ep=np.asarray(acc_ep);acc_fr=np.asarray(acc_fr)
 dt=[];dt_ep=[];dt_fr=[]
 for e in np.unique(ep):
  z=np.flatnonzero(ep==e);dt.extend(np.diff(ts[z]));dt_ep.extend([e]*(len(z)-1));dt_fr.extend(fr[z][1:])
 dt=np.asarray(dt);dt_ep=np.asarray(dt_ep);dt_fr=np.asarray(dt_fr);nominal=1/30
 per_channel=[]
 for j in range(14):
  imn=int(np.argmin(state[:,j]));imx=int(np.argmax(state[:,j]))
  per_channel.append({'index':j,'name':NAMES[j],'group':'gripper' if j in (6,13) else 'arm','position_min':float(state[imn,j]),'position_min_location':{'episode':int(ep[imn]),'frame':int(fr[imn])},'position_max':float(state[imx,j]),'position_max_location':{'episode':int(ep[imx]),'frame':int(fr[imx])},'state_step':summary(ds[:,j],ep[pair],fr[pair],j),'state_velocity':summary(dv_s[:,j],ep[pair],fr[pair],j),'state_acceleration':summary(acc_s[:,j],acc_ep,acc_fr,j),'action_step':summary(da[:,j],ep[pair],fr[pair],j),'action_velocity':summary(dv_a[:,j],ep[pair],fr[pair],j),'action_acceleration':summary(acc_a[:,j],acc_ep,acc_fr,j),'action_observation_error':summary(action[:,j]-state[:,j],ep,fr,j)})
 per_episode=[]
 for e in np.unique(ep):
  z=np.flatnonzero(ep==e);edt=np.diff(ts[z])
  for j in range(14):
   ss=state[z,j];aa=action[z,j]
   per_episode.append({'episode':int(e),'channel':j,'channel_name':NAMES[j],'group':'gripper' if j in (6,13) else 'arm','position_min':float(ss.min()),'position_max':float(ss.max()),'state_max_step':float(np.max(np.abs(np.diff(ss)),initial=0)),'state_max_velocity':float(np.max(np.abs(np.diff(ss)*30),initial=0)),'state_max_acceleration':float(np.max(np.abs(np.diff(ss,n=2)*900),initial=0)),'action_max_step':float(np.max(np.abs(np.diff(aa)),initial=0)),'action_max_velocity':float(np.max(np.abs(np.diff(aa)*30),initial=0)),'action_max_acceleration':float(np.max(np.abs(np.diff(aa,n=2)*900),initial=0)),'action_observation_max_error':float(np.max(np.abs(aa-ss),initial=0)),'timestamp_interval_median':float(np.median(edt)),'timestamp_interval_max':float(np.max(edt,initial=0))})
 groups={}
 for name,cols in {'arm':[i for i in range(14) if i not in (6,13)],'gripper':[6,13]}.items():
  groups[name]={'state_step':summary(ds[:,cols],ep[pair],fr[pair],channel_map=cols),'state_velocity':summary(dv_s[:,cols],ep[pair],fr[pair],channel_map=cols),'state_acceleration':summary(acc_s[:,cols],acc_ep,acc_fr,channel_map=cols),'action_step':summary(da[:,cols],ep[pair],fr[pair],channel_map=cols),'action_velocity':summary(dv_a[:,cols],ep[pair],fr[pair],channel_map=cols),'action_acceleration':summary(acc_a[:,cols],acc_ep,acc_fr,channel_map=cols),'action_observation_error':summary(action[:,cols]-state[:,cols],ep,fr,channel_map=cols)}
 dtstats=summary(dt,dt_ep,dt_fr);stale=dt>nominal*1.5;over=dt>nominal
 out={'envelope_type':'EMPIRICAL_SUCCESSFUL_OPERATION_ENVELOPE','vendor_hardware_limit':False,'dataset':str(Path(data).resolve()),'dataset_sha256':sha(data),'metadata':str(INFO),'raw_source_validation':validate_raw_sources(state,action,ts,ep),'episodes':50,'frames':len(state),'fps':30,'aligned_action_observation':True,'per_episode_per_channel':per_episode,'per_channel':per_channel,'groups':groups,'timestamp_interval_seconds':dtstats,'nominal_interval_seconds':nominal,'dropped_or_stale_definition':'interval > 1.5/30 seconds','dropped_or_stale_count':int(stale.sum()),'maximum_consecutive_stale_intervals':consecutive(stale),'overrun_definition':'interval > 1/30 seconds','overrun_count':int(over.sum()),'maximum_consecutive_overrun_intervals':consecutive(over),'gripper_observed_ranges':{'left':[float(state[:,6].min()),float(state[:,6].max())],'right':[float(state[:,13].min()),float(state[:,13].max())]}}
 return out,state,action,ep,fr
def candidate(envelope):
 action,_,_=load_action();seg=action[:30];cmd,_,_,_=resample_minimum_jerk(seg,30,.1,30,'hold-current',seg[0,[6,13]]);m=motion_metrics(cmd,30);g=envelope['groups'];dt=envelope['timestamp_interval_seconds']
 def item(value,unit,source,derivation,passed=None,status='CANDIDATE'):
  return {'status':status,'candidate_value':value,'unit':unit,'source':source,'derivation':derivation,'vendor_limit':False,'empirical_limit':value is not None,'requires_human_approval':True,'episode49_frames_0_29_pass':passed}
 arm_ranges=[[x['position_min'],x['position_max']] for x in envelope['per_channel'] if x['group']=='arm'];gr=[[envelope['per_channel'][i]['position_min'],envelope['per_channel'][i]['position_max']] for i in (6,13)]
 arm=np.delete(cmd,(6,13),axis=1)
 arm_range_pass=all(np.all((arm[:,j]>=r[0]) & (arm[:,j]<=r[1])) for j,r in enumerate(arm_ranges))
 return {'status':'CANDIDATE_NOT_APPROVED','hardware_replay_allowed':False,'envelope_type':'EMPIRICAL_SUCCESSFUL_OPERATION_ENVELOPE','reviewed_config_generated':False,
  'criteria':{'arm_position_range':item(arm_ranges,'rad',envelope['dataset'],'observed min/max per arm channel',bool(arm_range_pass),'EMPIRICAL_RANGE'),
  'gripper_position_range':item(gr,'m',envelope['dataset'],'observed state min/max; first test holds actual grippers',None,'REQUIRES_LIVE_CURRENT_STATE'),
  'max_command_step':item(g['arm']['action_step']['p99_9'],'rad',envelope['dataset'],'99.9 percentile absolute action frame step across successful demonstrations',m['max_step']<=g['arm']['action_step']['p99_9']),
  'max_velocity':item(g['arm']['action_velocity']['p99_9'],'rad/s',envelope['dataset'],'99.9 percentile action velocity at 30 Hz',max(m['max_velocity_per_channel'])<=g['arm']['action_velocity']['p99_9']),
  'max_acceleration':item(g['arm']['action_acceleration']['p99_9'],'rad/s^2',envelope['dataset'],'99.9 percentile action acceleration at 30 Hz',max(m['max_acceleration_per_channel'])<=g['arm']['action_acceleration']['p99_9']),
  'max_start_position_error':item(None,'rad and m','UNRESOLVED','successful demonstrations do not establish a safe arbitrary current-to-start displacement',None,'UNRESOLVED'),
  'max_tracking_error':item(g['arm']['action_observation_error']['p99_9'],'rad',envelope['dataset'],'99.9 percentile aligned arm action-observation absolute difference',None),
  'tracking_error_duration':item(None,'s','UNRESOLVED','dataset samples error but does not label unsafe persistence duration',None,'UNRESOLVED'),
  'state_timeout':item(dt['p99_9'],'s',envelope['dataset'],'99.9 percentile successful dataset timestamp interval; not a transport timeout',None),
  'control_loop_overrun_threshold':item(max(0.,dt['p99_9']-1/30),'s late',envelope['dataset'],'99.9 percentile timestamp interval minus nominal period',None),
  'consecutive_overrun_limit':item(envelope['maximum_consecutive_overrun_intervals'],'cycles',envelope['dataset'],'maximum consecutive intervals above nominal 1/30 s',None)}}
def markdown(e,c):
 lines=['# ALOHA 50-episode operational envelope','','**EMPIRICAL_SUCCESSFUL_OPERATION_ENVELOPE — not a vendor hardware safety limit.**','',f"Episodes: {e['episodes']}; frames: {e['frames']}; dropped/stale intervals: {e['dropped_or_stale_count']}",'','| Group/metric | median | p95 | p99 | p99.9 | max |','|---|---:|---:|---:|---:|---:|']
 for group in ('arm','gripper'):
  for metric in ('state_step','state_velocity','state_acceleration','action_step','action_velocity','action_acceleration','action_observation_error'):
   x=e['groups'][group][metric];lines.append(f"| {group} {metric} | {x['median']:.9g} | {x['p95']:.9g} | {x['p99']:.9g} | {x['p99_9']:.9g} | {x['maximum']:.9g} |")
 lines += ['','Candidate criteria remain human-unapproved. Null criteria are intentionally unresolved.','',f"Candidate config: `{CANDIDATE}`",'','Hardware replay allowed: **NO**',''];return '\n'.join(lines)
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--candidate',type=Path,default=CANDIDATE);a=p.parse_args(argv);e,*_=build(a.data);c=candidate(e);dump(a.output,e);a.output.with_suffix('.md').write_text(markdown(e,c));dump(a.candidate,c);print(json.dumps({'output':str(a.output),'candidate':str(a.candidate),'status':c['status']},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
