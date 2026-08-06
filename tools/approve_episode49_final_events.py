#!/usr/bin/env python3
"""Approve only the two final Episode-49 table-placement events manually."""
from __future__ import annotations
import argparse,json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
import approve_episode49_events_interactive as base
APPROVED=ROOT/'configs/episode49_task_timeline.approved.json';DRAFT=ROOT/'configs/episode49_final_events.draft.json';SEMANTICS=ROOT/'outputs/task_frame_registration/final_task_order_semantics.json'
FIXED={'left_phone_grasp_start':176,'phone_rotation_to_portrait_start':200,'phone_portrait_reached':223,'right_accessory_grasp_start':326,'accessory_detachment_start':329,'accessory_removed':341,'phone_move_to_charger_start':380,'phone_charger_attachment_complete':530,'left_phone_release_complete':586,'right_accessory_release_complete':646}
NEW=(('left_arm_return_near_home','phone 부착과 왼손 release 후 왼팔이 초기/대기 자세 근처로 거의 돌아온 순간'),('right_accessory_place_on_table_start','오른손이 accessory를 테이블 placement 방향으로 내려놓기 시작한 순간'))
CANONICAL=('left_phone_grasp_start','phone_rotation_to_portrait_start','phone_portrait_reached','right_accessory_grasp_start','accessory_detachment_start','accessory_removed','phone_move_to_charger_start','phone_charger_attachment_complete','left_phone_release_complete','left_arm_return_near_home','right_accessory_place_on_table_start','accessory_placed_on_table_complete','task_end')
def load_verified_existing(path:Path=APPROVED)->dict:
 if not path.exists():raise ValueError('Approved timeline missing')
 d=json.load(open(path));m={x['event']:int(x['frame']) for x in d.get('events',[])}
 bad={k:(m.get(k),v) for k,v in FIXED.items() if m.get(k)!=v}
 if d.get('status')!='APPROVED_MANUAL_VIDEO_REVIEW' or bad:raise ValueError(f'Existing approved timeline/status mismatch; refusing: {bad}')
 return d
def order_ok(new:dict)->tuple[bool,str]:
 a=int(new['left_arm_return_near_home']);b=int(new['right_accessory_place_on_table_start'])
 ok=530<586<=a<=b<=646;return ok,f'required: 530 < 586 <= {a} <= {b} <= 646'
def semantic_record(event,frame,note=''):return {'event':event,'frame':int(frame),'timestamp':frame/30.0,'source':'manual_video_review','confidence':'USER_CONFIRMED','automatic_phase_used_as_evidence':False,'note':note}
def write_semantics():
 d={'schema_version':1,'status':'FINAL_TASK_EVENTS_REQUIRED','canonical_event_order':CANONICAL,'fixed_existing_frames':FIXED,
  'right_accessory_phase_order':['ACCESSORY_APPROACH','ACCESSORY_GRASP','ACCESSORY_DETACHMENT','ACCESSORY_HOLD_AND_TRANSPORT','ACCESSORY_TABLE_PLACEMENT','RELEASED'],
  'phase_rules':{'ACCESSORY_HOLD_AND_TRANSPORT':{'start_event':'accessory_removed','end_event':'right_accessory_place_on_table_start','accessory_relation':'continues to follow right palm','includes_phone_move_and_attachment':True,'release_orientation_applied':False},'ACCESSORY_TABLE_PLACEMENT':{'start_event':'right_accessory_place_on_table_start','end_event':'accessory_placed_on_table_complete','meaning':'approach table surface; exact orientation/tolerance NOT_DEFINED'}},
  'kinematic_replay':{'initial':'accessory attached to phone back-center','right_accessory_grasp_start':'follow right palm diagnostic','accessory_removed':'detach from phone but continue following right palm','right_accessory_place_on_table_start':'table-placement stage','accessory_placed_on_table_complete':'remain on table and stop following hand','frame_341_release':False,'overlay':['KINEMATIC SEMANTIC REPLAY','ACCESSORY HELD BY RIGHT HAND','NOT PHYSICS GRASP']},
  'metric_events':['accessory_removed','phone_charger_attachment_complete','left_arm_return_near_home','right_accessory_place_on_table_start','accessory_placed_on_table_complete'],
  'raw_metrics':['accessory removal timing','accessory hold duration','accessory table-placement start timing','accessory placement/release completion timing','delay after phone charger attachment','left arm near-home timing','task completion timing'],
  'task_stages':['phone grasp','landscape-to-portrait rotation','accessory grasp','accessory detachment','phone move to charger','phone charger attachment','left phone release','left arm return near home','accessory placement on table','full task completion'],
  'evaluation_policy':['MEASURED_RAW_VALUE','MANUAL_REVIEW_REQUIRED','NO_FAKE_TASK_SUCCESS'],'trajectory_generated':False,'ik_run':False,'dds_or_publisher_used':False,'real_robot_used':False}
 SEMANTICS.parent.mkdir(parents=True,exist_ok=True);SEMANTICS.write_text(json.dumps(d,indent=2)+'\n');return d
def save_draft(chosen):DRAFT.write_text(json.dumps({'status':'FINAL_TASK_EVENTS_REQUIRED','existing_10_preserved':FIXED,'events':[semantic_record(k,v) for k,v in chosen.items()],'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
def rewrite(existing,chosen,path:Path=APPROVED)->Path:
 ok,msg=order_ok(chosen)
 if not ok:raise ValueError(msg)
 backup=path.with_name(path.stem+'.backup_'+datetime.now().strftime('%Y%m%d_%H%M%S')+path.suffix);shutil.copy2(path,backup)
 old={x['event']:dict(x) for x in existing['events']};events=[]
 for name,frame in FIXED.items():
  if name=='right_accessory_release_complete':continue
  events.append(old[name])
 events.extend(semantic_record(k,chosen[k]) for k,_ in NEW)
 placed=semantic_record('accessory_placed_on_table_complete',646);placed.update({'alias':'right_accessory_release_complete','semantic_meaning':'right gripper has placed the accessory on the table and fully released it'});events.append(placed)
 alias=dict(old['right_accessory_release_complete']);alias.update({'alias_of':'accessory_placed_on_table_complete','semantic_meaning':placed['semantic_meaning']});events.append(alias)
 out=dict(existing);out['status']='APPROVED_MANUAL_VIDEO_REVIEW';out['canonical_event_order']=list(CANONICAL);out['events']=sorted(events,key=lambda x:(x['frame'],CANONICAL.index(x['event']) if x['event'] in CANONICAL else 999));out['final_task_order_updated_at']=datetime.now(timezone.utc).isoformat();out['existing_10_frames_preserved']=True;path.write_text(json.dumps(out,indent=2)+'\n');return backup
def run(input_fn=input,open_enabled=True):
 existing=load_verified_existing();write_semantics();print('Existing 10 manually approved events (read-only):')
 for k,v in FIXED.items():print(f'  {k:42s} {v}')
 chosen={x['event']:x['frame'] for x in json.load(open(DRAFT)).get('events',[])} if DRAFT.exists() else {};i=next((j for j,(n,_) in enumerate(NEW) if n not in chosen),2)
 while i<2:
  name,desc=NEW[i];print(f'\n[{i+1}/2] {name}\n{desc}');v=input_fn('Frame number | p previous | s save | u UNKNOWN | q quit: ').strip().lower()
  if v=='p':i=max(0,i-1);continue
  if v=='s':save_draft(chosen);continue
  if v=='u':chosen.pop(name,None);save_draft(chosen);i+=1;continue
  if v=='q':save_draft(chosen);print('FINAL TASK EVENTS REQUIRED\nEXISTING 10 EVENTS PRESERVED\nNO TRAJECTORY GENERATED');return 0
  try:f=int(v)
  except ValueError:print('Invalid input; no frame selected.');continue
  if not 0<=f<=989:print('Frame must be 0..989.');continue
  preview=base.make_preview(name,f);print('Preview:',preview)
  if open_enabled:
   try:subprocess.Popen(['xdg-open',str(preview)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   except OSError:pass
  if input_fn(f'Confirm {name} at frame {f}? [y/n] ').strip().lower()!='y':print('Not saved.');continue
  chosen[name]=f;save_draft(chosen);i+=1
 if set(chosen)!={x[0] for x in NEW}:save_draft(chosen);return 2
 ok,msg=order_ok(chosen)
 if not ok:print('ORDER VIOLATION:',msg);save_draft(chosen);return 3
 if input_fn('Both final events are manually selected. Rewrite approved timeline? [y/n] ').strip().lower()!='y':save_draft(chosen);return 0
 backup=rewrite(existing,chosen);print('Backup:',backup);print('FINAL TASK ORDER APPROVED\nACCESSORY TABLE-PLACEMENT EVENTS ADDED\nREADY FOR SCENE-REGISTERED TASK VALIDATION\nNO TRAJECTORY GENERATED BY THIS PATCH');return 0
def main():
 p=argparse.ArgumentParser();p.add_argument('--no-open',action='store_true');a=p.parse_args();write_semantics();return run(open_enabled=not a.no_open)
if __name__=='__main__':raise SystemExit(main())
