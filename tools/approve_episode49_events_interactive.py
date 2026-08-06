#!/usr/bin/env python3
"""Manual-only CLI wizard for approving Episode-49 task events.

This tool displays evidence and records user input. It performs no IK,
retargeting, trajectory generation, DDS initialization, or hardware access.
"""
from __future__ import annotations
import argparse,csv,json,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
from typing import Callable
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('/home/jbnu/aloha_g1_dataset')
IMAGES=ROOT/'raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000'
VIDEO=ROOT/'outputs/scene_registered_task_validation/event_review/episode49_event_review.mp4'
PHASES=ROOT/'outputs/magsafe_gripper_phases.csv'
DRAFT=ROOT/'configs/episode49_task_timeline.draft.json'
APPROVED=ROOT/'configs/episode49_task_timeline.approved.json'
PREVIEWS=ROOT/'outputs/scene_registered_task_validation/event_review/manual_previews'
FPS=30.0;MIN_FRAME=0;MAX_FRAME=989
EVENTS=(
 ('left_phone_grasp_start','왼쪽 gripper가 phone 왼쪽 측면을 실제로 물기 시작한 순간'),
 ('phone_rotation_to_portrait_start','phone의 긴 축이 landscape에서 portrait로 회전하기 시작한 순간'),
 ('phone_portrait_reached','phone의 긴 축이 세로가 되고 회전이 거의 완료된 순간'),
 ('right_accessory_grasp_start','오른쪽 gripper가 원형 accessory를 실제로 물기 시작한 순간'),
 ('accessory_detachment_start','accessory가 phone 뒷면에서 떨어지기 시작한 순간'),
 ('accessory_removed','accessory와 phone 사이에 명확한 간격이 생긴 순간'),
 ('right_accessory_release_complete','오른쪽 gripper가 accessory에서 완전히 떨어진 순간'),
 ('phone_move_to_charger_start','왼손이 portrait phone을 charger 방향으로 이동시키기 시작한 순간'),
 ('phone_charger_attachment_complete','phone 뒷면이 charger pad에 붙어 자세가 안정된 순간'),
 ('left_phone_release_complete','왼손 gripper가 phone에서 완전히 떨어진 순간'))
EVENT_NAMES=tuple(x[0] for x in EVENTS)

def preview_frames(frame:int)->list[int]:
 if not MIN_FRAME<=frame<=MAX_FRAME:raise ValueError(f'frame must be {MIN_FRAME}..{MAX_FRAME}')
 return sorted(set(x for d in (-10,-5,-1,0,1,5,10) if MIN_FRAME<=(x:=frame+d)<=MAX_FRAME))
def timestamp(frame:int)->float:return frame/FPS
def record(event:str,frame:int,note:str='')->dict:
 if event not in EVENT_NAMES:raise ValueError(event)
 if not MIN_FRAME<=frame<=MAX_FRAME:raise ValueError(frame)
 return {'event':event,'frame':int(frame),'timestamp':timestamp(frame),'source':'manual_video_review','confidence':'USER_CONFIRMED','automatic_phase_used_as_evidence':False,'note':note}
def validate_order(events:list[dict])->tuple[bool,list[str]]:
 m={x['event']:int(x['frame']) for x in events};errors=[]
 for seq in (('left_phone_grasp_start','phone_rotation_to_portrait_start','phone_portrait_reached'),('right_accessory_grasp_start','accessory_detachment_start','accessory_removed','right_accessory_release_complete'),('phone_portrait_reached','phone_move_to_charger_start','phone_charger_attachment_complete','left_phone_release_complete')):
  if all(x in m for x in seq):
   for a,b in zip(seq,seq[1:]):
    if m[a]>m[b]:errors.append(f'{a} frame {m[a]} > {b} frame {m[b]}')
 return not errors,errors
def new_draft()->dict:return {'schema_version':3,'status':'DRAFT_WAITING_FOR_USER','dataset_episode':49,'fps':FPS,'frame_range':[MIN_FRAME,MAX_FRAME],'events':[],'unknown_events':[],'automatic_events_saved':False,'updated_at':datetime.now(timezone.utc).isoformat()}
def load_draft()->dict:
 if not DRAFT.exists():return new_draft()
 d=json.load(open(DRAFT));d['status']='DRAFT_WAITING_FOR_USER';d.setdefault('events',[]);d.setdefault('unknown_events',[]);d['automatic_events_saved']=False;return d
def save_draft(d:dict,path:Path|None=None)->None:
 path=DRAFT if path is None else path
 d=dict(d);d['status']='DRAFT_WAITING_FOR_USER';d['updated_at']=datetime.now(timezone.utc).isoformat();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,indent=2)+'\n')
def complete(d:dict)->bool:return {x['event'] for x in d.get('events',[])}==set(EVENT_NAMES) and validate_order(d['events'])[0]
def save_approved(d:dict,path:Path|None=None)->Path|None:
 path=APPROVED if path is None else path
 if not complete(d):raise ValueError('All 10 manually confirmed ordered events are required')
 if any(x.get('source')!='manual_video_review' or x.get('confidence')!='USER_CONFIRMED' for x in d['events']):raise ValueError('Every event must be manually confirmed')
 backup=None
 if path.exists():
  stamp=datetime.now().strftime('%Y%m%d_%H%M%S');backup=path.with_name(path.stem+f'.backup_{stamp}'+path.suffix);shutil.copy2(path,backup)
 out=dict(d);out['status']='APPROVED_MANUAL_VIDEO_REVIEW';out['approved_at']=datetime.now(timezone.utc).isoformat();out['events']=sorted(out['events'],key=lambda x:EVENT_NAMES.index(x['event']));path.write_text(json.dumps(out,indent=2)+'\n');return backup
def phase_rows()->list[dict]:return list(csv.DictReader(open(PHASES)))
def make_preview(event:str,frame:int,rows:list[dict]|None=None)->Path:
 rows=rows or phase_rows();ids=preview_frames(frame);thumb_w,thumb_h=320,240;cap=58;sheet=Image.new('RGB',(thumb_w*len(ids),thumb_h+cap),(245,245,245));draw=ImageDraw.Draw(sheet)
 for k,i in enumerate(ids):
  im=Image.open(IMAGES/f'frame_{i:06d}.png').convert('RGB');im.thumbnail((thumb_w,thumb_h));x=k*thumb_w+(thumb_w-im.width)//2;sheet.paste(im,(x,0));r=rows[i];text=f'frame {i}  t={timestamp(i):.3f}s\nL/R raw {float(r["left_gripper_raw"]):.4f}/{float(r["right_gripper_raw"]):.4f}\nauto {r["left_phase"]}/{r["right_phase"]}';draw.rectangle((k*thumb_w,thumb_h,k*thumb_w+thumb_w,thumb_h+cap),fill='black');draw.text((k*thumb_w+4,thumb_h+3),text,fill='white')
 draw.rectangle((0,0,sheet.width,28),fill='black');draw.text((8,7),f'{event} - selected frame {frame} - MANUAL CONFIRMATION REQUIRED',fill='red');PREVIEWS.mkdir(parents=True,exist_ok=True);path=PREVIEWS/f'{event}_frame_{frame:06d}.png';sheet.save(path);return path
def open_file(path:Path,enabled:bool)->None:
 if enabled:
  try:subprocess.Popen(['xdg-open',str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  except OSError as e:print(f'Could not open {path}: {e}')
def progress(d):
 m={x['event']:x for x in d['events']};print('\nProgress:')
 for i,(name,_) in enumerate(EVENTS,1):print(f'{i:2d}. {name:42s} '+(f"frame {m[name]['frame']} CONFIRMED" if name in m else ('UNKNOWN' if name in d['unknown_events'] else 'pending')))
def run(input_fn:Callable[[str],str]=input,open_enabled:bool=True)->int:
 print('Review video:',VIDEO);open_file(VIDEO,open_enabled);d=load_draft();confirmed={x['event']:x for x in d['events']};idx=next((i for i,(n,_) in enumerate(EVENTS) if n not in confirmed),len(EVENTS))
 while idx<len(EVENTS):
  name,desc=EVENTS[idx];print(f'\n[{idx+1}/10] {name}\n{desc}');value=input_fn('Frame number | p previous | s save | u UNKNOWN | q quit: ').strip().lower()
  if value=='p':idx=max(0,idx-1);continue
  if value=='s':d['events']=list(confirmed.values());save_draft(d);progress(d);continue
  if value=='u':
   if name not in d['unknown_events']:d['unknown_events'].append(name)
   confirmed.pop(name,None);d['events']=list(confirmed.values());save_draft(d);idx+=1;continue
  if value=='q':d['events']=list(confirmed.values());save_draft(d);print('Draft saved:',DRAFT);return 0
  try:frame=int(value)
  except ValueError:print('Invalid input. No frame selected.');continue
  if not MIN_FRAME<=frame<=MAX_FRAME:print('Frame must be 0..989.');continue
  preview=make_preview(name,frame);print('Preview:',preview);open_file(preview,open_enabled)
  if input_fn(f'Confirm {name} at frame {frame}? [y/n] ').strip().lower()!='y':print('Not saved.');continue
  confirmed[name]=record(name,frame);d['unknown_events']=[x for x in d['unknown_events'] if x!=name];d['events']=list(confirmed.values());save_draft(d);idx+=1
 if not complete(d):ok,errors=validate_order(d['events']);print('Cannot approve:',errors or 'missing events');save_draft(d);return 2
 progress(d)
 if APPROVED.exists():print(f'WARNING: existing approved-path file will be timestamp-backed up before replacement: {APPROVED}')
 if input_fn('All 10 events are manually selected. Create approved timeline? [y/n] ').strip().lower()!='y':save_draft(d);print('Approval cancelled; draft retained.');return 0
 backup=save_approved(d);print('\nevent | frame | timestamp | source');
 for name in EVENT_NAMES:
  x=next(e for e in d['events'] if e['event']==name);print(f"{name} | {x['frame']} | {x['timestamp']:.6f} | {x['source']}")
 if backup:print('Previous approved timeline backup:',backup)
 print('\nEVENT TIMELINE APPROVED\nNO TRAJECTORY GENERATED BY APPROVAL WIZARD\nREADY FOR SCENE-REGISTERED TASK VALIDATION')
 print('\ncd /home/jbnu/aloha_g1_dataset && \\\nMUJOCO_GL=egl \\\nMPLCONFIGDIR=/tmp/scene_task_validation_mpl \\\n/home/jbnu/miniconda3/envs/trossen_mujoco_env/bin/python \\\ntools/run_scene_registered_task_validation.py')
 return 0
def main():
 p=argparse.ArgumentParser();p.add_argument('--no-open',action='store_true',help='Do not launch xdg-open; useful over SSH.');a=p.parse_args();return run(open_enabled=not a.no_open)
if __name__=='__main__':raise SystemExit(main())
