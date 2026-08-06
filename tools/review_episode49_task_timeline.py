#!/usr/bin/env python3
"""Image-backed Episode-49 semantic timeline reviewer with explicit approval."""
from __future__ import annotations
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/task_frame_registration';APPROVED=ROOT/'configs/episode49_task_timeline.approved.json';DRAFT=ROOT/'configs/episode49_task_timeline.draft.json';MAN=ROOT/'reports/magsafe_lerobot_v3_manifest.csv';REPORT=ROOT/'reports/magsafe_lerobot_v3_build_report.json';RAW=ROOT/'raw_recordings/GoPark_20260729_111223';IMG=RAW/'images/observation.images.cam_high/episode_000000';ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz';PHASE=ROOT/'outputs/magsafe_gripper_phases.csv'
LABELS=('task_start','left_phone_approach_start','left_phone_grasp_start','phone_lift_start','phone_rotation_to_portrait_start','phone_portrait_reached','right_accessory_approach_start','right_accessory_grasp_start','accessory_detachment_start','accessory_removed','accessory_release_start','right_accessory_release_complete','phone_move_to_charger_start','phone_charger_alignment_start','phone_charger_attachment_complete','left_phone_release_start','left_phone_release_complete','task_end','unknown')
def mapping():
 rows=list(csv.DictReader(open(MAN)));r=next((x for x in rows if int(x['output_episode_index'])==49),None);imgs=sorted(IMG.glob('*.png'));z=np.load(ACTION,allow_pickle=False);n=len(z['optimized_action']);build=json.load(open(REPORT));listed=any(x.get('source_folder')==RAW.name and x.get('frames')==990 for x in build.get('source_frame_counts',[]));ok=bool(r and Path(r['source_folder']).resolve()==RAW.resolve() and Path(r['source_parquet']).name=='episode_000000.parquet' and int(r['source_frame_count'])==990 and int(r['cam_high_png_count'])==990 and len(imgs)==990 and n==990 and listed)
 d={'dataset_episode':49,'raw_recording_folder':str(RAW),'raw_episode':0,'frame_count':len(imgs),'fps':float(z['fps']) if 'fps' in z else 30.0,'first_frame_path':str(imgs[0]) if imgs else None,'last_frame_path':str(imgs[-1]) if imgs else None,'optimized_action_path':str(ACTION),'optimized_action_frames':n,'mapping_evidence':[str(MAN)+' row output_episode_index=49',str(REPORT)+' source_frame_counts entry',str(r['source_parquet']) if r else 'manifest row missing'],'status':'VERIFIED' if ok else 'UNKNOWN'};OUT.mkdir(parents=True,exist_ok=True);(OUT/'episode49_raw_source_mapping.json').write_text(json.dumps(d,indent=2)+'\n');return d
def load_draft():
 source=APPROVED if APPROVED.exists() and json.load(open(APPROVED)).get('status')=='EPISODE49_TIMELINE_APPROVED' else DRAFT
 d=json.load(open(source)) if source.exists() else {}
 d.update({'schema_version':2,'dataset_episode':49,'frame0':{'approved_task_stage':'INITIAL_POSTURE','approved_semantic_event':None,'object_contact':'NOT_EVALUATED','semantic_confidence':'UNASSIGNED'},'allowed_events':LABELS});d.setdefault('status','DRAFT_NEEDS_MANUAL_APPROVAL');d.setdefault('events',[]);return d
def frame_record(frame,d):
 rows=list(csv.DictReader(open(PHASE)));x=rows[frame];event=next((e for e in d['events'] if e['frame']==frame),None)
 return {'frame':frame,'time_sec':frame/30,'image':str(IMG/f'frame_{frame:06d}.png'),'left_gripper_raw':float(x['left_gripper_raw']),'right_gripper_raw':float(x['right_gripper_raw']),'left_automatic_phase':x['left_phase'],'right_automatic_phase':x['right_phase'],'approved_task_stage':'INITIAL_POSTURE' if frame==0 else (event['event'] if event else 'UNASSIGNED'),'approved_semantic_event':None if frame==0 else (event['event'] if event else None),'object_contact':'NOT_EVALUATED','semantic_confidence':'UNASSIGNED' if event is None else 'MANUALLY_ASSIGNED'}
def render(rec,d):
 import matplotlib.pyplot as plt
 from matplotlib.widgets import Button
 img=plt.imread(rec['image']);fig,ax=plt.subplots(figsize=(12,8));plt.subplots_adjust(bottom=.13);ax.imshow(img);ax.axis('off');warning='\nINITIAL POSTURE\nAUTOMATIC HOLD DOES NOT MEAN OBJECT HOLD' if rec['frame']==0 else ''
 ax.text(.01,.99,f"frame {rec['frame']} / 989 | {rec['time_sec']:.3f}s\nraw L/R={rec['left_gripper_raw']:.5f}/{rec['right_gripper_raw']:.5f}\nautomatic L/R={rec['left_automatic_phase']}/{rec['right_automatic_phase']}\napproved stage={rec['approved_task_stage']} | event={rec['approved_semantic_event']}\ncontact={rec['object_contact']} | confidence={rec['semantic_confidence']}{warning}",transform=ax.transAxes,va='top',color='white',fontsize=12,bbox={'facecolor':'black','alpha':.78})
 for x,label,delta in ((.36,'Previous',-1),(.52,'Next',1)):
  b=Button(fig.add_axes([x,.03,.12,.05]),label);b.on_clicked(lambda _,dd=delta:print(f"Run with --frame {max(0,min(989,rec['frame']+dd))}"))
 path=OUT/'approval_views'/f'timeline_frame_{rec["frame"]:06d}.png';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160,bbox_inches='tight');return fig,path
def main():
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,default=0);p.add_argument('--previous',action='store_true');p.add_argument('--next',action='store_true');p.add_argument('--step',type=int,choices=(-10,-5,-1,1,5,10),default=0);p.add_argument('--play',action='store_true');p.add_argument('--event',choices=LABELS);p.add_argument('--delete-event',action='store_true');p.add_argument('--note',default='');p.add_argument('--save-draft',action='store_true');p.add_argument('--approve',action='store_true');p.add_argument('--show',action='store_true');p.add_argument('--inspect',action='store_true');a=p.parse_args();mp=mapping();frame=max(0,min(989,a.frame+(-1 if a.previous else 1 if a.next else a.step)));d=load_draft()
 if frame==0 and a.event not in (None,'task_start','unknown'):raise SystemExit('Frame 0 is INITIAL_POSTURE; automatic HOLD cannot create semantic HOLD/contact.')
 if a.delete_event:d['events']=[e for e in d['events'] if e['frame']!=frame]
 if a.event:d['events']=[e for e in d['events'] if e['frame']!=frame]+[{'frame':frame,'time_sec':frame/30,'event':a.event,'note':a.note,'source':'manual_video_review'}]
 if a.save_draft or a.event or a.delete_event:DRAFT.parent.mkdir(parents=True,exist_ok=True);DRAFT.write_text(json.dumps(d,indent=2)+'\n')
 if a.approve:
  if mp['status']!='VERIFIED':raise SystemExit('Timeline approval blocked: raw source mapping UNKNOWN')
  required=set(LABELS)-{'unknown'};missing=sorted(required-{e['event'] for e in d['events']})
  if missing:raise SystemExit('Cannot approve; missing manually reviewed events: '+','.join(missing))
  d['status']='EPISODE49_TIMELINE_APPROVED';d['approved_at']=datetime.now(timezone.utc).isoformat();APPROVED.write_text(json.dumps(d,indent=2)+'\n')
 rec=frame_record(frame,d);fig,path=render(rec,d);print(json.dumps({'mapping':mp,'frame':rec,'draft_status':d['status'],'rendered':str(path)},indent=2));
 if a.show:
  import matplotlib.pyplot as plt;plt.show()
 if a.play:
  import cv2,time
  paused=False;i=frame
  while i<990:
   im=cv2.imread(str(IMG/f'frame_{i:06d}.png'));cv2.putText(im,f'frame {i} | SPACE pause | q quit',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2);cv2.imshow('Episode49 timeline reviewer',im);key=cv2.waitKey(0 if paused else 33)&255
   if key==ord('q'):break
   if key==32:paused=not paused
   elif key in (ord('a'),81):i=max(0,i-1)
   elif key in (ord('d'),83):i=min(989,i+1)
   elif not paused:i+=1
  cv2.destroyAllWindows()
 return 0
if __name__=='__main__':raise SystemExit(main())
