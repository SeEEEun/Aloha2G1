#!/usr/bin/env python3
"""Build read-only Episode-49 end-timeline review media and candidates."""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
import cv2,numpy as np

ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/episode49_end_timeline_review'
IMG=ROOT/'raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000'
ACTION=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
TIMELINE=ROOT/'configs/episode49_task_timeline.approved.json';MAPPING=ROOT/'outputs/task_frame_registration/episode49_raw_source_mapping.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def runs(x,lo,w,q=.35,minlen=6):
 v=np.r_[0,np.linalg.norm(np.diff(x,axis=0),axis=1)];rv=np.convolve(v,np.ones(w)/w,'same');th=float(np.quantile(rv[lo:],q));mask=rv<th;out=[];s=None
 for i in range(lo,len(x)):
  if mask[i] and s is None:s=i
  if (not mask[i] or i==len(x)-1) and s is not None:
   e=i-1 if not mask[i] else i
   if e-s+1>=minlen:out.append((s,e,float(rv[s:e+1].mean()),th))
   s=None
 return out,rv
def overlay(im,f,events,left_candidates,end_candidates):
 im=im.copy();lines=[f'EP49 frame {f:03d}/989  t={f/30:.3f}s']
 if f in events:lines.append('APPROVED EVENT: '+events[f])
 if f>=530:lines.append('PHONE ATTACHED')
 if f>=586:lines.append('LEFT RELEASED')
 if any(abs(f-x)<=5 for x in left_candidates):lines.append('LEFT RETURN CANDIDATE (DIAGNOSTIC)')
 if any(abs(f-x)<=5 for x in end_candidates):lines.append('TASK END CANDIDATE (DIAGNOSTIC)')
 y=28
 for line in lines:
  cv2.putText(im,line,(14,y),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,0,0),4,cv2.LINE_AA);cv2.putText(im,line,(14,y),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA);y+=25
 return im
def sheets(prefix,frames,events,lc,ec,per=25):
 paths=[];thumbw,thumbh=384,216;cols=5;rows=math.ceil(per/cols)
 for page,start in enumerate(range(0,len(frames),per),1):
  canvas=np.zeros((rows*thumbh,cols*thumbw,3),np.uint8)
  for k,f in enumerate(frames[start:start+per]):
   im=cv2.imread(str(IMG/f'frame_{f:06d}.png'));im=cv2.resize(im,(thumbw,thumbh));im=overlay(im,f,events,lc,ec);rr,cc=divmod(k,cols);canvas[rr*thumbh:(rr+1)*thumbh,cc*thumbw:(cc+1)*thumbw]=im
  path=OUT/f'{prefix}_{page:02d}.png';cv2.imwrite(str(path),canvas);paths.append(str(path))
 return paths
def candidates(name,rr,rv,maxn=5):
 out=[]
 for i,(s,e,mean,th) in enumerate(rr[:maxn]):
  f=s;before=float(np.mean(rv[max(0,f-5):f])) if f else 0.;after=float(np.mean(rv[f:min(len(rv),f+6)]))
  if name=='left_arm_return_near_home':
   reason='In the fine sheet the left hand is visibly separated from the phone and near the returned/resting arm posture; verify that no later left-arm motion is semantically meaningful.'
   recommended=(i==0)
  else:
   reason='Both arms appear stationary in this low-motion run; verify in the slow video whether this is the final episode state rather than a temporary pause.'
   recommended=(i==min(maxn,len(rr))-1)
  out.append({'event':name,'frame':int(f),'timestamp':f/30,'visual_reason':reason,'action_based_evidence':{'low_motion_run':[int(s),int(e)],'rolling_speed_mean':mean,'diagnostic_threshold':th},'previous_next_frame_difference':{'mean_speed_previous_5':before,'mean_speed_next_6':after},'recommended_candidate':recommended,'status':'USER_REVIEW_REQUIRED'})
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True);timeline=json.loads(TIMELINE.read_text());mapping=json.loads(MAPPING.read_text());events={e['frame']:e['event'] for e in timeline['events']};names=set(events.values())
 imgs=sorted(IMG.glob('frame_*.png'));z=np.load(ACTION);a=z['optimized_action'].astype(float);ts=z['timestamp'].astype(float);fps=float(z['fps'])
 if mapping.get('status')!='VERIFIED' or len(imgs)!=990 or a.shape!=(990,14) or len(ts)!=990 or fps!=30 or not np.isfinite(a).all():raise RuntimeError('source mapping/input validation failed')
 release=next((e['frame'] for e in timeline['events'] if e['event']=='left_phone_release_complete'),None);start=max(0,release-30) if release is not None and 0<=release<990 else 500;end=989
 left=a[:,:6];both=np.c_[a[:,:6],a[:,7:13]];lr,lv=runs(left,release or 500,15);er,ev=runs(both,max(646,release or 500),30)
 lc=[x[0] for x in lr[:5]];ec=[x[0] for x in er[:5]]
 coarse=sheets('contact_sheet_coarse',list(range(start,end+1,15)),events,lc,ec)
 leftfine=sheets('contact_sheet_left_return_fine',list(range(max(start,min(lc or [650])-15),min(end,max(lc or [850])+15)+1,3)),events,lc,ec)
 endfine=sheets('contact_sheet_task_end_fine',list(range(max(start,min(ec or [800])-15),end+1,3)),events,lc,ec)
 # Encode the original frames with diagnostics; no inferred label is promoted.
 sample=cv2.imread(str(imgs[0]));h,w=sample.shape[:2];fourcc=cv2.VideoWriter_fourcc(*'mp4v');video_paths=[]
 for name,outfps in [('episode49_end_review_realtime.mp4',30.0),('episode49_end_review_slow_0p25x.mp4',7.5)]:
  path=OUT/name;vw=cv2.VideoWriter(str(path),fourcc,outfps,(w,h))
  if not vw.isOpened():raise RuntimeError('VideoWriter failed: '+str(path))
  for f in range(start,end+1):vw.write(overlay(cv2.imread(str(IMG/f'frame_{f:06d}.png')),f,events,lc,ec))
  vw.release();video_paths.append(str(path))
 cand={'left_arm_return_near_home':{'status':'USER_REVIEW_REQUIRED','candidates':candidates('left_arm_return_near_home',lr,lv)},'task_end':{'status':'USER_REVIEW_REQUIRED','candidates':candidates('task_end',er,ev)},'automatic_approval':False,'approved_timeline_modified':False}
 (OUT/'candidate_frames.json').write_text(json.dumps(cand,indent=2)+'\n')
 manifest={'status':'EPISODE49_END_TIMELINE_CANDIDATES_READY_FOR_USER_REVIEW','source_episode':49,'source_range':[start,end],'fps':fps,'raw_image_count':len(imgs),'action_shape':list(a.shape),'timeline_sha256_before':sha(TIMELINE),'timeline_events_before':[(e['event'],e['frame']) for e in timeline['events']],'missing_events':sorted({'left_arm_return_near_home','task_end'}-names),'coarse_sheets':coarse,'left_return_fine_sheets':leftfine,'task_end_fine_sheets':endfine,'videos':video_paths,'automatic_approval':False,'approved_timeline_modified':False}
 (OUT/'review_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 first=lambda xs:xs[0] if xs else 'MISSING'
 report=f'''# Episode 49 end timeline review\n\nStatus: `EPISODE49_END_TIMELINE_CANDIDATES_READY_FOR_USER_REVIEW`\n\nReview range: frames {start}–{end}. Action thresholds are diagnostic only. No event was approved.\n\n- Left-return candidates: {lc}\n- Task-end candidates: {ec}\n- Coarse: `{first(coarse)}`\n- Left fine: `{first(leftfine)}`\n- Task-end fine: `{first(endfine)}`\n- Slow video: `{video_paths[1]}`\n\nNO EVENT FRAME WAS AUTOMATICALLY APPROVED\nNO TRAJECTORY GENERATED\nSIMULATION ONLY\n''';(OUT/'report.md').write_text(report)
 commands=f'''#!/usr/bin/env bash\nxdg-open {first(coarse)}\nxdg-open {first(leftfine)}\nxdg-open {first(endfine)}\nxdg-open {video_paths[1]}\n''';(OUT/'commands.sh').write_text(commands);os.chmod(OUT/'commands.sh',0o755)
 print(json.dumps({'status':manifest['status'],'range':[start,end],'left_candidates':lc,'task_end_candidates':ec,'coarse_pages':len(coarse),'left_fine_pages':len(leftfine),'task_end_fine_pages':len(endfine)},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
