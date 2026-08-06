#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');SRC=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz';OUT=ROOT/'outputs/behavior_comparison/collision';XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
def classify(b1,b2):
 sides=lambda b: 'left' if b.startswith('left_') else 'right' if b.startswith('right_') else None
 s1,s2=sides(b1),sides(b2)
 if s1==s2 and s1 and (('wrist' in b1 and 'hand_' in b2) or ('wrist' in b2 and 'hand_' in b1)):return 'same-hand wrist–finger'
 if b1.startswith('left_hand') and b2.startswith('left_hand') or b1.startswith('right_hand') and b2.startswith('right_hand'):return 'same-hand internal'
 if ('hand' in b1 and 'torso' in b2) or ('hand' in b2 and 'torso' in b1):return 'hand–torso'
 if b1.startswith('left_hand') and b2.startswith('right_hand') or b2.startswith('left_hand') and b1.startswith('right_hand'):return 'hand–hand'
 if s1 and s2 and s1!=s2 and (('hand' in b1 and any(x in b2 for x in ('wrist','elbow','shoulder'))) or ('hand' in b2 and any(x in b1 for x in ('wrist','elbow','shoulder')))):return 'cross-arm–hand'
 return 'other'
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=SRC);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--xml',type=Path,default=XML);a=p.parse_args();import mujoco
 with np.load(a.input,allow_pickle=False) as z:q=z['full_qpos'];ts=z['timestamps'];lp=z['left_phase'];rp=z['right_phase']
 m=mujoco.MjModel.from_xml_path(str(a.xml));d=mujoco.MjData(m);rows=[]
 for f,x in enumerate(q):
  d.qpos[:]=x;d.qvel[:]=0;mujoco.mj_forward(m,d)
  for c in d.contact:
   g1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or f'geom_{c.geom1}';g2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or f'geom_{c.geom2}';b1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom1])) or 'world';b2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[c.geom2])) or 'world';depth=max(0.,-float(c.dist));cat=classify(b1,b2);sev='none' if depth<=0 else 'shallow' if depth<.001 else 'moderate' if depth<.005 else 'severe';rows.append({'frame':f,'timestamp':float(ts[f]),'geom1':g1,'geom2':g2,'body1':b1,'body2':b2,'position_x':float(c.pos[0]),'position_y':float(c.pos[1]),'position_z':float(c.pos[2]),'normal_x':float(c.frame[0]),'normal_y':float(c.frame[1]),'normal_z':float(c.frame[2]),'penetration_m':depth,'signed_distance_m':float(c.dist),'left_phase':str(lp[f]),'right_phase':str(rp[f]),'category':cat,'severity':sev})
 a.output_dir.mkdir(parents=True,exist_ok=True);cols=list(rows[0]) if rows else ['frame'];
 with (a.output_dir/'collision_events.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 counts=Counter((r['category'],r['body1']+'|'+r['body2']) for r in rows)
 with (a.output_dir/'contact_pair_counts.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['category','body_pair','count']);w.writerows([(k[0],k[1],v) for k,v in sorted(counts.items())])
 cats=sorted(set(r['category'] for r in rows));first={c:min((r['frame'] for r in rows if r['category']==c),default=None) for c in cats};maximum={c:max((r['penetration_m'] for r in rows if r['category']==c),default=0) for c in cats};(a.output_dir/'first_contact_frames.json').write_text(json.dumps(first,indent=2)+'\n')
 summary={'structural_validation':'PASS','collision_validation':'CONTACTS_PRESENT_REQUIRES_REVIEW' if rows else 'PASS_NO_CONTACTS','real_robot_safety_validation':'NOT_PERFORMED','contact_records':len(rows),'frames_with_contact':len(set(r['frame'] for r in rows)),'category_counts':dict(Counter(r['category'] for r in rows)),'first_frames':first,'maximum_penetration_m':maximum,'maximum_penetration_record':max(rows,key=lambda r:r['penetration_m']) if rows else None,'xml_explicit_contact_exclusion_count':int(m.nexclude),'trajectory_modified':False};(a.output_dir/'collision_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(12,4));
 for j,c in enumerate(cats):ax.scatter([r['frame'] for r in rows if r['category']==c],[j]*sum(r['category']==c for r in rows),s=4,label=c)
 ax.set_xlabel('frame');ax.set_yticks(range(len(cats)),cats);ax.set_title('G1 generated target contact timeline');fig.tight_layout();fig.savefig(a.output_dir/'contact_timeline.png',dpi=160);plt.close(fig)
 # Offline key-frame renders; images are diagnostic, never corrective.
 try:
  renderer=mujoco.Renderer(m,480,640)
  keys={}
  for c in ('hand–torso','hand–hand','arm–hand'):
   rr=[r for r in rows if r['category']==c]
   if rr:keys['first_'+c]=min(rr,key=lambda r:r['frame']);keys['max_'+c]=max(rr,key=lambda r:r['penetration_m'])
  for name,r in keys.items():d.qpos[:]=q[r['frame']];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=-1);import imageio.v2 as imageio;imageio.imwrite(a.output_dir/(name.replace('–','_').replace(' ','_')+'.png'),renderer.render())
  renderer.close()
 except Exception as exc:summary['key_frame_rendering']=f'NOT_AVAILABLE: {exc}';(a.output_dir/'collision_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
