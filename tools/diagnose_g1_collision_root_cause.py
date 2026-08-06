#!/usr/bin/env python3
"""Offline, non-corrective root-cause experiment for G1 target contacts."""
from __future__ import annotations
import argparse,csv,html,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');SRC=ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz';ARM=ROOT/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz';CFG=ROOT/'configs/dex3_magsafe_grasp_primitives.sim.json';XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml');OUT=ROOT/'outputs/collision_root_cause';KEYS=(224,248,251,280,285,298,345,622,627)
def side(b):return 'left' if b.startswith('left_') else 'right' if b.startswith('right_') else None
def classify(b1,b2,parent=False):
 s1,s2=side(b1),side(b2)
 if parent:return 'adjacent-link'
 if s1==s2 and s1 and (('wrist' in b1 and 'hand_' in b2) or ('wrist' in b2 and 'hand_' in b1)):return 'same-hand wrist–finger'
 if ('hand' in b1 and 'torso' in b2) or ('hand' in b2 and 'torso' in b1):return 'hand–torso'
 if b1.startswith('left_hand') and b2.startswith('right_hand') or b2.startswith('left_hand') and b1.startswith('right_hand'):return 'hand–hand'
 if s1 and s2 and s1!=s2 and (('hand' in b1 and any(x in b2 for x in ('wrist','elbow','shoulder'))) or ('hand' in b2 and any(x in b1 for x in ('wrist','elbow','shoulder')))):return 'cross-arm–hand'
 if s1==s2 and s1 and 'hand_' in b1 and 'hand_' in b2:return 'same-hand internal'
 return 'other'
def names(model,c):
 import mujoco
 bs=[];gs=[]
 for g in (c.geom1,c.geom2):
  gs.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or f'geom_{g}')
  bs.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or 'world')
 return gs,bs
def contacts(model,data,q,ts,lp,rp,variant,filtered=False):
 import mujoco
 out=[]
 for f,x in enumerate(q):
  data.qpos[:]=x;data.qvel[:]=0;mujoco.mj_forward(model,data)
  for c in data.contact:
   gs,bs=names(model,c);ids=[int(model.geom_bodyid[g]) for g in (c.geom1,c.geom2)];parent=model.body_parentid
   cat=classify(*bs,parent=parent[ids[0]]==ids[1] or parent[ids[1]]==ids[0])
   if filtered and ('hand_' in bs[0] or 'hand_' in bs[1]):continue
   out.append(dict(variant=variant,frame=f,timestamp=float(ts[f]),geom1=gs[0],geom2=gs[1],body1=bs[0],body2=bs[1],position_x=float(c.pos[0]),position_y=float(c.pos[1]),position_z=float(c.pos[2]),normal_x=float(c.frame[0]),normal_y=float(c.frame[1]),normal_z=float(c.frame[2]),signed_distance_m=float(c.dist),penetration_m=max(0.,-float(c.dist)),left_phase=str(lp[f]),right_phase=str(rp[f]),category=cat))
 return out
def geom_mesh_name(m,g):
 import mujoco
 mid=int(m.geom_dataid[g]);return mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MESH,mid) if mid>=0 else None
def geom_ids(m,pred):
 import mujoco
 out=[]
 for g in range(m.ngeom):
  b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '';mesh=geom_mesh_name(m,g) or ''
  if pred(b,mesh,g):out.append(g)
 return out
def mindist(m,d,A,B,cut=.5):
 import mujoco
 best=cut;seg=np.zeros(6)
 for a in A:
  for b in B:
   x=np.zeros(6);v=float(mujoco.mj_geomDistance(m,d,a,b,cut,x))
   if v<best:best=v;seg=x.copy()
 return best,seg
def ranges(xs):
 xs=sorted(set(xs));out=[]
 for x in xs:
  if not out or x>out[-1][1]+1:out.append([x,x])
  else:out[-1][1]=x
 return out
def summarize(rows):
 cats={}
 for c in sorted({r['category'] for r in rows}):
  z=[r for r in rows if r['category']==c];cats[c]={'records':len(z),'unique_frames':len({r['frame'] for r in z}),'first_frame':min(r['frame'] for r in z),'maximum_penetration_m':max(r['penetration_m'] for r in z)}
 return {'records':len(rows),'unique_frames':len({r['frame'] for r in rows}),'categories':cats}
def render_frames(m,d,q,rows,out):
 import mujoco,imageio.v2 as imageio
 from PIL import Image,ImageDraw
 views={'front':(90,-8),'side':(0,-8),'top':(90,-89)}
 renderer=mujoco.Renderer(m,480,640)
 for f in KEYS:
  fd=out/'frames'/str(f);fd.mkdir(parents=True,exist_ok=True);d.qpos[:]=q[f];mujoco.mj_forward(m,d);rr=[r for r in rows if r['frame']==f]
  label=' | '.join(f"{r['body1']} ↔ {r['body2']} {r['penetration_m']*1000:.2f} mm" for r in rr) or 'no active contact'
  for mode in ('normal','collision'):
   opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[2 if mode=='normal' else 3]=1
   if mode=='collision':opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=1;opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=1
   for view,(az,el) in views.items():
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[.10,0,1.0];cam.distance=1.05;cam.azimuth=az;cam.elevation=el
    renderer.update_scene(d,camera=cam,scene_option=opt);im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,640,66),fill=(0,0,0));phase=(rr[0]['left_phase']+'/'+rr[0]['right_phase']) if rr else 'no contact';dr.text((8,5),f'frame {f} {mode} {view} phases={phase}',fill='white');dr.text((8,25),label[:110],fill='yellow');dr.text((8,45),'red contact markers / normal-force arrows (MuJoCo)',fill='red');im.save(fd/f'{mode}_{view}.png')
 renderer.close()
def montage(paths,out,title):
 from PIL import Image,ImageDraw
 imgs=[Image.open(p).resize((400,300)) for p in paths if p.exists()]
 if not imgs:return
 canvas=Image.new('RGB',(400*min(4,len(imgs)),330*((len(imgs)+3)//4)),(255,255,255));dr=ImageDraw.Draw(canvas);dr.text((5,5),title,fill='black')
 for i,im in enumerate(imgs):canvas.paste(im,((i%4)*400,(i//4)*330+30))
 canvas.save(out)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=SRC);p.add_argument('--arm-source',type=Path,default=ARM);p.add_argument('--primitives',type=Path,default=CFG);p.add_argument('--xml',type=Path,default=XML);p.add_argument('--output-dir',type=Path,default=OUT);a=p.parse_args();import mujoco
 a.output_dir.mkdir(parents=True,exist_ok=True)
 with np.load(a.input,allow_pickle=False) as z:q=z['full_qpos'].astype(float);arm=z['arm_qpos'].astype(float);ts=z['timestamps'];lp=z['left_phase'];rp=z['right_phase'];fn=z['full_joint_names'].astype(str);an=z['arm_joint_names'].astype(str);ln=z['left_dex3_joint_names'].astype(str);rn=z['right_dex3_joint_names'].astype(str)
 cfg=json.loads(a.primitives.read_text());qo=q.copy()
 for ns,key in ((ln,'LEFT_PHONE_OPEN'),(rn,'RIGHT_ACCESSORY_OPEN')):
  vals=np.asarray(cfg['primitives'][key]['qpos'],float)
  for n,v in zip(ns,vals):qo[:,list(fn).index(n)]=v
 ai=[list(fn).index(n) for n in an];assert np.array_equal(q[:,ai],arm) and np.array_equal(qo[:,ai],arm)
 m=mujoco.MjModel.from_xml_path(str(a.xml));d=mujoco.MjData(m);cur=contacts(m,d,q,ts,lp,rp,'current_full');opn=contacts(m,d,qo,ts,lp,rp,'all_fingers_open');masked=contacts(m,d,q,ts,lp,rp,'arm_wrist_only_collision',True)
 allrows=cur+opn+masked;cols=list(allrows[0]);
 with (a.output_dir/'collision_events_reclassified.csv').open('w',newline='') as f:w=csv.DictWriter(f,cols);w.writeheader();w.writerows(allrows)
 np.savez_compressed(a.output_dir/'diagnostic_variants.npz',current_full_qpos=q,all_fingers_open_qpos=qo,arm_qpos=arm,full_joint_names=fn,arm_joint_names=an,left_dex3_joint_names=ln,right_dex3_joint_names=rn,timestamps=ts,position_only_qpos=q,position_only_distinct=np.array(False))
 def pairs(rs):return defaultdict(list,((k,[x for x in rs if '|'.join(sorted((x['body1'],x['body2'])))==k]) for k in {'|'.join(sorted((x['body1'],x['body2']))) for x in rs}))
 pc,po=pairs(cur),pairs(opn);cc,oc=Counter(r['category'] for r in cur),Counter(r['category'] for r in opn);comparison={'current':summarize(cur),'all_open':summarize(opn),'category_record_difference':{k:cc[k]-oc[k] for k in sorted(set(cc)|set(oc))},'unique_contact_frame_difference':len({r['frame'] for r in cur})-len({r['frame'] for r in opn}),'first_collision_frame_difference':min(r['frame'] for r in cur)-min(r['frame'] for r in opn),'maximum_penetration_difference_m':max(r['penetration_m'] for r in cur)-max(r['penetration_m'] for r in opn),'pairs':{}}
 for key in sorted(set(pc)|set(po)):
  c,o=pc.get(key,[]),po.get(key,[]);comparison['pairs'][key]={'current_frames':ranges([x['frame'] for x in c]),'all_open_frames':ranges([x['frame'] for x in o]),'current_records':len(c),'all_open_records':len(o),'classification':'primitive_sensitive_contact' if c and not o else 'arm_or_wrist_trajectory_sensitive_contact' if o else 'not_applicable'}
 (a.output_dir/'current_vs_all_open.json').write_text(json.dumps(comparison,indent=2)+'\n')
 torso=geom_ids(m,lambda b,mesh,g:b=='torso_link' and m.geom_group[g]==3);wrist={s:geom_ids(m,lambda b,mesh,g,s=s:b==f'{s}_wrist_yaw_link' and 'wrist_yaw' in mesh and m.geom_group[g]==3) for s in ('left','right')};palm={s:geom_ids(m,lambda b,mesh,g,s=s:b==f'{s}_wrist_yaw_link' and mesh==f'{s}_hand_palm_link' and m.geom_group[g]==3) for s in ('left','right')};tip={s:geom_ids(m,lambda b,mesh,g,s=s:b.startswith(f'{s}_hand_') and b.endswith('_1_link') and m.geom_group[g]==3) for s in ('left','right')};hand={s:geom_ids(m,lambda b,mesh,g,s=s:(b.startswith(f'{s}_hand_') or mesh==f'{s}_hand_palm_link') and m.geom_group[g]==3) for s in ('left','right')}
 clear=[]
 for f,x in enumerate(q):
  d.qpos[:]=x;mujoco.mj_forward(m,d);row={'frame':f,'timestamp':float(ts[f])}
  for s in ('left','right'):
   row[f'{s}_wrist_torso_signed_distance_m']=mindist(m,d,wrist[s],torso)[0];row[f'{s}_palm_torso_signed_distance_m']=mindist(m,d,palm[s],torso)[0];row[f'{s}_fingertip_torso_signed_distance_m']=mindist(m,d,tip[s],torso)[0]
  row['inter_wrist_proxy_distance_m']=float(np.linalg.norm(d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'left_wrist_yaw_link')]-d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link')]));row['inter_hand_min_signed_distance_m']=mindist(m,d,hand['left'],hand['right'])[0];clear.append(row)
 with (a.output_dir/'clearance_over_time.csv').open('w',newline='') as f:w=csv.DictWriter(f,clear[0]);w.writeheader();w.writerows(clear)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 for name,keys in [('left_palm_torso_clearance.png',['left_wrist_torso_signed_distance_m','left_palm_torso_signed_distance_m']),('right_palm_torso_clearance.png',['right_wrist_torso_signed_distance_m','right_palm_torso_signed_distance_m']),('inter_hand_clearance.png',['inter_wrist_proxy_distance_m','inter_hand_min_signed_distance_m']),('fingertip_torso_clearance.png',['left_fingertip_torso_signed_distance_m','right_fingertip_torso_signed_distance_m'])]:
  fig,ax=plt.subplots(figsize=(12,4));[ax.plot([r['frame'] for r in clear],[r[k] for r in clear],label=k) for k in keys];ax.axhline(0,color='r',lw=.8);ax.set(xlabel='frame',ylabel='signed distance (m)');ax.legend();fig.tight_layout();fig.savefig(a.output_dir/name,dpi=160);plt.close(fig)
 cats=sorted({r['category'] for r in cur});fig,ax=plt.subplots(figsize=(12,4));[ax.scatter([r['frame'] for r in cur if r['category']==c],[i]*sum(r['category']==c for r in cur),s=5,label=c) for i,c in enumerate(cats)];ax.set(xlabel='frame',yticks=range(len(cats)),yticklabels=cats,title='Current collision timeline');fig.tight_layout();fig.savefig(a.output_dir/'current_collision_timeline.png',dpi=160);plt.close(fig)
 with np.load(a.arm_source,allow_pickle=False) as z:targl=z['g1_target_left_position'];targr=z['g1_target_right_position'];achl=z['g1_achieved_left_position'];achr=z['g1_achieved_right_position']
 win=[]
 for f in range(224,346):
  d.qpos[:]=q[f];mujoco.mj_forward(m,d);lb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'left_wrist_yaw_link');rb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link');row={'frame':f,'timestamp':float(ts[f]),'left_phase':str(lp[f]),'right_phase':str(rp[f]),'inter_wrist_distance_m':float(np.linalg.norm(d.xpos[lb]-d.xpos[rb]))}
  for s,tg,ac,b in [('left',targl,achl,lb),('right',targr,achr,rb)]:row.update({**{f'target_{s}_wrist_{c}':float(tg[f,i]) for i,c in enumerate('xyz')},**{f'achieved_{s}_wrist_{c}':float(ac[f,i]) for i,c in enumerate('xyz')},**{f'target_{s}_quat_{c}':float([1,0,0,0][i]) for i,c in enumerate('wxyz')},**{f'achieved_{s}_quat_{c}':float(d.xquat[b,i]) for i,c in enumerate('wxyz')},f'{s}_wrist_torso_signed_distance_m':clear[f][f'{s}_wrist_torso_signed_distance_m'],f'{s}_palm_torso_signed_distance_m':clear[f][f'{s}_palm_torso_signed_distance_m']})
  for n,v in zip(an,arm[f]):row[f'qpos_{n}']=float(v)
  win.append(row)
 with (a.output_dir/'target_vs_achieved_contact_window.csv').open('w',newline='') as f:w=csv.DictWriter(f,win[0]);w.writeheader();w.writerows(win)
 render_frames(m,d,q,cur,a.output_dir);render_frames(m,d,qo,opn,a.output_dir/'all_open');montage([a.output_dir/'frames'/str(f)/'collision_side.png' for f in KEYS],a.output_dir/'side_view_contact_sequence.png','current contact sequence side');montage([a.output_dir/'frames'/str(f)/'collision_top.png' for f in KEYS],a.output_dir/'top_view_contact_sequence.png','current contact sequence top');montage(sum(([a.output_dir/'frames'/str(f)/'normal_front.png',a.output_dir/'all_open/frames'/str(f)/'normal_front.png'] for f in (224,251,285,298)),[]),a.output_dir/'current_vs_all_open.png','Current then all-open, paired by frame');montage([a.output_dir/'frames'/str(f)/'normal_side.png' for f in (224,251,285,298)],a.output_dir/'current_vs_position_only.png','Position-only is exactly current: original IK orientation weight = 0')
 mins={k:float(min(r[k] for r in clear)) for k in clear[0] if k not in ('frame','timestamp')};curpairs={k:v for k,v in comparison['pairs'].items() if v['current_records']};removed=[k for k,v in curpairs.items() if v['current_records'] and not v['all_open_records']];persist=[k for k,v in curpairs.items() if v['all_open_records']]
 verdict=[]
 if removed:verdict.append('DEX3_PLACEHOLDER_DOMINANT')
 torso_persist=[k for k in persist if 'torso_link' in k]
 if torso_persist or mins['left_palm_torso_signed_distance_m']<0 or mins['right_palm_torso_signed_distance_m']<0:verdict.append('HAND_ANCHOR_TOO_CLOSE_TO_TORSO')
 if any('left_hand' in k and 'right_hand' in k for k in persist):verdict.append('BIMANUAL_TARGET_TOO_NARROW')
 if len(verdict)>1:verdict.append('MIXED_CAUSE')
 if not verdict:verdict=['INSUFFICIENT_EVIDENCE']
 summary={'verdicts':verdict,'current_vs_all_open':comparison,'arm_wrist_only':summarize(masked),'position_only':{'status':'NOT_AVAILABLE_AS_DISTINCT_VARIANT','reason':'authoritative converter already calls temporal_solve with orientation_weight=0.0','qpos_exactly_equal_to_current':True},'clearance_minima_m':mins,'primitive_sensitive_pairs':removed,'arm_or_wrist_trajectory_sensitive_pairs_by_requested_rule':persist,'persistent_pair_interpretation':{'right_hand_thumb_2_link|right_wrist_yaw_link':'same kinematic hand chain; all-open contact at every frame demonstrates simulation OPEN qpos/model self-contact, not a cross-arm collision'},'target_orientation_basis':'identity placeholder arrays in relative_targets; temporal IK orientation objective weight is exactly zero','palm_proxy_basis':{'left':'XML collision mesh left_hand_palm_link attached to left_wrist_yaw_link at local pos [0.0415,0.003,0]','right':'XML collision mesh right_hand_palm_link attached to right_wrist_yaw_link at local pos [0.0415,-0.003,0]'},'trajectory_modified':False,'real_g1_execution':'NOT_APPROVED','real_robot_safety_validation':'NOT_PERFORMED','visual_evidence':str((a.output_dir/'frames').resolve())};(a.output_dir/'root_cause_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 pairlines='\n'.join(f"- `{k}`: current {v['current_frames']}, all-open {v['all_open_frames']}, {v['classification']}" for k,v in comparison['pairs'].items())
 report='# G1 collision root-cause diagnostic\n\n**REAL G1 EXECUTION NOT APPROVED**  \n**REAL G1 SAFETY NOT PERFORMED**\n\n## Verdict\n\n'+', '.join(verdict)+'\n\n## Evidence\n\n- Current: '+json.dumps(summarize(cur))+'\n- All open: '+json.dumps(summarize(opn))+'\n- Arm/wrist-only after finger masking: '+json.dumps(summarize(masked))+'\n- Minimum clearances: '+json.dumps(mins)+'\n- Position-only: not a distinct experiment; authoritative IK already used orientation weight 0.\n- Visual evidence: `frames/<frame>/`, `current_vs_all_open.png`, and clearance plots.\n\n## Pair evidence\n\n'+pairlines+'\n\nNo trajectory was corrected, clamped, or overwritten.\n';(a.output_dir/'root_cause_report.md').write_text(report)
 imgs=''.join(f'<h3>Frame {f}</h3>'+''.join(f'<img width="31%" src="frames/{f}/collision_{v}.png">' for v in ('front','side','top')) for f in (224,251,285,298));dashboard=f'<html><body><h1>G1 collision root-cause dashboard</h1><h2 style="color:red">REAL G1 EXECUTION NOT APPROVED<br>REAL G1 SAFETY NOT PERFORMED</h2><pre>{html.escape(json.dumps(summary,indent=2))}</pre><img width="90%" src="current_collision_timeline.png"><img width="90%" src="current_vs_all_open.png"><img width="90%" src="current_vs_position_only.png"><img width="90%" src="left_palm_torso_clearance.png"><img width="90%" src="right_palm_torso_clearance.png"><img width="90%" src="inter_hand_clearance.png"><img width="90%" src="fingertip_torso_clearance.png">{imgs}</body></html>';(a.output_dir/'index.html').write_text(dashboard)
 print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
