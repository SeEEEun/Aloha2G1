#!/usr/bin/env python3
"""Render/approve immutable-object semantic-frame candidates offline."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'));from task_frame_registration import build
CFG=ROOT/'configs/magsafe_object_semantic_frames.sim.json';OUT=ROOT/'outputs/task_frame_registration'
def main():
 p=argparse.ArgumentParser();p.add_argument('--object',choices=('phone','accessory','charger','all'),default='all');p.add_argument('--front',action='store_true');p.add_argument('--side',action='store_true');p.add_argument('--top',action='store_true');p.add_argument('--edit',action='store_true');p.add_argument('--save-candidate',action='store_true');p.add_argument('--approve',action='store_true');p.add_argument('--config',type=Path,default=CFG);a=p.parse_args();build();c=json.load(open(a.config));sel=list(c.keys() & {'phone','accessory','charger'}) if a.object=='all' else [a.object]
 if a.edit:print('Editing is intentionally file-based: edit one fixed region in a copied candidate, record reason, then use --save-candidate. Object root poses must remain unchanged.')
 if a.save_candidate:(OUT/'semantic_frames_candidate.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'config':c,'objects_modified':False,'reviewed_objects':sel},indent=2)+'\n')
 if a.approve:
  pth=OUT/'semantic_frames_candidate.json'
  if not pth.exists():raise SystemExit('Save candidate before approval.')
  d=json.load(open(pth));d['status']='OBJECT_SEMANTIC_FRAMES_APPROVED';d['approved_at']=datetime.now(timezone.utc).isoformat();(OUT/'semantic_frames.approved.json').write_text(json.dumps(d,indent=2)+'\n')
 import matplotlib.pyplot as plt
 from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 import numpy as np
 ad=OUT/'approval_views';ad.mkdir(parents=True,exist_ok=True)
 def box(ax,center,size,color,alpha=.35,label=None):
  cc=np.asarray(center);h=np.asarray(size)/2;v=np.array([[i,j,k] for i in (-1,1) for j in (-1,1) for k in (-1,1)])*h+cc;faces=[[v[i] for i in f] for f in ((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))];ax.add_collection3d(Poly3DCollection(faces,facecolor=color,alpha=alpha,edgecolor=color));ax.text(*cc,label or '',color=color,fontsize=8)
 def arrow(ax,o,v,color,label):o=np.asarray(o);v=np.asarray(v);ax.quiver(*o,*v,color=color,length=.05,normalize=True);ax.text(*(o+v*.055),label,color=color,fontsize=8)
 views={'front':(-90,0),'side':(0,0),'top':(-90,90),'isometric':(-55,25),'phone_accessory_closeup':(-80,15),'charger_closeup':(-80,15)}
 l=json.load(open(ROOT/'isaaclab_magsafe_fixed_scene/scene_layout.json'))
 for view,(az,el) in views.items():
  fig=plt.figure(figsize=(10,8));ax=fig.add_subplot(111,projection='3d')
  ph=c['phone'];pc=ph['root_scene_position_m'];box(ax,pc,ph['size_xyz_m'],'forestgreen',.45,'phone visual/collision bbox');gf=ph['left_side_semantic_grasp_frame'];gc=np.asarray(gf['position_scene_m']);ax.scatter(*gc,color='orange',s=60);ax.text(*gc,'left-side semantic grasp frame\nTOLERANCE NOT_DEFINED',color='darkorange',fontsize=8);arrow(ax,gc,gf['approach_axis_scene'],'gold','approach axis');arrow(ax,pc,ph['back_surface_normal_scene'],'cyan','phone back normal')
  ac=c['accessory'];ap=np.asarray(ac['root_scene_position_m']);th=np.linspace(0,2*np.pi,80);ax.plot(ap[0]+ac['outer_radius_m']*np.cos(th),np.full_like(th,ap[1]),ap[2]+ac['outer_radius_m']*np.sin(th),color='black',lw=8,label='accessory physical annulus; tolerance NOT_DEFINED');arrow(ax,ap,ac['attachment_axis_scene'],'magenta','attachment');arrow(ax,ap,ac['initial_removal_direction_scene'],'red','initial removal outward')
  ch=c['charger'];cp=np.asarray(ch['root_scene_position_m']);box(ax,cp+[0,0,.012],[.105,.105,.024],'gray',.5,'charger base');n=np.asarray(ch['pad_normal_scene']);rad=ch['pad_radius_m'];tilt=np.radians(ch['pad_tilt_deg']);target=cp+np.array([0,.01,.16-rad*np.cos(tilt)]);u=np.linspace(0,2*np.pi,80);ax.plot(target[0]+rad*np.cos(u),target[1]+rad*np.sin(u),np.full_like(u,target[2]),color='purple',lw=3,label='physical pad extent; placement tolerance NOT_DEFINED');arrow(ax,target,n,'blue','charger pad normal 15°')
  for o,name in ((pc,'phone root/center'),(ap,'accessory root/center'),(cp,'charger root')):ax.scatter(*o,s=35);ax.text(*o,name,fontsize=8)
  ax.text2D(.02,.97,'UNAPPROVED TOLERANCE',transform=ax.transAxes,color='red',weight='bold',fontsize=14);ax.set(xlabel='scene X m',ylabel='scene Y m',zlabel='scene Z m');ax.view_init(el,az);ax.legend(fontsize=7)
  if view=='phone_accessory_closeup':ax.set_xlim(.42,.62);ax.set_ylim(.20,.32);ax.set_zlim(.78,.89)
  elif view=='charger_closeup':ax.set_xlim(.34,.50);ax.set_ylim(.44,.60);ax.set_zlim(.78,1.0)
  else:ax.set_xlim(0,.835);ax.set_ylim(0,.72);ax.set_zlim(.75,1.05)
  fig.savefig(ad/f'semantic_frames_{view}.png',dpi=160,bbox_inches='tight');plt.close(fig)
 print(json.dumps({'rendered':[str(ad/f'semantic_frames_{v}.png') for v in views],'objects':{k:c[k] for k in sel}},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
