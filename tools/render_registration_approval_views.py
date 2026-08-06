#!/usr/bin/env python3
"""Render approval evidence using the actual G1 MuJoCo visual meshes."""
from __future__ import annotations
import json,math,os,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
LAYOUT=ROOT/'isaaclab_magsafe_fixed_scene/scene_layout.json';POSE=ROOT/'isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json'
def add_axis(wb,origin,R=np.eye(3),scale=.16,prefix='axis'):
 for i,(name,color) in enumerate((('x','1 0 0 .8'),('y','0 1 0 .8'),('z','0 0 1 .8'))):
  end=np.asarray(origin)+R[:,i]*scale;ET.SubElement(wb,'geom',name=f'{prefix}_{name}',type='cylinder',size='.004',fromto=' '.join(map(str,[*origin,*end])),rgba=color,contype='0',conaffinity='0',group='4')
def add_scene(root):
 wb=root.find('worldbody');l=json.load(open(LAYOUT));z=l['table']['surface_height'];sx=l['table']['size_x'];sy=l['table']['size_y']
 ET.SubElement(wb,'geom',name='approval_table',type='box',size=f'{sx/2} {sy/2} .0225',pos=f'{sx/2} {sy/2} {z-.0225}',rgba='.45 .30 .18 .85',contype='0',conaffinity='0')
 phone=np.array([.525,.255,z+l['phone']['size_landscape_xyz'][2]/2]);size=np.array(l['phone']['size_landscape_xyz'])/2
 ET.SubElement(wb,'geom',name='approval_phone',type='box',size=' '.join(map(str,size)),pos=' '.join(map(str,phone)),rgba='.25 .45 .2 1',contype='0',conaffinity='0')
 acc=phone+np.array([0,l['phone']['size_landscape_xyz'][1]/2+l['accessory']['phone_back_clearance']+l['accessory']['main_depth']/2,0]);rad=l['accessory']['main_outer_diameter']/2
 for k in range(24):
  a=2*np.pi*k/24;p=acc+np.array([rad*np.cos(a),0,rad*np.sin(a)]);ET.SubElement(wb,'geom',name=f'approval_accessory_{k}',type='sphere',size='.0032',pos=' '.join(map(str,p)),rgba='.04 .04 .04 1',contype='0',conaffinity='0')
 ch=np.array([*l['charger']['center_xy'],z+(l['charger']['mount_plate']['size_xyz'][2] if l['charger']['mount_plate']['enabled'] else 0)]);ET.SubElement(wb,'geom',name='approval_charger_base',type='box',size=f"{l['charger']['base_size_xy'][0]/2} {l['charger']['base_size_xy'][1]/2} {l['charger']['base_height']/2}",pos=' '.join(map(str,ch+[0,0,l['charger']['base_height']/2])),rgba='.12 .12 .12 1',contype='0',conaffinity='0')
 tilt=np.radians(l['charger']['pad_tilt_degrees_up']);pad=ch+np.array([0,l['charger']['pad_center_y_offset'],l['charger']['total_height']-l['charger']['pad_diameter']/2*np.cos(tilt)]);q=[math.cos((math.pi/2-tilt)/2),math.sin((math.pi/2-tilt)/2),0,0]
 ET.SubElement(wb,'geom',name='approval_charger_pad',type='cylinder',size=f"{l['charger']['pad_diameter']/2} {l['charger']['pad_thickness']/2}",pos=' '.join(map(str,pad)),quat=' '.join(map(str,q)),rgba='.08 .08 .08 1',contype='0',conaffinity='0')
 add_axis(wb,[0,0,0],prefix='scene');Rt=np.column_stack(([0,1,0],[-1,0,0],[0,0,1]));add_axis(wb,[sx/2,0,z],Rt,prefix='task')
 return phone,acc,ch
def model_data():
 import mujoco
 root=ET.parse(XML).getroot();base=XML.parent
 meshdir=root.find('compiler').get('meshdir','') if root.find('compiler') is not None else ''
 for x in root.findall('.//*[@file]'):
  p=Path(x.get('file'));prefix=base/(meshdir if x.tag=='mesh' else '')
  x.set('file',str((prefix/p).resolve()) if not p.is_absolute() else str(p))
 phone,acc,ch=add_scene(root);m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m);pose=json.load(open(POSE))['g1'];d.qpos[:3]=pose['position_xyz_m'];d.qpos[3:7]=pose['orientation_wxyz'];mujoco.mj_forward(m,d);return m,d,phone,acc,ch
def metrics(m,d):
 import mujoco
 bid=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n);pelvis=d.xpos[bid('pelvis')];torso=d.xpos[bid('torso_link')];feet=[bid('left_ankle_roll_link'),bid('right_ankle_roll_link')]
 gids=[i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in feet and int(m.geom_type[i])==int(mujoco.mjtGeom.mjGEOM_SPHERE) and m.geom_size[i,0]<=.006];footz=min(float(d.geom_xpos[i,2]-m.geom_size[i,0]) for i in gids);table=.795;root=np.array(d.qpos[:3]);forward=d.xmat[bid('pelvis')].reshape(3,3)[:,0];charger=np.array([.42,.52,.807]);direction=(charger-root);direction/=np.linalg.norm(direction)
 return {'root_frame_meaning':'MuJoCo freejoint qpos belongs to body pelvis; preview root Xform positions the imported articulation asset','pelvis_height_m':float(pelvis[2]),'torso_height_m':float(torso[2]),'foot_minimum_z_conservative_m':footz,'table_surface_z_m':table,'pelvis_table_height_difference_m':float(pelvis[2]-table),'foot_ground_distance_m':footz,'g1_to_table_front_edge_m':.5,'g1_torso_forward_axis_scene':forward.tolist(),'charger_direction_from_root_scene':direction.tolist()}
def render_all(out):
 import mujoco
 os.environ.setdefault('MUJOCO_GL','egl');m,d,_,_,_=model_data();met=metrics(m,d);out.mkdir(parents=True,exist_ok=True);views={'front':(180,-5,2.2),'side':(90,-5,2.0),'top':(180,-89,2.4),'isometric':(135,-25,2.3)}
 for name,(azi,ele,dist) in views.items():
  cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[.4175,.18,.9];cam.azimuth=azi;cam.elevation=ele;cam.distance=dist
  with mujoco.Renderer(m,480,640) as r:r.update_scene(d,cam);img=r.render()
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(12,9));ax.imshow(img);ax.axis('off');p=json.load(open(POSE))['g1'];txt=f"G1 actual visual mesh + fixed scene geometry | {name}\nroot body: pelvis/freejoint | t={p['position_xyz_m']} m | q(wxyz)={p['orientation_wxyz']}\npelvis z={met['pelvis_height_m']:.3f} m | foot min z≈{met['foot_minimum_z_conservative_m']:.3f} m | table z={met['table_surface_z_m']:.3f} m\npelvis-table={met['pelvis_table_height_difference_m']:.3f} m | root-front-edge=0.500 m\nred/green/blue axes: +X/+Y/+Z; scene origin and task frame shown"
  ax.text(.01,.99,txt,transform=ax.transAxes,va='top',color='white',bbox={'facecolor':'black','alpha':.72});fig.savefig(out/f'g1_scene_{name}.png',dpi=140,bbox_inches='tight');plt.close(fig)
 (out/'g1_scene_metrics.json').write_text(json.dumps(met,indent=2)+'\n');return met
if __name__=='__main__':render_all(ROOT/'outputs/task_frame_registration/approval_views')
