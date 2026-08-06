#!/usr/bin/env python3
"""Render/view a persisted failed candidate. Never solves IK or steps physics."""
from __future__ import annotations
import argparse,json,os,tempfile,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/left_ab_view_mpl')
import mujoco,numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/home/jbnu/aloha_g1_dataset');DEFAULT=ROOT/'outputs/left_ab_grasp_visual_diagnosis/best_failed_candidate.npz';OUT=DEFAULT.parent;XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml');PHONE=np.array([.525,.07,.83075]);SIZE=np.array([.1496,.00795,.0715])
def args():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=DEFAULT);p.add_argument('--scene',type=Path);p.add_argument('--root-forward-offset-m',type=float,default=.15);p.add_argument('--show-targets',action='store_true');p.add_argument('--show-achieved',action='store_true');p.add_argument('--show-collisions',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--gui',action='store_true');return p.parse_args()
def vec(v):return ' '.join(f'{x:.10g}' for x in v)
def capsule(a,b,r,color,name):return f'<geom name="{name}" type="capsule" fromto="{vec(a)} {vec(b)}" size="{r}" rgba="{color}" contype="0" conaffinity="0"/>'
def sphere(p,r,color,name):return f'<site name="{name}" pos="{vec(p)}" size="{r}" rgba="{color}"/>'
def scene_model(z,q):
 text=XML.read_text();assets=XML.parent/'assets';text=text.replace('meshdir="assets"',f'meshdir="{assets}"').replace('meshdir="assets/"',f'meshdir="{assets}/"')
 extra=[f'<geom name="phone_collision" type="box" pos="{vec(PHONE)}" size="{vec(SIZE/2)}" rgba=".35 .55 .95 .48" contype="1" conaffinity="1"/>',f'<geom name="screen_surface" type="box" pos="{vec(PHONE-np.array([0,SIZE[1]/2+.0003,0]))}" size="{vec([SIZE[0]/2,.00025,SIZE[2]/2])}" rgba=".1 .8 1 .75" contype="0" conaffinity="0"/>',f'<geom name="back_surface" type="box" pos="{vec(PHONE+np.array([0,SIZE[1]/2+.0003,0]))}" size="{vec([SIZE[0]/2,.00025,SIZE[2]/2])}" rgba="1 .55 .1 .65" contype="0" conaffinity="0"/>',f'<geom name="phone_left_edge" type="box" pos="{vec(PHONE-np.array([SIZE[0]/2,0,0]))}" size=".0007 {SIZE[1]/2} {SIZE[2]/2}" rgba="1 1 0 1" contype="0" conaffinity="0"/>','<geom name="table_collision" type="box" pos=".4175 .36 .7725" size=".4175 .36 .0225" rgba=".5 .5 .5 .35" contype="1" conaffinity="1"/>']
 colors={'A_target_position':'1 0 0 1','A_achieved_position':'.5 0 0 1','B_target_position':'0 .35 1 1','B_achieved_position':'0 0 .45 1','C_achieved_position':'0 1 0 1','AB_midpoint_target':'1 1 0 1','AB_midpoint_achieved':'.6 0 .8 1'}
 for k,c in colors.items():extra.append(sphere(z[k],.005,c,k))
 meta_path=OUT/'best_failed_candidate.json'
 if meta_path.exists():
  for i,c in enumerate(json.loads(meta_path.read_text()).get('contact_pairs',[])):
   if 'position' in c:extra.append(sphere(np.asarray(c['position']),.004,'1 .35 0 1',f'collision_{i:02d}'))
 extra += [capsule(z['A_target_position'],z['A_achieved_position'],.0015,'1 0 1 1','A_error'),capsule(z['B_target_position'],z['B_achieved_position'],.0015,'1 0 1 1','B_error'),capsule(z['AB_midpoint_target'],z['AB_midpoint_target']+.055*z['pinch_axis_target'],.0018,'1 1 0 1','target_axis'),capsule(z['AB_midpoint_achieved'],z['AB_midpoint_achieved']+.055*z['pinch_axis_achieved'],.0018,'.6 0 .8 1','achieved_axis')]
 for prefix,pos,R in [('palm',z['palm_position'],z['palm_rotation']),('wrist',z['wrist_achieved_position'],z['wrist_achieved_rotation'])]:
  for i,c in enumerate(('1 0 0 1','0 1 0 1','0 0 1 1')):extra.append(capsule(pos,pos+.035*R[:,i],.0012,c,f'{prefix}_axis_{i}'))
 text=text.replace('<worldbody>','<worldbody>\n'+'\n'.join(extra),1).replace('</mujoco>','<visual><global offwidth="1300" offheight="1000"/></visual>\n</mujoco>');td=tempfile.TemporaryDirectory(prefix='left_ab_view_');p=Path(td.name)/'scene.xml';p.write_text(text);m=mujoco.MjModel.from_xml_path(str(p));d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d);return m,d,td
def metrics(z,meta):
 ae=np.linalg.norm(z['A_achieved_position']-z['A_target_position'])*1000;be=np.linalg.norm(z['B_achieved_position']-z['B_target_position'])*1000;me=np.linalg.norm(z['AB_midpoint_achieved']-z['AB_midpoint_target'])*1000;ang=np.degrees(np.arccos(np.clip(np.dot(z['pinch_axis_target'],z['pinch_axis_achieved']),-1,1)));cl=z['clearance_values'];palm=next((x[1] for x in meta.get('clearance_values',[]) if x[0]=='left_wrist_yaw_link'),float('nan'));return f"A {ae:.2f} mm | B {be:.2f} mm | midpoint {me:.2f} mm | pinch {ang:.2f} deg\nwrist {meta['position_error_m']*1000:.2f} mm / {meta['orientation_error_deg']:.2f} deg | joint margin {min(meta['joint_margins']):.3f} rad | min phone {np.min(cl) if len(cl) else np.nan:.4f} m | palm proxy {palm:.4f} m | contacts {len(meta['contact_pairs'])}"
def render_one(m,d,path,az,el,dist,look,text):
 r=mujoco.Renderer(m,height=800,width=1100);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=look;cam.distance=dist;cam.azimuth=az;cam.elevation=el;r.update_scene(d,cam);im=Image.fromarray(r.render());r.close();dr=ImageDraw.Draw(im);dr.rectangle((0,0,1100,58),fill=(255,255,255,220));dr.multiline_text((10,7),text,fill='black',spacing=3);im.save(path)
def main():
 a=args();assert abs(a.root_forward_offset_m-.15)<1e-12
 with np.load(a.input,allow_pickle=False) as z0:z={k:z0[k] for k in z0.files}
 meta=json.loads(a.input.with_suffix('.json').read_text());q=z['full_qpos'];m,d,tmp=scene_model(z,q);txt=metrics(z,meta);look=PHONE
 if a.render:
  views={'overview_front':(90,-5,1.55,np.array([.43,.02,1.0])),'overview_side':(0,-5,1.55,np.array([.43,.02,1.0])),'overview_top':(90,-89,1.35,np.array([.43,.02,.90])),'phone_closeup_front':(90,0,.38,look),'phone_closeup_side':(0,0,.38,look),'phone_closeup_top':(90,-89,.38,look),'palm_closeup':(130,-20,.32,z['palm_position']),'fingertip_target_vs_achieved':(90,-15,.28,look),'collision_closeup':(45,-15,.42,look),'transform_axes_comparison':(120,-25,.42,look)}
  for n,(az,el,di,lo) in views.items():render_one(m,d,OUT/f'{n}.png',az,el,di,lo,txt)
  # Same-camera comparison A/B/C/D; C is explicitly geometry-only markers.
  natural=q.copy();natural[16:23]=0;openq=q.copy();openq[37:44]=0
  panels=[]
  for label,qq in [('A best failed',q),('B natural + targets',natural),('C local relation markers (no ghost IK)',natural),('D achieved arm + open fingers',openq)]:
   mm,dd,tt=scene_model(z,qq);rr=mujoco.Renderer(mm,height=500,width=650);cam=mujoco.MjvCamera();cam.lookat[:]=look;cam.distance=.65;cam.azimuth=90;cam.elevation=-12;rr.update_scene(dd,cam);im=Image.fromarray(rr.render());rr.close();ImageDraw.Draw(im).text((8,8),label,fill='black',stroke_width=2,stroke_fill='white');panels.append(im)
  canvas=Image.new('RGB',(1300,1000),'white');[canvas.paste(im,((i%2)*650,(i//2)*500)) for i,im in enumerate(panels)];canvas.save(OUT/'pose_comparison.png')
 if a.gui:
  os.environ.pop('MUJOCO_GL',None);import importlib;viewer_module=importlib.import_module('mujoco.viewer')
  with viewer_module.launch_passive(m,d) as viewer:
   viewer.cam.lookat[:]=PHONE;viewer.cam.distance=.65;viewer.sync()
   print('STATIC FK ONLY; close viewer to exit. No mj_step is called.')
   while viewer.is_running():viewer.sync();time.sleep(.03)
 return 0
if __name__=='__main__':raise SystemExit(main())
