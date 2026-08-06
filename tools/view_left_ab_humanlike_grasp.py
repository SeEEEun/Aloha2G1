#!/usr/bin/env python3
"""Static FK renderer/viewer for persisted human-like candidate; no stepping."""
import argparse,importlib,os,tempfile,time
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_left_ab_humanlike_v6';XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml');PHONE=np.array([.525,.07,.83075]);SIZE=np.array([.1496,.00795,.0715])
def v(x):return ' '.join(map(str,np.asarray(x)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=OUT/'best_failed_humanlike_left_grasp.npz');p.add_argument('--root-forward-offset-m',type=float,default=.15);p.add_argument('--show-phone',action='store_true');p.add_argument('--show-contact-targets',action='store_true');p.add_argument('--show-arm-axes',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--gui',action='store_true');a=p.parse_args();assert abs(a.root_forward_offset_m-.15)<1e-12
 with np.load(a.input) as z:dct={k:z[k] for k in z.files}
 t=XML.read_text();assets=XML.parent/'assets';t=t.replace('meshdir="assets"',f'meshdir="{assets}"');extra=[f'<geom type="box" name="phone" pos="{v(PHONE)}" size="{v(SIZE/2)}" rgba=".2 .5 .9 .5"/>',f'<site pos="{v(dct["target_A"])}" size=".006" rgba="1 0 0 1"/>',f'<site pos="{v(dct["target_B"])}" size=".006" rgba="0 0 1 1"/>',f'<geom type="capsule" fromto="{v(dct["shoulder_position"])} {v(dct["elbow_position"])}" size=".006" rgba="1 .5 0 1" contype="0" conaffinity="0"/>',f'<geom type="capsule" fromto="{v(dct["elbow_position"])} {v(dct["wrist_position"])}" size=".006" rgba="1 1 0 1" contype="0" conaffinity="0"/>'];t=t.replace('<worldbody>','<worldbody>\n'+'\n'.join(extra),1).replace('</mujoco>','<visual><global offwidth="1200" offheight="900"/></visual></mujoco>');td=tempfile.TemporaryDirectory();x=Path(td.name)/'m.xml';x.write_text(t);m=mujoco.MjModel.from_xml_path(str(x));d=mujoco.MjData(m);d.qpos[:]=dct['full_qpos'];mujoco.mj_forward(m,d)
 text=f"BLOCKED candidate | forearm {float(dct['forearm_elevation_deg']):.2f} deg | wrist-neutral {float(dct['wrist_neutral_deviation_deg']):.2f} deg | swivel {float(dct['elbow_swivel_deg']):.1f} deg"
 if a.render:
  views={'front_view':(90,-5,1.5,[.43,.0,1.0]),'side_view':(0,-5,1.5,[.43,.0,1.0]),'top_view':(90,-89,1.3,[.43,.0,.9]),'phone_closeup':(90,-10,.42,PHONE),'elbow_posture_front':(90,-5,.9,dct['elbow_position']),'forearm_alignment_side':(0,-5,.8,(dct['elbow_position']+dct['wrist_position'])/2),'wrist_neutral_closeup':(90,-15,.35,dct['wrist_position']),'comparison_with_previous_failed_pose':(90,-5,1.3,[.43,.0,1.0])}
  for n,(az,el,di,lo) in views.items():
   r=mujoco.Renderer(m,height=800,width=1100);c=mujoco.MjvCamera();c.lookat[:]=lo;c.distance=di;c.azimuth=az;c.elevation=el;r.update_scene(d,c);im=Image.fromarray(r.render());r.close();dr=ImageDraw.Draw(im);dr.rectangle((0,0,1100,35),fill='white');dr.text((8,8),text,fill='black');im.save(OUT/f'{n}.png')
 if a.gui:
  vm=importlib.import_module('mujoco.viewer')
  with vm.launch_passive(m,d) as viewer:
   viewer.cam.lookat[:]=PHONE;viewer.cam.distance=.7;viewer.sync();print('STATIC FK ONLY; no physics step.')
   while viewer.is_running():viewer.sync();time.sleep(.03)
if __name__=='__main__':main()
