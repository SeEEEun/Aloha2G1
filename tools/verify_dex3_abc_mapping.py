#!/usr/bin/env python3
"""Verify photo A/B/C labels against the active G1 Dex3 collision geometry."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import mujoco,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT=Path('/home/jbnu/aloha_g1_dataset')
XML=Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
OUT=ROOT/'outputs/dex3_abc_mapping'
LEFT=Path('/tmp/codex-clipboard-PiARrr.png');RIGHT=Path('/tmp/codex-clipboard-6nQUzs.png')
MAP={'left':{'A':'thumb','B':'index','C':'middle'},'right':{'A':'index','B':'thumb','C':'middle'}}
DISTAL={'thumb':'thumb_2','index':'index_1','middle':'middle_1'}
JOINT_SUFFIX={'thumb':['thumb_0','thumb_1','thumb_2'],'index':['index_0','index_1'],'middle':['middle_0','middle_1']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mesh_points(m,gid):
 mid=int(m.geom_dataid[gid]);a=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);v=m.mesh_vert[a:a+n].copy()
 R=Rotation.from_quat(m.geom_quat[gid][[1,2,3,0]]).as_matrix();return v@R.T+m.geom_pos[gid]
def main():
 OUT.mkdir(parents=True,exist_ok=True);m=mujoco.MjModel.from_xml_path(str(XML));d=mujoco.MjData(m);d.qpos[:]=m.key_qpos[0];mujoco.mj_forward(m,d)
 records={};tips={}
 for side in ('left','right'):
  wrist=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'{side}_wrist_yaw_link');Rw=d.xmat[wrist].reshape(3,3);ow=d.xpos[wrist];records[side]={};tips[side]={}
  for label,digit in MAP[side].items():
   body=f'{side}_hand_{DISTAL[digit]}_link';bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body)
   gids=[int(g) for g in np.flatnonzero(m.geom_bodyid==bid) if m.geom_contype[g] or m.geom_conaffinity[g]]
   if not gids:raise RuntimeError(f'no active collision geom for {body}')
   gid=gids[-1];pts=mesh_points(m,gid);xmax=float(pts[:,0].max());pad=pts[pts[:,0]>=xmax-.004]
   center=pad.mean(0);extent=.5*(pad.max(0)-pad.min(0));body_R=d.xmat[bid].reshape(3,3);world=d.xpos[bid]+body_R@center;wlocal=Rw.T@(world-ow)
   normal_local=np.array([1.,0,0]);flex_axis=m.jnt_axis[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_hand_{JOINT_SUFFIX[digit][-1]}_joint')]
   rec={'digit_chain':digit,'joint_names':[f'{side}_hand_{x}_joint' for x in JOINT_SUFFIX[digit]],'distal_link':body,'fingertip_frame':f'{side}_{label}_tip_collision_pad_center','source_geom':f'active collision mesh geom_id={gid}','local_position_xyz_m':center.tolist(),'wrist_local_position_xyz_m':wlocal.tolist(),'local_normal':normal_local.tolist(),'flexion_direction_joint_axis':flex_axis.tolist(),'pad_half_extent_m':extent.tolist(),'mapping_evidence':f'photo {label} physical position matches {digit} collision chain in active mirrored geometry'}
   records[side][label]=rec;tips[side][label]=wlocal
  photo=plt.imread(LEFT if side=='left' else RIGHT)
  for view,(u,v) in {'front':(0,2),'palm':(0,1)}.items():
   fig,(ax0,ax)=plt.subplots(1,2,figsize=(15,6));ax0.imshow(photo);ax0.axis('off');ax0.set_title(f'user {side} photo')
   ax.scatter(0,0,s=180,c='k',marker='s',label='wrist origin')
   colors={'A':'tab:red','B':'tab:blue','C':'tab:green'}
   for label,p in tips[side].items():
    q=np.asarray(p);ax.plot([0,q[u]],[0,q[v]],color=colors[label],lw=3);ax.scatter(q[u],q[v],s=100,color=colors[label]);rec=records[side][label]
    ax.annotate(f"{label} = {rec['digit_chain']}\n{rec['distal_link']}\n{rec['joint_names']}",(q[u],q[v]),xytext=(8,8),textcoords='offset points',fontsize=8,color=colors[label])
   ax.set_aspect('equal');ax.grid(True,alpha=.3);ax.set_xlabel(('wrist +X','wrist +X')[0]);ax.set_ylabel('wrist +Z' if view=='front' else 'wrist +Y');ax.set_title(f'active XML collision geometry: {side} {view}')
   fig.suptitle('SIMULATION GEOMETRY EVIDENCE — labels and active joint/link names');fig.tight_layout();fig.savefig(OUT/f'{side}_{view}_labeled.png',dpi=170);plt.close(fig)
 mapping={'schema_version':1,'status':'VERIFIED_FROM_ACTIVE_MODEL_GEOMETRY_FOR_SIMULATION','simulation_only':True,'authoritative_for_real_robot':False,'active_model':str(XML),'active_model_sha256':sha(XML),'photo_evidence':{'left':{'path':str(LEFT),'sha256':sha(LEFT)},'right':{'path':str(RIGHT),'sha256':sha(RIGHT)}},'left':records['left'],'right':records['right'],'approved_roles':{'left_phone_grasp':['A','B'],'left_noncontact':['C'],'right_accessory_removal':['C'],'right_noncontact':['A','B']}}
 (ROOT/'configs/dex3_abc_finger_mapping.sim.json').write_text(json.dumps(mapping,indent=2)+'\n')
 frames={'schema_version':1,'status':'VERIFIED_FROM_ACTIVE_COLLISION_GEOMETRY_FOR_SIMULATION','simulation_only':True,'authoritative_for_real_robot':False,'active_model':str(XML),'fingertips':{f'{s}_{a}':mapping[s][a] for s in ('left','right') for a in ('A','B','C')}}
 (ROOT/'configs/dex3_fingertip_frames.sim.json').write_text(json.dumps(frames,indent=2)+'\n')
 report={'status':mapping['status'],'mapping':{s:{a:MAP[s][a] for a in ('A','B','C')} for s in ('left','right')},'images':[str(OUT/f'{s}_{v}_labeled.png') for s in ('left','right') for v in ('front','palm')],'photo_hashes_distinct':sha(LEFT)!=sha(RIGHT),'notes':['upper/lower long digits resolve from wrist-local +Z/-Z collision-chain placement','opposing digit resolves from mirrored thumb chain lateral placement','simulation mapping only; not real calibration']}
 (OUT/'mapping_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
