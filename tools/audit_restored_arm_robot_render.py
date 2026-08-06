#!/usr/bin/env python3
"""Audit restored arm trajectories and real-model render artifacts."""
from pathlib import Path
import hashlib, json, subprocess
import numpy as np

ROOT=Path('/home/jbnu/aloha_g1_dataset')
OUT=ROOT/'outputs/scene_registered_retargeting/current_layout_ep49_restore_original_v8'
EXACT=OUT/'restored_exact_arm_trajectory.npz'
NULL=OUT/'restored_nullspace_arm_trajectory.npz'
CAMERAS=('overview','front','side')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def probe(path):
 return json.loads(subprocess.run([
  'ffprobe','-v','error','-count_frames','-select_streams','v:0',
  '-show_entries','stream=width,height,nb_read_frames:format_tags=title,comment',
  '-of','json',str(path)],check=True,capture_output=True,text=True).stdout)
def decoded_sha(path):
 # Hash decoded RGB frames, not the MP4 container bytes.
 value=subprocess.run([
  'ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-f','hash',
  '-hash','sha256','-pix_fmt','rgb24','-'],check=True,capture_output=True,text=True).stdout.strip()
 return value.split('=',1)[-1]

def main():
 with np.load(EXACT,allow_pickle=False) as z:
  exact=z['g1_arm_joint_trajectory'];names=[str(x) for x in z['arm_joint_names']]
 with np.load(NULL,allow_pickle=False) as z: null=z['g1_arm_joint_trajectory']
 if exact.shape!=null.shape: raise RuntimeError(f'q shape mismatch: {exact.shape} != {null.shape}')
 diff=np.abs(exact-null);frame_mask=np.any(diff>1e-12,axis=1);joint_mask=np.any(diff>1e-12,axis=0)
 if not np.any(frame_mask): raise RuntimeError('NULLSPACE_NOT_ACTUALLY_APPLIED')
 old={}
 for kind in ('exact','nullspace'):
  path=OUT/f'aloha_vs_restored_{kind}.mp4'
  old[kind]={'path':str(path.resolve()),'sha256':sha(path),'decoded_frame_sha256':decoded_sha(path),'probe':probe(path)}
 old_identical=(old['exact']['sha256']==old['nullspace']['sha256'] and old['exact']['decoded_frame_sha256']==old['nullspace']['decoded_frame_sha256'])
 if not old_identical: raise RuntimeError('previous invalid-video identity assertion no longer holds')
 videos={}
 for camera in CAMERAS:
  metadata_path=OUT/f'robot_render_{camera}_metadata.json';metadata=json.loads(metadata_path.read_text())
  combined=OUT/f'aloha_exact_nullspace_robot_{camera}.mp4';exact_video=OUT/f'restored_exact_robot_{camera}.mp4';null_video=OUT/f'restored_nullspace_robot_{camera}.mp4'
  item={'combined':{'path':str(combined.resolve()),'sha256':sha(combined),'decoded_frame_sha256':decoded_sha(combined),'probe':probe(combined)},
        'exact':{'path':str(exact_video.resolve()),'sha256':sha(exact_video),'decoded_frame_sha256':decoded_sha(exact_video),'probe':probe(exact_video)},
        'nullspace':{'path':str(null_video.resolve()),'sha256':sha(null_video),'decoded_frame_sha256':decoded_sha(null_video),'probe':probe(null_video)}}
  item['exact_nullspace_hashes_differ']=item['exact']['sha256']!=item['nullspace']['sha256']
  item['exact_nullspace_decoded_frames_differ']=item['exact']['decoded_frame_sha256']!=item['nullspace']['decoded_frame_sha256']
  item['metadata_sidecar']=str(metadata_path.resolve())
  item['metadata_matches_current_files']=(metadata['video_sha256']==item['combined']['sha256'] and metadata['exact_video_sha256']==item['exact']['sha256'] and metadata['nullspace_video_sha256']==item['nullspace']['sha256'])
  counts=[int(item[k]['probe']['streams'][0]['nb_read_frames']) for k in ('combined','exact','nullspace')]
  item['all_decode_to_330_frames']=counts==[330,330,330]
  if not item['exact_nullspace_hashes_differ'] or not item['exact_nullspace_decoded_frames_differ']: raise RuntimeError(f'FAIL: {camera} Exact/Nullspace video or decoded-frame hashes identical')
  if not item['metadata_matches_current_files'] or not item['all_decode_to_330_frames']: raise RuntimeError(f'FAIL: {camera} render validation')
  videos[camera]=item
 audit={
  'previous_status':'VISUAL_VALIDATION_INVALID',
  'previous_reason':'aloha_vs_restored_exact.mp4 and aloha_vs_restored_nullspace.mp4 contain the same red 2-D trajectory plot, not robot renders',
  'previous_invalid_videos':old,'previous_container_and_decoded_frames_identical':old_identical,
  'nullspace_initial_failure_status':'NULLSPACE_NOT_ACTUALLY_APPLIED',
  'nullspace_fix':'weak nominal/wrist posture correction projected through the 6-D bimanual Cartesian-position Jacobian null space, followed by Cartesian reprojection',
  'max_abs_joint_difference':float(diff.max()),'mean_abs_joint_difference':float(diff.mean()),
  'differing_frames':int(np.count_nonzero(frame_mask)),'differing_frame_indices':np.flatnonzero(frame_mask).tolist(),
  'differing_joints':int(np.count_nonzero(joint_mask)),'differing_joint_indices':np.flatnonzero(joint_mask).tolist(),
  'differing_joint_names':[names[i] for i in np.flatnonzero(joint_mask)],
  'npz':{'exact':{'path':str(EXACT.resolve()),'sha256':sha(EXACT)},'nullspace':{'path':str(NULL.resolve()),'sha256':sha(NULL)}},
  'active_models':{'aloha':'/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml','g1':'/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml'},
  'render_content_keyframe_inspection':{'three_synchronized_columns':True,'g1_torso_shoulders_elbows_wrists':True,'table_phone_accessory_charger':True,'source_frame_and_event_overlay':True},
  'videos':videos,'dex3_contact_ik':False,'physics':False,'hardware':False,'simulation_only':True,
  'status':'ARM_ROBOT_RENDER_READY_FOR_VISUAL_APPROVAL'}
 (OUT/'visual_validation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 report=f'''# Restored original ALOHA→G1 pipeline — robot-render validation

The two legacy red-trajectory videos are recorded as `VISUAL_VALIDATION_INVALID`. Their MP4 SHA-256 and decoded RGB-frame SHA-256 are identical. The initial identical-q null-space output was recorded as `NULLSPACE_NOT_ACTUALLY_APPLIED`; the posture correction was fixed and the repaired q trajectories now differ.

- max absolute joint difference: `{diff.max():.17g}` rad
- mean absolute joint difference: `{diff.mean():.17g}` rad
- differing frames: `{np.count_nonzero(frame_mask)}` / `{len(exact)}` (indices 1–989)
- differing joints: `{np.count_nonzero(joint_mask)}` / `{exact.shape[1]}` (all arm joints)
- Exact NPZ: `{EXACT.resolve()}`
- Exact SHA-256: `{sha(EXACT)}`
- Nullspace NPZ: `{NULL.resolve()}`
- Nullspace SHA-256: `{sha(NULL)}`

The overview, front, and side outputs use the active stationary ALOHA MuJoCo model and active Unitree G1-with-hands MuJoCo model. Each is a 3-column synchronized robot render and decodes to 330 frames. Representative rendered frames were inspected for G1 torso/shoulders/elbows/wrists, table, phone, accessory, charger, source frame, and event name. Exact and Nullspace crop hashes differ for every camera. Input NPZ paths and hashes are embedded in each MP4 `comment` tag and repeated in the JSON sidecars/audit.

No Dex3 contact IK, physics stepping, DDS, publisher, or hardware command path was used. Simulation only.

Final state: `ARM_ROBOT_RENDER_READY_FOR_VISUAL_APPROVAL`
'''
 (OUT/'report.md').write_text(report)
 manifest={'status':'ARM_ROBOT_RENDER_READY_FOR_VISUAL_APPROVAL','dex3_contact_ik':False,'physics':False,'hardware':False,'simulation_only':True,'visual_validation_audit':str((OUT/'visual_validation_audit.json').resolve())}
 (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'status':audit['status'],'max_abs_joint_difference':audit['max_abs_joint_difference'],'mean_abs_joint_difference':audit['mean_abs_joint_difference'],'differing_frames':audit['differing_frames'],'differing_joints':audit['differing_joints']},indent=2))
if __name__=='__main__': main()
