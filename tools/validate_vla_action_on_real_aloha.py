#!/usr/bin/env python3
"""Safety-gated ALOHA action inspector; no hardware backend is implemented here."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/jbnu/aloha_g1_dataset');DEFAULT=ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz'
JOINTS=[f'left_joint_{i}' for i in range(7)]+[f'right_joint_{i}' for i in range(7)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=DEFAULT);p.add_argument('--dry-run',action='store_true',default=True);p.add_argument('--inspect',action='store_true');p.add_argument('--speed-scale',type=float,default=.25);p.add_argument('--start-frame',type=int,default=0);p.add_argument('--end-frame',type=int);p.add_argument('--no-object',action='store_true');p.add_argument('--record-actual',action='store_true');p.add_argument('--record-camera',action='store_true');p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/aloha_real_validation');p.add_argument('--require-confirmation',action='store_true');p.add_argument('--execute-hardware',action='store_true',help='always refused until a verified stationary ALOHA hardware replay backend is identified');a=p.parse_args()
 with np.load(a.input,allow_pickle=False) as z:
  if 'optimized_action' not in z:raise ValueError('optimized_action missing')
  q=z['optimized_action'].astype(float);ts=z['timestamp'] if 'timestamp' in z else np.arange(len(q))/30
 if q.shape!=(990,14) or not np.isfinite(q).all():raise ValueError(f'invalid action {q.shape}')
 end=len(q) if a.end_frame is None else min(a.end_frame,len(q));seg=q[a.start_frame:end];step=np.abs(np.diff(seg,axis=0));fps=30.;result={'status':'DRY_RUN_INSPECT_ONLY','command_enabled':False,'hardware_backend':'NOT_AVAILABLE_VERIFIED','reason':'Repository contains verified simulation replay but no verified stationary ALOHA hardware action replay/state-recorder wrapper was found. Hardware commands are intentionally not fabricated.','input':str(a.input.resolve()),'shape':list(q.shape),'fps':fps,'joint_order':JOINTS,'gripper_indices':{'left':6,'right':13},'gripper_range':{'left':[float(q[:,6].min()),float(q[:,6].max())],'right':[float(q[:,13].min()),float(q[:,13].max())]},'finite':True,'frame_range':[a.start_frame,end-1],'speed_scale':a.speed_scale,'no_object':a.no_object,'max_joint_step':float(step.max(initial=0)),'estimated_max_velocity_at_speed_scale':float(step.max(initial=0)*fps*a.speed_scale),'initial_target_actual_difference':'NOT_AVAILABLE_WITHOUT_READ_ONLY_HARDWARE_STATE','joint_limit_validation':'CODE_MAPPING_VALIDATED_IN_MUJOCO; HARDWARE_LIMIT_SOURCE_NOT_VERIFIED','state_timeout':'NOT_AVAILABLE_WITHOUT_BACKEND','emergency_stop':'UNKNOWN_NO_VERIFIED_HARDWARE_BACKEND','camera_recording_requested':a.record_camera,'actual_recording_requested':a.record_actual}
 a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'dry_run_inspection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if a.execute_hardware:raise RuntimeError('HARDWARE EXECUTION REFUSED: verified ALOHA command/state/emergency-stop backend is not available in this repository')
 return 0
if __name__=='__main__':raise SystemExit(main())
