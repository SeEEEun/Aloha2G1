#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));from g1_behavior_schema import sha256
ROOT=Path('/home/jbnu/aloha_g1_dataset');OUT=ROOT/'configs/g1_behavior_comparison_manifest.json'
FILES={'vla_action':ROOT/'evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz','generated_arm':ROOT/'converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz','generated_full':ROOT/'outputs/g1_magsafe_arm_dex3_full_trajectory.npz','primitive_config':ROOT/'configs/dex3_magsafe_grasp_primitives.sim.json','g1_xml':Path('/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml')}
def payload():
 with np.load(FILES['generated_full'],allow_pickle=False) as z:return {'task_name':'magsafe_phone_accessory','episode_id':49,'vla_checkpoint_identifier':'smolvla_20k','vla_action_source':str(FILES['vla_action']),'vla_action_key':'optimized_action','generated_arm_trajectory_source':str(FILES['generated_arm']),'generated_arm_key':'g1_arm_joint_trajectory','full_generated_trajectory_source':str(FILES['generated_full']),'primitive_config':str(FILES['primitive_config']),'g1_xml':str(FILES['g1_xml']),'fps':float(z['fps']),'expected_frame_count':len(z['timestamps']),'action_dimensions':14,'arm_joint_names':z['arm_joint_names'].tolist(),'left_dex3_joint_names':z['left_dex3_joint_names'].tolist(),'right_dex3_joint_names':z['right_dex3_joint_names'].tolist(),'coordinate_frame':'g1_base_fixed_mujoco_world_frame','fk_body_names':['left_wrist_yaw_link','right_wrist_yaw_link'],'comparison_protocol_version':'g1_behavior_comparison_v1','random_seed':20260803,'result_freeze_status':'FROZEN','sha256':{k:sha256(v) for k,v in FILES.items()}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--inspect',action='store_true');a=p.parse_args()
 if a.inspect:
  d=json.loads(a.output.read_text());warnings=[f'{k}: HASH_CHANGED' for k,v in FILES.items() if sha256(v)!=d['sha256'][k]];print(json.dumps({'manifest':d,'warnings':warnings,'valid':not warnings},indent=2));return 2 if warnings else 0
 d=payload();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
