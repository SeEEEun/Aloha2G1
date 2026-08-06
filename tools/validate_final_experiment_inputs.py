#!/usr/bin/env python3
"""Pure fail-closed validators for B2/C0/C1/F0 evidence."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from g1_behavior_schema import load_behavior
def task_valid(report):
 return bool(report.get('ik_success_rate',0)>=.99 and report.get('joint_limit_violation_count',1)==0 and report.get('branch_discontinuity_count',1)==0 and report.get('arm_level_collision_frames',1)==0 and report.get('task_region_checks_passed') is True and report.get('status')=='G1_TASK_VALID_TRAJECTORY_CANDIDATE')
def real_primitives(path):
 d=json.loads(Path(path).read_text());return d.get('source')=='real_robot_recording' and d.get('authoritative_for_real_robot') is True and all(d.get('primitives',{}).get(k,{}).get('qpos') for k in ('LEFT_PHONE_OPEN','LEFT_PHONE_PREGRASP','LEFT_PHONE_GRASP','RIGHT_ACCESSORY_OPEN','RIGHT_ACCESSORY_PREGRASP','RIGHT_ACCESSORY_GRASP'))
def object_physics_pass(report):
 return bool(report.get('task_valid_candidate_used') and report.get('real_dex3_primitives_used') and report.get('object_state_changed') and report.get('phone_grasp_assessable') and report.get('accessory_removal_assessable') and report.get('charger_placement_assessable') and not report.get('severe_instability') and report.get('front_video') and report.get('side_video') and report.get('top_video'))
def valid_actual(path,role):
 try:a,m=load_behavior(path)
 except Exception:return False
 return bool(m.get('behavior_role')==role and m.get('execution_status')=='executed' and not m.get('is_synthetic') and 'actual_full_qpos' in a and np.isfinite(a['timestamps']).all() and m.get('success_label')!='unlabeled')
def primary_available(generated,experts,minimum=5):return sum(valid_actual(x,'generated_executed') for x in generated)>=minimum and sum(valid_actual(x,'expert_actual') for x in experts)>=minimum
