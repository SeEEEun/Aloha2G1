from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path('/home/jbnu/aloha_g1_dataset');sys.path.insert(0,str(ROOT/'tools'))
import final_experiment_gates as gates
import measure_and_register_g1_magsafe_layout as layout
from validate_final_experiment_inputs import task_valid,real_primitives,object_physics_pass,primary_available

def test_no_result_cannot_create_pass_gate(tmp_path,monkeypatch):
 monkeypatch.setattr(gates,'OUT',tmp_path)
 with pytest.raises(ValueError,match='evidence'):gates.record('A0_ALOHA_PREFLIGHT_PASS','PASS')
def test_source_invalid_blocks_g1(tmp_path,monkeypatch):
 monkeypatch.setattr(gates,'OUT',tmp_path);gates.record('A4_ALOHA_SOURCE_RESULT','FAIL',metrics={'classification':'SOURCE_ACTION_INVALID'})
 assert not gates.source_allows_g1()
def test_g1_offset_and_missing_measurement_block():
 d=layout.template();assert d['root_forward_offset_m']==.20;assert layout.validate(d)
 d['root_forward_offset_m']=.1;assert 'offset' in ' '.join(layout.validate(d))
def test_task_valid_gate_is_strict():
 good={'status':'G1_TASK_VALID_TRAJECTORY_CANDIDATE','ik_success_rate':.99,'joint_limit_violation_count':0,'branch_discontinuity_count':0,'arm_level_collision_frames':0,'task_region_checks_passed':True};assert task_valid(good)
 bad=dict(good,ik_success_rate=.989);assert not task_valid(bad)
def test_sim_primitive_cannot_be_real():assert not real_primitives(ROOT/'configs/dex3_magsafe_grasp_primitives.sim.json')
def test_no_object_change_cannot_be_success():
 report={'task_valid_candidate_used':True,'real_dex3_primitives_used':True,'object_state_changed':False,'phone_grasp_assessable':True,'accessory_removal_assessable':True,'charger_placement_assessable':True,'severe_instability':False,'front_video':'a','side_video':'b','top_video':'c'};assert not object_physics_pass(report)
def test_actual_missing_or_trial_count_blocks_primary(tmp_path):
 assert not primary_available([],[],5)
 assert not primary_available([tmp_path/'missing.npz']*5,[tmp_path/'missing2.npz']*5,5)
def test_three_comparison_classes_are_documented():
 text=(ROOT/'docs/GENERATED_EXPERT_FINAL_ANALYSIS_RUNBOOK.md').read_text();assert all(x in text for x in ('Generated–Expert','Expert–Expert','Generated–Generated'))
def test_no_reviewed_config_and_no_hardware_commands_added():
 assert not (ROOT/'configs/aloha_source_validation_safety.reviewed.json').exists()
 assert 'ChannelPublisher' not in (ROOT/'tools/measure_and_register_g1_magsafe_layout.py').read_text()
def test_public_ui_and_authoritative_scene_not_write_targets():
 for p in (ROOT/'tools/final_experiment_gates.py',ROOT/'tools/measure_and_register_g1_magsafe_layout.py'):
  text=p.read_text();assert '/home/jbnu/.lerobot_trossen_ai_data_collection_ui' not in text
  assert 'scene_layout.json' not in text
