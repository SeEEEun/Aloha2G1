from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path('/home/jbnu/aloha_g1_dataset');TOOLS=ROOT/'tools';sys.path.insert(0,str(TOOLS))
SPEC=importlib.util.spec_from_file_location('aloha_runner',TOOLS/'replay_smolvla_on_real_aloha.py');runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)
from aloha_source_validation_common import EXPECTED_SHA256, CommandActualLog, minimum_jerk_transition, resample_minimum_jerk, sha
from analyze_aloha_successful_operation_envelope import build as build_envelope

def args(*extra):return runner.parser().parse_args(list(extra))
def report():return {'sha256_matches_frozen_episode49':True}
def reviewed():return {'status':'REVIEWED','hardware_replay_allowed':True}

def test_frozen_trajectory_and_public_ui_is_outside_project():
 assert sha(runner.TRAJECTORY)==EXPECTED_SHA256
 assert not str(Path('/home/jbnu/.lerobot_trossen_ai_data_collection_ui')).startswith(str(ROOT)+'/')

def test_default_and_inspect_never_construct_hardware(monkeypatch):
 called=[];monkeypatch.setattr(runner,'create_stationary_robot',lambda:called.append(1))
 assert args().mode=='inspect';assert runner.main([])==0;assert runner.main(['--mode','inspect'])==0;assert called==[]

def test_dry_run_has_no_hardware_and_logging_schema(tmp_path,monkeypatch):
 called=[];monkeypatch.setattr(runner,'create_stationary_robot',lambda:called.append(1))
 out=tmp_path/'trial';assert runner.main(['--mode','dry-run','--end-frame','2','--output',str(out)])==0;assert called==[]
 with np.load(out/'command_actual_log.npz',allow_pickle=False) as z:assert set(CommandActualLog.FIELDS)<=set(z.files);assert z['command_action_14d'].shape[1]==14
 meta=json.loads((out/'trial_metadata.json').read_text());assert meta['hardware_connection']=='NOT PERFORMED';assert meta['mode_change']=='NOT PERFORMED'

def test_hardware_missing_flags_blocks_before_process_or_connect(monkeypatch):
 called=[];monkeypatch.setattr(runner,'process_conflicts',lambda *_:called.append('process'));monkeypatch.setattr(runner,'create_stationary_robot',lambda:called.append('robot'))
 with pytest.raises(runner.Blocked,match='missing hardware gates'):runner.hardware_preflight(args('--mode','hardware'),report(),reviewed())
 assert called==[]

def fully_gated():
 flags=['--mode','hardware','--enable-hardware','--confirmed-ui-closed','--confirmed-empty-workspace','--confirmed-power-switch-ready','--confirmed-operator-present','--confirmed-thresholds-reviewed','--confirmed-vendor-disconnect-motion','--normal-exit-policy','vendor-disconnect','--hardware-confirmation',runner.HARDWARE_PHRASE,'--start-transition-confirmation',runner.START_PHRASE]
 return args(*flags)
def test_ui_process_and_duplicate_lock_block(tmp_path):
 a=fully_gated();a.lock_file=tmp_path/'lock'
 with pytest.raises(runner.Blocked,match='conflicting'):runner.hardware_preflight(a,report(),reviewed(),'123 control_robot.py')
 with runner.ControlLock(a.lock_file):
  with pytest.raises(runner.Blocked,match='lock exists'):runner.ControlLock(a.lock_file).__enter__()

def test_sha_shape_and_full_prerequisites_block(tmp_path):
 bad=tmp_path/'bad.npz';np.savez(bad,optimized_action=np.zeros((2,14)))
 with pytest.raises(ValueError,match='shape'):runner.load_action(bad)
 a=fully_gated();a.end_frame=989
 with pytest.raises(runner.Blocked,match='short test'):runner.hardware_preflight(a,report(),reviewed(),'')

def test_interpolation_reduces_step_and_holds_grippers():
 q=np.zeros((2,14));q[1]=1;q[:,6]=[.02,.08];q[:,13]=[.03,.09]
 out,_,_,n=resample_minimum_jerk(q,30,.1,30,'hold-current',[.04,.05])
 assert n==10;assert np.max(np.abs(np.diff(out[:,:6],axis=0)))<1
 assert np.all(out[:,6]==.04);assert np.all(out[:,13]==.05)
 tr=minimum_jerk_transition(np.zeros(14),np.ones(14),1,30,True);assert np.all(tr[:,[6,13]]==0)

def test_pause_is_logged_without_synthetic_command(tmp_path):
 log=CommandActualLog();log.event('PAUSED','SIGINT');meta={'trajectory_sha256':'x','stop_reason':'PAUSED'};log.save(tmp_path,meta,np.zeros((1,14)))
 assert 'PAUSED' in (tmp_path/'safety_events.jsonl').read_text()

def test_g1_authoritative_files_are_not_imported_by_runner():
 text=(TOOLS/'replay_smolvla_on_real_aloha.py').read_text()
 assert 'isaaclab_magsafe_fixed_scene' not in text
 assert 'import g1_' not in text.lower() and 'from g1_' not in text.lower()

def test_50_episode_envelope_and_candidate_is_not_reviewed():
 envelope,*_=build_envelope();assert envelope['episodes']==50;assert envelope['frames']==50302
 assert envelope['envelope_type']=='EMPIRICAL_SUCCESSFUL_OPERATION_ENVELOPE'
 candidate=json.loads((ROOT/'configs/aloha_source_validation_safety.candidate.json').read_text())
 assert candidate['status']=='CANDIDATE_NOT_APPROVED' and candidate['hardware_replay_allowed'] is False
 assert not (ROOT/'configs/aloha_source_validation_safety.reviewed.json').exists()
 with pytest.raises(runner.Blocked):runner.reviewed_safety(ROOT/'configs/aloha_source_validation_safety.candidate.json')

class FakeRobot:
 def __init__(self):self.connected=False;self.disconnected=False;self.send_count=0
 def connect(self):self.connected=True
 def disconnect(self):self.disconnected=True
 def capture_observation(self):return {'observation.state':np.arange(14,dtype=float)/100}
 def send_action(self,*_):self.send_count+=1;raise AssertionError('preflight must not send')
def test_connect_only_preflight_records_but_never_sends(tmp_path,monkeypatch):
 a=args('--mode','hardware-preflight','--enable-hardware','--confirmed-ui-closed','--confirmed-empty-workspace','--confirmed-power-switch-ready','--confirmed-operator-present','--confirmed-vendor-disconnect-motion','--normal-exit-policy','vendor-disconnect','--preflight-confirmation',runner.PREFLIGHT_PHRASE,'--preflight-duration','0.001','--output',str(tmp_path/'p'),'--lock-file',str(tmp_path/'lock'))
 monkeypatch.setattr(runner,'process_conflicts',lambda *_:[]);robot=FakeRobot();action=runner.load_action()[0];answers=iter((runner.HOME_OBSERVED_PHRASE,runner.DISCONNECT_PHRASE));out=runner.run_connect_only_preflight(a,action,runner.trajectory_report(),lambda:robot,lambda _:None,lambda _:next(answers))
 assert robot.connected and robot.disconnected and robot.send_count==0
 assert (out/'actual_state.npz').exists() and (out/'current_to_action0.json').exists() and (out/'planned_start_transition_metrics.json').exists()
 assert json.loads((out/'planned_start_transition_metrics.json').read_text())['executed'] is False

def monitor_cfg():return {'max_tracking_error':.1,'tracking_error_duration':.2,'state_timeout':.1,'control_loop_overrun_threshold':.01,'consecutive_overrun_limit':2}
def test_start_and_tracking_thresholds_are_independent_and_persistent():
 cfg=monitor_cfg();cfg['max_start_position_error']=.01;m=runner.RuntimeSafetyMonitor(cfg)
 assert runner.threshold(cfg,'max_start_position_error') != runner.threshold(cfg,'max_tracking_error')
 m.check(np.ones(14)*.2,0,.1)
 with pytest.raises(runner.Blocked,match='PERSISTENT_TRACKING_ERROR'):m.check(np.ones(14)*.2,0,.1)
def test_state_timeout_and_consecutive_overrun_stop():
 with pytest.raises(runner.Blocked,match='STATE_TIMEOUT'):runner.RuntimeSafetyMonitor(monitor_cfg()).check(np.zeros(14),.2,.03)
 m=runner.RuntimeSafetyMonitor(monitor_cfg());m.check(np.zeros(14),0,.05)
 with pytest.raises(runner.Blocked,match='CONSECUTIVE'):m.check(np.zeros(14),0,.05)
