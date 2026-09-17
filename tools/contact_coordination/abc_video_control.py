"""Short isolated TRAIN controls for full-attempt recording and reset parity."""
from pathlib import Path
import numpy as np
from .io import read, record, atomic_json
from .full_attempt import prepare, launch


def run(out,resume=False):
    old=Path(read(out/'ABC_STUDY.json')['historical_run'])
    rows=read(old/'TRAIN40_conversion/LEDGER.json')['rows']
    golden=next(r for r in rows if r['source_id']=='GoPark_20260820_152058' and r.get('full_task_plan'))
    other=min((r for r in rows if r.get('full_task_plan') and r['source_id']!=golden['source_id']),key=lambda r:r['source_id'])
    late_failure=min((r for r in rows if r['terminal']=='VALID_PLAN_TASK_FAILURE' and r.get('physical_validity')),key=lambda r:r['source_id'])
    # Selection uses saved plan availability solely to test the interface;
    # no new calibration outcome participates in source/parameter selection.
    cases=[('golden_flag_30',golden,30,'OFFICIAL_NOMINAL',180),
           ('golden_flag_90',golden,90,'OFFICIAL_NOMINAL',180),
           ('golden_diagnostic',golden,30,'DIAGNOSTIC_FULL_CONTINUATION',180),
           ('other_reset',other,30,'OFFICIAL_NOMINAL',180),
           ('late_stop_nominal',late_failure,None,'OFFICIAL_NOMINAL',None),
           ('late_stop_diagnostic',late_failure,None,'DIAGNOSTIC_FULL_CONTINUATION',None)]
    reports=[]
    for name,row,flag,channel,prefix in cases:
        folder=out/'video_contract_controls'/name
        if not folder.exists():
            folder=prepare(out,Path(row['plan']),name,'C_COUPLED',channel,
                prefix_frames=prefix,injected_failure_frame=flag,contract_test=True)
        process=launch(folder,resume=resume)
        if process['returncode']!=0:
            result=dict(status='FAIL',first_error=str(folder/'engine.log'),completed=reports)
            atomic_json(out/'VIDEO_CONTRACT_CONTROL.json',result);return result
        a=np.load(folder/'event_log.npz',allow_pickle=False)
        cfg=read(folder/'input/FULL_ATTEMPT_CONFIG.json');reset=read(folder/'PRE_COMMAND_SCENE.json')
        rec=read(folder/'FULL_ATTEMPT_RECORDING.json')
        n=int(a['control_frame'][-1])+1
        expected=cfg['nominal_frames']+round(cfg['observation_s']*30)
        if n!=expected or len(a['control_frame'])!=expected*8:
            # A successful process exit does not certify the recording horizon.
            # Preserve the failed control explicitly instead of leaving the
            # previous short-control PASS as the latest aggregate evidence.
            failed=dict(name=name,source_id=cfg['source_id'],channel=channel,
                frames=n,expected_frames=expected,physics_rows=len(a['control_frame']),
                expected_physics_rows=expected*8,last_timestamp_s=float(a['timestamp_s'][-1]),
                recording_end_reason=rec['recording_end_reason'],validity_abort=rec.get('validity_abort'),
                trace=record(folder/'event_log.npz'),recording=record(folder/'FULL_ATTEMPT_RECORDING.json'),
                reset=record(folder/'PRE_COMMAND_SCENE.json'),process=record(folder/'PROCESS.json'))
            result=dict(status='FAIL',reason='INCOMPLETE_DECLARED_RECORDING_HORIZON',
                controls=reports,failed_control=failed,continue_downstream=False,
                no_calibration_score_or_training_data=True,
                acceptance_criteria_changed=False)
            atomic_json(out/'VIDEO_CONTRACT_CONTROL.json',result);return result
        if flag is not None:assert rec['first_failure_latched']['control_frame']==flag
        assert rec['recorded_control_frames']==n and not rec['eligible_for_ACT_data']
        assert reset['controller_initial_frame']==-1 and reset['controller_trace_length']==0 and reset['phase_stop'] is None
        np.testing.assert_allclose(reset['actual_named_q_rad'],reset['natural_initial_q_rad'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(reset['actual_object_pose_xyzw'],reset['requested_object_pose_xyzw'],atol=1e-7,rtol=0)
        assert np.all(a['RECORDING_CHANNEL']==channel)
        speed_evidence=None
        if name=='late_stop_diagnostic':
            telemetry=read(folder/'OBJECT_LINEAR_SPEED_TELEMETRY.json')
            speed=np.linalg.norm(a['object_linear_velocity_m_s'],axis=1)
            np.testing.assert_array_equal(a['object_linear_speed_m_s'],speed)
            np.testing.assert_array_equal(a['object_linear_speed_warning'],speed>telemetry['warning_threshold_m_s'])
            assert telemetry['warning_threshold_m_s']==1.0 and not telemetry['terminal_condition']
            assert telemetry['first_exceedance'] is not None
            crossing=telemetry['first_exceedance'];peak=telemetry['maximum']
            assert crossing['speed_m_s']==speed[crossing['physics_step']]
            assert crossing['timestamp_s']==a['timestamp_s'][crossing['physics_step']]
            assert peak['speed_m_s']==float(speed.max())
            assert a['timestamp_s'][-1]>crossing['timestamp_s']
            assert rec['validity_abort'] is None and rec['recording_end_reason']=='COMPLETE_DECLARED_HORIZON'
            speed_evidence=record(folder/'OBJECT_LINEAR_SPEED_TELEMETRY.json')
        reports.append(dict(name=name,source_id=cfg['source_id'],channel=channel,frames=n,
            failure_frame=rec['first_failure_latched']['control_frame'] if rec['first_failure_latched'] else None,
            post_failure_s=(n-1-rec['first_failure_latched']['control_frame'])/30. if rec['first_failure_latched'] else 0.,last_timestamp_s=float(a['timestamp_s'][-1]),
            trace=record(folder/'event_log.npz'),recording=record(folder/'FULL_ATTEMPT_RECORDING.json'),
            reset=record(folder/'PRE_COMMAND_SCENE.json'),process=record(folder/'PROCESS.json'),
            linear_speed_telemetry=speed_evidence))
    first=np.load(out/'video_contract_controls/golden_flag_30/event_log.npz')
    second=np.load(out/'video_contract_controls/golden_flag_90/event_log.npz')
    np.testing.assert_array_equal(first['commanded_q_rad'],second['commanded_q_rad'])
    np.testing.assert_array_equal(first['timestamp_s'],second['timestamp_s'])
    nominal=read(out/'video_contract_controls/late_stop_nominal/FULL_ATTEMPT_RECORDING.json')
    diagnostic=read(out/'video_contract_controls/late_stop_diagnostic/FULL_ATTEMPT_RECORDING.json')
    assert nominal['official_admission_stop'] is not None
    assert diagnostic['official_admission_stop'] is None and diagnostic['interventions']
    result=dict(status='PASS',controls=reports,nonfatal_failure_timestamp_does_not_change_horizon=True,
        explicit_nominal_diagnostic_channels_verified=True,two_source_reset_states_verified=True,
        linear_speed_exceedance_is_nonterminal_and_measured=True,
        no_calibration_score_or_training_data=True,renderer_tests='PENDING_MEASURED_REPLAYS',
        note='Independent recorder tests use stored TRAIN command prefixes, not new converter outcomes.')
    atomic_json(out/'VIDEO_CONTRACT_CONTROL.json',result);return result
