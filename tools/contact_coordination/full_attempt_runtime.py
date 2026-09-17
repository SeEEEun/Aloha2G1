"""Keep official controller stops separate from the measurement horizon.

This adapter cannot write object/robot state. Task failures latch for scoring
while the pre-existing command sequence continues through its true horizon.
"""
from pathlib import Path
import numpy as np
from .io import read, atomic_json, record
from .phase_clock_runtime import build_runtime as phase_runtime
from .phase_physics import (lift_allowed, giver_clearance_allowed,
    right_transport_allowed, receiver_candidate_ready, receiver_departure_allowed)


def state_validity_reason(pose, velocity, q, qd, previous_position, gates):
    if not all(np.isfinite(v).all() for v in (pose,velocity,q,qd)):
        return 'NONFINITE_MEASURED_STATE'
    # Finite speed/step excursions fail the unchanged physical-validity audit,
    # but are not proof of numerical corruption and must not cut a recording.
    # Preserve them as measured telemetry while executing the existing horizon.
    return None


def record_linear_speed_telemetry(arrays, gates, folder):
    """Retain every measured velocity/timestamp; warnings never control execution."""
    speed = np.linalg.norm(arrays['object_linear_velocity_m_s'], axis=1)
    threshold = float(gates['maximum_object_linear_speed_m_s'])
    warning = speed > threshold
    arrays['object_linear_speed_m_s'] = speed
    arrays['object_linear_speed_warning'] = warning
    def event(index):
        return dict(speed_m_s=float(speed[index]),
            timestamp_s=float(arrays['timestamp_s'][index]),
            physics_step=int(arrays['physics_step'][index]),
            control_frame=int(arrays['control_frame'][index]))
    crossings = np.flatnonzero(warning)
    result = dict(schema='nonterminal_object_linear_speed_telemetry_v1',
        warning_threshold_m_s=threshold, terminal_condition=False,
        warning_samples=int(np.count_nonzero(warning)),
        maximum=event(int(np.argmax(speed))) if len(speed) else None,
        first_exceedance=event(int(crossings[0])) if len(crossings) else None,
        measured_velocity_and_timestamp_unchanged=True,
        basis='Removal of unjustified legacy terminal heuristic; not threshold tuning')
    atomic_json(Path(folder)/'OBJECT_LINEAR_SPEED_TELEMETRY.json', result)
    record_finite_state_validity_telemetry(arrays,gates,folder)
    return result


def record_finite_state_validity_telemetry(arrays,gates,folder):
    angular=np.linalg.norm(arrays['object_angular_velocity_rad_s'],axis=1)
    position=arrays['object_position_world_m']
    step=np.r_[0.,np.linalg.norm(np.diff(position,axis=0),axis=1)]
    reports={}
    for name,values,threshold in (
        ('angular_speed',angular,float(gates['maximum_object_angular_speed_rad_s'])),
        ('com_step',step,float(gates['maximum_object_com_step_m']))):
        failed=values>threshold;indices=np.flatnonzero(failed)
        arrays['object_'+name+'_validity_failure']=failed
        def event(index):
            return dict(value=float(values[index]),timestamp_s=float(arrays['timestamp_s'][index]),
                control_frame=int(arrays['control_frame'][index]),physics_step=int(arrays['physics_step'][index]))
        reports[name]=dict(threshold=threshold,failed_samples=int(failed.sum()),
            maximum=event(int(np.argmax(values))) if len(values) else None,
            first_exceedance=event(int(indices[0])) if len(indices) else None)
    atomic_json(Path(folder)/'FINITE_STATE_VALIDITY_TELEMETRY.json',dict(
        terminal_condition=False,scoring_thresholds_unchanged=True,measurements_unchanged=True,
        first_position_sample_is_trace_origin=True,checks=reports))


def admission_failure(controller, snapshot, frame, label, previous):
    if frame == 0 or label == previous:
        return None
    if label == 'LIFT_5CM' and not lift_allowed(controller):
        return 'NO_ACQUISITION_BEFORE_LIFT'
    if label == 'GIVER_CLEARANCE':
        if getattr(controller, 'coordinated_giver_release', False):
            if not receiver_candidate_ready(controller, snapshot):
                return 'NO_CURRENT_RECEIVER_CANDIDATE_FOR_COORDINATED_RELEASE'
        elif not giver_clearance_allowed(controller, frame):
            return 'NO_RECEIVER_SUPPORT_OR_INCOMPLETE_GIVER_OPENING'
    if label == 'RECEIVER_DEPARTURE' and not receiver_departure_allowed(controller, snapshot, frame):
        return 'NO_CURRENT_RECEIVER_SUPPORT_OR_INCOMPLETE_GIVER_OPENING'
    if label == 'RIGHT_TRANSPORT' and not right_transport_allowed(controller):
        return 'NO_RIGHT_OWNERSHIP_AFTER_GIVER_WITHDRAWAL_AND_WAIT'
    return None


class FullAttemptRuntime:
    def __init__(self, delegate, config, stages):
        self.delegate = delegate;self.config = config;self.stages = np.asarray(stages)
        self.folder = Path(config['output_dir']);self.frame = -1
        self.stop = None;self.first_failure = None;self.interventions = []
        self.last_command = np.asarray(delegate.initial_q_rad).copy()
        self.nominal_frames = int(config['nominal_frames'])
        self.channel = config['evidence_channel']
        if self.channel not in ('OFFICIAL_NOMINAL','DIAGNOSTIC_FULL_CONTINUATION','RECONSTRUCTED_DIAGNOSTIC'):
            raise ValueError('Unknown evidence channel')

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    @property
    def event_field_names(self):
        return (*self.delegate.event_field_names, 'RECORDING_CHANNEL', 'NOMINAL_PHASE',
                'FULL_ATTEMPT_RUNTIME_PHASE', 'TASK_FAILURE_LATCHED', 'CONTROL_ADMISSION_STOPPED')

    def latch_task_failure(self, reason, frame):
        # Observer/scorer flags never control playback or the servo schedule.
        if self.first_failure is None:
            self.first_failure = dict(reason=reason, control_frame=int(frame))

    def step(self, frame, snapshot):
        self.frame = frame;label = str(self.stages[frame])
        if frame == self.config.get('injected_nonfatal_failure_frame', -1):
            if not self.config.get('contract_test_only',False):
                raise ValueError('Synthetic failure injection is restricted to software contract controls')
            self.latch_task_failure('INJECTED_NONFATAL_RECORDING_TEST', frame)
        if frame >= self.nominal_frames:
            self.current_phase = 'POST_HORIZON_OBSERVATION_HOLD'
            return self.last_command.copy()
        reason = admission_failure(self.controller, snapshot, frame, label,
                                   str(self.stages[frame-1]) if frame else '') if self.stop is None else None
        if reason:
            failure = dict(reason=reason, control_frame=frame, phase=label)
            self.latch_task_failure(reason, frame)
            if self.channel == 'OFFICIAL_NOMINAL' and not self.config.get('continue_existing_command_after_task_failure',True):
                self.stop = failure;atomic_json(self.folder/'PHASE_STOP.json', dict(failure,
                    arm_rescue=False, recording_continues=True, action='HOLD_LAST_SERVO_COMMAND'))
            elif self.channel != 'OFFICIAL_NOMINAL':
                self.interventions.append(dict(failure, action='DIAGNOSTIC_ADMISSION_BYPASS'))
            else:
                atomic_json(self.folder/'TASK_FAILURE.json',dict(failure,
                    action='CONTINUE_EXISTING_COMMAND_SEQUENCE',arm_rescue=False,
                    task_failure_changes_commands=False))
        if self.stop:
            self.current_phase = 'OFFICIAL_HOLD_AFTER_PHASE_STOP'
            return self.last_command.copy()
        self.current_phase = label
        self.last_command = np.asarray(self.delegate.step(frame, snapshot)).copy()
        if not np.isfinite(self.last_command).all():
            raise ValueError('NONFINITE_EXECUTED_COMMAND')
        return self.last_command.copy()

    def event_values(self, measured):
        values = self.delegate.event_values(measured)
        # The frozen delegate decision is intentionally not advanced on hold.
        # Logged actual command is the servo target that is actually issued.
        values['EXECUTED_COMMAND'] = self.last_command.copy()
        values.update(RECORDING_CHANNEL=self.channel, NOMINAL_PHASE=str(self.stages[self.frame]),
            FULL_ATTEMPT_RUNTIME_PHASE=self.current_phase,
            TASK_FAILURE_LATCHED=self.first_failure is not None,
            CONTROL_ADMISSION_STOPPED=self.stop is not None)
        return values

    def write_summary(self, path):
        self.delegate.write_summary(path)
        abort=read(self.folder/'NUMERICAL_ABORT.json') if (self.folder/'NUMERICAL_ABORT.json').exists() else None
        atomic_json(self.folder/'FULL_ATTEMPT_RECORDING.json', dict(
            evidence_channel=self.channel, eligible_for_official_scoring=self.channel=='OFFICIAL_NOMINAL' and not self.config.get('contract_test_only',False),
            eligible_for_ACT_data=self.channel=='OFFICIAL_NOMINAL' and not self.config.get('contract_test_only',False) and not self.config.get('architecture_repair_verification_only',False),
            architecture_repair_verification_only=self.config.get('architecture_repair_verification_only',False),
            source_id=self.config['source_id'], method_key=self.config['method_key'],
            nominal_frames=self.nominal_frames, requested_recording_frames=len(self.stages),
            recorded_control_frames=self.frame+1, post_horizon_observation_s=self.config['observation_s'],
            first_failure_latched=self.first_failure, official_admission_stop=self.stop,
            continue_existing_command_after_task_failure=self.config.get('continue_existing_command_after_task_failure',True),
            interventions=self.interventions, recording_end_reason=abort['reason'] if abort else 'COMPLETE_DECLARED_HORIZON',
            validity_abort=abort,
            synthetic_contract_test=self.config.get('contract_test_only',False),
            object_pose_writes_after_initialization=0, controller_state_advanced_during_official_hold=False,
            implementation=record(__file__)))


def build_runtime(command_path, commands, names):
    config=read(Path(command_path).parent/'FULL_ATTEMPT_CONFIG.json')
    with np.load(command_path,allow_pickle=False) as archive:
        stages=archive['stage'].copy()
    return FullAttemptRuntime(phase_runtime(command_path,commands,names),config,stages)
