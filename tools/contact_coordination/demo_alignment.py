"""Validate pre-action target observations against independent physics telemetry."""
from pathlib import Path
import numpy as np
from .io import read, record, atomic_json


def scored_task_success(score):
    """Interpret the scorer's string enum explicitly; never use its truthiness."""
    label = score['source_conditioned_full_task']
    if label not in ('DEMONSTRATED', 'NOT_DEMONSTRATED'):
        raise ValueError('Unknown source-conditioned success label')
    success = score['stages']['FULL_TASK']
    if not isinstance(success, bool) or success != (label == 'DEMONSTRATED'):
        raise ValueError('Inconsistent boolean stage and source-conditioned label')
    if success and not score['physical_validity']:
        raise ValueError('Invalid physics cannot demonstrate success')
    return success


def check_arrays(observations, telemetry):
    frames = observations['control_frame']
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError('Observation frames must be contiguous from natural startup')
    state = observations['measured_state']
    action = observations['executed_command']
    if state.shape != (len(frames), 28) or action.shape != state.shape:
        raise ValueError('Expected named28 measured state and absolute executed action')
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError('Nonfinite supervision')
    if not np.allclose(observations['timestamp_s'], frames / 30., atol=1e-9, rtol=0):
        raise ValueError('Observation timestamps do not follow the declared30Hz clock')
    physics_frames = telemetry['control_frame']
    max_state_error = max_action_error = 0.
    for frame in frames:
        rows = np.flatnonzero(physics_frames == frame)
        if len(rows) != 8:
            raise ValueError(f'Incomplete physical transition at frame{frame}')
        error = float(np.max(np.abs(telemetry['EXECUTED_COMMAND'][rows] - action[frame])))
        max_action_error = max(max_action_error, error)
        if error > 1e-9:
            raise ValueError(f'Saved action differs from actual servo command at frame{frame}')
        if frame:
            previous = np.flatnonzero(physics_frames == frame - 1)[-1]
            error = float(np.max(np.abs(telemetry['MEASURED_Q'][previous] - state[frame])))
            max_state_error = max(max_state_error, error)
            if error > 1e-7:
                raise ValueError(f'Pre-action state does not equal the preceding measured transition at frame{frame}')
    return dict(frames=len(frames), maximum_state_alignment_error_rad=max_state_error,
                maximum_action_alignment_error_rad=max_action_error,
                first_state_has_no_preceding_logged_transition=True)


def validate(folder, command_path):
    import cv2
    folder, command_path = Path(folder), Path(command_path)
    observations = dict(np.load(folder / 'OBSERVATION_ACTION.npz'))
    telemetry = dict(np.load(folder / 'event_log.npz'))
    commands = dict(np.load(command_path))
    result = check_arrays(observations, telemetry)
    if not np.array_equal(observations['joint_names'], commands['joint_names']):
        raise ValueError('Observation/action joint order differs from command contract')
    image_records = []
    for frame in observations['control_frame']:
        row = read(folder / 'observation_action' / f'{frame:06d}.json')
        current = record(row['image']['path'])
        if current != row['image'] or not row['observation_precedes_action']:
            raise ValueError(f'Observation provenance mismatch at frame{frame}')
        rgb = cv2.imread(current['path'])
        if rgb is None or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError(f'Invalid target RGB at frame{frame}')
        image_records.append(current)
    score = read(folder / 'HYBRID_SCORE.json')
    result.update(status='DYNAMIC_OBSERVATION_ACTION_ALIGNMENT_VERIFIED',schema_version=2,
                  complete_command_sequence=len(observations['control_frame']) == len(commands['stage']),
                  physical_validity=score['physical_validity'],
                  task_success=scored_task_success(score),
                  source_conditioned_status=score['source_conditioned_full_task'],
                  task_success_is_not_alignment_evidence=True,
                  observations=record(folder / 'OBSERVATION_ACTION.npz'),
                  physics_trace=record(folder / 'event_log.npz'),
                  commands=record(command_path), images=image_records,
                  implementation=record(__file__))
    target=folder / 'DEMONSTRATION_ALIGNMENT.json'
    if target.exists():
        result['previous_alignment_record']=record(target)
        target=folder / 'DEMONSTRATION_ALIGNMENT_V2.json'
    atomic_json(target, result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-folder', type=Path, required=True)
    parser.add_argument('--commands', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.trace_folder, args.commands)
    print(result['status'], result['frames'], result['complete_command_sequence'])
