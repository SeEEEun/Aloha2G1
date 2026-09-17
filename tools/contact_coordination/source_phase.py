"""Thin, recording-ID-bound adapter over existing FK, events and image registration.

T_XY maps Y to X. Source forces and dynamic object tracking are unavailable.
Rigid carry extrapolations are explicitly inferred, never observed ownership.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT, read, record, atomic_json, atomic_npz

PRIOR = ROOT / 'outputs/final_single_variable_ab'
CALIBRATION = ROOT / 'outputs/doll_handoff_retargeting/scene_recalibration.json'
COMMON = ROOT / 'outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4/config/common_config.json'
PHYSICS = ROOT / 'configs/dex3_simple_graspable_doll_grasp_v2.json'


def pose(rotation, position):
    t = np.eye(4); t[:3, :3] = rotation; t[:3, 3] = position
    return t


def mean_pose(values):
    values = np.asarray(values)
    return pose(Rotation.from_matrix(values[:, :3, :3]).mean().as_matrix(),
                np.median(values[:, :3, 3], axis=0))


def load_source(entry):
    ref = entry['current_rebuild_raw_cartesian_reference']
    assert record(ref['path']) == ref
    assert record(entry['source_parquet'])['sha256'] == entry['source_parquet_sha256']
    with np.load(ref['path'], allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    assert str(arrays['source_recording_id']) == entry['source_recording_id']
    assert np.all(np.diff(arrays['source_timestamp']) > 0)
    return arrays


def acquisition_window(times,events,source_state):
    """Recover observed closing when legacy event timestamps collapse a phase.

    This infers a pre-lift tool/object relation, never contact or ownership.
    Valid existing event windows are preserved exactly.
    """
    times=np.asarray(times);ids=np.flatnonzero((times>=events['LEFT_CLOSE_BEGIN'])&(times<events['LEFT_LIFT_BEGIN']))
    if len(ids):return ids,None
    before=np.flatnonzero((times>=events['APPROACH_START'])&(times<events['LEFT_LIFT_BEGIN']))
    if len(before)<2:raise ValueError('SOURCE_EVIDENCE_MISSING: a pre-lift acquisition relation sample')
    # ALOHA named left gripper is source state index6; a decreasing aperture
    # is observed closure. Require a contiguous measured closing run ending
    # before the existing lift event, not samples from lifted transport.
    aperture=np.asarray(source_state)[:,6];end=int(before[-1]);start=end
    while start>int(before[0]) and aperture[start-1]>aperture[start]+1e-6:start-=1
    if end-start<2:raise ValueError('SOURCE_EVIDENCE_MISSING: observed pre-lift closing run for collapsed event window')
    ids=np.arange(start,end+1)
    correction=dict(status='INFERRED_FROM_OBSERVED_GRIPPER_CLOSURE',old_close_begin_s=events['LEFT_CLOSE_BEGIN'],new_close_begin_s=float(times[start]),lift_begin_unchanged_s=events['LEFT_LIFT_BEGIN'],state_channel=6,sample_count=len(ids),rule='Contiguous strictly decreasing measured left aperture before lift; minimum3 samples; 1micrometre numerical difference floor. No contact/force or physical ownership inference.')
    return ids,correction


def phase_record(entry, arrays, g1, task_normalization=None):
    from tools.build_eval35_episode_object_registration import episode_visual_yaw
    calibration = read(CALIBRATION)
    source_id = entry['source_recording_id']
    measurement = next((r for r in calibration['per_episode'] if r['source_name'] == source_id), None)
    if measurement is None:
        extra = read(ROOT / 'outputs/doll_handoff_dataset_b_final/new_episode_conversion/source_image_object_estimates.json')
        measurement = next((v for v in extra.values() if source_id in str(v['observations'])), None)
    if measurement is None:
        raise ValueError('SOURCE_EVIDENCE_MISSING: initial_object_image_registration for ' + source_id)
    for observation in measurement['observations']:
        image = ROOT / observation['image']
        if 'image_sha256' in observation:
            assert record(image)['sha256'] == observation['image_sha256']
    yaw = episode_visual_yaw(source_id, measurement, calibration)
    physics = read(PHYSICS)
    z = physics['object']['table_surface_world_z_m'] + physics['object']['visual_dimensions_m'][2] / 2
    x0 = pose(Rotation.from_quat(yaw['quaternion_xyzw']).as_matrix(),
              [*measurement['doll_initial_center_task_xy_m'], z])
    times = arrays['source_timestamp']
    events = dict(zip(arrays['source_event_names'].astype(str), arrays['source_event_times_sec'].astype(float)))
    close, lift = events['LEFT_CLOSE_BEGIN'], events['LEFT_LIFT_BEGIN']
    prelift,closing_correction=acquisition_window(times,events,arrays['source_state'])
    if closing_correction is not None:
        close=closing_correction['new_close_begin_s'];events['LEFT_CLOSE_BEGIN']=close
    wrists = {}
    for side in ('left', 'right'):
        rotation = g1.model_to_world_rotation(arrays[f'WRIST_{side}_wrist_rotation_model'])
        position = g1.model_to_world_position(arrays[f'WRIST_{side}_wrist_position_model'])
        wrists[side] = np.asarray([pose(r, p) for r, p in zip(rotation, position)])
    gl_values = np.linalg.inv(wrists['left'][prelift]) @ x0
    gl = mean_pose(gl_values)
    inferred_object = wrists['left'] @ gl
    overlap = np.flatnonzero((times >= events['RIGHT_ACQUIRE_SOURCE']) & (times <= events['LEFT_RELEASE_BEGIN']))
    if len(overlap) < 3:
        raise ValueError('SOURCE_EVIDENCE_MISSING: receiver/giver overlap window')
    gr_values = np.linalg.inv(wrists['right'][overlap]) @ inferred_object[overlap]
    gr = mean_pose(gr_values)
    approach_window = np.flatnonzero((times >= close - .3) & (times < close))
    if len(approach_window) < 2:
        raise ValueError('SOURCE_EVIDENCE_MISSING: pregrasp approach axis')
    delta = wrists['left'][prelift[0], :3, 3] - wrists['left'][approach_window[0], :3, 3]
    approach = delta / np.linalg.norm(delta) if np.linalg.norm(delta) > 1e-6 else None
    phases = [('pregrasp', 'APPROACH_START', 'LEFT_CLOSE_BEGIN', 'OPEN'),
              ('acquisition', 'LEFT_CLOSE_BEGIN', 'LEFT_LIFT_BEGIN', 'LEFT_CANDIDATE'),
              ('lift_clearance', 'LEFT_LIFT_BEGIN', 'RIGHT_APPROACH_BEGIN', 'LEFT_CARRY_INFERRED'),
              ('handoff', 'RIGHT_ACQUIRE_SOURCE', 'LEFT_RELEASE_BEGIN', 'DUAL_SUPPORT_INFERRED'),
              ('right_transport', 'RIGHT_TRANSPORT_BEGIN', 'FINAL_RELEASE_BEGIN', 'RIGHT_CARRY_INFERRED'),
              ('placement', 'FINAL_RELEASE_BEGIN', 'TASK_END', 'RELEASE_INTENT')]
    relations = {'left': gl, 'right': gr}
    source_tools={s:np.asarray([pose(r,p) for r,p in zip(arrays[f'source_fk_{s}_tcp_rotation_world'],arrays[f'source_fk_{s}_tcp_position_world'])]) for s in wrists}
    source_left_relation=mean_pose(np.linalg.inv(source_tools['left'][prelift])@x0)
    source_object_prior=source_tools['left']@source_left_relation
    source_right_relation=mean_pose(np.linalg.inv(source_tools['right'][overlap])@source_object_prior[overlap])
    relation_spread = {side: {'position_max_m': float(np.max(np.linalg.norm(v[:, :3, 3] - relations[side][:3, 3], axis=1))),
                             'orientation_max_rad': float(np.max(Rotation.from_matrix(v[:, :3, :3] @ relations[side][:3, :3].T).magnitude()))}
                       for side, v in [('left', gl_values), ('right', gr_values)]}
    result = dict(schema='hybrid_source_phase_v1', source_id=source_id, split='TRAIN40' if entry['TRAIN40'] else 'DEV35',
        transform_convention='T_XY maps Y into X', hand_roles={'giver': 'left', 'receiver': 'right'},
        initial_object_pose_world=x0, registered_wrist_object_relations=relations,
        source_functional_tool_object_relations={'left':source_left_relation,'right':source_right_relation},
        relation_status={'left': 'INFERRED_PRELIFT_RIGID_RELATION', 'right': 'INFERRED_FROM_LEFT_RIGID_CARRY'},
        observed_source_forces='UNKNOWN', observed_source_contacts='UNKNOWN', observed_dynamic_object_poses='UNKNOWN',
        supported_grasp_patches='UNKNOWN_NOT_CERTIFIED_BY_CLOSING_INTENT',
        approach_axis_world=approach, source_closing_axes={s: arrays[f'source_fk_{s}_tcp_rotation_world'][prelift[-1], :, 1] for s in wrists},
        events={k: {'time_s': v, 'status': 'INFERRED_FROM_EXISTING_EVENT_EXTRACTION'} for k, v in events.items()},
        phases=[dict(name=n, time_window_s=[events[a], events[b]], contact_mode=m) for n, a, b, m in phases],
        prelift_sample_indices=prelift, handoff_sample_indices=overlap,
        prelift_relation_evidence=dict(sample_count=len(prelift),status='INFERRED',
            spread_estimable=len(prelift)>1,rule='Use all actual source samples in close-begin <= t < lift-begin. One sample supplies a pose but cannot estimate within-window uncertainty.'),
        closing_window_correction=closing_correction,
        uncertainty={'registration': '2 pixel annotated homography corners; no dynamic object tracking',
                     'object_z': 'INFERRED table height + half existing visual height; not source depth measurement',
                     'orientation': yaw, 'relation_spread': relation_spread,
                     'rigid_carry': 'Assumes no source slip; cannot certify a G1 contact patch'},
        placement={'bin_center_xy_m': read(ROOT/'isaaclab_doll_handoff_scene/scene_layout.json')['bin']['center_world_xy_m'],
                   'bin_height_m': .150, 'settle_min_s': 1., 'orientation_requirement': 'UNKNOWN beyond inferred carried pose'},
        provenance={'raw': record(entry['source_parquet']), 'reference': record(entry['current_rebuild_raw_cartesian_reference']['path']),
                    'image_registration': record(CALIBRATION), 'physics': record(PHYSICS), 'yaw_images': yaw['frames']})
    if task_normalization is not None:
        if task_normalization['mode'] not in ('SOURCE_BIN_TO_QUALIFIED_TARGET_BIN_TRANSLATION','SOURCE_TASK_AXIS_TO_G1_BILATERAL_AXIS'):
            raise ValueError('Unknown common task-frame normalization')
        if 'bin_center_task_xy_m' not in measurement:
            raise ValueError('SOURCE_EVIDENCE_MISSING: registered source bin center')
        source_bin=np.asarray(measurement['bin_center_task_xy_m'],float)
        target_bin=np.asarray(result['placement']['bin_center_xy_m'],float)
        common=np.eye(4)
        if task_normalization['mode']=='SOURCE_TASK_AXIS_TO_G1_BILATERAL_AXIS':
            direction=source_bin-x0[:2,3]
            if np.linalg.norm(direction)<1e-6:raise ValueError('SOURCE_EVIDENCE_MISSING: distinct object/bin task axis')
            # G1 model -Y points from its left arm toward its right arm.
            bilateral=-g1.root_pose[:2,1]
            angle=np.arctan2(bilateral[1],bilateral[0])-np.arctan2(direction[1],direction[0])
            common[:3,:3]=Rotation.from_euler('z',angle).as_matrix()
        common[:2,3]=target_bin-common[:2,:2]@source_bin
        # Change one world/task frame for BOTH wrists, object, and bin.
        # The relative contact transforms were computed before this change.
        result['source_task_geometry']=dict(initial_object_pose=x0.copy(),bin_center_xy_m=source_bin)
        result['task_registration']=dict(mode=task_normalization['mode'],T_target_source=common,
            target_bin_center_xy_m=target_bin,source_bin_center_xy_m=source_bin,
            source_object_to_bin_xy_m=source_bin-x0[:2,3],
            target_object_to_bin_xy_m=target_bin-(common@x0)[:2,3],
            independent_hand_rebasing=False,method_or_outcome_used=False,
            reference='Qualified fixed bin/table/G1 scene; bin position is not a method output')
        result['initial_object_pose_world']=common@x0
        if result['approach_axis_world'] is not None:
            result['approach_axis_world']=common[:3,:3]@result['approach_axis_world']
        wrists={side:common@values for side,values in wrists.items()}
        inferred_object=common@inferred_object
    return result, wrists, inferred_object


def run(out, resume=False):
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    split = read(out/'bootstrap/SPLITS.json')['entries']
    selected = read(out/'bootstrap/SELECTION.json')['train_source_ids']
    common = load_common_config(COMMON); g1 = G1Kinematics(common, load_scene(common))
    rows = []
    for source_id in selected:
        entry = next(e for e in split if e['source_recording_id'] == source_id)
        assert entry['TRAIN40'] and not entry['DEV35_DIAGNOSTIC35']
        arrays = load_source(entry)
        record_value, wrists, objects = phase_record(entry, arrays, g1)
        folder = out/'source_phase'/source_id
        atomic_json(folder/'PHASE_RECORD.json', record_value)
        atomic_npz(folder/'SOURCE_PRIORS.npz', source_timestamp=arrays['source_timestamp'],
                   left_wrist_world=wrists['left'], right_wrist_world=wrists['right'],
                   inferred_object_from_left=objects, **{k: v for k, v in arrays.items() if k.startswith('source_fk_')})
        rows.append({'source_id': source_id, 'phase_record': record(folder/'PHASE_RECORD.json')})
    result = dict(status='COMPLETE_WITH_EXPLICIT_UNCERTAINTY', entries=rows)
    atomic_json(out/'source_phase/RESULT.json', result)
    return result
