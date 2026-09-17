"""Recover established DEV35 scenes from saved source-only registration.

The legacy entry contains policy diagnostics too. This adapter consumes only
the source identity, initial object pose, registration transform and raw-image
evidence. It never reads a policy target or an evaluation outcome.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT,read,record,atomic_json,atomic_npz
from .source_phase import COMMON,PHYSICS,pose,mean_pose,load_source,acquisition_window

REGISTRATION=ROOT/'outputs/final_episode_registered_eval35/00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json'


def phase_from_registered_scene(entry,arrays,g1,x0,yaw,registration):
    """Same pre-lift relation arithmetic as source_phase; explicit scene input."""
    times=arrays['source_timestamp'];events=dict(zip(arrays['source_event_names'].astype(str),arrays['source_event_times_sec'].astype(float)))
    prelift,correction=acquisition_window(times,events,arrays['source_state'])
    if correction:events['LEFT_CLOSE_BEGIN']=correction['new_close_begin_s']
    wrists={s:np.asarray([pose(r,p) for r,p in zip(g1.model_to_world_rotation(arrays[f'WRIST_{s}_wrist_rotation_model']),g1.model_to_world_position(arrays[f'WRIST_{s}_wrist_position_model']))]) for s in ('left','right')}
    gl_values=np.linalg.inv(wrists['left'][prelift])@x0;gl=mean_pose(gl_values);objects=wrists['left']@gl
    overlap=np.flatnonzero((times>=events['RIGHT_ACQUIRE_SOURCE'])&(times<=events['LEFT_RELEASE_BEGIN']))
    if len(overlap)<3:raise ValueError('SOURCE_EVIDENCE_MISSING: receiver/giver overlap window')
    gr_values=np.linalg.inv(wrists['right'][overlap])@objects[overlap];gr=mean_pose(gr_values)
    approach_window=np.flatnonzero((times>=events['LEFT_CLOSE_BEGIN']-.3)&(times<events['LEFT_CLOSE_BEGIN']))
    if len(approach_window)<2:raise ValueError('SOURCE_EVIDENCE_MISSING: pregrasp approach axis')
    delta=wrists['left'][prelift[0],:3,3]-wrists['left'][approach_window[0],:3,3]
    approach=delta/np.linalg.norm(delta) if np.linalg.norm(delta)>1e-6 else None
    tools={s:np.asarray([pose(r,p) for r,p in zip(arrays[f'source_fk_{s}_tcp_rotation_world'],arrays[f'source_fk_{s}_tcp_position_world'])]) for s in wrists}
    gl_source=mean_pose(np.linalg.inv(tools['left'][prelift])@x0)
    gr_source=mean_pose(np.linalg.inv(tools['right'][overlap])@(tools['left']@gl_source)[overlap])
    spec=[('pregrasp','APPROACH_START','LEFT_CLOSE_BEGIN','OPEN'),('acquisition','LEFT_CLOSE_BEGIN','LEFT_LIFT_BEGIN','LEFT_CANDIDATE'),
        ('lift_clearance','LEFT_LIFT_BEGIN','RIGHT_APPROACH_BEGIN','LEFT_CARRY_INFERRED'),('handoff','RIGHT_ACQUIRE_SOURCE','LEFT_RELEASE_BEGIN','DUAL_SUPPORT_INFERRED'),
        ('right_transport','RIGHT_TRANSPORT_BEGIN','FINAL_RELEASE_BEGIN','RIGHT_CARRY_INFERRED'),('placement','FINAL_RELEASE_BEGIN','TASK_END','RELEASE_INTENT')]
    relations={'left':gl,'right':gr};spread={s:dict(position_max_m=float(np.max(np.linalg.norm(v[:,:3,3]-relations[s][:3,3],axis=1))),orientation_max_rad=float(np.max(Rotation.from_matrix(v[:,:3,:3]@relations[s][:3,:3].T).magnitude()))) for s,v in [('left',gl_values),('right',gr_values)]}
    phase=dict(schema='hybrid_source_phase_v1',source_id=entry['source_recording_id'],split='DEV35',transform_convention='T_XY maps Y into X',hand_roles={'giver':'left','receiver':'right'},
        initial_object_pose_world=x0,registered_wrist_object_relations=relations,source_functional_tool_object_relations={'left':gl_source,'right':gr_source},
        relation_status={'left':'INFERRED_PRELIFT_RIGID_RELATION','right':'INFERRED_FROM_LEFT_RIGID_CARRY'},observed_source_forces='UNKNOWN',observed_source_contacts='UNKNOWN',observed_dynamic_object_poses='UNKNOWN',
        supported_grasp_patches='UNKNOWN_NOT_CERTIFIED_BY_CLOSING_INTENT',approach_axis_world=approach,
        source_closing_axes={s:arrays[f'source_fk_{s}_tcp_rotation_world'][prelift[-1],:,1] for s in wrists},
        events={k:dict(time_s=v,status='INFERRED_FROM_EXISTING_EVENT_EXTRACTION') for k,v in events.items()},
        phases=[dict(name=n,time_window_s=[events[a],events[b]],contact_mode=m) for n,a,b,m in spec],prelift_sample_indices=prelift,handoff_sample_indices=overlap,
        prelift_relation_evidence=dict(sample_count=len(prelift),status='INFERRED',spread_estimable=len(prelift)>1,rule='All actual close-begin <= t < lift-begin samples; no lifted pose is an initial object estimate.'),closing_window_correction=correction,
        uncertainty=dict(registration='Existing source image registration, not target-policy output',object_z='INFERRED table+half existing visual height',orientation=yaw,relation_spread=spread,rigid_carry='Assumes no source slip; cannot certify target contact patch'),
        placement=dict(bin_center_xy_m=read(ROOT/'isaaclab_doll_handoff_scene/scene_layout.json')['bin']['center_world_xy_m'],bin_height_m=.150,settle_min_s=1.,orientation_requirement='UNKNOWN beyond inferred carried pose'),
        provenance=dict(raw=record(entry['source_parquet']),reference=record(entry['current_rebuild_raw_cartesian_reference']['path']),image_registration=registration,physics=record(PHYSICS),yaw_images=yaw['frames']))
    return phase,wrists,objects


def normalize_registered_phase(phase,wrists,objects,measurement,g1,normalization):
    """Apply the frozen source_phase SE(2) rule to legacy saved scene inputs.

    This adapter is checked numerically against source_phase on every source
    with the original measurement record. No robot solution enters the map.
    """
    import copy
    result=copy.deepcopy(phase);x=np.asarray(phase['initial_object_pose_world'])
    source_bin=np.asarray(measurement['bin_center_task_xy_m'],float)
    target_bin=np.asarray(phase['placement']['bin_center_xy_m'],float)
    common=np.eye(4)
    mode=normalization['mode']
    if mode=='SOURCE_TASK_AXIS_TO_G1_BILATERAL_AXIS':
        direction=source_bin-x[:2,3]
        if np.linalg.norm(direction)<1e-6:raise ValueError('SOURCE_EVIDENCE_MISSING: distinct object/bin task axis')
        bilateral=-g1.root_pose[:2,1]
        angle=np.arctan2(bilateral[1],bilateral[0])-np.arctan2(direction[1],direction[0])
        common[:3,:3]=Rotation.from_euler('z',angle).as_matrix()
    elif mode!='SOURCE_BIN_TO_QUALIFIED_TARGET_BIN_TRANSLATION':
        raise ValueError('Unknown common task-frame normalization')
    common[:2,3]=target_bin-common[:2,:2]@source_bin
    result['source_task_geometry']=dict(initial_object_pose=x.copy(),bin_center_xy_m=source_bin)
    result['task_registration']=dict(mode=mode,T_target_source=common,
        target_bin_center_xy_m=target_bin,source_bin_center_xy_m=source_bin,
        source_object_to_bin_xy_m=source_bin-x[:2,3],
        target_object_to_bin_xy_m=target_bin-(common@x)[:2,3],
        independent_hand_rebasing=False,method_or_outcome_used=False,
        reference='Qualified fixed bin/table/G1 scene; bin position is not a method output')
    result['initial_object_pose_world']=common@x
    if result['approach_axis_world'] is not None:
        result['approach_axis_world']=common[:3,:3]@result['approach_axis_world']
    return result,{s:common@v for s,v in wrists.items()},common@objects


def build_normalized(out):
    """Export all35 through the TRAIN-frozen rule before reference/policy freeze."""
    import shutil
    from .generalization_gate import require
    require(out,'reference_coupling10')
    from .source_contract import registered_functional_wrists
    from .source_phase import CALIBRATION,phase_record
    from tools.finalize_doll_handoff_dataset_b import source_image_object_estimate
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    area=out/'DEV35_scenes';manifest_path=area/'MANIFEST.json'
    if manifest_path.exists():
        manifest=read(manifest_path)
        if manifest.get('task_normalization')!=record(out/'target_repair/TASK_FRAME_NORMALIZATION.json'):
            raise ValueError('Existing DEV scene version differs; preserve/version it before preparing a new study')
        for dep in manifest['dependencies']:
            if record(dep['path'])!=dep:raise ValueError('Changed DEV scene input or output: '+dep['path'])
        return manifest
    split=read(out/'SPLIT_CONTRACT.json');old=read(REGISTRATION)
    if old['physical_eval35_outcomes_read'] or old['policy_output_derived_object_placement']:
        raise ValueError('Registration must be source-derived')
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg))
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json')
    normalization=read(out/'target_repair/TASK_FRAME_NORMALIZATION.json')
    known={r['source_name']:r for r in read(CALIBRATION)['per_episode']}
    extra=read(ROOT/'outputs/doll_handoff_dataset_b_final/new_episode_conversion/source_image_object_estimates.json')
    for value in extra.values():
        sid=next(part for part in Path(value['observations'][0]['image']).parts if part.startswith('GoPark_'))
        known[sid]=value
    entries=read(out/'bootstrap/SPLITS.json')['entries'];rows=[];regressions=[];dependencies=[]
    for sid in split['evaluation_source_ids']:
        saved=next(e for e in old['entries'] if e['source_recording']==sid)
        source=next(e for e in entries if e['source_recording_id']==sid)
        measurement=known[sid] if sid in known else source_image_object_estimate(sid)
        # Reuse the existing source-only detector; never use a held-out policy
        # pose, event-based object center, or fitted target-robot trajectory.
        np.testing.assert_allclose(measurement['doll_initial_center_task_xy_m'],saved['source_object_pose']['position_xyz_m'][:2],atol=1e-7,rtol=0)
        np.testing.assert_allclose(saved['source_object_pose']['position_xyz_m'],saved['target_object_pose']['position_xyz_m'],atol=1e-12,rtol=0)
        for image in saved['orientation_evidence']['frames']:
            if record(image['image'])['sha256']!=image['image_sha256']:raise ValueError('Changed raw source image')
        measurement_path=area/sid/'SOURCE_TASK_MEASUREMENT.json';atomic_json(measurement_path,measurement)
        input_records=[record(ROOT/o['image']) for o in measurement['observations']]
        if measurement.get('reference_image'):input_records.append(record(measurement['reference_image']))
        x=pose(Rotation.from_quat(saved['source_object_pose']['quaternion_xyzw']).as_matrix(),saved['source_object_pose']['position_xyz_m'])
        arrays=load_source(source)
        phase,wrists,objects=phase_from_registered_scene(source,arrays,g,x,saved['orientation_evidence'],record(REGISTRATION))
        phase,wrists,objects=normalize_registered_phase(phase,wrists,objects,measurement,g,normalization)
        if sid in known:
            expected,ew,eo=phase_record(source,arrays,g,normalization)
            for field in ('initial_object_pose_world','approach_axis_world'):
                np.testing.assert_allclose(phase[field],expected[field],atol=1e-12,rtol=0)
            for side in ('left','right'):
                np.testing.assert_allclose(wrists[side],ew[side],atol=1e-12,rtol=0)
                np.testing.assert_allclose(phase['source_functional_tool_object_relations'][side],expected['source_functional_tool_object_relations'][side],atol=1e-12,rtol=0)
            np.testing.assert_allclose(objects,eo,atol=1e-12,rtol=0)
            regressions.append(sid)
        phase['provenance'].update(task_measurement=record(measurement_path),task_normalization=record(out/'target_repair/TASK_FRAME_NORMALIZATION.json'),source_measurement_images=input_records)
        folder=out/'source_phase'/sid;backup=area/'before_task_normalization'/sid
        if folder.exists() and not backup.exists():shutil.copytree(folder,backup)
        atomic_json(folder/'PHASE_RECORD.json',phase)
        atomic_npz(folder/'SOURCE_PRIORS.npz',source_timestamp=arrays['source_timestamp'],left_wrist_world=wrists['left'],right_wrist_world=wrists['right'],inferred_object_from_left=objects,**{k:v for k,v in arrays.items() if k.startswith('source_fk_')})
        adapted={};transforms={}
        for side in ('left','right'):
            previous=arrays[f'WRIST_{side}_wrist_to_tool'];current=np.asarray(cal['contacts']['handoff_'+side]['T_wrist_H'])
            adapted[side]=registered_functional_wrists(wrists[side],previous,current)
            transforms[side]=dict(original_wrist_to_tool=previous,shared_wrist_to_tool=current)
            np.testing.assert_allclose(adapted[side]@current,wrists[side]@previous,atol=1e-12,rtol=0)
        atomic_npz(folder/'FUNCTIONAL_WRIST_PRIORS.npz',source_timestamp=arrays['source_timestamp'],**{s+'_wrist_world':v for s,v in adapted.items()})
        atomic_json(folder/'FUNCTIONAL_FRAME_AUDIT.json',dict(status='REGISTERED_FUNCTIONAL_TOOL_POSES_PRESERVED',source_id=sid,phase=record(folder/'PHASE_RECORD.json'),transforms=transforms,independent_hand_rebasing=False,calibration=record(out/'target_repair/CONTACT_CALIBRATION.json'),implementation=record(__file__)))
        x=np.asarray(phase['initial_object_pose_world'])
        target=dict(position_xyz_m=x[:3,3],quaternion_xyzw=Rotation.from_matrix(x[:3,:3]).as_quat())
        path=area/sid/'SOURCE_SCENE.json'
        scene=dict(schema_version='eval35_episode_source_derived_object_registration_v1',status='PASS',physical_eval35_outcomes_read=False,one_global_canonical_object_pose=False,
            authoritative_inputs={str(PHYSICS):record(PHYSICS)['sha256']},entries=[dict(stable_episode_id=sid,A_B_identical_object_pose=True,runtime_initialization_rule=dict(bin_pose_fixed=True),target_object_pose=target,source_registration=record(REGISTRATION),task_registration=phase['task_registration'],original_stable_id=saved['stable_episode_id'])])
        atomic_json(path,scene)
        rows.append(dict(source_id=sid,source_scene=record(path),phase=record(folder/'PHASE_RECORD.json'),raw_images_verified=True))
        dependencies += [record(p) for p in sorted(folder.iterdir()) if p.suffix in ('.json','.npz')]+[record(path),record(measurement_path),*input_records]
    dependencies += [record(p) for p in (REGISTRATION,CALIBRATION,PHYSICS,COMMON,out/'SPLIT_CONTRACT.json',out/'target_repair/TASK_FRAME_NORMALIZATION.json',Path(__file__),ROOT/'tools/finalize_doll_handoff_dataset_b.py',ROOT/'tools/calibrate_doll_handoff_scene_from_cam_high.py',ROOT/'tools/contact_coordination/source_phase.py')]
    result=dict(status='ALL35_SOURCE_SCENES_RECOVERED',rows=rows,ordered_source_ids=split['evaluation_source_ids'],registration=record(REGISTRATION),
        task_normalization=record(out/'target_repair/TASK_FRAME_NORMALIZATION.json'),dependencies=dependencies,
        common_frozen_source_phase_equivalence_sources=regressions,initial_object_height_compatible=True,legacy_method_diagnostics_consumed=False,implementation=record(__file__),
        source_bin_orientation='UNKNOWN; metric center and object-to-bin direction only',new_policy_or_reference_trials_run=0)
    atomic_json(manifest_path,result);return result


def build(out):
    if (out/'target_repair/TASK_FRAME_NORMALIZATION.json').exists():return build_normalized(out)
    return build_legacy(out)


def build_legacy(out):
    from .source_contract import registered_functional_wrists
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    split=read(out/'SPLIT_CONTRACT.json');old=read(REGISTRATION);cal=read(out/'target_repair/CONTACT_CALIBRATION.json')
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));entries=read(out/'bootstrap/SPLITS.json')['entries'];rows=[];reg_record=record(REGISTRATION)
    physics=read(PHYSICS);height=physics['object']['table_surface_world_z_m']+physics['object']['visual_dimensions_m'][2]/2
    if old['physical_eval35_outcomes_read'] or old['policy_output_derived_object_placement']:raise ValueError('Registration provenance is not source-only')
    for sid in split['evaluation_source_ids']:
        saved=next(r for r in old['entries'] if r['source_recording']==sid);source=next(r for r in entries if r['source_recording_id']==sid)
        if not saved['A_B_identical_object_pose'] or saved['source_object_task_frame_source']['ACT_outputs_used_for_object_pose']:raise ValueError('Invalid source scene provenance')
        for image in saved['orientation_evidence']['frames']:
            if record(image['image'])['sha256']!=image['image_sha256']:raise ValueError('Changed raw source image')
        target=saved['target_object_pose'];x=pose(Rotation.from_quat(target['quaternion_xyzw']).as_matrix(),np.asarray(target['position_xyz_m']))
        if abs(x[2,3]-height)>1e-9:raise ValueError('Registered initial height incompatible with current object geometry')
        folder=out/'source_phase'/sid;arrays=load_source(source)
        phase,wrists,objects=phase_from_registered_scene(source,arrays,g,x,saved['orientation_evidence'],reg_record)
        if (folder/'PHASE_RECORD.json').exists():
            existing=read(folder/'PHASE_RECORD.json')
            for field in ('initial_object_pose_world','prelift_sample_indices','handoff_sample_indices'):
                np.testing.assert_allclose(existing[field],phase[field],atol=1e-12)
            for side in ('left','right'):np.testing.assert_allclose(existing['registered_wrist_object_relations'][side],phase['registered_wrist_object_relations'][side],atol=1e-12)
        else:
            atomic_json(folder/'PHASE_RECORD.json',phase)
            atomic_npz(folder/'SOURCE_PRIORS.npz',source_timestamp=arrays['source_timestamp'],left_wrist_world=wrists['left'],right_wrist_world=wrists['right'],inferred_object_from_left=objects,**{k:v for k,v in arrays.items() if k.startswith('source_fk_')})
            adapted={};transforms={}
            for side in ('left','right'):
                previous=arrays[f'WRIST_{side}_wrist_to_tool'];current=np.asarray(cal['contacts']['handoff_'+side]['T_wrist_H'])
                adapted[side]=registered_functional_wrists(wrists[side],previous,current);transforms[side]=dict(original_wrist_to_tool=previous,shared_wrist_to_tool=current)
                np.testing.assert_allclose(adapted[side]@current,wrists[side]@previous,atol=1e-12)
            atomic_npz(folder/'FUNCTIONAL_WRIST_PRIORS.npz',source_timestamp=arrays['source_timestamp'],**{s+'_wrist_world':v for s,v in adapted.items()})
            atomic_json(folder/'FUNCTIONAL_FRAME_AUDIT.json',dict(status='REGISTERED_FUNCTIONAL_TOOL_POSES_PRESERVED',source_id=sid,phase=record(folder/'PHASE_RECORD.json'),transforms=transforms,independent_hand_rebasing=False,calibration=record(out/'target_repair/CONTACT_CALIBRATION.json'),implementation=record(__file__)))
        scene=dict(schema_version='eval35_episode_source_derived_object_registration_v1',status='PASS',physical_eval35_outcomes_read=False,one_global_canonical_object_pose=False,
            authoritative_inputs={str(PHYSICS):record(PHYSICS)['sha256']},entries=[dict(stable_episode_id=sid,A_B_identical_object_pose=True,runtime_initialization_rule=dict(bin_pose_fixed=True),target_object_pose=target,source_registration=reg_record,original_stable_id=saved['stable_episode_id'])])
        path=out/'DEV35_scenes'/sid/'SOURCE_SCENE.json';atomic_json(path,scene)
        rows.append(dict(source_id=sid,source_scene=record(path),phase=record(folder/'PHASE_RECORD.json'),raw_images_verified=True))
    result=dict(status='ALL35_SOURCE_SCENES_RECOVERED',rows=rows,ordered_source_ids=split['evaluation_source_ids'],registration=reg_record,
        compared_existing_phase_records=8,initial_object_height_compatible=True,legacy_method_diagnostics_consumed=False,implementation=record(__file__))
    atomic_json(out/'DEV35_scenes/MANIFEST.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();print(build(a.run_dir)['status'])
