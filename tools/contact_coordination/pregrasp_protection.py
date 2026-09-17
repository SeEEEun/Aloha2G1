"""Read-only PhysX contact telemetry and pre-acquisition protection audit."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import read,record,atomic_json,atomic_npz


def instrument(source):
    anchor='    table_sensor = ContactSensor(\n';assert source.count(anchor)==1
    source=source.replace(anchor,'''    # Supported one-to-many filter: one object against individual robot bodies.
    # The filter index provides the exact contacting robot link.
    protected_robot_paths = sorted(str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(G1))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI))
    if not protected_robot_paths: raise RuntimeError("No robot bodies for protected-object telemetry")
    protected_object_sensor = ContactSensor(ContactSensorCfg(
        prim_path=DOLL, update_period=0.0, filter_prim_paths_expr=protected_robot_paths,
        track_contact_points=True, max_contact_data_count_per_prim=64*len(protected_robot_paths), force_threshold=0.0))
    protected_contact_rows = []
    protected_sensor_errors = []
'''+anchor)
    anchor='        table_sensor.update(dt, force_recompute=True)\n';assert source.count(anchor)==1
    source=source.replace(anchor,'''        protected_object_sensor.update(dt, force_recompute=True)
'''+anchor)
    anchor='            table_rows, table_error = contact_rows(table_sensor, dt)\n';assert source.count(anchor)==1
    source=source.replace(anchor,'''            try:
                forces, points, _, separations, counts, starts = protected_object_sensor.contact_view.get_contact_data(dt)
                forces = numpy(forces).reshape(-1); separations = numpy(separations).reshape(-1)
                points = numpy(points).reshape(-1,3)
                counts = numpy(counts).reshape(-1).astype(np.int64); starts = numpy(starts).reshape(-1).astype(np.int64)
                if len(counts) != len(protected_robot_paths): raise RuntimeError("Unexpected protected-object filter layout")
                for filter_index, robot_path in enumerate(protected_robot_paths):
                    for contact_index in range(int(starts[filter_index]),int(starts[filter_index]+counts[filter_index])):
                        force=float(abs(forces[contact_index])); separation=float(separations[contact_index])
                        if force>1.e-8 or separation<=0.:
                            protected_contact_rows.append(dict(control_frame=control_frame,physics_step=physics_step,
                                timestamp_s=physics_step*dt,phase=str(label),robot_link=robot_path.rsplit("/",1)[-1],
                                force_n=force,separation_m=separation,point_world_m=points[contact_index].tolist(),impulse_ns=force*dt))
            except Exception as error:
                message=type(error).__name__+": "+str(error)
                if message not in protected_sensor_errors: protected_sensor_errors.append(message)
'''+anchor)
    anchor='    arrays = {key: np.asarray(value) for key, value in records.items()}\n';assert source.count(anchor)==1
    source=source.replace(anchor,'''    from tools.contact_coordination.io import atomic_json as save_contacts
    save_contacts(output_dir / "ROBOT_OBJECT_CONTACTS.json", dict(
        status="PASS" if not protected_sensor_errors else "SENSOR_ERROR",
        rows=protected_contact_rows, errors=protected_sensor_errors,
        kind="MEASURED_PHYSX_CONTACT_DATA", controls_changed=False,
        covered_links=protected_robot_paths, filter_contract="ONE_OBJECT_TO_MANY_INDIVIDUAL_ROBOT_BODIES", force_floor_n=1.e-8))
'''+anchor)
    return source


def audit_trial(folder,trace):
    folder=Path(folder);contact=read(folder/'ROBOT_OBJECT_CONTACTS.json')
    if contact['status']!='PASS':raise RuntimeError('Read-only object contact sensor failure: '+str(contact['errors']))
    phase=np.asarray(trace['NOMINAL_PHASE']).astype(str);frames=np.asarray(trace['control_frame']);times=np.asarray(trace['timestamp_s'])
    start=np.flatnonzero(phase=='ACQUISITION_INGRESS')
    if not len(start):raise AssertionError('Complete command lacks an acquisition ingress boundary')
    acquisition_index=int(start[0]);acquisition_frame=int(frames[acquisition_index]);rows=contact['rows']
    first=min(rows,key=lambda r:r['physics_step']) if rows else None
    premature=[r for r in rows if r['control_frame']<acquisition_frame]
    initial=read(folder/'PRE_COMMAND_SCENE.json')['actual_object_pose_xyzw'];position=np.asarray(trace['object_position_world_m'])
    rotation=Rotation.from_quat(trace['object_quaternion_xyzw']);origin=Rotation.from_quat(initial[3:])
    count=max(1,acquisition_index)
    displacement=float(np.max(np.linalg.norm(position[:count]-np.asarray(initial[:3]),axis=1)))
    angle=float(np.max((origin.inv()*rotation[:count]).magnitude()))
    grasp=np.flatnonzero(phase=='POWER_GRASP');grasp_begin=int(grasp[0]) if len(grasp) else acquisition_index
    grasp_count=max(1,grasp_begin)
    before_grasp_displacement=float(np.max(np.linalg.norm(position[:grasp_count]-np.asarray(initial[:3]),axis=1)))
    before_grasp_rotation=float(np.max((origin.inv()*rotation[:grasp_count]).magnitude()))
    # The available body origins give an observed approach direction. Compare
    # it against the commanded ingress FK direction; report errors, never tune.
    names=list(trace['body_names']);wrist=names.index('left_wrist_yaw_link');ingress=np.flatnonzero(phase=='ACQUISITION_INGRESS')
    measured=np.asarray(trace['body_position_world_m']);direction=measured[ingress[-1],wrist]-measured[ingress[0],wrist]
    command=np.asarray(trace['EXECUTED_COMMAND']);deviation=None
    from .source_phase import COMMON
    from .planning_kinematics import G1Kinematics
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));points=[]
    for i in (ingress[0],ingress[-1]):g.assign(command[i,:14]);points.append(g.model_to_world_position(g.wrist_pose('left')[:3,3]))
    intended=points[1]-points[0]
    if np.linalg.norm(direction)*np.linalg.norm(intended)>1e-12:deviation=float(np.arccos(np.clip(direction@intended/(np.linalg.norm(direction)*np.linalg.norm(intended)),-1,1)))
    source=read(folder/'input/SOURCE_SCENE.json')['entries'][0]['source_phase']['path']
    source_phase=read(source);source_axis=np.asarray(source_phase['approach_axis_world'])
    source_deviation=None
    if np.linalg.norm(direction)*np.linalg.norm(source_axis)>1e-12:
        source_deviation=float(np.arccos(np.clip(direction@source_axis/(np.linalg.norm(direction)*np.linalg.norm(source_axis)),-1,1)))
    issued_indices=np.r_[np.flatnonzero(np.diff(frames)!=0),len(frames)-1];issued=command[issued_indices]
    measured_q=np.asarray(trace['MEASURED_Q'])[issued_indices]
    result=dict(status='PREGRASP_OBJECT_CONTACT' if premature else 'PASS',first_robot_object_contact=first,
        first_contact_time_s=first['timestamp_s'] if first else None,first_contact_phase=first['phase'] if first else None,
        contacting_robot_link=first['robot_link'] if first else None,acquisition_begin_frame=acquisition_frame,
        premature_contact_count=len(premature),first_premature_contact=premature[0] if premature else None,
        maximum_object_displacement_before_acquisition_m=displacement,maximum_object_rotation_before_acquisition_rad=angle,
        maximum_object_displacement_before_grasp_m=before_grasp_displacement,maximum_object_rotation_before_grasp_rad=before_grasp_rotation,
        measured_vs_commanded_ingress_direction_deviation_rad=deviation,
        measured_vs_source_approach_direction_deviation_rad=source_deviation,source_approach_axis_world=source_axis,
        maximum_command_velocity_rad_s=float(np.max(abs(np.diff(issued,axis=0))*30.)),
        maximum_command_acceleration_rad_s2=float(np.max(abs(np.diff(issued,n=2,axis=0))*900.)),
        maximum_measured_joint_velocity_rad_s=float(np.max(abs(np.diff(measured_q,axis=0))*30.)),
        maximum_measured_joint_acceleration_rad_s2=float(np.max(abs(np.diff(measured_q,n=2,axis=0))*900.)),
        maximum_measured_object_angular_speed_rad_s=float(np.max(np.linalg.norm(trace['object_angular_velocity_rad_s'],axis=1))),
        trace=record(folder/'event_log.npz'),contact_evidence=record(folder/'ROBOT_OBJECT_CONTACTS.json'),
        contact_is_measured_not_reconstructed=True,scoring_thresholds_unchanged=True,full_failure_recording_continues=True)
    policy_path=folder/'input/INCIDENTAL_CONTACT_POLICY.json'
    if policy_path.exists() and read(policy_path)['enabled']:
        from .incidental_contact import measured_assessment
        assessment=measured_assessment(read(policy_path),premature,trace,acquisition_index,initial)
        result.update(status='PASS' if assessment['accepted'] else 'PREGRASP_OBJECT_CONTACT',
            incidental_contact_assessment=assessment,incidental_contacts_logged_not_terminal=True,
            frozen_task_tolerance=record(policy_path))
    path=folder/'PREGRASP_OBJECT_PROTECTION.json';atomic_json(path,result);return path


def audit_all(out):
    value=read(out/'PHYSICAL_VERIFICATION.json');rows=[]
    old=Path(read(out/'SOURCE_GUIDED_RRT_REBUILD.json')['old_run'])
    old_rows={(r['source_id'],r['method_key']):r for r in read(old/'PHYSICAL_VERIFICATION.json')['rows']};paired=[]
    for row in value['rows']:
        if not row['physics_executed']:continue
        path=Path(row['folder'])/'PREGRASP_OBJECT_PROTECTION.json'
        rows.append(dict(source_id=row['source_id'],method=row['method_key'],audit=record(path),**read(path)))
        previous=old_rows.get((row['source_id'],row['method_key']))
        if previous and previous['physics_executed']:
            folder=Path(previous['folder']);initial=read(folder/'PRE_COMMAND_SCENE.json')['actual_object_pose_xyzw']
            with np.load(folder/'event_log.npz') as trace:
                phase=trace['NOMINAL_PHASE'].astype(str);begin=int(np.flatnonzero(phase=='ACQUISITION_INGRESS')[0]);count=max(1,begin)
                distance=float(np.max(np.linalg.norm(trace['object_position_world_m'][:count]-np.asarray(initial[:3]),axis=1)))
                angle=float(np.max((Rotation.from_quat(initial[3:]).inv()*Rotation.from_quat(trace['object_quaternion_xyzw'][:count])).magnitude()))
            current=rows[-1]
            paired.append(dict(source_id=row['source_id'],method=row['method_key'],
                old_pregrasp_displacement_m=distance,new_pregrasp_displacement_m=current['maximum_object_displacement_before_acquisition_m'],
                old_pregrasp_rotation_rad=angle,new_pregrasp_rotation_rad=current['maximum_object_rotation_before_acquisition_rad'],
                old_trace=previous['trace'],new_audit=record(path),old_exact_contact_time_available=False))
    path=out/'PREGRASP_OBJECT_PROTECTION_RESULTS.json';atomic_json(path,dict(status='AUDITED',rows=rows,matched_before_after=paired));return [path]
