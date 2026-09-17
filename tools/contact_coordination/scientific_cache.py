"""Content/semantic keys for the existing phase and connection directories.

No source solution is a cache default. Paths and report/plot metadata do not
identify a scientific problem. Source identity, all consumed source arrays,
contacts, backend, geometry, constraints and method code do.
"""
from pathlib import Path
from functools import lru_cache
import hashlib,json,sys
import numpy as np
from .io import ROOT,read,record
from .source_phase import COMMON

REPORT_KEYS={'plotting','plot','figure','figures','report','report_metadata','notes','note',
    'description','interpretation','limitation','limitations','basis','use','rule',
    'evidence','runtime_s','runtime_seconds','planning_seconds','created_unix_s'}
PHASE_KEYS=('source_id','hand_roles','initial_object_pose_world',
    'registered_wrist_object_relations','source_functional_tool_object_relations',
    'relation_status','approach_axis_world','source_closing_axes','prelift_sample_indices',
    'handoff_sample_indices','placement','source_task_geometry','task_registration')
CONTACT_KEYS=('side','T_HO','T_wrist_H','seed_q','measured_finger_q','commanded_finger_q',
    'measured_T_HO','T_commandHand_measuredHand','acquisition_reference_T_HO','source_receiving_region')
CALIBRATION_FILES=('CONTACT_CALIBRATION.json','CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json',
    'LOADED_CARRY_GEOMETRY_f83d2e29156f.json','CONTACT_SEPARATION_CANDIDATES.json',
    'CALIBRATED_HANDOFF_GRAVITY.json','CALIBRATED_RIGHT_CARRY_GRAVITY.json')
CODE_FILES=('io.py','source_phase.py','source_contract.py','receiving_relation.py','scientific_cache.py',
    'episode_context.py','conversion_attempt.py','acquisition_plan.py','handoff_repair.py',
    'full_task_plan.py','wrist_reference.py','targets.py','prototype.py','morphology_repair.py',
    'planning_kinematics.py','planner.py','runtime_hulls.py','loaded_contact_geometry.py',
    'giver_clearance.py','physical_attempt.py','phase_clock_runtime.py','phase_physics.py',
    'demonstration_physics.py','demonstration_observation.py','g1_rgb_observer.py','score_hybrid.py','demo_alignment.py',
    'contact_transition_geometry.py','contact_candidate_region.py','passive_hand_clearance.py','calibration_parameters.py',
    'joint_path_planner.py','path_quality.py','source_motion_prior.py','interaction_candidates.py','shared_handoff_candidates.py','interaction_chain.py','execution_timing.py','contact_command_retiming.py','practical_parameters.py','incidental_contact.py')


def canonical(value):
    if isinstance(value,np.ndarray):
        a=np.ascontiguousarray(value)
        return {'array_shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in sorted(value.items()) if k not in REPORT_KEYS}
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    if isinstance(value,Path):return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_payload(phase,priors):
    result={key:phase.get(key) for key in PHASE_KEYS}
    result['raw_recording_sha256']=phase['provenance']['raw']['sha256']
    result['events']={k:v['time_s'] for k,v in phase['events'].items()}
    result['priors']=priors
    return result


def calibration_payload(value):
    if 'contacts' in value:
        return dict(joint_names=value['joint_names'],contacts={name:{k:c[k] for k in CONTACT_KEYS if k in c} for name,c in value['contacts'].items()},
            receiver_capture_transition={k:value.get('receiver_capture_transition',{}).get(k) for k in ('T_preobject_postobject','gravity_up_in_preobject')})
    keys=('joint_names','arm_measured_minus_command_rad','uncommanded_joint_positions_rad',
        'T_wrist_H','measured_T_HO','command_T_HO','measured_seed_q','command_seed_q',
        'offset_object_frame_m','maximum_offset_m','fractions','up_direction_object_frame')
    return {k:value[k] for k in keys if k in value}


@lru_cache(maxsize=512)
def _file_hash(path,size,mtime_ns,ctime_ns):
    # Stat tuple only avoids repeated reads within one process. The returned
    # key is a content digest, not a stat or folder-name cache identity.
    return record(path)['sha256']


def content_hash(path):
    p=Path(path).resolve();s=p.stat()
    return _file_hash(str(p),s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def file_payload(path):
    p=Path(path)
    if p.suffix=='.npz':
        with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
    if p.suffix=='.json':return read(p)
    return dict(content_sha256=content_hash(p))


def scientific_inputs(out,sid,extra_paths=()):
    import scipy,mujoco
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    source=out/'source_phase'/sid;phase=read(source/'PHASE_RECORD.json')
    priors=file_payload(source/'SOURCE_PRIORS.npz')
    functional=source/'FUNCTIONAL_WRIST_PRIORS.npz'
    if functional.exists():priors['registered_functional_wrists']=file_payload(functional)
    cfg=load_common_config(COMMON);scene=load_scene(cfg)
    from tools.common_execution_isaac_runtime import PHYSICAL_ENVIRONMENT,COMMON_PHYSICAL_CONTROLLER
    from .calibration_parameters import parameters
    from .practical_parameters import parameters as practical_parameters
    result=dict(schema='hybrid_scientific_cache_v3',source=source_payload(phase,priors),
        calibrated_parameters=parameters(out),
        practical_parameters=practical_parameters(out),
        calibration={name:calibration_payload(read(out/'target_repair'/name)) for name in CALIBRATION_FILES if (out/'target_repair'/name).exists()},
        code=CODE_AT_PROCESS_IMPORT,
        shared_code={name:content_hash(ROOT/name) for name in ('tools/doll_handoff_retargeting/models.py','tools/doll_handoff_retargeting/common.py',
            'tools/common_execution_layer.py','tools/common_execution_isaac_runtime.py','tools/direct_physical_execution_layer.py','tools/finalize_common_dex3_grasp_qualification.py')},
        backend=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,mujoco=mujoco.__version__),
        common_configuration=cfg,scene=scene,
        model={key:content_hash(cfg['models'][key]) for key in ('g1_xml','dex3_whole_hand_geometry')},
        hulls={k:read(out/'target_repair/runtime_bin150/RUNTIME_HULLS.json')[k] for k in ('rows','bin_geometry')},
        hull_vertices=file_payload(out/'target_repair/runtime_bin150/RUNTIME_HULL_VERTICES.npz'),
        physics=read(ROOT/'configs/dex3_simple_graspable_doll_grasp_v2.json'),
        physical_environment=read(PHYSICAL_ENVIRONMENT),common_hand_controller=read(COMMON_PHYSICAL_CONTROLLER),
        joint_contract=read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json')['joint_specs'],
        limits=read(ROOT/'outputs/policy_execution_stability_review/jerk_limited_otg/common_g1_28d_ruckig_config.json'),
        initial=read(ROOT/'outputs/paper_core_ab/COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json'),
        extra=[file_payload(p) for p in extra_paths])
    return result


def key(out,sid,stage,parameters=None,extra_paths=()):
    return digest(dict(stage=stage,parameters=parameters or {},inputs=scientific_inputs(out,sid,extra_paths)))


# A long worker must never key an old imported implementation using new bytes
# edited on disk partway through its computation. The outer runner invalidates
# the stage if current dependencies change; these inner caches retain the
# actual process-start code identity rather than claiming the new revision.
CODE_AT_PROCESS_IMPORT={name:content_hash(ROOT/'tools/contact_coordination'/name) for name in CODE_FILES}
