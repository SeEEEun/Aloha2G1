"""Hash-compatible reuse and independent raw-state common-control audit."""
import numpy as np
from .io import ROOT, read, record, atomic_json

OLD=ROOT/'outputs/contact_coordination_retargeting/20260907T063704Z/common_control'


def launch_if_absent(out):
    """One isolated calibration process; never overwrite an interrupted attempt."""
    import os,subprocess,time
    from .io import ISAAC
    folder=out/'common_control/scripted_captured'
    if (folder/'INVOCATION.json').exists():
        return
    prior=read(OLD/'scripted_d42c796d7e59/INVOCATION.json')
    args=list(prior['command']);args[0]=ISAAC;args[1]=str(ROOT/'tools/contact_coordination/physics_capture.py')
    args[args.index('--output-dir')+1]=str(folder)
    freeze=read(OLD/'CURRENT_DEPENDENCIES_d42c796d7e59.json')
    for row in freeze['files']:
        assert record(row['path'])['sha256']==row['sha256']
    freeze['files'] += [record(ROOT/p) for p in ['tools/contact_coordination/physics_capture.py','tools/contact_coordination/io.py','tools/reconciled_ab/runtime_geometry.py']]
    atomic_json(folder/'DEPENDENCIES.json',freeze)
    args[args.index('--direct-freeze-manifest')+1]=str(folder/'DEPENDENCIES.json')
    atomic_json(folder/'INVOCATION.json',dict(command=args,type='COMMON_CONTROL',method_result=False,read_only_instrumentation=True,timeout_s=1200))
    started=time.monotonic()
    try:
        with (folder/'engine.log').open('w') as log:
            proc=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200,
                                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        result=dict(returncode=proc.returncode,wall_seconds=time.monotonic()-started)
    except subprocess.TimeoutExpired:
        result=dict(returncode='TIMEOUT',status='INFRASTRUCTURE_INVALID',wall_seconds=time.monotonic()-started)
    atomic_json(folder/'PROCESS.json',result)


def run(out,resume=False):
    from tools.finalize_common_dex3_grasp_qualification import standalone,full_task
    freeze=read(OLD/'CURRENT_DEPENDENCIES_d42c796d7e59.json')
    for dependency in freeze['files']:
        assert record(dependency['path'])['sha256']==dependency['sha256'],dependency['path']
    results=[]
    for side in ('left','right'):
        folder=OLD/f'{side}_d42c796d7e59'
        old=read(folder/'QUALIFICATION.json')
        for label,name in [('event_log','event_log.npz'),('trial_result','trial_result.json'),('runtime_summary','DIRECT_EXECUTION_RUNTIME_SUMMARY.json')]:
            assert record(folder/name)['sha256']==old['artifact_sha256'][label]
        score=standalone(folder,side)
        score.update(control=side,reused=True,method_result=False,dependency_manifest=record(OLD/'CURRENT_DEPENDENCIES_d42c796d7e59.json'))
        results.append(score)
    folder=out/'common_control/scripted_captured'
    launch_if_absent(out)
    if not (folder/'trial_result.json').exists():
        result=dict(status='COMMON_CONTROL_STILL_RUNNING_OR_INCOMPLETE',results=results,M1=False)
        atomic_json(out/'common_control/RESULT.json',result);return result
    for dep in read(folder/'DEPENDENCIES.json')['files']:
        assert record(dep['path'])['sha256']==dep['sha256']
    score=full_task(folder);score.update(control='scripted',reused=False,method_result=False)
    results.append(score)
    joint=read(ROOT/'outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json')['joint_specs']
    lower=np.asarray([r['minimum'] for r in joint]);upper=np.asarray([r['maximum'] for r in joint])
    with np.load(folder/'event_log.npz') as z:
        q=z['measured_q_rad'];excess=np.maximum(np.maximum(lower-q,q-upper),0.)
        raw=dict(maximum_raw_joint_bound_excess_rad=float(excess.max()),
                 raw_joint_bound_excursion_samples=int(np.count_nonzero(excess)),
                 joint_names=[r['joint_name'] for r in joint],maximum_raw_excess_per_joint_rad=excess.max(axis=0),
                 trace=record(folder/'event_log.npz'),states_clipped=False)
        events=z['DIRECT_COMMON_EXECUTION_EVENTS'].astype(str)
        ownership=score['right_ownership_frame'];release=score['final_release_frame'];frames=z['control_frame']
        mask=(frames>=ownership)&(frames<release) if ownership is not None and release is not None else np.zeros(len(frames),bool)
        right=np.max(np.column_stack([z['right_'+s+'_force_n'] for s in ('thumb','index','middle')]),axis=1)
        left=np.max(np.column_stack([z['left_'+s+'_force_n'] for s in ('thumb','index','middle')]),axis=1)
        retained=mask&(right>=.015)&(left<.015)&(z['table_contact_force_n']<=.02)
        from tools.score_episode_registered_physical_eval35_run import longest_rows
        raw['longest_measured_right_only_retention_s']=float(longest_rows(retained)/240.)
    atomic_json(folder/'RAW_NUMERICAL_AND_RETENTION_AUDIT.json',raw)
    geometry=read(folder/'RUNTIME_COLLISION_MODEL.json')
    collider_audit=dict(runtime_inventory=record(folder/'RUNTIME_COLLISION_MODEL.json'),
        articulation=geometry['articulations'],
        runtime_convex_hulls=sum(c['approximation']=='convexHull' and c['enabled'] is not False for c in geometry['colliders']),
        offline_runtime_equivalence_verified=False,
        explanation='Offline robot primitives/detailed visual surfaces differ from authored runtime convex colliders and filters. Self response disabled in reused runtime. No proxy whitelist used to certify execution.',
        method_execution_gate='Complete source-conditioned command must certify runtime collision correspondence before physical execution.')
    atomic_json(out/'common_control/COLLISION_PARITY.json',collider_audit)
    # M1 mechanical controls can pass while the required offline/runtime parity
    # check remains unresolved; report those qualifications separately.
    result=dict(status='MECHANICAL_CONTROLS_VERIFIED_PARITY_UNRESOLVED' if all(r['status']=='PASS' for r in results) else 'COMMON_CONTROL_FAILED',
        mechanical_controls_verified=all(r['status']=='PASS' for r in results),
        M1=False, M1_remaining='offline/runtime collider equivalence and natural-start approach not qualified',
        results=results,new_physics_process_invocations=1+int((out/'common_control/scripted_current/PROCESS.json').exists()),new_completed_physical_controls=1,
        interrupted_infrastructure_invalid=int((out/'common_control/scripted_current/PROCESS.json').exists()),reused_standalone_controls=2,method_rollouts=0,
        natural_start_qualification=False,raw_audit=record(folder/'RAW_NUMERICAL_AND_RETENTION_AUDIT.json'))
    atomic_json(out/'common_control/RESULT.json',result)
    return result
