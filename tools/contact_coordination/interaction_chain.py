"""Bounded hierarchical complete-chain selection with grasp backtracking.

Only final complete commands are eligible for PhysX. Failed branches retain
their first causal error and never become diagnostic commands by implication.
"""
import copy
import shutil
import time
from pathlib import Path
import numpy as np
from .io import read,record,atomic_json,atomic_text
from .scientific_cache import key as cache_key
from .interaction_candidates import acquisition_bank
from .runtime_hulls import object_dimensions


def preserve_incomplete(context,out,sid):
    """Called only by the locked stage worker; preserve partial atomic outputs."""
    prototype=context/'prototype'/sid
    targets=[]
    acquisition=prototype/'morphology_acquisition_v4'
    if acquisition.exists() and not (acquisition/'RESULT.json').exists():targets.append(acquisition)
    for parent in (prototype/'full_task_connection',prototype/'shared_handoff_candidates'):
        if parent.exists():
            for path in parent.iterdir():
                if path.is_dir() and not (path/'CACHE_CONTRACT.json').exists():targets.append(path)
    for path in targets:
        archive=context/'INTERRUPTED_EVIDENCE'/(path.name+'_'+str(time.time_ns()))
        archive.parent.mkdir(exist_ok=True);path.rename(archive)
        from .run_converter_architecture_repair import log
        log(out,'preserved_incomplete_stage',source=str(path),archive=str(archive),source_id=sid)
        with (out/'DECISIONS.md').open('a') as stream:
            stream.write(f'\n- Preserved incomplete connection `{path}` as `{archive}` before deterministic rebuild. Completed upstream artifacts reused.\n')


def branch_context(out,sid,method,signature,candidate,scope='ARCHITECTURE_REPAIR'):
    destination=out/scope/'candidate_contexts'/sid/method/signature[:12]/('grasp_'+candidate['candidate_id'].rsplit(':',1)[1])
    receipt=destination/'BRANCH_INPUTS.json'
    if receipt.exists():
        saved=read(receipt)
        if saved['signature']!=signature:raise ValueError('Immutable branch identity changed')
        return destination
    for relative in ('target_repair','source_phase/'+sid):
        source=out/relative;target=destination/relative
        target.mkdir(parents=True,exist_ok=True)
        for path in source.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.npz') and path.name not in ('CURRENT_CONTACT_REGION_FIT.json',):
                dest=target/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for relative in ('SPLIT_CONTRACT.json','CALIBRATION_PARAMETERS.json'):
        shutil.copy2(out/relative,destination/relative)
    if (out/'PRACTICAL_PARAMETERS.json').exists():shutil.copy2(out/'PRACTICAL_PARAMETERS.json',destination/'PRACTICAL_PARAMETERS.json')
    selection=read(out/'bootstrap/SELECTION.json')
    atomic_json(destination/'bootstrap/SELECTION.json',dict(selection,prototype_source_id=sid,
        context_scope=scope,representation_condition=method))
    atomic_json(destination/'target_repair/CONTACT_CALIBRATION.json',candidate['calibration'])
    carry=read(destination/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json')
    delta=np.asarray(candidate['chart_transform'])
    for contact in carry['contacts'].values():
        if contact['side']=='left':
            for field in ('T_HO','measured_T_HO'):
                if field in contact:contact[field]=np.asarray(contact[field])@np.linalg.inv(delta)
    atomic_json(destination/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json',carry)
    atomic_json(receipt,dict(source_id=sid,method=method,signature=signature,
        candidate_id=candidate['candidate_id'],source_relation=candidate['source_relation'],
        chart_transform=delta,source_pose_or_trajectory_copied_from_other_episode=False,
        dependencies=[record(out/'source_phase'/sid/'PHASE_RECORD.json'),record(out/'target_repair/CONTACT_CALIBRATION.json')]))
    return destination


def summarize_failure(context,sid,connection=None):
    path=context/'prototype'/sid/'morphology_acquisition_v4/PLAN_RESULT.json'
    if path.exists():
        for phase in read(path)['phases']:
            if not phase['admissible']:return dict(cause=phase.get('causal_failure','NO_VALID_CANDIDATE'),phase=phase['phase'],evidence=record(path))
    if connection is not None:
        rows=read(connection/'RESULT.json')['results']
        if not rows:return dict(cause='NO_VALID_CANDIDATE',phase='HANDOFF',evidence=record(connection/'RESULT.json'))
        deepest=max(rows,key=lambda r:r['completed_phases'])
        detail=connection/deepest['subdirectory']/'PHASE_IK.json'
        for phase in read(detail)['phases']:
            if not phase['admissible']:return dict(cause=phase.get('causal_failure','NO_CONNECTING_PATH'),phase=phase['phase'],evidence=record(detail))
        return dict(cause='NO_COMPLETE_CHAIN',phase='HANDOFF_RELEASE',evidence=record(detail))
    return dict(cause='NO_COMPLETE_CHAIN',phase='UNKNOWN')


def attempt(out,sid,method,scope='ARCHITECTURE_REPAIR'):
    from .abc_contract import require
    require(out,scope)
    if method not in ('B_INDEPENDENT','C_COUPLED'):raise ValueError('Interaction chain cannot receive wrist baseline')
    phase=read(out/'source_phase'/sid/'PHASE_RECORD.json')
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json')
    from .practical_parameters import parameters as practical_parameters
    candidates=acquisition_bank(phase,cal,object_dimensions(out),practical_parameters(out))
    signature=cache_key(out,sid,'complete_candidate_chain',dict(method=method,scope=scope,grasp_budget=len(candidates)))
    folder=out/'planning'/sid/method/signature[:12];receipt=folder/'RESULT.json'
    if receipt.exists():
        value=read(receipt)
        if all(record(a['path'])==a for a in value.get('artifacts',[])):
            value['receipt']=str(receipt);return value
        raise ValueError('Changed completed chain artifact; preserve and invalidate dependencies')
    from .acquisition_plan import build as acquisition
    from .handoff_repair import fit_region
    from .full_task_plan import build as connect
    from .physical_attempt import export_full,validate_full
    # Best-first source-preservation ordering; failure at any downstream phase
    # returns here instead of permanently committing the earliest grasp.
    ranked=sorted(candidates,key=lambda c:(c['score']['source_deviation'],c['candidate_id']))
    attempts=[];selected=None;complete=[]
    for rank,candidate in enumerate(ranked):
        golden=read(out/'COVERAGE8.json')['golden'] if (out/'COVERAGE8.json').exists() else read(out/'bootstrap/SELECTION.json')['prototype_source_id']
        atomic_json(out/'WORK_PROGRESS.json',dict(stage='golden_regression' if sid==golden else 'coverage8_planning',
            source_id=sid,method=method,substage='complete-chain branch '+str(rank+1),completed_work=rank,remaining_work=len(ranked)-rank))
        context=branch_context(out,sid,method,signature,candidate,scope)
        preserve_incomplete(context,out,sid)
        row=dict(candidate_id=candidate['candidate_id'],rank=rank,context=str(context),source_score=candidate['score'])
        result=acquisition(context,context/'target_repair/CONTACT_CALIBRATION.json',candidate['goal_overrides'])
        candidate['IK_result']=record(context/'prototype'/sid/'morphology_acquisition_v4/PLAN_RESULT.json')
        candidate['planner_result']=result['status']
        if result['status']!='ACQUISITION_PLAN_BUILT_PENDING_FINAL_VALIDATION':
            row.update(complete=False,first_failure=summarize_failure(context,sid));attempts.append(row)
            atomic_json(folder/'BRANCH_PROGRESS.json',attempts);continue
        fit_region(context,'translation',context/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json','calibrated_gravity','receiver_acquisition_intent')
        connection=connect(context,giver_release_policy='middle_first',receiver_departure=True,
            loaded_geometry=context/'target_repair/LOADED_CARRY_GEOMETRY_f83d2e29156f.json',enable_coupling=method=='C_COUPLED')
        row['connection']=str(connection)
        if read(connection/'RESULT.json')['status']!='FULL_PATH_BUILT':
            row.update(complete=False,first_failure=summarize_failure(context,sid,connection));attempts.append(row)
            atomic_json(folder/'BRANCH_PROGRESS.json',attempts);continue
        plan=connection/'physical_plan'
        if not (plan/'PLAN.json').exists():plan=export_full(context,connection)
        validation=validate_full(context,plan)
        row.update(complete=validation['status']=='VALID',plan=str(plan),post_retime_validation=record(plan/'FINAL_COMMAND_VALIDATION.json'))
        if not row['complete']:
            row['first_failure']=dict(cause='RETIMING_FAIL',phase='POST_RETIME_GEOMETRY');attempts.append(row)
            atomic_json(folder/'BRANCH_PROGRESS.json',attempts);continue
        early=read(context/'prototype'/sid/'morphology_acquisition_v4/PLAN_RESULT.json')
        early_scores=[p['selected_path_quality']['score'] for p in early['phases'] if p.get('selected_path_quality')]
        late=read(connection/'RESULT.json')['results'][0]
        source_score=float(candidate['score']['source_deviation'])
        row['chain_quality']=.25*source_score/(1.+source_score)+.75*float(np.mean([*early_scores,late['complete_chain_quality']]))
        complete.append(row);attempts.append(row)
        if len(complete)>=2:break
    if complete:selected=min(complete,key=lambda r:(r['chain_quality'],r['rank']))
    atomic_json(folder/'GRASP_CANDIDATES.json',candidates)
    artifacts=[record(folder/'GRASP_CANDIDATES.json')]
    if selected:
        plan=Path(selected['plan']);artifacts += [record(plan/n) for n in ('COMMANDS.npz','PLAN.json','FINAL_COMMAND_VALIDATION.json','SOURCE_SCENE.json')]
    value=dict(source_id=sid,method_key=method,full_task_plan=selected is not None,physical_run=False,
        terminal='PLAN_VALID_PENDING_PHYSICS' if selected else 'NO_COMPLETE_CHAIN',
        first_failure=None if selected else attempts[-1]['first_failure'],
        generated_grasp_candidates=len(candidates),attempted_grasp_chains=len(attempts),attempts=attempts,
        selected_candidate_id=selected['candidate_id'] if selected else None,
        selected_chain_score=dict(source=selected['source_score'],complete_quality=selected['chain_quality']) if selected else None,
        selection_policy='Bounded source-ordered backtracking; retain up to2 complete grasp chains and2 complete handoff chains; rank endpoint/path quality before final commitment; no physical success oracle',
        context=selected['context'] if selected else attempts[-1]['context'],plan=selected['plan'] if selected else None,artifacts=artifacts)
    atomic_json(receipt,value);value['receipt']=str(receipt);return value


def coverage(out,golden_only=False):
    fixed=read(out/'COVERAGE8.json');ids=[fixed['golden']] if golden_only else fixed['source_ids'];rows=[]
    for sid in ids:
        for method in ('B_INDEPENDENT','C_COUPLED'):
            rows.append(attempt(out,sid,method))
            atomic_json(out/('GOLDEN_PLANNING.json' if golden_only else 'COVERAGE8_PLANNING.json'),dict(rows=rows,complete=sum(r['full_task_plan'] for r in rows)))
    path=out/('GOLDEN_PLANNING.json' if golden_only else 'COVERAGE8_PLANNING.json')
    if golden_only and not all(r['full_task_plan'] for r in rows):raise AssertionError('Golden common-pipeline complete chain missing')
    if not golden_only and any(sum(r['full_task_plan'] and r['method_key']==m for r in rows)<3 for m in ('B_INDEPENDENT','C_COUPLED')):
        raise AssertionError('Fewer than3 non-Golden complete plans per method; aggregate causes and repair general code')
    return [path,*[Path(r['receipt']) for r in rows if 'receipt' in r]]
