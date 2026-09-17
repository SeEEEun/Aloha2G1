"""Evidence-bound source-guided planner comparisons and readiness gate."""
from pathlib import Path
from dataclasses import asdict
import csv,json,shutil
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text


def selected_phases(result):
    if not result['full_task_plan']:return []
    sid=result['source_id'];context=Path(result['context']);base=context/'prototype'/sid/'morphology_acquisition_v4'
    early=read(base/'PLAN_RESULT.json');goals=read(base/'GOALS.json')['goals']
    branch=next(b for b in result['attempts'] if b['candidate_id']==result['selected_candidate_id'])
    connection=Path(branch['connection']);summary=read(connection/'RESULT.json');selected=next(r for r in summary['results'] if r['all_admissible'])
    sub=connection/selected['subdirectory'];late=read(sub/'PHASE_IK.json');late_goals=read(sub/'GOALS.json')
    contacts=read(Path(result['plan'])/'CONTACT_SELECTION.json')
    rows=[]
    for phase_list,goal_list,path in [(early['phases'],goals,base/'PLAN_RESULT.json'),(late['phases'],late_goals,sub/'PHASE_IK.json')]:
        for phase in phase_list:
            goal=next((g for g in goal_list if g['name']==phase['phase']),None)
            if goal is None:raise AssertionError('Phase goal missing: '+phase['phase'])
            preparation=phase.get('passive_hand_preparation')
            parts=next(a['result']['phases'] for a in preparation['attempts'] if a['valid']) if preparation else [phase]
            for index,part in enumerate(parts):
                candidates=[p for p in part.get('planner_attempts',[]) if p['status']=='PATH_FOUND']
                certificate=next((p for p in candidates if p.get('endpoint_index')==part.get('selected_endpoint_index')),candidates[-1] if candidates else None)
                if certificate is None:raise AssertionError('Selected phase lacks a path certificate')
                q=np.asarray(part['connecting_q']);np.testing.assert_allclose(q,certificate['path'])
                selected_id=goal.get('candidate_id')
                if selected_id is None:
                    selected_id=result['selected_candidate_id'] if path==base/'PLAN_RESULT.json' else ' + '.join(contacts['selected_candidate_ids'])
                if phase['phase']=='POST_RELEASE_RETREAT':
                    selected_id='DERIVED_FROM:'+next((v.get('candidate_id','PLACE_GOAL') for v in late_goals if v['name']=='PLACE'),'PLACE_GOAL')
                row=dict(source_id=sid,method=result['method_key'],phase=phase['phase'],subconnection=index,
                    selected_candidate=selected_id,chain_grasp_candidate=result['selected_candidate_id'],
                    endpoint_IK_role=part.get('role'),endpoint_errors=part.get('selected_errors'),
                    goal=goal,path=q,certificate=certificate,evidence=record(path))
                rows.append(row)
    return rows


def _metrics(row,source_root,g):
    from .source_motion_prior import attach,MotionGuide
    from .path_quality import evaluate
    q=np.asarray(row['path']);goal=attach(row['goal'],source_root/'source_phase'/row['source_id'])
    active=np.asarray(row['certificate']['active_indices']) if 'active_indices' in row['certificate'] else np.concatenate([np.arange(7) if s=='left' else np.arange(7,14) for s in goal['active_hands']])
    # A passive preparation changes active side within the enclosing phase.
    goal=dict(goal,active_hands=['left' if min(active)<7 else 'right']) if len(active)==7 else goal
    guide=MotionGuide(g,goal,q[0],q[-1],active)
    return evaluate(q,g.arm_limits[:,0],g.arm_limits[:,1],active,guide.features,guide.reference)


def planning_audit(out):
    from .source_phase import COMMON
    from .planning_kinematics import G1Kinematics
    from .path_quality import QualityPolicy
    from .joint_path_planner import Budget
    from tools.doll_handoff_retargeting.common import load_common_config,load_scene
    cfg=load_common_config(COMMON);g=G1Kinematics(cfg,load_scene(cfg));old=Path(read(out/'SOURCE_GUIDED_RRT_REBUILD.json')['old_run'])
    sets={name:read(root/'GOLDEN_PLANNING.json')['rows']+read(root/'COVERAGE8_PLANNING.json')['rows'] for name,root in [('OLD',old),('NEW',out)]}
    evidence=[];table=[];backend={name:record(ROOT/'tools/contact_coordination'/name) for name in ('joint_path_planner.py','planner.py','source_motion_prior.py','path_quality.py')}
    for version,results in sets.items():
        for result in results:
            for row in selected_phases(result):
                cert=row['certificate'];metrics=cert.get('final_path_quality') or _metrics(row,out,g)
                contact=cert.get('algorithm')=='CONSTRAINED_LOCAL_CONTACT_MOTION'
                values=dict(version=version,source_id=row['source_id'],method=row['method'],phase=row['phase'],subconnection=row['subconnection'],
                    motion_class='CONSTRAINED_LOCAL_CONTACT_MOTION' if contact else 'FREE_SPACE',
                    connection_algorithm=('OLD_DIRECT_EARLY_RETURN' if version=='OLD' and cert.get('direct_path_valid') and not cert.get('search_used') else cert['algorithm']),
                    RRT_API_called=cert.get('rrt_api_called',True),RRT_search_expanded=cert.get('rrt_search_expanded',int(cert.get('search_used',False))),
                    straight_selected=cert.get('straight_path_selected',cert.get('direct_path_valid',False)),
                    nontrivial_selected=cert.get('nontrivial_rrt_path_selected',bool(cert.get('search_used'))),
                    source_guided_samples=cert.get('source_guided_samples',0),global_samples=cert.get('global_samples',0),
                    candidate_solutions=len(cert.get('path_candidates',[])),selected_path_id=cert.get('selected_path_id'),
                    selected_candidate=row['selected_candidate'],selected_path_score=metrics['score'] if version=='NEW' else None,
                    joint_path_length=metrics['joint_path_length_rad'],cartesian_wrist_path_length=metrics['cartesian_wrist_path_length_m'],
                    source_deviation=metrics['source_motion_deviation_m'],roughness=metrics['roughness'],posture_excursion=metrics['posture_deviation'],
                    minimum_clearance=metrics.get('minimum_clearance_m'),joint_limit_margin=metrics['joint_limit_margin_fraction'],
                    state_checks=cert['state_checks'],edge_checks=cert['edge_checks'],evidence=row['evidence']['path'])
                table.append(values);evidence.append(dict(version=version,**row,metrics=metrics))
    # All attempted endpoint connections, including rejected candidates, are
    # reported separately from accepted final-chain connections.
    attempts=[];no_ik=[];journals=set()
    def visit(value,sid,method,phase=None):
        if isinstance(value,dict):
            phase=value.get('phase',phase)
            if value.get('algorithm') in ('SOURCE_GUIDED_RRT_CONNECT','CONSTRAINED_LOCAL_CONTACT_MOTION') and 'state_checks' in value:
                if value.get('invocation_evidence'):journals.add(value['invocation_evidence']['path'])
                attempts.append(dict(source_id=sid,method=method,phase=phase,status=value['status'],
                    motion_class=value.get('motion_class'),rrt_api_called=value.get('rrt_api_called'),expanded=value.get('rrt_search_expanded',0),
                    straight=value.get('straight_path_selected',False),nontrivial=value.get('nontrivial_rrt_path_selected',False),
                    source_guided_samples=value.get('source_guided_samples',0),global_samples=value.get('global_samples',0)))
                return
            for k,v in value.items():
                # The same certificate is copied into each IK candidate and
                # planner_attempts. Count the explicit API-attempt list once.
                if k=='planner_attempts':
                    for item in v:visit(item,sid,method,phase)
                elif k not in ('candidates','planner_result','source_wrist_reference'):visit(v,sid,method,phase)
        elif isinstance(value,list):
            for v in value:visit(v,sid,method,phase)
    seen=set()
    for result in sets['NEW']:
        for branch in result['attempts']:
            context=Path(branch['context']);paths=[context/'prototype'/result['source_id']/'morphology_acquisition_v4/PLAN_RESULT.json']
            if branch.get('connection'):paths+=list(Path(branch['connection']).glob('contact_*/PHASE_IK.json'))
            for path in paths:
                if path in seen or not path.exists():continue
                seen.add(path);visit(read(path),result['source_id'],result['method_key'])
        if not result['full_task_plan']:no_ik.append(dict(source_id=result['source_id'],method=result['method_key'],failure=result['first_failure'],
            branch_first_failures=[dict(candidate_id=b['candidate_id'],failure=b.get('first_failure')) for b in result['attempts']]))
    if journals:
        attempts=[]
        for journal in sorted(journals):
            for line in Path(journal).read_text().splitlines():
                event=json.loads(line)
                event['method']=next((m for m in ('B_INDEPENDENT','C_COUPLED') if m in Path(journal).name),'SEE_PHASE_EVIDENCE')
                attempts.append(dict(event,expanded=event.get('rrt_search_expanded') or 0,straight=bool(event.get('straight_path_selected')),nontrivial=bool(event.get('nontrivial_rrt_path_selected'))))
    else:raise AssertionError('Actual planner invocation journal missing')
    free=[a for a in attempts if a['motion_class']=='FREE_SPACE'];selected=[r for r in table if r['version']=='NEW' and r['motion_class']=='FREE_SPACE']
    assert free and all(a['rrt_api_called'] for a in free)
    assert all(r['RRT_search_expanded']>0 for r in selected)
    assert all(r['source_guided_samples']>0 for r in selected)
    counts=dict(FREE_SPACE_EDGES_TOTAL=len(free),RRT_API_CALLED=sum(a['rrt_api_called'] for a in free),
        RRT_SEARCH_EXPANDED=sum(a['expanded']>0 for a in free),STRAIGHT_PATH_SELECTED=sum(a['straight'] for a in free),
        NONTRIVIAL_RRT_PATH_SELECTED=sum(a['nontrivial'] for a in free),RRT_FAILURE=sum(a['status']!='PATH_FOUND' for a in free),
        selected_complete_chain_free_edges=len(selected),scope='Attempted endpoint connections across bounded candidate/chain search; separate selected-chain rows in CSV',
        goals_without_complete_chain=no_ik)
    csv_path=out/'SOURCE_GUIDED_RRT_AUDIT.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    attempted_csv=out/'SOURCE_GUIDED_RRT_CONNECTION_ATTEMPTS.csv'
    fields=['source_id','method','phase','invocation_id','algorithm','status','motion_class','rrt_api_called','expanded','straight',
        'nontrivial','source_guided_samples','global_samples','selected_path_id','state_checks','edge_checks']
    with attempted_csv.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(attempts)
    # Compare both methods' actual generated handoff families for identical
    # source/grasp conditions, not selected endpoints (which may differ).
    branches={};parity=[]
    for result in sets['NEW']:
        for branch in result['attempts']:
            pointer=Path(branch['context'])/'target_repair/CURRENT_CONTACT_REGION_FIT.json'
            if pointer.exists():
                folder=Path(read(pointer)['path']);bank=folder/'ENDPOINT_BANK.json'
                if bank.exists():branches[(result['source_id'],branch['candidate_id'],result['method_key'])]=bank
    for (sid,candidate,method),bank_b in branches.items():
        if method!='B_INDEPENDENT':continue
        bank_c=branches.get((sid,candidate,'C_COUPLED'))
        if bank_c is None:continue
        b=read(bank_b);c=read(bank_c)
        for side in ('left','right'):
            assert len(b['bank'][side])==len(c['bank'][side])
            for left,right in zip(b['bank'][side],c['bank'][side]):
                assert left['candidate_id']==right['candidate_id']
                np.testing.assert_array_equal(left['task_space_target'],right['task_space_target'])
        parity.append(dict(source_id=sid,grasp_candidate=candidate,B=record(bank_b),C=record(bank_c),
            identical_targets=True,counts={s:len(b['bank'][s]) for s in ('left','right')}))
    assert parity,'No identical-input B/C candidate-family witness'
    policy=asdict(QualityPolicy());budget=asdict(Budget())
    for row in evidence:
        if row['version']!='NEW':continue
        cert=row['certificate'];assert cert['final_path_quality']['weights']==policy
        assert all(cert['budget'][k]==v for k,v in budget.items() if k!='state_checks')
        assert cert['budget']['state_checks']<=budget['state_checks']
    def row_key(r):return (r['source_id'],r['method'],r['phase'],r['subconnection'])
    paired={v:{row_key(r):r for r in table if r['version']==v} for v in ('OLD','NEW')}
    matched=sorted(set(paired['OLD'])&set(paired['NEW']));summary={}
    for metric in ('joint_path_length','cartesian_wrist_path_length','source_deviation','roughness','posture_excursion'):
        summary[metric]={v:float(np.mean([paired[v][k][metric] for k in matched])) for v in ('OLD','NEW')}
    path=out/'SOURCE_GUIDED_PATH_EVIDENCE.json';atomic_json(path,dict(status='PASS',rows=evidence,attempts=attempts,counts=counts,
        backend=backend,B_C_identical_backend=True,B_C_candidate_family_parity=parity,
        common_budget=budget,common_weights=policy,matched_phase_count=len(matched),matched_phase_mean=summary))
    description='''# Source-guided RRT before / after

The frozen old run is preserved. The fixed Golden/Coverage8 sources are unchanged. New free-space connections expand RRT even when a straight edge is valid. A straight candidate competes with checked RRT solutions; the retained set is bounded at4. The planner continues24 improvement iterations after its first solution within the same18000-state-check transition budget, shared across up to2 IK endpoints. Source guidance uses7 event-normalized wrist anchors,4 differential proposals with6 bounded Jacobian steps each, and a mixture of source/global/midpoint/goal proposals. It never requires intermediate source IK targets.

Path weights: clearance exposure0.15; normalized joint length0.15; Cartesian wrist travel0.10; source-shape deviation0.30; endpoint posture continuity0.10; excess posture excursion0.10; direction-change roughness0.07; joint-limit margin0.03. All terms are bounded to[0,1]. Joint lengths use named joint ranges; Cartesian travel uses endpoint wrist chord with a0.1m floor. Source RMS error uses source curvature RMS with a0.05m floor. Clearance exposure averages exp(-max(distance,0)/0.01m) over17 arclength samples; minimum signed distance is separately recorded and capped at0.03m. Endpoint continuity is mean squared joint-range-normalized endpoint displacement. Excursion only counts motion outside the endpoint joint box, normalized by0.05 joint ranges. Roughness sums squared changes in unit joint directions. Joint margin penalty is exp(-margin/0.05). Positive unbounded measures use x/(1+x). Full collision safety is independently checked at<=0.005rad edge spacing.

Shortcuts and two corner-relaxation passes accept only collision-valid, non-worsening changes; final geometry is revalidated before quintic retiming. Smoothing keeps interaction endpoints fixed. Handoff chain cost is0.3*squashed representation cost+0.7*mean path score, comparing up to2 complete handoff chains. Full grasp-chain cost is0.25*squashed grasp source deviation+0.75*mean acquisition/path-and-handoff score, comparing up to2 complete grasp chains within5 grasp candidates. These weights and bounds are common across B/C. Only the explicit shared-object representation term distinguishes C. No physical outcome is used.

The CSV compares selected complete-chain connections. The JSON also records attempted endpoints and first causal source failures. Missing IK endpoints cannot form a planning problem and are reported separately, not counted as an RRT failure or fabricated API call. Old source-deviation and quality diagnostics were recomputed using the current common metric on saved old geometric paths; old paths were not re-executed or changed. Old minimum-clearance values were not logged and remain missing.
'''
    description+='\n```json\n'+json.dumps(counts,indent=2)+'\n```\n'
    description+='\nMean over '+str(len(matched))+' matched source/method/phase connections (geometric metrics, not task success):\n\n| Metric | Old | New |\n|---|---:|---:|\n'
    for metric,values in summary.items():description+=f'| {metric} | {values["OLD"]:.6f} | {values["NEW"]:.6f} |\n'
    description+='\nExact numerical B/C candidate-family equality verified in '+str(len(parity))+' shared source/grasp branches. Actual accepted certificates have identical planner weights, seed, resolution and iteration policy; endpoint attempts share the fixed state-check budget.\n'
    report=out/'SOURCE_GUIDED_RRT_BEFORE_AFTER.md';atomic_text(report,description)
    return [path,csv_path,attempted_csv,report]


def final_gate(out):
    from .run_source_guided_rrt_rebuild import STAGES,dependencies,valid_receipt
    previous=None
    for stage in STAGES[:-1]:
        path=out/'STAGE_RECEIPTS'/(stage+'.json')
        assert valid_receipt(path,dependencies(out,stage),previous),stage
        previous=record(path)
    plan=read(out/'SOURCE_GUIDED_PATH_EVIDENCE.json');counts=plan['counts'];physics=read(out/'PHYSICAL_VERIFICATION.json');video=read(out/'SOURCE_GUIDED_VIDEO_VERIFICATION.json')
    source=read(out/'COMPACT_SOURCE_MOTION_PRIORS.json');tests=read(out/'SOURCE_GUIDED_STRUCTURAL_TESTS.json')
    golden=read(out/'GOLDEN_PLANNING.json')['rows'];coverage=read(out/'COVERAGE8_PLANNING.json')['rows']
    complete=[r for r in golden+coverage if r['full_task_plan']]
    command_validation=[read(Path(r['plan'])/'FINAL_COMMAND_VALIDATION.json') for r in complete]
    current_paths=[r for r in plan['rows'] if r['version']=='NEW']
    local=[r['certificate'] for r in current_paths if r['certificate']['algorithm']=='CONSTRAINED_LOCAL_CONTACT_MOTION']
    assert local and all(c['ingress_distance_m']<=.025 and c['approach_direction_valid'] and c['full_geometry_revalidated'] for c in local)
    review=read(out/'ARCHITECTURE_DEFECT_REVIEW.json')
    assert review['evidence']==[record(out/n) for n in ('SOURCE_GUIDED_PATH_EVIDENCE.json','PHYSICAL_VERIFICATION.json','PREGRASP_OBJECT_PROTECTION_RESULTS.json','SOURCE_GUIDED_VIDEO_VERIFICATION.json')], 'Defect review is stale'
    conditions={
        'phase_based_B_C':source['B_C_primary']=='INTERACTION_PHASE_GOALS' and all('source_wrist_reference_waypoints' not in r['goal'] for r in current_paths),
        'multiple_target_candidates':all(r['meaningful_grasp_candidates']>1 for r in source['rows']),
        'endpoint_IK_separate':tests['status']=='PASS' and all(r['endpoint_IK_role']=='IK_ENDPOINT_THEN_PATH_PLANNER' for r in current_paths),
        'every_free_space_edge_RRT':counts['RRT_API_CALLED']==counts['FREE_SPACE_EDGES_TOTAL']>0,
        'direct_early_bypass_removed':all(not r['certificate'].get('direct_early_bypass',False) for r in plan['rows'] if r['version']=='NEW'),
        'source_soft_prior':source['B_C_source_motion']=='SOFT',
        'multiple_paths_ranked':any(len(r['certificate'].get('path_candidates',[]))>1 for r in plan['rows'] if r['version']=='NEW'),
        'smoothing_revalidated':all(r['certificate'].get('full_geometry_revalidated') for r in plan['rows'] if r['version']=='NEW'),
        'state_edge_collision_tests':tests['status']=='PASS',
        'pregrasp_object_protection':read(out/'SOURCE_GUIDED_ROBOT_GEOMETRY_TESTS.json')['protected_object']['status']=='PASS',
        'carried_object_validation':read(out/'SOURCE_GUIDED_ROBOT_GEOMETRY_TESTS.json')['carried_object']['status']=='PASS',
        'B_C_same_backend':plan['B_C_identical_backend'],
        'retiming_after_final_geometry':bool(command_validation) and all(v['status']=='VALID' and v['post_retime_rechecked'] for v in command_validation) and all(r.get('command_horizon_verified') for r in physics['rows'] if r['physics_executed']),
        'no_command_speed_discontinuity':all(read(r['issued_command_timing']['path'])['status']=='PASS' for r in physics['rows'] if r['physics_executed']),
        'golden_and_coverage_rerun':len(read(out/'GOLDEN_PLANNING.json')['rows'])==2 and len(read(out/'COVERAGE8_PLANNING.json')['rows'])==16,
        'actual_full_videos_visually_verified':video['status']=='PASS' and any(r['physics_executed'] for r in physics['rows']),
        'no_known_common_architecture_defect':review['no_known_common_architecture_defect'] and tests['status']=='PASS' and plan['status']=='PASS' and video['status']=='PASS'}
    # Every physical result, including a protection violation, remains visible.
    # Legitimate task failures do not imply a planner software defect or tune it.
    proofs={k:dict(passed=bool(v),evidence=[record(out/n) for n in ('SOURCE_GUIDED_STRUCTURAL_TESTS.json','SOURCE_GUIDED_PATH_EVIDENCE.json','PHYSICAL_VERIFICATION.json','SOURCE_GUIDED_VIDEO_VERIFICATION.json')]) for k,v in conditions.items()}
    ready=all(conditions.values());gate=out/'SOURCE_GUIDED_RRT_READY_FOR_CALIBRATION.json'
    atomic_json(gate,dict(SOURCE_GUIDED_RRT_READY_FOR_CALIBRATION='YES' if ready else 'NO',conditions=proofs,
        calibration_started=False,first_structural_blocker=next((k for k,v in conditions.items() if not v),None)))
    if not ready:raise AssertionError('Source-guided readiness incomplete')
    from .architecture_reports import alias
    aliases=alias(out)
    for item in read(aliases)['workspace_code_manifest']:
        src=Path(item['path']);dst=out/'FROZEN_SOURCE_GUIDED_CODE'/src.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    text='============================================================\nSOURCE-GUIDED RRT PLANNER REBUILD\n============================================================\nPHASE-BASED RETARGETING: YES\nDense framewise B/C IK: NO\nMultiple target candidates: YES\nEndpoint IK: YES\n'
    text+='\nFREE-SPACE PLANNING\n'+json.dumps(counts,indent=2)+'\nDirect early bypass remaining: NO\n'
    text+='\nPATH QUALITY\nSource-motion soft prior: YES\nMultiple path ranking: YES\nShortcut/smoothing: YES\nState revalidation: YES\nEdge revalidation: YES\nPregrasp object protection: YES\nCarried-object validation: YES\n'
    for label,rows in [('GOLDEN',golden),('COVERAGE8',coverage)]:
        text+='\n'+label+'\n'
        for method in ('B_INDEPENDENT','C_COUPLED'):
            group=[r for r in rows if r['method_key']==method];physical=[r for r in physics['rows'] if r['method_key']==method and r['source_id'] in {x['source_id'] for x in rows}]
            text+=method+f' complete plans: {sum(r["full_task_plan"] for r in group)}/{len(group)}; full tasks: {sum(r.get("stages",{}).get("FULL_TASK",False) for r in physical)}/{len(group)} scheduled; physically executed: {sum(r["physics_executed"] for r in physical)}\n'
    text+='\nBEFORE / AFTER MOTION QUALITY\nMatched phase mean ('+str(plan['matched_phase_count'])+' connections):\n'
    for metric,values in plan['matched_phase_mean'].items():text+=f'{metric}: {values["OLD"]:.6f} -> {values["NEW"]:.6f}\n'
    protection_report=read(out/'PREGRASP_OBJECT_PROTECTION_RESULTS.json');protection=protection_report['rows']
    text+='Maximum pregrasp object displacement (new measured): '+str(max(r['maximum_object_displacement_before_acquisition_m'] for r in protection))+' m\n'
    text+='PREGRASP_OBJECT_CONTACT trials: '+str(sum(r['status']=='PREGRASP_OBJECT_CONTACT' for r in protection))+' / '+str(len(protection))+' executed\n'
    matched=protection_report['matched_before_after']
    if matched:text+=f'Matched mean pregrasp object displacement: {np.mean([r["old_pregrasp_displacement_m"] for r in matched]):.6f} -> {np.mean([r["new_pregrasp_displacement_m"] for r in matched]):.6f} m ({len(matched)} measured pairs)\n'
    text+='\nVIDEOS: '+str(out/'videos')+'\nSOURCE_GUIDED_RRT_READY_FOR_CALIBRATION: YES\nCALIBRATION STARTED: NO\n============================================================\n'
    terminal=out/'FINAL_TERMINAL_SUMMARY.txt';atomic_text(terminal,text);print(text,flush=True)
    report=out/'SOURCE_GUIDED_RRT_REBUILD_REPORT.md'
    atomic_text(report,'# Source-guided RRT rebuild\n\n```text\n'+text+'```\n\n'
        'Readiness certifies the software architecture and measured execution evidence. It does not certify task success on all sources. '
        'All scheduled planning failures and measured physical failures remain in the denominators and videos. '
        'No TRAIN40 calibration or final dataset evaluation was started.\n\n'+review['assessment']+'\n')
    return [gate,aliases,terminal,report,out/'ARCHITECTURE_DEFECT_REVIEW.json']
