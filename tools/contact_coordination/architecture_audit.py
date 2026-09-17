"""Executable inventory of the preserved pre-repair implementation/evidence."""
import ast
import csv
import io
import json
from pathlib import Path
import numpy as np
from .io import atomic_json, atomic_text, read, record


def csv_file(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def run(out):
    snapshot = out/'before/tools/contact_coordination'
    old = Path(read(out/'ARCHITECTURE_REPAIR.json')['old_run'])
    # Resolve the actual imported entrypoints in a reproducible syntax inventory.
    graph = []
    for path in sorted(snapshot.glob('*.py')):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                graph.append(dict(file=path.name, function=node.name, line=node.lineno,
                    calls=[dict(line=c.lineno, expression=ast.unparse(c.func))
                           for c in ast.walk(node) if isinstance(c, ast.Call)],
                    imports=[ast.unparse(c) for c in ast.walk(node)
                             if isinstance(c, (ast.Import, ast.ImportFrom))]))
    atomic_json(out/'CURRENT_CALL_GRAPH.json', graph)
    evidence = []
    for path in sorted((old/'calibration').rglob('PLAN_RESULT.json')):
        if path.parent.name != 'morphology_acquisition_v4':
            continue
        result = read(path)
        parts = path.parts
        condition = parts[parts.index('contexts')+2]
        method = {'A':'A_WRIST','B':'C_COUPLED','B_NO_COUPLING':'B_INDEPENDENT'}[condition]
        for phase in result.get('phases', []):
            candidates = phase.get('candidates', [])
            evidence.append(dict(method=method, source_id=path.parent.parent.name,
                phase=phase['phase'], task_space_targets=1,
                endpoint_attempts=len(candidates),
                IK_valid=sum(bool(c.get('pose_prior_within_tolerance',c.get('goal_satisfied'))) for c in candidates),
                geometry_valid='NOT_LOGGED_SEPARATELY_FROM_CHORD',
                chord_connectable=sum(bool(c.get('admissible')) for c in candidates),
                planner_connectable=0, selected=phase.get('selected_seed'),
                score='least-squares endpoint residual; first admissible phase; no chain score',
                evidence=str(path)))
    for path in sorted((old/'calibration').rglob('coupling_*.json')):
        if path.stem not in ('coupling_True','coupling_False'): continue
        rows = read(path)
        if not isinstance(rows,list): continue
        source = next((p for p in path.parts if p.startswith('GoPark_')), 'UNKNOWN')
        # Two IK seeds for one contact relation are not two task candidates.
        # Count the actual numerical contact targets, including any collapse
        # across named proposal families, independently of realized q values.
        target_count = len({np.asarray(r['contacts']['right']['T_HO'],float).round(10).tobytes() for r in rows})
        evidence.append(dict(method='C_COUPLED' if path.stem.endswith('True') else 'B_INDEPENDENT',
            source_id=source, phase='HANDOFF', task_space_targets=target_count,
            endpoint_attempts=len(rows), IK_valid='NO_HARD_ENDPOINT_IK_GATE_IN_FIT',
            geometry_valid=sum(not r.get('forbidden') for r in rows),
            chord_connectable='NOT_CHECKED_AT_FIT', planner_connectable=0,
            selected='DEFERRED_TO_FIRST_ADMISSIBLE_CHAIN',
            score='source position/yaw + gravity + table region + seed posture + optional cross residual',
            evidence=str(path)))
    for path in sorted((old/'calibration').rglob('PHASE_IK.json')):
        result=read(path);parts=path.parts
        condition=parts[parts.index('contexts')+2]
        method={'A':'A_WRIST','B':'C_COUPLED','B_NO_COUPLING':'B_INDEPENDENT'}[condition]
        source=parts[parts.index('contexts')+1]
        names=('LEFT_CARRY','RECEIVER_APPROACH','RECEIVER_DEPARTURE','RIGHT_TRANSPORT','PLACE')
        searches={2:'receiver_departure_search',3:'transport_region_search',4:'placement_region_search'}
        for index,phase in enumerate(result.get('phases',[])):
            search=result.get(searches.get(index,'')) or {}
            attempts=search.get('attempts',[])
            targets={json.dumps(a['goal']['wrist_pose_world'],sort_keys=True) for a in attempts if a.get('goal',{}).get('wrist_pose_world')}
            endpoints=[c for a in attempts for p in a.get('result',{}).get('phases',[]) for c in p.get('candidates',[])] if attempts else phase.get('candidates',[])
            evidence.append(dict(method=method,source_id=source,phase=names[index] if index<len(names) else phase['phase'],
                task_space_targets=len(targets) if targets else 1,endpoint_attempts=len(endpoints),
                IK_valid=sum(bool(c.get('pose_prior_within_tolerance',c.get('goal_satisfied'))) for c in endpoints),
                geometry_valid='NOT_LOGGED_SEPARATELY_FROM_CHORD',
                chord_connectable=sum(bool(c.get('admissible')) for c in endpoints),planner_connectable=0,
                selected=search.get('selected',phase.get('selected_seed')),
                score='Source/task-region priority then sampled chord feasibility; no obstacle-search cost',evidence=str(path)))
    if evidence: csv_file(out/'CURRENT_CANDIDATE_EVIDENCE.csv', evidence)
    transitions = [
        ('NATURAL','PREGRASP','acquisition_plan.py:38-66; planner.py:38','common natural q0 from INITIAL','one calibrated object-relative pregrasp; intermediate hover',1,'OPEN_FREE/OPEN_ACQUIRE'),
        ('PREGRASP','LEFT_ACQUISITION','acquisition_plan.py:38-66','previous selected pregrasp IK','one calibrated acquisition wrist target',1,'OPEN_ACQUIRE'),
        ('LEFT_ACQUISITION','LIFT','acquisition_plan.py:40-66','acquisition IK; stationary close/hold','one acquisition target + 63mm world up',1,'LEFT_HOLD'),
        ('LIFT','LEFT_CARRY','full_task_plan.py:123-200,279-303','last HOLD_ELEVATED prefix command','left arm of ranked handoff fit', '21 contact proposals x2 IK fits; actual attempted branches in evidence CSV','LEFT_HOLD'),
        ('LEFT_CARRY','RECEIVER_APPROACH','full_task_plan.py:151-200,279-303','previous left-carry IK','right arm of ranked fit + up to6 predefined detours','21 contact proposals x2 IK fits; actual attempted branches in evidence CSV','RECEIVE_OPEN'),
        ('RECEIVER_APPROACH','DUAL_SUPPORT','physical_attempt.py:37; contact_transition_geometry.py:36','receiver endpoint','same arm q; nominal receiver finger closing',1,'DUAL'),
        ('DUAL_SUPPORT','GIVER_RELEASE','full_task_plan.py:449-469; phase_clock_runtime.py:10-32','same handoff arm q','same arm q; simultaneous/thumb/middle-first opening',1,'DUAL_TO_RIGHT'),
        ('GIVER_RELEASE','RIGHT_OWNERSHIP','phase_clock_runtime.py:86-115; physical_attempt.py:39','stationary arms or post-departure endpoint','observed contact counter; no separate arm goal',1,'RIGHT_HOLD'),
        ('RIGHT_OWNERSHIP','RECEIVER_DEPARTURE','loaded_contact_geometry.py:82; full_task_plan.py:202-207,304-320','handoff q (departure precedes ownership verification in export)','six shoulder/lateral/up half/full-object-extent goals',6,'RIGHT_HOLD'),
        ('RECEIVER_DEPARTURE','RIGHT_TRANSPORT','full_task_plan.py:327-346','selected departure q','bin-overhead q; fallback yaw/height bank','up to8 conditional fallback targets; actual generated counts in evidence CSV','RIGHT_HOLD'),
        ('RIGHT_TRANSPORT','PLACE','full_task_plan.py:348-446','selected transport q','center insertion + pitch/XY/height fallbacks','up to54 initial region goals + refinements','RIGHT_HOLD'),
        ('PLACE','RELEASE','physical_attempt.py:44-49; full_task_plan.py:262-277','place endpoint; then retreat IK','finger opening at fixed q; retreat +85mm and optional delayed opening',1,'RELEASE'),
    ]
    rows = []
    for method in ('A_WRIST','B_INDEPENDENT','C_COUPLED'):
        for a,b,location,start,goal,count,mode in transitions:
            local = b in ('DUAL_SUPPORT','GIVER_RELEASE','RIGHT_OWNERSHIP','RELEASE')
            if method == 'A_WRIST' and b in ('PREGRASP','LEFT_ACQUISITION','LIFT','LEFT_CARRY','RECEIVER_APPROACH','RIGHT_TRANSPORT','PLACE'):
                goal='registered functional source wrist event sample/overlap mean; hard_pose_constraint=False'
                count=1
                location += '; wrist_reference.py:40,63'
            rows.append(dict(method=method,transition=a+' -> '+b,
                classification='STATIC_ENDPOINT' if local else 'DIRECT_JOINT_INTERPOLATION',
                fallback='nominal finger primitive' if local else 'TEMPORAL_IK:6 Cartesian IK waypoints or predefined IK detours; no obstacle search',
                file_function_line=location+'; function build/export_full/step',q_start=start,q_goal=goal,
                multiple_task_space_candidates=str(count)!= '1',actual_candidate_count=count,
                observed_counts='CURRENT_CANDIDATE_EVIDENCE.csv; unevaluated phases have no measured count',
                intermediate_state_collision_checks='YES finite samples (except stationary arm ownership counter)',
                edge_collision_checks='DISCRETE_ONLY <=0.02rad arm spacing or finger/frame sweep; no continuous guarantee',
                table_collision='YES authored hull checker',self_collision='PARTIAL excludes same-hand wrist/digit pairs and non-arm body pairs',
                opposite_arm_hand_collision='YES',bin_collision='YES authored runtime bin150 hulls',
                carried_object='YES LEFT_HOLD/RECEIVE_OPEN and loaded RIGHT_HOLD; partial during contact/release',
                contact_mode=mode,
                no_plan_cause='conversion_attempt.attempt: acquisition or full connection failure collapsed; abc_calibration_worker.run: SIGALRM at900s; no RRT budget exists'))
    csv_file(out/'CURRENT_PHASE_CONNECTION_AUDIT.csv', rows)
    text = '''# Current converter architecture audit — preserved pre-repair implementation

This audit describes the code saved in `before/`, not a design diagram. `CURRENT_CALL_GRAPH.json` records actual function calls/imports and line numbers. `CURRENT_PHASE_CONNECTION_AUDIT.csv` covers all 12 requested transitions for all three methods; `CURRENT_CANDIDATE_EVIDENCE.csv` counts retained interrupted-run attempts. A count of zero planner-connectable candidates means no search planner was invoked, not a proof of infeasibility.

Active call chain: `abc_calibration_worker.run -> source_contract.run -> conversion_attempt.attempt -> episode_context.prepare`. A calls `wrist_reference.acquisition/full_task`; B/C call `acquisition_plan.build -> handoff_repair.fit_region -> full_task_plan.build`. Export follows `physical_attempt.export_full -> validate_full`. Current calibration physical recording uses `full_attempt.prepare/launch -> full_attempt_physics.instrument -> full_attempt_runtime.FullAttemptRuntime -> phase_clock_runtime -> DirectPhysicalDex3ExecutionLayer -> PhysX`. `abc_score.score` delegates measured scoring; `full_attempt_replay.render` replays measured states.

**Real collision-aware global planner: NO.** Repository search found no RRT/OMPL or equivalent general obstacle-search connector. The named `planner.realize_phase_goals` is endpoint least-squares IK with three seeds, optional collision-residual refinement, a direct joint chord validator, six Cartesian continuation IK waypoints, and a small predefined prepose bank. `passive_hand_clearance.connect` also chains IK chords. Neither searches joint-space connectivity around obstacles. Finite edge sampling exists and should be reused, with explicit finer motion resolution and limits.

**Effective multiplicity:** acquisition/pregrasp/lift each have one task-space pose, normally three IK seeds (occasionally a fourth collision refinement or one preferred q). Changing an IK seed is not a new task-space candidate. Acquisition hover has one alternative orientation only after failure. Handoff `fit_region` has a genuine local receiver contact bank from `translation_bank`, two optimization seeds per proposal, but separate B/C solves. The unused `handoff_repair.run` and `targets.selection_costs` are not evidence that the active fit implements shared candidate selection. A's contact variants keep identical wrist pose targets. Departure generates six targets. Transport generates up to eight fallback pose candidates; some height entries coincide when handoff is below nominal overhead. Placement constructs up to54 initial fallbacks plus later refinements, only after failure; it returns the first admissible region entry. RIGHT_OWNERSHIP is a contact counter, not a selected spatial endpoint. Saved executed counts are recorded separately and must not be inferred for phases never reached.

**B/C:** `fit_region` removes six cross residual entries for B, but still applies the same <=3mm/0.05rad pair consistency gate to B and C. Each branch optimizes a 14D q from the same seed families; B's objective is separable but both unary seed indices are tied together and no independent left/right bank selection exists. C has explicit cross residuals, but its endpoint bank differs from B's. Full path selection breaks at the first admissible ranked handoff chain; the already chosen acquisition is never reconsidered. This is not complete-chain backtracking over grasp/lift/handoff/place alternatives.

**Geometry:** `runtime_hulls.make_model/Checker.check` loads authored/cooked convex hulls for both arms, hands, body, table, bin and rounded carried object; `loaded_contact_geometry.check` checks raw robot geometry and calibrated loaded object geometry. Same-hand adjacent assembly is filtered, as are non-arm-only body pairs. Joint limits are bounded in IK, with loaded offsets checked in the loaded helper. Arm chords sample max joint increments0.02rad; no swept-volume proof exists. Receiver closing and giver release are sampled finger sweeps. Some contact sweeps omit object-environment checks. Accepted endpoints alone are insufficient.

**Retiming:** `quintic_retime` follows selected IK knots/chords and uses analytic stop-to-stop velocity/acceleration bounds. It does not search or repair geometry. `validate_full` checks command-frame states, not independently validated edges between every retimed frame, and omits finger velocity/acceleration certification.

**Failure and video:** `conversion_attempt.attempt` conflates NO_IK, geometry rejection and disconnected chains into NO_PLAN_WITHIN_FIXED_BUDGET/TRAIN_NO_PLAN_WITHIN_BOUNDED_POLICY. A separate SIGALRM900-second cap does the same. `FullAttemptRuntime.step` records to horizon but OFFICIAL_NOMINAL admission failure returns HOLD_LAST_SERVO_COMMAND, so later arm sequence is hidden. The measured renderer is solid and does not cut at failure; fixing only rendering cannot restore commands never executed. This controller adapter must continue the pre-existing command schedule while keeping failure labels latched.

**A mismatch:** the current baseline samples source wrist events/overlap means rather than tracking the dense source wrist motion. A must retain source wrist/TCP trajectory as its primary spatial reference when the common backend is repaired; it must never receive C's shared-object objective.

All historical evidence is preserved. The interrupted calibration has been explicitly marked ABORTED_FOR_CONVERTER_ARCHITECTURE_REPAIR. No calibration has been resumed.
'''
    atomic_text(out/'CURRENT_CONVERTER_ARCHITECTURE_AUDIT.md', text)
    return [out/name for name in ('CURRENT_CALL_GRAPH.json','CURRENT_PHASE_CONNECTION_AUDIT.csv',
                                  'CURRENT_CANDIDATE_EVIDENCE.csv','CURRENT_CONVERTER_ARCHITECTURE_AUDIT.md')]
