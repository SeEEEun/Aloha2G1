"""Read-only historical/source inventory for the TRAIN40-calibrated study."""
import csv
import io
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import ROOT, read, record, atomic_json, atomic_text
from .abc_contract import METHODS


def csv_write(path, rows):
    stream = io.StringIO();writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);atomic_text(path, stream.getvalue())


def inspect(out):
    old = Path(read(out/'ABC_STUDY.json')['historical_run'])
    split = read(out/'SPLIT_CONTRACT.json');entries = read(out/'bootstrap/SPLITS.json')['entries']
    train = split['authorized_training_source_ids'];dev = split['evaluation_source_ids']
    assert len(train) == len(set(train)) == 40 and len(dev) == len(set(dev)) == 35
    assert not set(train) & set(dev)
    raw = []
    for entry in entries:
        r = record(entry['source_parquet'])
        assert r['sha256'] == entry['source_parquet_sha256']
        raw.append(r)
    aliases = dict(schema='immutable_abc_method_aliases_v1', methods=METHODS,
        historical_run=record(old/'TRAIN40_conversion/LEDGER.json'),
        historical_aliases={'A':'A_WRIST','B':'C_COUPLED','B_NO_COUPLING':'B_INDEPENDENT'},
        historical_B_is_not_independent=True,
        implementation_before=[r for r in read(out/'bootstrap/DEPENDENCIES_BEFORE.json')
            if Path(r['path']).name in ('conversion_attempt.py','handoff_repair.py','full_task_plan.py','wrist_reference.py')])
    atomic_json(out/'METHOD_ALIAS_MANIFEST.json', aliases)
    milestone = read(old/'FIRST_SOURCE_CONDITIONED_FULL_TASK.json')
    for field in ('trace','score','source_phase','plan','video'):
        assert record(milestone[field]['path']) == milestone[field]
    score = read(milestone['score']['path'])
    assert score['physical_validity'] and score['stages']['FULL_TASK']
    ledger = read(old/'TRAIN40_conversion/LEDGER.json');assert ledger['completed'] == len(ledger['rows']) == 80
    inventory = []
    for r in ledger['rows']:
        for f in r['artifacts']:
            assert record(f['path']) == f
        context = Path(r['context']);causal = r['first_failure'] or 'NONE';layer = 'NONE';detail = ''
        plan = context/'prototype'/r['source_id']/'morphology_acquisition_v4/PLAN_RESULT.json'
        phase = next((p for p in read(plan).get('phases',[]) if not p['admissible']), None) if plan.exists() else None
        if phase:
            causal = phase.get('phase', phase.get('name','UNKNOWN'))
            candidates = phase.get('candidates',[])
            contacts = [h for c in candidates for h in c.get('connection_validation',{}).get('forbidden_contacts',[])]
            layer = 'contact_mode_geometry_or_connecting_path' if contacts else 'IK_or_target_construction'
            if contacts:detail = ' / '.join(contacts[0].get('bodies',[]))
        elif r.get('physical_run'):
            layer = 'physical_validity' if r.get('physical_validity') is False else 'physical_task' if not r.get('task_success') else 'NONE'
        elif causal == 'POST_RETIME_GEOMETRY':layer = 'retiming_geometry'
        elif causal != 'NONE':layer = 'full_chain_connection_or_candidate_selection'
        inventory.append(dict(source_id=r['source_id'],historical_method=r['condition'],
            canonical_method=aliases['historical_aliases'][r['condition']],terminal=r['terminal'],
            outer_first_failure=r['first_failure'],earliest_saved_phase=causal,causal_layer=layer,
            first_geometry_pair=detail,physical_run=r.get('physical_run',False),
            physical_validity=r.get('physical_validity'),full_task=r.get('task_success'),
            context=str(context),phase_evidence=str(plan) if plan.exists() else 'MISSING',
            subsequent_symptoms='NOT_RECLASSIFIED_AS_EARLIEST_CAUSE'))
    csv_write(out/'HISTORICAL_TRAIN40_FIRST_FAILURES.csv', inventory)
    atomic_json(out/'bootstrap/VERIFIED_HISTORY.json',dict(raw_records=raw,
        train_ids=train,dev_ids=dev,overlap=[],exact_signal_duplicate_groups=split['exact_signal_duplicate_groups'],
        golden_source=milestone['source_id'],golden_evidence=milestone,
        historical_counts={m:dict(Counter(r['terminal'] for r in inventory if r['canonical_method']==m)) for m in ('A_WRIST','C_COUPLED')},
        independent_TRAIN40_outcomes='NOT_PREVIOUSLY_MEASURED'))
    atomic_text(out/'HISTORICAL_EXPOSURE.md', '# Historical exposure\n\n'
        'All TRAIN40 sources and the DEV35 development scenes have been inspected previously. '
        'Golden GoPark_20260820_152058 supplied repeated target-robot controller/contact/load development evidence. '
        'Historical coupled B is current C; no full Golden trajectory copy was found by the preserved episode audit. '
        'The prior final TRAIN40 run includes 40 Wrist and 40 coupled attempts, with 10 coupled physical traces and 6 valid full tasks. '
        'The separate historical paired-10 reference ablation also exposed DEV outcomes. No new DEV outcomes may select parameters.\n\n'
        'Folds are development selection partitions. Neither grouping nor leave-Golden-out scoring removes inherited code, calibration or historical inspection. '
        'No legacy score will be inserted into new repaired-backend fold results. New outcome-driven design requires a versioned record.\n')
    reuse = [
        ('source FK/identities/events','source_phase.py','load_source / phase_record','Preserved raw parquet and corrected-reference hashes; source fields retain inference/unknown status'),
        ('task registration / functional TCP','source_contract.py','registered_functional_wrists / run','One common task transform; same tool poses with fixed transform applied once'),
        ('receiving interaction','receiving_relation.py','build_region','Source object-relative receiving relation enters local morphology chart'),
        ('acquisition','acquisition_plan.py','build','Existing phase IK and complete-hand edge validation'),
        ('handoff selection','handoff_repair.py','fit_region / coupling_residual','Explicit cross factors on/off; same unary terms and candidate bank'),
        ('connection / retiming','full_task_plan.py','build','Retained knots, contact release/egress and whole-chain checks'),
        ('IK / timing','planner.py','realize_phase_goals / quintic_retime','Authoritative named model and joint bounds'),
        ('runtime geometry','runtime_hulls.py','Checker','Existing authored/cooked hull bundle; runtime filters unchanged'),
        ('loaded carry','loaded_contact_geometry.py','check / object_pose','Local load-state prediction; not a universal mechanical transform'),
        ('hand control','phase_clock_runtime.py','PhaseClockDex3','Preserve contact candidate, release order and right ownership'),
        ('dynamic simulator/logger','phase_physics.py','instrument','Existing PhysX wrapper; add separately versioned full recording adapter'),
        ('independent scoring','score_hybrid.py','score','Measured stage thresholds; new failure latch must prevent later rescue'),
        ('measured rendering','paper_replays.py','existing replay functions','Read measured q/object states; new videos cannot use target-only stand-ins'),
        ('observations/alignment','demonstration_observation.py','MeasuredDemonstrationRuntime','G1 pre-action RGB/current state/executed command'),
        ('ACT implementation','act_training.py','prepare / train','Installed lerobot-train; canonical B/C adapter needed; no legacy checkpoint substitution'),
    ]
    lines=['# Reuse map','','| Component | Exact module / function | Compatibility evidence / minimal action |','|---|---|---|']
    for c,f,fn,e in reuse:lines.append(f'| {c} | `tools/contact_coordination/{f}` : `{fn}` | {e} |')
    atomic_text(out/'REUSE_MAP.md','\n'.join(lines)+'\n')
    atomic_text(out/'REFERENCE_ADOPTION.md', '''# Reference adoption and implementation traceability

These are limited conceptual adaptations, not reproductions of the cited systems.

| Reference | Borrowed principle | Local implementation / evidence | Adaptation and exclusions |
|---|---|---|---|
| [Mayr, Ahmad, Chatzilygeroudis, Nardi and Krueger, 2022, arXiv:2203.10033](https://arxiv.org/abs/2203.10033) | Tune a small set of skill parameters in simulation | `abc_contract.require`; parameter inventory and bounded selection stages; empirical search remains gated until recording/scene checks pass | Existing converter parameters only. No SkiREIL/SkiROS, PDDL, KUKA, RL or Bayesian-optimization implementation claimed. |
| [Calinon, 2016, Intelligent Service Robotics 9(1):1–29, DOI 10.1007/s11370-015-0187-9](https://calinon.ch/paper4018.htm) | Represent task relations in appropriate frames and instantiate them per episode | `source_contract.registered_functional_wrists`, `receiving_relation.build_region`; environment and source-sensitivity regression | Deterministic rigid transforms and local contact chart. No TP-GMM/TP-GMR implementation claimed. |
| [Englert and Toussaint, IJRR 37(1), 2018, DOI 10.1177/0278364917743795](https://journals.sagepub.com/doi/10.1177/0278364917743795) | Separate analytic constraints from interaction outcomes requiring trials | Existing `planner.realize_phase_goals`, `runtime_hulls.Checker`, independent `score_hybrid.score` | Geometry/IK are not physical grasp success. No inverse-optimal-control or constrained Bayesian-optimization framework ported. Publisher lists online publication on 5 December 2017; issue citation is 2018. |
| [Cawley and Talbot, JMLR 11(70):2079–2107, 2010](https://jmlr.org/papers/v11/cawley10a.html) | Disclose selection bias and distinguish selection from later fixed evaluation | `abc_audit.folds`, `abc_contract.require`, HISTORICAL_EXPOSURE.md | Grouped development folds, finite recipes, frozen downstream gates. CV does not erase historical exposure or provide an untouched test. |

Bibliographic title/author/year and the stated principles were checked against the linked primary pages. Proposed downstream stages are not described as measured implementations until their receipts exist.
''')
    return dict(status='HISTORY_AND_IDENTITIES_VERIFIED',TRAIN40=40,DEV35=35,overlap=0,
                historical_A_attempts=40,historical_C_attempts=40,new_trials=0)


def environment(out):
    old=Path(read(out/'ABC_STUDY.json')['historical_run']);split=read(out/'SPLIT_CONTRACT.json')
    entries={r['source_recording_id']:r for r in read(out/'bootstrap/SPLITS.json')['entries']}
    physical={r['source_id']:r for r in read(old/'TRAIN40_conversion/LEDGER.json')['rows'] if r.get('physical_run')}
    rows=[];evidence=[]
    for subset,key in [('TRAIN40','authorized_training_source_ids'),('DEV35','evaluation_source_ids')]:
        for index,sid in enumerate(split[key]):
            path=out/'source_phase'/sid/'PHASE_RECORD.json';p=read(path)
            x=np.asarray(p['initial_object_pose_world']);source=np.asarray(p['source_task_geometry']['initial_object_pose'])
            t=np.asarray(p['task_registration']['T_target_source']);error=float(np.max(np.abs(t@source-x)))
            before=np.asarray(p['task_registration']['source_object_to_bin_xy_m']);after=np.asarray(p['task_registration']['target_object_to_bin_xy_m'])
            metric_error=abs(float(np.linalg.norm(before)-np.linalg.norm(after)))
            assert error<1e-10 and metric_error<1e-10 and not p['task_registration']['independent_hand_rebasing']
            actual='NOT_MEASURED_NO_SAVED_RUNTIME';trans_error=rot_error=None;runtime_hash=None
            if sid in physical:
                trial_path=Path(physical[sid]['attempt'])/'trial_result.json';trial=read(trial_path)
                init=trial['object_task_frame_registration']['runtime_initial_pose_verification']
                actual=init['actual_position_xyz_m_before_frame_0'];trans_error=init['translation_error_mm'];rot_error=init['rotation_error_deg']
                assert trans_error<=.1 and rot_error<=.1 and trial['object_pose_writes_during_timed_loop']==0
                assert not trial['prohibited_attachment_used'] and not trial['real_robot']
                runtime_hash=record(trial_path);evidence.append(runtime_hash)
            rows.append(dict(split=subset,episode_index=index+1,source_id=sid,
                original_stable_id=entries[sid]['stable_episode_id'],raw_sha256=entries[sid]['source_parquet_sha256'],
                source_object_xyz=source[:3,3].tolist(),source_bin_xy=p['source_task_geometry']['bin_center_xy_m'],
                source_bin_orientation='UNKNOWN',registered_object_xyz=x[:3,3].tolist(),
                registered_object_xyzw=Rotation.from_matrix(x[:3,:3]).as_quat().tolist(),
                registered_bin_xy=p['placement']['bin_center_xy_m'],object_to_bin_distance_m=float(np.linalg.norm(after)),
                task_transform_max_error=error,metric_distance_error_m=metric_error,
                before_command_actual_xyz=actual,runtime_position_error_mm=trans_error,runtime_rotation_error_deg=rot_error,
                runtime_trial_hash=runtime_hash['sha256'] if runtime_hash else 'NOT_MEASURED',
                runtime_reset_process='fresh process per saved trial; new reset/controller-state regression pending',
                object_orientation_status='INFERRED_SOURCE_IMAGE_PLANAR_AXIS; absolute dynamic orientation UNKNOWN',
                source_contacts='UNKNOWN',scene_contract='source metric relation rigidly registered to fixed target bin; target object varies',
                A_B_C_scene_rule='identical episode input / natural q0 / initializer; new nominal reset verification required'))
    csv_write(out/'EPISODE_ENVIRONMENT_AUDIT.csv',rows)
    atomic_json(out/'RUNTIME_SCENE_HASHES.json',dict(historical_runtime_evidence=evidence,
        new_runtime_checks='PENDING',source_phases=[record(out/'source_phase'/r['source_id']/'PHASE_RECORD.json') for r in rows]))
    atomic_text(out/'EPISODE_ENVIRONMENT_AUDIT.md', '# Source and runtime environment audit\n\n'
        '75/75 source manifests satisfy one shared rigid transform and preserve object-to-bin metric separation. '
        'TRAIN40 and DEV35 source identities are disjoint. The scene contract is source-conditioned metric transfer into the fixed target bin/G1 layout: '
        'the bin is shared; each object location and all source wrist priors use the same episode task transform. This is not a world-coordinate relabeling that leaves the robot relation unchanged.\n\n'
        'Source bin orientation is UNKNOWN. Planar object orientation is an inferred image axis, not a measured full dynamic orientation or symmetry certificate. '
        'The rigid rounded plush/ball surrogate uses the unchanged qualified geometry/mass/contact model and 150 mm bin. No soft-body claim.\n\n'
        '10 historical TRAIN physical trials have explicit requested-versus-actual object readback before frame zero; all pass the existing 0.1 mm/0.1 degree checks. '
        'Absent runtime states for other source episodes are NOT MEASURED, not fabricated. New nominal controls must additionally log reset/readback and clean controller state. '
        'Fresh simulator processes isolate trials. Metadata audit alone does not certify every source has a valid reachable grasp.\n')
    return dict(status='SOURCE_SCENES_AUDITED_RUNTIME_REGRESSION_PENDING',source_scenes=75,historical_runtime_readbacks=10)


def folds(out):
    split=read(out/'SPLIT_CONTRACT.json');ids=split['authorized_training_source_ids']
    groups=[set(g) for g in split['exact_signal_duplicate_groups']]
    if groups:raise RuntimeError('Grouped duplicate schema needs explicit inventory before fold assignment')
    # Geometry-neutral round-robin through the immutable source manifest.
    # Source membership, not observed task outcome, determines folds.
    value=dict(schema='TRAIN40_development_folds_v1',source_manifest=record(out/'SPLIT_CONTRACT.json'),
        rule='ordered TRAIN40 source index modulo 5; one verified unique original recording per group',
        folds=[dict(fold=i,held_out=ids[i::5],fitting=[s for j,s in enumerate(ids) if j%5!=i]) for i in range(5)],
        proposal_and_screening_fold_order=[0,1,2,3,4],recipes_max=4,selection_attempt_ceiling=224,
        historical_exposure_disclosed=True,unbiased_test_claim=False,
        fit_rule='Any data-derived statistics/modes must be fit only on the listed fitting sources; fixed mechanical calibration is separately disclosed',
        selection='Equal source/equal B,C full-task mean; then weaker-method full success, handoff/ownership, plan coverage, lower deviation/compute/complexity',
        screening='Four recipes on folds0,1; retain current comparator and best challenger; both on remaining three; final comparison only complete five-fold records')
    path=out/'TRAIN40_FOLDS.json'
    if path.exists() and read(path)!=value:raise RuntimeError('Immutable folds changed')
    atomic_json(path,value);return dict(status='FOLDS_PREDECLARED',fold_sizes=[8]*5,new_outcomes=0)


def parameters_inventory(out):
    from .calibration_parameters import DEFAULT,BOUNDS,RECIPES
    cal=read(out/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json')
    rows=[]
    def add(name,file,needle,units,classification,usage,origin,conditions,tunable=False,bounds='FIXED'):
        path=Path(file);path=path if path.is_absolute() else ROOT/path
        matches=[i+1 for i,line in enumerate(path.read_text().splitlines()) if needle in line]
        rows.append(dict(name=name,file=str(path),line=matches[0] if matches else 'JSON_FIELD',units=units,
            classification=classification,actual_usage=usage,originating_evidence=origin,
            state_contact_conditions=conditions,tunable=tunable,allowed_range=bounds,file_sha256=record(path)['sha256']))
    for name,c in cal['contacts'].items():
        add('contacts.'+name+'.T_HO',out/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json','"'+name+'"',
            'm / SO(3)','STATE_DEPENDENT_CALIBRATION','Object-in-functional-hand relation used by wrist_target or loaded carry prediction',
            str(c.get('control',c.get('evidence',cal.get('control','See nested contact lineage')))),
            'Only '+str(c.get('stage',name))+' with recorded fingers/load; not a universal rigid mechanical frame')
    add('T_wrist_H','tools/contact_coordination/morphology_repair.py','tools[side]=','m / SO(3)',
        'STATE_DEPENDENT_CALIBRATION','Fixed computational hand frame defined at qualified HOLD morphology; applied exactly once',
        'Named measured grasp geometry in extract_calibration','Frame convention subsequently fixed; digit contact surfaces still move')
    add('receiver_capture_transition',out/'target_repair/CONTACT_CALIBRATION_CURRENT_CARRY_265626bb3e67.json','receiver_capture_transition','m / SO(3)',
        'STATE_DEPENDENT_CALIBRATION','Prediction of receiving-induced object displacement during overlap',
        str(cal.get('receiver_capture_transition',{})),'Qualified receiver approach/closing and giver support; not a robot-fixed 12 mm transform')
    add('receiver_separation_span',out/'target_repair/CONTACT_SEPARATION_CANDIDATES.json','maximum_offset_m','m',
        'OBJECT_OR_CONTROLLER_CALIBRATION','Outer displacement radius and separating-direction prior',
        'Measured geometry overlap 1.497 mm plus fixed 1.5 mm contact offset; outer radius 2.997 mm',
        'Current object/hand state; search may shrink sampling inside existing radius, never expand it')
    add('loaded_arm_bias',out/'target_repair/LOADED_CARRY_GEOMETRY_f83d2e29156f.json','arm_measured_minus_command_rad','rad',
        'STATE_DEPENDENT_CALIBRATION','Predict full loaded arm/waist geometry in right carry',
        str(read(out/'target_repair/LOADED_CARRY_GEOMETRY_f83d2e29156f.json').get('trace')),
        'Local quasistatic right-carry calibration; raw measurements and physical scorer remain independent')
    add('giver_release_policy','tools/contact_coordination/conversion_attempt.py',"giver_release_policy='middle_first'",'discrete',
        'OBJECT_OR_CONTROLLER_CALIBRATION','Common progressive release order','Preserved TRAIN controller development',
        'Receiver candidate while giver supports; release before right-only verification')
    add('receiver_departure_family','tools/contact_coordination/loaded_contact_geometry.py','directions=[','unit vectors / m',
        'GENERAL_ALGORITHM_PARAMETER' if False else 'SEARCH_HYPERPARAMETER',
        'Directions recomputed from current geometry/shoulder, with half/full object short extent',
        'Existing modeled hand/object/shoulder geometry','No fixed Golden world departure vector')
    locations={
        'acquisition_posture_multiplier':('acquisition_plan.py','acquisition','Existing acquisition joint_prior_scale .001'),
        'connection_posture_multiplier':('full_task_plan.py','connection','Existing connection joint_prior_scale .001'),
        'handoff_posture_multiplier':('handoff_repair.py','handoff','Existing common posture length .1 m/rad; previous TRAIN range .01 to .1'),
        'receiver_spread_fraction':('handoff_repair.py','receiver','Shrink samples inside the fixed calibrated 2.997 mm displacement domain'),
    }
    for name,(file,phase,evidence) in locations.items():
        add(name,'tools/contact_coordination/'+file,'scaled_ik_config' if phase in ('acquisition','connection') else 'tuning',
            'dimensionless multiplier','SEARCH_HYPERPARAMETER',evidence,
            'Pre-outcome bounded recipe proposal using existing default and historical TRAIN numerical scales',
            'Identical per-source B/C application; source goal centers and outer safety domain unchanged',True,str(BOUNDS[name]))
    csv_write(out/'GOLDEN_PARAMETER_CLASSIFICATION.csv',rows)
    search=dict(schema='four_scalar_existing_converter_search_v1',values={name:dict(default=DEFAULT[name],
        range=list(BOUNDS[name]),units='dimensionless multiplier',consumer=locations[name][0],
        meaning=locations[name][2],proposal_evidence='Existing default and conservative multiplicative neighborhood; no new outcome was read to propose these values') for name in DEFAULT},
        recipes=RECIPES,recipe_order=list(RECIPES),proposal_frozen_before_new_calibration_outcomes=True,
        max_recipes=4,attempt_ceiling=224,shared_objective='0.5*(mean full_task_B + mean full_task_C)',
        normalization='All knobs multiply existing residual scales. Position residuals use metres, orientation uses rotation-vector radians, and q uses radians. Acquisition and connection have disjoint phase blocks (100/m position, 1/rad rotation, .001/rad posture at default). Handoff uses 1/m position and .1/rad posture at default. A common scaling of a whole residual vector is immaterial. No duplicate coefficient multiplies the same residual block.',
        data_fitting='No new data-derived feature normalization or modes are introduced in this version. Each fold uses the same disclosed fixed historical embodiment calibration. All episode priors are instantiated from that episode; no all-TRAIN mean task is fitted.',
        modes=1,outer_contact_domain='Existing calibration/geometry span; receiver_spread_fraction can only shrink samples within it',
        non_tunable=['scene registration','joint/numerical/geometry validity','physics','mass/friction','bin/object size','source semantics','release strategy','capture displacement','loaded-state correction','seed set','900-second total planning cap','retry policy','coupling strength'],
        A_opportunity='Same acquisition/connection multipliers and common backend; no contact-pair optimizer. No new A-specific sweep has started.',
        implementation=record(ROOT/'tools/contact_coordination/calibration_parameters.py'))
    path=out/'PARAMETER_SEARCH_SPACE.json'
    if path.exists() and read(path)!=search:raise RuntimeError('Immutable recipe proposal changed')
    atomic_json(path,search)
    atomic_json(out/'CALIBRATION_PARAMETERS.json',dict(recipe='CURRENT',values=DEFAULT))
    atomic_text(out/'GOLDEN_INFLUENCE_AUDIT.md', '# Golden influence inventory\n\n'
        'The preserved audit found no Golden full-path copying. Current source scene, acquisition target position, receiving chart and handoff prior are instantiated per source. '
        'Golden-influenced contact transforms, capture transition, posture seeds and loaded-arm/waist bias remain explicitly inherited calibration. Their limits are state-, load-, finger- and object-dependent. '
        'They are not classified as universal mechanical constants, and are not averaged across unrelated episodes.\n\n'
        'The current acquisition object-relative contact family is fixed target-embodiment calibration. Its world goal uses each source initial object pose. '
        'The source left relation also supplies the inferred carry/handoff prior; exact parallel-gripper axes are not equated to Dex3 axes. '
        'Source contact patches and dynamic object poses remain inferred/unknown, so semantic-preservation claims are limited to the documented task roles and supported relation model.\n\n'
        'Leave-Golden-out fold results will exclude its new trial score from selection in that fold, but cannot remove the historical trace influence in these fixed assets. '
        'No clean unseen-Golden claim is possible. No leave-out statistic has yet been measured. '
        'Current versus selected recipe comparisons must use the same repaired backend and recording rules, not archived Golden scores.\n')
    atomic_text(out/'ABC_METHOD_PARITY.md', '# Controlled A/B/C contract\n\n'
        'A_WRIST uses the calibrated source wrist/TCP spatial prior. B_INDEPENDENT and C_COUPLED use identical source inputs, receiving chart, contact samples, unary residuals, seeds, region, planner, geometry, retiming and controller. '
        'The existing `handoff_repair.coupling_residual` adds predicted shared-object position/rotation disagreement only for C. '
        'B still undergoes complete robot/object collision and overlap safety validation; that validator does not optimize a new C solution. '
        'The existing `fit_region` computes both factor settings and saves separately; every consumed solve counts against the same 900-second instance cap. No free Golden result or cache transfer is an output.\n\n'
        'Historical B/Ours is C. The old coupling-disabled branch is B provenance, not a relabeling of coupled results. '
        'Acquisition and connection weights act on separate phase blocks; they do not duplicate the handoff posture residual. '
        'Geometry, source-semantic validity, physical outcomes and full task scores remain separate. '
        'The small independent/coupled difference has not yet been evaluated under this new protocol.\n')
    return dict(status='FOUR_RECIPES_PREDECLARED',parameters=4,modes=1,new_calibration_outcomes=0)


def run(out,stage,resume=False):
    if stage=='inspect':return inspect(out)
    if stage=='environment_golden_audit':return environment(out)
    if stage=='folds':return folds(out)
    if stage=='parameter_inventory':return parameters_inventory(out)
    raise RuntimeError('ABC_STAGE_NOT_IMPLEMENTED_OR_QUALIFIED: '+stage)
