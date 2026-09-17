"""Measured paper tables/figures; denominators include all40 scheduled sources."""
from pathlib import Path
import shutil
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text
from .practical_planning import csv_file
from .practical_finalize import verify_freeze

METRICS=('VALID_TARGET','COMPLETE_PLAN','PHYSICS_EXECUTED','GRASP','LIFT','HANDOFF','RIGHT_OWNERSHIP','TRANSPORT','PLACE','FULL_TASK')
FAILURES=('NO_IK','NO_VALID_CANDIDATE','NO_CONNECTING_PATH','NO_COMPLETE_CHAIN','RETIMING_FAIL','PREGRASP_OBJECT_CONTACT',
    'PHYSICS_GRASP_FAIL','PHYSICS_HANDOFF_FAIL','PHYSICS_OWNERSHIP_FAIL','COLLISION_FAIL','PLACE_FAIL','NUMERICAL_ABORT','PHYSICS_FAILURE')
LABELS={'VALID_TARGET':'Valid Target','COMPLETE_PLAN':'Complete Plan','PHYSICS_EXECUTED':'Physics Executed','GRASP':'Grasp','LIFT':'Lift',
    'HANDOFF':'Handoff','RIGHT_OWNERSHIP':'Ownership','TRANSPORT':'Transport','PLACE':'Place','FULL_TASK':'FULL TASK'}


def episode(row):
    p=row['physical'];stages=p.get('stages',{});plan=row['planning'];failure=p.get('first_failure')
    if isinstance(failure,dict):failure=failure.get('cause',failure.get('causal_label','NO_COMPLETE_CHAIN'))
    if p.get('status')=='NUMERICAL_ABORT':failure='NUMERICAL_ABORT'
    detail=p.get('first_failure_detail') or {}
    if 'COLLISION' in detail.get('reason',''):failure='COLLISION_FAIL'
    if failure=='PHYSICS_PLACE_FAIL':failure='PLACE_FAIL'
    values=dict(source_id=row['source_id'],paper_method=row['paper_method'],internal_method=row['internal_method'],
        VALID_TARGET=bool(plan['summary']['VALID_TARGET']),COMPLETE_PLAN=bool(p['complete_plan']),PHYSICS_EXECUTED=bool(p['physics_executed']),
        **{k:bool(stages.get(k,False)) for k in ('GRASP','LIFT','HANDOFF','RIGHT_OWNERSHIP','TRANSPORT','FULL_TASK')},
        PLACE=bool(stages.get('BIN_ENTRY') and stages.get('BIN_SETTLE')),first_failure=failure,
        first_failure_detail=detail if detail else plan['summary']['first_causal_failure'],
        physical_validity=p.get('physical_validity'),planning_seconds=plan['planning_seconds'],source_deviation_m2=plan['summary']['source_deviation_m2'],
        trace=p.get('trace'),command_horizon_verified=p.get('command_horizon_verified',False))
    return values


def metrics(out):
    verify_freeze(out);ids=read(out/'PRACTICAL_STUDY.json')['train_source_ids'];episodes=[];aggregates={};dataset_manifests=[]
    for code in ('A','B'):
        report=read(out/('PAPER_'+code+'_RESULTS.json'));assert report['status']=='PASS' and len(report['rows'])==40
        assert [r['source_id'] for r in report['rows']]==ids
        values=[episode(r) for r in report['rows']];episodes+=values
        counts={key:sum(r[key] for r in values) for key in METRICS}
        counts.update(TOTAL=40,failure_counts={k:sum(r['first_failure']==k for r in values) for k in FAILURES},
            conditional_full_task_rate=counts['FULL_TASK']/counts['PHYSICS_EXECUTED'] if counts['PHYSICS_EXECUTED'] else None)
        aggregates[code]=counts
        # Actual frozen-method outputs only; no reconstructed diagnostic enters
        # either dataset. Explicit failure manifests retain all scheduled IDs.
        base=ROOT/'datasets'/('PAPER_A_WRIST_TRAIN40' if code=='A' else 'PAPER_B_INTERACTION_TRAIN40')/out.name
        for raw,value in zip(report['rows'],values,strict=True):
            folder=base/raw['source_id'];folder.mkdir(parents=True,exist_ok=True)
            plan=raw['planning']['plan'];physical=raw['physical'];copied=[]
            if plan['full_task_plan']:
                for name in ('COMMANDS.npz','PLAN.json','SOURCE_SCENE.json','FINAL_COMMAND_VALIDATION.json','CONTACT_SELECTION.json'):
                    src=Path(plan['plan'])/name
                    if src.exists():
                        target=folder/name
                        if not target.exists():shutil.copy2(src,target)
                        assert record(target)['sha256']==record(src)['sha256'];copied.append(record(target))
            if physical['physics_executed']:
                for name in ('event_log.npz','ROBOT_OBJECT_CONTACTS.json','PREGRASP_OBJECT_PROTECTION.json','ABC_NOMINAL_SCORE.json','FULL_ATTEMPT_RECORDING.json'):
                    src=Path(physical['folder'])/name
                    if src.exists():
                        target=folder/name
                        if not target.exists():shutil.copy2(src,target)
                        assert record(target)['sha256']==record(src)['sha256'];copied.append(record(target))
            source_phase=out/'FINAL_FROZEN/source_phase'/raw['source_id']/'PHASE_RECORD.json'
            from .practical_dataset import export_evidence
            packaged=export_evidence(raw,folder,source_phase.parent)
            atomic_json(folder/'EPISODE.json',dict(result=value,source_events=record(source_phase),selected_plan=plan,
                artifacts=copied,freeze=record(out/'FINAL_TRAIN40_FREEZE_MANIFEST.json'),
                packaged_planning_evidence=record(packaged),
                actual_measured_physics=physical['physics_executed'],diagnostic_reconstruction=False,
                measured_arrays='event_log.npz: MEASURED_Q, EXECUTED_COMMAND, object poses/velocities, contact forces, nominal/runtime phases, timestamps',
                RGB_observations_available=False,RGB_note='Full video is a measured-state replay; no recorded simulator RGB observation array is claimed',
                eligible_supervision=bool(physical.get('physical_validity') and physical.get('command_horizon_verified'))))
        manifest=base/'DATASET_MANIFEST.json';atomic_json(manifest,dict(paper_method='PAPER_'+code,source_ids=ids,
            episode_manifests=[record(base/sid/'EPISODE.json') for sid in ids],freeze=record(out/'FINAL_TRAIN40_FREEZE_MANIFEST.json'),diagnostics_included=False))
        dataset_manifests.append(manifest)
    path=out/'PAPER_AB_METRICS.json';atomic_json(path,dict(status='PASS',methods=aggregates,episodes=episodes,
        denominator='All40 scheduled source episodes per method; unexecuted and no-plan rows remain in denominator',
        valid_target_definition='At least one task-satisfying IK endpoint among the visited acquisition-phase candidates; geometry-valid endpoint counts are recorded separately in planning summaries',
        place_definition='Measured BIN_ENTRY and BIN_SETTLE both true',conditional_metric='FULL_TASK / PHYSICS_EXECUTED',
        full_task_difference_percentage_points=100*(aggregates['B']['FULL_TASK']-aggregates['A']['FULL_TASK'])/40.,
        dataset_manifests=[record(p) for p in dataset_manifests]))
    per=out/'PER_EPISODE_PAPER_AB_RESULTS.csv';csv_file(per,episodes)
    table=[dict(Metric=LABELS[k],A_Wrist_Centric=f'{aggregates["A"][k]} / 40',B_Interaction_Centric=f'{aggregates["B"][k]} / 40') for k in METRICS]
    table.append(dict(Metric='FULL TASK / EXECUTED (conditional)',A_Wrist_Centric=f'{aggregates["A"]["FULL_TASK"]} / {aggregates["A"]["PHYSICS_EXECUTED"]}',B_Interaction_Centric=f'{aggregates["B"]["FULL_TASK"]} / {aggregates["B"]["PHYSICS_EXECUTED"]}'))
    csv_path=out/'TABLE_PAPER_AB_TRAIN40_RESULTS.csv';csv_file(csv_path,table)
    md=out/'TABLE_PAPER_AB_TRAIN40_RESULTS.md';text='| Metric | A Wrist-Centric | B Interaction-Centric |\n|---|---:|---:|\n'
    for r in table:text+=f'| {r["Metric"]} | {r["A_Wrist_Centric"]} | {r["B_Interaction_Centric"]} |\n'
    text+='\nPAPER_A=A_WRIST; PAPER_B=C_COUPLED. B_INDEPENDENT is the internal coupling ablation. These are TRAIN calibration/evaluation results, not held-out generalization estimates.\n'
    atomic_text(md,text)
    definitions=out/'PAPER_AB_METRIC_DEFINITIONS.md'
    atomic_text(definitions,'# Paper A/B metric definitions\n\n'
        'PAPER_A=A_WRIST; PAPER_B=C_COUPLED. Both methods use all40 scheduled TRAIN identities as the primary denominator.\n\n'
        '- VALID TARGET: at least one task-satisfying IK endpoint among the actually visited acquisition-phase candidates. The planning summaries separately report admissible endpoint counts as VALID_CANDIDATES.\n'
        '- COMPLETE PLAN: a complete command sequence whose whole-chain geometric and post-retiming validation passed.\n'
        '- PHYSICS EXECUTED: an official complete command was launched and an actual measured trace exists. A genuine numerical abort remains an executed failure with its true shorter recorded horizon.\n'
        '- GRASP, LIFT, HANDOFF, OWNERSHIP and TRANSPORT: the unchanged common ABC scorer flags, including its physical-validity requirements.\n'
        '- PLACE: both measured BIN_ENTRY and BIN_SETTLE are true.\n'
        '- FULL TASK: the frozen full-task scorer flag; the primary rate divides by40. FULL TASK / PHYSICS EXECUTED is separately labelled conditional and is undefined when no trial executed.\n'
        '- Failure-category counts use each episode\'s first recorded causal failure. Complete-plan failures and physical failures remain distinct.\n'
        '- No-plan video status cards provide source coverage for visual review; they do not count as executed physics. Diagnostic reconstructed commands are excluded.\n\n'
        'These are TRAIN-calibrated single-task results. No DEV35 or ACT evaluation is included.\n')
    return [path,per,csv_path,md,definitions,*dataset_manifests]


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    value=read(out/'PAPER_AB_METRICS.json');methods=value['methods'];keys=['COMPLETE_PLAN','GRASP','LIFT','HANDOFF','RIGHT_OWNERSHIP','TRANSPORT','PLACE','FULL_TASK']
    labels=['PLAN','GRASP','LIFT','HANDOFF','OWNERSHIP','TRANSPORT','PLACE','FULL TASK'];colors=['#65788c','#158f89'];artifacts=[]
    fig,ax=plt.subplots(figsize=(11,4.3));x=np.arange(len(keys))
    for code,offset,color,title in [('A',-.18,colors[0],'(a) Baseline: Wrist-Centric Retargeting'),('B',.18,colors[1],'(b) Ours: Interaction-Centric Retargeting')]:
        y=[methods[code][k]/40*100 for k in keys];bars=ax.bar(x+offset,y,.36,color=color,label=title)
        ax.bar_label(bars,labels=[f'{methods[code][k]}/40' for k in keys],fontsize=8,padding=3)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,110),ylabel='Success Rate (%)');ax.set_yticks(np.arange(0,101,20));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper center',fontsize=9,frameon=False)
    fig.tight_layout()
    for extension in ('png','pdf','svg'):
        path=out/('FIG_PAPER_AB_STAGE_SUCCESS.'+extension);fig.savefig(path,dpi=300,bbox_inches='tight');artifacts.append(path)
    plt.close(fig)
    stages=['TOTAL','VALID_TARGET','COMPLETE_PLAN','PHYSICS_EXECUTED','FULL_TASK'];names=['Scheduled sources','Valid target','Complete plan','PhysX executed','Full task']
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
    for ax,code,color in zip(axes,['A','B'],colors,strict=True):
        y=np.arange(len(stages));bars=ax.barh(y,[methods[code][k] for k in stages],color=color)
        ax.bar_label(bars,padding=3);ax.set(yticks=y,yticklabels=names,xlim=(0,44),xlabel='Episodes /40',title='(a) Baseline: Wrist-Centric Retargeting');ax.set_title('(a) Baseline: Wrist-Centric Retargeting' if code=='A' else '(b) Ours: Interaction-Centric Retargeting',fontsize=10);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.tight_layout();path=out/'FIG_PAPER_AB_PIPELINE_COVERAGE.png';fig.savefig(path,dpi=300,bbox_inches='tight');plt.close(fig);artifacts.append(path)
    ids=read(out/'PRACTICAL_STUDY.json')['train_source_ids'];lookup={(r['source_id'],r['paper_method']):r for r in value['episodes']};columns=['COMPLETE_PLAN','GRASP','HANDOFF','FULL_TASK']
    matrix=np.array([[int(lookup[(sid,'PAPER_'+code)][key]) for code in ('A','B') for key in columns] for sid in ids])
    fig,ax=plt.subplots(figsize=(8,13));ax.imshow(matrix,aspect='auto',cmap=ListedColormap(['#e5e8eb','#188d86']),vmin=0,vmax=1)
    ax.set(xticks=range(8),xticklabels=['A PLAN','A GRASP','A HANDOFF','A FULL','B PLAN','B GRASP','B HANDOFF','B FULL'],yticks=range(40),yticklabels=ids)
    ax.tick_params(axis='x',rotation=45,labelsize=9);ax.tick_params(axis='y',labelsize=8);ax.axvline(3.5,color='white',lw=3)
    ax.set_title('TRAIN40 per-episode outcomes | green: success; gray: no success',pad=14);fig.tight_layout();path=out/'FIG_PAPER_AB_EPISODE_MATRIX.png';fig.savefig(path,dpi=300,bbox_inches='tight');plt.close(fig);artifacts.append(path)
    manifest=out/'PAPER_AB_FIGURE_MANIFEST.json';atomic_json(manifest,dict(figures=[record(p) for p in artifacts],metrics=record(out/'PAPER_AB_METRICS.json'),unmeasured_results_invented=False))
    return [manifest,*artifacts]


def final_report(out):
    verify_freeze(out);videos=read(out/'TRAIN40_AB_VIDEO_VERIFICATION.json');review=read(out/'TRAIN40_VISUAL_INSPECTION.json')
    assert review['pass'] and sorted(review['videos'],key=lambda r:r['path'])==sorted([r['video'] for r in videos['videos']],key=lambda r:r['path'])
    assert videos['status']=='AUTOMATED_PASS_VISUAL_PENDING' or videos['status']=='PASS'
    acceptance=out/'TRAIN40_AB_VIDEO_ACCEPTANCE.json'
    atomic_json(acceptance,dict(status='PASS',automated_verification=record(out/'TRAIN40_AB_VIDEO_VERIFICATION.json'),
        visual_inspection=record(out/'TRAIN40_VISUAL_INSPECTION.json'),individual_A=40,individual_B=40))
    value=read(out/'PAPER_AB_METRICS.json');ranking=read(out/'RECIPE_RANKING.json');config=read(out/'TRAIN40_FINAL_SHARED_CONFIG.json')
    from .practical_dataset import verify_dataset
    datasets=[]
    for item in value['dataset_manifests']:
        assert record(item['path'])==item
        datasets.append(verify_dataset(item['path']))
    dataset_receipt=out/'FINAL_DATASET_PAYLOAD_VERIFICATION.json'
    atomic_json(dataset_receipt,dict(status='PASS',datasets=datasets))
    text='============================================================\nTRAIN40 PRACTICAL CALIBRATION + PAPER A/B\n============================================================\n\nPLANNING SWEEP\n'
    for r in sorted(ranking['ranked'],key=lambda r:r['complexity']):text+=f'{r["recipe"]}: B {r["complete"]["B_INDEPENDENT"]}/40; C {r["complete"]["C_COUPLED"]}/40\n'
    text+='Selected recipe: '+config['recipe']+'\n\nPHYSICAL VERIFICATION\n'
    rows=read(out/'TOP_RECIPE_PHYSICS.json')['rows']
    for method in ('B_INDEPENDENT','C_COUPLED'):
        selected=[r for r in rows if r['recipe']==config['recipe'] and r['method_key']==method]
        text+=method+f' full task: {sum(r.get("stages",{}).get("FULL_TASK",False) for r in selected)} / {len(selected)} scheduled\n'
    text+='\nFINAL FROZEN CONFIG: '+str(out/'TRAIN40_FINAL_SHARED_CONFIG.json')+'\n'
    for code,title in [('A','WRIST'),('B','INTERACTION')]:
        r=value['methods'][code];text+=f'\nPAPER {code} — {title}\nPlan: {r["COMPLETE_PLAN"]}/40\nGrasp: {r["GRASP"]}/40\nHandoff: {r["HANDOFF"]}/40\nFull task: {r["FULL_TASK"]}/40\nFull task /executed (conditional): {r["FULL_TASK"]}/{r["PHYSICS_EXECUTED"]}\n'
    text+='\nA → B FULL TASK DIFFERENCE: '+str(value['full_task_difference_percentage_points'])+' percentage points\n'
    text+='\nTABLE: '+str(out/'TABLE_PAPER_AB_TRAIN40_RESULTS.md')+'\nFIGURES: '+str(out/'PAPER_AB_FIGURE_MANIFEST.json')+'\nVIDEOS: '+str(out/'videos')+'\nDEV35 STARTED: NO\nACT STARTED: NO\n============================================================\n'
    path=out/'FINAL_TERMINAL_SUMMARY.txt';atomic_text(path,text);print(text,flush=True)
    report=out/'TRAIN40_PRACTICAL_CALIBRATION_PAPER_AB_REPORT.md';atomic_text(report,'# TRAIN40 practical calibration and paper A/B\n\n```text\n'+text+'```\n\n'
        'All40 scheduled identities per method are included. Physical task failures retain their full recorded horizons; no-plan videos are clearly labelled status cards. '
        'No reconstructed diagnostic contributes to measured results or converted datasets. This is a TRAIN-calibrated single-task comparison, not a held-out generalization claim. '
        'The repaired representation and common planner/controller/physics architecture remain unchanged apart from the explicitly authorized scalar exposure and bounded incidental-contact policy.\n')
    done=out/'TRAIN40_STUDY_COMPLETE.json';atomic_json(done,dict(complete=True,metrics=record(out/'PAPER_AB_METRICS.json'),freeze=record(out/'FINAL_TRAIN40_FREEZE_MANIFEST.json'),
        video_verification=record(acceptance),visual_review=record(out/'TRAIN40_VISUAL_INSPECTION.json'),DEV35_started=False,ACT_started=False))
    return [path,report,done,acceptance,dataset_receipt,out/'TRAIN40_VISUAL_INSPECTION.json']
