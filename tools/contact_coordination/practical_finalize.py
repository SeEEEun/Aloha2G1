"""Select one shared configuration, then freeze before any paper outcome."""
from pathlib import Path
import shutil
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text,fingerprint
from .practical_study import assert_backend,stage_inputs,now


def final_selection(out):
    study=read(out/'PRACTICAL_STUDY.json');ranking=read(out/'RECIPE_RANKING.json')
    physical=read(out/'TOP_RECIPE_PHYSICS.json');assert physical['status']=='PASS';evaluated=[]
    for name in ranking['top_recipes']:
        rows=[r for r in physical['rows'] if r['recipe']==name];planning=next(r for r in ranking['ranked'] if r['recipe']==name)
        assert len(rows)==2*len(study['physical_source_ids'])
        success=sum(r.get('stages',{}).get('FULL_TASK',False) for r in rows)
        interaction=sum(r.get('stages',{}).get(k,False) for r in rows for k in ('HANDOFF','RIGHT_OWNERSHIP'))
        contacts=[read(Path(r['folder'])/'PREGRASP_OBJECT_PROTECTION.json') for r in rows if r['physics_executed'] and (Path(r['folder'])/'PREGRASP_OBJECT_PROTECTION.json').exists()]
        displacement=float(np.mean([r['maximum_object_displacement_before_acquisition_m'] for r in contacts])) if contacts else None
        evaluated.append(dict(recipe=name,full_task_success=success,scheduled=len(rows),physical_rate=success/len(rows),
            planning_rate=planning['score'],handoff_ownership_count=interaction,source_deviation=planning['source_deviation'],
            mean_preacquisition_displacement_m=displacement,planning_seconds=planning['planning_seconds'],complexity=planning['complexity']))
    key=lambda r:(-r['physical_rate'],-r['planning_rate'],-r['handoff_ownership_count'],r['source_deviation'] if r['source_deviation'] is not None else float('inf'),
        r['mean_preacquisition_displacement_m'] if r['mean_preacquisition_displacement_m'] is not None else float('inf'),r['planning_seconds'],r['complexity'])
    evaluated.sort(key=key);name=evaluated[0]['recipe'];area=out/'recipes'/name
    config=dict(status='SELECTED_ONE_SHARED_CONFIG',recipe=name,selection=evaluated,
        calibration_parameters=read(area/'CALIBRATION_PARAMETERS.json'),practical_parameters=read(area/'PRACTICAL_PARAMETERS.json'),
        B_C_shared=True,selection_optimizes_C_minus_B=False,selected_at=now(),physical_subset=record(out/'PHYSICAL_SUBSET_PREDECLARATION.json'))
    path=out/'TRAIN40_FINAL_SHARED_CONFIG.json';atomic_json(path,config)
    report=out/'TRAIN40_FINAL_SHARED_CONFIG.md';atomic_text(report,'# Final shared TRAIN40 configuration\n\nSelected recipe: **'+name+'**.\n\n'
        'Selection uses measured full-task success over the entire predeclared paired TRAIN subset, then TRAIN40 planning coverage and the remaining predeclared common tie-breakers. '
        'No method-specific recipe or C-minus-B objective is used.\n\n```json\n'+__import__('json').dumps(config,indent=2)+'\n```\n')
    return [path,report]


def freeze(out):
    assert_backend(out);config=read(out/'TRAIN40_FINAL_SHARED_CONFIG.json');source=out/'recipes'/config['recipe']
    area=out/'FINAL_FROZEN';stage_inputs(source,area)
    for name in ('ABC_STUDY.json','CALIBRATION_READY.json','CALIBRATION_PARAMETERS.json','PRACTICAL_PARAMETERS.json'):
        shutil.copy2(source/name,area/name)
    for destination in (out,area):
        state=read(destination/'ABC_STUDY.json');state['FINAL_CONFIG_FROZEN']=True;atomic_json(destination/'ABC_STUDY.json',state)
    from .architecture_reports import alias
    aliases=alias(out);paper=read(aliases)
    paper['final_TRAIN40_results']='AUTHORIZED_CURRENT_PRACTICAL_TRAIN40_STUDY';paper['calibration_started']=True
    atomic_json(aliases,paper);shutil.copy2(aliases,area/'PAPER_METHOD_ALIAS.json')
    # No learned distribution statistic exists in this converter. Summarize all
    # TRAIN40 geometry, but do not invent an outcome-driven fit or world offset.
    ids=read(out/'PRACTICAL_STUDY.json')['train_source_ids'];positions=[];relations=[]
    for sid in ids:
        phase=read(area/'source_phase'/sid/'PHASE_RECORD.json');positions.append(np.asarray(phase['initial_object_pose_world'])[:3,3])
        relations.append(np.asarray(phase['source_functional_tool_object_relations']['left'])[:3,3])
    stats=out/'TRAIN40_SOURCE_STATISTICS.json';atomic_json(stats,dict(source_ids=ids,source_count=40,
        object_position_mean=np.mean(positions,axis=0),object_position_min=np.min(positions,axis=0),object_position_max=np.max(positions,axis=0),
        source_left_relation_mean=np.mean(relations,axis=0),fitted_converter_statistics={},
        reason='No learned TRAIN-distribution parameters in frozen architecture; all40 episode priors are read directly. Empirical summary does not alter task registration or embodiment calibration.',outcomes_used=False))
    paths=[Path(x['path']) for x in paper['workspace_code_manifest']]
    paths += [Path(x['path']) for x in read(out/'IMPLEMENTATION_FREEZE.json')['files']]
    paths += [out/n for n in ('CALIBRATION_SEARCH_SPACE.json','PRACTICAL_STUDY.json','TRAIN40_FINAL_SHARED_CONFIG.json','TRAIN40_SOURCE_STATISTICS.json','PAPER_METHOD_ALIAS.json')]
    paths += [p for name in ('bootstrap','source_phase','target_repair') for p in (area/name).rglob('*') if p.is_file()]
    paths += [area/n for n in ('ABC_STUDY.json','CALIBRATION_READY.json','CALIBRATION_PARAMETERS.json','PRACTICAL_PARAMETERS.json','SPLIT_CONTRACT.json','COVERAGE8.json','PAPER_METHOD_ALIAS.json')]
    digest,files=fingerprint(paths)
    manifest=out/'FINAL_TRAIN40_FREEZE_MANIFEST.json'
    atomic_json(manifest,dict(status='FROZEN',hash=digest,files=files,frozen_at=now(),recipe=config['recipe'],
        converter_code_hash=paper['workspace_code_hash'],planner=record(ROOT/'tools/contact_coordination/joint_path_planner.py'),
        source_prior=record(ROOT/'tools/contact_coordination/source_motion_prior.py'),
        shared_configuration=record(out/'TRAIN40_FINAL_SHARED_CONFIG.json'),
        seed_schedule='Same deterministic seed1729 for every RRT API invocation; no per-source reseeding or retries',
        compute_budget=read(out/'PRACTICAL_STUDY.json')['planner_budget'],
        no_more_TRAIN40_outcome_driven_tuning=True,DEV35_started=False,ACT_started=False))
    for x in paper['workspace_code_manifest']:
        src=Path(x['path']);dst=out/'FROZEN_FINAL_CODE'/src.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    gate=dict(FINAL_CONFIG_FROZEN=True,checks=dict(one_shared_config=True,code_and_parameters_frozen=True),evidence=[record(manifest)])
    for destination in (out,area):atomic_json(destination/'FINAL_CONFIG_FROZEN.json',gate)
    # These files are gates, deliberately outside their own recursive manifest.
    return [manifest,aliases,stats,out/'FINAL_CONFIG_FROZEN.json',area/'FINAL_CONFIG_FROZEN.json']


def verify_freeze(out):
    manifest=read(out/'FINAL_TRAIN40_FREEZE_MANIFEST.json')
    for x in manifest['files']:
        if record(x['path'])!=x:raise RuntimeError('FROZEN_FINAL_DEPENDENCY_CHANGED: '+x['path'])
    return manifest
