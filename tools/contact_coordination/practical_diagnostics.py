"""Explain the fixed sweep using recorded causes; never propose more recipes."""
from pathlib import Path
from collections import Counter
from .io import read,record,atomic_json,atomic_text


def run(out):
    study=read(out/'PRACTICAL_STUDY.json');sweep=read(out/'PLANNING_SWEEP.json')
    ranking=read(out/'RECIPE_RANKING.json');assert sweep['status']=='PASS'
    space=read(out/'CALIBRATION_SEARCH_SPACE.json');rows=sweep['rows'];causes={};witnesses=[]
    all_causes=Counter();default={}
    for row in rows:
        if row['recipe']=='CURRENT_DEFAULT':default[(row['source_id'],row['method'])]=row
    changes=[]
    for recipe in study['recipes']:
        group=[r for r in rows if r['recipe']==recipe];counts=Counter()
        for row in group:
            failure=row['summary']['first_causal_failure']
            if failure:
                cause=failure.get('cause','UNKNOWN');phase=failure.get('phase','UNKNOWN')
                counts[(cause,phase)]+=1;all_causes[(cause,phase)]+=1
                reference=failure.get('evidence');witness=dict(recipe=recipe,source_id=row['source_id'],method=row['method'],cause=cause,phase=phase,evidence=reference)
                if reference and Path(reference['path']).exists():
                    assert record(reference['path'])==reference
                    phases=read(reference['path']).get('phases',[])
                    value=next((p for p in phases if p.get('phase')==phase),None)
                    if value:
                        candidates=value.get('candidates',[])
                        witness.update(at_failed_phase_generated_IK_solutions=len(candidates),
                            at_failed_phase_IK_pass=sum(bool(c.get('goal_satisfied')) for c in candidates),
                            at_failed_phase_geometry_pass=sum(bool(c.get('admissible')) for c in candidates),
                            selected_errors=value.get('selected_errors'),
                            IK_evaluation_counts=sorted({c['nfev'] for c in candidates if 'nfev' in c}))
                witnesses.append(witness)
            prior=default[(row['source_id'],row['method'])]
            changes.append(dict(recipe=recipe,source_id=row['source_id'],method=row['method'],
                default_complete=prior['summary']['COMPLETE_PLAN'],recipe_complete=row['summary']['COMPLETE_PLAN'],
                default_cause=prior['summary']['first_causal_failure'],recipe_cause=failure))
        causes[recipe]=[dict(cause=c,phase=p,count=n) for (c,p),n in counts.most_common()]
    leading=[dict(cause=c,phase=p,count=n) for (c,p),n in all_causes.most_common()]
    path=out/'PLANNING_CALIBRATION_CAUSAL_DIAGNOSTICS.json'
    atomic_json(path,dict(status='PASS',causes_by_recipe=causes,aggregate_causes=leading,
        failure_witnesses=witnesses,paired_default_changes=changes,
        no_new_recipes=True,no_architecture_changes=True,physics_started=False))
    if ranking['planning_improved']:return [path]
    text='# Planning calibration insufficient\n\nAll four predeclared planning-only recipes have finished. '
    text+='The predeclared improvement gate did not pass; expensive physics and final paper A/B evaluation are not started.\n\n'
    text+='| Recipe | B complete /40 | C complete /40 | Mean net plan gain | Newly covered distinct sources |\n|---|---:|---:|---:|---:|\n'
    for r in sorted(ranking['ranked'],key=lambda r:r['complexity']):
        text+=f'| {r["recipe"]} | {r["complete"]["B_INDEPENDENT"]} | {r["complete"]["C_COUPLED"]} | {r["average_net_gain"]:.1f} | {len(r["new_distinct_sources"])} |\n'
    text+='\nThe gate requires at least two additional complete plans per method on average and at least two distinct newly covered sources. Weaker-method coverage is a ranking tie-breaker, as requested.\n\n'
    if leading:
        dominant=leading[0]
        text+=f'The most frequent recorded first cause is **{dominant["cause"]} at {dominant["phase"]}**, affecting {dominant["count"]} of the 320 scheduled recipe/method/source cases. '
        if dominant['cause']=='NO_IK':
            text+='This means the fixed bounded endpoint solver did not produce a target-satisfying configuration at that phase. It is a candidate/IK coverage bottleneck before path connection, not evidence that RRT failed, and not a proof that the source is physically unreachable. '
        elif dominant['cause']=='NO_CONNECTING_PATH':
            text+='This is a connection failure within the unchanged collision checks and RRT budget; it does not establish global infeasibility. '
        elif dominant['cause']=='NO_VALID_CANDIDATE':
            text+='The endpoint candidate filters rejected the available bounded family before a complete connection could be admitted. '
        text+='Exact source/method witnesses, saved evidence hashes, solver evaluation counts and available selected pose errors are in PLANNING_CALIBRATION_CAUSAL_DIAGNOSTICS.json.\n\n'
    text+='| Recipe | First causal failures |\n|---|---|\n'
    for recipe in study['recipes']:
        text+='| '+recipe+' | '+('; '.join(f'{r["cause"]} @ {r["phase"]}: {r["count"]}' for r in causes[recipe]) or 'None')+' |\n'
    text+='\nThe changed parameter bundles were:\n\n'
    baseline=space['recipes']['CURRENT_DEFAULT']
    for recipe in study['recipes'][1:]:
        difference={k:v for k,v in space['recipes'][recipe].items() if v!=baseline[k]}
        text+='- '+recipe+': '+', '.join(f'{k}={v} (default {baseline[k]})' for k,v in difference.items())+'.\n'
    text+='\nThese four bundles did not satisfy the shared coverage-improvement rule. This small non-factorial sweep does not isolate the causal effect of each scalar. '
    text+='The displacement tolerance is a measured physical acceptance bound; it cannot repair an endpoint NO_IK result. '
    text+='The converter representation, source-motion prior, RRT, collision checks, retiming, controller and scorer remain frozen. No fifth recipe, source-specific rescue, outcome-driven architecture change or expensive physics is attempted.\n\n'
    text+='FINAL SHARED CONFIG: NOT SELECTED\n\nFINAL PAPER A/B: NOT RUN\n\nDEV35 STARTED: NO\n\nACT STARTED: NO\n'
    report=out/'PLANNING_CALIBRATION_INSUFFICIENT.md';atomic_text(report,text)
    terminal='============================================================\nTRAIN40 PRACTICAL CALIBRATION + PAPER A/B\n============================================================\n\nPLANNING SWEEP\n'
    for r in sorted(ranking['ranked'],key=lambda r:r['complexity']):
        terminal+=f'{r["recipe"]}: B {r["complete"]["B_INDEPENDENT"]}/40; C {r["complete"]["C_COUPLED"]}/40\n'
    terminal+='\nSELECTED RECIPE: NONE — PLANNING CALIBRATION INSUFFICIENT\nPHYSICAL VERIFICATION: NOT STARTED\nFINAL FROZEN CONFIG: NOT SELECTED\nFINAL PAPER A/B: NOT RUN\nFINAL TABLE / FIGURES / VIDEOS: NOT GENERATED\n'
    terminal+='\nREPORT: '+str(report)+'\nDEV35 STARTED: NO\nACT STARTED: NO\n============================================================\n'
    summary=out/'FINAL_TERMINAL_SUMMARY.txt';atomic_text(summary,terminal)
    return [path,summary]
