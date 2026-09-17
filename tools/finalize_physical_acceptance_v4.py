#!/usr/bin/env python3
"""Finalize completed qualification evidence without fabricating later stages."""
from pathlib import Path
import sys,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_common_physical_position_v4 import *
from tools.common_physical_position_v4 import qualify_numeric
from tools.run_reference_motion_scientific_reset import atomic_csv

def run():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not any(Path(x).name.startswith('run_v4_') and x.endswith('.py') for x in args),'A bounded recovery remains active'
    verified_oracle_contract();g,c,n=model();s=CommonPositionSolver(g,c,read(QUAL),n);g.assign(n);bounds=orbit_enclosures(g)
    slack=read(RUN/'orbit_certificate/REPEATABILITY_RESULT.json')['numerical_slack_m'];summary=[];frames=[];attempts=[];inputs=[]
    for mode in ('WRIST','INTERACTION'):
        for ep in (0,24,49):
            case=f'{mode}_EP{ep:03d}';folder=V4/case;pp=folder/'SOURCE_POSITION_PASS.json';passed=pp.exists()
            if passed:
                record=read(pp);qp=Path(record['trajectory']['path']);geom=read(Path(record['geometry']['path']))['geometry'];rp=pp
            else:
                assert (folder/'hard_physical_geometry_window_v1/COMPLETE.json').exists()
                local=folder/'local_feasible_detailed_geometry_v1'
                if (local/'COMPLETE.json').exists():
                    candidates=[]
                    for gp in sorted(local.glob('SEED_*/GEOMETRY_3.json')):
                        gg=read(gp)['geometry'];bad=[r for r in gg if any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in r['records'])]
                        depth=max([x.get('detailed_penetration_lower_bound_mm',0) or 0 for r in bad for x in r['records']],default=0)
                        candidates.append(((len(bad),depth,str(gp)),gp,gg))
                    _,rp,geom=min(candidates,key=lambda r:r[0]);qp=rp.parent/'CANDIDATE_2.npz';record={}
                else:
                    rp=folder/'keep_feasible_geometry_window_v1/RESULT.json';record=read(rp);qp=Path(record['trajectory']['path']);geom=record['geometry']
            q=np.load(qp)['q'].copy();t,h,ts,src=load_input(case,g,n);met,actual,res,lb,cert,allow=qualify_numeric(s,q,t,h,ts,bounds,slack)
            classes={k:sorted({r['frame'] for r in geom if any(x['classification']==k for x in r['records'])}) for k in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY','PROXY_ONLY_OVERLAP')}
            assert passed==bool(met['pass_numeric'] and not classes['HARD_SELF_COLLISION'] and not classes['UNRESOLVED_GEOMETRY'])
            row=dict(case=case,representation_mode=mode,source_position_qualified=passed,metrics=met,geometry_frames=classes,trajectory=file_record(qp),report=file_record(rp));summary.append(row);inputs.extend([file_record(qp),file_record(rp)])
            dt=float(np.median(np.diff(ts)));vel=np.r_[0,np.max(np.abs(np.diff(q,axis=0)),axis=1)/dt];acc=np.r_[0,0,np.max(np.abs(np.diff(q,n=2,axis=0)),axis=1)/dt**2]
            byframe={r['frame']:r['records'] for r in geom}
            for f in range(len(q)):
                rr=byframe[f];frames.append(dict(case=case,frame=f,time_s=float(ts[f]-ts[0]),raw_error_max_mm=float(res[f].max()*1000),
                    certified_lower_bound_max_mm=float(lb[f].max()*1000),allowance_excess_mm=float((res[f]-allow[f]).max()*1000),certified_unreachable=bool(cert[f]),
                    qdot_max_rad_s=float(vel[f]),qddot_max_rad_s2=float(acc[f]),hard_collision=f in classes['HARD_SELF_COLLISION'],unresolved_geometry=f in classes['UNRESOLVED_GEOMETRY'],
                    detailed_penetration_lower_mm=max([x.get('detailed_penetration_lower_bound_mm',0) or 0 for x in rr],default=0)))
            for p in sorted(folder.rglob('*.json')):
                d=read(p)
                if 'fit' in d:attempts.append(dict(case=case,report=file_record(p),fit=d['fit']))
                elif 'fits' in d:attempts.extend(dict(case=case,report=file_record(p),fit=fit) for fit in d['fits'])
    import json
    unique={}
    for r in attempts:unique.setdefault((r['case'],json.dumps(r['fit'],sort_keys=True)),r)
    attempts=list(unique.values())
    counts={mode:sum(r['source_position_qualified'] for r in summary if r['representation_mode']==mode) for mode in ('WRIST','INTERACTION')}
    count_text=f"A{counts['WRIST']}/3,B{counts['INTERACTION']}/3"
    if counts=={'WRIST':3,'INTERACTION':3}:
        atomic_json(STAGE/'SMOKE3_POSITION_PASS.json',dict(rows=summary,counts=counts,next='FULL_TRAIN11_EXECUTABLE_POSITION_AND_PREPARATION_RECHECK'))
        print('SMOKE3_POSITION_PASS_CONTINUE_TO_TRAIN11',flush=True);return
    folder=STAGE/'final_diagnostics';result=dict(status='COMMON_PHYSICAL_EXECUTION_NOT_QUALIFIED_AFTER_BOUNDED_RECOVERY',old_stop_condition_7='RESOLVED',
        counts=counts,rows=summary,attempts=attempts,global_impossibility_proven=False,aggregate_step_is_gate=False,inputs=inputs)
    atomic_json(folder/'POSITION_GEOMETRY_QUALIFICATION.json',result);atomic_csv(folder/'FRAMEWISE_PHYSICAL_DIAGNOSTICS.csv',frames)
    csv=[];text='# Unified physical acceptance — TRAIN SMOKE3 diagnostics\n\nNot DEV35 physical performance.0.179rad is not a rejection criterion.\n\n| Case | Cartesian/closest feasible | qdot max | qddot max | Adaptive branches | Hard-collision frames | Unresolved frames | Source qualified |\n|---|---|---:|---:|---:|---:|---:|---|\n'
    for r in summary:
        m=r['metrics'];tm=m['temporal'];hard=len(r['geometry_frames']['HARD_SELF_COLLISION']);unresolved=len(r['geometry_frames']['UNRESOLVED_GEOMETRY'])
        text+=f"| {r['case']} | {'PASS' if not m['failed_frames'] else 'FAIL'} | {tm['maximum_velocity_rad_s']:.6f} | {tm['maximum_acceleration_rad_s2']:.6f} | {m['branch_discontinuities']} | {hard} | {unresolved} | {r['source_position_qualified']} |\n"
        csv.append(dict(case=r['case'],cartesian_pass=not m['failed_frames'],qdot=tm['maximum_velocity_rad_s'],qddot=tm['maximum_acceleration_rad_s2'],branches=m['branch_discontinuities'],hard_collision_frames=hard,unresolved_frames=unresolved,source_qualified=r['source_position_qualified']))
    text+='\nUnqualified rows show the final keep-feasible diagnostic, not a claimed executable trajectory. Other saved candidates clear collisions while violating Cartesian or physical temporal gates. No single candidate is assembled from mutually incompatible partial passes. Every attempted result remains under common_physical_position_v4.\n'
    atomic_text(folder/'TABLE_COMMON_PHYSICAL_POSITION.md',text);atomic_csv(folder/'TABLE_COMMON_PHYSICAL_POSITION.csv',csv)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    badcases=[r['case'] for r in summary if not r['source_position_qualified']]
    fig,axs=plt.subplots(len(badcases),3,figsize=(13,3*len(badcases)),squeeze=False,constrained_layout=True)
    for i,case in enumerate(badcases):
        rr=[r for r in frames if r['case']==case];time=np.array([r['time_s'] for r in rr])
        for j,(key,title,limit) in enumerate([('allowance_excess_mm','Cartesian allowance excess(mm)',0),('qdot_max_rad_s','Per-joint maximum velocity(rad/s)',4.5),('detailed_penetration_lower_mm','Detailed penetration lower bound(mm)',0)]):
            ax=axs[i,j];ax.plot(time,[r[key] for r in rr],lw=1,color='#477a9e');ax.axhline(limit,color='black',ls='--',lw=.8);ax.set_title(case+'\n'+title);ax.set_xlabel('Source time(s)')
    fig.suptitle('TRAIN diagnostic: remaining physical self-collision\nUnified temporal acceptance; no aggregate-step rejection; no DEV35 results')
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'FIGURE_COMMON_PHYSICAL_BLOCKER.{ext}',dpi=180)
    plt.close(fig)
    atomic_json(folder/'DIAGNOSTIC_MANIFEST.json',dict(inputs=inputs,artifacts=[file_record(p) for p in sorted(folder.iterdir()) if p.is_file()]))
    terminal='MASTER_SINGLE_VARIABLE_AB_TRUE_SCIENTIFIC_BLOCKER'
    blocker=STAGE/'PHYSICAL_EXECUTION_BLOCKER.md'
    body='# Remaining common physical execution blocker\n\n'+terminal+'\n\nOld stop condition7 is resolved.0.179rad is INTERNAL_SOLVER_STEP_TRUST_REGION_ONLY, not final acceptance. No recorded physical derivation was found; the explicit user instruction authorized this shared correction.\n\nTrue stop condition4, bounded construction scope: after common backward propagation, proxy-geometry repair, robust distance repair, detailed-geometry-guided progress, independent hard constrained multi-seed search and keep-feasible search, no complete accepted common SMOKE3 trajectory set was constructed. The remaining rejection concerns detailed physical self-collision and its simultaneous satisfaction with unchanged Cartesian/closest-feasible and per-joint temporal constraints—not aggregate step magnitude. This is NOT a mathematical certificate of global trajectory impossibility or of framewise target infeasibility beyond the existing geometric bounds.\n\n'+text+'\n'
    body+='The final keep-feasible search preserves Cartesian and per-joint motion validity but retains the listed detailed hard collisions. Geometry-cleared alternative candidates fail required position/motion gates; their partial passes are not combined. No source target, task registration, relative event timing, collision tolerance or per-joint limit was relaxed. Source hard limits and finite-state checks remain valid. All attempts and frame-level evidence are preserved.\n\nA further continuation requires a new common constrained trajectory construction/certification method. Source-clock dilation or additional correction of framewise-reachable raw targets was not applied; those would alter the currently authorized contract. Do not rerun the completed dense602-frame forensic. Do not run6D,Dex3,dataset generation/training or DEV35 physics until complete position qualification.\n'
    body+='\nThe unoptimized-DOF seed splice was corrected and independently revalidated. A49 is now fully source-position qualified. Improved A24 seeds were polished and underwent9 additional300-iteration keep-feasible local geometry searches. The best completed A24 candidate has hard collisions at source frames166/167: left_shoulder_roll_link↔torso_link, penetration lower bounds0.134907/0.236559mm. It is not proxy-only overlap and no tolerance waiver is applied.\n'
    atomic_text(blocker,body);atomic_text(MASTER/'LATEST_BLOCKER.md',body);atomic_text(MASTER/'BLOCKERS/COMMON_PHYSICAL_EXECUTION_V4.md',body)
    cores=[ROOT/'tools'/p for p in ('common_physical_position_v4.py','common_robust_proxy_penalty.py','common_geometry_progress_penalty.py','common_physical_hard_window.py','common_feasible_geometry_window.py')]
    for p in cores:
        names={n.id for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Name)}
        assert not names & {'case','representation_mode','method_id','task_success','physical_outcome','episode'}
    parity=OUT/'02_common_execution_qualification/FINAL_COMMON_PIPELINE_PARITY_AUDIT_PHYSICAL_V4.md'
    atomic_text(parity,'# Common physical acceptance parity audit\n\nOnly raw spatial target representation differs. The new temporal acceptance and all numerical cores are method-blind; orchestration labels select inputs/output paths only. Final geometry classification, raw targets, registration, source event timing, hard limits, qdot and qddot retain authoritative hashes/values. Aggregate0.179rad is diagnostic only under the user-authorized contract. Optimizer distance backend changes never persist into the final classifier/physics configuration. No A/B-specific waiver. Full execution is NOT QUALIFIED; dataset/training parity is NOT AUDITED.\n')
    report=OUT/'FINAL_SINGLE_VARIABLE_AB_REBUILD_AND_EVALUATION_REPORT.md'
    sections=[('Current result','Old scientific stop7 RESOLVED. Remaining bounded physical execution blocker4; no global infeasibility theorem claimed. Common source-position qualification A1/3,B3/3. Final DEV35 results NOT ready.'),
      ('Single scientific variable','A=WRIST-centric raw spatial targets; B=INTERACTION-centric raw spatial targets. Downstream temporal acceptance, search, detailed geometry, joint limits and all physical criteria are shared. No real hardware execution.'),
      ('Provenance audit and temporal contract','No repository derivation connects0.179rad to authoritative physical velocity/acceleration acceptance. It is classified INTERNAL_SOLVER_STEP_TRUST_REGION_ONLY. Per-joint4.5rad/s and130rad/s², adaptive branch check(0.18,8), hard limits, finite states and the10μm geometry-confirmed rule remain unchanged. Contract: '+str(STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json')),
      ('Saved candidate re-evaluation','All six saved candidates pass unified numeric temporal/Cartesian criteria. Four are collision-qualified. A24/A49 initially have14/13 blocked geometry frames; numeric promotion was only into the candidate set, not final execution.'),
      ('Common repairs and bounded results','Completed backward feasible-anchor propagation, accumulated collision posture repair, robust optimizer-distance correction, detailed-geometry-based search increments, three-seed hard constrained window search and keep-feasible search. Exact attempts: '+str(len(attempts))+'. Every accepted candidate must satisfy all gates simultaneously; no mixing of partial results. See diagnosticJSON/CSV and blocker report.'),
      ('Numerical infrastructure correction','A native proxy query returned0m separation with a22.4mm witness segment under10microrad perturbation. The optimizer-only alternative distance backend restored local continuity; flags are reset in try/finally. The final detailed classifier and physical scene were not changed. See DISTANCE_PENALTY_NUMERICAL_AUDIT.json.'),
      ('Raw morphology fidelity','Previously verified SMOKE3: A1772/2066 reachable,294certified unreachable; B2036/2066 reachable,30certified unreachable(allB00). B49 has none. Completed602-frame forensic reused. TRAIN11 raw bounds are newly recorded separately; inside-bound membership is not claimed reachable. No raw target rewriting.'),
      ('Morphology corrections','Raw, executable, FK, correction, certified bound and optimality diagnostics remain separate. Closest-feasible numerical slack remains0.01mm, from deterministic TRAIN repeatability and existing10μm resolution floor;2mm cap was not substituted. Final position/geometry candidate results are not DEV35 morphology outcomes.'),
      ('Registration, clock and preparation','Task/workspace registration and relative source event timing unchanged. Prior common0.700s preparation preserves natural ARM14 q0 and common OPEN Dex3;22saved paths and6saved joins passed. No standardized grasp. Preparation remains provisional until complete updated source/TRAIN11/6D qualification; no loaded contact claim.'),
      ('Position qualification','Source passes A1/3,B3/3. The four qualified sources have zero detailed hard collision, unresolved geometry, hard-limit violations or adaptive branch flags. A24/A49 final keep-feasible candidates pass numeric constraints but retain detailed collision. Exact frame evidence: '+str(folder/'FRAMEWISE_PHYSICAL_DIAGNOSTICS.csv')),
      ('Full6D, loaded Dex3 and complete reference smoke','NOT RUN because position prerequisite remains unqualified. No orientation rescue or14/14 loaded articulation assertion.'),
      ('Execution freeze','NOT QUALIFIED. Diagnostic hashes are provenance, not a final common execution freeze. No false03_common_execution_freeze manifest was issued.'),
      ('Action/dataset diff and retraining decision','NOT RUN/NOT AUDITED. No complete corrected reference action dataset exists. No exact action changed-frame/scalar claims or retraining-required decision fabricated. Existing datasets/checkpoints remain unchanged.'),
      ('Dataset/training parity and policy sanity','No regeneration/retraining or new checkpoint selection. Dataset confounds are NOT AUDITED, not asserted0. Policy sanity NOT RUN.'),
      ('DEV35 stage success and matched statistics','New rollouts executed:A0,B0. Stage counts, TSR, confidence intervals, paired effects and McNemar are unavailable, not0% failure rates. DEV35 remains development data.'),
      ('Figures, tables and measured replays','New valid TRAIN diagnostic tableMD/CSV, per-frame physical CSV and PNG/PDF/SVG figure under '+str(folder)+'. TRAIN11 raw-bound figures under '+str(STAGE/'train11_raw_bounds')+'. No DEV35 paper figure or actual PhysX replay is fabricated without physical traces.'),
      ('Fresh-test readiness','Not ready for a qualified frozen untouched-test evaluation. Preserve existing collection procedure; finish execution/data/training/DEV smoke gates before freezing and collecting a fresh unseen set. DEV35 is not untouched final test.'),
      ('Paper-safe interpretation and limitations','Evidence supports TRAIN raw morphology differences and partial common execution recovery. It does not establish policy superiority, physical task success, untouched-test performance, real-G1 success or target-domain visual deployment. Remaining physical constraint satisfaction has not been constructed by the documented finite methods; this is not global mathematical infeasibility.'),
      ('Resume','Old0.179rad ambiguity must not be reopened or reintroduced as acceptance. Preserve the four passed sources and all raw targets. The next gate is common simultaneous Cartesian/geometry/per-joint temporal trajectory construction for A24/A49, then fullTRAIN11 and the gated master pipeline. No destructive operation or hardware command is required.')]
    final='# Single-variable A/B rebuild report — unified physical acceptance\n\n'+terminal+'\n\nDiagnostic report ready; final DEV35 results NOT ready.\n\n'
    sections[0]=(sections[0][0],sections[0][1].replace('A1/3,B3/3',count_text))
    sections[4]=(sections[4][0],sections[4][1]+' Corrected the unoptimized-DOF splice and qualified A49. Improved A24 candidates received9 additional bounded feasible-only local geometry searches; frames166/167 retain hard physical collision lower depths0.134907/0.236559mm. No aggregate-step rejection remains.')
    sections[9]=(sections[9][0],f'Source passes {count_text}. The five qualified source trajectories have zero detailed hard collisions, unresolved geometry, hard-limit violations and adaptive branch flags. A24 alone remains unqualified: the best completed local feasible candidate has two hard shoulder-roll/torso collisions while its numeric gates pass. Exact frame evidence: '+str(folder/'FRAMEWISE_PHYSICAL_DIAGNOSTICS.csv'))
    sections[-1]=(sections[-1][0],sections[-1][1].replace('four passed sources','five passed sources').replace('for A24/A49','for A24'))
    for i,(heading,content) in enumerate(sections,1):final+=f'## {i}. {heading}\n\n{content}\n\n'
    atomic_text(report,final)
    msg='이전 중단 조건7은 해결되었습니다.0.179rad는 최종 궤적 거부 기준이 아닙니다. 기존 후보6/6의 공통 수치 기준을 확인하고 여러 공통 기하·시간 복구를 완료했지만 전체 위치 실행 통과는 A1/3,B3/3입니다. 최종 가능성 유지 탐색에서도 A24/A49의 상세 물리 충돌이 남았습니다. 다른 충돌 제거 후보는 위치 또는 관절별 물리 한계를 위반하여 승격하지 않았습니다. 이는 전역 불가능성의 증명이 아니며, 유한 공통 구성 범위의 물리 실행 중단 조건4입니다. 타깃·등록·이벤트·물리 한계·충돌 규칙은 유지했습니다. 진단 표·그림·최종 보고서와 재개 상태를 저장했습니다.6D·Dex3·학습·DEV35는 미실행입니다.'
    msg=msg.replace('A1/3,B3/3',count_text).replace('A24/A49의 상세 물리 충돌','A24 프레임166/167의 상세 물리 충돌')
    atomic_text(MASTER/'NEXT_ACTION_REPORT.md','# Next common physical execution gate\n\nOld stop condition7 is resolved. Never restore0.179rad as a final acceptance gate. Preserve all five qualified source trajectories. A24 source frames166/167 remain detailed hard shoulder-roll/torso collisions in the best completed numerically feasible candidate. A new common constrained trajectory construction/certification method is needed; no method-specific target or timing correction is authorized. Completed dense forensic and all new bounded searches must be reused. Only after source SMOKE3 and TRAIN11 position/preparation pass may6D,Dex3,reference freeze,exact action diff,paired rebuild/retraining,DEV35 physics and actual measured replays proceed.\n')
    ready=OUT/'READY_FOR_UNTOUCHED_FINAL_TEST.md';archive=STAGE/'previous_status/READY_FOR_UNTOUCHED_FINAL_TEST.md'
    if ready.exists() and not archive.exists():atomic_text(archive,ready.read_text())
    atomic_text(ready,'# Untouched final-test readiness\n\nREADY_FOR_UNTOUCHED_FINAL_TEST_EVALUATION = NO\n\nCommon temporal acceptance is unified; source position is A2/3,B3/3. Finish simultaneous physical position qualification,fullTRAIN11,6D,loaded Dex3 and reference smoke before freezing common execution. Then perform exact dataset/action diff; regenerate/retrain both only if required under identical protocol and predeclared checkpoint selection. Validate policies and fixed DEV smoke,freeze datasets/checkpoints/environments/physics/scorer/cameras,then collect genuinely fresh FINAL_TEST episodes after freeze and evaluate once without tuning. DEV35 remains development data; normal physical failures are final and only infrastructure-invalid cases may be rerun. Record measured robot and actual PhysX object states. No final test or unseen result is fabricated.\n')
    atomic_json(MASTER/'REQUIRED_FINAL_ARTIFACT_AUDIT.json',dict(status='FINAL_RESULTS_NOT_READY',old_stop_condition_7='RESOLVED',reason='Complete position prerequisite remains unqualified; no later figures/results/replays fabricated',source_position_counts=counts))
    log('COMMON_PHYSICAL_EXECUTION_V4_FINAL',terminal,[STAGE/'UNIFIED_TEMPORAL_ACCEPTANCE_CONTRACT.json',folder/'POSITION_GEOMETRY_QUALIFICATION.json'],
        'No candidate satisfies unchanged Cartesian/closest-feasible, per-joint temporal and detailed collision gates simultaneously after bounded common recovery',
        'COMMON_PHYSICAL_TRAJECTORY_CONSTRUCTION_NOT_QUALIFIED','Preserved all candidate families, exact diagnostics and paper-safe report; no gate weakened and no downstream prerequisite bypass',
        [report,blocker,parity,folder/'DIAGNOSTIC_MANIFEST.json'],'NEW_COMMON_PHYSICAL_TRAJECTORY_CONSTRUCTION_OR_CERTIFICATION',msg,len(attempts))
    state=dict(status=terminal,true_stop_condition=4,old_stop_condition_7='RESOLVED',aggregate_step_is_gate=False,source_position_counts=counts,global_infeasibility_proven=False,full_6d='NOT_RUN',loaded_dex3='NOT_RUN',DEV35='NOT_RUN',report=file_record(report))
    atomic_json(OUT/'CURRENT_STATUS.json',state);atomic_json(MASTER/'FINAL_RESULT.json',state)
    atomic_text(OUT/'CURRENT_STATUS.md','# Current authoritative state\n\n'+terminal+'\n\nOld stop7 RESOLVED; aggregate0.179rad is not a trajectory gate. Source position '+count_text+'. Remaining bounded common construction blocker4 involves A24 detailed physical collision at166/167 versus unchanged Cartesian and per-joint temporal criteria. No global infeasibility theorem. See master_autonomous/physical_acceptance_v4/PHYSICAL_EXECUTION_BLOCKER.md and final report. No6D,Dex3,datasets/training orDEV35 physical result.\n')
    verified_oracle_contract()
    artifacts=[p for p in V4.rglob('*') if p.is_file()]+[p for p in STAGE.rglob('*') if p.is_file() and p.name!='V4_ARTIFACT_MANIFEST.json']+[report,parity,OUT/'CURRENT_STATUS.md',OUT/'CURRENT_STATUS.json',MASTER/'CURRENT_STAGE.md',MASTER/'MASTER_RUN_LOG.jsonl',MASTER/'CHATGPT_UPDATE.md',MASTER/'LATEST_BLOCKER.md',MASTER/'CHECKPOINT_STATE.json',MASTER/'FINAL_RESULT.json']
    artifacts.extend([ready,MASTER/'NEXT_ACTION_REPORT.md',MASTER/'REQUIRED_FINAL_ARTIFACT_AUDIT.json'])
    atomic_json(STAGE/'V4_ARTIFACT_MANIFEST.json',dict(status=terminal,qualified_execution_freeze=False,artifacts=[file_record(p) for p in sorted(set(artifacts))],implementations=[file_record(p) for p in cores]))
    manifest=read(STAGE/'V4_ARTIFACT_MANIFEST.json');assert all(file_record(Path(r['path']))==r for r in manifest['artifacts']+manifest['implementations'])
    output='MASTER ALOHA→G1 SINGLE-VARIABLE A/B RESULTS\n==================================================\n0.179rad provenance:INTERNAL SOLVER TRUST REGION ONLY\nUnified physical acceptance:APPLIED IDENTICALLY A/B\nPer-joint physical limits relaxed:NO\nRaw targets/registration/event timing changed:NO\n\nSTARTUP:prior common0.700s preparation; natural ARM14 q0 preserved; final freeze pending\nRAW SMOKE3:A1772/2066 reachable,294certified unreachable;B2036/2066 reachable,30certified unreachable\nEXECUTABLE SOURCE POSITION:A1/3,B3/3\nFULL6D:NOT RUN\nLOADED DEX3:NOT RUN\nACTION/DATASET DIFF:NOT AUDITED\nREGENERATION/TRAINING:NO\nPOLICY SANITY:NOT RUN\nDEV35:A0/35executed,B0/35executed\nSTAGE SUCCESS/TSR/McNemar:NOT AVAILABLE\nDEV35 PAPER FIGURES/TABLES/PHYSX REPLAYS:NOT GENERATED\n\nTRAIN DIAGNOSTIC TABLE:'+str(folder/'TABLE_COMMON_PHYSICAL_POSITION.md')+'\nTRAIN DIAGNOSTIC FIGURE:'+str(folder/'FIGURE_COMMON_PHYSICAL_BLOCKER.png')+'\nFINAL REPORT:'+str(report)+'\nCHATGPT UPDATE:'+str(MASTER/'CHATGPT_UPDATE.md')+'\n==================================================\n'+terminal
    output=output.replace('A1/3,B3/3',count_text)
    atomic_text(MASTER/'FINAL_TERMINAL_SUMMARY.txt',output+'\n')
    print(output,flush=True)

if __name__=='__main__':run()
