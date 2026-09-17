#!/usr/bin/env python3
"""Paper tables/statistics/plots from completed frozen outcomes, never estimates."""
from pathlib import Path
import csv,io,json,sys
import numpy as np
from scipy.stats import beta,binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.final_paper_reference_physics import DEST,OUT,STAGES,read,atomic_json,atomic_text,file_record,cases
PAPER=OUT/'07_paper_artifacts/final'
MODES=('WRIST','INTERACTION');LABELS=('A — Wrist','B — Interaction');COLORS=('#cc6b42','#277f9d')

def ci(k,n):return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def distribution(x):
    a=np.asarray(x).ravel();return dict(mean=float(a.mean()),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else None
def table(name,headers,rows,notes):
    PAPER.mkdir(parents=True,exist_ok=True);text='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    text+=''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'+notes+'\n'
    atomic_text(PAPER/(name+'.md'),text);buffer=io.StringIO();w=csv.writer(buffer);w.writerow(headers);w.writerows(rows);atomic_text(PAPER/(name+'.csv'),buffer.getvalue())
def save(fig,name,vector=False):
    for suffix in (('png','pdf','svg') if vector else ('png',)):fig.savefig(PAPER/f'{name}.{suffix}',dpi=250,bbox_inches='tight')
    plt.close(fig)

def characterize(rows,groups=('TRAIN40','DEV35')):
    result={};collections={}
    for group in groups:
        result[group]={};collections[group]={}
        for mode in MODES:
            selected=[r for r in rows if r['group']==group and r['representation_mode']==mode];records=[];res=[];lb=[];valid_res=[];orient=[];counts={};runtime=[];witnessed=0;frames=0;hard_frames=0;unresolved=0;proxy=0;six_count=0;prep_hard=0;branches=0;six_res=[];six_reasons={}
            for row in selected:
                p=read(DEST/'position'/row['key']/'RESULT.json');f=read(DEST/'full6d'/row['key']/'RESULT.json')
                assert p['outcome']!='INFRASTRUCTURE_INVALID' and f['outcome']!='INFRASTRUCTURE_INVALID','Never turn missing/invalid infrastructure into scientific losses'
                counts[p['outcome']]=counts.get(p['outcome'],0)+1;r=p['selected'];runtime.append(p['runtime_s'])
                with np.load(r['trajectory']['path']) as z:
                    error=z['POSITION_CORRECTION_MM'].max(axis=1);lower=z['CERTIFIED_LOWER_BOUND_MM'].max(axis=1)
                with np.load(DEST/'raw_certificate'/f"{row['key']}.npz") as z:assert np.array_equal(lower,z['CERTIFIED_LOWER_BOUND_MM'].max(axis=1)), 'Independent raw-bound audit disagrees'
                prep_hard+=sum(any(x['classification']=='HARD_SELF_COLLISION' for x in rr['records']) for rr in read(r['geometry']['path'])['preparation'])
                branches+=len(r['complete_temporal']['branch_discontinuity_frames'])
                geom=read(r['geometry']['path'])['source'];gvalid=np.array([not any(x['classification'] in ('HARD_SELF_COLLISION','UNRESOLVED_GEOMETRY') for x in rr['records']) for rr in geom])
                assert len(gvalid)==len(error)
                witnessed+=int(np.sum((error<=10)&gvalid));frames+=len(error)
                hard_frames+=sum(any(x['classification']=='HARD_SELF_COLLISION' for x in rr['records']) for rr in geom)
                unresolved+=sum(any(x['classification']=='UNRESOLVED_GEOMETRY' for x in rr['records']) for rr in geom)
                proxy+=sum(any(x['classification']=='PROXY_ONLY_OVERLAP' for x in rr['records']) for rr in geom)
                res.extend(error);lb.extend(lower)
                if p['outcome']=='POSITION_EXECUTABLE':valid_res.extend(error)
                if f['outcome']=='FULL6D_EXECUTABLE':six_count+=1
                if 'trajectory' in f:
                    with np.load(f['trajectory']['path']) as z:
                        orient.extend(z['ORIENTATION_RESIDUAL_RAD'].max(axis=1));six_res.extend(z['POSITION_CORRECTION_MM'].max(axis=1))
                    reasons=dict(CARTESIAN_FINAL_GATE=bool(f['metrics']['failed_frames']),ANGULAR_FINAL_GATE=bool(f['orientation_failed_frames']),
                        HARD_SELF_COLLISION=bool(f['geometry_counts']['HARD_SELF_COLLISION']),UNRESOLVED_GEOMETRY=bool(f['geometry_counts']['UNRESOLVED_GEOMETRY']),
                        PHYSICAL_TEMPORAL_GATE=not f['temporal']['pass_temporal'],JOINT_LIMIT=bool(f['metrics']['hard_limit_violations']),NONFINITE=not f['metrics']['finite'])
                    for k,v in reasons.items():six_reasons[k]=six_reasons.get(k,0)+int(v)
                records.append(dict(case=row,position_outcome=p['outcome'],full6d_outcome=f['outcome'],position_report=file_record(DEST/'position'/row['key']/'RESULT.json'),full6d_report=file_record(DEST/'full6d'/row['key']/'RESULT.json')))
            res=np.array(res);lb=np.array(lb);n=len(selected)
            complete_counts={}
            for row in selected:
                cr=read(DEST/'complete_action'/row['key']/'RESULT.json');co=cr['outcome'];complete_counts[co]=complete_counts.get(co,0)+1
            result[group][mode]=dict(episodes=n,frames=frames,position_outcomes=counts,executable_position=counts.get('POSITION_EXECUTABLE',0),full6d_executable=six_count,
                raw_numerical_within10_fraction=float(np.mean(res<=10)),framewise_geometry_valid_witness_fraction=witnessed/frames,
                certified_raw_unreachable_frames=int(np.sum(lb>10.00001)),certified_raw_unreachable_fraction=float(np.mean(lb>10.00001)),
                attempt_residual_mm=distribution(res),executable_correction_mm=distribution(valid_res),certified_raw_lower_bound_mm=distribution(lb),
                orientation_attempt_residual_rad=distribution(orient),full6d_attempt_cartesian_residual_mm=distribution(six_res),full6d_failure_component_counts=six_reasons,hard_candidate_frames=hard_frames,unresolved_candidate_frames=unresolved,proxy_only_candidate_frames=proxy,
                runtime_s=distribution(runtime),records=records,certified_executable_trajectory_infeasibility_count=0,preparation_hard_candidate_frames=prep_hard,adaptive_branch_discontinuities=branches,
                complete_action_outcomes=complete_counts,complete_action_executable=complete_counts.get('COMPLETE_ACTION_EXECUTABLE',0),
                aggregation='One value per source frame: maximum residual/lower bound across the two wrists. Prefix excluded from raw fidelity but included in executable temporal/geometry acceptance.',
                caveat='A framewise witness is not a temporally valid trajectory. A raw certificate does not prove absence of an allowed closest-feasible trajectory. Rejected-candidate residuals are not executable corrections.')
            collections[group][mode]=dict(res=res,lb=lb,valid=np.array(valid_res))
    return result,collections

def physical():
    rows=read(DEST/'REFERENCE_PHYSICS_COMPLETE.json')['results'];assert len(rows)==70
    assert all(r['outcome']!='INFRASTRUCTURE_INVALID' for r in rows)
    methods={m:sorted((r for r in rows if r['case']['representation_mode']==m),key=lambda r:r['case']['index']) for m in MODES}
    counts={m:{s:sum(r['outcomes'][s] for r in methods[m]) for s in STAGES} for m in MODES}
    for m in MODES:
        assert len(methods[m])==35
        assert [r['case']['index'] for r in methods[m]]==list(range(35))
        for r in methods[m]:assert all(not r['outcomes'][STAGES[k+1]] or r['outcomes'][STAGES[k]] for k in range(len(STAGES)-1))
    assert [r['case']['source_recording_id'] for r in methods['WRIST']]==[r['case']['source_recording_id'] for r in methods['INTERACTION']]
    a=np.array([r['outcomes']['FULL_TASK_SUCCESS'] for r in methods['WRIST']],int);b=np.array([r['outcomes']['FULL_TASK_SUCCESS'] for r in methods['INTERACTION']],int)
    afbs=int(((a==0)&(b==1)).sum());asbf=int(((a==1)&(b==0)).sum());discord=afbs+asbf
    boot=np.random.default_rng(1000).choice(b-a,(10000,35),replace=True).mean(axis=1)*100
    stats=dict(denominator=35,label='DEV35 DEVELOPMENT EVALUATION',counts=counts,
        full_task_ci95={m:ci(counts[m]['FULL_TASK_SUCCESS'],35) for m in MODES},
        paired_outcomes=dict(A_fail_B_fail=int(((a==0)&(b==0)).sum()),A_success_B_fail=asbf,A_fail_B_success=afbs,A_success_B_success=int(((a==1)&(b==1)).sum())),
        difference_percentage_points=float((b-a).mean()*100),paired_bootstrap95_percentage_points=np.quantile(boot,[.025,.975]).tolist(),
        bootstrap_seed=1000,bootstrap_replicates=10000,exact_mcnemar_pvalue=float(binomtest(afbs,discord,.5).pvalue) if discord else 1.,
        mcnemar_note='No discordant pairs; no evidence of a paired difference.' if not discord else 'Exact conditional two-sided binomial test on discordant pairs.',
        actual_physics_trace_counts={m:sum(bool(r.get('physical_trace')) for r in methods[m]) for m in MODES})
    stats['first_failure_class_counts']={m:{s:sum(r.get('failure_class',r['first_failure_stage'])==s for r in methods[m]) for s in sorted({r.get('failure_class',r['first_failure_stage']) for r in methods[m]})} for m in MODES}
    stats['stage_statistics']={}
    for stage in STAGES:
        sa=np.array([r['outcomes'][stage] for r in methods['WRIST']],int);sb=np.array([r['outcomes'][stage] for r in methods['INTERACTION']],int)
        ab=int(((sa==1)&(sb==0)).sum());ba=int(((sa==0)&(sb==1)).sum());nd=ab+ba
        boot=np.random.default_rng(1000).choice(sb-sa,(10000,35),replace=True).mean(axis=1)*100
        stats['stage_statistics'][stage]=dict(proportion_ci95={m:ci(counts[m][stage],35) for m in MODES},
            paired_table=dict(A_fail_B_fail=int(((sa==0)&(sb==0)).sum()),A_success_B_fail=ab,A_fail_B_success=ba,A_success_B_success=int(((sa==1)&(sb==1)).sum())),
            paired_difference_percentage_points=float((sb-sa).mean()*100),paired_bootstrap95_percentage_points=np.quantile(boot,[.025,.975]).tolist(),
            exact_mcnemar_pvalue=float(binomtest(ba,nd,.5).pvalue) if nd else 1.,inference_scope='Exploratory stage-wise statistics; no multiplicity-adjusted significance claim')
    return stats,methods

def paired_feasibility_statistics(feas):
    output={}
    for group in ('TRAIN40','DEV35'):
        aa=feas[group]['WRIST']['records'];bb=feas[group]['INTERACTION']['records']
        assert [r['case']['source_recording_id'] for r in aa]==[r['case']['source_recording_id'] for r in bb]
        output[group]={}
        for field,success in [('position_outcome','POSITION_EXECUTABLE'),('full6d_outcome','FULL6D_EXECUTABLE')]:
            a=np.array([r[field]==success for r in aa],int);b=np.array([r[field]==success for r in bb],int);n=len(a)
            ab=int(((a==1)&(b==0)).sum());ba=int(((a==0)&(b==1)).sum());nd=ab+ba
            boot=np.random.default_rng(1000).choice(b-a,(10000,n),replace=True).mean(axis=1)*100
            output[group][field]=dict(n=n,counts=dict(WRIST=int(a.sum()),INTERACTION=int(b.sum())),
                proportion_ci95=dict(WRIST=ci(int(a.sum()),n),INTERACTION=ci(int(b.sum()),n)),
                paired_table=dict(A_fail_B_fail=int(((a==0)&(b==0)).sum()),A_success_B_fail=ab,A_fail_B_success=ba,A_success_B_success=int(((a==1)&(b==1)).sum())),
                difference_percentage_points=float((b-a).mean()*100),paired_bootstrap95_percentage_points=np.quantile(boot,[.025,.975]).tolist(),
                exact_mcnemar_pvalue=float(binomtest(ba,nd,.5).pvalue) if nd else 1.,bootstrap_seed=1000,bootstrap_replicates=10000,
                interpretation='Descriptive/exploratory matched analysis conditional on the frozen procedure. TRAIN is calibration data and DEV35 is repeatedly inspected development data; no untouched-test or multiplicity-adjusted significance claim.')
    return output

def geometry_reason_audit(rows):
    reasons={};inputs=[];unexpected=[]
    allowed={('ValueError','ambiguous bidirectional solid-containment rays'),('RuntimeError','finite geometry triangle-pair budget exceeded'),('UNCERTAIN_SHELL_ENCLOSURE_OVERLAP',''),('NEAR_SURFACE_WITHOUT_STRICT_INTERIOR_WITNESS','')}
    for row in rows:
        result=read(DEST/'position'/row['key']/'RESULT.json');records=[('position',result['selected']['geometry'])]
        for stage in ('full6d','complete_action'):
            rr=read(DEST/stage/row['key']/'RESULT.json')
            if rr.get('geometry'):records.append((stage,rr['geometry']))
        for stage,record in records:
            inputs.append(record);g=read(record['path'])
            for phase,frames in g.items():
                for frame in frames:
                    for pair in frame['records']:
                        if pair['classification']!='UNRESOLVED_GEOMETRY':continue
                        for reason in pair.get('unresolved_reasons',[pair]):
                            k=(reason.get('reason','UNSPECIFIED'),reason.get('detail',''));label=' | '.join(k);reasons[label]=reasons.get(label,0)+1
                            if k not in allowed:unexpected.append(dict(case=row['key'],stage=stage,phase=phase,frame=frame['frame'],reason=reason))
    audit=dict(scope='All150 selected position candidates and every attempted6D/complete-action geometry; reason records may repeat within a frame',reason_counts=reasons,unexpected=unexpected,inputs=inputs,
        interpretation='Known fail-closed mesh/finite-budget ambiguity is not a confirmed hard collision or an infeasibility certificate. No tolerance, geometry budget or outcome is changed.')
    atomic_json(PAPER/'DETAILED_GEOMETRY_FAILURE_REASON_AUDIT.json',audit)
    if unexpected:raise RuntimeError('Unexpected geometry exception needs infrastructure review; do not silently count it as scientific failure')
    return file_record(PAPER/'DETAILED_GEOMETRY_FAILURE_REASON_AUDIT.json')

def draw_stages(ax,stats):
    names=['Executable','Approach','Grasp','Lift','Handoff','Ownership','Transport','Bin','Settle','Full task'];x=np.arange(len(names))
    for m,label,color,offset in zip(MODES,LABELS,COLORS,(-.18,.18)):
        values=[stats['counts'][m][s] for s in STAGES];bars=ax.bar(x+offset,values,.35,label=label,color=color);ax.bar_label(bars,fontsize=6,padding=2)
    ax.set(ylim=(0,35),ylabel='Cumulative episodes /35');ax.set_xticks(x,names,rotation=45,ha='right');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    if not sum(stats['actual_physics_trace_counts'].values()):
        ax.text(.5,.60,'No qualified complete trajectories\n0 physical trials; upstream feasibility failures',ha='center',va='center',transform=ax.transAxes,color='#555555',fontsize=9)
    intervals=stats['full_task_ci95']
    label='Full task:\n'+ '\n'.join(f"{letter} {stats['counts'][m]['FULL_TASK_SUCCESS']}/35, 95% CI {100*intervals[m][0]:.1f}–{100*intervals[m][1]:.1f}%" for letter,m in zip(('A','B'),MODES))
    ax.text(.02,.91,label+'\n'+f"B−A {stats['difference_percentage_points']:+.1f} pp",transform=ax.transAxes,fontsize=7,va='top')

def matrix(ax,methods,compact=False):
    classes=sorted({r.get('failure_class',r['first_failure_stage']) for rows in methods.values() for r in rows})
    codes={v:k for k,v in enumerate(classes)};values=np.array([[codes[r.get('failure_class',r['first_failure_stage'])] for r in methods[m]] for m in MODES])
    from matplotlib.colors import ListedColormap
    palette=['#b75c54','#a88a51','#817193','#577b98','#8a8a8a','#9c705e','#6e8790']
    cmap=ListedColormap([('#45936a' if v=='SUCCESS' else palette[k%len(palette)]) for v,k in codes.items()]);ax.imshow(values,aspect='auto',cmap=cmap,vmin=-.5,vmax=len(classes)-.5)
    ticks=[0,4,9,14,19,24,29,34] if compact else list(range(35))
    ax.set_yticks([0,1],['A','B']);ax.set_xticks(ticks,[f'{i+1:02}' for i in ticks],rotation=0 if compact else 90,fontsize=7)
    if compact:ax.set_box_aspect(.34)
    ax.set_xlabel('Matched DEV35 episode (source-manifest order)')
    from matplotlib.patches import Patch
    short=lambda v:v.replace('POSITION_','Position: ').replace('FULL6D_','6D: ').replace('NO_SOLUTION_WITHIN_COMMON_BUDGET','no solution within budget').replace('HARD_COLLISION','hard collision').replace('TEMPORAL_INVALID','temporal invalid').replace('_',' ')
    ax.legend(handles=[Patch(color=cmap(k),label=short(v)) for v,k in codes.items()],loc='upper center',bbox_to_anchor=(.5,-.38),ncol=1 if compact else 2,fontsize=7,frameon=False)

def main():
    PAPER.mkdir(parents=True,exist_ok=True);rows=cases();geometry_reason_audit(rows);feas,data=characterize(rows);stats,methods=physical()
    selection_path=OUT/'00_contract/AUTHORITATIVE_SPLIT_MANIFEST.json';ids=read(selection_path)['selection']['qualification_ids'];assert len(ids)==11
    diagnostic={}
    for mode in MODES:
        subset=[r for r in rows if r['group']=='TRAIN40' and r['representation_mode']==mode and r['index'] in ids];assert len(subset)==11
        outcomes={str(row['index']):read(DEST/'position'/row['key']/'RESULT.json')['outcome'] for row in subset}
        diagnostic[mode]=dict(executable=sum(x=='POSITION_EXECUTABLE' for x in outcomes.values()),denominator=11,outcomes=outcomes)
    atomic_json(PAPER/'FINAL_NUMERICAL_RESULTS.json',dict(feasibility=feas,paired_feasibility_statistics=paired_feasibility_statistics(feas),reference_dev35=stats,reference_episodes=methods,uniform_frozen_train11=diagnostic,train11_selection_source=file_record(selection_path)))
    episode_rows=[]
    for i in range(35):
        for mode in MODES:
            rr=methods[mode][i];case=rr['case'];pr=read(DEST/'position'/case['key']/'RESULT.json');sr=read(DEST/'full6d'/case['key']/'RESULT.json')
            episode_rows.append([i+1,case['source_recording_id'],mode,pr['outcome'],sr['outcome'],rr['outcome'],rr.get('failure_class',rr['first_failure_stage']),bool(rr.get('physical_trace')),*[int(rr['outcomes'][s]) for s in STAGES]])
    table('MATCHED_DEV35_EPISODE_OUTCOMES',['DEV index','Source recording','Representation','Position class','6D class','Reference class','First failure','Actual PhysX trace',*STAGES],episode_rows,'Every matched episode is retained. False later cumulative stages after retargeting failure are end-to-end outcomes, not fabricated physical observations.')
    table_rows=[]
    for group in ('TRAIN40','DEV35'):
        a,b=[feas[group][m] for m in MODES];n=a['episodes']
        for name,field,scale in [('Raw numerical residual ≤10 mm (%)','raw_numerical_within10_fraction',100),('Geometry-valid framewise witness (%)','framewise_geometry_valid_witness_fraction',100),('Certified raw-unreachable (%)','certified_raw_unreachable_fraction',100)]:
            av,bv=a[field]*scale,b[field]*scale;table_rows.append([group,name,f'{av:.2f}',f'{bv:.2f}',f'{bv-av:+.2f} pp'])
        for name,field in [('Executable position','executable_position'),('Executable 6D','full6d_executable'),('Complete G1 + Dex3 action executable','complete_action_executable')]:table_rows.append([group,name,f'{a[field]}/{n}',f'{b[field]}/{n}',f'{100*(b[field]-a[field])/n:+.2f} pp'])
        for metric in ('mean','p95','max'):
            values=[x['executable_correction_mm'][metric] if x['executable_correction_mm'] else None for x in (a,b)]
            table_rows.append([group,f'Qualified position correction {metric} (mm)',*['N/A — no qualified position' if v is None else f'{v:.3f}' for v in values],'N/A' if None in values else f'{values[1]-values[0]:+.3f}'])
        table_rows.append([group,'Hard-collision candidate frames',a['hard_candidate_frames'],b['hard_candidate_frames'],b['hard_candidate_frames']-a['hard_candidate_frames']])
        table_rows.append([group,'Preparation hard-collision candidate frames',a['preparation_hard_candidate_frames'],b['preparation_hard_candidate_frames'],b['preparation_hard_candidate_frames']-a['preparation_hard_candidate_frames']])
        table_rows.append([group,'Unresolved detailed-geometry source frames',a['unresolved_candidate_frames'],b['unresolved_candidate_frames'],b['unresolved_candidate_frames']-a['unresolved_candidate_frames']])
        for title,key in [('6D attempted Cartesian residual mean (mm)','full6d_attempt_cartesian_residual_mm'),('6D attempted orientation residual mean (rad)','orientation_attempt_residual_rad')]:
            values=[v[key]['mean'] if v[key] else None for v in (a,b)]
            table_rows.append([group,title,*['N/A — no 6D attempt' if x is None else f'{x:.4f}' for x in values],'N/A' if None in values else f'{values[1]-values[0]:+.4f}'])
        av,bv=[100*x['hard_candidate_frames']/x['frames'] for x in (a,b)]
        table_rows.append([group,'Hard-collision candidate frame fraction (%)',f'{av:.2f}',f'{bv:.2f}',f'{bv-av:+.2f} pp'])
        for metric in ('mean','p95','max'):
            av,bv=[x['runtime_s'][metric] for x in (a,b)]
            table_rows.append([group,f'Position wall time {metric} (s)',f'{av:.2f}',f'{bv:.2f}',f'{bv-av:+.2f}'])
    table('TABLE_1_RETARGETING_FEASIBILITY',['Split','Metric',*LABELS,'B−A'],table_rows,'Numerical raw residual is not a reachability proof. Certificates concern raw fidelity; failed searches do not certify trajectory impossibility. Correction rows contain qualified trajectories only, using the worse wrist per source frame (preparation excluded). Candidate collisions are not executed physical collisions. TRAIN11 A7/11 versus B10/11 remains prior diagnostic provenance, not this uniform frozen run.')
    stage_rows=[]
    for s in STAGES:
        a,b=[stats['counts'][m][s] for m in MODES];stage_rows.append([s,f'{a}/35 ({100*a/35:.1f}%)',f'{b}/35 ({100*b/35:.1f}%)',f'{100*(b-a)/35:+.1f} pp'])
    table('TABLE_2_REFERENCE_DEV35_PHYSICAL',['Metric',*LABELS,'B−A'],stage_rows,'DEV35 DEVELOPMENT EVALUATION. All cumulative denominators remain35, including retargeting failures without a physical rollout. Actual attempted physics counts: '+json.dumps(stats['actual_physics_trace_counts'])+'.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(8.2,6.4),layout='constrained')
    ax=axes[0,0];ax.axis('off');ax.set_title('(a) One controlled representation switch',loc='left',fontweight='bold')
    nodes=[(.5,.91,'Same source / event clock / registration'),(.27,.68,'A: WRIST'),(.73,.68,'B: INTERACTION'),(.5,.44,'Same frozen bounded G1 realization'),(.5,.22,'Same source-clock Dex3\nSame contact-constrained physics')]
    for x,y,text in nodes:ax.text(x,y,text,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#edf1f3',ec='#71808a'),transform=ax.transAxes,fontsize=8)
    for xy,xytext in [((.27,.74),(.46,.86)),((.73,.74),(.54,.86)),((.44,.51),(.27,.61)),((.56,.51),(.73,.61)),((.5,.29),(.5,.38))]:ax.annotate('',xy=xy,xytext=xytext,xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#4b5961'})
    ax.text(.5,.04,'Bounded failures retained; no individual rescue',ha='center',transform=ax.transAxes,fontsize=7)
    ax=axes[0,1];ax.set_title('(b) DEV35 retargeting feasibility',loc='left',fontweight='bold');keys=['framewise_geometry_valid_witness_fraction','certified_raw_unreachable_fraction','executable_position','full6d_executable'];names=['Raw ≤10 mm\ngeometry witness','Certified raw\nunreachable','Executable\nposition','Executable\n6D'];x=np.arange(4)
    for m,label,color,offset in zip(MODES,LABELS,COLORS,(-.18,.18)):
        r=feas['DEV35'][m];y=[100*r[k] if k.endswith('fraction') else 100*r[k]/35 for k in keys];bars=ax.bar(x+offset,y,.35,color=color,label=label);ax.bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
    ax.set_xticks(x,names,fontsize=7);ax.set(ylim=(0,110),ylabel='Frames (%) / episodes (%)');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    draw_stages(axes[1,0],stats);axes[1,0].set_title('(c) Reference cumulative end-to-end success',loc='left',fontweight='bold')
    matrix(axes[1,1],methods,compact=True);axes[1,1].set_title('(d) All35 matched first failures',loc='left',fontweight='bold')
    fig.suptitle('Single-variable ALOHA → G1\nDEV35 DEVELOPMENT EVALUATION',fontsize=12,fontweight='bold');save(fig,'Fig_Main_SingleVariable_AB',True)
    fig,ax=plt.subplots(figsize=(10,4));draw_stages(ax,stats);ax.set_title('Reference DEV35 development evaluation');save(fig,'Fig_DEV35_Stage_Success_AB')
    fig,ax=plt.subplots(figsize=(14,3.5),layout='constrained');matrix(ax,methods);ax.set_title('DEV35 development: matched first-failure classification');save(fig,'Fig_Matched_DEV35_Matrix')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,group in zip(axes,('TRAIN40','DEV35')):
        for m,label,color in zip(MODES,LABELS,COLORS):
            v=np.sort(data[group][m]['lb']);ax.plot(v,np.arange(1,len(v)+1)/len(v),label=label,color=color)
        ax.axvline(10,color='black',ls='--',lw=1);ax.set(title=group+' raw morphology certificate',xlabel='Certified raw residual lower bound (mm)',ylabel='Empirical CDF');ax.legend();ax.grid(alpha=.2)
    save(fig,'Fig_Raw_Cartesian_Feasibility_AB')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,group in zip(axes,('TRAIN40','DEV35')):
        any_valid=False
        for m,label,color in zip(MODES,LABELS,COLORS):
            v=np.sort(data[group][m]['valid'])
            if len(v):any_valid=True;ax.plot(v,np.arange(1,len(v)+1)/len(v),label=label,color=color)
        if not any_valid:ax.text(.5,.5,'No qualified position trajectories\nNo executable-correction distribution',ha='center',va='center',transform=ax.transAxes)
        else:
            ax.legend()
            missing=[label for m,label in zip(MODES,LABELS) if not len(data[group][m]['valid'])]
            if missing:ax.text(.03,.08,'; '.join(missing)+': no qualified trajectory',transform=ax.transAxes,fontsize=8)
        ax.set(title=group,xlabel='Qualified position correction (mm)',ylabel='Empirical CDF');ax.grid(alpha=.2)
    save(fig,'Fig_Morphology_Correction_Distribution')
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained');classes=sorted({r.get('failure_class',r['first_failure_stage']) for rows2 in methods.values() for r in rows2});x=np.arange(len(classes))
    for m,label,color,offset in zip(MODES,LABELS,COLORS,(-.18,.18)):ax.bar(x+offset,[sum(r.get('failure_class',r['first_failure_stage'])==k for r in methods[m]) for k in classes],.35,label=label,color=color)
    ax.set_xticks(x,[s.replace('_','\n') for s in classes],fontsize=8);ax.set_ylabel('Episodes /35');ax.legend();ax.set_title('DEV35 first failure: failures are not excluded');save(fig,'Fig_DEV35_First_Failure_Stage')
    # Objective matched examples; no visually selected cases.
    categories={}
    for i,(a,b) in enumerate(zip(methods['WRIST'],methods['INTERACTION'])):
        key=('success' if a['outcomes']['FULL_TASK_SUCCESS'] else 'fail')+'_'+('success' if b['outcomes']['FULL_TASK_SUCCESS'] else 'fail')
        categories.setdefault(key,i)
    atomic_json(PAPER/'OBJECTIVE_REPRESENTATIVE_SELECTION.json',dict(rule='Lowest manifest index in each observed paired success/failure category; absent categories not fabricated',categories=categories))
    atomic_json(PAPER/'ANALYSIS_MANIFEST.json',dict(results=file_record(PAPER/'FINAL_NUMERICAL_RESULTS.json'),implementation=file_record(Path(__file__)),artifacts=[file_record(p) for p in sorted(PAPER.glob('*')) if p.is_file() and p.name!='ANALYSIS_MANIFEST.json']))
    print('PAPER_ANALYSIS_COMPLETE',json.dumps(stats),flush=True)

if __name__=='__main__':main()
