"""Verified three-layer summaries; absence of physics never becomes observed0%."""
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
from .score import STAGES

COLORS={'WRIST':'#4779ad','INTERACTION':'#df813f'}
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-r),float(c+r)]
def summary(rows):
    actual=[r for r in rows if r.get('actual_physics_executed')];valid=[r for r in actual if r.get('physical_valid')]
    physical={s:sum(r['physical_stages'][s] is True for r in valid) for s in STAGES}
    cumulative={s:sum(r['cumulative_pipeline'][s] is True for r in rows) for s in STAGES}
    unknown={s:sum(r['cumulative_pipeline'][s] is None for r in rows) for s in STAGES}
    return dict(cases=len(rows),actual_rollouts=len(actual),valid_rollouts=len(valid),invalid_or_incomplete=len(actual)-len(valid),not_executed=len(rows)-len(actual),
        observed_stage_counts=physical,cumulative_stage_counts=cumulative,unknown_stage_counts=unknown,
        observed_tsr=physical['FULL_TASK_SUCCESS']/len(valid) if valid else None,observed_tsr_label='MEASURED' if valid else 'NOT MEASURED',observed_tsr_wilson95=wilson(physical['FULL_TASK_SUCCESS'],len(valid)))
def main():
    cases=read(PRIOR/'CASE_MANIFEST.json');construction=[]
    for row in cases:
        p=RUN/'construction'/row['key']/'RESULT.json'
        if p.exists():construction.append(read(p))
    reference=[]
    for row in cases:
        if row['group']!='DEV35':continue
        p=RUN/'reference_physics'/row['key']/'RESULT.json'
        if p.exists():reference.append(read(p))
    methods={m:summary([r for r in reference if r['case']['representation_mode']==m]) for m in COLORS}
    ctables=[];ftables=[];counts={}
    for group in ('TRAIN40','DEV35'):
        for mode in COLORS:
            rows=[r for r in construction if r['case']['group']==group and r['case']['representation_mode']==mode];selected=[r['selected'] for r in rows if r.get('selected')]
            counts[f'{group}_{mode}']=dict(completed=len(rows),total=40 if group=='TRAIN40' else 35,executable=len(selected),no_candidate=sum(r['outcome']=='NO_EXECUTABLE_TRAJECTORY_UNDER_FIXED_PROTOCOL' for r in rows),infrastructure_invalid=sum(r['outcome']=='INFRASTRUCTURE_INVALID' for r in rows))
            for r in rows:
                sel=r.get('selected');best=sel or min(r.get('candidates',[]),key=lambda x:x['tracking_cost'],default=None)
                ctables.append(dict(group=group,method=mode,index=r['case']['index'],recording=r['case']['source_recording_id'],outcome=r['outcome'],execution_valid=sel is not None,
                    fidelity_candidate='selected_valid' if sel else 'best_attempt_diagnostic_only',position_mean_mm=best['position_mm']['mean'] if best else None,position_p95_mm=best['position_mm']['p95'] if best else None,position_max_mm=best['position_mm']['max'] if best else None,
                    raw_10mm_fraction=best['raw_frame_fraction_10mm'] if best else None,orientation_mean_rad=best['orientation_rad']['mean'] if best else None,orientation_max_rad=best['orientation_rad']['max'] if best else None,full6d_invocations=r.get('full6d_invocations'),selected_family=sel['family'] if sel else None,selected_candidate=sel['candidate'] if sel else None))
    for r in reference:
        row=dict(method=r['case']['representation_mode'],index=r['case']['index'],recording=r['case']['source_recording_id'],actual_physics_executed=r.get('actual_physics_executed'),physical_valid=r.get('physical_valid'),outcome=r['outcome'],first_failure_stage=r['first_failure_stage'],release_classification=r.get('release_classification','NOT_ATTEMPTED'))
        for s in STAGES:
            row['observed_'+s]=r['physical_stages'][s] if r['physical_stages'][s] is not None else 'UNKNOWN'
            row['cumulative_'+s]=r['cumulative_pipeline'][s] if r['cumulative_pipeline'][s] is not None else 'UNKNOWN'
        ftables.append(row)
    csvsave(RUN/'TABLE_REFERENCE_FIDELITY_AND_EXECUTABILITY.csv',ctables);csvsave(RUN/'TABLE_REFERENCE_PHYSICAL_OUTCOMES.csv',ftables)
    matched=[]
    for i in range(35):
        pair=[next((r for r in reference if r['case']['index']==i and r['case']['representation_mode']==m),None) for m in COLORS]
        if all(pair):
            vals=[r['cumulative_pipeline']['FULL_TASK_SUCCESS'] for r in pair]
            matched.append(dict(index=i,A=vals[0],B=vals[1],both_actually_executed=all(r.get('physical_valid') for r in pair)))
    known=[x for x in matched if x['A'] is not None and x['B'] is not None];matrix=[[sum(x['A']==a and x['B']==b for x in known) for b in (False,True)] for a in (False,True)]
    disc=matrix[0][1]+matrix[1][0];p=float(binomtest(matrix[0][1],disc,.5).pvalue) if disc else 1.
    if known:
        values=np.array([int(x['B'])-int(x['A']) for x in known]);rng=np.random.default_rng(20260907);boot=np.mean(rng.choice(values,(10000,len(values))),axis=1)*100;ci=np.quantile(boot,[.025,.975]).tolist();effect=float(values.mean()*100)
    else:ci=None;effect=None
    stats=dict(known_matched_pipeline_cases=len(known),unknown_matched_cases=35-len(known),paired_pipeline_matrix=matrix,exact_mcnemar_p=p if known else None,paired_pipeline_effect_pp=effect,paired_bootstrap95_pp=ci,
        interpretation='Cumulative pipeline outcomes, not observed physical A/B success unless both actually executed. Unknown runs excluded explicitly.',both_physically_executed_pairs=sum(x['both_actually_executed'] for x in matched))
    report=dict(created_at=now(),construction_counts=counts,reference=methods,matched_statistics=stats,construction_completed=len(construction),reference_cases_completed=len(reference),reference_episodes=reference)
    save(RUN/'analysis/RESULTS.json',report)
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained');ax=axs[0,0];ax.axis('off');ax.set_title('(a) Single representation switch',loc='left')
    ax.text(.03,.85,'ALOHA source + common event clock + registration\n\nA WRIST target   /   B INTERACTION target\n\nSame bounded G1 solver + candidate validity\n\nSame0.700s natural prefix + Dex3 + PhysX\n\nFidelity ≠ execution validity ≠ task outcome',va='top',fontsize=12)
    ax=axs[0,1];ax.set_title('(b) Fidelity versus execution validity — DEV35',loc='left')
    for k,m in enumerate(COLORS):
        rows=[r for r in ctables if r['group']=='DEV35' and r['method']==m];valid=[r for r in rows if r['execution_valid']];bad=[r for r in rows if not r['execution_valid']]
        for values,marker,alpha in [(valid,'o',1),(bad,'x',.45)]:
            ax.scatter([r['position_mean_mm'] for r in values],[k+.09*np.sin(r['index']) for r in values],c=COLORS[m],marker=marker,alpha=alpha)
    ax.axvline(10,color='gray',ls='--',lw=1,label='10mm diagnostic');ax.set_yticks([0,1],['A WRIST','B INTERACTION']);ax.set_xlabel('Mean raw position residual (mm); ○ executable, × invalid')
    ax=axs[1,0];ax.set_title('(c) Observed physical stages',loc='left');labels=['Grasp','Lift','Handoff','Ownership','Transport','Bin','Settle','Full'];st=STAGES[2:]
    for k,m in enumerate(COLORS):
        v=methods[m];n=v['valid_rollouts'];x=np.arange(len(st))+(k-.5)*.36
        if n:ax.bar(x,[v['observed_stage_counts'][s]/n*100 for s in st],.36,color=COLORS[m],label=f'{m}: valid executed n={n}')
        else:ax.plot([],[],color=COLORS[m],label=f'{m}: NOT MEASURED (n=0)')
    ax.set_xticks(np.arange(8),labels,rotation=30,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Observed completion (%)');ax.legend(fontsize=8)
    ax=axs[1,1];ax.set_title('(d) Matched development cases',loc='left');grid=np.full((2,35),-1.)
    for k,m in enumerate(COLORS):
        for r in reference:
            if r['case']['representation_mode']!=m:continue
            grid[k,r['case']['index']]=0 if not r.get('actual_physics_executed') else 1 if not r.get('physical_valid') else 3 if r['physical_stages']['FULL_TASK_SUCCESS'] else 2
    from matplotlib.colors import ListedColormap,BoundaryNorm
    cmap=ListedColormap(['#ffffff','#cccccc','#a679b4','#d79973','#69a38c']);ax.imshow(grid,aspect='auto',cmap=cmap,norm=BoundaryNorm(np.arange(-1.5,4.5),5));ax.set_yticks([0,1],['A','B']);ax.set_xticks(range(0,35,4),[str(x+1) for x in range(0,35,4)]);ax.set_xlabel('Gray: not executed; purple: invalid; orange: task fail; green: success',fontsize=8)
    fig.suptitle('ALOHA → G1: reconciled A/B evaluation\nDEV35 DEVELOPMENT EVALUATION — no untouched-test claim',fontsize=15)
    for ext in ('png','pdf','svg'):
        pth=RUN/f'figures/Fig_AB_Reconciled_Physical_Evaluation.{ext}';pth.parent.mkdir(parents=True,exist_ok=True);fig.savefig(pth,dpi=200)
    plt.close(fig)
    for name,component in [('Fig_Reference_Stage_Completion','stages'),('Fig_Reference_Correction_Distribution','correction')]:
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        if component=='correction':
            for m in COLORS:
                vals=[r['position_mean_mm'] for r in ctables if r['group']=='DEV35' and r['method']==m and r['position_mean_mm'] is not None];ax.hist(vals,bins=20,alpha=.55,color=COLORS[m],label=m)
            ax.set_xlabel('Per-episode mean raw correction (mm), best available diagnostic');ax.set_ylabel('Episodes')
        else:
            for m in COLORS:ax.plot(range(len(STAGES)),[methods[m]['cumulative_stage_counts'][s] for s in STAGES],'-o',color=COLORS[m],label=m)
            ax.set_xticks(range(len(STAGES)),['Executable','Approach',*labels],rotation=30,ha='right');ax.set_ylabel('Cumulative completions /35 (unknowns disclosed)');ax.set_ylim(0,35)
        ax.legend();ax.set_title('DEV35 development evaluation');fig.savefig(RUN/f'figures/{name}.png',dpi=200);plt.close(fig)
    print('ANALYSIS',len(construction),'/150 construction',len(reference),'/70 reference records','actual',sum(x['actual_rollouts'] for x in methods.values()),flush=True)

if __name__=='__main__':main()
