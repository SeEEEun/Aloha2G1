"""Editable plots from verified current-study numeric accounting."""
from pathlib import Path
import numpy as np
from .io import read, record, atomic_json, atomic_text
from .paper_results import STAGES


def generate(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    n = read(out/'PAPER_NUMERIC_SUMMARY.json')
    for dep in n['evidence']:
        if record(dep['path']) != dep:
            raise ValueError('Numeric inputs changed; rerun analysis before plotting')
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    products = []
    colors = {'A':'#bb6b36', 'B':'#217b85'}
    def save(fig, name):
        for suffix in ('png','svg','pdf'):
            path = folder/(name+'.'+suffix)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            products.append(record(path))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(16,4.5))
    ax.axis('off')
    actual = n['matched_counts']['paired_usable'] if n['matched_counts'] else 'pending'
    trained=(out/'act_training/ACT_TRAINING_COMPLETE.json').exists()
    policy_executed=any(n['ACT']['conditions'][c]['recorded'] for c in ('A','B'))
    labels = [(0.08,.5,'ALOHA TRAIN40\nRegistered scene / TCP\nEvents + uncertainty'),
              (.31,.72,'A: Wrist reference\nCalibrated spatial prior'),
              (.31,.28,'B: Interaction\nPhase/contact goals\nCoupled handoff'),
              (.54,.5,'Common G1 planning\nRetiming + dynamic physics\nRGB / measured state / action'),
              (.77,.5,f'Matched G1 sources: {actual}\nSame ACT training protocol\n'+('Selected ACT-A / ACT-B' if trained else 'Policies not trained')),
              (1.0,.5,'ACT-A / ACT-B\nMatched DEV35 physics\n'+('Recorded policy outcomes' if policy_executed else 'NOT EXECUTED'))]
    for i,(x,y,label) in enumerate(labels):
        pending=(i==4 and not trained) or (i==5 and not policy_executed)
        ax.text(x,y,label,ha='center',va='center',fontsize=9,
                bbox=dict(boxstyle='round,pad=.65',fc='#f4f4f4' if pending else '#edf1f3',
                          ec='#53636b',linestyle='--' if pending else '-'))
    for x0,y0,x1,y1 in [(.17,.52,.23,.72),(.17,.48,.23,.28),(.40,.72,.45,.54),(.40,.28,.45,.46),(.645,.5,.675,.5),(.865,.5,.925,.5)]:
        ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='->',color='#53636b'))
    ax.text(.53,.015,f'Demonstrated: one TRAIN full task. Frozen generation: {n["generation_recorded"]}/80 attempts; paired ACT sources = {actual}.\nDashed stages have not been executed.',ha='center',fontsize=10)
    ax.set_title('Hybrid retargeting and the required target-domain ACT study',pad=18)
    ax.set_xlim(-.05,1.12);ax.set_ylim(-.1,1.05)
    save(fig,'Fig1_METHOD_PIPELINE')

    fig = plt.figure(figsize=(13,6))
    grid = fig.add_gridspec(2,2,width_ratios=[2.4,1],height_ratios=[1,1.4])
    primary = n['ACT'];summary = fig.add_subplot(grid[0,0]);summary.axis('off')
    lines = []
    for c in ('A','B'):
        p = primary['conditions'][c]
        value = f"{p['full_task_successes']}/35 = {100*p['full_task_rate']:.1f}%" if p['full_task_rate'] is not None else p['primary_status'].replace('_',' ')
        lines.append(f'ACT-{c} FULL TASK: {value}\n{p["valid_physical_rollouts"]} valid rollouts; {p["not_attempted"]} not attempted; {p["infrastructure_unknown"]} infrastructure unknown')
    difference = primary['paired'].get('difference_pp')
    summary.text(0, .98, '\n\n'.join(lines), va='top', fontsize=13)
    summary.text(0,.04,'B − A: '+('NOT ESTIMABLE' if difference is None else f'{difference:.1f} percentage points'),fontsize=14,weight='bold')
    stage_ax = fig.add_subplot(grid[1,0])
    if all(primary['conditions'][c]['resolved']==35 for c in ('A','B')):
        x=np.arange(len(STAGES))
        for shift,c in [(-.18,'A'),(.18,'B')]:
            stage_ax.bar(x+shift,[primary['conditions'][c]['cumulative_valid_stage_counts'][s] for s in STAGES],.36,color=colors[c],label='ACT-'+c)
        stage_ax.set_ylim(0,35);stage_ax.set_ylabel('Cumulative valid completion / 35')
        stage_ax.set_xticks(x,['Grasp','Lift','Handoff','Ownership','Transport','Entry','Settle','Full'],rotation=25)
        stage_ax.legend()
    else:
        stage_ax.axis('off')
        stage_ax.text(.5,.58,'Policy stage success NOT MEASURED\nNo reference score substituted',ha='center',va='center',fontsize=16)
    matrix_ax=fig.add_subplot(grid[:,1]);keyed={(r['source_id'],r['condition']):r for r in primary['per_source']}
    matrix=np.zeros((35,2),int)
    for i,sid in enumerate(primary['source_units']):
        for j,c in enumerate(('A','B')):
            row=keyed[sid,c]
            matrix[i,j]=(0 if row['terminal']=='NOT_ATTEMPTED_UPSTREAM' else 1 if row['full_task_success'] is None else 3 if row['full_task_success'] else 2)
    cmap=ListedColormap(['#dedede','#dcc58d','#c78474','#65aaa0'])
    matrix_ax.imshow(matrix,aspect='auto',cmap=cmap,norm=BoundaryNorm(np.arange(-.5,4.5),4))
    matrix_ax.set_xticks([0,1],['ACT-A','ACT-B']);matrix_ax.set_yticks([0,4,9,14,19,24,29,34],[1,5,10,15,20,25,30,35]);matrix_ax.set_ylabel('Ordered DEV35 source row')
    matrix_ax.set_title('All matched source instances',fontsize=11)
    fig.legend(handles=[Patch(facecolor=cmap(i),label=l) for i,l in enumerate(['Not attempted','Unknown','Observed failure/abort','Success'])],loc='lower center',ncol=4,fontsize=9)
    fig.suptitle('Primary ACT-A versus ACT-B — DEV35 development evaluation')
    fig.subplots_adjust(bottom=.13,top=.89,hspace=.15,wspace=.3)
    save(fig,'Fig2_MAIN_ACT_RESULT')

    fig, axes_grid=plt.subplots(2,2,figsize=(14,8.5))
    axes=axes_grid.ravel()
    for i,(c,label) in enumerate([('B','Ours'),('B_NO_COUPLING','Coupling off')]):
        r=n['reference'][c]
        axes[0].bar(i,r['complete_valid_plans'],color=colors['B'] if c=='B' else '#797f90')
        axes[0].text(i,r['complete_valid_plans']+.2,f"{r['complete_valid_plans']}/10",ha='center')
    axes[0].set_xticks([0,1],['Ours','Coupling off']);axes[0].set_ylim(0,11);axes[0].set_ylabel('Complete valid reference plans / 10')
    axes[0].set_title(f"{n['reference_recorded']}/20 frozen reference attempts recorded")
    paired=n['reference_paired'];physical_n=paired.get('paired_physical_N',0)
    stage_x=np.arange(2)
    for shift,c,label,color in [(-.18,'B','Ours',colors['B']),(.18,'B_NO_COUPLING','Coupling off','#797f90')]:
        values=[n['reference'][c]['cumulative_valid_stage_counts'][s] for s in ('HANDOFF','FULL_TASK')]
        axes[1].bar(stage_x+shift,values,.36,label=label,color=color)
        for x,value in zip(stage_x+shift,values):axes[1].text(x,value+.16,str(value),ha='center',fontsize=9)
    axes[1].set_xticks(stage_x,['Handoff','Full task']);axes[1].set_ylim(0,11)
    axes[1].set_ylabel('Cumulative valid completions / 10 scheduled')
    axes[1].set_title(f"Valid rollouts: Ours {n['reference']['B']['valid_physical_rollouts']}, off {n['reference']['B_NO_COUPLING']['valid_physical_rollouts']}\nPaired valid physical N = {physical_n}",fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].text(.5,-.2,('Paired physical effect NOT MEASURED' if not physical_n else 'Exploratory reference-level comparison')+'\nNo learned-policy coupling claim',ha='center',transform=axes[1].transAxes,fontsize=8)
    geometry=n.get('reference_geometry',[])
    available=sorted({r['source_id'] for r in geometry if all(any(x['source_id']==r['source_id'] and x['condition']==c and x['contact_candidate']==2 and x['seed']==0 for x in geometry) for c in ('B','B_NO_COUPLING'))})
    if available:
        sid=available[0]
        rows=[next(r for r in geometry if r['source_id']==sid and r['condition']==c and r['contact_candidate']==2 and r['seed']==0) for c in ('B','B_NO_COUPLING')]
        axes[2].bar([0,1],[r['position_disagreement_mm'] for r in rows],color=[colors['B'],'#797f90'])
        axes[2].set_xticks([0,1],['Ours','Coupling off']);axes[2].set_ylabel('Predicted object disagreement (mm)')
        axes[2].set_title('Illustration: contact2 / seed0\n'+sid,fontsize=9)
        axes[2].text(.5,-.22,'Lowest source ID with both saved fits;\nnot a physical measurement or typical-case estimate',ha='center',transform=axes[2].transAxes,fontsize=8)
    else:
        axes[2].axis('off');axes[2].text(.5,.55,'No matched completed handoff fit\nGeometric difference unavailable',ha='center',va='center',fontsize=12)
    reference_contract=read(out/'reference_coupling10/CONTRACT.json')
    reference_rows=read(out/'reference_coupling10/LEDGER.json')['rows']
    keyed={(r['source_id'],r['condition']):r for r in reference_rows}
    matrix=np.zeros((10,2),int)
    for i,sid in enumerate(reference_contract['source_ids']):
        for j,c in enumerate(('B','B_NO_COUPLING')):
            row=keyed.get((sid,c),{})
            matrix[i,j]=(0 if not row.get('physical_run') else 1 if not row.get('physical_validity')
                         else 3 if row.get('task_success') else 2)
    cmap=ListedColormap(['#dedede','#dcc58d','#c78474','#65aaa0'])
    axes[3].imshow(matrix,aspect='auto',cmap=cmap,norm=BoundaryNorm(np.arange(-.5,4.5),4))
    axes[3].set_xticks([0,1],['Ours','Coupling off'])
    axes[3].set_yticks(np.arange(10),[sid.removeprefix('GoPark_') for sid in reference_contract['source_ids']],fontsize=8)
    axes[3].set_title('All ten predeclared matched source instances',fontsize=11)
    axes[3].legend(handles=[Patch(facecolor=cmap(i),label=l) for i,l in enumerate(
        ['No physical rollout','Physical validity failure','Valid task failure','Valid full success'])],
        loc='upper center',bbox_to_anchor=(.5,-.1),ncol=2,fontsize=8)
    fig.suptitle('Same Ours implementation, coupling factors enabled / disabled')
    fig.tight_layout(rect=(0,.02,1,.95),h_pad=3.5);save(fig,'Fig3_COUPLING_ATTRIBUTION')
    result=dict(status='CURRENT_NUMERIC_FIGURES_GENERATED',numeric_summary=record(out/'PAPER_NUMERIC_SUMMARY.json'),
                plotting_code=record(__file__),products=products,ACT_results_substituted=False)
    atomic_json(folder/'CURRENT_FIGURE_PROVENANCE.json',result)
    atomic_text(out/'PAPER_FIGURE_DIRECTION.md',f'''# 그림 배치와 주장 범위

1. Methods: `{folder}/Fig1_METHOD_PIPELINE.png` 및 `.svg`/`.pdf`. 원본 → A/B 변환 → 실제 G1 자료 → 동일 ACT 학습 → 대응 DEV35 순서를 유지한다. 구현된 연결과 실제 완료한 실험을 구분한다.
   English caption: “Source-conditioned Wrist and Interaction converters share G1 calibration, connection, retiming, demonstration control and physics. Real target-domain RGB, measured state and executed actions define the matched ACT learning interface. Frozen generation produced {actual} paired usable sources; dashed stages have not been executed. The full-task TRAIN development milestone is reference evidence; it is not an ACT rollout. Demonstration generation uses privileged simulator state. No real-robot or VLA claim is made.”

2. Main Results: `{folder}/Fig2_MAIN_ACT_RESULT.png` 및 `.svg`/`.pdf`, `TABLE_ACT_DEV35_RESULTS.csv`, `PER_EPISODE_ACT_RESULTS.csv`. Full Task, B−A, 단계별 누적 결과, 전체35 대응 행렬 순서. NOT MEASURED를0%로 바꾸거나 기준 궤적 결과를 넣지 않는다. 회색은 미실행이다.
   English caption: “Primary ACT-A versus ACT-B accounting for all35 matched DEV35 development instances per policy. Rates, paired uncertainty and stage counts are shown only when supported by actual policy outcomes. Unexecuted and infrastructure-unknown cells remain distinct from observed failures. Reference conversion results never substitute for policy task success. DEV35 is not untouched testing.”

3. Separate reference diagnostic: `{folder}/Fig3_COUPLING_ATTRIBUTION.png` 및 `.svg`/`.pdf`, `TABLE_COUPLING_ABLATION.csv`. 계획 → 누적 유효 단계 → 예측 기하 → 전체10개 대응 행렬 순서로 배치한다. 사전 고정10개 원본과20개 기준 변환 시도만 사용한다. 실제 실행 표본 수를 유지하고 계획 실패를 관측된 전달 실패로 해석하지 않는다.
   English caption: “Exploratory reference-level paired10 comparison of the same Interaction converter with only cross-hand consistency factors enabled or disabled. Candidate banks, unary priors, scenes, planner, controller and budgets are shared. Complete plans were {n['reference']['B']['complete_valid_plans']}/10 for Ours and {n['reference']['B_NO_COUPLING']['complete_valid_plans']}/10 for coupling off; paired valid physical N={physical_n}. Planning and physical denominators are separate. The geometric illustration uses the lowest source ID with both saved fits, at contact candidate2 and seed0; it is a predicted fit rather than a physical measurement or typical-case estimate. The source-conditioned scenes and demonstration controller use privileged simulated state. Stage bars count cumulative valid completion per10 scheduled instances; no-plan cases have unattempted physical stages in the ledger. The matrix retains all ten matched sources. This experiment does not estimate the effect of coupling after ACT learning, real-robot performance, or VLA learning.”

Development illustration: `{out}/replays/FIRST_SOURCE_CONDITIONED_FULL_TASK.mp4`. 고정 TRAIN 원본에서 최초의 유효 전체 과제 시도를 보여준다. 이전 실패는 보존되며, 일반적인 성공률을 추정하는 대표 표본으로 부르지 않는다. 실제 측정 G1/물체 상태와 물리 시간을 사용한다.
''')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    print(generate(p.parse_args().run_dir)['status'])
