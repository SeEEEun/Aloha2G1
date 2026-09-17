"""Partial-study figures and actual measured replays, never invented outcomes."""
from pathlib import Path
import numpy as np
from .io import ROOT,read,record,atomic_json,atomic_text


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    n=read(out/'NUMERIC_SUMMARY.json');folder=out/'figures';folder.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ('png','svg','pdf'):fig.savefig(folder/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4));ax.axis('off')
    blocks=[('ALOHA source\nactual IDs / events\nregistered object scene',.08,.56,True),
        ('A: Wrist reference\nB: Interaction goals\ncommon connection / timing\nPARTIAL TRAIN INTEGRATION',.31,.56,False),
        ('Matched dynamic G1 data\ncurrent RGB + measured state\nexecuted commands\nNOT YET AVAILABLE',.56,.56,False),
        ('Same ACT training\nACT-A / ACT-B\nNOT YET RUN',.78,.56,False),
        ('Closed-loop G1\nACT-A35 / ACT-B35\nNOT MEASURED',.96,.56,False)]
    for text,x,y,done in blocks:
        ax.text(x,y,text,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.5',fc='#e1eeee' if done else '#eeeeee',ec='#536878'))
    for x0,x1 in [(0.17,.22),(.405,.445),(.665,.70),(.855,.89)]:ax.annotate('',xy=(x1,.56),xytext=(x0,.56),arrowprops=dict(arrowstyle='->',color='#536878'))
    ax.text(.52,.15,'Implemented evidence: source acquisition/lift/release + actual legacy-checkpoint interface diagnostics.\nFull source task, matched datasets and the primary ACT study remain unresolved.',ha='center',fontsize=10)
    ax.set_xlim(-.04,1.075);ax.set_title('Required converter → target-robot demonstrations → ACT study',fontsize=14)
    save(fig,'Fig1_Method')
    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.4,1]})
    axes[0].axis('off');axes[0].text(.5,.76,'ACT-A FULL TASK SUCCESS\nNOT MEASURED\n\nACT-B FULL TASK SUCCESS\nNOT MEASURED',ha='center',va='center',fontsize=17,weight='bold')
    axes[0].text(.5,.2,'35 intended cases per policy; 0 primary physical trials\nB − A: NOT ESTIMABLE\nAll physical stage counts unavailable',ha='center',fontsize=11)
    axes[1].imshow(np.zeros((35,2)),aspect='auto',cmap=ListedColormap(['#e7e7e7']),vmin=0,vmax=1)
    axes[1].set_xticks([0,1],['ACT-A','ACT-B']);axes[1].set_yticks([0,4,9,14,19,24,29,34],['1','5','10','15','20','25','30','35'])
    axes[1].set_ylabel('Ordered DEV35 source instance');axes[1].set_title('All70 intended outcomes: NOT_ATTEMPTED_UPSTREAM',fontsize=9)
    fig.suptitle('PRIMARY ACT TASK SUCCESS — policy comparison unavailable',fontsize=14)
    fig.text(.5,.02,'Grey is unobserved, not failure. DEV35 is development evaluation. Short legacy TRAIN diagnostics are excluded.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94));save(fig,'Fig2_PRIMARY_ACT_TASK_SUCCESS')
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].axis('off')
    axes[0].text(.5,.65,'REFERENCE-LEVEL PAIRED10\nPhysical paired N=0\nHandoff / Full Task: NOT MEASURED',ha='center',va='center',fontsize=14)
    axes[0].text(.5,.25,'No policy-learning attribution\nNo ACT-no-coupling policy trained',ha='center',fontsize=10)
    c=n['train_geometric_coupling'];x=np.arange(2)
    axes[1].bar(x-.18,np.asarray(c['on_position_disagreement_m'])*1000,.36,label='Coupling on',color='#277d88')
    axes[1].bar(x+.18,np.asarray(c['off_position_disagreement_m'])*1000,.36,label='Coupling off',color='#bd7949')
    axes[1].set_xticks(x,['Seed0','Seed1']);axes[1].set_ylabel('Predicted handoff object disagreement (mm)');axes[1].legend(fontsize=9)
    axes[1].set_title('Separate TRAIN algebraic diagnostic: one source',fontsize=10);axes[1].spines[['top','right']].set_visible(False)
    fig.suptitle('Reference coupling ablation unavailable; bounded mechanism diagnostic only')
    fig.text(.5,.02,'Same contact, seeds, unary terms and budget. Seeds are not independent episodes. This is not physical coupling evidence.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93));save(fig,'Fig3_Reference_Coupling_Ablation')
    path=out/'prototype'/n['source_id']/'morphology_acquisition_v4/physics_attempt_01/event_log.npz';a=dict(np.load(path));t=a['control_frame']/30.+(a['physics_step']%8)/240.
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
    axes[0].plot(t,a['object_position_world_m'][:,2],color='#277d88');axes[0].set_ylabel('Measured object z (m)');axes[0].set_title('Actual source-conditioned acquisition, lift, retention and natural release')
    for digit,color in [('thumb','#277d88'),('index','#bd7949'),('middle','#77528d')]:axes[1].plot(t,a[f'left_{digit}_force_n'],lw=.8,label=digit,color=color)
    axes[1].set_xlabel('Physical time (s)');axes[1].set_ylabel('Measured doll contact (N)');axes[1].legend();fig.tight_layout();save(fig,'SOURCE_ACQUISITION_MEASURED_EVIDENCE')
    atomic_json(folder/'FIGURE_PROVENANCE.json',dict(generator=record(__file__),summary=record(out/'NUMERIC_SUMMARY.json'),source_trace=record(path),primary_ACT_outcomes_available=False,physical_ablation_available=False))
    return dict(figures=4,primary_result='NOT_MEASURED')


def closeup(out):
    import cv2,mujoco
    from scipy.spatial.transform import Rotation
    from tools.render_final_episode_registered_physical_evidence import PhysicalRenderer
    n=read(out/'NUMERIC_SUMMARY.json');folder=Path(n['current_final_plan']['path']);d=read(folder/'seed_0/PHASE_IK.json')
    chosen=next(a for a in d['placement_region_search']['attempts'] if a['result']['phases'][0]['goal_satisfied'])
    r=chosen['result']['phases'][0];q=np.asarray(r['candidates'][r['selected_seed']]['q'])
    cal=read(out/'target_repair/CONTACT_CALIBRATION.json');f=np.asarray(cal['contacts']['pregrasp']['commanded_finger_q']);f[7:]=cal['contacts']['handoff_right']['measured_finger_q'][7:]
    x=np.asarray(chosen['goal']['object_pose']);renderer=PhysicalRenderer()
    try:
        renderer.set_state(np.r_[q,f],x[:3,3],Rotation.from_matrix(x[:3,:3]).as_quat());view=renderer.view('overview')
    finally:renderer.close()
    p=read(out/'source_phase'/n['source_id']/'PHASE_RECORD.json');source=cv2.imread(p['provenance']['yaw_images'][0]['image']);source=cv2.resize(source,(540,405))
    board=np.full((520,1080,3),248,np.uint8);board[75:480,:540]=source;board[75:480,540:]=view
    cv2.putText(board,'Source evidence / rejected PLACE goal (kinematic diagnostic)',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),1,cv2.LINE_AA)
    cv2.putText(board,'Original source RGB',(12,57),cv2.FONT_HERSHEY_SIMPLEX,.53,(25,25,25),1,cv2.LINE_AA)
    cv2.putText(board,'PLANNED, NOT EXECUTED | bin-wall collision',(550,57),cv2.FONT_HERSHEY_SIMPLEX,.53,(35,35,150),1,cv2.LINE_AA)
    cv2.putText(board,'Fixed rule: prototype source; lowest seed and first IK-satisfied rejected placement candidate',(12,508),cv2.FONT_HERSHEY_SIMPLEX,.49,(25,25,25),1,cv2.LINE_AA)
    target=out/'figures/FIRST_FAILURE_CLOSEUP.png';assert cv2.imwrite(str(target),board)
    atomic_json(out/'figures/FIRST_FAILURE_CLOSEUP.json',dict(image=record(target),source=record(p['provenance']['yaw_images'][0]['image']),goal=chosen,physically_executed=False,selection_rule='Lowest seed and first IK-satisfied rejected candidate'))


def replays(out,resume=False):
    from .continuation_media import render_one
    sid=read(out/'NUMERIC_SUMMARY.json')['source_id'];products=[]
    specs=[(out/'prototype'/sid/'morphology_acquisition_v4/physics_attempt_01/event_log.npz',
        'SOURCE_TRAIN_ACQUISITION_LIFT_RELEASE_MEASURED','Source TRAIN | actual acquisition / 60.5 mm lift / retention / release')]
    for m in ('A','B'):
        specs.append((out/f'ACT_interface_diagnostics/v1/LEGACY_ACT_{m}/event_log.npz',
            f'LEGACY_ACT_{m}_TRAIN_INTERFACE_MEASURED',f'Legacy ACT-{m} TRAIN interface | current RGB / measured state | not DEV35'))
    for path,name,title in specs:
        product=render_one(out,path,name,title,resume=resume)
        product['experiment']='LEGACY_ACT_TRAIN_INTERFACE_DIAGNOSTIC' if name.startswith('LEGACY') else 'SOURCE_ACQUISITION_SHORT_INTEGRATION'
        product['primary_ACT_trial']=False
        product['physical_full_task_outcome']='NOT_EVALUATED_SHORT_DIAGNOSTIC'
        product['source_conditioning']='Initial scene only; current RGB and measured joints drive ACT' if name.startswith('LEGACY') else 'Source phase/contact relations determine acquisition goals and initial scene'
        product['full_task_success']=None
        atomic_json(out/'replays'/f'{name}.json',product);products.append(product)
    missing=['ACT_A_DEV35_TOP_35SPLIT.mp4','ACT_A_DEV35_OVERVIEW_35SPLIT.mp4',
        'ACT_B_DEV35_TOP_35SPLIT.mp4','ACT_B_DEV35_OVERVIEW_35SPLIT.mp4','ACT_AB_DEV35_MATCHED_REVIEW.mp4']
    atomic_json(out/'replays/REPLAY_STATUS.json',dict(actual_short_replays=products,
        required_primary_videos=[dict(file=f,status='UNAVAILABLE_NO_PRIMARY_ACT_ROLLOUTS') for f in missing],
        reference_ablation_clip='UNAVAILABLE_PAIRED_PHYSICAL_N0',all_card_videos_created=False,physics_rerun_for_render=False))
    return dict(actual_short_replays=len(products),primary_ACT_videos=0)


def direction(out):
    atomic_text(out/'PAPER_FIGURE_DIRECTION.md',f'''# 그림과 영상 배치 지침

완성된 정책 비교 논문으로 표시하지 않는다. 모든 주 결과는 현재 NOT MEASURED이다. 아래 경로는 본 실행의 부분 보고용 파일이다. PNG와 동일 이름의 SVG/PDF, 편집 코드 `{ROOT}/tools/contact_coordination/act_prerequisite_media.py`를 함께 보존한다.

1. Methods: `{out}/figures/Fig1_Method.png`. ALOHA 원본 → A/B 변환 → 동적 G1 관측·실행 행동 데이터 → 동일 ACT 학습 → 폐루프 비교 순서. 미완료 데이터·학습·평가 상자를 회색으로 유지한다.
   Caption: “Required hybrid retargeting and downstream ACT study. Existing source evidence conditions task goals and the common G1 simulation scene. Dynamic matched demonstration datasets and the ACT-A35/ACT-B35 comparison remain unavailable. Privileged simulator state supports generation and safety; no real-robot or VLA claim is made.”

2. Primary Results/accounting: `{out}/figures/Fig2_PRIMARY_ACT_TASK_SUCCESS.png`, `{out}/TABLE_ACT_A_VS_ACT_B_DEV35.md`, `{out}/ACT_DEV35_PER_EPISODE_RESULTS.csv`. 왼쪽에 ACT-A/B Full Task Success NOT MEASURED, 오른쪽에35원본×2정책 행렬. 회색을 실패/0%로 변경하지 않는다. 별도 짧은 TRAIN 진단을 분모35에 섞지 않는다.
   Caption: “Primary ACT policy comparison on DEV35 development instances:35 intended cases per policy and zero primary physical trials. All70 entries are upstream-not-attempted; completion rates, stage counts, uncertainty and the paired B−A effect are unmeasured. Reference replay and legacy-checkpoint TRAIN interface diagnostics are excluded.”

3. Mechanism/limitations: `{out}/figures/Fig3_Reference_Coupling_Ablation.png`, `{out}/TABLE_REFERENCE_COUPLING_ABLATION_PAIRED10.md`. 왼쪽 물리 쌍 N=0, 오른쪽 원본1개의 계획 잔차 진단. 두 초기값을 두 원본으로 세지 않는다.
   Caption: “The predeclared reference-level paired10 coupling experiment has observed physical N=0. A separate one-source TRAIN optimization diagnostic compares identical contacts, seeds, unary terms and budgets with cross-hand factors on/off. Predicted handoff consistency is not a physical effect estimate and does not establish an ACT learning effect.”

4. Engineering evidence: `{out}/figures/SOURCE_ACQUISITION_MEASURED_EVIDENCE.png`, `{out}/figures/FIRST_FAILURE_CLOSEUP.png`. 실제 물체 높이·접촉력 이후 계획 실패 확대를 배치한다. 확대 이미지는 실행되지 않은 후보로 명시한다.
   Caption: “Measured source-conditioned natural acquisition,60.5mm lift, sustained retention and natural release of the rigid doll surrogate. Full task completion is not demonstrated. The separate source/placement close-up is an unexecuted rejected planning illustration, not a rollout.”

실제 영상: `{out}/replays/SOURCE_TRAIN_ACQUISITION_LIFT_RELEASE_MEASURED.mp4`, `{out}/replays/LEGACY_ACT_A_TRAIN_INTERFACE_MEASURED.mp4`, `{out}/replays/LEGACY_ACT_B_TRAIN_INTERFACE_MEASURED.mp4`. 전체 측정 관절·물체 상태, 실제30fps 시간, 상단/개요 공통 카메라이다. 모든 해당 짧은 진단을 제시하며 성공 사례를 선별하지 않았다. 예시 영상은 일반적인 성능 추정이 아니다. 요구된 DEV35 모자이크와 참조 제거 실험 클립은 원본 물리 실행이 없어 제공할 수 없다. all-card 물리 영상은 만들지 않았다.
''')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--stage',choices=['figures','replays'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.stage=='figures':figures(a.run_dir);closeup(a.run_dir);direction(a.run_dir)
    else:print(replays(a.run_dir,a.resume))
